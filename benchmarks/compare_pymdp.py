"""Reproducible PyAIF versus inferactively-pymdp microbenchmarks.

Run this script in separate environments containing either pymdp 0.0.7.1
(classic NumPy) or pymdp 1.x (JAX). It emits machine-readable JSON so results
from the two environments can be combined without installing incompatible
pymdp releases together.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    DeepTemporalInference,
    GenerativeModel,
    ShallowInference,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    states: int
    factors: int
    actions: int
    horizon: int
    iterations: int

    @property
    def deep(self) -> bool:
        return self.horizon > 1


SCENARIOS = (
    Scenario("shallow_1f", states=16, factors=1, actions=4, horizon=1, iterations=16),
    Scenario("shallow_2f", states=8, factors=2, actions=4, horizon=1, iterations=16),
    Scenario("temporal_81p", states=4, factors=2, actions=3, horizon=3, iterations=8),
)


def object_array(*values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = np.asarray(value, dtype=float)
    return result


def model_arrays(scenario: Scenario):
    rng = np.random.default_rng(20260724)
    likelihood_shape = (scenario.states,) + (scenario.states,) * scenario.factors
    likelihood = rng.random(likelihood_shape) + 0.05
    likelihood /= likelihood.sum(axis=0, keepdims=True)

    transitions = []
    for factor in range(scenario.factors):
        transition = np.empty(
            (scenario.states, scenario.states, scenario.actions),
            dtype=float,
        )
        for action in range(scenario.actions):
            deterministic = np.roll(np.eye(scenario.states), action, axis=0)
            transition[:, :, action] = 0.95 * deterministic + 0.05 / scenario.states
        transitions.append(transition)

    priors = [np.full(scenario.states, 1.0 / scenario.states)] * scenario.factors
    preferences = np.zeros(scenario.states)
    return likelihood, transitions, priors, preferences


def summarize(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = np.asarray(samples_ns, dtype=float) / 1e6
    return {
        "median_ms": float(np.median(samples_ms)),
        "mean_ms": float(np.mean(samples_ms)),
        "p10_ms": float(np.percentile(samples_ms, 10)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "stdev_ms": float(np.std(samples_ms)),
    }


def benchmark(
    function: Callable[[], object],
    synchronize: Callable[[object], None],
    *,
    prepare: Callable[[], None] | None,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float], object]:
    result = None
    for _ in range(warmups):
        if prepare is not None:
            prepare()
        result = function()
        synchronize(result)

    samples = []
    gc.disable()
    try:
        for _ in range(repeats):
            if prepare is not None:
                prepare()
            start = time.perf_counter_ns()
            result = function()
            synchronize(result)
            samples.append(time.perf_counter_ns() - start)
    finally:
        gc.enable()
    return summarize(samples), result


def create_pyaif(scenario: Scenario):
    likelihood, transitions, priors, preferences = model_arrays(scenario)
    if scenario.deep:
        inference = DeepTemporalInference(
            horizon=scenario.horizon,
            message_passing_iterations=scenario.iterations,
        )
        preference_array = np.zeros((scenario.states, scenario.horizon))
    else:
        inference = ShallowInference(
            message_passing_iterations=scenario.iterations,
        )
        preference_array = preferences

    model = GenerativeModel(
        B=object_array(*transitions),
        D=object_array(*priors),
        controls_dim=[scenario.actions] * scenario.factors,
        controllable_factors=list(range(scenario.factors)),
    )
    categorical = CategoricalLikelihood(
        A=object_array(likelihood),
        preferences=object_array(preference_array),
        modality_dependencies=[list(range(scenario.factors))],
    )
    agent = ActiveInfAgent(
        model=model,
        likelihood=categorical,
        inference=inference,
        action_selection="deterministic",
    )
    return agent


def benchmark_pyaif(scenario: Scenario, warmups: int, repeats: int):
    agent = create_pyaif(scenario)

    def prepare_state():
        agent.reset()
        agent.observe([0])

    state_stats, _ = benchmark(
        agent.infer_states,
        lambda _: None,
        prepare=prepare_state,
        warmups=warmups,
        repeats=repeats,
    )

    prepare_state()
    agent.infer_states()
    policy_stats, _ = benchmark(
        agent.infer_policies,
        lambda _: None,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    def full_step():
        agent.infer_states()
        return agent.infer_policies()

    full_stats, _ = benchmark(
        full_step,
        lambda _: None,
        prepare=prepare_state,
        warmups=warmups,
        repeats=repeats,
    )
    if scenario.deep:
        iteration_values = np.asarray(agent.last_state_inference.iterations)
        state_iterations = {
            "minimum": int(np.min(iteration_values)),
            "median": float(np.median(iteration_values)),
            "maximum": int(np.max(iteration_values)),
        }
    else:
        state_iterations = agent.last_state_inference.iterations

    return {
        "implementation": "PyAIF",
        "num_policies": agent.num_policies,
        "state": state_stats,
        "policy": policy_stats,
        "full": full_stats,
        "state_iterations": state_iterations,
    }, agent


def classic_object_array(values):
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


def benchmark_pymdp_classic(scenario: Scenario, warmups: int, repeats: int):
    from pymdp.agent import Agent

    likelihood, transitions, priors, preferences = model_arrays(scenario)
    inference_algo = "MMP" if scenario.deep else "VANILLA"
    inference_params = (
        {"num_iter": scenario.iterations, "grad_descent": True, "tau": 0.25}
        if scenario.deep
        else {
            "num_iter": scenario.iterations,
            "dF": 1.0,
            "dF_tol": 1e-4,
        }
    )
    agent = Agent(
        A=classic_object_array([likelihood]),
        B=classic_object_array(transitions),
        C=classic_object_array([preferences]),
        D=classic_object_array(priors),
        num_controls=[scenario.actions] * scenario.factors,
        control_fac_idx=list(range(scenario.factors)),
        policy_len=scenario.horizon - 1 if scenario.deep else 1,
        inference_horizon=scenario.horizon,
        inference_algo=inference_algo,
        inference_params=inference_params,
        action_selection="deterministic",
    )

    def prepare_state():
        agent.reset()
        if scenario.deep:
            # Classic reset() leaves observation/action histories intact.
            agent.prev_obs = []
            agent.prev_actions = None

    state_stats, posterior = benchmark(
        lambda: agent.infer_states([0]),
        lambda _: None,
        prepare=prepare_state,
        warmups=warmups,
        repeats=repeats,
    )
    prepare_state()
    agent.infer_states([0])
    policy_stats, _ = benchmark(
        agent.infer_policies,
        lambda _: None,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    def full_step():
        agent.infer_states([0])
        return agent.infer_policies()

    full_stats, _ = benchmark(
        full_step,
        lambda _: None,
        prepare=prepare_state,
        warmups=warmups,
        repeats=repeats,
    )
    return {
        "implementation": "pymdp-classic",
        "num_policies": len(agent.policies),
        "state": state_stats,
        "policy": policy_stats,
        "full": full_stats,
    }, posterior


def block_jax(value):
    import jax

    jax.block_until_ready(value)


def benchmark_pymdp_jax(scenario: Scenario, warmups: int, repeats: int):
    import jax
    import jax.numpy as jnp
    from pymdp.agent import Agent

    likelihood, transitions, priors, preferences = model_arrays(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = Agent(
            A=[likelihood],
            B=transitions,
            C=[preferences],
            D=priors,
            A_dependencies=[list(range(scenario.factors))],
            num_controls=[scenario.actions] * scenario.factors,
            control_fac_idx=list(range(scenario.factors)),
            policy_len=scenario.horizon - 1 if scenario.deep else 1,
            inference_horizon=scenario.horizon,
            inference_algo="mmp" if scenario.deep else "fpi",
            num_iter=scenario.iterations,
            action_selection="deterministic",
        )

    observations = [jnp.asarray([[0]])] if scenario.deep else [jnp.asarray([0])]
    prior = agent.D

    def state_call():
        return agent.infer_states(observations, prior)

    cold_start = time.perf_counter_ns()
    posterior = state_call()
    block_jax(posterior)
    cold_state_ms = (time.perf_counter_ns() - cold_start) / 1e6

    state_stats, posterior = benchmark(
        state_call,
        block_jax,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    def policy_call():
        return agent.infer_policies(posterior)

    cold_policy = time.perf_counter_ns()
    cold_policy_result = policy_call()
    block_jax(cold_policy_result)
    cold_policy_ms = (time.perf_counter_ns() - cold_policy) / 1e6
    policy_stats, _ = benchmark(
        policy_call,
        block_jax,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    def full_step():
        beliefs = state_call()
        return agent.infer_policies(beliefs)

    full_stats, _ = benchmark(
        full_step,
        block_jax,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    jitted_state = jax.jit(lambda obs, p: agent.infer_states(obs, p))
    compile_start = time.perf_counter_ns()
    jitted_posterior = jitted_state(observations, prior)
    block_jax(jitted_posterior)
    jit_state_compile_ms = (time.perf_counter_ns() - compile_start) / 1e6
    jit_state_stats, jitted_posterior = benchmark(
        lambda: jitted_state(observations, prior),
        block_jax,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    jitted_policy = jax.jit(lambda beliefs: agent.infer_policies(beliefs))
    compile_start = time.perf_counter_ns()
    jitted_policy_result = jitted_policy(jitted_posterior)
    block_jax(jitted_policy_result)
    jit_policy_compile_ms = (time.perf_counter_ns() - compile_start) / 1e6
    jit_policy_stats, _ = benchmark(
        lambda: jitted_policy(jitted_posterior),
        block_jax,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    def jitted_full_step():
        beliefs = jitted_state(observations, prior)
        return jitted_policy(beliefs)

    jit_full_stats, _ = benchmark(
        jitted_full_step,
        block_jax,
        prepare=None,
        warmups=warmups,
        repeats=repeats,
    )

    return {
        "implementation": "pymdp-jax",
        "num_policies": int(agent.policies.num_policies),
        "state": state_stats,
        "policy": policy_stats,
        "full": full_stats,
        "cold_state_ms": cold_state_ms,
        "cold_policy_ms": cold_policy_ms,
        "jit": {
            "state": jit_state_stats,
            "policy": jit_policy_stats,
            "full": jit_full_stats,
            "compile_state_ms": jit_state_compile_ms,
            "compile_policy_ms": jit_policy_compile_ms,
        },
    }, posterior


def posterior_vectors(posterior, scenario: Scenario, pymdp_major: int):
    if scenario.deep:
        return None
    if pymdp_major >= 1:
        return [np.asarray(factor)[0, -1] for factor in posterior]
    return [np.asarray(factor) for factor in posterior]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    pymdp_version = importlib.metadata.version("inferactively-pymdp")
    pymdp_major = int(pymdp_version.split(".", maxsplit=1)[0])
    jax_x64 = None
    if pymdp_major >= 1:
        import jax

        jax_x64 = bool(jax.config.x64_enabled)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    results = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pyaif": importlib.metadata.version("pyaif-toolkit"),
            "pyaif_git_commit": git_commit,
            "pymdp": pymdp_version,
            "jax_enable_x64": jax_x64,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "scenarios": [],
    }

    for scenario in SCENARIOS:
        pyaif_result, pyaif_agent = benchmark_pyaif(
            scenario,
            args.warmups,
            args.repeats,
        )
        if pymdp_major >= 1:
            pymdp_result, pymdp_posterior = benchmark_pymdp_jax(
                scenario,
                args.warmups,
                args.repeats,
            )
        else:
            pymdp_result, pymdp_posterior = benchmark_pymdp_classic(
                scenario,
                args.warmups,
                args.repeats,
            )

        validation = None
        reference_vectors = posterior_vectors(
            pymdp_posterior,
            scenario,
            pymdp_major,
        )
        if reference_vectors is not None:
            differences = [
                np.max(np.abs(pyaif_agent.posteriors[index] - reference))
                for index, reference in enumerate(reference_vectors)
            ]
            validation = {"max_state_posterior_abs_difference": float(max(differences))}

        results["scenarios"].append(
            {
                "scenario": scenario.__dict__,
                "validation": validation,
                "results": [pyaif_result, pymdp_result],
            }
        )

    rendered = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

"""Profile receding inference and policy components across grid resolutions."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable

import numpy as np

import PyAIF.inference.deep_temporal as deep_module
import PyAIF.inference.filtered_receding as filtered_module
from PyAIF import (
    ActiveInfAgent,
    CategoricalLikelihood,
    FilteredRecedingHorizonInference,
    GenerativeModel,
    RecedingHorizonInference,
    __version__ as pyaif_version,
)


def object_array(*values: np.ndarray) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = np.asarray(value, dtype=float)
    return result


def create_agent(resolution: int, mode: str, *, outcomes: int = 16):
    x_index, y_index = np.indices((resolution, resolution))
    outcome_index = np.floor(
        (x_index + y_index) * outcomes / (2 * resolution - 1)
    ).astype(int)
    likelihood = np.full(
        (outcomes, resolution, resolution),
        0.05 / outcomes,
        dtype=float,
    )
    for outcome in range(outcomes):
        likelihood[outcome][outcome_index == outcome] += 0.95
    likelihood /= likelihood.sum(axis=0, keepdims=True)

    transitions = np.empty((resolution, resolution, 3), dtype=float)
    for action, shift in enumerate((-1, 0, 1)):
        deterministic = np.roll(np.eye(resolution), shift, axis=0)
        transitions[:, :, action] = 0.98 * deterministic + 0.02 / resolution

    inference_type = {
        "receding-mmp": RecedingHorizonInference,
        "filtered-receding": FilteredRecedingHorizonInference,
    }[mode]
    horizon = 3
    inference = inference_type(
        horizon=horizon,
        message_passing_iterations=8,
        policy_workers=1,
    )
    model = GenerativeModel(
        B=object_array(transitions, transitions),
        D=object_array(np.ones(resolution), np.ones(resolution)),
        controls_dim=[3, 3],
        controllable_factors=[0, 1],
    )
    preference = np.linspace(-1.0, 1.0, outcomes)[:, None]
    preference = np.repeat(preference, horizon, axis=1)
    categorical = CategoricalLikelihood(
        A=object_array(likelihood),
        preferences=object_array(preference),
        modality_dependencies=[[0, 1]],
    )
    return ActiveInfAgent(
        model=model,
        likelihood=categorical,
        inference=inference,
        action_selection="deterministic",
    )


def summarize(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = np.asarray(samples_ns, dtype=float) / 1e6
    return {
        "median_ms": float(np.median(samples_ms)),
        "p10_ms": float(np.percentile(samples_ms, 10)),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "mean_ms": float(np.mean(samples_ms)),
    }


class PolicyTimingHooks:
    """Instrument named operations without changing the policy implementation."""

    def __init__(self):
        self.elapsed_ns: dict[str, int] = defaultdict(int)
        self._originals: list[tuple[Any, str, Callable[..., Any]]] = []

    def _patch(self, owner: Any, attribute: str, label: str) -> None:
        original = getattr(owner, attribute)

        @wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.elapsed_ns[label] += time.perf_counter_ns() - start

        self._originals.append((owner, attribute, original))
        setattr(owner, attribute, timed)

    def __enter__(self):
        self._patch(filtered_module, "roll_out_policy_states", "rollout")
        self._patch(
            deep_module,
            "_deep_categorical_policy_terms_batch",
            "efe_terms",
        )
        self._patch(deep_module, "map_policies", "information_gain_dispatch")
        self._patch(ActiveInfAgent, "perform_modal_average", "modal_average")
        self._patch(ActiveInfAgent, "update_policy_posterior", "posterior_update")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        for owner, attribute, original in reversed(self._originals):
            setattr(owner, attribute, original)

    def reset(self) -> None:
        self.elapsed_ns.clear()


def profile_modes(
    resolution: int,
    *,
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    mode_names = ("receding-mmp", "filtered-receding")
    agents = {mode: create_agent(resolution, mode) for mode in mode_names}
    samples = {
        mode: {
            "state": [],
            "full": [],
            "policy": defaultdict(list),
        }
        for mode in mode_names
    }

    def prepare_observation(agent: ActiveInfAgent) -> None:
        agent.reset()
        agent.observe([0])

    def run_state(mode: str) -> None:
        agent = agents[mode]
        prepare_observation(agent)
        start = time.perf_counter_ns()
        agent.infer_states()
        samples[mode]["state"].append(time.perf_counter_ns() - start)

    def run_policy(mode: str, hooks: PolicyTimingHooks) -> None:
        agent = agents[mode]
        prepare_observation(agent)
        agent.infer_states()
        hooks.reset()
        start = time.perf_counter_ns()
        agent.infer_policies()
        total = time.perf_counter_ns() - start

        rollout = hooks.elapsed_ns.get("rollout", 0)
        efe_terms = hooks.elapsed_ns.get("efe_terms", 0)
        information_gain = hooks.elapsed_ns.get("information_gain_dispatch", 0)
        posterior_update = hooks.elapsed_ns.get("posterior_update", 0)
        modal_average = hooks.elapsed_ns.get("modal_average", 0)
        accounted = rollout + efe_terms + information_gain + posterior_update
        component_samples = samples[mode]["policy"]
        component_samples["total"].append(total)
        component_samples["rollout"].append(rollout)
        component_samples["efe_terms"].append(efe_terms)
        component_samples["information_gain_dispatch"].append(information_gain)
        component_samples["posterior_update"].append(posterior_update)
        component_samples["modal_average"].append(modal_average)
        component_samples["posterior_update_other"].append(
            posterior_update - modal_average
        )
        component_samples["bookkeeping_and_overhead"].append(max(0, total - accounted))

    def run_full(mode: str) -> None:
        agent = agents[mode]
        prepare_observation(agent)
        start = time.perf_counter_ns()
        agent.infer_states()
        agent.infer_policies()
        samples[mode]["full"].append(time.perf_counter_ns() - start)

    # Interleave both algorithms and all three measurements, then rotate their
    # order. This limits bias from CPU frequency changes and background load.
    with PolicyTimingHooks() as hooks:
        for index in range(warmups + repeats):
            operations = tuple(
                operation
                for mode in mode_names
                for operation in (
                    lambda mode=mode: run_state(mode),
                    lambda mode=mode: run_policy(mode, hooks),
                    lambda mode=mode: run_full(mode),
                )
            )
            offset = index % len(operations)
            rotated = operations[offset:] + operations[:offset]
            if index < warmups:
                for operation in rotated:
                    operation()
                for mode_samples in samples.values():
                    mode_samples["state"].clear()
                    mode_samples["full"].clear()
                    mode_samples["policy"].clear()
                continue
            gc.disable()
            try:
                for operation in rotated:
                    operation()
            finally:
                gc.enable()

    return [
        {
            "mode": mode,
            "num_policies": agents[mode].num_policies,
            "state": summarize(samples[mode]["state"]),
            "policy": {
                component: summarize(component_values)
                for component, component_values in samples[mode]["policy"].items()
            },
            "full": summarize(samples[mode]["full"]),
        }
        for mode in mode_names
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    if any(resolution < 2 for resolution in args.resolutions):
        raise ValueError("Every resolution must be at least two.")
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
            "pyaif": pyaif_version,
            "pyaif_git_commit": git_commit,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "constants": {
            "spatial_factors": 2,
            "outcomes": 16,
            "actions_per_factor": 3,
            "horizon": 3,
            "policies": 81,
            "message_passing_iterations": 8,
            "policy_workers": 1,
        },
        "resolutions": [],
    }
    for resolution in args.resolutions:
        modes = profile_modes(
            resolution,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        results["resolutions"].append(
            {
                "states_per_spatial_factor": resolution,
                "grid_cells": resolution**2,
                "modes": modes,
            }
        )

    rendered = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

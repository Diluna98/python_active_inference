from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import sys
import numpy as np
sys.path.append(r"C:\Users\dawarn\Documents\matrices_for_ICRA_paper")
from q_learning_meta_baseline import ACTION_LABELS, QLearningMetaController, QLearningConfig


METHOD_TO_ACTION = {
    "2": 0,
    "5": 1,
    "10": 2,
    "20": 3,
    "2x2": 0,
    "5x5": 1,
    "10x10": 2,
    "20x20": 3,
}


@dataclass(frozen=True)
class Transition:
    obs: tuple[float, float, float, float, int]
    action: int
    reward: float
    next_obs: tuple[float, float, float, float, int]
    done: bool


def parse_artifact_name(path: Path) -> tuple[str, str]:
    pattern = re.compile(r"Artifacts_cpu_load_(?P<load>.+?)_212\.5_312\.5_(?P<method>.+)$")
    match = pattern.match(path.stem)
    if not match:
        return "unknown", path.stem
    return match.group("load"), match.group("method")


def action_from_confidence(confidence_row: Sequence[float]) -> int:
    if not confidence_row:
        return 4
    return int(np.argmax(np.asarray(confidence_row, dtype=float)))


def reward_from_meta_transition(
    reward_obs: Sequence[float],
    *,
    latency_weight: float,
    error_weight: float,
    latency_scale_ms: float,
    info_gain_clip: float,
    switch_weight: float,
    previous_resolution: int,
    action: int,
) -> float:
    info_gain_proxy = min(float(reward_obs[0]), info_gain_clip)
    prediction_error = float(reward_obs[1])
    latency_ms = float(reward_obs[2])
    selected_resolution = previous_resolution if action == 4 else action
    switched = selected_resolution != previous_resolution
    return float(
        -(latency_ms / 100.0)
        -prediction_error * info_gain_proxy
        -switch_weight * float(switched)*0
    )


def normalize_obs(obs: Sequence[float], current_resolution: int) -> tuple[float, float, float, float, int]:
    if len(obs) >= 5:
        return (float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), int(obs[4]))
    if len(obs) >= 4:
        return (float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), int(current_resolution))
    raise ValueError(f"Expected at least 4 observation values, got {obs}")


def transitions_from_artifact(
    path: Path,
    *,
    latency_weight: float,
    error_weight: float,
    latency_scale_ms: float,
    info_gain_clip: float,
    switch_weight: float,
) -> list[Transition]:
    _load, method = parse_artifact_name(path)
    if path.stat().st_size == 0:
        print(f"WARNING: skipping empty artifact file: {path.name}")
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except JSONDecodeError as exc:
        print(f"WARNING: skipping invalid JSON artifact file: {path.name} ({exc})")
        return []

    fixed_action = METHOD_TO_ACTION.get(method)
    transitions: list[Transition] = []
    meta_obs_trials = data.get("meta_obs", [])
    confidence_trials = data.get("meta_action_confidance", [])

    for trial_idx, meta_obs in enumerate(meta_obs_trials):
        if len(meta_obs) < 2:
            continue
        confidence = confidence_trials[trial_idx] if trial_idx < len(confidence_trials) else []
        current_resolution = fixed_action if fixed_action is not None else int(meta_obs[0][4]) if len(meta_obs[0]) >= 5 else 0
        for idx in range(len(meta_obs) - 1):
            obs = normalize_obs(meta_obs[idx], current_resolution)
            if fixed_action is None:
                conf_row = confidence[idx] if idx < len(confidence) else []
                action = action_from_confidence(conf_row)
            else:
                action = fixed_action
            selected_resolution = obs[4] if action == 4 else action
            next_obs = normalize_obs(meta_obs[idx + 1], selected_resolution)
            # This mirrors the corrected online update:
            # action at meta step k is updated when meta observation k+1 arrives.
            # Therefore the reward/cost signal is taken from next_obs, not obs.
            reward = reward_from_meta_transition(
                next_obs,
                latency_weight=latency_weight,
                error_weight=error_weight,
                latency_scale_ms=latency_scale_ms,
                info_gain_clip=info_gain_clip,
                switch_weight=switch_weight,
                previous_resolution=obs[4],
                action=action,
            )
            done = idx == len(meta_obs) - 2
            transitions.append(Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done))
            current_resolution = selected_resolution
    return transitions


def train_offline(
    transitions: list[Transition],
    controller: QLearningMetaController,
    *,
    epochs: int,
    seed: int,
) -> tuple[list[float], int]:
    rng = random.Random(seed)
    epoch_rewards = []
    skipped_disallowed = 0
    allowed_actions = set(controller.config.allowed_actions)
    for _epoch in range(epochs):
        rng.shuffle(transitions)
        total_reward = 0.0
        used_transitions = 0
        for tr in transitions:
            if tr.action not in allowed_actions:
                skipped_disallowed += 1
                continue
            controller.learn(tr.obs, tr.action, tr.reward, tr.next_obs, tr.done)
            if tr.action == tr.obs[4] and 4 in allowed_actions:
                # In the simulator, selecting the already-current resolution and
                # choosing keep are equivalent no-op decisions at the meta level.
                controller.learn(tr.obs, 4, tr.reward, tr.next_obs, tr.done)
            total_reward += tr.reward
            used_transitions += 1
        epoch_rewards.append(total_reward / max(1, used_transitions))
    return epoch_rewards, skipped_disallowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Q-learning from logged artifact transitions.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Folder containing artifact files.")
    parser.add_argument("--out", type=Path, default=Path("profiling_results/offline_q_policy.json"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help=(
            "Discount factor for artifact-only offline training. Use 0.0 for a conservative "
            "one-step cost table; use a larger value only with exploratory RL logs."
        ),
    )
    parser.add_argument("--latency-weight", type=float, default=1.0)
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument("--latency-scale-ms", type=float, default=1000.0)
    parser.add_argument("--info-gain-clip", type=float, default=2.0)
    parser.add_argument("--switch-weight", type=float, default=0.05)
    parser.add_argument(
        "--initial-q-values",
        nargs=5,
        type=float,
        default=(-1.0, -1.0, -1.0, -2.0, -1.0),
        metavar=("Q_2X2", "Q_5X5", "Q_10X10", "Q_20X20", "Q_KEEP"),
        help="Initial Q-values for unseen states.",
    )
    parser.add_argument(
        "--allowed-actions",
        nargs="+",
        type=int,
        default=(0, 1, 2, 4),
        help="Action indices allowed during greedy/exploratory selection. Use 0 1 2 4 to exclude 20x20.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=["2", "5", "10", "20", "Qlearning", "RL"],
        help="Artifact method suffixes to include.",
    )
    args = parser.parse_args()

    include = set(args.include)
    artifacts = []
    for path in sorted(args.root.glob("Artifacts_cpu_load_*_212.5_312.5_*.jsonl")):
        _load, method = parse_artifact_name(path)
        if method in include:
            artifacts.append(path)

    transitions: list[Transition] = []
    for path in artifacts:
        loaded = transitions_from_artifact(
            path,
            latency_weight=args.latency_weight,
            error_weight=args.error_weight,
            latency_scale_ms=args.latency_scale_ms,
            info_gain_clip=args.info_gain_clip,
            switch_weight=args.switch_weight,
        )
        transitions.extend(loaded)
        print(f"{path.name}: {len(loaded)} transitions")

    if not transitions:
        raise SystemExit("No transitions found. Check artifact paths and --include.")

    controller = QLearningMetaController(
        q_config=QLearningConfig(
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=0.0,
            epsilon_end=0.0,
            initial_q_values=tuple(args.initial_q_values),
            allowed_actions=tuple(args.allowed_actions),
        )
    )
    rewards, skipped_disallowed = train_offline(transitions, controller, epochs=args.epochs, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    controller.save(args.out)

    diagnostics = {
        "n_artifacts": len(artifacts),
        "n_transitions": len(transitions),
        "n_skipped_disallowed_action_updates": skipped_disallowed,
        "epochs": args.epochs,
        "alpha": args.alpha,}
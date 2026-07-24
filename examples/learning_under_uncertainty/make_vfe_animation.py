"""Create an animated VFE-surprise trace from a saved simulation result."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate trial-wise VFE surprise around the workload transition."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Simulation .npy file containing the F_policies result.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vfe_transition.gif"),
        help="Destination GIF path.",
    )
    parser.add_argument(
        "--transition-trial",
        type=int,
        default=100,
        help="Trial at which the environment changes.",
    )
    parser.add_argument(
        "--ema-rate",
        type=float,
        default=0.03,
        help="Adaptive-baseline update rate.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=15,
        help="Centered smoothing window for display.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Trials revealed per animation frame.",
    )
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def load_best_policy_vfe(path: Path) -> np.ndarray:
    result = np.load(path, allow_pickle=True).item()
    policy_vfe = np.vstack(
        [
            np.asarray(trial_values, dtype=float).reshape(-1)
            for trial_values in result["F_policies"]
        ]
    )
    if not np.isfinite(policy_vfe).all():
        raise ValueError("F_policies contains non-finite values.")

    # F is log model evidence in the agent implementation. Its negative is the
    # conventional variational free-energy magnitude, and the best policy is
    # the policy with the greatest evidence (smallest negative evidence).
    return -np.max(policy_vfe, axis=1)


def adaptive_vfe_surprise(vfe: np.ndarray, rate: float) -> np.ndarray:
    if not 0.0 < rate <= 1.0:
        raise ValueError("ema-rate must be in (0, 1].")

    baseline = np.empty_like(vfe)
    surprise = np.zeros_like(vfe)
    baseline[0] = vfe[0]

    for trial in range(1, len(vfe)):
        surprise[trial] = abs(vfe[trial] - baseline[trial - 1])
        baseline[trial] = (
            (1.0 - rate) * baseline[trial - 1] + rate * vfe[trial]
        )
    return surprise


def centered_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing-window must be a positive odd integer.")
    half_window = window // 2
    padded = np.pad(values, (half_window, half_window), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def create_animation(
    surprise: np.ndarray,
    output: Path,
    *,
    transition_trial: int,
    frame_step: int,
    fps: int,
) -> None:
    trials = np.arange(len(surprise))
    if not 0 <= transition_trial < len(trials):
        raise ValueError("transition-trial must fall inside the simulation.")
    if frame_step < 1:
        raise ValueError("frame-step must be positive.")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    figure.patch.set_facecolor("#f7f8fa")
    axis.set_facecolor("#ffffff")

    transition_end = min(transition_trial + 20, len(trials) - 1)
    axis.axvspan(
        transition_trial,
        transition_end,
        color="#e78a3f",
        alpha=0.14,
        linewidth=0,
    )
    axis.axvline(
        transition_trial,
        color="#b45f1f",
        linewidth=1.5,
        linestyle="--",
    )
    axis.text(
        transition_trial + 3,
        np.max(surprise) * 0.96,
        "Workload transition",
        color="#8b4a1a",
        fontsize=10,
        va="top",
    )

    (trace,) = axis.plot([], [], color="#2f6fa3", linewidth=2.6)
    marker = axis.scatter([], [], s=46, color="#2f6fa3", zorder=4)
    status = axis.text(
        0.015,
        0.95,
        "",
        transform=axis.transAxes,
        fontsize=10,
        va="top",
        color="#28323c",
    )

    axis.set_xlim(0, len(trials) - 1)
    axis.set_ylim(0, np.max(surprise) * 1.08)
    axis.set_xlabel("Trial")
    axis.set_ylabel("VFE surprise (a.u.)")
    axis.set_title(
        "Learning response to a workload transition",
        fontsize=15,
        color="#1f2933",
    )
    axis.text(
        0.5,
        -0.16,
        "15-trial smoothed |best-policy VFE − adaptive baseline|",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        color="#59636e",
    )
    axis.grid(axis="y", color="#d9dde2", linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#aeb6bf")

    frame_ends = list(range(1, len(trials) + 1, frame_step))
    if frame_ends[-1] != len(trials):
        frame_ends.append(len(trials))

    def update(frame_end: int):
        visible_trials = trials[:frame_end]
        visible_surprise = surprise[:frame_end]
        trace.set_data(visible_trials, visible_surprise)
        marker.set_offsets(
            [[visible_trials[-1], visible_surprise[-1]]]
        )
        current_trial = visible_trials[-1]
        if current_trial < transition_trial:
            phase = "learning stable regime"
        elif current_trial < transition_end + 40:
            phase = "adapting to changed regime"
        else:
            phase = "adapted to changed regime"
        status.set_text(
            f"Trial {current_trial:03d}  ·  {phase}"
        )
        return trace, marker, status

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_ends,
        interval=1000 / fps,
        blit=True,
        repeat=True,
        repeat_delay=1200,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    vfe = load_best_policy_vfe(args.input)
    surprise = adaptive_vfe_surprise(vfe, args.ema_rate)
    smoothed_surprise = centered_smooth(
        surprise,
        args.smoothing_window,
    )
    create_animation(
        smoothed_surprise,
        args.output,
        transition_trial=args.transition_trial,
        frame_step=args.frame_step,
        fps=args.fps,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

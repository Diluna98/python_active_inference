"""Configuration for single-step state and policy inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShallowInference:
    message_passing_iterations: int = 16
    convergence_tolerance: float = 1e-4
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.message_passing_iterations < 1:
            raise ValueError("message_passing_iterations must be positive.")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")

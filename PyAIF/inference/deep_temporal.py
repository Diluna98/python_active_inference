"""Configuration for marginal-message-passing over a temporal horizon."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepTemporalInference:
    horizon: int
    message_passing_iterations: int = 16
    convergence_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("Deep temporal inference requires horizon >= 2.")
        if self.message_passing_iterations < 1:
            raise ValueError("message_passing_iterations must be positive.")
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")

"""Shared inference configuration types."""

from typing import Protocol


class InferenceConfiguration(Protocol):
    horizon: int
    message_passing_iterations: int

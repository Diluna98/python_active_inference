"""Inference configurations exposed by PyAIF."""

from .deep_temporal import DeepTemporalInference
from .shallow import (
    ShallowInference,
    ShallowStateInferenceResult,
    infer_shallow_states,
)

__all__ = [
    "DeepTemporalInference",
    "ShallowInference",
    "ShallowStateInferenceResult",
    "infer_shallow_states",
]

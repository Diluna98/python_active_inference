"""Inference configurations exposed by PyAIF."""

from .deep_temporal import DeepTemporalInference
from .shallow import (
    ShallowInference,
    ShallowPolicyInferenceResult,
    ShallowStateInferenceResult,
    infer_shallow_policies,
    infer_shallow_states,
)

__all__ = [
    "DeepTemporalInference",
    "ShallowInference",
    "ShallowPolicyInferenceResult",
    "ShallowStateInferenceResult",
    "infer_shallow_policies",
    "infer_shallow_states",
]

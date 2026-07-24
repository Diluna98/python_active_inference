"""Inference configurations exposed by PyAIF."""

from .deep_temporal import (
    DeepStateInferenceResult,
    DeepTemporalInference,
    infer_deep_temporal_states,
)
from .shallow import (
    ShallowInference,
    ShallowPolicyInferenceResult,
    ShallowStateInferenceResult,
    infer_shallow_policies,
    infer_shallow_states,
)

__all__ = [
    "DeepTemporalInference",
    "DeepStateInferenceResult",
    "ShallowInference",
    "ShallowPolicyInferenceResult",
    "ShallowStateInferenceResult",
    "infer_shallow_policies",
    "infer_shallow_states",
    "infer_deep_temporal_states",
]

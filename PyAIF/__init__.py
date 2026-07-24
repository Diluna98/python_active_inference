"""Public API for PyAIF active-inference agents."""

from . import utils
from .aif_agent import ActiveInfAgent
from .generative_model import GenerativeModel
from .inference import (
    DeepPolicyInferenceResult,
    DeepStateInferenceResult,
    DeepTemporalInference,
    ShallowInference,
    ShallowPolicyInferenceResult,
    ShallowStateInferenceResult,
    deep_categorical_policy_ambiguity,
    deep_categorical_policy_risk,
    deep_expected_free_energy,
)
from .likelihoods import CategoricalLikelihood, ContinuousLikelihood
from .learning import (
    CategoricalLearningResult,
    categorical_observation_evidence,
    categorical_transition_evidence,
    update_dirichlet_parameters,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "ActiveInfAgent",
    "CategoricalLikelihood",
    "ContinuousLikelihood",
    "CategoricalLearningResult",
    "DeepPolicyInferenceResult",
    "DeepStateInferenceResult",
    "DeepTemporalInference",
    "deep_categorical_policy_ambiguity",
    "deep_categorical_policy_risk",
    "deep_expected_free_energy",
    "GenerativeModel",
    "ShallowInference",
    "ShallowPolicyInferenceResult",
    "ShallowStateInferenceResult",
    "categorical_observation_evidence",
    "categorical_transition_evidence",
    "update_dirichlet_parameters",
    "utils",
]

"""Public API for PyAIF's discrete active-inference package."""

from importlib.metadata import PackageNotFoundError, version

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
from .likelihoods import CategoricalLikelihood
from .learning import (
    CategoricalLearningResult,
    categorical_observation_evidence,
    categorical_transition_evidence,
    update_dirichlet_parameters,
)

try:
    __version__ = version("pyaif-toolkit")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "__version__",
    "ActiveInfAgent",
    "CategoricalLikelihood",
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

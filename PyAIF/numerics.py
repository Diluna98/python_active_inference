"""Domain-independent numerical operations used by PyAIF.

These functions deliberately preserve the numerical conventions of the
original ``ActiveInfAgent`` implementation. Keeping them free of agent state
makes their behavior independently testable and reusable by future inference
strategies.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln, psi


def softmax(x, axis=0, gamma=1.0):
    """Return the original PyAIF precision-weighted softmax."""

    values = gamma * np.asarray(x)
    exp_x = np.exp(values - np.max(values))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_stable_probability(array, eps=1e-16):
    """Log probabilities after clipping to PyAIF's supported interval."""

    return np.log(np.clip(array, eps, 1.0))


def log_stable_additive(array, val=np.exp(-16)):
    """Original worker-process convention: add a small value before logging."""

    return np.log(np.asarray(array) + val)


def log_stable_object_array(array, eps=1e-16):
    """Convert a homogeneous object array to numeric form and log it safely."""

    numeric = np.array(array.tolist(), dtype=float)
    return log_stable_probability(numeric, eps=eps)


def factor_dot(tensor, factors: Sequence[np.ndarray]):
    """Contract a likelihood tensor with one posterior per state factor."""

    result = np.asarray(tensor).copy()
    for factor in reversed(range(len(factors))):
        result = np.tensordot(result, factors[factor], axes=(factor + 1, 0))
    return result


def spm_dot(tensor, factors: Sequence[np.ndarray]):
    """Equivalent tensor contraction expressed with NumPy's indexed einsum."""

    dims = list(range(1, len(factors) + 1))
    arguments = [tensor, list(range(np.asarray(tensor).ndim))]
    for index, factor in enumerate(factors):
        arguments.extend([factor, [dims[index]]])
    arguments.append([0])
    result = np.einsum(*arguments)

    if np.prod(result.shape) <= 1:
        return np.array([result.item()], dtype="float64")
    return result


def transpose_transition(
    transition,
    *,
    epsilon=0.0,
    normalize=False,
    replace_nan=False,
):
    """Transpose the two state axes of one action-conditioned transition."""

    transposed = copy.deepcopy(transition) + epsilon
    transposed = np.transpose(transposed, (1, 0))
    if normalize:
        transposed = np.divide(transposed, transposed.sum(axis=0))
    if replace_nan:
        transposed = np.nan_to_num(transposed, nan=0.0)
    return transposed


def one_hot(value, size):
    """Return an integer one-hot vector."""

    encoded = np.zeros(int(size), dtype=int)
    encoded[int(value)] = 1
    return encoded


def wnorm(parameters, val=np.exp(-16)):
    """Weighting term used by PyAIF's Dirichlet learning equations."""

    adjusted = np.asarray(parameters) + val
    norm = np.divide(1.0, np.sum(adjusted, axis=0))
    average = np.divide(1.0, adjusted)
    return 0.5 * (norm - average)


def categorical_kl_terms(p, q, eps=1e-16):
    """Elementwise categorical KL contributions."""

    return p * np.log((p + eps) / (q + eps))


def spm_psi(parameters):
    """Python equivalent of SPM's column-wise ``spm_psi`` helper."""

    return psi(parameters) - psi(np.sum(parameters, axis=0, keepdims=True))


def log_beta(parameters):
    """Compute the log multivariate beta function along the first axis."""

    values = np.asarray(parameters)
    if values.ndim == 1:
        positive = values[values > 0]
        return np.sum(gammaln(positive)) - gammaln(np.sum(positive))

    result = np.zeros(values.shape[1:])
    for index in np.ndindex(values.shape[1:]):
        result[index] = log_beta(values[(slice(None),) + index])
    return result


def dirichlet_kl(q, p):
    """Preserve the Dirichlet-divergence convention used by ActiveInfAgent."""

    p_values = np.asarray(p).copy()
    q_values = np.asarray(q).copy()
    divergence = (
        log_beta(p_values)
        - log_beta(q_values)
        - np.sum(
            (p_values - q_values) * spm_psi(q_values + 1 / 32),
            axis=0,
        )
    )
    return np.sum(divergence)

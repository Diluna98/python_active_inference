import numpy as np

from PyAIF.numerics import (
    categorical_kl_terms,
    dirichlet_kl,
    factor_dot,
    log_beta,
    log_stable_probability,
    one_hot,
    softmax,
    spm_dot,
    transpose_transition,
    wnorm,
)


def test_softmax_preserves_precision_and_axis_convention():
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = softmax(values, axis=0, gamma=2.0)

    assert result.shape == values.shape
    np.testing.assert_allclose(result.sum(axis=0), [1.0, 1.0])
    assert result[1, 0] > result[0, 0]


def test_factor_dot_and_spm_dot_match():
    likelihood = np.arange(12, dtype=float).reshape(3, 2, 2)
    factors = [np.array([0.75, 0.25]), np.array([0.4, 0.6])]

    np.testing.assert_allclose(
        factor_dot(likelihood, factors),
        spm_dot(likelihood, factors),
    )


def test_transition_transpose_options_match_legacy_conventions():
    transition = np.array([[0.8, 0.3], [0.2, 0.7]])

    plain = transpose_transition(transition, epsilon=1e-16)
    normalized = transpose_transition(transition, normalize=True)

    np.testing.assert_allclose(plain, transition.T + 1e-16)
    np.testing.assert_allclose(normalized.sum(axis=0), [1.0, 1.0])


def test_probability_and_learning_helpers():
    probabilities = np.array([0.0, 0.25, 1.0])
    assert np.isfinite(log_stable_probability(probabilities)).all()
    np.testing.assert_array_equal(one_hot(1, 3), [0, 1, 0])

    p = np.array([0.7, 0.3])
    q = np.array([0.5, 0.5])
    assert np.isclose(categorical_kl_terms(p, q).sum(), 0.0822828785)
    assert wnorm(np.array([2.0, 3.0])).shape == (2,)


def test_dirichlet_helpers_are_zero_for_identical_parameters():
    parameters = np.array([[2.0, 3.0], [4.0, 5.0]])

    assert log_beta(parameters).shape == (2,)
    assert np.isclose(dirichlet_kl(parameters, parameters), 0.0)

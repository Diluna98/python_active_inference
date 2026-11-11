import PyAIF


def test_public_api_exposes_version_and_core_components():
    assert PyAIF.__version__ == "0.5.0"

    expected = {
        "ActiveInfAgent",
        "CategoricalLikelihood",
        "ContinuousLikelihood",
        "DeepTemporalInference",
        "FilteredRecedingHorizonInference",
        "FilteredStateInferenceResult",
        "RecedingHorizonInference",
        "GenerativeModel",
        "ShallowInference",
        "expected_free_energy",
        "categorical_expected_utility",
        "categorical_state_information_gain",
        "utils",
    }
    assert expected.issubset(PyAIF.__all__)
    for name in expected:
        assert hasattr(PyAIF, name)

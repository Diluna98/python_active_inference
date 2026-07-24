import PyAIF


def test_public_api_exposes_version_and_core_components():
    assert PyAIF.__version__ == "0.2.0"

    expected = {
        "ActiveInfAgent",
        "CategoricalLikelihood",
        "ContinuousLikelihood",
        "DeepTemporalInference",
        "GenerativeModel",
        "ShallowInference",
        "utils",
    }
    assert expected.issubset(PyAIF.__all__)
    for name in expected:
        assert hasattr(PyAIF, name)

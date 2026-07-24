import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "learning_under_uncertainty"


@pytest.mark.parametrize(
    ("script_name", "result_name"),
    [
        ("epistemic_uncertainty.py", "epistemic_simulation_results_1.npy"),
        ("aleatoric_uncertainty.py", "aleatoric_simulation_results_1.npy"),
    ],
)
def test_learning_example_smoke(tmp_path, script_name, result_name):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYAIF_EXAMPLE_TRIALS"] = "1"
    env["PYAIF_EXAMPLE_OUTPUT_DIR"] = str(tmp_path)
    env["PYAIF_EXAMPLE_SEED"] = "1"

    subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / script_name)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=60,
    )

    result = np.load(tmp_path / result_name, allow_pickle=True).item()
    assert result["decisions"].shape == (1, 4)
    assert result["selected_actions"].shape == (1, 3)
    assert np.isfinite(result["decisions"]).all()
    assert np.isfinite(result["trust"]).all()
    np.testing.assert_array_equal(
        result["learning_updates"],
        [[True, True, True, False, False]],
    )
    assert result["learning_update_labels"] == (
        "likelihood",
        "transition",
        "initial_state",
        "habit",
        "preference",
    )

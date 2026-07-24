import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

"""
This script compares the success rates of the epistemic-only uncertainty model
(`epistemic_uncertainty.py`) and the full-uncertainty model
(`aleatoric_uncertainty.py`).

One can simply run the file 'Python main.py' to reproduce the results presented in our pre-print.

"""

def run_simulation(script_path):
    subprocess.run([sys.executable, script_path], check=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.environ.get("PYAIF_EXAMPLE_OUTPUT_DIR", script_dir)

run_simulation(os.path.join(script_dir, "epistemic_uncertainty.py"))
print("running epistemic-only uncertainty model............")
run_simulation(os.path.join(script_dir, "aleatoric_uncertainty.py"))
print("now running full-uncertainty model............")

# Load dictionaries
epi_data = np.load(
    os.path.join(output_dir, "epistemic_simulation_results_1.npy"),
    allow_pickle=True,
).item()
aleo_data = np.load(
    os.path.join(output_dir, "aleatoric_simulation_results_1.npy"),
    allow_pickle=True,
).item()

epi = epi_data["decisions"][:, :3]
aleo = aleo_data["decisions"][:, :3]

def compute_rates(arr):
    first = arr[:100]
    rest = arr[100:]
    rate_first = first.sum() / first.size * 100
    rate_rest = rest.sum() / rest.size * 100
    rate_all = arr.sum() / arr.size * 100
    return [rate_first, rate_rest, rate_all]

epi_rates = compute_rates(epi)
aleo_rates = compute_rates(aleo)

x = np.arange(3)
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, epi_rates, width, color='red', label='Epistemic Only')
plt.bar(x + width/2, aleo_rates, width, color='blue', label='Fully Uncertainty')

plt.xticks(x, ["Before work-transition", "After Work-transition", "Overall"])
plt.ylabel("Success rate percent")
plt.legend()
plt.tight_layout()

plt.show()
plt.show()

import pandas as pd
import numpy as np

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

data = pd.read_csv("meta_context_labeled_dataset.csv")

# ensure cpu exists (fixed case)
if "cpu" not in data.columns:
    data["cpu"] = 2

# -------------------------------------------------------
# Modalities
# -------------------------------------------------------

modalities = {
    "div": "pred_divergence",
    "acc": "mean_surprise",
    "lat": "inference_time_ms"
}

# -------------------------------------------------------
# Helper: empirical stats
# -------------------------------------------------------

def stats(df, group_cols, col):
    g = df.groupby(group_cols)[col]
    mu = g.mean()
    sigma = g.std().fillna(1e-6)
    return mu, sigma

# -------------------------------------------------------
# Modality 0: divergence (depends on context)
# -------------------------------------------------------

mu_div, sigma_div = stats(
    data,
    ["context"],
    modalities["div"]
)

# -------------------------------------------------------
# Modality 1: expected accuracy (resolution only)
# -------------------------------------------------------

mu_acc, sigma_acc = stats(
    data,
    ["resolution", "context"],
    modalities["acc"]
)

# -------------------------------------------------------
# Modality 4: latency (resolution only, cpu fixed at 2)
# -------------------------------------------------------

mu_lat, sigma_lat = stats(
    data,
    ["resolution"],
    modalities["lat"]
)

# -------------------------------------------------------
# Convert to arrays for fast lookup
# -------------------------------------------------------

num_res = data["resolution"].nunique()
num_ctx = data["context"].nunique()

mu_div_arr = np.array([mu_div[i] for i in sorted(mu_div.index)])
sigma_div_arr = np.array([sigma_div[i] for i in sorted(sigma_div.index)])

# info gain and entropy need 2D tables
mu_acc_arr = np.zeros((num_res, num_ctx))
sigma_acc_arr = np.zeros((num_res, num_ctx))

for (r, c), v in mu_acc.items():
    mu_acc_arr[r, c] = v
for (r, c), v in sigma_acc.items():
    sigma_acc_arr[r, c] = v

mu_lat_arr = np.array([mu_lat[i] for i in sorted(mu_lat.index)])
sigma_lat_arr = np.array([sigma_lat[i] for i in sorted(sigma_lat.index)])

# -------------------------------------------------------
# Save for reuse
# -------------------------------------------------------

np.savez(
    "meta_likelihood_tables.npz",
    mu_div=mu_div_arr,
    sigma_div=sigma_div_arr,
    mu_acc=mu_acc_arr,
    sigma_acc=sigma_acc_arr,
    mu_lat=mu_lat_arr,
    sigma_lat=sigma_lat_arr
)

print("Saved meta likelihood tables.")
import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# -------------------------------------------------------
# 1. load all csv files
# -------------------------------------------------------

files = glob.glob("results_res_260526_*.csv")

dfs = []

for f in files:

    df = pd.read_csv(f)

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
"""
# -------------------------------------------------------
# 2. select runtime statistics
# -------------------------------------------------------

feature_cols = [
    "pred_divergence"
]

X = data[feature_cols].values

# -------------------------------------------------------
# 3. normalize features
# -------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------
# 4. fit Gaussian Mixture Model
# -------------------------------------------------------

num_contexts = 4

gmm = GaussianMixture(
    n_components=num_contexts,
    covariance_type="full",
    random_state=0
)

gmm.fit(X_scaled)

# -------------------------------------------------------
# 5. infer contexts
# -------------------------------------------------------

contexts = gmm.predict(X_scaled)

# soft probabilities if needed
context_probs = gmm.predict_proba(X_scaled)

# -------------------------------------------------------
# 6. attach raw context labels
# -------------------------------------------------------

data["context"] = contexts

# -------------------------------------------------------
# 6.1 remap contexts (ordered by divergence)
# -------------------------------------------------------

context_stats = data.groupby("context")["pred_divergence"].mean()

print("\nContext means (before remap):")
print(context_stats)

ordered_contexts = context_stats.sort_values().index.to_list()

print("\nOrdered contexts (low -> high divergence):")
print(ordered_contexts)

context_map = {old: new for new, old in enumerate(ordered_contexts)}

data["context"] = data["context"].map(context_map)

print("\nContext counts after remap:")
print(data["context"].value_counts().sort_index())

# -------------------------------------------------------
# 6.2 remap resolution to ordered indices
# -------------------------------------------------------

res_order = sorted(data["resolution"].unique())
res_map = {old: i for i, old in enumerate(res_order)}

data["resolution"] = data["resolution"].map(res_map)

# -------------------------------------------------------
# 7. inspect discovered contexts
# -------------------------------------------------------

summary = data.groupby("context")[feature_cols].mean()

print("\nContext Means:\n")
print(summary)
"""
# -------------------------------------------------------
# 6.2 remap resolution to ordered indices
# -------------------------------------------------------

res_order = sorted(data["resolution"].unique())
res_map = {old: i for i, old in enumerate(res_order)}

data["resolution"] = data["resolution"].map(res_map)

# -------------------------------------------------------
# 8. save labeled dataset
# -------------------------------------------------------

data.to_csv(
    "meta_context_labeled_dataset.csv",
    index=False
)

print("\nSaved:")
print("meta_context_labeled_dataset.csv")
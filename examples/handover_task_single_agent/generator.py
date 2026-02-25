import numpy as np

# Original A0 for context A
A0 = np.array([[[11.46, 1.0],
                [0.0, 0.0],
                [2.83, 1.0]],
               [[0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0]]])

# Step 1: Compress the matrix
# Store indices of non-zero entries and their values
nonzero_idx = np.nonzero(A0)
nonzero_vals = A0[nonzero_idx]

# Save compressed context
context_store = {
    "context_A": {
        "indices": nonzero_idx,
        "values": nonzero_vals
    }
}

# Step 2: Later, reconstruct A0 for that context
def retrieve_context(context_name, shape):
    idx = context_store[context_name]["indices"]
    vals = context_store[context_name]["values"]
    A_reconstructed = np.zeros(shape)
    A_reconstructed[idx] = vals
    return A_reconstructed

# Usage
A_loaded = retrieve_context("context_A", A0.shape)
print("Reconstructed A0:\n", A_loaded)

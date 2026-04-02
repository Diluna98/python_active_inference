import numpy as np

def get_posterior_entropy(qs):
    # Calculate the entropy of the posterior distribution over policies
    entropy = 0
    for factor_idx in range(len(qs)):
        log_qs = np.log(qs[factor_idx] + 1e-16)  # Add small value to avoid log(0)
        qs_reshape = qs[factor_idx].reshape(-1, 1)
        entropy += -np.sum(qs_reshape * log_qs)
        entropy += -np.sum(qs_reshape * log_qs)/len(qs[factor_idx])
    return entropy

def refine_factor(coarse_array, new_size):
    old_size = len(coarse_array)
    upscale_factor = new_size // old_size
    
    # Repeat each element to fill the new resolution
    # We divide by the upscale_factor to ensure the sum remains 1.0
    refined_array = np.repeat(coarse_array, upscale_factor) / upscale_factor
    return refined_array

# Your specific data
qs_5 = [
    np.array([9.59757606e-04, 9.99026321e-01, 4.64054639e-06, 4.64054639e-06, 4.64054639e-06]),
    np.array([4.64498289e-06, 4.64498289e-06, 4.64498289e-06, 4.64498289e-06, 9.99981420e-01]),
    np.array([0.18522814, 0.04999293, 0.18522818, 0.27859599, 0.30095477]),
    np.array([0.28406222, 0.28083511, 0.24810679, 0.13572975, 0.05126613])
]

# Convert all factors to size 20
qs_20 = [refine_factor(f, 20) for f in qs_5]

entropy_5 = get_posterior_entropy(qs_5)
entropy_20 = get_posterior_entropy(qs_20)

# Verification
print(f"Original shape: {qs_5[0].shape}, Sum: {qs_5[0].sum()}")
print(f"Refined shape: {qs_20[0].shape}, Sum: {qs_20[0].sum()}")
# Receding-inference resolution scaling and policy-cost breakdown

Date: 2026-09-16  
PyAIF version: 0.4.0  
Python: 3.12.14  
NumPy: 2.5.1  
Platform: Windows 11, Intel64 Family 6 Model 191  
Protocol: 10 warm-ups and 100 measured repetitions; median wall time;
single process; CPU execution. Algorithms and measurement stages were
interleaved and their order rotated on every repetition to reduce bias from
background load and CPU-frequency changes.

## Controlled resolution sweep

This benchmark varies only the number of states along two spatial hidden-state
factors. A resolution of 32 therefore represents a 32 × 32 latent grid with
1,024 joint cells. The observation space remains fixed at 16 categorical
outcomes, isolating hidden-state resolution rather than simultaneously changing
sensor resolution.

The following values remain constant:

- two spatial hidden-state factors;
- 16 observation outcomes;
- three actions per factor;
- horizon three;
- all 81 policies;
- eight permitted message-passing iterations;
- one policy worker; and
- identical dense-transition and likelihood construction rules.

Each state-stage sample starts from a reset agent with one observation. Policy
timings start after state inference and include all work performed by
`infer_policies()`. A full step measures state and policy inference together.
Action selection and environment simulation are excluded.

## End-to-end scaling

All values are median milliseconds.

| Resolution | Grid cells | MMP state | MMP policy | MMP full | Filtered state | Filtered policy | Filtered full | Full-step speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 × 4 | 16 | 7.238 | 1.812 | 9.028 | 1.748 | 3.389 | 5.126 | 1.76× |
| 8 × 8 | 64 | 8.716 | 2.037 | 10.722 | 2.319 | 3.628 | 6.029 | 1.78× |
| 16 × 16 | 256 | 10.851 | 2.983 | 14.293 | 2.484 | 4.461 | 7.017 | 2.04× |
| 32 × 32 | 1,024 | 17.468 | 6.333 | 23.953 | 2.776 | 6.989 | 9.791 | 2.45× |

The full-step latency reduction grows from 43.2% at 4 × 4 to 59.1% at
32 × 32. At the largest tested grid, policy-independent current-state
filtering is 6.29× faster than policy-conditioned MMP state inference.

## Filtered policy-evaluation breakdown

The profiler instruments the normal implementation rather than replacing it
with a synthetic calculation. Components are:

- **Rollout:** propagate the shared filtered belief through `B` for every
  policy and future time point.
- **EFE terms:** batched expected-observation, risk, and ambiguity tensor
  contractions.
- **Information dispatch:** policy information-gain callback dispatch. All
  learning flags are disabled here, so the numerical information gain is zero.
- **Posterior update:** policy precision/softmax update. Policy-weighted future
  state averaging is now skipped in the default online-control path; its
  measured column is retained to make that zero cost explicit.
- **Bookkeeping:** storing diagnostics and expected observations, function
  dispatch, and profiler residual.

| Resolution | Rollout | EFE terms | Information dispatch | Posterior update | Future-state average | Bookkeeping | Policy total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 × 4 | 2.278 | 0.658 | 0.011 | 0.190 | 0.000 | 0.246 | 3.389 |
| 8 × 8 | 2.347 | 0.813 | 0.012 | 0.194 | 0.000 | 0.260 | 3.628 |
| 16 × 16 | 2.509 | 1.448 | 0.012 | 0.199 | 0.000 | 0.264 | 4.461 |
| 32 × 32 | 2.784 | 3.590 | 0.012 | 0.204 | 0.000 | 0.274 | 6.989 |

At 4 × 4, rollout is 67.2% of filtered policy time and EFE contractions are
19.4%. At 32 × 32, rollout falls to 39.8% of the total while EFE contractions
rise to 51.4%. The rollout still becomes slower, but its Python loop and
object-array overhead dominate these relatively small dense matrix-vector
products. EFE contractions respond more strongly to the growing joint-state
tensor and become the main resolution-sensitive policy cost.

The filtered current posterior is copied directly into the carried current
belief. Future policy-weighted state averages are not required for policy
scoring or first-action selection, so their default cost is zero. They remain
available with `average_future_states=True` for trajectory visualization and
diagnostics.

## Receding-MMP policy breakdown

MMP constructs its policy-conditioned trajectories during state inference, so
its policy stage has no separate rollout component.

| Resolution | EFE terms | Information dispatch | Posterior update (model average) | Bookkeeping | Policy total |
|---:|---:|---:|---:|---:|---:|
| 4 × 4 | 0.970 | 0.011 | 0.518 (0.319) | 0.297 | 1.812 |
| 8 × 8 | 1.191 | 0.012 | 0.524 (0.323) | 0.307 | 2.037 |
| 16 × 16 | 2.128 | 0.012 | 0.532 (0.327) | 0.308 | 2.983 |
| 32 × 32 | 5.424 | 0.012 | 0.547 (0.339) | 0.314 | 6.333 |

MMP scores the current time point plus two future points, whereas filtered
receding inference scores only the two genuinely future points. At low
resolution, avoiding one EFE time point does not compensate for the explicit
rollout cost, so the filtered policy stage is slower. As resolution increases,
the avoided tensor contraction becomes more expensive: the policy-stage gap
narrows from 87.0% at 4 × 4 to 10.4% at 32 × 32.

## Conclusions

- The new method's end-to-end advantage increases with latent-grid resolution
  in the tested range.
- Policy rollout explains the low-resolution policy-stage penalty.
- EFE risk/ambiguity contractions, not rollout, become the dominant
  resolution-sensitive policy cost.
- Redundant future-state averaging has been removed from the default filtered
  control path without changing expected free energy, policy probabilities, or
  the selected action.
- These results isolate categorical hidden-state resolution. They do not yet
  measure continuous sensor integration, changing observation-grid resolution,
  sparse transitions, or application-specific source-localization likelihoods.
- Dense `B` matrices scale quadratically per state factor. Substantially larger
  robotic grids should also be profiled with sparse or structured transitions.

The measurements are reproducible with `benchmarks/profile_resolution.py`.

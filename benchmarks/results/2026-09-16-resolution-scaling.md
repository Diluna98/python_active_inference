# Receding-inference resolution scaling and policy-cost breakdown

Date: 2026-09-16  
PyAIF version: 0.4.0  
Python: 3.12.14  
NumPy: 2.5.1  
Platform: Windows 11, Intel64 Family 6 Model 191  
Protocol: 10 warm-ups and 100 measured repetitions; median wall time;
single process; CPU execution.

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
| 4 × 4 | 16 | 2.856 | 1.233 | 4.144 | 0.671 | 1.789 | 2.508 | 1.65× |
| 8 × 8 | 64 | 3.411 | 1.379 | 4.819 | 0.884 | 1.923 | 2.802 | 1.72× |
| 16 × 16 | 256 | 4.414 | 1.816 | 6.252 | 0.911 | 2.270 | 3.201 | 1.95× |
| 32 × 32 | 1,024 | 8.016 | 3.482 | 11.488 | 1.057 | 3.473 | 4.525 | 2.54× |

The full-step latency reduction grows from 39.5% at 4 × 4 to 60.6% at
32 × 32. At the largest tested grid, policy-independent current-state
filtering is 7.58× faster than policy-conditioned MMP state inference.

## Filtered policy-evaluation breakdown

The profiler instruments the normal implementation rather than replacing it
with a synthetic calculation. Components are:

- **Rollout:** propagate the shared filtered belief through `B` for every
  policy and future time point.
- **EFE terms:** batched expected-observation, risk, and ambiguity tensor
  contractions.
- **Information dispatch:** policy information-gain callback dispatch. All
  learning flags are disabled here, so the numerical information gain is zero.
- **Posterior update:** policy precision/softmax update plus Bayesian model
  averaging. The model-average portion is shown in parentheses.
- **Bookkeeping:** storing diagnostics and expected observations, function
  dispatch, and profiler residual.

| Resolution | Rollout | EFE terms | Information dispatch | Posterior update (model average) | Bookkeeping | Policy total |
|---:|---:|---:|---:|---:|---:|---:|
| 4 × 4 | 0.753 | 0.339 | 0.006 | 0.567 (0.480) | 0.123 | 1.789 |
| 8 × 8 | 0.769 | 0.435 | 0.006 | 0.581 (0.490) | 0.128 | 1.923 |
| 16 × 16 | 0.819 | 0.722 | 0.007 | 0.587 (0.494) | 0.132 | 2.270 |
| 32 × 32 | 0.909 | 1.798 | 0.007 | 0.610 (0.513) | 0.139 | 3.473 |

At 4 × 4, rollout is 42.1% of filtered policy time and EFE contractions are
19.0%. At 32 × 32, rollout falls to 26.2% of the total while EFE contractions
rise to 51.8%. The rollout still becomes slower, but its Python loop and
object-array overhead dominate these relatively small dense matrix-vector
products. EFE contractions respond more strongly to the growing joint-state
tensor and become the main resolution-sensitive policy cost.

Bayesian model averaging is most of the posterior-update cost. It changes only
slightly over this range because policy count, horizon, and factor count remain
fixed and the state vectors are still small enough for fixed overhead to
dominate.

## Receding-MMP policy breakdown

MMP constructs its policy-conditioned trajectories during state inference, so
its policy stage has no separate rollout component.

| Resolution | EFE terms | Information dispatch | Posterior update (model average) | Bookkeeping | Policy total |
|---:|---:|---:|---:|---:|---:|
| 4 × 4 | 0.501 | 0.006 | 0.573 (0.482) | 0.152 | 1.233 |
| 8 × 8 | 0.630 | 0.006 | 0.583 (0.490) | 0.154 | 1.379 |
| 16 × 16 | 1.056 | 0.006 | 0.595 (0.499) | 0.159 | 1.816 |
| 32 × 32 | 2.698 | 0.007 | 0.611 (0.513) | 0.165 | 3.482 |

MMP scores the current time point plus two future points, whereas filtered
receding inference scores only the two genuinely future points. At low
resolution, avoiding one EFE time point does not compensate for the explicit
rollout cost, so the filtered policy stage is 45.2% slower. As resolution
increases, the avoided tensor contraction becomes more expensive. At 32 × 32,
the two complete policy stages are effectively equal: filtered is 0.27% faster
in this run.

## Conclusions

- The new method's end-to-end advantage increases with latent-grid resolution
  in the tested range.
- Policy rollout explains the low-resolution policy-stage penalty.
- EFE risk/ambiguity contractions, not rollout, become the dominant
  resolution-sensitive policy cost.
- The largest remaining nearly fixed policy cost is Bayesian model averaging,
  suggesting a useful target for vectorization independent of resolution.
- These results isolate categorical hidden-state resolution. They do not yet
  measure continuous sensor integration, changing observation-grid resolution,
  sparse transitions, or application-specific source-localization likelihoods.
- Dense `B` matrices scale quadratically per state factor. Substantially larger
  robotic grids should also be profiled with sparse or structured transitions.

The measurements are reproducible with `benchmarks/profile_resolution.py`.

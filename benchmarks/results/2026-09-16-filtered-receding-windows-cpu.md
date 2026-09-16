# PyAIF filtered receding-horizon CPU benchmark

Date: 2026-09-16  
PyAIF version: 0.4.0  
PyAIF commit: `170cf056ac6b50cfb33c3b7081e4da396a2b4a50`  
Python: 3.12.14  
NumPy: 2.5.1  
Platform: Windows 11, Intel64 Family 6 Model 191  
Protocol: 10 warm-ups and 100 measured repetitions; median wall time;
single process; CPU execution.

The temporal workload has two hidden-state factors with four states each,
three actions per factor, horizon three, eight permitted state-inference
iterations, and all 81 policies. All implementations received the same
normalized categorical `A`, `B`, and `D` arrays and observation.

Each state-stage sample starts from a reset agent with one observation. Each
policy-stage sample starts from an agent for which state inference has already
completed; that preparation is outside the timed region. A full step measures
state inference followed by policy inference. Action selection, environment
simulation, and parameter learning are not included.

## PyAIF temporal modes

These measurements came from the classic-pymdp environment. The separate JAX
environment produced the same qualitative result.

| PyAIF mode | State inference | Policy evaluation | Full step | Relative to receding MMP |
|---|---:|---:|---:|---:|
| Fixed-window MMP | 2.829 ms | 1.180 ms | 4.002 ms | 1.03× faster |
| Receding MMP | 2.857 ms | 1.184 ms | 4.114 ms | baseline |
| Filtered receding | 0.679 ms | 1.772 ms | 2.473 ms | **1.66× faster** |

Filter-then-plan reduced current-state inference time by 76.2% (4.21×), because
the current posterior is inferred once rather than once per policy trajectory.
Its policy stage was 49.7% slower because that stage now includes explicit
transition rollouts from the shared filtered belief. The state-stage saving was
larger, producing a 39.9% reduction in complete decision latency.

This is a comparison of different inference semantics. Receding MMP allows
future-to-present messages; filtered receding inference does not. The timing
result should therefore support algorithm selection, not imply numerical
equivalence.

## Classic NumPy pymdp 0.0.7.1

| Scenario | Policies | PyAIF mode | PyAIF full step | pymdp full step | PyAIF speed-up |
|---|---:|---|---:|---:|---:|
| One factor, 16 states | 4 | Shallow | 0.313 ms | 0.860 ms | 2.74× |
| Two factors, 8 states each | 16 | Shallow | 1.390 ms | 11.868 ms | 8.54× |
| Temporal, two factors, horizon 3 | 81 | Filtered receding | 2.473 ms | 133.526 ms | 53.99× |

For reference, fixed-window PyAIF took 4.002 ms and receding-MMP PyAIF took
4.114 ms in the temporal workload. Classic pymdp uses its MMP configuration in
that row, so none of the temporal ratios establish objective equivalence.

The single-factor shallow posterior agreed with classic pymdp to
`1.4e-16`. The two-factor posterior differed by at most `1.6e-3`, consistent
with the different factor-update and convergence schedules reported in the
earlier benchmark.

## Current JAX pymdp 1.0.3

JAX 64-bit mode was enabled. Every timing blocks until JAX computation is
complete. Compilation is excluded from warmed-JIT execution times.

| Scenario | PyAIF | pymdp eager API | pymdp warmed JIT | JIT advantage over PyAIF | JIT compile cost |
|---|---:|---:|---:|---:|---:|
| One factor, 16 states | 0.307 ms | 74.698 ms | 0.017 ms | 17.86× | 132 ms |
| Two factors, 8 states each | 1.537 ms | 166.499 ms | 0.039 ms | 39.32× | 183 ms |
| Temporal, 81 policies (filtered receding) | 2.713 ms | 191.609 ms | 0.049 ms | 55.71× | 758 ms |

In the temporal workload, PyAIF filter-then-plan was 70.63× faster than the
current pymdp eager API. Warmed, explicitly JIT-compiled pymdp was 55.71×
faster than PyAIF. Using the measured compilation and steady-state times, the
approximate break-even point was 285 repeated, shape-stable inference steps.

The temporal row remains a workload comparison. Current pymdp separates
sequence inference from future-policy evaluation, but its objective, public
API, and JAX execution model are not identical to PyAIF's filtered controller.

## Comparison with the earlier profile

The earlier 2026-07-24 report measured PyAIF's fixed-window temporal algorithm
at approximately 4.2–4.4 ms per full step. The new runs measured fixed-window
and receding-MMP modes at approximately 4.0–4.2 ms, so the original conclusion
is reproducible within ordinary machine and harness variation. The new 2.5–2.7
ms result belongs specifically to `FilteredRecedingHorizonInference`; it does
not retroactively replace the older algorithm's measurement.

## Interpretation

- Filter-then-plan materially reduces online decision latency for this
  many-policy categorical workload.
- It moves work from policy-conditioned state inference into explicit future
  rollouts, explaining why policy evaluation itself is slower.
- Classic and eager JAX pymdp remain much slower for these small online calls.
- Warmed JAX JIT remains the highest-throughput option after compilation is
  amortized over hundreds of shape-stable calls.
- Results are machine- and model-specific. Continuous likelihoods, larger
  state tensors, different policy counts, and thread settings require separate
  measurements.

See the follow-up
[resolution-scaling and policy-cost profile](2026-09-16-resolution-scaling.md)
for 4 × 4 through 32 × 32 hidden-state grids.

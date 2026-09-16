# PyAIF–pymdp performance comparison

This benchmark compares identical categorical model arrays and policy spaces
using:

- PyAIF from the current source checkout;
- `inferactively-pymdp==0.0.7.1`, the classic NumPy release;
- `inferactively-pymdp==1.0.3`, the current JAX release.

The script reports state inference, policy inference, and their combined
execution separately. Temporal scenarios report fixed-window MMP, receding
MMP, and filtered receding-horizon PyAIF as distinct algorithms. For JAX it
reports ordinary public-API execution, first-call latency, explicit JIT
compilation cost, and warmed JIT execution. Every JAX timing blocks until
computation is complete.

The shallow scenarios use equivalent factorized categorical state inference.
Their resulting posteriors are compared numerically. The temporal scenario is
a workload comparison, not a claim that every mode implements an identical
objective. PyAIF's fixed-window and receding-MMP modes perform
policy-conditioned marginal message passing. Its filtered receding mode
estimates the present once and then rolls each policy forward. Current pymdp
separates sequence inference from future-policy evaluation.

Run the script in separate virtual environments because the two pymdp
releases cannot be installed together:

```bash
python benchmarks/compare_pymdp.py \
  --warmups 10 \
  --repeats 100 \
  --output results.json
```

Profile receding MMP and filter-then-plan across hidden-state grid resolutions,
including an instrumented policy-evaluation breakdown:

```bash
python benchmarks/profile_resolution.py \
  --resolutions 4 8 16 32 \
  --warmups 10 \
  --repeats 100 \
  --output resolution-results.json
```

Timing microbenchmarks are machine- and version-specific. Use a quiet machine,
the same Python and NumPy versions, and multiple repetitions. Do not compare
JAX asynchronous dispatch without blocking or mix compilation time into
steady-state measurements.

Published results:

- [Resolution scaling and policy-cost breakdown (2026-09-16)](results/2026-09-16-resolution-scaling.md)
- [Filtered receding-horizon profile (2026-09-16)](results/2026-09-16-filtered-receding-windows-cpu.md)
- [Original PyAIF–pymdp profile (2026-07-24)](results/2026-07-24-windows-cpu.md)

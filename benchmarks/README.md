# PyAIF–pymdp performance comparison

This benchmark compares identical categorical model arrays and policy spaces
using:

- PyAIF from the current source checkout;
- `inferactively-pymdp==0.0.7.1`, the classic NumPy release;
- `inferactively-pymdp==1.0.3`, the current JAX release.

The script reports state inference, policy inference, and their combined
execution separately. For JAX it reports ordinary public-API execution,
first-call latency, explicit JIT compilation cost, and warmed JIT execution.
Every JAX timing blocks until computation is complete.

The shallow scenarios use equivalent factorized categorical state inference.
Their resulting posteriors are compared numerically. The temporal scenario is
a workload comparison, not a claim that the libraries implement identical
temporal objectives: PyAIF performs policy-conditioned marginal message
passing, while current pymdp separates sequence inference from future-policy
evaluation.

Run the script in separate virtual environments because the two pymdp
releases cannot be installed together:

```bash
python benchmarks/compare_pymdp.py \
  --warmups 10 \
  --repeats 100 \
  --output results.json
```

Timing microbenchmarks are machine- and version-specific. Use a quiet machine,
the same Python and NumPy versions, and multiple repetitions. Do not compare
JAX asynchronous dispatch without blocking or mix compilation time into
steady-state measurements.

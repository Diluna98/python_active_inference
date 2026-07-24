# PyAIF versus pymdp CPU benchmark

Date: 2026-07-24  
PyAIF commit: `d68c21dddab311482c186023183469b46751cd45`  
Python: 3.12.13  
NumPy: 2.5.1  
Platform: Windows 11, Intel64 Family 6 Model 191  
Protocol: 10 warm-ups and 100 measured repetitions; median wall time;
single process; CPU execution.

The benchmark used identical normalized categorical `A`, `B`, and `D`
matrices, control dimensions, observations, iteration limits, and policy
spaces. A full step contains state inference followed by policy inference.

## Classic NumPy pymdp 0.0.7.1

| Scenario | Policies | PyAIF full step | pymdp full step | PyAIF speed-up |
|---|---:|---:|---:|---:|
| One factor, 16 states | 4 | 0.344 ms | 0.852 ms | 2.48× |
| Two factors, 8 states each | 16 | 1.479 ms | 12.132 ms | 8.20× |
| Temporal, two factors, horizon 3 | 81 | 4.359 ms | 131.783 ms | 30.23× |

The single-factor state posteriors agreed to `1.4e-16`. The two-factor
posteriors differed by at most `1.6e-3`, reflecting different factor-update
and convergence schedules.

## Current JAX pymdp 1.0.3

JAX 64-bit mode was enabled to match PyAIF's NumPy `float64` calculations.

| Scenario | PyAIF | pymdp eager API | pymdp warmed JIT | JIT advantage over PyAIF | JIT compile cost |
|---|---:|---:|---:|---:|---:|
| One factor, 16 states | 0.348 ms | 73.047 ms | 0.016 ms | 21.28× | 143 ms |
| Two factors, 8 states each | 1.457 ms | 160.259 ms | 0.027 ms | 54.99× | 213 ms |
| Temporal, 81 policies | 4.201 ms | 180.280 ms | 0.029 ms | 145.36× | 651 ms |

Approximate break-even points for paying the explicit JIT compilation cost
instead of repeatedly running PyAIF were 432, 149, and 156 full inference
steps, respectively. First un-jitted pymdp calls took approximately
0.63–0.99 seconds across these models.

The JAX shallow posteriors agreed with PyAIF to `3.5e-17` for one factor and
`9.3e-6` for two factors.

## Interpretation

- PyAIF is faster than classic NumPy pymdp for complete online inference in
  these small and medium categorical models.
- PyAIF is also much faster than the current pymdp public API when it is
  executed eagerly.
- Explicitly JIT-compiled current pymdp is much faster once compilation has
  been amortized over many repeated calls with fixed array shapes.
- PyAIF has no compilation delay and is therefore attractive for interactive
  use, changing model shapes, short simulations, and moderate online
  workloads.
- JAX pymdp is the stronger throughput choice for long, shape-stable,
  repeatedly executed simulations.

The temporal row is a workload comparison, not proof of equivalent inference
objectives. PyAIF performs policy-conditioned marginal message passing;
current pymdp separates sequence inference from future-policy evaluation.

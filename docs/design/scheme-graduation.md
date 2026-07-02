# Scheme graduation study — which opt-ins should you turn on?

**Date:** 2026-07-02. Five configurations swept over five cases with measurable
truth, to decide the recommended flag bundle (single-rank CPU, this repo at
feat/stage-bgk-gpu; sweep script logic preserved below the tables).

Configs: `legacy` (all defaults), `prec` (`ho_pressure_recon`), `sbgk`
(`stage_bgk`), `prec+sbgk`, `prec+sbgk+lim` (+`ho_realizability_limiter`).

## Case 1 — exact stationary contact (Kn=0, 1000:1, Nx=64, t=0.05)

Exact solution: nothing moves. Every number is pure numerical error.

| config | max\|u\| | max\|p-1\| |
|---|---|---|
| legacy | 6.4e-02 | 3.9e-02 |
| prec | 4.9e-03 | 1.4e-02 |
| sbgk | 7.1e-02 | 4.1e-02 |
| prec+sbgk | **3.2e-16** | **2.8e-15** |
| prec+sbgk+lim | 3.2e-16 | 2.8e-15 |

## Case 2 — exact smooth stationary state (uniform-p Gaussian bubble, u=0, Kn=0, 32², t=0.05)

| config | max\|u\| | max\|p-1\| |
|---|---|---|
| legacy | 2.6e-02 | 1.9e-02 |
| prec | 2.3e-03 | 6.4e-03 |
| sbgk | 2.8e-02 | 2.5e-02 |
| prec+sbgk | **2.2e-16** | **3.0e-15** |
| prec+sbgk+lim | 2.2e-16 | 3.0e-15 |

## Case 3 — smooth traveling pulse, self-convergence (1D periodic, Kn=1, t=0.05)

| config | L1 (100 vs 200) | L1 (200 vs 400) | order |
|---|---|---|---|
| legacy | 1.74e-04 | 4.54e-05 | 1.93 |
| prec | 2.03e-04 | 5.31e-05 | 1.94 |
| sbgk | 1.74e-04 | 4.55e-05 | 1.93 |
| prec+sbgk | 2.03e-04 | 5.30e-05 | 1.93 |
| prec+sbgk+lim | 2.03e-04 | 5.30e-05 | 1.93 |

All second order; `prec` costs ~17% in the error constant on this case.

## Case 4 — shock tube (rho 1|0.125, uniform T, Kn=1, N=200 vs N=1600 reference)

| config | L1(rho) | overshoot | undershoot |
|---|---|---|---|
| legacy | 6.70e-03 | 0 | 0 |
| prec | 6.58e-03 | 0 | 0 |
| sbgk | 7.47e-03 | 0 | 0 |
| prec+sbgk | 7.34e-03 | 0 | 0 |
| prec+sbgk+lim | 7.34e-03 | 0 | 0 |

## Case 5 — crossing jets Ma=100 (32³, Kn=1000): robustness + CPU cost

| config | min rho | mass drift | wall s/step |
|---|---|---|---|
| legacy | 1.0e-03 | 1.8e-16 | 25.6 |
| prec | 1.0e-03 | 1.8e-16 | 25.4 |
| sbgk | 1.0e-03 | 0 | 25.0 |
| prec+sbgk | 1.0e-03 | 1.8e-16 | 26.1 |
| prec+sbgk+lim | 1.0e-03 | 1.8e-16 | 36.5 |

## Verdict

**Recommended bundle: `ho_pressure_recon = true` + `stage_bgk = true`**
(exposed as `scheme = :recommended`).

- Machine-exact on both exact-solution cases (legacy: 2.6–6.4% error that does
  not converge in max norm). The two fixes only work together — `sbgk` alone is
  marginally WORSE than legacy.
- Clean second order on smooth flow; the price is a ~10–17% larger error
  constant on smooth/shock transport, which converges away (unlike the contact
  error it removes).
- No robustness change at Ma=100; runtime cost indistinguishable from zero.
- The scaling limiter is NOT in the bundle: +40% wall time and no accuracy
  benefit on these cases. It remains the right tool for deep-vacuum
  realizability stress (see docs on `ho_realizability_limiter`), and it
  composes with the bundle when needed.

**As of 2026-07-02 the package default IS `scheme = :recommended`** (on the CPU
runner and `run_gpu_3d`). `scheme = :legacy` reproduces the historical bit-exact
behavior (pre-July-2026 results, MATLAB parity). Function-level golden and
MATLAB-parity tests are unaffected by the flip — they do not go through params.

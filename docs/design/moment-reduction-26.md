# 26-moment reduction (Rodney Fox) — CPU + GPU, opt-in

## What it is

An opt-in projection of the 35-moment state onto a 26-moment reduced manifold. The
nine **odd** fourth-order standardized moments are replaced by closure values; only
the six **even** fourth-order moments (S400, S040, S004, S220, S202, S022) stay
independent. Rodney Fox's conjecture (2026-07-22): the reduced system is globally
hyperbolic and loses no accuracy.

The nine dropped moments and their closures:

- **s310-type** (six: S310, S130, S301, S103, S031, S013), eq (43):
  `S310 = S110·S400 + (3/2)·S300·(S210 − S110·S300)`  (+ permutations)
- **s211-type** (three: S211, S121, S112):
  `S211 = S011 + S300·S111`  (+ permutations)

Applied per cell per step as an operator-split projection: standardize → overwrite the
nine odd moments with their closure → destandardize back to raw moments. Only those
nine raw moments change (each 4th-order raw moment `M_ijk = f(means, lower centrals) +
M000·C_ijk`, so changing only the 4th-order centrals touches only the matching raws).

## Status: diagnostic / comparison feature, default OFF

Both paths are opt-in and byte-identical when off:

- **CPU**: `src/numerics/moment_reduce26.jl`, toggled by `Riemann35.REDUCE26[] = true`
  (env `REDUCE26=1` in `examples/run_3d_custom_jets.jl`). Per-step hook in
  `src/simulation_runner.jl`.
- **GPU** (order-3 single-GPU path): `gpu/reduce26_gpu.jl`, toggled by the
  `reduce26=true` kwarg on `run_gpu_3d` / `march3d_order3_gpu!`. Per-step
  `_reduce26_interior!` kernel after the RK stages. Reuses the same verbatim `@fastmath`
  device subsets the flux/realizability kernels use (`_recon_centrals` raw→central,
  `_c4tom4_35` central→raw), so it reproduces the CPU `reduce26_moments` to
  **4.2e-16** across 5000 random realizable states
  (`scripts/test_reduce26_gpu.jl`). `reduce26` is rejected on any path other than
  single-GPU order-3.

## Accuracy finding (2026-07-22)

Tested on 3D crossing jets, Ma=100, Kn=1, order-3 WENO5+θ*-IDP.

**The reduction is an accurate closure.** The per-step closure residual
`‖reduce26(M) − M‖ / ‖M‖` on the *evolved* full-35 field is sub-percent and
*decreases* with resolution:

| grid | global residual | odd-9 residual |
|------|-----------------|----------------|
| 12³  | 1.5%            | 2.0%           |
| 24³  | 0.32%           | 0.42%          |
| 36³  | 0.22%           | 0.29%          |
| 48³  | 0.13%           | 0.18%          |
| 64³  | 0.109%          | 0.147%         |
| 96³  | 0.096%          | 0.129%         |
| 128³ | 0.098%          | 0.131%         |

The flow stays close to (and closer as resolved) the reduced manifold; the residual
shrinks then PLATEAUS at ~0.1% — a small, irreducible closure cost, not asymptotically
exact. Norm-weighted decomposition (128³) shows it is highly localized: the top 1% of
cells contribute 74% of Σ‖ΔM‖², the top 5% contribute 98%. Those cells sit at
intermediate density (ρ≈0.02–0.03, jet peripheries / mixing zones) with moderate
gradients (~51% overlap the top-10% |∇ρ| interface cells, ~95% the top-10% ρ
jet-influenced cells). So the reduction is essentially exact in the smooth bulk; its
~0.1% cost is concentrated in the thin jet-collision/mixing layer — physically where the
VDF goes bimodal and carries the odd 4th-order content the reduction discards.

**Pointwise dynamic comparison at Ma=100 is NOT a valid accuracy test — it is
chaos-limited.** The Ma=100 crossing-jets flow is sensitively dependent: identical
order-3 code and IC, differing only in the dt sequence (55 vs 136 steps to t=0.004),
diverge by **0.18** in relative L2; CPU vs GPU order-3 differ by 0.10–0.22. The
"closure effect" from separately-evolved reduced-vs-full runs (~0.15–0.30, *growing*
with resolution) is this chaotic amplification of a sub-percent per-step perturbation,
not closure inaccuracy — the per-moment breakdown is roughly uniform across moment
orders (density 0.17 … 4th-order 0.29), the signature of trajectory drift, not a
targeted high-moment error. The residual (which *shrinks* with resolution) and the
pointwise difference (which *grows*) move in opposite directions; that is how the two
are distinguished.

Consequence: assess this reduction (and closures generally) at Ma=100 via the per-step
residual or statistical/structural metrics, or on a non-chaotic case (the 1D DVM-BGK
ground-truth harness) — not via pointwise L2 against a finer full-35 solve.

## GPU throughput

The order-3 GPU march is kernel-launch-latency-bound at small grids and only wins at
scale: measured V100 wall time to t=0.004 (Ma=100 crossing jets) was 6.6 s/step at 24³
(14k cells) but 3.0 s/step at 48³ (110k cells) — 8× the cells at half the per-step time.
Extrapolated CPU cost at 48³ is ~10 h vs ~15 min on the GPU (~40×). Use `-g0` for fast
ptxas. Reference/sweep drivers: `scripts/build_ic.jl`, `scripts/gpu_ref_run.jl`.

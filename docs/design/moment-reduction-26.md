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

## Wall-bounded finding (2026-07-26): no steady state in force-driven channel flow

**The projection has no steady state in planar Poiseuille at Kn_H = lambda/H in
0.2–1.0, while full-35 under identical numerics converges to six figures.**

Force-driven channel, two stationary fully-diffuse walls, ES-BGK at Pr = 2/3 and
omega = 0.81, body force applied exactly (a uniform force is a rigid translation in
velocity space, so central moments are untouched — `src/numerics/body_force.jl`).
Dimensionless flow rate Q against march length, at Kn_H = 1:

| TENDF (diffusion times) | Q full-35 | Q reduced-26 |
|---|---|---|
| 1.2 | 0.99288 | 1.15738 |
| 3.0 | 0.99530 | 1.85082 |
| 6.0 | 0.99531 | 2.63706 |

Full-35 plateaus (0.99530 → 0.99531, drift 1.4e-6). Reduced-26 grows without bound,
and cleanly: fitting Q ∝ T^p gives p = 0.513 and 0.511 across the two intervals, so
Q ~ sqrt(T). That exponent is the signature of a momentum sink that fails to close —
momentum diffusing into an effectively unbounded medium rather than pinning to the
channel half-width — which is what a wrong wall-normal momentum transport would do.

**It is not a timestep artifact.** Halving and quartering dt leaves the reduced result
essentially unchanged (1.85082 → 1.85747 → 1.86479, 0.4% per halving), so the growth
is intrinsic to the dynamics rather than to the once-per-step operator split.

**The Ma=100 chaos argument above does not cover this case, and that is the point.**
The section on the Ma=100 crossing jets observes separately-evolved reduced-vs-full
differences that *grow* (~0.15–0.30) and attributes them to trajectory drift, then
recommends assessing the reduction by the per-step residual instead. That reasoning is
sound where chaos is genuinely present. It is unavailable here: planar Poiseuille at
Kn ~ 1 is laminar, steady and one-dimensional, the companion full-35 run converges to
six digits under the same numerics, and the relative difference reaches 0.86 → 1.65 —
far past what was seen at Ma=100. Chaos cannot produce unbounded growth in a flow whose
companion solve converges to six figures.

The per-step residual and the closed-loop dynamics answer different questions. The
residual is an *instantaneous, linearised* measure: it says the reduced manifold lies
close to the current full-35 state. It does not bound the reduced *trajectory*. Because
`reduce26_moments` is an exact idempotent projection (verified: R∘R = R to machine
zero — the nine closed values depend only on retained moments, which R never touches),
a small residual is expected whether or not the closed loop is stable. So a shrinking
residual is necessary, not sufficient, and the two metrics can and here do disagree.

### What this does and does not indict

It indicts **this form of the reduction** — the per-cell per-step overwrite, which is
what `REDUCE26[]` implements and what the notes' `sec:oblique-reduce26` measured. It
does not settle the reduced *system*, because that system does not exist in code: the
35-moment fifth-order closure consumes the nine dropped moments (`S410 ← S310`,
`S311 ← S211`, `S221 ← S121, S211`, plus permutations), so carrying 26 moments requires
its own fifth-order closure rather than the 35-moment one with nine arguments
overwritten. Deriving those 21 closures for a 26-moment state is the open item.

This is the dynamical counterpart of a static result already on record (S. Bryngelson to
R. O. Fox, 2026-07-22): setting s310 to eq. (43) *as a value* while keeping it as an
evolved variable restores real eigenvalues on only ~25% of the planes where dropping it
*as a variable* works structurally. Overwrite-as-value and drop-as-variable were already
known to differ for hyperbolicity; they differ for the dynamics too.

### Reproducing

    julia --project=. test/probe_poiseuille.jl          # PS_KNH, PS_TEND, PS_CFL
    julia --project=. test/probe_reduce26_residual.jl   # per-moment closure residual

Caveat on the companion residual measurement: in wall-bounded Couette the closure
reproduces the *streamwise* odd moment S310 to 0.7–2% but misses the *wall-normal* S130
by up to 12% (worst-cell residuals 11–22%, concentrated in wall cells). Both are ~3x the
shear stress, which is exactly what the closure predicts at leading order — S310 =
S110*S400 with S400 → 3 for a near-Maxwellian — so magnitude alone is not evidence
against the reduction. The streamwise/wall-normal asymmetry is.

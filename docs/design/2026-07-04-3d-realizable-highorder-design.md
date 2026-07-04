# 3D realizability-robust high-order reconstruction (WENO5 + θ*-IDP) — design

**Date:** 2026-07-04. **Repo:** comp-physics/Riemann35.jl (production). **Goal:**
port the 1D realizability-robust high-order scheme to the full 3D solver, CPU and
GPU, as single-source device-safe code, opt-in via `spatial_order = 3`. The win
over the shipped MUSCL2 path is high order that stays realizable at the Ma=100
boundary — proven in 1D (5th order on smooth, survives the crossing where
unlimited WENO5 NaNs). This is the 3D follow-on to the 1D spec
(`roe-hyqmom-research`, `sec:hiorder-idp`).

## Scope (this spec)

- Reconstruction basis: **central-moment recon variables** (the existing
  `recon_dev.jl` `to_recon_vars_dev`), WENO5-Z faces. NOT log-Jacobi.
- Realizability: the **θ*-IDP flux limiter** (update-level), reusing the shipped
  device-safe `_state_realizable` and the θ* bisection from `riemann_flux_dev.jl`.
- CPU + GPU together, single-source device-safe (the `riemann_flux_dev.jl`
  pattern): NTuples, no heap allocation, no dynamic dispatch, no ENV/CPU-runtime
  calls inside kernels.
- Opt-in: `spatial_order = 3`; `spatial_order = 1, 2` byte-identical.

**Out of scope (deferred to v2):** log-Jacobi per-marginal reconstruction and the
machine-exact contacts it brings (needs the EC/Suliciu flux as F_HO). This spec
delivers the realizability-robustness win; contact-exactness is a follow-on.

## Architecture

Two new device-safe files + minimal integration:

- **`src/numerics/weno5_dev.jl`** (new, module `Weno5Dev`) — device-safe WENO5-Z
  (Borges) face reconstruction. `weno5_faces_dev(Vm2,Vm1,V0,Vp1,Vp2)` where each V
  is the recon-var `NTuple{35,Float64}` (from `to_recon_vars_dev`); returns
  `(vLface, vRface)` NTuples. Pure arithmetic (smoothness indicators + nonlinear
  weights). Consumers map back with `from_recon_vars_dev` and check
  `_state_realizable`, falling back to the cell mean per face on failure.
- **`src/numerics/idp_limiter_dev.jl`** (new, module `IdpLimiterDev`) — the θ*-IDP
  layer. `idp_theta_cell(Mi, λx,λy,λz, G⁻ˣ,G⁺ˣ,G⁻ʸ,G⁺ʸ,G⁻ᶻ,G⁺ᶻ, Mlo) ->
  (θ⁻ˣ,θ⁺ˣ,...)` : given the cell's first-order anchor `Mlo`, the six HO−LO flux
  corrections `G`, and the per-axis `λ_d = dt/dx_d`, solve the per-face θ* (the
  six one-sided half-updates `Mlo ∓ 6 λ_d θ G`, each realizable) by bisection on
  `_state_realizable`. Interface θ is the min over the two adjacent cells,
  computed by the caller. Reuses `theta_star_update`-style bisection and
  `_state_realizable` (imported from `riemann_flux_dev.jl`'s module).

Integration (modify, `order == 3` branch only; `1, 2` untouched):

- **`src/numerics/highorder_3d.jl`** (CPU `residual_line`/`residual_ho_3d!`):
  add the two-pass structure for `order == 3` — flux pass (store F_HO via WENO5 +
  F_LO via HLL at all faces) then limit+update pass (per cell: gather 6 face
  corrections, θ*, blend, update).
- **`gpu/residual3d_gpu.jl`** (GPU): same two passes as kernels — a flux kernel
  storing F_HO/F_LO per face into the existing `flat` face-scratch buffers (add a
  second buffer for the anchor), and a limit+update kernel doing the per-cell θ*.
- **Exposure:** `simulation_runner` (CPU) and `march3d_gpu!` / `run_gpu_3d` (GPU)
  accept `spatial_order = 3` → the new path. Default unchanged.

## Mechanics

**WENO5 reconstruction (per direction).** Central moments are NONLINEAR in the
raw moments, so WENO5 applied directly to central-var *cell averages* caps at 2nd
order (§9(k'); this is exactly the bug the 1D scheme hit until the convolution
step was added). The 5th-order pipeline, per direction, is therefore:
(1) deconvolve the RAW-moment cell averages to raw point values (linear,
order-preserving 5-point stencil); (2) map point values to recon vars
(`to_recon_vars_dev`); (3) convolve recon-var point values back to recon-var cell
averages (5-point stencil); (4) WENO5-Z on the recon-var averages → faces;
(5) `from_recon_vars_dev` → raw-moment faces. Both stencils (`deconv`, `conv`,
device-safe arithmetic, ported from `roe1d.jl`) are needed for genuine 5th order.
A per-cell smoothness gate drops the deconvolution correction near jumps (keeps
the cell average), preventing Gibbs. If a face fails `_state_realizable`, replace
it with the cell mean (first-order) for that face — the reconstruction-layer net.

**θ*-IDP limiter (joint 6-face Zhang–Shu).** The update
`Mᵢⁿ⁺¹ = Mᵢ − Σ_d λ_d(F_{d,+} − F_{d,−})` depends on all six faces. Blend each
face `F = F_LO + θ_face(F_HO − F_LO)`. Decompose the update into six one-sided
half-updates `H_face = Mlo ∓ 6 λ_d θ_face G_face` (G = F_HO − F_LO,
Mlo = the first-order update); require each `H_face` realizable ⇒ per-face θ* via
bisection on `_state_realizable`; interface θ = min over the two adjacent cells.
Convexity of the cone ⇒ `Mᵢⁿ⁺¹ = (1/6)Σ H_face` realizable. Conservative (blends
fluxes). This forces the **two-pass residual** (fluxes stored, then limited).

**Anchor + CFL.** `F_LO` = first-order HLL, realizability-preserving under
`λ_d v_max ≤ 1/6` (½ decomposition × 3 dims); SSP-RK3 (convex combination of
substeps) inherits IDP. Measure the constant on the Ma=100 case.

**Realizability test.** The shipped device-safe `_state_realizable` (ρ>0, per-axis
b₂>0, b₃>0 = Hamburger K ≥ 1+q̂²), reused verbatim. No new realizability code.

## Validation (success bar)

Intrinsic:
1. **Order (smooth 3D):** sinusoidal density, quadrature 3-pt-Gauss cell-average
   IC (avoid the §9(j') order-2 artifact), nx = 16/24/32/48, L1 vs a fine
   reference → high order (target ≈5). Guard: θ*-active ≈ 0 on smooth (limiter
   off ⇒ order preserved).
2. **Ma=100 crossing jets (headline):** WENO5+θ*-IDP survives at high order —
   realizable, NaN-free, ρ>0 — where it matters. θ* fires only near the front.
3. **Conservation:** total moments invariant (blends fluxes).
4. **CPU/GPU parity:** bitwise where possible, else ulp-class (the documented
   @fastmath-closure tolerance, GOLDEN_TOL=1e-10), matched-dt.
5. **Defaults byte-identical:** `spatial_order = 1, 2` unchanged (golden suite).

Head-to-head: vs MUSCL2 (order 5 vs 2, Ma=100 survival, wall-clock); vs
WENO5-no-limiter (breaks at Ma=100 — necessity).

Compute note: Ma=100 3D and parity are GPU-bound (V100/H100); order studies stay
small (nx³).

## Error handling (safety hierarchy)

1. Face realizability fallback (reconstruction → cell mean per bad face).
2. θ* = 0 → pure first-order HLL (universal net at a degenerate/very-out face).
3. First-order-anchor check: if `Mlo` itself is non-realizable, CFL is violated →
   the run should reduce dt (guard + report).
4. Existing realizability projection as belt-and-suspenders; its firing count is a
   diagnostic (should be ~0 if the θ* layer is correct).

## Tests

- Unit (device-safe, CPU-run): `weno5_faces_dev` recovers a smooth polynomial to
  5th order; `idp_theta_cell` vs brute-force bisection on `_state_realizable`
  (θ=1 interior, ∈(0,1) boundary, 0 when HO wildly out); conservation of the
  blended update.
- GPU: a kernel-compile smoke test for both new files (the `_fhat`-style
  InvalidIR guard) + CPU/GPU parity on a small box.
- Integration: the intrinsic tests above; the `spatial_order=1,2` golden
  regression.

## Open risks (resolve during implementation)

1. The 3D CFL constant for HLL-IDP on the 35-moment cone (fan-convexity
   guarantees one; the value 1/6 is the decomposition target, unmeasured).
2. Two-pass GPU memory: storing F_HO + F_LO at all faces doubles the face-scratch
   buffer; confirm it fits the H200/V100 budget at production nx.
3. If θ*-active > 0 on smooth (limiter misfiring at smooth extrema), add a
   smooth-extrema relaxation (skip layer 2 where `smooth_cell`-analog holds and
   both LO/HO updates are realizable with margin). Flag, don't pre-engineer.
4. The deconv/conv pipeline (in Mechanics) is required for 5th order and is in the
   design; the residual risk is whether it composes cleanly per-direction in 3D
   (halo width: WENO5 needs 3 ghost cells, deconv/conv another ±2 — confirm the
   existing halo `g` is wide enough or widen it). Measure order early to confirm.
5. `weno5_dev.jl`/`idp_limiter_dev.jl` must be included by BOTH the package
   (`src/Riemann35.jl`) and the standalone GPU modules — mirror how
   `riemann_flux_dev.jl` is dual-included; watch the @noinline @fastmath device-
   helper parity gotcha (shared helpers must be @noinline for byte parity).

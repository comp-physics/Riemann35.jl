# Stage-wise BGK (`stage_bgk`) + GPU support for both new opt-ins — design + plan

**Date:** 2026-07-02 · **Status:** approved (conversation) · **Branch:** `feat/stage-bgk-gpu`

## Problem

After `ho_pressure_recon` (PR #9) the order-2 stationary-contact gate still shows
a saturating residual (maxvel 0.0027/0.0049/0.0060 at Nx=32/64/128): SSP-RK3
stages are collisionless (BGK applied once per full step), so at Kn=0 stage 1
pumps transient M300 from the M400 = 3p²/ρ variation, later stages flux it into
pressure, and the u-error persists (collision preserves u). First-order is exact
because it is single-stage. Separately, the GPU march has **no collision at all**
and PR #9's `ho_pressure_recon` is CPU-only. User requirements: BGK per RK
substep, and a DRY implementation of both features that works on GPU.

## Design

### Single-source helpers (src/numerics/recon_dev.jl, module ReconDev)

Plain (non-fastmath) device-compatible NTuple helpers, used verbatim by CPU and GPU:

- `bgk_relax_tup(C::NTuple{35}, dt, Kn) -> NTuple{35}` — exact-exponential BGK:
  ρ,u,v,w,Θ from C (Θ floored at 1e-14, matching `collision35` with positivity on);
  `tc = Kn/(2ρ√Θ)`; `e = exp(-dt/tc)`; Maxwellian target
  `MG = from_recon_vars_dev(ρ,u,v,w,Θ,Θ,Θ, <Maxwellian S: S400=S040=S004=3,
  S220=S202=S022=1, rest 0>)` — **reuses the existing single-source kernel as the
  Maxwellian builder**; returns `MG - e·(MG - C)` (a convex combination ⇒
  realizability preserved). ρ≤0 returns C unchanged. Kn=0 ⇒ e=0 (instant
  Maxwellianization); Kn=Inf ⇒ e=1 (no-op).
- `pressurize_recon_tup(V)` / `depressurize_recon_tup(V)` — slots 5–7 × / ÷ V[1].
  CPU `to_recon_vars`/`from_recon_vars` delegate to these (same arithmetic as the
  current inline ops); GPU kernels call them directly.

Value-parity test: `bgk_relax_tup` vs legacy `collision35` (rtol 1e-12). Legacy
`collision35` and the default post-step CPU path stay byte-identical (untouched).

### CPU stage BGK

- `step_highorder_3d!` gains `stage_bgk_kn=nothing`; when a number, after each
  stage's `_project_interior!` apply `bgk_relax_tup` to every interior cell with
  the full `dt` (each SSP building block is a Lie-split forward-Euler step; convex
  combinations keep first-order splitting consistency — same formal order as the
  current once-per-step splitting).
- simulation_runner: opt-in `stage_bgk = get(params, :stage_bgk, false)`
  (order-2 only); passes `stage_bgk_kn = Kn`; **skips the post-step collision
  block** when active (no double relaxation). Default byte-identical.

### GPU support (both features)

- `pressure_recon::Bool=false` threaded through `run_gpu_3d` →
  `march3d_gpu!`/`march3d_slab_gpu!` → `residual3d_box_gpu!` → kernels:
  pressurize after `to_recon_vars_tup` (vbuf fill), depressurize before each of
  the 4 `from_recon_vars_tup` sites in `_face_flux_core`.
- **Limiter + pressure_recon supported** (added on user request): `prec` is
  threaded down the θ chain (`scaling_theta_dev` → `_faces_realizable_dev` →
  `is_realizable_recon_dev`, all defaulting `prec=false` for byte-identity), with
  the depressurize at the single conversion point. Verified: engaged-limiter θ
  (0.00509…) identical through CPU wrapper (flag), device oracle (prec=true),
  and the C-form reference; contact stays machine-exact with the limiter on
  (both gated in test_rodney_cases.jl); GPU `lpb` mode parity 3.2e-13.
- `stage_bgk::Bool=false, Kn::Real=Inf` on the marches: `_bgk_kernel!` (per-cell
  `bgk_relax_tup` on the `(35,ncl)` view) launched after each stage's `proj!` in
  `_rk3_step!`. This is also the GPU's first collision capability (default off ⇒
  existing byte-identical collisionless behavior).

### Tests

CPU (test/test_rodney_cases.jl): (1) helper vs collision35 value parity;
(2) **acceptance: contact gate with `ho_pressure_recon=true, stage_bgk=true` ⇒
maxvel, pdev < 1e-12 (machine-exact prediction)**; (3) flag-off bitwise identity;
(4) finite-Kn (0.01) stage_bgk run: finite, mass conserved to 1e-10 (BGK
conserves ρ, momentum, energy).
GPU: `gpu/validate_stage_bgk.jl` — CPU-vs-GPU trajectory parity on a tiny case
with both flags on (and flags-off parity via existing validators); run via sbatch
on a PACE GPU node (account gts-sbryngelson3).

### Non-goals

Changing default behavior anywhere (all opt-in, byte-identical); IMEX treatment
of stiff collision (exact-exponential is unconditionally stable already);
per-stage-weighted dt variants.

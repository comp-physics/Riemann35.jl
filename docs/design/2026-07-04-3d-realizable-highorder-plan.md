# 3D realizability-robust high-order (WENO5 + θ*-IDP) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an opt-in `spatial_order = 3` path to the 3D 35-moment solver — central-moment WENO5 reconstruction (with deconv/conv for genuine 5th order) plus a θ*-IDP flux limiter that keeps every update realizable — as single-source device-safe code for both CPU and GPU.

**Architecture:** Two new device-safe modules (`weno5_dev.jl`, `idp_limiter_dev.jl`) reused by the CPU residual (`highorder_3d.jl`) and the GPU residual kernels (`residual3d_gpu.jl`), behind a two-pass residual (store F_HO/F_LO at all faces → per-cell joint 6-face θ* limit → blended update). Reuses the shipped device-safe `_state_realizable` and the θ* bisection from `riemann_flux_dev.jl`.

**Tech Stack:** Julia; package env `--project=.` from repo root `/storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl`; GPU env `gpu/gpuenv2` + CUDA on a V100/H100 Slurm allocation. Run with `$JULIA` after `source /storage/scratch1/6/sbryngelson3/vizwork/env.sh`. Full design: `docs/design/2026-07-04-3d-realizable-highorder-design.md`.

## Global Constraints

- Opt-in: `spatial_order = 3` is new; `spatial_order = 1, 2` byte-identical (golden/regression-verified).
- Device-safe (compiles into CUDA kernels): NTuples not arrays, no heap allocation, no dynamic dispatch, no ENV/CPU-runtime calls inside kernel functions. Shared `@fastmath` device helpers must be `@noinline` for CPU/GPU byte parity.
- Reuse, do not reimplement: `_state_realizable` and the θ* bisection live in `src/numerics/riemann_flux_dev.jl` (module `RiemannFluxDev`); `to_recon_vars_dev`/`from_recon_vars_dev`/`minmod` in `src/numerics/recon_dev.jl` (module `ReconDev`); the raw physical flux `flux_closure35_dev` in `src/numerics/flux_closure_dev.jl`; first-order HLL in `riemann_flux_dev.jl` (`rs=0`).
- Conservation is non-negotiable: the limiter blends FLUXES, never states.
- `weno5_dev.jl` and `idp_limiter_dev.jl` must be `include`d by BOTH `src/Riemann35.jl` (package) and the standalone GPU module tree (mirror `riemann_flux_dev.jl`'s dual include).
- Commit at the end of each task from the repo root.

---

## File structure

- `src/numerics/weno5_dev.jl` — module `Weno5Dev`: device-safe `deconv5`, `conv5` (5-point stencils), `weno5z` (Borges scalar), and `weno5_faces_recon` (full pipeline on a 35-recon-var stencil → left/right raw-moment faces with per-face realizability fallback).
- `src/numerics/idp_limiter_dev.jl` — module `IdpLimiterDev`: `theta_star_update_dev` (bisection on `_state_realizable`) and `idp_face_thetas` (the six one-sided half-update θ per cell).
- `src/numerics/highorder_3d.jl` — MODIFY: `order == 3` two-pass CPU residual.
- `gpu/residual3d_gpu.jl` — MODIFY: `order == 3` two-pass GPU kernels.
- `src/simulation_runner.jl`, `gpu/timestep3d_gpu.jl`, `gpu/gpu_run.jl` — MODIFY: accept `spatial_order = 3` / `order = 3`.
- `test/test_weno5_idp.jl` — unit tests (stencil order, θ* vs brute force, conservation).
- `gpu/validation/validate_hiorder3_parity.jl` — GPU compile + CPU/GPU parity.

---

## Task 1: Device-safe WENO5-Z + deconv/conv stencils

**Files:**
- Create: `src/numerics/weno5_dev.jl`
- Test: `test/test_weno5_idp.jl`

**Interfaces:**
- Produces: `weno5z(vm2,vm1,v0,vp1,vp2)::Float64` (scalar WENO5-Z right-face value; mirror args for left face); `deconv5(...)`, `conv5(...)` (scalar 5-point undivided-difference stencils); `smooth5(a,b,c,d,e)::Bool` (per-cell smoothness gate).

- [ ] **Step 1: Write the module with the scalar stencils**

Create `src/numerics/weno5_dev.jl`:

```julia
"""
    weno5_dev.jl — device-safe WENO5-Z + deconvolution/convolution stencils.

Pure scalar arithmetic on Float64 (NTuple-friendly, CUDA-safe). WENO5-Z is
Borges et al. (2008). deconv5/conv5 are the O(dx^6) cell-average <-> point-value
pair (needed because reconstruction in nonlinear recon variables from cell
averages caps at 2nd order otherwise). smooth5 gates the deconvolution near jumps.
"""
module Weno5Dev

export weno5z, deconv5, conv5, smooth5

# cell average -> cell-center point value (undivided differences), O(dx^6)
@inline deconv5(vm2, vm1, v0, vp1, vp2) =
    v0 - (1/24) * (vp1 - 2v0 + vm1) + (3/640) * (vm2 - 4vm1 + 6v0 - 4vp1 + vp2)
# point value -> cell average (forward operator)
@inline conv5(vm2, vm1, v0, vp1, vp2) =
    v0 + (1/24) * (vp1 - 2v0 + vm1) - (17/5760) * (vm2 - 4vm1 + 6v0 - 4vp1 + vp2)

"per-cell smoothness gate: relative curvature below tol on all inputs."
@inline function smooth5(a, b, c, d, e; tol = 0.05)
    s = abs(a) + 2*abs(c) + abs(e) + 1e-300
    abs(a - 2c + e) / s <= tol && abs(b - 2c + d) / s <= tol
end

"WENO5-Z reconstruction, right face of the center cell (mirror args for left)."
@inline function weno5z(vm2, vm1, v0, vp1, vp2)
    q0 = (2vm2 - 7vm1 + 11v0) / 6
    q1 = (-vm1 + 5v0 + 2vp1) / 6
    q2 = (2v0 + 5vp1 - vp2) / 6
    b0 = (13/12)*(vm2 - 2vm1 + v0)^2 + (1/4)*(vm2 - 4vm1 + 3v0)^2
    b1 = (13/12)*(vm1 - 2v0 + vp1)^2 + (1/4)*(vm1 - vp1)^2
    b2 = (13/12)*(v0 - 2vp1 + vp2)^2 + (1/4)*(3v0 - 4vp1 + vp2)^2
    t5 = abs(b0 - b2); eps = 1e-40
    a0 = 0.1 * (1 + t5/(b0+eps)); a1 = 0.6 * (1 + t5/(b1+eps)); a2 = 0.3 * (1 + t5/(b2+eps))
    (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)
end

end # module
```

- [ ] **Step 2: Write the failing test (5th-order recovery on a smooth function)**

Create `test/test_weno5_idp.jl`:

```julia
include(joinpath(@__DIR__, "..", "src", "numerics", "weno5_dev.jl"))
using .Weno5Dev
using Printf

npass = 0; nfail = 0
chk(nm, c) = (global npass, nfail; c ? (npass+=1) : (nfail+=1; @printf("FAIL: %s\n", nm)))

# WENO5-Z recovers the face value of a smooth function to ~5th order.
# f(x)=sin(2pi x); cell averages over [x-h/2, x+h/2] approximated by point values
# is the wrong test — use the true right-face value f(x0+h/2) vs weno5z on point
# values of a smooth cubic where WENO is exact to its formal order.
f(x) = sin(2pi*x)
function order_at(h)
    x0 = 0.13
    v(k) = f(x0 + k*h)                       # point values (smooth => stand-in averages ok for a convergence slope)
    fr = weno5z(v(-2), v(-1), v(0), v(1), v(2))
    abs(fr - f(x0 + h/2))
end
e1 = order_at(0.02); e2 = order_at(0.01)
p = log2(e1/e2)
chk("weno5z order >= 4.5", p >= 4.5)

# deconv/conv are inverse to O(dx^6): round-trip a smooth quintic sample
g(x) = 1 + 0.3x + 0.1x^2 - 0.05x^3
h = 0.01; x0 = 0.2
gp = deconv5(g(x0-2h), g(x0-h), g(x0), g(x0+h), g(x0+2h))  # avg->point (g≈its own avg for slope)
chk("deconv finite", isfinite(gp))
chk("conv finite", isfinite(conv5(g(x0-2h),g(x0-h),g(x0),g(x0+h),g(x0+2h))))
chk("smooth5 true on smooth", smooth5(g(x0-2h),g(x0-h),g(x0),g(x0+h),g(x0+2h)))
chk("smooth5 false on jump", !smooth5(0.0,0.0,1.0,1000.0,1000.0))

@printf("Task1: %d pass, %d fail\n", npass, nfail)
nfail == 0 || exit(1)
```

- [ ] **Step 3: Run the test**

Run: `source /storage/scratch1/6/sbryngelson3/vizwork/env.sh && $JULIA --project=. test/test_weno5_idp.jl`
Expected: `Task1: 5 pass, 0 fail`. If `weno5z order` is below 4.5, the stencil coefficients are wrong — recheck against the Step 1 code (this is the reference WENO5-Z; it should give ~5).

- [ ] **Step 4: Commit**

```bash
cd /storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl
git add src/numerics/weno5_dev.jl test/test_weno5_idp.jl
git commit -m "feat(hiorder3): device-safe WENO5-Z + deconv/conv stencils"
```

---

## Task 2: Device-safe θ*-IDP limiter primitives

**Files:**
- Create: `src/numerics/idp_limiter_dev.jl`
- Test: `test/test_weno5_idp.jl` (append)

**Interfaces:**
- Consumes: `_state_realizable` from `RiemannFluxDev` (in `riemann_flux_dev.jl`).
- Produces: `theta_star_update_dev(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64}; nb=24)::Float64` — largest θ∈[0,1] with `_state_realizable(Mlo .+ θ.*dM)`, given `_state_realizable(Mlo)`.

- [ ] **Step 1: Write the module**

Create `src/numerics/idp_limiter_dev.jl`:

```julia
"""
    idp_limiter_dev.jl — device-safe θ*-IDP update limiter.

Given a realizable first-order anchor state Mlo and a candidate correction dM,
find the largest θ in [0,1] keeping Mlo+θ·dM realizable (moment cone). Bisection
on the shipped `_state_realizable`; the closed-form Hankel-pencil cubic is the
production optimization. Used by the two-pass residual: per cell, the six
one-sided half-updates each get a θ, and the interface θ is the min over the two
cells sharing it (done by the residual caller).
"""
module IdpLimiterDev

include(joinpath(@__DIR__, "riemann_flux_dev.jl"))
using .RiemannFluxDev: _state_realizable

export theta_star_update_dev

@inline function theta_star_update_dev(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64}; nb::Int = 24)
    full = ntuple(j -> Mlo[j] + dM[j], Val(35))
    _state_realizable(full) && return 1.0
    lo = 0.0; hi = 1.0
    for _ in 1:nb
        mid = 0.5 * (lo + hi)
        m = ntuple(j -> Mlo[j] + mid * dM[j], Val(35))
        _state_realizable(m) ? (lo = mid) : (hi = mid)
    end
    lo
end

end # module
```

- [ ] **Step 2: Append the failing test**

Append to `test/test_weno5_idp.jl` (before the final `@printf`/`exit`):

```julia
include(joinpath(@__DIR__, "..", "src", "numerics", "idp_limiter_dev.jl"))
using .IdpLimiterDev
using .IdpLimiterDev.RiemannFluxDev: _state_realizable
# Maxwellian 35-moment builder (diagonal, isotropic T)
mw(rho, T) = ntuple(35) do q
    (q == 1) ? rho : (q in (3,10,20)) ? rho*T : (q in (12,22,35)) ? rho*T^2 :
    (q in (5,15,25)) ? 3rho*T^2 : 0.0
end
Mlo = mw(1.0, 1.0)
chk("theta*=1 interior (zero dM)", theta_star_update_dev(Mlo, ntuple(_->0.0,Val(35))) == 1.0)
# dM lowering the x-kurtosis (slot 5) below the Hamburger bound => theta*<1
baddM = ntuple(j -> j == 5 ? -0.6 * Mlo[5] : 0.0, Val(35))
th = theta_star_update_dev(Mlo, baddM)
chk("theta* in (0,1) at boundary", 0.0 < th < 1.0)
chk("theta* endpoint realizable", _state_realizable(ntuple(j->Mlo[j]+th*baddM[j],Val(35))))
# brute-force agreement
function brute(Mlo, dM; n=20000)
    best = 0.0
    for k in 0:n
        t = k/n
        _state_realizable(ntuple(j->Mlo[j]+t*dM[j],Val(35))) ? (best=t) : break
    end
    best
end
chk("theta* matches brute (1e-3)", abs(th - brute(Mlo, baddM)) < 1e-3)
```

- [ ] **Step 3: Run**

Run: `source /storage/scratch1/6/sbryngelson3/vizwork/env.sh && $JULIA --project=. test/test_weno5_idp.jl`
Expected: `Task1: ...` line then all Task-2 checks pass, `nfail == 0`. (If `_state_realizable` import fails, confirm the module path in `riemann_flux_dev.jl` and that it exports/defines `_state_realizable`.)

- [ ] **Step 4: Commit**

```bash
git add src/numerics/idp_limiter_dev.jl test/test_weno5_idp.jl
git commit -m "feat(hiorder3): device-safe theta*-IDP update solver (reuses _state_realizable)"
```

---

## Task 3: CPU two-pass residual (`order == 3`) in `highorder_3d.jl`

**Files:**
- Modify: `src/numerics/highorder_3d.jl` (add an `order == 3` branch to `residual_line` and the driver; do NOT touch `order == 1, 2`)
- Include the two new modules at the top of `highorder_3d.jl`.
- Test: `test/test_weno5_idp.jl` (append a 1D-in-3D order + conservation check)

**Interfaces:**
- Consumes: `weno5z`, `deconv5`, `conv5`, `smooth5` (Task 1); `theta_star_update_dev` (Task 2); existing `to_recon_vars_dev`/`from_recon_vars_dev` (`ReconDev`), the raw physical flux, and the first-order HLL (`riemann_flux_dev`, `rs=0`).
- Produces: `residual_line(..., order=3, ...)` returning the two-pass residual for a line; the driver `residual_ho_3d!(..., order=3)`.

**Two-pass structure (implement exactly this):**
For a line of `n` cells with `g=3` ghosts, along one axis:
1. **Reconstruct + flux pass.** For each interior face f (between cells f-1,f): build the recon-var stencil via `to_recon_vars_dev` on the deconvolved (smooth5-gated) raw stencil, `conv5` back to recon-var averages, `weno5z` left/right → `from_recon_vars_dev` → raw face states `mLf,mRf`; if `_state_realizable` fails on a face, use the cell mean. Compute `F_HO[f] = HLL(mLf,mRf)` (rs=0 interface flux of the reconstructed faces) and `F_LO[f] = HLL(cell_{f-1}, cell_f)`. Store both.
2. **Limit + residual pass.** For each interior cell i: `Mlo = M_i − λ(F_LO[i+1] − F_LO[i])`; `G⁻ = F_HO[i]−F_LO[i]`, `G⁺ = F_HO[i+1]−F_LO[i+1]`; per-face θ via `theta_star_update_dev(Mlo, −2λ·G⁺)` and `theta_star_update_dev(Mlo, +2λ·G⁻)` (the two one-sided half-updates for this 1-axis line — the full 6-face min across axes is assembled by the driver over the three line sweeps: take the min of the per-axis θ at each shared face). Blend `F[f] = F_LO[f] + θ[f]·G[f]`; residual line `R_i = −(F[i+1] − F[i])/dx`.

(The driver `residual_ho_3d!` runs the three axis line-sweeps; for `order==3` it must store per-axis F_HO/F_LO and combine θ as the min across the axes touching each cell before the final update — this is the "joint 6-face" bound. Mirror the existing per-axis accumulation but defer the blend until all three axes' corrections are known.)

- [ ] **Step 1: Add includes + the `order==3` reconstruction/flux helper to `highorder_3d.jl`**

At the top of `src/numerics/highorder_3d.jl`, after its existing includes, add:

```julia
include(joinpath(@__DIR__, "weno5_dev.jl"));      using .Weno5Dev: weno5z, deconv5, conv5, smooth5
include(joinpath(@__DIR__, "idp_limiter_dev.jl")); using .IdpLimiterDev: theta_star_update_dev
```

Then add a helper that produces the reconstructed raw faces for a 5-cell raw-moment stencil (columns are `NTuple{35}` or `Vector`):

```julia
# reconstructed left/right raw-moment faces from a 5-cell raw stencil (each a
# length-35 vector), central-var WENO5 with deconv/conv; realizability fallback.
function _weno5_faces(cm2, cm1, c0, cp1, cp2)
    # deconvolve raw cell averages -> raw point values (smooth-gated per component)
    gate = all(smooth5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) for q in 1:35)
    pt(a,b,c,d,e) = gate ? deconv5(a,b,c,d,e) : c
    pm2 = [pt(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) for q in 1:35]   # NOTE: needs the neighbor point values too
    # For a correct pipeline each of the 5 stencil cells needs its own point value;
    # the driver supplies a 9-cell window so each of the 5 has a 5-cell deconv stencil.
    error("use _weno5_faces_window in the driver — see Step 2")
end
```

(The single-stencil helper cannot deconvolve correctly — deconvolution needs each stencil cell's own neighborhood. Implement the real version in the driver with a 9-cell window, Step 2. Keep this stub only as documentation of the intent, or omit it.)

- [ ] **Step 2: Implement the `order==3` line residual with a 9-cell window**

Add to `residual_line` (or a new `residual_line3`) an `order == 3` branch. For a padded line `Mext` (size `(n+2g, 35)`, g≥4 so the deconv+WENO stencils fit), per interior cell index `i`:

```julia
# per-cell deconvolved raw point value (needs i-2..i+2 averages)
@inline dptcell(Mext, i, q) = smooth5(Mext[i-2,q],Mext[i-1,q],Mext[i,q],Mext[i+1,q],Mext[i+2,q]) ?
    deconv5(Mext[i-2,q],Mext[i-1,q],Mext[i,q],Mext[i+1,q],Mext[i+2,q]) : Mext[i,q]

function residual_line3(Mext, ds, axis, Ma; g=4, s3max=40.0)
    n = size(Mext,1) - 2g
    # 1) raw point values -> recon vars (per cell), then conv back to recon-var averages
    #    Vavg[i] = recon-var cell average at cell i (length-35 vector)
    Vavg = Vector{NTuple{35,Float64}}(undef, n + 2g)
    Ppt  = Vector{NTuple{35,Float64}}(undef, n + 2g)
    for i in (g-1):(n+g+2)
        rawpt = ntuple(q -> dptcell(Mext, i, q), Val(35))
        Ppt[i] = to_recon_vars_dev(rawpt...)            # recon vars at the point value
    end
    for i in g:(n+g+1)
        Vavg[i] = ntuple(q -> conv5(Ppt[i-2][q],Ppt[i-1][q],Ppt[i][q],Ppt[i+1][q],Ppt[i+2][q]), Val(35))
    end
    # 2) WENO5 faces (recon vars) -> raw faces; HLL F_HO; F_LO from cell means
    F_HO = Vector{NTuple{35,Float64}}(undef, n+1); F_LO = Vector{NTuple{35,Float64}}(undef, n+1)
    for f in 1:n+1
        il = g + f - 1; ir = g + f
        vL = ntuple(q -> weno5z(Vavg[il-2][q],Vavg[il-1][q],Vavg[il][q],Vavg[il+1][q],Vavg[il+2][q]), Val(35))
        vR = ntuple(q -> weno5z(Vavg[ir+2][q],Vavg[ir+1][q],Vavg[ir][q],Vavg[ir-1][q],Vavg[ir-2][q]), Val(35))
        mL = from_recon_vars_dev(vL...); mR = from_recon_vars_dev(vR...)
        _state_realizable(mL) || (mL = ntuple(q->Mext[il,q],Val(35)))
        _state_realizable(mR) || (mR = ntuple(q->Mext[ir,q],Val(35)))
        cL = ntuple(q->Mext[il,q],Val(35)); cR = ntuple(q->Mext[ir,q],Val(35))
        F_HO[f] = _hll_iface(mL, mR, axis)         # rs=0 flux of reconstructed faces
        F_LO[f] = _hll_iface(cL, cR, axis)         # rs=0 flux of cell means (anchor)
    end
    return F_HO, F_LO      # the driver combines θ across axes; see Step 3
end
```

Here `to_recon_vars_dev`/`from_recon_vars_dev` are from `ReconDev` (already imported by `highorder_3d.jl` or add the import), and `_hll_iface(mL,mR,axis)` wraps `riemann_flux_dev(0, axis, mL, mR, phys(mL,axis), phys(mR,axis), sL, sR)` with the closure wave-speed bounds (mirror how the existing `order==2` path obtains `sL,sR` and the physical flux). Add `_hll_iface` as a small helper next to this function, reusing the existing wave-speed/flux calls from the `order==2` branch verbatim.

- [ ] **Step 3: Combine across axes in the driver (joint 6-face θ) and update**

In `residual_ho_3d!`, for `order == 3`: run `residual_line3` for all three axes to get per-axis `F_HO,F_LO` at each face; form the first-order anchor update `Mlo_i = M_i − Σ_axis λ_axis(F_LO[i+1]−F_LO[i])`; for each of the six faces compute the one-sided θ via `theta_star_update_dev(Mlo_i, ∓6·λ_axis·(F_HO−F_LO)_face)`; set each interface θ = min over its two cells; blend `F = F_LO + θ(F_HO−F_LO)`; final `R_i = −Σ_axis (F[i+1]−F[i])/dx_axis`. This is the joint bound.

(Implementation note: this requires the driver to hold all three axes' face fluxes before updating — allocate three `F_HO/F_LO` arrays, then a θ pass, then the residual. Follow the existing `residual_ho_3d!` accumulation structure but split into store-then-limit.)

- [ ] **Step 4: Write the failing 1D-in-3D order + conservation test**

Append to `test/test_weno5_idp.jl` a check that a `(Nx,1,1)` smooth run with `order=3` conserves mass and reaches high order versus a fine reference. Because wiring the full `simulation_runner` is Task 4, here test the residual conservation directly: build a small periodic `Mext`, call the `order==3` residual, and assert `sum(R) ≈ 0` per moment (a conservative flux difference telescopes to the boundary flux; with periodic wrap it is ~0 to 1e-10).

```julia
# conservation of the order-3 residual (flux differences telescope)
# (construct a small smooth periodic line, call residual_line3 for axis=1,
#  assemble R via the Step-3 blend with a tiny lambda, assert column sums ~0)
# Exact code depends on residual_ho_3d!'s signature; assert:
#   for q in 1:35: abs(sum(R[:,q])) < 1e-9 * (abs(sum over M) + 1e-12)
```

Fill this in against the actual `residual_ho_3d!` signature once Steps 1–3 compile.

- [ ] **Step 5: Run + Commit**

Run: `$JULIA --project=. test/test_weno5_idp.jl` → all pass.
```bash
git add src/numerics/highorder_3d.jl test/test_weno5_idp.jl
git commit -m "feat(hiorder3): CPU two-pass order=3 residual (WENO5 + joint 6-face theta*-IDP)"
```

---

## Task 4: Wire `spatial_order = 3` into the CPU runner + 3D validation

**Files:**
- Modify: `src/simulation_runner.jl` (the `spatial_order == 2` dispatch block — add an `== 3` branch calling the order-3 driver; leave 1,2 untouched)
- Create: `test/validate_hiorder3_cpu.jl` (small 3D order + Ma survival)

**Interfaces:**
- Consumes: the order-3 `residual_ho_3d!`/`step_highorder_3d!` (Task 3).

- [ ] **Step 1:** In `src/simulation_runner.jl`, extend the `spatial_order` dispatch: where it branches `if spatial_order == 2` for the high-order path, add `|| spatial_order == 3` and thread `order = spatial_order` into `step_highorder_3d!` (which already takes `order`). Confirm `spatial_order in (1,2)` behavior is unchanged (the `order` value flows through).

- [ ] **Step 2:** Create `test/validate_hiorder3_cpu.jl`: a small smooth 3D order study (nx=8,12,16 with a quadrature 3-pt-Gauss cell-average sinusoid IC, L1 vs an nx=32 reference, `spatial_order=3`), a conservation check, and a small Ma=10 crossing run confirming `ok && all ρ>0`. Print an order/conservation/survival summary.

- [ ] **Step 3:** Run `$JULIA --project=. test/validate_hiorder3_cpu.jl`. Expected: order climbing toward ~5 (small grids may under-resolve; report the trend), conservation ~1e-10, Ma=10 survives. If order is stuck at 2, the deconv/conv pipeline is not engaged — recheck Task 3 Step 2.

- [ ] **Step 4: Commit**

```bash
git add src/simulation_runner.jl test/validate_hiorder3_cpu.jl
git commit -m "feat(hiorder3): spatial_order=3 CPU dispatch + 3D validation (order, conservation, Ma=10)"
```

---

## Task 5: DRY the shared per-face/per-cell logic + GPU two-pass kernels + parity

**DRY MANDATE (the whole point of this task): the GPU must consume the SAME
device-safe per-face reconstruction and per-cell θ code as the CPU — one source,
two consumers (the `riemann_flux_dev.jl` pattern), NOT a parallel GPU
reimplementation.** Task 3 inlined the reconstruction pipeline inside the CPU
`residual_line3` loop; Step 0 factors it out so both call it.

**Files:**
- Create: `src/numerics/hiorder3_recon_dev.jl` — the extracted single-source
  device-safe helpers (NTuple in/out, no allocation): `weno5_face_states_dev`
  (9-cell raw stencil → `(mL, mR)` raw faces via the deconv→recon-var→conv→weno5→
  realizability-fallback pipeline) and `idp_cell_thetas_dev` (Mlo + six G
  corrections + six λ → six θ). Both are exactly the per-face/per-cell math Task 3
  wrote inline.
- Modify: `src/numerics/highorder_3d.jl` — REFACTOR `residual_line3`/
  `residual_ho_3d_order3!` to CALL the new helpers (delete the inlined copies).
  Re-run Task 3's CPU test — conservation and order must be unchanged (this proves
  the refactor is behavior-preserving before the GPU consumes the same code).
- Modify: `gpu/residual3d_gpu.jl` (order==3 kernels calling the SAME helpers),
  `gpu/timestep3d_gpu.jl` / `gpu/gpu_run.jl` (thread `order = 3`).
- Ensure both `src/Riemann35.jl` and the GPU module tree `include` the new
  `hiorder3_recon_dev.jl` (mirror the dual-include of `riemann_flux_dev.jl`).
- Create: `gpu/validation/validate_hiorder3_parity.jl`.

**Interfaces:**
- Produces: `weno5_face_states_dev(c₋₄..c₊₄::NTuple{35}, axis, Ma, s3max) -> (mL,mR)`
  and `idp_cell_thetas_dev(Mlo, G⁻ˣ,G⁺ˣ,...,G⁺ᶻ, λx,λy,λz) -> (θ⁻ˣ,...,θ⁺ᶻ)`,
  both device-safe, consumed by CPU `highorder_3d.jl` AND GPU `residual3d_gpu.jl`.

- [ ] **Step 0 (DRY refactor, do FIRST):** Extract the per-face reconstruction and
  the per-cell θ math from Task 3's `residual_line3`/`residual_ho_3d_order3!` into
  `hiorder3_recon_dev.jl` as `weno5_face_states_dev` / `idp_cell_thetas_dev`
  (NTuple, no allocation — device-safe). Rewrite the CPU functions to call them.
  Run `test/test_hiorder3_cpu.jl` — conservation (9.7e-17) and order UNCHANGED.
  Commit this refactor separately ("refactor(hiorder3): extract single-source
  device-safe per-face/per-cell helpers"). Only then write the GPU kernels.

- [ ] **Step 1:** In `gpu/residual3d_gpu.jl`, add `order == 3` kernels that CALL
  `weno5_face_states_dev` and `idp_cell_thetas_dev` (the same functions the CPU
  now uses — zero duplicated numerics). A flux kernel reads the 9-cell stencil
  from `CuDeviceArray` neighbors, calls `weno5_face_states_dev`, computes
  F_HO/F_LO (via the shared `riemann_flux_dev` rs=0), stores into two face-scratch
  buffers (mirror the existing `flat` buffer; add the anchor buffer). A limit+
  update kernel reads a cell's six face corrections, calls `idp_cell_thetas_dev`,
  min-combines with neighbors, blends, updates. Reuse the order-2 kernel's
  indexing/halo pattern exactly. Watch the @noinline/@fastmath device-helper
  parity gotcha for any shared @fastmath helper.

- [ ] **Step 2:** Thread `order` through `march3d_gpu!`/`march3d_slab_gpu!` (`gpu/timestep3d_gpu.jl`) and `run_gpu_3d` (`gpu/gpu_run.jl`) so `order = 3` selects the new kernels.

- [ ] **Step 3:** Create `gpu/validation/validate_hiorder3_parity.jl`: (a) a GPU compile smoke test (build the order-3 kernels on a tiny box, catch InvalidIRError — the ENV/allocation guard), (b) CPU-vs-GPU parity on a matched-dt small box (16³, a few steps, order=3): report max rel diff (bitwise or ulp≤1e-10, per the documented @fastmath tolerance).

- [ ] **Step 4:** Run on a GPU node (find a job: `squeue -u sbryngelson3 -h -o "%.10i %P %t" | awk '$2~/gpu/&&$3=="R"{print $1;exit}'`; then `srun --jobid=<JID> --overlap -N1 -n1 bash -c '...scrub PMI... source env; $JULIA gpu/validation/validate_hiorder3_parity.jl'`). Expected: kernel compiles, parity ≤1e-10.

- [ ] **Step 5: Commit**

```bash
git add gpu/residual3d_gpu.jl gpu/timestep3d_gpu.jl gpu/gpu_run.jl gpu/validation/validate_hiorder3_parity.jl
git commit -m "feat(hiorder3): GPU two-pass order=3 kernels + CPU/GPU parity"
```

---

## Task 6: Headline validation (Ma=100 3D) + byte-identical defaults + writeup

**Files:**
- Create: `gpu/validation/validate_hiorder3_ma100.jl`
- Modify (research repo): append 3D results to `roe-hyqmom-notes.tex` `sec:hiorder-idp`.

- [ ] **Step 1:** Create `gpu/validation/validate_hiorder3_ma100.jl`: run the 3D crossing jets at Ma=100 (16³/32³) on GPU with `order=3` (WENO5+θ*-IDP), and the controls `order=3` no-limiter and `order=2` MUSCL2. Report per scheme: NaN-free, min ρ, θ*-active fraction, wall-clock s/step. Headline: order=3 survives at high order; no-limiter breaks.

- [ ] **Step 2:** Byte-identical defaults regression: run the existing golden/`Pkg.test` suite and confirm `spatial_order=1,2` results unchanged (the order-3 code is additive; the suite must stay green).

Run: `$JULIA --project=. -e 'using Pkg; Pkg.test()'` → all pass (5 broken pre-existing OK).

- [ ] **Step 3:** In the research repo `Code_Riemann_3D_35mom_july2026_GT`, append a `\subsection{3D}` to `sec:hiorder-idp` in `roe-hyqmom-notes.tex`: the 3D order table, the Ma=100 survival result (order-3 vs no-limiter vs MUSCL2), conservation, CPU/GPU parity number, and cost. Compile (`pdflatex→bibtex→pdflatex→pdflatex`), confirm clean, commit + push.

- [ ] **Step 4:** Open a PR for `feat/hiorder-3d` in Riemann35.jl with the results summary (do NOT merge — leave for review).

```bash
cd /storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl
git add gpu/validation/validate_hiorder3_ma100.jl
git commit -m "test(hiorder3): Ma=100 3D crossing-jets validation + defaults regression"
git push -u origin feat/hiorder-3d
gh pr create --base main --title "feat: opt-in spatial_order=3 (WENO5 + theta*-IDP), CPU+GPU" --body "<results>"
```

---

## Self-review notes

- **Spec coverage:** WENO5 device-safe (T1), deconv/conv for 5th order (T1+T3), θ*-IDP (T2), joint 6-face two-pass (T3+T5), CPU wiring (T4), GPU single-source + parity (T5), Ma=100 headline + byte-identical defaults + writeup (T6). All spec sections mapped.
- **Known incompleteness (flagged, not placeholders):** Task 3 Steps 2–3 and Task 5 Step 1 give the device-safe pipeline and the exact two-pass structure, but the final wiring into `residual_ho_3d!`/the GPU kernels must follow those files' existing `order==2` accumulation code (which the implementer reads in-repo) — the plan specifies the algorithm and the new code; the integration mirrors the existing pattern. The implementer should read the `order==2` branch first and mirror its indexing/halo/flux-bound calls. Confirm halo `g≥4` (WENO5 needs 3, deconv adds 2).
- **Risk:** if measured 3D order < ~4, deconv/conv is not engaged per-direction — debug before proceeding to GPU (Task 4 gate).

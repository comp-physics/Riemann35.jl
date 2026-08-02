module DVMBGKGPU
# dvm_bgk_gpu.jl — THE DVM GROUND TRUTH ON THE GPU.
#
# WHY. The DVM is the reference every wall result is measured against, and it had become the
# rate limiter: the Couette and Fourier references run single-threaded, hours per point, which
# is why three of the four points in the published zeta curve were never convergence-checked
# and why the velocity grid sat at nv = 24. A reference you cannot afford to refine is a
# reference you cannot fully trust.
#
# The structure is close to ideal for a GPU and it was simply never exploited:
#   * transport is first-order upwind in x, elementwise in the 3D velocity index;
#   * the collision is per spatial cell, a REDUCTION over the velocity grid followed by a
#     5-parameter Newton solve that costs nothing next to the reduction.
#
# LAYOUT: f[a,b,c,i], velocity-major. Both hot kernels then read coalesced -- transport has
# consecutive threads touching consecutive `a` at fixed neighbour `i-1`, and the collision
# reduction walks the velocity block contiguously. The CPU code uses f[i,a,b,c], which is
# wrong for both on a device.
#
# THE NEWTON IS WARM-STARTED from the previous step's coefficients, held per cell in `A`.
# This is the single most important optimisation: from a cold start the Mieussens solve takes
# 5-10 iterations, from the previous timestep it takes 1-2, and the collision is ~95% of the
# cost. The CPU version restarts cold every cell every step because it never keeps state.
#
# WHAT IS NOT APPROXIMATED. The discrete Maxwellian still matches rho, rho*u and rho*E
# EXACTLY on the grid (that is the whole point of Mieussens -- it makes the collision
# conservative on a finite velocity grid), and the wall is still the exact half-space
# condition with rho_w set from the DISCRETE half-fluxes. This is a port, not a model change,
# and validate_dvm_gpu.jl checks it against the CPU field to roundoff rather than trusting
# that claim.
using CUDA, LinearAlgebra
using Riemann35.Weno5Dev: weno5z

export VGridG, WALL_DV_MAX, wall_dv_ok, dvm_alloc, transport_upwind!, transport_weno5!, dvm_alloc_weno5,
       transport_walls!, transport_walls_weno5!, collide!, moments35_field,
       discrete_maxwellian_host!, wall_pair_host, cellT_from_M,
       # added with the ES-BGK collision and the exact wall fluxes (#47, #48). These were
       # PRESENT but not exported, so `using .DVMBGKGPU` did not bring them into scope --
       # invisible here because every call site in the sweeps is module-qualified.
       collide_es!, wall_flux, freemolecular_stress, freemolecular_heatflux

# CANONICAL ORDER, and it must match src/moments/moment_indices.jl exactly. An earlier
# transcription of this table PERMUTED positions 31-33 -- it read
# (0,1,2),(1,1,2),(0,3,1) where the canonical order is (0,3,1),(0,1,2),(1,1,2). The multiset
# was right, so no moment was missing and every total/trace was correct; three moments were
# simply MISLABELLED. Nothing published was affected because every wall observable uses
# indices <= 30 (rho, u, and the diagonal second moments), but those three are exactly the
# odd fourth-order cross-moments that moment_reduce26.jl drops, so a reduced-26 comparison
# run through this module would have been silently wrong.
#
# It surfaced only because a heat-flux estimator built on index 32 returned exactly -u on a
# pure Maxwellian, where the answer had to be 0. validate_dvm_gpu.jl now asserts the two
# tables agree elementwise, which is the check that should have existed from the start.
using Riemann35.MomentIndices: IJK
# Canonical table, imported not copied (issue #61). It was a local transcription; the
# table has been mis-transcribed once before -- positions 31-33 permuted, three moments
# mislabelled while every total and trace stayed correct, so nothing caught it.
const IJK35 = IJK

struct VGridG
    v::CuVector{Float64}
    vh::Vector{Float64}
    dv::Float64
    n::Int
end
"""
    VGridG(vmax, n)

Uniform velocity grid on `[-vmax, vmax]` with `n` points, so `dv = 2*vmax/(n-1)`.

RESOLUTION IS SET BY `dv`, NOT BY `n`, and at a wall it matters far more than in the bulk.
Measured at Kn = 0.73 (ES-BGK, nx = 384) against 8-seed DSMC, Knudsen-layer curvature
1.426 +/- 0.050 (SEM):

    nv    dv      curvature   gap to DSMC
    32    0.387   1.160       5.3 SEM
    48    0.255   1.296       2.6 SEM
    64    0.190   1.348       1.5 SEM
    96    0.126   1.366       1.2 SEM
    128   0.094   1.375       1.0 SEM

so `nv = 32` is 18% low on the layer and only `dv <~ 0.13` reaches the reference's own
noise floor. `vmax` is NOT the knob: a matched-`dv` pair (nv=64/vmax=6, dv=0.1905) and
(nv=86/vmax=8, dv=0.1882) agree to 0.1% despite a 33% wider domain, while coarsening `dv`
at fixed vmax moves it 4%. Tail truncation at 6 thermal speeds is negligible.

IN THE BULK IT BARELY MATTERS: on a wall-free shear mode, refining nv 32 -> 48 changes
the decay rate in the SEVENTH digit. A diffuse wall is a discontinuity in velocity space
at `v_n = 0`; a smooth bulk mode is not, so a grid adequate for transport is not
adequate at a wall.

The threshold is on `dv/sqrt(T)` -- it is a resolution per thermal speed. `WALL_DV_MAX`
is quoted for `T = 1`, the normalisation every driver here uses.
"""
function VGridG(vmax::Real, n::Int)
    vh = collect(range(-Float64(vmax), Float64(vmax), length = n))
    VGridG(CuArray(vh), vh, vh[2] - vh[1], n)
end

"Largest `dv` (at T = 1) that reaches the DSMC noise floor on the Knudsen layer; see `VGridG`."
const WALL_DV_MAX = 0.13

"""
    wall_dv_ok(dv; T = 1.0) -> Bool

Is this velocity resolution adequate for a WALL-bounded run? The criterion is
`dv/sqrt(T) <= WALL_DV_MAX`, measured; see `VGridG` for the data behind it. Bulk-only
problems are fine far coarser and this does not apply to them.

Pure -- no device state. It still lives in this module, which needs CUDA to load, so it
is exercised by `gpu/runtests_gpu.jl` rather than by the CI suite.
"""
wall_dv_ok(dv::Real; T::Real = 1.0) = dv / sqrt(T) <= WALL_DV_MAX

"f, work buffer, and the per-cell Newton coefficient state."
function dvm_alloc(Nx::Int, g::VGridG)
    (CUDA.zeros(Float64, g.n, g.n, g.n, Nx),
     CUDA.zeros(Float64, g.n, g.n, g.n, Nx),
     CUDA.zeros(Float64, 5, Nx))
end

# =======================================================================================
# TRANSPORT — first-order upwind in x. Identical stencil to the CPU `transport!`.
# =======================================================================================
function _upwind_kernel!(fn, f, v, n, Nx, lam, bc::Int32)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = n * n * n * Nx
    if t <= N
        @inbounds begin
            a = (t - 1) % n + 1;   r = (t - 1) ÷ n
            b = r % n + 1;         r = r ÷ n
            c = r % n + 1;         i = r ÷ n + 1
            s = v[a]
            if s > 0
                up = i == 1 ? (bc == Int32(1) ? f[a,b,c,Nx] : f[a,b,c,1]) : f[a,b,c,i-1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(f[a,b,c,i] - up)
            elseif s < 0
                up = i == Nx ? (bc == Int32(1) ? f[a,b,c,1] : f[a,b,c,Nx]) : f[a,b,c,i+1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(up - f[a,b,c,i])
            else
                fn[a,b,c,i] = f[a,b,c,i]
            end
        end
    end
    return nothing
end

"Periodic (bc=:periodic) or zero-gradient (bc=:copy) upwind transport."
function transport_upwind!(f, fn, dt, dx, g::VGridG; bc::Symbol = :copy)
    Nx = size(f, 4); N = g.n^3 * Nx
    thr = 256; blk = cld(N, thr)
    @cuda threads=thr blocks=blk _upwind_kernel!(fn, f, g.v, g.n, Nx,
                                                 dt/dx, bc === :periodic ? Int32(1) : Int32(0))
    copyto!(f, fn); f
end

# ---------------------------------------------------------------------------------------
# TRANSPORT — WENO5 + SSP-RK3, for when the DVM has to be a REFERENCE rather than a solver.
#
# WHY THIS EXISTS. First-order upwind damps a travelling wave at a rate O(dx) that is
# indistinguishable from physical attenuation on any single grid. That is tolerable when the
# DVM is being compared on a steady profile and fatal when it is asked for a decay rate, a
# sound attenuation, or the thickness of a Knudsen layer -- numerical diffusion smooths a
# Knudsen layer exactly the way a weak collision model does, so the two are confounded.
# Every DVM reference in the wall study used the first-order path, including the curvature
# ratio (DVM 1.247 against DSMC's 1.426) that was read as BGK's collision-model deficit.
# This makes that attribution checkable instead of assumed.
#
# WHY IT IS CHEAP TO DO HERE, and much easier than the moment side. The transport substep is
# LINEAR SCALAR ADVECTION AT CONSTANT SPEED, independently for each velocity node:
# df/dt + v_a df/dx = 0. There is no Riemann problem, no coupling between nodes, and --
# unlike the 35-moment system -- no realizable set to stay inside, so no projection and no
# limiter interaction to reason about. Standard Jiang-Shu WENO5 with the stencil biased by
# sign(v_a), and SSP-RK3 in time so the scheme is not left first-order in dt.
#
# `transport_upwind!` is KEPT and unchanged: every published number in the notes was produced
# with it, and silently upgrading it would make those irreproducible.
# RECONSTRUCTION IS THE SHIPPED `weno5z`, NOT A LOCAL COPY. An earlier version of this
# routine hand-rolled classic Jiang-Shu WENO5 here. That was wrong on two counts. It
# duplicated math the package already provides device-safe (src/numerics/weno5_dev.jl), against
# this file's own convention; and it used DIFFERENT nonlinear weights from the moment solver,
# which runs WENO5-Z (Borges et al. 2008) -- so part of any closure-versus-DVM gap would have
# been the two codes reconstructing differently rather than the closure being wrong. With
# `weno5z` on both sides the spatial reconstruction and the SSP-RK3 time integration are
# IDENTICAL, and what remains is the moment truncation and the moment solver's theta*-IDP
# limiter (which has no DVM analogue: scalar advection has no realizable set to protect).
#
# The local copy also carried a trap worth recording. Its centre value was named `f0` with
# juxtaposition multiplication, so `11f0` was not `11*f0` but Julia's Float32 LITERAL 11.0f0,
# and the centre value silently dropped out of three of the four polynomials -- a 78%
# spatial-operator error that did NOT converge under refinement. `weno5z` names its centre
# value `v0`, and `v` is not an exponent marker, so it is immune. Any future local stencil
# here must avoid `<digit>f<digit>` and `<digit>e<digit>` in the same way.
#
# NOTE ON THE USAGE. `weno5z`'s docstring describes reconstruction from CELL AVERAGES, which
# is how the moment solver uses it. Here it is used in FINITE-DIFFERENCE mode, on point values
# of the flux. The coefficients coincide -- that is the standard FD/FV WENO equivalence -- and
# test_dvm_weno5.jl measures the resulting order rather than taking it on faith.
@inline _idx(i, Nx, per::Bool) = per ? mod(i - 1, Nx) + 1 : clamp(i, 1, Nx)

"One WENO5 spatial residual: du[a,b,c,i] = -(F_{i+1/2} - F_{i-1/2})/dx."
function _weno5_rhs_kernel!(du, f, v, n, Nx, invdx, per::Int32)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = n * n * n * Nx
    if t <= N
        @inbounds begin
            a = (t - 1) % n + 1;   r = (t - 1) ÷ n
            b = r % n + 1;         r = r ÷ n
            c = r % n + 1;         i = r ÷ n + 1
            s = v[a]
            if s == 0.0
                du[a,b,c,i] = 0.0
            else
                # The seven stencil values are read into locals EXPLICITLY. An earlier
                # version used a closure `g(k) = f[a,b,c,_idx(i+k,Nx,p)]`, which is correct
                # Julia and compiled without complaint, but produced a spatial operator with
                # an O(1) error that did not converge under refinement (78% at nx = 32, 64
                # and 128 alike) -- the captured, reassigned locals do not survive into the
                # device closure the way they do on the host. Hand-evaluating the same
                # stencil off-device reproduced the exact derivative to 5th order, which is
                # what localised it to the closure rather than the formula.
                p = per == Int32(1)
                fm3 = f[a, b, c, _idx(i-3, Nx, p)]
                fm2 = f[a, b, c, _idx(i-2, Nx, p)]
                fm1 = f[a, b, c, _idx(i-1, Nx, p)]
                f00 = f[a, b, c, _idx(i,   Nx, p)]
                fp1 = f[a, b, c, _idx(i+1, Nx, p)]
                fp2 = f[a, b, c, _idx(i+2, Nx, p)]
                fp3 = f[a, b, c, _idx(i+3, Nx, p)]
                if s > 0
                    # upwind-biased from the left for both interfaces
                    fr = weno5z(fm2, fm1, f00, fp1, fp2)     # at i+1/2
                    fl = weno5z(fm3, fm2, fm1, f00, fp1)     # at i-1/2
                else
                    # mirror the stencil about the interface
                    fr = weno5z(fp3, fp2, fp1, f00, fm1)     # at i+1/2
                    fl = weno5z(fp2, fp1, f00, fm1, fm2)     # at i-1/2
                end
                du[a,b,c,i] = -s*(fr - fl)*invdx
            end
        end
    end
    return nothing
end

function _weno5_rhs!(du, f, dx, g::VGridG, per::Bool)
    Nx = size(f, 4); N = g.n^3 * Nx
    thr = 256; blk = cld(N, thr)
    @cuda threads=thr blocks=blk _weno5_rhs_kernel!(du, f, g.v, g.n, Nx, 1.0/dx,
                                                    per ? Int32(1) : Int32(0))
    du
end

"""
    transport_weno5!(f, u1, u2, du, dt, dx, g; bc=:periodic)

WENO5 + SSP-RK3 transport, the high-order counterpart of `transport_upwind!`. `u1`, `u2` and
`du` are scratch arrays shaped like `f` (see `dvm_alloc_weno5`).

`bc = :periodic` wraps the stencil; `:copy` clamps it, which is a zero-gradient extrapolation
and is NOT a wall -- a diffuse wall needs the half-space treatment in `transport_walls!`,
which remains first-order and is unaffected by this routine.
"""
function transport_weno5!(f, u1, u2, du, dt, dx, g::VGridG; bc::Symbol = :periodic)
    per = bc === :periodic
    _weno5_rhs!(du, f, dx, g, per)
    @. u1 = f + dt*du                                   # stage 1
    _weno5_rhs!(du, u1, dx, g, per)
    @. u2 = 0.75*f + 0.25*(u1 + dt*du)                  # stage 2
    _weno5_rhs!(du, u2, dx, g, per)
    @. f = (1.0/3.0)*f + (2.0/3.0)*(u2 + dt*du)         # stage 3
    f
end

# ---------------------------------------------------------------------------------------
# WALL TRANSPORT — WENO5 + SSP-RK3.  High-order counterpart of `transport_walls!` (#58).
#
# WHY. `transport_walls!` is first-order upwind, and it is the path every wall reference in
# the notes was produced with. First-order smooths a Knudsen layer exactly the way a weak
# collision model does, so the two are confounded: the DVM's layer deficit could not be
# attributed to BGK versus to the scheme without a grid study standing in for a direct
# measurement.
#
# WHY IT IS TRACTABLE, which is not obvious from the physics. `f` is discontinuous at
# v_n = 0 -- but that is a discontinuity in VELOCITY space, and each velocity node advects
# independently in PHYSICAL space. Per node this is ordinary scalar advection with a
# Dirichlet inflow, so a standard one-sided/ghost WENO5 applies. The first-order kernel
# already encodes exactly that BC: at i = 1 with v > 0 its upwind value is `rw[1]*ML`, the
# wall Maxwellian scaled by rho_w.
#
# THE GHOST RULE IS SIGN-DEPENDENT, and getting it backwards silently injects wall
# material into an outflow:
#     v > 0 : ghosts to the LEFT  are the wall value rw[1]*ML   (inflow)
#             ghosts to the RIGHT clamp                          (outflow, zero-gradient)
#     v < 0 : ghosts to the RIGHT are the wall value rw[2]*MR   (inflow)
#             ghosts to the LEFT  clamp                          (outflow)
#
# AND rho_w IS RESTAGED. `rw` is fixed by the discrete half-flux balance and therefore
# depends on `f`, so each SSP-RK3 stage recomputes it from that stage's state. Reusing the
# step-start rho_w would make the boundary condition lag the solution by a stage and quietly
# cost the order this routine exists to provide.
function _wall_weno5_rhs_kernel!(du, f, v, n, Nx, invdx, ML, MR, rw)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = n * n * n * Nx
    if t <= N
        @inbounds begin
            a = (t - 1) % n + 1;   r = (t - 1) ÷ n
            b = r % n + 1;         r = r ÷ n
            c = r % n + 1;         i = r ÷ n + 1
            s = v[a]
            if s == 0.0
                du[a,b,c,i] = 0.0
            else
                wl = rw[1]*ML[a,b,c]        # what streams in at the LO wall  (v > 0)
                wr = rw[2]*MR[a,b,c]        # what streams in at the HI wall  (v < 0)
                # sample with the sign-dependent ghost rule above
                g(k) = begin
                    j = i + k
                    j < 1  ? (s > 0 ? wl : f[a,b,c,1])  :
                    j > Nx ? (s < 0 ? wr : f[a,b,c,Nx]) : f[a,b,c,j]
                end
                fm3 = g(-3); fm2 = g(-2); fm1 = g(-1); f00 = g(0)
                fp1 = g(1);  fp2 = g(2);  fp3 = g(3)
                if s > 0
                    fr = weno5z(fm2, fm1, f00, fp1, fp2)
                    fl = weno5z(fm3, fm2, fm1, f00, fp1)
                else
                    fr = weno5z(fp3, fp2, fp1, f00, fm1)
                    fl = weno5z(fp2, fp1, f00, fm1, fm2)
                end
                du[a,b,c,i] = -s*(fr - fl)*invdx
            end
        end
    end
    return nothing
end

function _wall_weno5_rhs!(du, f, dx, g::VGridG, ML, MR, rw)
    Nx = size(f, 4); N = g.n^3 * Nx
    thr = 256; blk = cld(N, thr)
    @cuda threads=thr blocks=blk _wall_weno5_rhs_kernel!(du, f, g.v, g.n, Nx, 1.0/dx, ML, MR, rw)
    du
end

"Recompute rho_w at both walls from the CURRENT state's discrete half-fluxes."
function _restage_rhow!(rw, inflx, f, g::VGridG, outL::Float64, outR::Float64)
    Nx = size(f, 4)
    fill!(inflx, 0.0)
    @cuda threads=256 blocks=cld(g.n^3, 256) _influx_kernel!(inflx, f, g.v, g.n, Nx, g.dv^3)
    h = Array(inflx)
    copyto!(rw, [h[1]/outL, h[2]/outR])
    (h[1]/outL, h[2]/outR)
end

"""
    transport_walls_weno5!(f, u1, u2, du, dt, dx, g, ML, MR, outL, outR, inflx, rw)

WENO5 + SSP-RK3 transport between two diffuse walls — the high-order counterpart of
`transport_walls!`, which is kept byte-unchanged so published numbers stay reproducible.
`u1`, `u2`, `du` are scratch shaped like `f` (`dvm_alloc_weno5`). Returns the final
`(rho_wL, rho_wR)`.
"""
function transport_walls_weno5!(f, u1, u2, du, dt, dx, g::VGridG, ML, MR,
                                outL::Float64, outR::Float64, inflx, rw)
    _restage_rhow!(rw, inflx, f,  g, outL, outR)
    _wall_weno5_rhs!(du, f,  dx, g, ML, MR, rw); @. u1 = f + dt*du
    _restage_rhow!(rw, inflx, u1, g, outL, outR)
    _wall_weno5_rhs!(du, u1, dx, g, ML, MR, rw); @. u2 = 0.75*f + 0.25*(u1 + dt*du)
    r = _restage_rhow!(rw, inflx, u2, g, outL, outR)
    _wall_weno5_rhs!(du, u2, dx, g, ML, MR, rw); @. f = (1.0/3.0)*f + (2.0/3.0)*(u2 + dt*du)
    r
end

"Scratch for `transport_weno5!`: three arrays shaped like `f`."
dvm_alloc_weno5(f) = (similar(f), similar(f), similar(f))

# ---------------------------------------------------------------------------------------
# WALLS — the exact half-space diffuse condition. f for the INCOMING velocities IS the wall
# Maxwellian; no ghost cell and no Gaussian assumption about the interior. rho_w comes from
# the DISCRETE half-fluxes, so zero net mass flux holds by construction on the finite grid.
# The continuum rho*sqrt(T/Tw) is the formula the moment solver had to abandon (rem:wall-fix)
# and using it here would quietly reintroduce a mass leak into the reference itself.
# ---------------------------------------------------------------------------------------
function _influx_kernel!(out, f, v, n, Nx, dv3)
    # out[1] = influx at the LO wall (v>0 leaves, so v<0 arrives), out[2] = at the HI wall
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= n*n*n
        @inbounds begin
            a = (t - 1) % n + 1; r = (t - 1) ÷ n
            b = r % n + 1;       c = r ÷ n + 1
            vx = v[a]
            if vx < 0
                CUDA.@atomic out[1] += -vx*f[a,b,c,1]*dv3
            elseif vx > 0
                CUDA.@atomic out[2] +=  vx*f[a,b,c,Nx]*dv3
            end
        end
    end
    return nothing
end

function _wall_transport_kernel!(fn, f, v, n, Nx, lam, ML, MR, rw)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = n*n*n*Nx
    if t <= N
        @inbounds begin
            a = (t - 1) % n + 1;   r = (t - 1) ÷ n
            b = r % n + 1;         r = r ÷ n
            c = r % n + 1;         i = r ÷ n + 1
            s = v[a]
            if s > 0
                up = i == 1 ? rw[1]*ML[a,b,c] : f[a,b,c,i-1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(f[a,b,c,i] - up)
            elseif s < 0
                up = i == Nx ? rw[2]*MR[a,b,c] : f[a,b,c,i+1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(up - f[a,b,c,i])
            else
                fn[a,b,c,i] = f[a,b,c,i]
            end
        end
    end
    return nothing
end

"""
    transport_walls!(f, fn, dt, dx, g, ML, MR, outL, outR, inflx, rw)

Upwind transport with exact diffuse walls at both ends. `ML`/`MR` are unit-density wall
Maxwellians on the device, `outL`/`outR` their outgoing half-fluxes (host scalars from
`wall_pair_host`). `inflx` and `rw` are length-2 device scratch.
"""
function transport_walls!(f, fn, dt, dx, g::VGridG, ML, MR, outL::Float64, outR::Float64,
                          inflx, rw)
    # Warn ONCE. Under-resolving the velocity grid at a wall is silent -- it produces a
    # smooth, plausible profile that is simply wrong in the Knudsen layer (18% at nv = 32),
    # and no convergence study in nx can see it. Not an error and not a default change:
    # published numbers came from nv = 24-32 and altering the default would make them
    # irreproducible.
    wall_dv_ok(g.dv) || @warn(
        "DVM wall transport with an under-resolved velocity grid: dv = $(round(g.dv, digits=4)) " *
        "> WALL_DV_MAX = $WALL_DV_MAX (at T = 1). The Knudsen layer will be too weak " *
        "(measured: 18% low at dv = 0.387). Refine nv, not vmax -- see `VGridG`.",
        maxlog = 1)
    Nx = size(f, 4); n = g.n
    fill!(inflx, 0.0)
    @cuda threads=256 blocks=cld(n^3, 256) _influx_kernel!(inflx, f, g.v, n, Nx, g.dv^3)
    h = Array(inflx)
    # NB both `out*` and the influx are POSITIVE magnitudes -- dividing by a negated outflux
    # gives a negative ghost density and a singular Newton downstream. That sign error cost
    # an afternoon in the CPU version.
    copyto!(rw, [h[1]/outL, h[2]/outR])
    @cuda threads=256 blocks=cld(n^3*Nx, 256) _wall_transport_kernel!(
        fn, f, g.v, n, Nx, dt/dx, ML, MR, rw)
    copyto!(f, fn)
    (h[1]/outL, h[2]/outR)
end

# =======================================================================================
# COLLISION — one block per spatial cell.
#
# Each block runs the Mieussens Newton on its own cell: threads accumulate the 5 moments and
# the 15 unique entries of the symmetric 5x5 Jacobian over the velocity grid, tree-reduce in
# shared memory, and thread 1 does the Cholesky solve. The Jacobian is a Gram matrix
# sum(b b' f) with f > 0, hence symmetric positive definite, so Cholesky is both valid and
# cheaper than the generic solve the CPU uses.
#
# Warm start: A[:,i] carries the previous step's coefficients. `tol` is checked on the
# residual so a converged cell exits after ONE reduction.
# =======================================================================================
const NRED = 20   # 5 moments + 15 unique Jacobian entries

function _collide_kernel!(f, v, n, Nx, dv3, A, dt, tau, iters::Int32, tol)
    i = blockIdx().x
    tid = threadIdx().x
    nt = blockDim().x
    sh = CUDA.CuDynamicSharedArray(Float64, NRED * nt)
    sa = CUDA.CuDynamicSharedArray(Float64, 8, NRED * nt * sizeof(Float64))
    npts = n*n*n

    # ---- target collision invariants from the CURRENT f: <1>, <v>, <|v|^2> ----
    m0 = 0.0; m1 = 0.0; m2 = 0.0; m3 = 0.0; m4 = 0.0
    @inbounds for t in tid:nt:npts
        a = (t - 1) % n + 1; r = (t - 1) ÷ n
        b = r % n + 1;       c = r ÷ n + 1
        vx = v[a]; vy = v[b]; vz = v[c]
        w = f[a,b,c,i]*dv3
        m0 += w; m1 += w*vx; m2 += w*vy; m3 += w*vz
        m4 += w*(vx*vx + vy*vy + vz*vz)
    end
    @inbounds begin
        sh[tid] = m0; sh[nt+tid] = m1; sh[2nt+tid] = m2; sh[3nt+tid] = m3; sh[4nt+tid] = m4
    end
    sync_threads()
    s = nt >> 1
    while s > 0
        if tid <= s
            @inbounds for q in 0:4; sh[q*nt+tid] += sh[q*nt+tid+s]; end
        end
        sync_threads(); s >>= 1
    end
    @inbounds if tid == 1
        for q in 0:4; sa[q+1] = sh[q*nt+1]; end     # sa[1..5] = target moments
    end
    sync_threads()
    @inbounds tgt0 = sa[1]; @inbounds tgt1 = sa[2]; @inbounds tgt2 = sa[3]
    @inbounds tgt3 = sa[4]; @inbounds tgt4 = sa[5]

    # ---- warm start; fall back to the Maxwellian coefficients on the first visit ----
    @inbounds a1 = A[1,i]; a2 = A[2,i]; a3 = A[3,i]; a4 = A[4,i]; a5 = A[5,i]
    if a5 == 0.0                                   # never initialised
        rho = tgt0
        ux = tgt1/rho; uy = tgt2/rho; uz = tgt3/rho
        T = (tgt4/rho - (ux*ux + uy*uy + uz*uz))/3
        a1 = log(rho*(2*pi*T)^(-1.5)) - (ux*ux + uy*uy + uz*uz)/(2T)
        a2 = ux/T; a3 = uy/T; a4 = uz/T; a5 = -1/(2T)
    end

    it = Int32(0)
    while it < iters
        it += Int32(1)
        # accumulate moments (5) and the unique upper triangle of J (15)
        r1 = 0.0; r2 = 0.0; r3 = 0.0; r4 = 0.0; r5 = 0.0
        j11=0.0; j12=0.0; j13=0.0; j14=0.0; j15=0.0
        j22=0.0; j23=0.0; j24=0.0; j25=0.0
        j33=0.0; j34=0.0; j35=0.0
        j44=0.0; j45=0.0; j55=0.0
        @inbounds for t in tid:nt:npts
            aa = (t - 1) % n + 1; rr = (t - 1) ÷ n
            bb = rr % n + 1;      cc = rr ÷ n + 1
            vx = v[aa]; vy = v[bb]; vz = v[cc]
            s2 = vx*vx + vy*vy + vz*vz
            fv = exp(a1 + a2*vx + a3*vy + a4*vz + a5*s2)*dv3
            r1 += fv;      r2 += vx*fv;   r3 += vy*fv;  r4 += vz*fv;  r5 += s2*fv
            j11 += fv;     j12 += vx*fv;  j13 += vy*fv; j14 += vz*fv; j15 += s2*fv
            j22 += vx*vx*fv; j23 += vx*vy*fv; j24 += vx*vz*fv; j25 += vx*s2*fv
            j33 += vy*vy*fv; j34 += vy*vz*fv; j35 += vy*s2*fv
            j44 += vz*vz*fv; j45 += vz*s2*fv; j55 += s2*s2*fv
        end
        @inbounds begin
            sh[      tid] = r1;  sh[   nt+tid] = r2;  sh[ 2nt+tid] = r3
            sh[ 3nt+tid] = r4;  sh[ 4nt+tid] = r5
            sh[ 5nt+tid] = j11; sh[ 6nt+tid] = j12; sh[ 7nt+tid] = j13
            sh[ 8nt+tid] = j14; sh[ 9nt+tid] = j15; sh[10nt+tid] = j22
            sh[11nt+tid] = j23; sh[12nt+tid] = j24; sh[13nt+tid] = j25
            sh[14nt+tid] = j33; sh[15nt+tid] = j34; sh[16nt+tid] = j35
            sh[17nt+tid] = j44; sh[18nt+tid] = j45; sh[19nt+tid] = j55
        end
        sync_threads()
        s = nt >> 1
        while s > 0
            if tid <= s
                @inbounds for q in 0:(NRED-1); sh[q*nt+tid] += sh[q*nt+tid+s]; end
            end
            sync_threads(); s >>= 1
        end

        @inbounds if tid == 1
            b1 = sh[0nt+1] - tgt0; b2 = sh[1nt+1] - tgt1; b3 = sh[2nt+1] - tgt2
            b4 = sh[3nt+1] - tgt3; b5 = sh[4nt+1] - tgt4
            nrm = sqrt(b1*b1 + b2*b2 + b3*b3 + b4*b4 + b5*b5)
            if nrm < tol*max(1.0, tgt0)
                sa[6] = 1.0                       # converged: signal the block to stop
            else
                sa[6] = 0.0
                # Cholesky on the SPD Gram matrix J = sum(b b' f), then solve J d = b.
                L11 = sqrt(sh[5nt+1])
                L21 = sh[6nt+1]/L11; L31 = sh[7nt+1]/L11
                L41 = sh[8nt+1]/L11; L51 = sh[9nt+1]/L11
                L22 = sqrt(sh[10nt+1] - L21*L21)
                L32 = (sh[11nt+1] - L31*L21)/L22
                L42 = (sh[12nt+1] - L41*L21)/L22
                L52 = (sh[13nt+1] - L51*L21)/L22
                L33 = sqrt(sh[14nt+1] - L31*L31 - L32*L32)
                L43 = (sh[15nt+1] - L41*L31 - L42*L32)/L33
                L53 = (sh[16nt+1] - L51*L31 - L52*L32)/L33
                L44 = sqrt(sh[17nt+1] - L41*L41 - L42*L42 - L43*L43)
                L54 = (sh[18nt+1] - L51*L41 - L52*L42 - L53*L43)/L44
                L55 = sqrt(sh[19nt+1] - L51*L51 - L52*L52 - L53*L53 - L54*L54)
                y1 = b1/L11
                y2 = (b2 - L21*y1)/L22
                y3 = (b3 - L31*y1 - L32*y2)/L33
                y4 = (b4 - L41*y1 - L42*y2 - L43*y3)/L44
                y5 = (b5 - L51*y1 - L52*y2 - L53*y3 - L54*y4)/L55
                d5 = y5/L55
                d4 = (y4 - L54*d5)/L44
                d3 = (y3 - L43*d4 - L53*d5)/L33
                d2 = (y2 - L32*d3 - L42*d4 - L52*d5)/L22
                d1 = (y1 - L21*d2 - L31*d3 - L41*d4 - L51*d5)/L11
                sa[1] = d1; sa[2] = d2; sa[3] = d3; sa[4] = d4; sa[5] = d5
            end
        end
        sync_threads()
        @inbounds if sa[6] == 1.0
            break
        end
        @inbounds begin
            a1 -= sa[1]; a2 -= sa[2]; a3 -= sa[3]; a4 -= sa[4]; a5 -= sa[5]
        end
        sync_threads()
    end

    @inbounds if tid == 1
        A[1,i] = a1; A[2,i] = a2; A[3,i] = a3; A[4,i] = a4; A[5,i] = a5
    end

    # ---- relax: f = feq + (f - feq)*exp(-dt/tau) ----
    e = exp(-dt/tau)
    @inbounds for t in tid:nt:npts
        aa = (t - 1) % n + 1; rr = (t - 1) ÷ n
        bb = rr % n + 1;      cc = rr ÷ n + 1
        vx = v[aa]; vy = v[bb]; vz = v[cc]
        feq = exp(a1 + a2*vx + a3*vy + a4*vz + a5*(vx*vx + vy*vy + vz*vz))
        f[aa,bb,cc,i] = feq + (f[aa,bb,cc,i] - feq)*e
    end
    return nothing
end

"BGK collision on every cell. `A` is the persistent per-cell Newton state from `dvm_alloc`."
function collide!(f, g::VGridG, A, dt, tau; iters::Int = 30, tol = 1e-13, threads::Int = 128)
    Nx = size(f, 4)
    shmem = (NRED*threads + 8)*sizeof(Float64)
    @cuda threads=threads blocks=Nx shmem=shmem _collide_kernel!(
        f, g.v, g.n, Nx, g.dv^3, A, Float64(dt), Float64(tau), Int32(iters), Float64(tol))
    f
end

# =======================================================================================
# DIAGNOSTICS — the 35 raw moments per cell, on the host (called once at the end).
# =======================================================================================
function moments35_field(f, g::VGridG)
    Nx = size(f, 4); n = g.n; H = Array(f); dv3 = g.dv^3; v = g.vh
    M = zeros(Nx, 35)
    P = [v[a]^p for a in 1:n, p in 0:4]
    @inbounds for i in 1:Nx, c in 1:n, b in 1:n, a in 1:n
        w = H[a,b,c,i]*dv3
        w == 0.0 && continue
        for (m,(p,q,r)) in enumerate(IJK35)
            M[i,m] += w*P[a,p+1]*P[b,q+1]*P[c,r+1]
        end
    end
    M
end

"T = (C200+C020+C002)/3 from a row of 35 raw moments."
function cellT_from_M(Mrow)
    r = Mrow[1]; ux = Mrow[2]/r; uy = Mrow[6]/r; uz = Mrow[16]/r
    ((Mrow[3]/r - ux^2) + (Mrow[10]/r - uy^2) + (Mrow[20]/r - uz^2))/3
end

# ---- host-side discrete Maxwellian, for building the wall states once at setup ----
function discrete_maxwellian_host!(feq, rho, ux, uy, uz, T, vh::Vector{Float64}, dv::Float64;
                                   iters = 80, tol = 1e-14)
    n = length(vh); dv3 = dv^3
    E = 1.5*T + 0.5*(ux^2 + uy^2 + uz^2)
    tgt = [rho, rho*ux, rho*uy, rho*uz, rho*2E]
    a = [log(rho*(2pi*T)^(-1.5)) - (ux^2+uy^2+uz^2)/(2T), ux/T, uy/T, uz/T, -1/(2T)]
    for _ in 1:iters
        mom = zeros(5); J = zeros(5,5)
        @inbounds for k in 1:n, j in 1:n, i in 1:n
            vx = vh[i]; vy = vh[j]; vz = vh[k]; s2 = vx^2 + vy^2 + vz^2
            fval = exp(a[1] + a[2]*vx + a[3]*vy + a[4]*vz + a[5]*s2)*dv3
            b = (1.0, vx, vy, vz, s2)
            for p in 1:5
                mom[p] += b[p]*fval
                for q in 1:5; J[p,q] += b[p]*b[q]*fval; end
            end
        end
        r = mom .- tgt
        norm(r) < tol*max(1.0, rho) && break
        a .-= J\r
    end
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        feq[i,j,k] = exp(a[1] + a[2]*vh[i] + a[3]*vh[j] + a[4]*vh[k] +
                         a[5]*(vh[i]^2 + vh[j]^2 + vh[k]^2))
    end
    feq
end

"""
    wall_pair_host(g, Tw, sgn) -> (CuArray unit-density wall Maxwellian, outgoing half-flux)

`sgn = +1` for the LO wall (outgoing means vx > 0), `-1` for the HI wall.
"""
function wall_pair_host(g::VGridG, Tw::Float64, sgn::Int)
    n = g.n
    M1 = zeros(n, n, n)
    discrete_maxwellian_host!(M1, 1.0, 0.0, 0.0, 0.0, Tw, g.vh, g.dv)
    dv3 = g.dv^3; out = 0.0
    @inbounds for k in 1:n, j in 1:n, a in 1:n
        vx = g.vh[a]
        (sgn*vx > 0) && (out += sgn*vx*M1[a,j,k]*dv3)
    end
    (CuArray(M1), out)
end

# =======================================================================================
# ES-BGK — the collision that gets the Prandtl number right.
#
# WHY THIS MATTERS MORE THAN IT LOOKS. Every comparison against DSMC is currently
# CONFOUNDED, because BGK forces one relaxation time to serve two transport coefficients.
# Measured against real VHS argon it comes out +43% in the shear rate and -28 to -32% in the
# temperature jump: two errors of OPPOSITE SIGN that partially cancel against the closure's
# own error and thereby flatter it. With Pr = 2/3 the reference matches a real gas, and
# closure-versus-reference becomes a direct measurement of truncation error instead of an
# inference from two references that disagree.
#
# THE ALGORITHM IS CHEAPER THAN THE CPU'S, not just ported. The CPU solves a 7x7 Newton for
# the exponential-family parameters, which costs 25 distinct reductions over n^3 velocity
# points PER ITERATION. But the diagonal-covariance Gaussian FACTORISES,
#
#     f = A * gx(vx) * gy(vy) * gz(vz),   g(v) = exp(b v + c v^2)
#
# and on a tensor-product velocity grid that factorisation is EXACT, not approximate. The
# seven constraints then decouple: <vx>/<1> and <vx^2>/<1> involve only (bx, cx), so each
# axis is an independent 2x2 Newton over n points, and A follows from normalisation. One
# O(n^3) reduction for the input moments, three O(n) solves, one O(n^3) write -- against the
# CPU's 25 O(n^3) reductions per iteration.
#
# The same factorisation applies to the isotropic Maxwellian in `_collide_kernel!` above,
# which is therefore also leaving performance on the table. That kernel is validated to
# 1e-14 and is not touched here; the observation is recorded rather than acted on.
#
# Diagonal covariance is COMPLETE for this geometry, not an approximation: in 1D-physical x
# 3D-velocity with x-transport, symmetry prevents any xy/xz/yz correlation from developing.
# A general 10-parameter version would be needed in 2D or 3D physical space.

"kappa = (a-e)/(1-e), a=exp(-y), e=exp(-Pr y); expm1 form for small-y stability. Matches
 the CPU `kappa_es` and Riemann35's ReconDev.bgk_relax_tup, deliberately -- if the two
 solvers integrated the collision differently, a closure comparison would conflate
 truncation error with a splitting difference."
@inline function _kappa_es(Pr::Float64, y::Float64)
    Pr == 1.0 && return 0.0
    -exp(-Pr*y) * expm1((Pr - 1.0)*y) / expm1(-Pr*y)
end

"""
One axis of the factorised fit: find (b, c) with g(v) = exp(b v + c v^2) whose normalised
first and second moments on the grid are `m1t` and `m2t`. Returns (b, c, converged).

Newton on a symmetric 2x2 whose entries are grid sums S0..S4 of v^p g. The initial guess is
the Gaussian's own parameters, b = u/l and c = -1/(2l), which is exact when the target
moments are consistent with a Gaussian -- so this typically converges in 2-3 iterations.
"""
@inline function _fit_axis(v, n::Int, m1t::Float64, m2t::Float64,
                           b0::Float64, c0::Float64, iters::Int, tol::Float64)
    b = b0; c = c0
    @inbounds for _ in 1:iters
        S0 = 0.0; S1 = 0.0; S2 = 0.0; S3 = 0.0; S4 = 0.0
        for a in 1:n
            vv = v[a]
            gg = exp(b*vv + c*vv*vv)
            S0 += gg; S1 += vv*gg; S2 += vv*vv*gg
            S3 += vv*vv*vv*gg; S4 += vv*vv*vv*vv*gg
        end
        (S0 > 0.0 && isfinite(S0)) || return (b, c, false)
        e1 = S1/S0 - m1t
        e2 = S2/S0 - m2t
        (abs(e1) + abs(e2) < tol) && return (b, c, true)
        # J = d(moments)/d(b,c), symmetric: the covariance matrix of (v, v^2) under g
        J11 = S2/S0 - (S1/S0)^2
        J12 = S3/S0 - (S1/S0)*(S2/S0)
        J22 = S4/S0 - (S2/S0)^2
        det = J11*J22 - J12*J12
        (abs(det) > 1e-300) || return (b, c, false)
        b -= ( J22*e1 - J12*e2)/det
        c -= (-J12*e1 + J11*e2)/det
    end
    (b, c, true)
end

function _collide_es_kernel!(f, v, n, Nx, dv3, dt, Kn, Pr, omega, iters::Int32, tol)
    i = blockIdx().x
    tid = threadIdx().x
    nt = blockDim().x
    sh = CUDA.CuDynamicSharedArray(Float64, 7*nt)
    sa = CUDA.CuDynamicSharedArray(Float64, 8, 7*nt*sizeof(Float64))
    npts = n*n*n

    # ---- ONE reduction: rho, rho*u (3), and <v_i^2> (3) -- exactly the inputs needed ----
    m0=0.0; m1=0.0; m2=0.0; m3=0.0; m4=0.0; m5=0.0; m6=0.0
    @inbounds for t in tid:nt:npts
        a = (t-1) % n + 1; r = (t-1) ÷ n
        b = r % n + 1;     c = r ÷ n + 1
        vx = v[a]; vy = v[b]; vz = v[c]
        w = f[a,b,c,i]*dv3
        m0 += w; m1 += w*vx; m2 += w*vy; m3 += w*vz
        m4 += w*vx*vx; m5 += w*vy*vy; m6 += w*vz*vz
    end
    @inbounds begin
        sh[tid]=m0; sh[nt+tid]=m1; sh[2nt+tid]=m2; sh[3nt+tid]=m3
        sh[4nt+tid]=m4; sh[5nt+tid]=m5; sh[6nt+tid]=m6
    end
    sync_threads()
    s = nt >> 1
    while s > 0
        if tid <= s
            @inbounds for q in 0:6; sh[q*nt+tid] += sh[q*nt+tid+s]; end
        end
        sync_threads(); s >>= 1
    end

    # ---- thread 1: the anisotropic target, then three independent 2x2 fits -------------
    @inbounds if tid == 1
        rho = sh[1]
        if rho > 0.0
            ux = sh[nt+1]/rho; uy = sh[2nt+1]/rho; uz = sh[3nt+1]/rho
            cxx = sh[4nt+1]/rho - ux*ux
            cyy = sh[5nt+1]/rho - uy*uy
            czz = sh[6nt+1]/rho - uz*uz
            Theta = (cxx + cyy + czz)/3
            Theta = Theta > 1e-14 ? Theta : 1e-14
            tau = (Kn/2)*Theta^(omega - 1.0)/rho
            y = dt/tau
            if y > 0.0
                k = _kappa_es(Pr, y)
                lxx = max((1-k)*Theta + k*cxx, 1e-14)
                lyy = max((1-k)*Theta + k*cyy, 1e-14)
                lzz = max((1-k)*Theta + k*czz, 1e-14)
                bx, cx, _ = _fit_axis(v, n, ux, lxx + ux*ux, ux/lxx, -1/(2lxx), Int(iters), tol)
                by, cy, _ = _fit_axis(v, n, uy, lyy + uy*uy, uy/lyy, -1/(2lyy), Int(iters), tol)
                bz, cz, _ = _fit_axis(v, n, uz, lzz + uz*uz, uz/lzz, -1/(2lzz), Int(iters), tol)
                # normalisation: A * Gx*Gy*Gz * dv^3 = rho
                Gx = 0.0; Gy = 0.0; Gz = 0.0
                for a in 1:n
                    vv = v[a]
                    Gx += exp(bx*vv + cx*vv*vv)
                    Gy += exp(by*vv + cy*vv*vv)
                    Gz += exp(bz*vv + cz*vv*vv)
                end
                sa[1] = rho/(Gx*Gy*Gz*dv3)
                sa[2] = bx; sa[3] = cx; sa[4] = by
                sa[5] = cy; sa[6] = bz; sa[7] = cz
                sa[8] = exp(-Pr*y)          # NB Pr*y, not y -- the ES-BGK relaxation rate
            else
                sa[1] = -1.0
            end
        else
            sa[1] = -1.0
        end
    end
    sync_threads()

    @inbounds if sa[1] >= 0.0
        A = sa[1]; bx = sa[2]; cx = sa[3]; by = sa[4]
        cy = sa[5]; bz = sa[6]; cz = sa[7]; e = sa[8]
        for t in tid:nt:npts
            a = (t-1) % n + 1; r = (t-1) ÷ n
            b = r % n + 1;     c = r ÷ n + 1
            vx = v[a]; vy = v[b]; vz = v[c]
            feq = A*exp(bx*vx + cx*vx*vx)*exp(by*vy + cy*vy*vy)*exp(bz*vz + cz*vz*vz)
            f[a,b,c,i] = feq + (f[a,b,c,i] - feq)*e
        end
    end
    return nothing
end

"""
    collide_es!(f, g, dt, Kn, Pr, omega; ...)

ES-BGK collision on every cell. `Pr = 1` reduces to BGK exactly (kappa == 0 makes the target
isotropic), which `validate_dvm_esbgk_gpu.jl` uses as its first gate. `tau` is computed per
cell from the local Theta and rho with the VHS exponent `omega`, matching the moment solver's
convention rather than taking a fixed tau -- so a closure comparison differs only in the flux.
"""
function collide_es!(f, g::VGridG, dt, Kn, Pr, omega;
                     iters::Int = 40, tol = 1e-14, threads::Int = 128)
    Nx = size(f, 4)
    shmem = (7*threads + 8)*sizeof(Float64)
    @cuda threads=threads blocks=Nx shmem=shmem _collide_es_kernel!(
        f, g.v, g.n, Nx, g.dv^3, Float64(dt), Float64(Kn), Float64(Pr), Float64(omega),
        Int32(iters), Float64(tol))
    f
end

# =======================================================================================
# EXACT WALL FLUXES — an observable with no extrapolation in it.
#
# WHY. Every wall coefficient in the notes is read by fitting the core of the channel and
# extrapolating to the wall. That estimator has two measured failure modes. At Kn = 0.8 the
# core-fit R^2 falls to 0.9991 in BOTH codes, because the profile stops being linear across
# the middle half -- the fit is being asked for something the data no longer supports. And
# the temperature jump turns out to depend on WHICH wall it is read at, by 6.6% at
# dT/T = 10%, because lambda_eff = tau*sqrt(T) differs between a hot and a cold plate.
#
# The wall stress and wall heat flux have neither problem. They are exact half-space
# integrals of f at the wall face -- no fit, no extrapolation, no choice of window -- and
# they are what DSMC reports natively (the deck already dumps `press shx shy shz`), so the
# comparison tightens on both sides at once. They are also the physically primary quantities:
# drag and heat load are what a wall actually experiences.
#
# The split matters. At the lo wall the INCOMING half (vx < 0) comes from the interior and
# the OUTGOING half (vx > 0) is the wall's own Maxwellian at rho_w. Summing f over all
# velocities in cell 1 would be the cell-centre flux, not the wall flux, and would silently
# omit the wall's contribution.
function _wall_flux_kernel!(out, f, Mw, v, n, Nx, dv3, rho_w, icell, sgn, uw, Tw)
    t = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    if t <= n*n*n
        @inbounds begin
            a = (t-1) % n + 1; r = (t-1) ÷ n
            b = r % n + 1;     c = r ÷ n + 1
            vx = v[a]; vy = v[b]; vz = v[c]
            # outgoing (sgn*vx > 0) is the wall Maxwellian; incoming is the interior cell
            fv = (sgn*vx > 0) ? rho_w*Mw[a,b,c] : f[a,b,c,icell]
            w = fv*dv3
            CUDA.@atomic out[1] += vx*(vy - uw)*w                       # tangential stress
            cx = vx; cy = vy - uw; cz = vz
            CUDA.@atomic out[2] += 0.5*vx*(cx*cx + cy*cy + cz*cz)*w     # normal heat flux
            CUDA.@atomic out[3] += vx*w                                 # net mass flux (~0)
        end
    end
    return nothing
end

"""
    wall_flux(f, g, Mw, rho_w, side; uw=0.0, Tw=1.0) -> (P_xy, q_x, mdot)

Exact tangential momentum flux and normal heat flux at one wall, with the half-space split
done explicitly. `side = :lo` or `:hi`. `mdot` should be ~0 by construction and is returned
as a check, not a diagnostic to be interpreted.
"""
function wall_flux(f, g::VGridG, Mw, rho_w::Float64, side::Symbol; uw = 0.0, Tw = 1.0)
    Nx = size(f, 4); n = g.n
    icell = side === :lo ? 1 : Nx
    sgn   = side === :lo ? 1.0 : -1.0
    out = CUDA.zeros(Float64, 3)
    @cuda threads=256 blocks=cld(n^3, 256) _wall_flux_kernel!(
        out, f, Mw, g.v, n, Nx, g.dv^3, rho_w, icell, sgn, Float64(uw), Float64(Tw))
    h = Array(out)
    (Pxy = h[1], qx = h[2], mdot = h[3])
end

"""
    freemolecular_stress(rho, Uw, T)
    freemolecular_heatflux(rho, Tcold, Thot)

Collisionless limits, used to normalise the wall fluxes into O(1) numbers whose Kn-dependence
is interpretable. Normalising by the CONTINUUM value instead would divide by something that
vanishes in the rarefied limit, which is the wrong way round for a transition-regime sweep.

THE HEAT-FLUX FORM TOOK THREE ATTEMPTS and the two wrong ones are recorded because each
was wrong for an instructive reason. It matters because it is a DENOMINATOR: a wrong
coefficient made the normalised wall heat flux read 1.06, 1.36 and 1.59 at Kn = 0.2, 0.4 and
0.8 -- above the collisionless limit, which is impossible, and is what exposed it. GATE 4
validated the stress limit and nothing validated heat; GATE 5 now does.

  (i)  rho*dT*sqrt(T/2pi)                 -- 2x too SMALL. Written by analogy with the stress
                                             formula rather than derived at all.
  (ii) 2 rho (Th sqrt(Th/2pi) - Tc sqrt(Tc/2pi))
                                          -- 1.5x too LARGE. The one-sided energy flux
                                             2 rho T sqrt(T/2pi) is right (2T per molecule,
                                             not (3/2)T), but this takes the DIFFERENCE of two
                                             beams each at rho = 1, and they cannot both be:
                                             zero net MASS flux forces
                                             rho_c sqrt(Tc) = rho_h sqrt(Th), so the hot beam
                                             is thinner, and that cancels most of the
                                             temperature dependence.
  (iii) 2 rho sqrt(Tbar/2pi) dT            -- correct. Substituting the mass-flux constraint
                                             into (ii) collapses it to this, and it reproduces
                                             the measured collisionless flux to 1.7% -- the
                                             SAME residual GATE 4 reports for stress, i.e. the
                                             shared discretisation error rather than anything
                                             specific to heat.

Linear in dT and therefore exact only to O(dT^2), which is the regime these runs use
(dT/T <= 10%, and 2.5% by default since rem:zetaT-wall-ambiguity).
"""
freemolecular_stress(rho, Uw, T) = rho*Uw*sqrt(T/(2pi))
freemolecular_heatflux(rho, Tcold, Thot) =
    2*rho*sqrt(0.5*(Tcold + Thot)/(2pi))*(Thot - Tcold)

end # module

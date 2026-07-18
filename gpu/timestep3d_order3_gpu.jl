"""
    timestep3d_order3_gpu.jl — on-device 3D order-3 (WENO5 + θ*-IDP) SSP-RK3 march.

GPU analogue of the CPU `step_highorder_3d!` (`src/numerics/highorder_3d.jl`) with
`order=3`.  The state lives on a FULLY-HALOED cube `G (35, nf, nf, nf)`,
`nf = n + 2g`, `g = HALO3 = 8` (the same layout `residual3d_order3_box_gpu!` consumes).

A multi-GPU z-slab march (`march3d_slab_order3_gpu!`, MPI) is provided alongside the
single-GPU march: each rank keeps its interior `(35,n,n,nz_loc)` resident on a haloed
cube `(35,n+2g,n+2g,nz_loc+2g)`; per RK stage the g=8 z-halo planes are host-staged and
exchanged with the z-neighbour ranks (outflow at the global z ends), and the residual is
told which z sides are RANK boundaries so its θ layer takes the min with the z-neighbour
halo cell's θ — rank-consistent, conservative, and bit-identical to the single-GPU march.

Each SSP-RK3 stage mirrors the CPU `step_highorder_3d!` sequence EXACTLY:

  1. refill the cube's outflow halos (clamp / edge-copy in all three axes — the
     single-rank equivalent of `halo_exchange_3d!(M, decomp, :copy)` plus the
     z-outflow pad the CPU driver builds per line; the residual parity harness
     already proved a full clamp reproduces the CPU halo state).
  2. `residual3d_order3_box_gpu!` → interior residual `R`  (passes the FULL `dt`,
     so the θ*-IDP λ = dt/ds is identical to CPU).
  3. RK3 combine on the interior:  Gint = a·G0 + b·Gint + (c·dt)·R.
  4. per-cell realizability projection (`realizable_3D_M4_dev`) — the CPU
     `_project_interior!`.
  5. optional exact-exponential stage-BGK (`bgk_relax_tup`) — the CPU `bgk!`.

RK3 weights (SSP, `step_highorder_3d!`):
  stage1: M = M0 + dt·L(M)                (a,b,c) = (1,   0,   1  )
  stage2: M = ¾M0 + ¼(M + dt·L(M))        (a,b,c) = (¾,   ¼,   ¼  )
  stage3: M = ⅓M0 + ⅔(M + dt·L(M))        (a,b,c) = (⅓,   ⅔,   ⅔  )

CFL dt (matches the GPU order-2 helper `_cfl_from_vmax`):
  dt = (1/3)·dx / max_cell(max(|u|,|v|,|w|) + 4·2.334·√(max(cx,cy,cz)+1e-12)).

fp64 throughout.  Pure addition under `gpu/`; the order-1/2 paths are untouched.
"""
module Timestep3DOrder3GPU

using CUDA, MPI
using Random: randn!          # CUDA.jl extends randn! for CuArrays (CUDA.randn! not exported)

include(joinpath(@__DIR__, "residual3d_order3_gpu.jl"))
using .Residual3DOrder3GPU: residual3d_order3_box_gpu!
using .Residual3DOrder3GPU.RealizeDev: realizable_3D_M4_dev
using .Residual3DOrder3GPU.ReconDev: bgk_relax_tup

export march3d_order3_gpu!, march3d_slab_order3_gpu!, build_haloed_cube, interior_from_cube!

# Halo width for the order-3 WENO5 + θ*-IDP cube. g=8 (matching the CPU order-3
# path, be37651): the WENO5 reconstruction footprint at the outermost interior
# interface reaches ±7 cells, so every interior interface — and the z rank-boundary
# θ halo cell + its first-order anchor — reads real (exchanged / clamped) data, not
# a cell-average fallback. This makes the z-slab multi-GPU march rank-consistent and
# bit-identical to the single-GPU march. (The residual only requires g≥4.)
const HALO3 = 8

@inline _clampi(a::Int, lo::Int, hi::Int) = a < lo ? lo : (a > hi ? hi : a)

# ---------------------------------------------------------------------------
# Outflow halo refill: every halo cell copies the nearest INTERIOR cell (clamp).
# Interior cells map to themselves and are never written, so halo threads only
# ever read interior cells no thread mutates — race-free.  Equivalent to
# `apply_physical_bc_3d!(:copy)` in x/y and the per-line z-outflow pad on the CPU.
# Rectangular (nx,ny,nz); the cubic path is the nx==ny==nz special case (byte-identical).
# ---------------------------------------------------------------------------
function _refill_halo!(G, nfx::Int, nfy::Int, nfz::Int, g::Int, nx::Int, ny::Int, nz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            ca = _clampi(a, g + 1, g + nx)
            cb = _clampi(b, g + 1, g + ny)
            cc = _clampi(c, g + 1, g + nz)
            if a != ca || b != cb || c != cc
                for m in 1:35; G[m, a, b, c] = G[m, ca, cb, cc]; end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Crossflow halo refill (opt-in :crossflow BC; GPU analogue of the CPU
# `apply_physical_bc_3d!(:crossflow)`):
#   low-x  ghosts : INLET   — Dirichlet Maxwellian `inlet` (35 device scalars)
#   high-x ghosts : OUTFLOW — copy nearest interior x cell
#   y ghosts      : PERIODIC (wrap; single-GPU, x/y not decomposed)
#   z ghosts      : OUTFLOW — copy nearest interior z cell
# The inlet is y-uniform, so an x-left ghost is `inlet` regardless of the y wrap
# (inlet dominates the corner, consistently). Ghost threads read only interior
# cells (periodic-y images land in the interior; x/z copies read interior edges)
# and write only ghost cells — race-free, exactly like `_refill_halo!`.
# ---------------------------------------------------------------------------
function _refill_halo_crossflow!(G, nfx::Int, nfy::Int, nfz::Int, g::Int,
                                 nx::Int, ny::Int, nz::Int, inlet)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            interior = (g + 1 <= a <= g + nx) && (g + 1 <= b <= g + ny) && (g + 1 <= c <= g + nz)
            if !interior
                if a <= g
                    # inlet (x-left Dirichlet); y-uniform so periodic is moot
                    for m in 1:35; G[m, a, b, c] = inlet[m]; end
                else
                    ax = a >= g + nx + 1 ? g + nx : a                     # copy x-right, else interior x
                    by = b <= g ? b + ny : (b >= g + ny + 1 ? b - ny : b) # periodic y
                    cz = _clampi(c, g + 1, g + nz)                        # copy z
                    for m in 1:35; G[m, a, b, c] = G[m, ax, by, cz]; end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Direction-agnostic halo refill (unifies _refill_halo! and _refill_halo_crossflow!).
# Per axis, the lo/hi face codes are 0=outflow, 1=inlet, 2=periodic. A ghost lying
# in ANY inlet face takes the `inlet` vector (inlet is transverse-uniform, so it
# dominates corners consistently); otherwise each axis resolves its own source
# index (outflow clamp or periodic wrap) and the interior cell is copied. This is
# byte-identical to _refill_halo! for all-outflow codes and to
# _refill_halo_crossflow! for codes (inlet,outflow,periodic,periodic,outflow,outflow).
# A :sponge face is :outflow here (code 0); its absorbing effect is _sponge_interior!.
# ---------------------------------------------------------------------------
@inline function _axis_src(i::Int, n::Int, g::Int, clo::Int, chi::Int)
    if i <= g
        clo == 1 && return (i, true)              # inlet
        clo == 2 && return (i + n, false)         # periodic wrap (lo ghost <- hi interior)
        return (g + 1, false)                     # outflow clamp
    elseif i >= g + n + 1
        chi == 1 && return (i, true)
        chi == 2 && return (i - n, false)
        return (g + n, false)
    else
        return (i, false)                         # interior along this axis
    end
end

function _refill_halo_faces!(G, nfx::Int, nfy::Int, nfz::Int, g::Int, nx::Int, ny::Int, nz::Int,
                             inlet, cxlo::Int, cxhi::Int, cylo::Int, cyhi::Int, czlo::Int, czhi::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            interior = (g+1 <= a <= g+nx) && (g+1 <= b <= g+ny) && (g+1 <= c <= g+nz)
            if !interior
                sa, ia = _axis_src(a, nx, g, cxlo, cxhi)
                sb, ib = _axis_src(b, ny, g, cylo, cyhi)
                sc, ic = _axis_src(c, nz, g, czlo, czhi)
                if ia || ib || ic
                    for m in 1:35; G[m, a, b, c] = inlet[m]; end
                else
                    for m in 1:35; G[m, a, b, c] = G[m, sa, sb, sc]; end
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Absorbing sponge source: exact-exponential relaxation of the interior toward a
# freestream Maxwellian `ref`, with a precomputed per-cell `ramp` (linear i+(j-1)nx
# +(k-1)nx*ny, in [0,1]). No-op where ramp==0. Same integrator as BGK; the GPU
# analogue of apply_sponge! (src/numerics/sponge.jl).
# ---------------------------------------------------------------------------
function _sponge_interior!(G, nx::Int, ny::Int, nz::Int, g::Int, ramp, ref, rate::Float64, dt::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            rp = ramp[idx]
            if rp > 0.0
                i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
                j = r % ny + 1;         k = r ÷ ny + 1
                f = exp(-rate * rp * dt)
                for m in 1:35
                    G[m, g+i, g+j, g+k] = ref[m] + (G[m, g+i, g+j, g+k] - ref[m]) * f
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Stochastic forcing (thermal-noise mimic): add a small zero-mean random transverse
# (y) velocity kick to M010 (moment index 6) of interior cells inside the box
# [i0,i1]x[j0,j1] (cell indices), per step. `noise` is a per-cell N(0,1) device
# array (regenerated each step); the kick is amp*sqrt(dt)*rho*noise (Brownian
# scaling so the perturbation variance is dt-independent). Mimics the DSMC thermal
# fluctuations that trigger the subcritical Kármán onset our deterministic solver
# otherwise lacks. Kick is tiny => central moments (hence realizability) essentially
# unchanged; the per-stage projection absorbs any drift.
# ---------------------------------------------------------------------------
function _noise_forcing!(G, noise, nx::Int, ny::Int, nz::Int, g::Int, kick::Float64,
                         i0::Int, i1::Int, j0::Int, j1::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            if i0 <= i <= i1 && j0 <= j <= j1
                G[6, g+i, g+j, g+k] += kick * G[1, g+i, g+j, g+k] * noise[idx]
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# FDT-calibrated fluctuating stress (Landau-Lifshitz thermal noise) — the moment-
# space analogue of the paper's stochastic-flux term. A random symmetric stress
# s_ij is added to the momentum flux; it enters the momentum moments M100(2),
# M010(6), M001(16) as dt*(div s). Two kernels: (1) `_scale_fluct_stress!` turns a
# raw N(0,1) buffer `sbuf(6,nx,ny,nz)` (components 1=xx,2=yy,3=zz,4=xy,5=xz,6=yz)
# into a stress with per-cell FDT amplitude sigma = intensity*sqrt(2*T*eta/(dV*dt))
# from the local temperature T and the BGK dilute-gas viscosity eta = Kn*sqrt(T)/2;
# (2) `_apply_fluct_div!` adds the discrete (central-difference => face-averaged,
# telescoping) divergence to the momentum moments. Stress is 0 outside the interior
# (fluctuation vanishes at the boundary). For Nz=1 the z-divergence auto-vanishes.
# ---------------------------------------------------------------------------
function _scale_fluct_stress!(sbuf, G, nx::Int, ny::Int, nz::Int, g::Int,
                              intensity::Float64, dV::Float64, dt::Float64, Kn::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            ia = g+i; ja = g+j; ka = g+k
            invr = 1.0 / G[1, ia, ja, ka]
            ux = G[2, ia, ja, ka] * invr; uy = G[6, ia, ja, ka] * invr; uz = G[16, ia, ja, ka] * invr
            Txx = G[3, ia, ja, ka] * invr - ux*ux
            Tyy = G[10, ia, ja, ka] * invr - uy*uy
            Tzz = G[20, ia, ja, ka] * invr - uz*uz
            T = (Txx + Tyy + Tzz) / 3.0
            T = T > 0.0 ? T : 0.0
            eta = 0.5 * Kn * sqrt(T)                        # BGK dilute-gas viscosity
            sigma = intensity * sqrt(2.0 * T * eta / (dV * dt) + 1e-300)
            for c in 1:6
                sbuf[c, i, j, k] *= sigma
            end
        end
    end
    return nothing
end

function _apply_fluct_div!(G, sbuf, nx::Int, ny::Int, nz::Int, g::Int,
                           dt::Float64, hdx::Float64, hdy::Float64, hdz::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            xp = i < nx; xm = i > 1; yp = j < ny; ym = j > 1; zp = k < nz; zm = k > 1
            # M100: d_x s_xx + d_y s_xy + d_z s_xz   (xx=1, xy=4, xz=5)
            dM2 = ((xp ? sbuf[1,i+1,j,k] : 0.0) - (xm ? sbuf[1,i-1,j,k] : 0.0)) * hdx +
                  ((yp ? sbuf[4,i,j+1,k] : 0.0) - (ym ? sbuf[4,i,j-1,k] : 0.0)) * hdy +
                  ((zp ? sbuf[5,i,j,k+1] : 0.0) - (zm ? sbuf[5,i,j,k-1] : 0.0)) * hdz
            # M010: d_x s_xy + d_y s_yy + d_z s_yz   (xy=4, yy=2, yz=6)
            dM6 = ((xp ? sbuf[4,i+1,j,k] : 0.0) - (xm ? sbuf[4,i-1,j,k] : 0.0)) * hdx +
                  ((yp ? sbuf[2,i,j+1,k] : 0.0) - (ym ? sbuf[2,i,j-1,k] : 0.0)) * hdy +
                  ((zp ? sbuf[6,i,j,k+1] : 0.0) - (zm ? sbuf[6,i,j,k-1] : 0.0)) * hdz
            # M001: d_x s_xz + d_y s_yz + d_z s_zz   (xz=5, yz=6, zz=3)
            dM16 = ((xp ? sbuf[5,i+1,j,k] : 0.0) - (xm ? sbuf[5,i-1,j,k] : 0.0)) * hdx +
                   ((yp ? sbuf[6,i,j+1,k] : 0.0) - (ym ? sbuf[6,i,j-1,k] : 0.0)) * hdy +
                   ((zp ? sbuf[3,i,j,k+1] : 0.0) - (zm ? sbuf[3,i,j,k-1] : 0.0)) * hdz
            ia = g+i; ja = g+j; ka = g+k
            G[2,  ia, ja, ka] += dt * dM2
            G[6,  ia, ja, ka] += dt * dM6
            G[16, ia, ja, ka] += dt * dM16
        end
    end
    return nothing
end

# copy the interior of the haloed cube into a compact (35,nx,ny,nz) buffer (RK M0).
function _copy_interior!(G0, G, nx::Int, ny::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            for m in 1:35; G0[m, i, j, k] = G[m, g + i, g + j, g + k]; end
        end
    end
    return nothing
end

# write a compact interior (35,nx,ny,nz) into the interior region of a haloed cube.
function _set_interior!(G, Mi, nx::Int, ny::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            for m in 1:35; G[m, g + i, g + j, g + k] = Mi[m, i, j, k]; end
        end
    end
    return nothing
end

# RK combine on the interior:  Gint = a*G0 + b*Gint + cdt*R  (cdt already = c*dt).
function _rk_combine!(G, G0, R, nx::Int, ny::Int, nz::Int, g::Int, a::Float64, b::Float64, cdt::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            ga = g + i; gb = g + j; gc = g + k
            for m in 1:35
                G[m, ga, gb, gc] = a * G0[m, i, j, k] + b * G[m, ga, gb, gc] + cdt * R[m, i, j, k]
            end
        end
    end
    return nothing
end

# per-cell realizability projection on the interior (CPU `_project_interior!`).
function _proj_interior!(G, nx::Int, ny::Int, nz::Int, g::Int, Ma::Float64, s3max::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            ga = g + i; gb = g + j; gc = g + k
            P = realizable_3D_M4_dev(
                G[1,ga,gb,gc],  G[2,ga,gb,gc],  G[3,ga,gb,gc],  G[4,ga,gb,gc],  G[5,ga,gb,gc],
                G[6,ga,gb,gc],  G[7,ga,gb,gc],  G[8,ga,gb,gc],  G[9,ga,gb,gc],  G[10,ga,gb,gc],
                G[11,ga,gb,gc], G[12,ga,gb,gc], G[13,ga,gb,gc], G[14,ga,gb,gc], G[15,ga,gb,gc],
                G[16,ga,gb,gc], G[17,ga,gb,gc], G[18,ga,gb,gc], G[19,ga,gb,gc], G[20,ga,gb,gc],
                G[21,ga,gb,gc], G[22,ga,gb,gc], G[23,ga,gb,gc], G[24,ga,gb,gc], G[25,ga,gb,gc],
                G[26,ga,gb,gc], G[27,ga,gb,gc], G[28,ga,gb,gc], G[29,ga,gb,gc], G[30,ga,gb,gc],
                G[31,ga,gb,gc], G[32,ga,gb,gc], G[33,ga,gb,gc], G[34,ga,gb,gc], G[35,ga,gb,gc],
                Ma, s3max)
            for m in 1:35; G[m, ga, gb, gc] = P[m]; end
        end
    end
    return nothing
end

# optional exact-exponential stage-BGK relaxation on the interior (CPU `bgk!`).
function _bgk_interior!(G, nx::Int, ny::Int, nz::Int, g::Int, dt::Float64, kn::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            ga = g + i; gb = g + j; gc = g + k
            C = ntuple(m -> G[m, ga, gb, gc], Val(35))
            out = bgk_relax_tup(C, dt, kn)
            for m in 1:35; G[m, ga, gb, gc] = out[m]; end
        end
    end
    return nothing
end

# per-interior-cell CFL speed (mirrors the order-2 `_speed_box_kernel!`).
function _speed_interior!(svec, G, nx::Int, ny::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r0 = (idx - 1) ÷ nx
            j = r0 % ny + 1;        k = r0 ÷ ny + 1
            ga = g + i; gb = g + j; gc = g + k
            r = G[1, ga, gb, gc]
            if r > 0.0
                u = G[2,ga,gb,gc]/r; v = G[6,ga,gb,gc]/r; w = G[16,ga,gb,gc]/r
                cx = G[3,ga,gb,gc]/r - u*u; cy = G[10,ga,gb,gc]/r - v*v; cz = G[20,ga,gb,gc]/r - w*w
                if cx < 0.0; cx = 0.0; end
                if cy < 0.0; cy = 0.0; end
                if cz < 0.0; cz = 0.0; end
                au = abs(u); av = abs(v); aw = abs(w)
                amax = au > av ? au : av; amax = amax > aw ? amax : aw
                cmax = cx > cy ? cx : cy; cmax = cmax > cz ? cmax : cz
                svec[idx] = amax + 4.0 * 2.334 * sqrt(cmax + 1e-12)
            else
                svec[idx] = 0.0
            end
        end
    end
    return nothing
end

# Rigid immersed obstacle: interior cells within the disk (center (cx,cy) in
# cell-index units, r2 = radius-in-cells^2) are set to the held rest state.
function _apply_obstacle!(G, nx::Int, ny::Int, nz::Int, g::Int,
                          cx::Float64, cy::Float64, r2::Float64, ostate)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r0 = (idx - 1) ÷ nx
            j = r0 % ny + 1;        k = r0 ÷ ny + 1
            di = i - cx; dj = j - cy
            if di * di + dj * dj <= r2
                ga = g + i; gb = g + j; gc = g + k
                for m in 1:35; G[m, ga, gb, gc] = ostate[m]; end
            end
        end
    end
    return nothing
end

"""
    build_haloed_cube(Mi::CuArray{Float64,4}; threads=128) -> CuArray{Float64,4}

Build a `g=HALO3` outflow-haloed cube `(35, n+2g, n+2g, n+2g)` from a device interior
`(35, n, n, n)`: the interior is copied in and the halos are clamp-filled (outflow,
nearest-interior). This is the SINGLE source of the interior→haloed-cube bridge shared
by every `march3d_order3_gpu!` caller (the standalone validation driver AND the standard
`run_gpu_3d` order-3 path) — the host clamp-fill that used to live in
`gpu/validation/run_hiorder3_ma100_gpu.jl` produced the exact same cube.
"""
function build_haloed_cube(Mi::CuArray{Float64,4}; threads::Int=128)
    @assert size(Mi, 1) == 35 "interior must be (35,nx,ny,nz)"
    nx = size(Mi, 2); ny = size(Mi, 3); nz = size(Mi, 4)
    g = HALO3; nfx = nx + 2g; nfy = ny + 2g; nfz = nz + 2g
    G = CUDA.zeros(Float64, 35, nfx, nfy, nfz)
    @cuda threads=threads blocks=cld(nx * ny * nz, threads) _set_interior!(G, Mi, nx, ny, nz, g)
    @cuda threads=threads blocks=cld(nfx * nfy * nfz, threads) _refill_halo!(G, nfx, nfy, nfz, g, nx, ny, nz)
    CUDA.synchronize()
    return G
end

"""
    interior_from_cube!(Mi::CuArray{Float64,4}, G::CuArray{Float64,4}; threads=128) -> Mi

Copy the interior `(35, nx, ny, nz)` out of a `g=HALO3` haloed cube `G` into `Mi`, in place
(the inverse of `build_haloed_cube`). Reuses the same `_copy_interior!` kernel the march
uses for its RK `M0` snapshot. Rectangular; the cubic case is nx==ny==nz.
"""
function interior_from_cube!(Mi::CuArray{Float64,4}, G::CuArray{Float64,4}; threads::Int=128)
    nx = size(Mi, 2); ny = size(Mi, 3); nz = size(Mi, 4); g = HALO3
    @assert size(G) == (35, nx + 2g, ny + 2g, nz + 2g) "G must be the matching haloed cube"
    @cuda threads=threads blocks=cld(nx * ny * nz, threads) _copy_interior!(Mi, G, nx, ny, nz, g)
    return Mi
end

# Preset -> ((xlo,xhi,ylo,yhi,zlo,zhi) face codes, sponge bools). Codes: 0=outflow,
# 1=inlet, 2=periodic. Mirrors the CPU face_bc.jl presets (this GPU module is
# standalone — no Riemann35 dependency — so the small preset table lives here too).
const _GPU_BC_PRESETS = Dict{Symbol,Tuple{NTuple{6,Int},NTuple{6,Bool}}}(
    :copy               => ((0,0,0,0,0,0), (false,false,false,false,false,false)),
    :crossflow          => ((1,0,2,2,0,0), (false,false,false,false,false,false)),
    :crossflow_absorb_y => ((1,0,0,0,0,0), (false,false,true, true, false,false)),
)

# bc may be a preset Symbol or an explicit ((codes...),(sponge...)) tuple.
_gpu_bc_codes(bc::Symbol) = get(_GPU_BC_PRESETS, bc) do
    error("march3d_order3_gpu!: unknown bc=:$bc (known: $(sort(collect(keys(_GPU_BC_PRESETS)))))")
end
_gpu_bc_codes(bc::Tuple) = bc

# Per-cell sponge ramp in [0,1], linear column-major (i+(j-1)nx+(k-1)nx*ny) to match
# the kernel index. Mirrors src/numerics/sponge.jl build_sponge_ramp.
function _build_sponge_ramp_host(spg::NTuple{6,Bool}, nx::Int, ny::Int, nz::Int, width::Int; power::Int=2)
    ramp = zeros(Float64, nx, ny, nz)
    width <= 0 && return vec(ramp)
    n = (nx, ny, nz)
    for (a, loi, hii) in ((1, 1, 2), (2, 3, 4), (3, 5, 6))
        na = n[a]; w = min(width, na)
        if spg[loi]
            for d in 1:w
                r = ((width - d + 1) / width)^power
                idxs = ntuple(x -> x == a ? d : Colon(), 3)
                @views ramp[idxs...] .= max.(ramp[idxs...], r)
            end
        end
        if spg[hii]
            for d in 1:w
                r = ((width - d + 1) / width)^power
                idxs = ntuple(x -> x == a ? na - d + 1 : Colon(), 3)
                @views ramp[idxs...] .= max.(ramp[idxs...], r)
            end
        end
    end
    vec(ramp)
end

"""
    march3d_order3_gpu!(G, dx, Ma, nstep; dts=nothing, s3max=…, stage_bgk=false,
                        Kn=Inf, threads=128) -> Vector{Float64}

Advance the fully-haloed order-3 cube `G (35, n+2g, n+2g, n+2g)` (g=HALO3=8) for
`nstep` SSP-RK3 steps.  Outflow BC on all six faces.  If `dts` is given those dt are
used verbatim; else local CFL each step.  Returns the dt vector used.  `G` is updated
in place and left with its outflow halos refilled.
"""
function march3d_order3_gpu!(G::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer;
                             dts=nothing, s3max::Real = max(40.0, 4.0 + abs(Ma)/2.0),
                             stage_bgk::Bool = false, Kn::Real = Inf, threads::Int = 128,
                             theta_closed::Bool = true, use_logjacobi_recon::Bool = false,
                             first_order::Bool = false, bc = :copy, inlet = nothing,
                             obst_state = nothing, obst_cx::Real = 0.0, obst_cy::Real = 0.0,
                             obst_r2::Real = 0.0,
                             sponge_ref = nothing, sponge_width::Int = 0, sponge_rate::Real = 0.0,
                             noise_amp::Real = 0.0, noise_box = nothing,
                             fluct_intensity::Real = 0.0)
    @assert size(G, 1) == 35 "G must be (35,nfx,nfy,nfz)"
    g = HALO3
    nfx = size(G, 2); nfy = size(G, 3); nfz = size(G, 4)
    nx = nfx - 2g; ny = nfy - 2g; nz = nfz - 2g
    @assert min(nx, ny, nz) >= 1 "interior extents nx,ny,nz = nf-2g must be ≥ 1 (got $nx,$ny,$nz)"

    dxf = Float64(dx); Maf = Float64(Ma); s3f = Float64(s3max); knf = Float64(Kn)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))

    # Direction-agnostic per-face BC: expand to face codes + sponge flags.
    (codes, spg) = _gpu_bc_codes(bc)
    cxlo, cxhi, cylo, cyhi, czlo, czhi = codes
    any_inlet  = any(==(1), codes)

    # inlet Maxwellian on the device (35,). A dummy zeros vector when no inlet face
    # (never read by the kernel then). :copy path byte-identical (all-outflow codes).
    inlet_h = inlet === nothing ? nothing : Float64.(collect(inlet))
    if any_inlet
        inlet_h === nothing && error("march3d_order3_gpu!: an inlet face requires inlet (35-vector)")
        length(inlet_h) == 35 || error("inlet must have 35 entries")
    end
    inlet_d = CuArray(inlet_h === nothing ? zeros(Float64, 35) : inlet_h)

    # Absorbing sponge layer (opt-in via :sponge faces). ref = freestream Maxwellian
    # (sponge_ref, or the inlet). Ramp built once on host, uploaded.
    sponge_ramp_d = nothing; sponge_ref_d = nothing; sprate = Float64(sponge_rate)
    if any(spg)
        sref = sponge_ref === nothing ? inlet_h : Float64.(collect(sponge_ref))
        sref === nothing && error("a :sponge face requires sponge_ref (or inlet) as the freestream state")
        length(sref) == 35 || error("sponge_ref must have 35 entries")
        sw = sponge_width > 0 ? sponge_width : max(4, round(Int, 0.08 * min(nx, ny)))
        sponge_ramp_d = CuArray(_build_sponge_ramp_host(spg, nx, ny, nz, sw))
        sponge_ref_d  = CuArray(sref)
    end

    # Stochastic forcing (opt-in): per-cell N(0,1) buffer + forcing box in cell
    # indices (default: whole interior). kick = noise_amp*sqrt(dt) applied each step.
    noise_buf = nothing; noise_kickf = Float64(noise_amp); ni0 = 1; ni1 = nx; nj0 = 1; nj1 = ny
    if noise_kickf > 0
        noise_buf = CUDA.zeros(Float64, nx * ny * nz)
        if noise_box !== nothing
            ni0 = clamp(Int(noise_box[1]), 1, nx); ni1 = clamp(Int(noise_box[2]), 1, nx)
            nj0 = clamp(Int(noise_box[3]), 1, ny); nj1 = clamp(Int(noise_box[4]), 1, ny)
        end
    end

    # FDT fluctuating-stress (Landau-Lifshitz thermal noise): 6-component stress
    # buffer, refilled with N(0,1) each step, scaled to the FDT amplitude, then its
    # divergence added to the momentum moments. dV = cubic cell volume.
    sbuf = nothing; flucti = Float64(fluct_intensity); dVf = dxf^3
    if flucti > 0
        sbuf = CUDA.zeros(Float64, 6, nx, ny, nz)
    end

    # Rigid immersed obstacle (opt-in): held rest Maxwellian on a disk of interior
    # cells, re-imposed at the end of every step (rigid no-slip cylinder).
    obst_d = obst_state === nothing ? nothing : CuArray(Float64.(collect(obst_state)))
    obst_cxf = Float64(obst_cx); obst_cyf = Float64(obst_cy); obst_r2f = Float64(obst_r2)

    R    = CUDA.zeros(Float64, 35, nx, ny, nz)
    G0   = CUDA.zeros(Float64, 35, nx, ny, nz)
    svec = CUDA.zeros(Float64, nx * ny * nz)

    bcube = cld(nfx * nfy * nfz, threads)
    bint  = cld(nx * ny * nz, threads)

    # Refill halos via the direction-agnostic face-code kernel (hoisted; dispatch
    # once). Byte-identical to _refill_halo! for :copy and _refill_halo_crossflow!
    # for :crossflow (same memory ops).
    refill!() = (@cuda threads=threads blocks=bcube _refill_halo_faces!(
        G, nfx, nfy, nfz, g, nx, ny, nz, inlet_d, cxlo, cxhi, cylo, cyhi, czlo, czhi))

    # (a, b, c) RK3 stage weights: Gint = a*G0 + b*Gint + (c*dt)*R
    stages = ((1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1.0/3.0, 2.0/3.0, 2.0/3.0))

    used = Vector{Float64}(undef, nstep)
    for s in 1:nstep
        if dts_host === nothing
            @cuda threads=threads blocks=bint _speed_interior!(svec, G, nx, ny, nz, g)
            vmax = CUDA.@allowscalar maximum(svec)
            dt = (1.0/3.0) * dxf / max(vmax, 1e-12)
        else
            dt = dts_host[s]
        end
        used[s] = dt

        @cuda threads=threads blocks=bint _copy_interior!(G0, G, nx, ny, nz, g)
        for (a, b, c) in stages
            refill!()
            residual3d_order3_box_gpu!(R, G, nx, ny, nz, g, dxf, dxf, dxf, Maf, dt;
                                       s3max=s3f, threads=threads, theta_closed=theta_closed,
                                       use_logjacobi_recon=use_logjacobi_recon,
                                       first_order=first_order)
            @cuda threads=threads blocks=bint _rk_combine!(G, G0, R, nx, ny, nz, g, a, b, c * dt)
            @cuda threads=threads blocks=bint _proj_interior!(G, nx, ny, nz, g, Maf, s3f)
            if stage_bgk
                @cuda threads=threads blocks=bint _bgk_interior!(G, nx, ny, nz, g, dt, knf)
            end
        end
        # stochastic forcing: random transverse-velocity kick (before the obstacle
        # re-imposition so held cells stay exactly fixed).
        if noise_buf !== nothing
            randn!(noise_buf)
            @cuda threads=threads blocks=bint _noise_forcing!(G, noise_buf, nx, ny, nz, g,
                noise_kickf * sqrt(dt), ni0, ni1, nj0, nj1)
        end
        # FDT fluctuating-stress thermal noise: random symmetric stress -> div added
        # to the momentum moments (Landau-Lifshitz). Brownian sqrt(dt) is inside the
        # FDT sigma (1/sqrt(dt) factor), so the dt weight of div(s) gives net sqrt(dt).
        if sbuf !== nothing
            randn!(sbuf)
            @cuda threads=threads blocks=bint _scale_fluct_stress!(sbuf, G, nx, ny, nz, g, flucti, dVf, dt, knf)
            hd = 1.0 / (2.0 * dxf)
            @cuda threads=threads blocks=bint _apply_fluct_div!(G, sbuf, nx, ny, nz, g, dt, hd, hd, hd)
        end
        # rigid immersed obstacle: re-impose the held rest state each step
        if obst_d !== nothing
            @cuda threads=threads blocks=bint _apply_obstacle!(G, nx, ny, nz, g, obst_cxf, obst_cyf, obst_r2f, obst_d)
        end
        # absorbing sponge: relax the boundary zone toward the freestream each step
        if sponge_ramp_d !== nothing
            @cuda threads=threads blocks=bint _sponge_interior!(G, nx, ny, nz, g, sponge_ramp_d, sponge_ref_d, sprate, dt)
        end
    end
    refill!()
    CUDA.synchronize()
    return used
end

# ---------------------------------------------------------------------------
# z-slab x/y outflow halo refill. Clamps ONLY the x/y ghost columns to the nearest
# interior column, over EVERY z plane (interior AND z-ghost). The z-ghost planes are
# filled separately (neighbour exchange / global outflow) BEFORE this runs, so at a
# z-ghost plane the x/y clamp reads the freshly-exchanged neighbour interior — making
# the x/y ghosts at that plane bit-identical to what the neighbour rank computes for
# its own interior plane (needed by the z rank-boundary θ halo-cell anchor).
# ---------------------------------------------------------------------------
function _refill_xy_halo_slab!(G, nfx::Int, nfz::Int, g::Int, n::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfx * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfx + 1;         c = r ÷ nfx + 1
            ca = _clampi(a, g + 1, g + n)
            cb = _clampi(b, g + 1, g + n)
            if a != ca || b != cb
                for m in 1:35; G[m, a, b, c] = G[m, ca, cb, c]; end
            end
        end
    end
    return nothing
end

"""
    march3d_slab_order3_gpu!(Mi, dx, Ma, nstep, comm; dts=nothing, s3max=…,
                             stage_bgk=false, Kn=Inf, threads=128) -> Vector{Float64}

Multi-GPU z-slab order-3 (WENO5 + θ*-IDP) SSP-RK3 march. `Mi` is this rank's resident
interior slab `(35, n, n, nz_loc)` (x/y are global — not decomposed; z is split across
ranks in rank order). Each rank keeps a haloed cube `(35, n+2g, n+2g, nz_loc+2g)`,
`g = HALO3 = 8`. Per RK stage:

  1. z-halo exchange: host-stage this rank's top/bottom `g` interior z-planes (interior
     x/y block) and `MPI.Sendrecv!` them to the z-neighbour ranks (outflow edge-copy at
     the global z ends); then clamp the x/y outflow ghosts over all z planes.
  2. `residual3d_order3_box_gpu!` on the extended slab, with `rank_bnd=(zlo,zhi)` set for
     the z sides that face a neighbour rank (rank>0 → zlo; rank<nranks-1 → zhi) — so the
     shared z-interface θ = min(own θ, z-neighbour halo-cell θ), single-valued across
     ranks. Global z ends keep own-cell θ.
  3. RK3 combine + per-cell realizability projection (+ optional stage-BGK) on the interior.

`dt` = `Allreduce(max)` of per-rank max wave speed (exact ⇒ identical to single-GPU),
unless `dts` is supplied. Returns the dt vector used. `Mi` is updated in place.
"""
function march3d_slab_order3_gpu!(Mi::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer, comm;
                                  dts=nothing, s3max::Real = max(40.0, 4.0 + abs(Ma)/2.0),
                                  stage_bgk::Bool = false, Kn::Real = Inf, threads::Int = 128,
                                  theta_closed::Bool = true)
    rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
    @assert size(Mi, 1) == 35 "Mi must be (35,n,n,nz_loc)"
    n = size(Mi, 2); nzloc = size(Mi, 4)
    @assert size(Mi) == (35, n, n, nzloc) "Mi must be (35,n,n,nz_loc) with nx==ny==n"
    g = HALO3
    @assert nzloc >= g "order-3 z-slab needs nz_loc >= g=$g (got nz_loc=$nzloc)"
    nfx = n + 2g; nfz = nzloc + 2g
    down = rank > 0          ? rank - 1 : MPI.PROC_NULL   # lower-z neighbour
    up   = rank < nranks - 1 ? rank + 1 : MPI.PROC_NULL   # upper-z neighbour
    # rank-boundary flags: multi-GPU decomposes z only, so x/y stay false (their
    # axis-generic branches in the residual are present but DORMANT on the GPU).
    rb   = (xlo = false, xhi = false, ylo = false, yhi = false,
            zlo = rank > 0, zhi = rank < nranks - 1)

    dxf = Float64(dx); Maf = Float64(Ma); s3f = Float64(s3max); knf = Float64(Kn)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))

    G    = CUDA.zeros(Float64, 35, nfx, nfx, nfz)   # resident haloed slab cube
    R    = CUDA.zeros(Float64, 35, n, n, nzloc)
    G0   = CUDA.zeros(Float64, 35, n, n, nzloc)
    svec = CUDA.zeros(Float64, n * n * nzloc)

    bint = cld(n * n * nzloc, threads)
    bxy  = cld(nfx * nfx * nfz, threads)

    @cuda threads=threads blocks=bint _set_interior!(G, Mi, n, n, nzloc, g)

    # pinned host halo buffers: the interior-x/y block of g z-planes.
    pin() = (h = Array{Float64}(undef, 35, n, n, g); CUDA.pin(h); h)
    hsB = pin(); hsT = pin(); hrB = pin(); hrT = pin()
    ib = g + 1;         it = 2g                    # bottom interior halo z-planes [g+1 .. 2g]
    tb = g + nzloc - g + 1; tt = g + nzloc         # top interior halo z-planes
    gb0 = g + nzloc + 1                            # first top z-ghost plane

    used = Vector{Float64}(undef, nstep)
    stages = ((1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1.0/3.0, 2.0/3.0, 2.0/3.0))

    # L!: refresh halos (z exchange + x/y clamp), then residual on the extended slab.
    function refresh_halos!()
        # stage this rank's top/bottom g interior z-planes (interior x/y block) to host
        copyto!(hsB, @view G[:, g+1:g+n, g+1:g+n, ib:it])
        copyto!(hsT, @view G[:, g+1:g+n, g+1:g+n, tb:tt])
        CUDA.synchronize()
        MPI.Sendrecv!(hsT, hrB, comm; dest=up,   source=down, sendtag=1, recvtag=1)
        MPI.Sendrecv!(hsB, hrT, comm; dest=down, source=up,   sendtag=2, recvtag=2)
        # bottom z-ghosts (planes 1..g): neighbour interior, or outflow at global z=lo
        if down == MPI.PROC_NULL
            @inbounds for c in 1:g
                G[:, g+1:g+n, g+1:g+n, c] .= @view G[:, g+1:g+n, g+1:g+n, g+1]
            end
        else
            copyto!(@view(G[:, g+1:g+n, g+1:g+n, 1:g]), reshape(hrB, 35, n, n, g))
        end
        # top z-ghosts (planes gb0..nfz): neighbour interior, or outflow at global z=hi
        if up == MPI.PROC_NULL
            @inbounds for c in 1:g
                G[:, g+1:g+n, g+1:g+n, gb0+c-1] .= @view G[:, g+1:g+n, g+1:g+n, g+nzloc]
            end
        else
            copyto!(@view(G[:, g+1:g+n, g+1:g+n, gb0:nfz]), reshape(hrT, 35, n, n, g))
        end
        # x/y outflow ghosts over ALL z planes (reads freshly-set z-ghost interior x/y)
        @cuda threads=threads blocks=bxy _refill_xy_halo_slab!(G, nfx, nfz, g, n)
        return nothing
    end

    for s in 1:nstep
        if dts_host === nothing
            @cuda threads=threads blocks=bint _speed_interior!(svec, G, n, n, nzloc, g)
            lvmax = CUDA.@allowscalar maximum(svec)
            gmax  = MPI.Allreduce(lvmax, max, comm)
            dt = (1.0/3.0) * dxf / max(gmax, 1e-12)
        else
            dt = dts_host[s]
        end
        used[s] = dt

        @cuda threads=threads blocks=bint _copy_interior!(G0, G, n, n, nzloc, g)
        for (a, b, c) in stages
            refresh_halos!()
            residual3d_order3_box_gpu!(R, G, n, n, nzloc, g, dxf, dxf, dxf, Maf, dt;
                                       s3max=s3f, threads=threads, rank_bnd=rb, theta_closed=theta_closed)
            @cuda threads=threads blocks=bint _rk_combine!(G, G0, R, n, n, nzloc, g, a, b, c * dt)
            @cuda threads=threads blocks=bint _proj_interior!(G, n, n, nzloc, g, Maf, s3f)
            if stage_bgk
                @cuda threads=threads blocks=bint _bgk_interior!(G, n, n, nzloc, g, dt, knf)
            end
        end
    end

    @cuda threads=threads blocks=bint _copy_interior!(Mi, G, n, n, nzloc, g)  # sync interior back
    CUDA.synchronize()
    return used
end

end # module

"""
    timestep3d_order3_gpu.jl — on-device 3D order-3 (WENO5 + θ*-IDP) SSP-RK3 march.

GPU analogue of the CPU `step_highorder_3d!` (`src/numerics/highorder_3d.jl`) with
`order=3`.  The state lives on a FULLY-HALOED cube `G (35, nf, nf, nf)`,
`nf = n + 2g`, `g = 4` (the same layout `residual3d_order3_box_gpu!` consumes).

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

using CUDA

include(joinpath(@__DIR__, "residual3d_order3_gpu.jl"))
using .Residual3DOrder3GPU: residual3d_order3_box_gpu!
using .Residual3DOrder3GPU.RealizeDev: realizable_3D_M4_dev
using .Residual3DOrder3GPU.ReconDev: bgk_relax_tup

export march3d_order3_gpu!, build_haloed_cube, interior_from_cube!

const HALO3 = 4

@inline _clampi(a::Int, lo::Int, hi::Int) = a < lo ? lo : (a > hi ? hi : a)

# ---------------------------------------------------------------------------
# Outflow halo refill: every halo cell copies the nearest INTERIOR cell (clamp).
# Interior cells map to themselves and are never written, so halo threads only
# ever read interior cells no thread mutates — race-free.  Equivalent to
# `apply_physical_bc_3d!(:copy)` in x/y and the per-line z-outflow pad on the CPU.
# ---------------------------------------------------------------------------
function _refill_halo!(G, nf::Int, g::Int, n::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nf * nf * nf
        @inbounds begin
            a = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            b = r % nf + 1;         c = r ÷ nf + 1
            ca = _clampi(a, g + 1, g + n)
            cb = _clampi(b, g + 1, g + n)
            cc = _clampi(c, g + 1, g + n)
            if a != ca || b != cb || c != cc
                for m in 1:35; G[m, a, b, c] = G[m, ca, cb, cc]; end
            end
        end
    end
    return nothing
end

# copy the interior of the haloed cube into a compact (35,n,n,n) buffer (RK M0)
function _copy_interior!(G0, G, n::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
            for m in 1:35; G0[m, i, j, k] = G[m, g + i, g + j, g + k]; end
        end
    end
    return nothing
end

# write a compact interior (35,n,n,n) into the interior region of a haloed cube.
function _set_interior!(G, Mi, n::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
            for m in 1:35; G[m, g + i, g + j, g + k] = Mi[m, i, j, k]; end
        end
    end
    return nothing
end

# RK combine on the interior:  Gint = a*G0 + b*Gint + cdt*R  (cdt already = c*dt).
function _rk_combine!(G, G0, R, n::Int, g::Int, a::Float64, b::Float64, cdt::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
            ga = g + i; gb = g + j; gc = g + k
            for m in 1:35
                G[m, ga, gb, gc] = a * G0[m, i, j, k] + b * G[m, ga, gb, gc] + cdt * R[m, i, j, k]
            end
        end
    end
    return nothing
end

# per-cell realizability projection on the interior (CPU `_project_interior!`).
function _proj_interior!(G, n::Int, g::Int, Ma::Float64, s3max::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
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
function _bgk_interior!(G, n::Int, g::Int, dt::Float64, kn::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
            ga = g + i; gb = g + j; gc = g + k
            C = ntuple(m -> G[m, ga, gb, gc], Val(35))
            out = bgk_relax_tup(C, dt, kn)
            for m in 1:35; G[m, ga, gb, gc] = out[m]; end
        end
    end
    return nothing
end

# per-interior-cell CFL speed (mirrors the order-2 `_speed_box_kernel!`).
function _speed_interior!(svec, G, n::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * n
        @inbounds begin
            i = (idx - 1) % n + 1; r0 = (idx - 1) ÷ n
            j = r0 % n + 1;        k = r0 ÷ n + 1
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
    @assert size(Mi, 1) == 35 "interior must be (35,n,n,n)"
    n = size(Mi, 2)
    @assert size(Mi) == (35, n, n, n) "interior must be a cube (35,n,n,n); got $(size(Mi))"
    g = HALO3; nf = n + 2g
    G = CUDA.zeros(Float64, 35, nf, nf, nf)
    @cuda threads=threads blocks=cld(n * n * n, threads) _set_interior!(G, Mi, n, g)
    @cuda threads=threads blocks=cld(nf * nf * nf, threads) _refill_halo!(G, nf, g, n)
    CUDA.synchronize()
    return G
end

"""
    interior_from_cube!(Mi::CuArray{Float64,4}, G::CuArray{Float64,4}; threads=128) -> Mi

Copy the interior `(35, n, n, n)` out of a `g=HALO3` haloed cube `G` into `Mi`, in place
(the inverse of `build_haloed_cube`). Reuses the same `_copy_interior!` kernel the march
uses for its RK `M0` snapshot.
"""
function interior_from_cube!(Mi::CuArray{Float64,4}, G::CuArray{Float64,4}; threads::Int=128)
    n = size(Mi, 2); g = HALO3
    @assert size(Mi) == (35, n, n, n) "interior must be a cube (35,n,n,n); got $(size(Mi))"
    @assert size(G) == (35, n + 2g, n + 2g, n + 2g) "G must be the matching haloed cube"
    @cuda threads=threads blocks=cld(n * n * n, threads) _copy_interior!(Mi, G, n, g)
    return Mi
end

"""
    march3d_order3_gpu!(G, dx, Ma, nstep; dts=nothing, s3max=…, stage_bgk=false,
                        Kn=Inf, threads=128) -> Vector{Float64}

Advance the fully-haloed order-3 cube `G (35, n+2g, n+2g, n+2g)` (g=4) for `nstep`
SSP-RK3 steps.  Outflow BC on all six faces.  If `dts` is given those dt are used
verbatim; else local CFL each step.  Returns the dt vector used.  `G` is updated
in place and left with its outflow halos refilled.
"""
function march3d_order3_gpu!(G::CuArray{Float64,4}, dx::Real, Ma::Real, nstep::Integer;
                             dts=nothing, s3max::Real = max(40.0, 4.0 + abs(Ma)/2.0),
                             stage_bgk::Bool = false, Kn::Real = Inf, threads::Int = 128)
    @assert size(G, 1) == 35 "G must be (35,nf,nf,nf)"
    nf = size(G, 2)
    @assert size(G) == (35, nf, nf, nf) "G must be a cube (35,nf,nf,nf)"
    g = HALO3
    n = nf - 2g
    @assert n >= 1 "interior extent n = nf-2g must be ≥ 1 (got n=$n)"

    dxf = Float64(dx); Maf = Float64(Ma); s3f = Float64(s3max); knf = Float64(Kn)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))

    R    = CUDA.zeros(Float64, 35, n, n, n)
    G0   = CUDA.zeros(Float64, 35, n, n, n)
    svec = CUDA.zeros(Float64, n * n * n)

    bcube = cld(nf * nf * nf, threads)
    bint  = cld(n * n * n, threads)

    # (a, b, c) RK3 stage weights: Gint = a*G0 + b*Gint + (c*dt)*R
    stages = ((1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1.0/3.0, 2.0/3.0, 2.0/3.0))

    used = Vector{Float64}(undef, nstep)
    for s in 1:nstep
        if dts_host === nothing
            @cuda threads=threads blocks=bint _speed_interior!(svec, G, n, g)
            vmax = CUDA.@allowscalar maximum(svec)
            dt = (1.0/3.0) * dxf / max(vmax, 1e-12)
        else
            dt = dts_host[s]
        end
        used[s] = dt

        @cuda threads=threads blocks=bint _copy_interior!(G0, G, n, g)
        for (a, b, c) in stages
            @cuda threads=threads blocks=bcube _refill_halo!(G, nf, g, n)
            residual3d_order3_box_gpu!(R, G, n, n, n, g, dxf, dxf, dxf, Maf, dt;
                                       s3max=s3f, threads=threads)
            @cuda threads=threads blocks=bint _rk_combine!(G, G0, R, n, g, a, b, c * dt)
            @cuda threads=threads blocks=bint _proj_interior!(G, n, g, Maf, s3f)
            if stage_bgk
                @cuda threads=threads blocks=bint _bgk_interior!(G, n, g, dt, knf)
            end
        end
    end
    @cuda threads=threads blocks=bcube _refill_halo!(G, nf, g, n)
    CUDA.synchronize()
    return used
end

end # module

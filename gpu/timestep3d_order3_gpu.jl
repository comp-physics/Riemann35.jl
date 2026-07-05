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

# copy the interior of the haloed cube into a compact (35,n,n,nz) buffer (RK M0).
# nz == n for the single-GPU cube; nz == nz_loc for a z-slab.
function _copy_interior!(G0, G, n::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * nz
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
            for m in 1:35; G0[m, i, j, k] = G[m, g + i, g + j, g + k]; end
        end
    end
    return nothing
end

# write a compact interior (35,n,n,nz) into the interior region of a haloed cube.
function _set_interior!(G, Mi, n::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * nz
        @inbounds begin
            i = (idx - 1) % n + 1; r = (idx - 1) ÷ n
            j = r % n + 1;         k = r ÷ n + 1
            for m in 1:35; G[m, g + i, g + j, g + k] = Mi[m, i, j, k]; end
        end
    end
    return nothing
end

# RK combine on the interior:  Gint = a*G0 + b*Gint + cdt*R  (cdt already = c*dt).
function _rk_combine!(G, G0, R, n::Int, nz::Int, g::Int, a::Float64, b::Float64, cdt::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * nz
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
function _proj_interior!(G, n::Int, nz::Int, g::Int, Ma::Float64, s3max::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * nz
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
function _bgk_interior!(G, n::Int, nz::Int, g::Int, dt::Float64, kn::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * nz
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
function _speed_interior!(svec, G, n::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n * n * nz
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
    @cuda threads=threads blocks=cld(n * n * n, threads) _set_interior!(G, Mi, n, n, g)
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
    @cuda threads=threads blocks=cld(n * n * n, threads) _copy_interior!(Mi, G, n, n, g)
    return Mi
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
            @cuda threads=threads blocks=bint _speed_interior!(svec, G, n, n, g)
            vmax = CUDA.@allowscalar maximum(svec)
            dt = (1.0/3.0) * dxf / max(vmax, 1e-12)
        else
            dt = dts_host[s]
        end
        used[s] = dt

        @cuda threads=threads blocks=bint _copy_interior!(G0, G, n, n, g)
        for (a, b, c) in stages
            @cuda threads=threads blocks=bcube _refill_halo!(G, nf, g, n)
            residual3d_order3_box_gpu!(R, G, n, n, n, g, dxf, dxf, dxf, Maf, dt;
                                       s3max=s3f, threads=threads)
            @cuda threads=threads blocks=bint _rk_combine!(G, G0, R, n, n, g, a, b, c * dt)
            @cuda threads=threads blocks=bint _proj_interior!(G, n, n, g, Maf, s3f)
            if stage_bgk
                @cuda threads=threads blocks=bint _bgk_interior!(G, n, n, g, dt, knf)
            end
        end
    end
    @cuda threads=threads blocks=bcube _refill_halo!(G, nf, g, n)
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
                                  stage_bgk::Bool = false, Kn::Real = Inf, threads::Int = 128)
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

    @cuda threads=threads blocks=bint _set_interior!(G, Mi, n, nzloc, g)

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
            @cuda threads=threads blocks=bint _speed_interior!(svec, G, n, nzloc, g)
            lvmax = CUDA.@allowscalar maximum(svec)
            gmax  = MPI.Allreduce(lvmax, max, comm)
            dt = (1.0/3.0) * dxf / max(gmax, 1e-12)
        else
            dt = dts_host[s]
        end
        used[s] = dt

        @cuda threads=threads blocks=bint _copy_interior!(G0, G, n, nzloc, g)
        for (a, b, c) in stages
            refresh_halos!()
            residual3d_order3_box_gpu!(R, G, n, n, nzloc, g, dxf, dxf, dxf, Maf, dt;
                                       s3max=s3f, threads=threads, rank_bnd=rb)
            @cuda threads=threads blocks=bint _rk_combine!(G, G0, R, n, nzloc, g, a, b, c * dt)
            @cuda threads=threads blocks=bint _proj_interior!(G, n, nzloc, g, Maf, s3f)
            if stage_bgk
                @cuda threads=threads blocks=bint _bgk_interior!(G, n, nzloc, g, dt, knf)
            end
        end
    end

    @cuda threads=threads blocks=bint _copy_interior!(Mi, G, n, nzloc, g)  # sync interior back
    CUDA.synchronize()
    return used
end

end # module

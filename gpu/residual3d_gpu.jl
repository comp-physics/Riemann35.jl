"""
    residual3d_gpu.jl — on-device 3D order-2 (MUSCL) HLL residual.

Composes the validated per-cell / per-line device kernels
(`src/numerics/recon_dev.jl`, `src/numerics/flux_closure_dev.jl`, `gpu/wavespeed_dev.jl`,
`src/realizability/realize_dev.jl`, `gpu/schur4.jl`) into the 3D unsplit residual, the GPU
analogue of the CPU `residual_ho_3d!` (`src/numerics/highorder_3d.jl`) at
`order=2`. The residual is the SUM of the per-line order-2 1D HLL residual along
the three axes:

    R[:,i,j,k] = Lx(M)[:,i,j,k] + Ly(M)[:,i,j,k] + Lz(M)[:,i,j,k]

where each `L_axis` reconstructs MUSCL faces along that axis, applies the
per-face `realizable_3D_M4` projection (`project_faces=true`, as the CPU
`face_flux_1d` always does), forms the HLL flux with the axis-appropriate
physical flux (`flux_closure35_dev` returns Fx|Fy|Fz; axis a uses block a) and
wave speeds (`realize_and_speed_Mr_dev(..., axis, Ma)`), then differences.

DEVICE LAYOUT (documented):
  `M_dev` and `R_dev` are `(35, nx, ny, nz)` CuArrays — 35 moments CONTIGUOUS
  per cell (fastest), then i, j, k. This matches the on-disk `r3d_M.f64`
  flatten (`permutedims(interior,(4,1,2,3))` then `vec`) and the column-per-cell
  `(35,N)` convention of the 1D kernels, so a line along ANY axis reads the
  35-block of each cell as a unit; threads within a warp stream consecutive
  cells along the line.

BOUNDARY (outflow): the interior is the real n^3 field with NO stored halo. Each
axis treats the line of n real cells as bounded by `halo` outflow ghosts (= copy
of the edge cell). This is realized by CLAMPing the cell index to [1,n] when
gathering the 4 MUSCL stencil cells of each face — EXACTLY reproducing the CPU
`residual_ho_3d!` which pads x/y with outflow halos and pads the z column with
`repeat(edge)`. (n+1 faces per line: face f between cells f and f+1, f=0..n,
ghost cells 0 and n+1 clamped to 1 and n.) All n interior cells get a real
residual — nothing is zeroed (the CPU 3D path computes every interior cell from
its ghost-backed fluxes; a standalone 1D "zero the boundary" rule does not apply
to the 3D composition).

Per axis: kernel 1 computes the face flux `Fhat` at every face into a reusable
`(35, n+1, p1, p2)` buffer (each face computed ONCE); kernel 2 differences and
ACCUMULATES `-(Fhat[f] - Fhat[f-1])/ds` into `R`. `R` is zeroed once up front,
then the three axes add into it.

`@fastmath` stays OFF in the wave-speed path (inherited). fp64 throughout. No
tuple-splat `f(x...)` anywhere on the device. Pure addition under `gpu/`; not
wired into production.
"""
module Residual3DGPU

using CUDA

include(joinpath(@__DIR__, "wavespeed_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "flux_closure_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "realizability", "realize_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "riemann_flux_dev.jl"))
using .RiemannFluxDev: riemann_flux_dev, rs_code
using .WavespeedDev: realize_and_speed_Mr_dev
using .FluxClosureDev: flux_closure35_dev, flux_closure35_central_dev

# Flux-closure path — SELECTED BY MULTIPLE DISPATCH on a singleton type.
#   StdClosure()     -> the standardized closure (standardize -> 21 closures -> destandardize).
#   CentralClosure() -> central-direct closure (same result, skips the sigma round-trip; the
#                       variance powers cancel by parity, removing 2 sqrt + ~56 live values
#                       from the per-face critical path). ~7e-14 vs StdClosure; opt-in.
struct StdClosure end
struct CentralClosure end
const FLUX_CLOSURE = StdClosure()
# Opt-in θ-cache for the realizability scaling limiter: precompute each cell's θ once
# per axis (vs twice, inline) — byte-identical, ~halves the limiter's scaling_theta cost.
# Byte-identical to the inline path (validated: 0/9.2M residual values differ, 0 diff over
# a 20-step limiter march), and ~1.34x faster on a limiter-active step — so default ON, like
# the always-on recon-var cache. Runtime Ref: flip to false to recover the exact inline path
# without recompiling (host-side branch; both kernel sets precompile).
const LIMITER_THETA_CACHE = Ref(true)
# 35 explicit args (NO tuple-splat on the device).
@inline _flux35(::StdClosure,
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai) =
    flux_closure35_dev(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai)
@inline _flux35(::CentralClosure,
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai) =
    flux_closure35_central_dev(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai)
using .ReconDev: to_recon_vars_tup, from_recon_vars_tup, recon_vars_ok_tup, minmod,
                 muscl_right_face_tup, muscl_left_face_tup,
                 pressurize_recon_tup, depressurize_recon_tup
using .RealizeDev: realizable_3D_M4_dev, delta2star_mineig_dev, scaling_theta_dev

export residual3d_gpu!, residual3d_gpu

# ---------------------------------------------------------------------------
# Per-face core: given the four MUSCL-stencil cells (35-tuples) C_{f-1}, C_f,
# C_{f+1}, C_{f+2} along an axis, return the order-2 HLL face flux Fhat (35-tuple)
# for the face between C_f and C_{f+1}. Faithful to CPU `face_flux_1d` composed
# with the MUSCL `recon_face_pair` gate (default, vacuum-floored). project=true
# applies `realizable_3D_M4` to both face states first (CPU always does).
# ---------------------------------------------------------------------------
# proj-first-order flag == CPU `realizability_margin(M) < 0`: M000<=0, any directional
# variance (C200/C020/C002, UNfloored) <= 0, non-finite delta2star, or min eigenvalue < 0.
# C is the raw 35-moment cell tuple; V = to_recon_vars_tup(C) (standardized in V[8:35]).
@inline function _proj_flag(C::NTuple{35,Float64}, V::NTuple{35,Float64})
    C[1] <= 0.0 && return true
    iM = 1.0 / C[1]
    c200 = C[3]*iM  - (C[2]*iM)^2
    c020 = C[10]*iM - (C[6]*iM)^2
    c002 = C[20]*iM - (C[16]*iM)^2
    (c200 <= 0.0 || c020 <= 0.0 || c002 <= 0.0) && return true
    m = delta2star_mineig_dev(V[8],V[9],V[10],V[11],V[12],V[13],V[14],V[15],V[16],V[17],
        V[18],V[19],V[20],V[21],V[22],V[23],V[24],V[25],V[26],V[27],V[28],V[29],V[30],
        V[31],V[32],V[33],V[34],V[35])
    return !isfinite(m) || m < 0.0
end

# Recon vars (Vfm1,Vf,Vfp1,Vfp2) are PRECOMPUTED once per cell by `_recon_kernel!` and passed
# in; to_recon_vars is axis-independent and was being recomputed per face per axis (~12x), so
# caching it is a pure compute dedup (byte-identical). Cf/Cfp1 (the raw inner cells) are still
# needed for the vacuum floor, the proj-first-order flag, and the first-order/cell-mean fallback.
@inline function _face_flux_core(Cf::NTuple{35,Float64}, Cfp1::NTuple{35,Float64},
                                 Vfm1::NTuple{35,Float64}, Vf::NTuple{35,Float64},
                                 Vfp1::NTuple{35,Float64}, Vfp2::NTuple{35,Float64},
                                 axis::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool,
                                 order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool,
                                 θLin::Float64=-1.0, θRin::Float64=-1.0)
    ML0 = Cf[1]
    MR0 = Cfp1[1]

    use_recon = false
    Li = Cf       # order-1 / fallback face = cell mean
    Ri = Cfp1
    # order==1: faces ARE the cell means (no MUSCL), matching CPU residual_line order=1.
    # order>=2: MUSCL reconstruction in recon variables, with the recon-validity +
    # vacuum-floor fallback to the cell mean (byte-identical to the prior default).
    if lim == 1 && order >= 2
        # ho_realizability_limiter (Zhang--Shu / Fan--Huang--Wu scaling limiter): the
        # shared `scaling_limited_faces_dev` returns faces realizable by construction, so
        # there is NO vacuum-floor / recon-validity fallback and NO proj override here
        # (the limiter takes precedence) — matching CPU `face_states_lim`.
        # θ precomputed once per cell per axis (LIMITER_THETA_CACHE path, byte-identical:
        # a cell's θ is otherwise computed twice — as θR of one face and θL of the next);
        # sentinel <0 (default path) recomputes inline exactly as before.
        θL = θLin >= 0.0 ? θLin : scaling_theta_dev(Vfm1, Vf, Vfp1, prec)   # limiter coeff for cell f
        θR = θRin >= 0.0 ? θRin : scaling_theta_dev(Vf, Vfp1, Vfp2, prec)   # limiter coeff for cell f+1
        Vlp = muscl_right_face_tup(Vfm1, Vf, Vfp1, θL)   # right face of cell f   == Vplus(cell f)
        Vlm = muscl_left_face_tup(Vf, Vfp1, Vfp2, θR)    # left face of cell f+1  == Vminus(cell f+1)
        Li = from_recon_vars_tup(prec ? depressurize_recon_tup(Vlp) : Vlp)
        Ri = from_recon_vars_tup(prec ? depressurize_recon_tup(Vlm) : Vlm)
        use_recon = true
    elseif order >= 2 && !(vacf > 0.0 && (ML0 < vacf || MR0 < vacf))
        Vp = muscl_right_face_tup(Vfm1, Vf, Vfp1, 1.0)  # MUSCL right face of cell f
        Vm = muscl_left_face_tup(Vf, Vfp1, Vfp2, 1.0)   # MUSCL left face of cell f+1
        # ho_proj_first_order (Rodney): a cell whose mean is flagged for the realizability
        # projection (smallest delta2star eigenvalue < 0) reconstructs FIRST-ORDER (face =
        # cell mean in recon vars). Same realizability signal the projection uses.
        if proj
            if _proj_flag(Cf, Vf);     Vp = Vf;   end   # flagged cell f   -> first-order right face
            if _proj_flag(Cfp1, Vfp1); Vm = Vfp1; end   # flagged cell f+1 -> first-order left face
        end
        if recon_vars_ok_tup(Vp) && recon_vars_ok_tup(Vm)
            Lc = from_recon_vars_tup(prec ? depressurize_recon_tup(Vp) : Vp)
            Rc = from_recon_vars_tup(prec ? depressurize_recon_tup(Vm) : Vm)
            finL = true; finR = true
            for k in 1:35
                finL &= isfinite(Lc[k]); finR &= isfinite(Rc[k])
            end
            if Lc[1] > 0.0 && Rc[1] > 0.0 && finL && finR
                use_recon = true; Li = Lc; Ri = Rc
            end
        end
    end

    MLf = use_recon ? Li : Cf
    MRf = use_recon ? Ri : Cfp1

    # per-face realizability projection (CPU face_flux_1d always projects)
    if project
        MLf = realizable_3D_M4_dev(
            MLf[1],  MLf[2],  MLf[3],  MLf[4],  MLf[5],  MLf[6],  MLf[7],
            MLf[8],  MLf[9],  MLf[10], MLf[11], MLf[12], MLf[13], MLf[14],
            MLf[15], MLf[16], MLf[17], MLf[18], MLf[19], MLf[20], MLf[21],
            MLf[22], MLf[23], MLf[24], MLf[25], MLf[26], MLf[27], MLf[28],
            MLf[29], MLf[30], MLf[31], MLf[32], MLf[33], MLf[34], MLf[35], Ma, s3f)
        MRf = realizable_3D_M4_dev(
            MRf[1],  MRf[2],  MRf[3],  MRf[4],  MRf[5],  MRf[6],  MRf[7],
            MRf[8],  MRf[9],  MRf[10], MRf[11], MRf[12], MRf[13], MRf[14],
            MRf[15], MRf[16], MRf[17], MRf[18], MRf[19], MRf[20], MRf[21],
            MRf[22], MRf[23], MRf[24], MRf[25], MRf[26], MRf[27], MRf[28],
            MRf[29], MRf[30], MRf[31], MRf[32], MRf[33], MRf[34], MRf[35], Ma, s3f)
    end

    # hyperbolicity correction + wave speeds (axis-aware)
    MLr, lminL, lmaxL = realize_and_speed_Mr_dev(
        MLf[1],  MLf[2],  MLf[3],  MLf[4],  MLf[5],  MLf[6],  MLf[7],
        MLf[8],  MLf[9],  MLf[10], MLf[11], MLf[12], MLf[13], MLf[14],
        MLf[15], MLf[16], MLf[17], MLf[18], MLf[19], MLf[20], MLf[21],
        MLf[22], MLf[23], MLf[24], MLf[25], MLf[26], MLf[27], MLf[28],
        MLf[29], MLf[30], MLf[31], MLf[32], MLf[33], MLf[34], MLf[35], axis, Ma)
    MRr, lminR, lmaxR = realize_and_speed_Mr_dev(
        MRf[1],  MRf[2],  MRf[3],  MRf[4],  MRf[5],  MRf[6],  MRf[7],
        MRf[8],  MRf[9],  MRf[10], MRf[11], MRf[12], MRf[13], MRf[14],
        MRf[15], MRf[16], MRf[17], MRf[18], MRf[19], MRf[20], MRf[21],
        MRf[22], MRf[23], MRf[24], MRf[25], MRf[26], MRf[27], MRf[28],
        MRf[29], MRf[30], MRf[31], MRf[32], MRf[33], MRf[34], MRf[35], axis, Ma)

    # physical flux: flux_closure35_dev returns (Fx|Fy|Fz) flattened (105); axis a -> block a
    FLall = _flux35(FLUX_CLOSURE,
        MLr[1],  MLr[2],  MLr[3],  MLr[4],  MLr[5],  MLr[6],  MLr[7],
        MLr[8],  MLr[9],  MLr[10], MLr[11], MLr[12], MLr[13], MLr[14],
        MLr[15], MLr[16], MLr[17], MLr[18], MLr[19], MLr[20], MLr[21],
        MLr[22], MLr[23], MLr[24], MLr[25], MLr[26], MLr[27], MLr[28],
        MLr[29], MLr[30], MLr[31], MLr[32], MLr[33], MLr[34], MLr[35])
    FRall = _flux35(FLUX_CLOSURE,
        MRr[1],  MRr[2],  MRr[3],  MRr[4],  MRr[5],  MRr[6],  MRr[7],
        MRr[8],  MRr[9],  MRr[10], MRr[11], MRr[12], MRr[13], MRr[14],
        MRr[15], MRr[16], MRr[17], MRr[18], MRr[19], MRr[20], MRr[21],
        MRr[22], MRr[23], MRr[24], MRr[25], MRr[26], MRr[27], MRr[28],
        MRr[29], MRr[30], MRr[31], MRr[32], MRr[33], MRr[34], MRr[35])

    off = (axis - 1) * 35
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)

    # single-source flux + selector (src/numerics/riemann_flux_dev.jl; shared
    # verbatim with the CPU face_flux_1d — byte-identical op order)
    return riemann_flux_dev(rs, axis, MLr, MRr,
                            ntuple(j -> FLall[off + j], Val(35)),
                            ntuple(j -> FRall[off + j], Val(35)), sL, sR)
end

@inline _cell(M, i::Int, j::Int, k::Int) =
    ntuple(m -> @inbounds(M[m, i, j, k]), Val(35))

@inline _clamp(a::Int, n::Int) = a < 1 ? 1 : (a > n ? n : a)

# Precompute the reconstruction variables ONCE per cell into Vbuf (35,nx,ny,nz). to_recon_vars
# is axis-independent, so this replaces the per-face/per-axis recompute in the fhat kernels.
function _recon_kernel!(Vbuf, M, nx::Int, ny::Int, nz::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            V = to_recon_vars_tup(_cell(M, i, j, k))
            if prec
                # ho_pressure_recon: slots 5-7 carry P_ii = rho*C2ii (single-source
                # transform shared with the CPU wrappers; see recon_dev.jl)
                V = pressurize_recon_tup(V)
            end
            for m in 1:35; Vbuf[m, i, j, k] = V[m]; end
        end
    end
    return nothing
end

"""
    residual3d_gpu!(R, Fbuf, M, n, dx, Ma; vacuum_floor=0.001, project_faces=true, threads=128)

In-place 3D order-2 HLL residual. `M::CuArray{Float64,4}` (35,n,n,n) interior
field (no stored halo). `R::CuArray{Float64,4}` (35,n,n,n) is OVERWRITTEN
(zeroed then summed over the 3 axes). `Fbuf::CuArray{Float64,4}` is a
(35, n+1, n, n) scratch reused for each axis. `vacuum_floor` = HO_VACUUM_FLOOR.
Cubic grid (dx=dy=dz). project_faces=true matches the CPU 3D path.
"""
function residual3d_gpu!(R::CuArray{Float64,4}, Fbuf::CuArray{Float64,4},
                         M::CuArray{Float64,4}, n::Int, dx::Real, Ma::Real;
                         vacuum_floor::Real=0.001, project_faces::Bool=true, order::Int=2, proj_first_order::Bool=false, riemann_solver::Symbol=:hll, limiter::Bool=false,
                         threads::Int=128)
    @assert size(Fbuf) == (35, n + 1, n, n) "Fbuf must be (35,n+1,n,n)"
    # The cubic residual is exactly the nx==ny==nz case of `residual3d_box_gpu!`
    # (same kernels). Reuse the caller's Fbuf as the box face-scratch — its element
    # count (n+1)*n*n equals the box `fmax` for a cube — so this stays alloc-free.
    flat = reshape(Fbuf, 35, (n + 1) * n * n)
    residual3d_box_gpu!(R, M, n, n, n, dx, Ma;
                        vacuum_floor=vacuum_floor, project_faces=project_faces, order=order, proj_first_order=proj_first_order, riemann_solver=riemann_solver, limiter=limiter,
                        threads=threads, flat=flat)
    return nothing
end

"""
    residual3d_gpu(M_host, n, dx, Ma; vacuum_floor=0.001, project_faces=true, threads=128)
        -> Array{Float64,4}

Host convenience: upload (35,n,n,n), compute the 3D residual, return (35,n,n,n).
"""
function residual3d_gpu(M_host::Array{Float64,4}, n::Int, dx::Real, Ma::Real;
                        vacuum_floor::Real=0.001, project_faces::Bool=true, order::Int=2, proj_first_order::Bool=false, riemann_solver::Symbol=:hll, limiter::Bool=false,
                        threads::Int=128)
    @assert size(M_host) == (35, n, n, n) "M_host must be (35,n,n,n)"
    Md   = CuArray(M_host)
    R    = CUDA.zeros(Float64, 35, n, n, n)
    Fbuf = CUDA.zeros(Float64, 35, n + 1, n, n)
    residual3d_gpu!(R, Fbuf, Md, n, dx, Ma;
                    vacuum_floor=vacuum_floor, project_faces=project_faces, order=order, proj_first_order=proj_first_order, riemann_solver=riemann_solver, limiter=limiter, threads=threads)
    CUDA.synchronize()
    return Array(R)
end

# ===========================================================================
# RECTANGULAR-BOX generalization (nx,ny,nz independent), outflow BC on all 6
# faces. This is a STRICT generalization of the cubic kernels above: with
# nx==ny==nz==n the indexing and arithmetic are identical, so results are
# bit-for-bit the same. Used for multi-GPU z-slab decomposition: each rank runs
# this on its EXTENDED slab (35, nx, ny, nz_loc+2*halo) whose halo z-planes are
# ghosts (neighbor data, or outflow copies at the global z-boundary); the
# interior nz_loc planes' residuals are then bit-identical to the single-GPU
# full-domain result because every interior cell sees its real +/-2 neighbors.
# Same `_face_flux_core` / `_cell` / `_clamp` as the cubic path -> bit parity.
# ===========================================================================
function _fhat_x_g!(Fbuf, M, Vbuf, nx::Int, ny::Int, nz::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool, order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nx + 1
    if idx <= nf * ny * nz
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            j = r % ny + 1;         k = r ÷ ny + 1
            f = t - 1
            c0 = _cell(M, _clamp(f, nx), j, k); cp1 = _cell(M, _clamp(f + 1, nx), j, k)
            vm1 = _cell(Vbuf, _clamp(f - 1, nx), j, k); v0 = _cell(Vbuf, _clamp(f, nx), j, k)
            vp1 = _cell(Vbuf, _clamp(f + 1, nx), j, k); vp2 = _cell(Vbuf, _clamp(f + 2, nx), j, k)
            Fh = _face_flux_core(c0, cp1, vm1, v0, vp1, vp2, 1, Ma, s3f, vacf, project, order, proj, rs, lim, prec)
            for m in 1:35; Fbuf[m, t, j, k] = Fh[m]; end
        end
    end
    return nothing
end

function _fhat_y_g!(Fbuf, M, Vbuf, nx::Int, ny::Int, nz::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool, order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = ny + 1
    if idx <= nf * nx * nz
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         k = r ÷ nx + 1
            f = t - 1
            c0 = _cell(M, i, _clamp(f, ny), k); cp1 = _cell(M, i, _clamp(f + 1, ny), k)
            vm1 = _cell(Vbuf, i, _clamp(f - 1, ny), k); v0 = _cell(Vbuf, i, _clamp(f, ny), k)
            vp1 = _cell(Vbuf, i, _clamp(f + 1, ny), k); vp2 = _cell(Vbuf, i, _clamp(f + 2, ny), k)
            Fh = _face_flux_core(c0, cp1, vm1, v0, vp1, vp2, 2, Ma, s3f, vacf, project, order, proj, rs, lim, prec)
            for m in 1:35; Fbuf[m, t, i, k] = Fh[m]; end
        end
    end
    return nothing
end

function _fhat_z_g!(Fbuf, M, Vbuf, nx::Int, ny::Int, nz::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool, order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nz + 1
    if idx <= nf * nx * ny
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         j = r ÷ nx + 1
            f = t - 1
            c0 = _cell(M, i, j, _clamp(f, nz)); cp1 = _cell(M, i, j, _clamp(f + 1, nz))
            vm1 = _cell(Vbuf, i, j, _clamp(f - 1, nz)); v0 = _cell(Vbuf, i, j, _clamp(f, nz))
            vp1 = _cell(Vbuf, i, j, _clamp(f + 1, nz)); vp2 = _cell(Vbuf, i, j, _clamp(f + 2, nz))
            Fh = _face_flux_core(c0, cp1, vm1, v0, vp1, vp2, 3, Ma, s3f, vacf, project, order, proj, rs, lim, prec)
            for m in 1:35; Fbuf[m, t, i, j] = Fh[m]; end
        end
    end
    return nothing
end

# ===========================================================================
# LIMITER_THETA_CACHE (opt-in): the scaling-limiter coefficient θ for a cell is
# axis-dependent but, in the inline path, computed TWICE per cell (as θR of the
# face to its left and θL of the face to its right). Precompute it ONCE per cell
# per axis here (one thread/cell, three axes), then the _fhat_*_g_tc! kernels read
# θL=Tb[axis,cell_f], θR=Tb[axis,cell_{f+1}]. Byte-identical: interior cells use the
# same stencil the inline path uses; boundary faces have zero MUSCL slope (neighbor
# clamps to self), so their θ never affects the face value. Tb is (3,nx,ny,nz):
# row 1=x, 2=y, 3=z. Only launched when lim==1 && order>=2.
# ===========================================================================
function _theta_kernel!(Tb, V, nx::Int, ny::Int, nz::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            v0  = _cell(V, i, j, k)
            vxm = _cell(V, _clamp(i - 1, nx), j, k); vxp = _cell(V, _clamp(i + 1, nx), j, k)
            vym = _cell(V, i, _clamp(j - 1, ny), k); vyp = _cell(V, i, _clamp(j + 1, ny), k)
            vzm = _cell(V, i, j, _clamp(k - 1, nz)); vzp = _cell(V, i, j, _clamp(k + 1, nz))
            Tb[1, i, j, k] = scaling_theta_dev(vxm, v0, vxp, prec)
            Tb[2, i, j, k] = scaling_theta_dev(vym, v0, vyp, prec)
            Tb[3, i, j, k] = scaling_theta_dev(vzm, v0, vzp, prec)
        end
    end
    return nothing
end

function _fhat_x_g_tc!(Fbuf, M, Vbuf, Tb, nx::Int, ny::Int, nz::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool, order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nx + 1
    if idx <= nf * ny * nz
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            j = r % ny + 1;         k = r ÷ ny + 1
            f = t - 1
            c0 = _cell(M, _clamp(f, nx), j, k); cp1 = _cell(M, _clamp(f + 1, nx), j, k)
            vm1 = _cell(Vbuf, _clamp(f - 1, nx), j, k); v0 = _cell(Vbuf, _clamp(f, nx), j, k)
            vp1 = _cell(Vbuf, _clamp(f + 1, nx), j, k); vp2 = _cell(Vbuf, _clamp(f + 2, nx), j, k)
            θL = Tb[1, _clamp(f, nx), j, k]; θR = Tb[1, _clamp(f + 1, nx), j, k]
            Fh = _face_flux_core(c0, cp1, vm1, v0, vp1, vp2, 1, Ma, s3f, vacf, project, order, proj, rs, lim, prec, θL, θR)
            for m in 1:35; Fbuf[m, t, j, k] = Fh[m]; end
        end
    end
    return nothing
end

function _fhat_y_g_tc!(Fbuf, M, Vbuf, Tb, nx::Int, ny::Int, nz::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool, order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = ny + 1
    if idx <= nf * nx * nz
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         k = r ÷ nx + 1
            f = t - 1
            c0 = _cell(M, i, _clamp(f, ny), k); cp1 = _cell(M, i, _clamp(f + 1, ny), k)
            vm1 = _cell(Vbuf, i, _clamp(f - 1, ny), k); v0 = _cell(Vbuf, i, _clamp(f, ny), k)
            vp1 = _cell(Vbuf, i, _clamp(f + 1, ny), k); vp2 = _cell(Vbuf, i, _clamp(f + 2, ny), k)
            θL = Tb[2, i, _clamp(f, ny), k]; θR = Tb[2, i, _clamp(f + 1, ny), k]
            Fh = _face_flux_core(c0, cp1, vm1, v0, vp1, vp2, 2, Ma, s3f, vacf, project, order, proj, rs, lim, prec, θL, θR)
            for m in 1:35; Fbuf[m, t, i, k] = Fh[m]; end
        end
    end
    return nothing
end

function _fhat_z_g_tc!(Fbuf, M, Vbuf, Tb, nx::Int, ny::Int, nz::Int, Ma::Float64, s3f::Float64, vacf::Float64, project::Bool, order::Int, proj::Bool, rs::Int, lim::Int, prec::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nz + 1
    if idx <= nf * nx * ny
        @inbounds begin
            t = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         j = r ÷ nx + 1
            f = t - 1
            c0 = _cell(M, i, j, _clamp(f, nz)); cp1 = _cell(M, i, j, _clamp(f + 1, nz))
            vm1 = _cell(Vbuf, i, j, _clamp(f - 1, nz)); v0 = _cell(Vbuf, i, j, _clamp(f, nz))
            vp1 = _cell(Vbuf, i, j, _clamp(f + 1, nz)); vp2 = _cell(Vbuf, i, j, _clamp(f + 2, nz))
            θL = Tb[3, i, j, _clamp(f, nz)]; θR = Tb[3, i, j, _clamp(f + 1, nz)]
            Fh = _face_flux_core(c0, cp1, vm1, v0, vp1, vp2, 3, Ma, s3f, vacf, project, order, proj, rs, lim, prec, θL, θR)
            for m in 1:35; Fbuf[m, t, i, j] = Fh[m]; end
        end
    end
    return nothing
end

function _diff_x_g!(R, Fbuf, nx::Int, ny::Int, nz::Int, ds::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            for m in 1:35; R[m, i, j, k] += -(Fbuf[m, i + 1, j, k] - Fbuf[m, i, j, k]) / ds; end
        end
    end
    return nothing
end

function _diff_y_g!(R, Fbuf, nx::Int, ny::Int, nz::Int, ds::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        # j (the diff/face axis) is the fastest thread index so adjacent threads read the
        # y face-buffer By=(35,ny+1,nx,nz) contiguously (its 2nd dim is the y-face index) —
        # mirrors _diff_x, where the face axis i is already R's fast axis. Byte-identical
        # (each cell's residual is written once into a distinct slot; thread order is free).
        @inbounds begin
            j = (idx - 1) % ny + 1; r = (idx - 1) ÷ ny
            i = r % nx + 1;         k = r ÷ nx + 1
            for m in 1:35; R[m, i, j, k] += -(Fbuf[m, j + 1, i, k] - Fbuf[m, j, i, k]) / ds; end
        end
    end
    return nothing
end

function _diff_z_g!(R, Fbuf, nx::Int, ny::Int, nz::Int, ds::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        # k (the diff/face axis) fastest -> adjacent threads read Bz=(35,nz+1,nx,ny)
        # contiguously (2nd dim is the z-face index). Byte-identical thread remap.
        @inbounds begin
            k = (idx - 1) % nz + 1; r = (idx - 1) ÷ nz
            i = r % nx + 1;         j = r ÷ nx + 1
            for m in 1:35; R[m, i, j, k] += -(Fbuf[m, k + 1, i, j] - Fbuf[m, k, i, j]) / ds; end
        end
    end
    return nothing
end

export residual3d_box_gpu!, residual3d_box_gpu

"""
    residual3d_box_gpu!(R, M, nx, ny, nz, dx, Ma; vacuum_floor=0.001, project_faces=true, threads=128)

Rectangular generalization of `residual3d_gpu!` with outflow BC on all 6 faces.
`M`,`R` are `(35,nx,ny,nz)`. Allocates a reused face buffer internally. With
nx==ny==nz this is bit-identical to `residual3d_gpu!`.
"""
function residual3d_box_gpu!(R::CuArray{Float64,4}, M::CuArray{Float64,4},
                             nx::Int, ny::Int, nz::Int, dx::Real, Ma::Real;
                             vacuum_floor::Real=0.001, project_faces::Bool=true, order::Int=2, proj_first_order::Bool=false, riemann_solver::Symbol=:hll, limiter::Bool=false,
                             pressure_recon::Bool=false, s3max::Real=4.0 + abs(Ma) / 2.0,
                             threads::Int=128, flat::Union{Nothing,CuMatrix{Float64}}=nothing,
                             vbuf::Union{Nothing,CuArray{Float64,4}}=nothing,
                             tbuf::Union{Nothing,CuArray{Float64,4}}=nothing)
    @assert size(M) == (35, nx, ny, nz) "M must be (35,nx,ny,nz)"
    @assert size(R) == (35, nx, ny, nz) "R must be (35,nx,ny,nz)"
    Maf = Float64(Ma); dxf = Float64(dx); vacf = Float64(vacuum_floor); s3f = Float64(s3max)
    rs = rs_code(riemann_solver)   # single-source selector (riemann_flux_dev.jl)
    lim = limiter ? 1 : 0
    prec = pressure_recon
    fx = (nx + 1) * ny * nz; fy = (ny + 1) * nx * nz; fz = (nz + 1) * nx * ny
    fmax = max(fx, fy, fz)
    # reusable face buffer: caller may supply a (35, >=fmax) scratch to avoid per-call alloc
    if flat === nothing
        flat = CUDA.zeros(Float64, 35, fmax)
    else
        @assert size(flat, 1) == 35 && size(flat, 2) >= fmax "flat must be (35, >=fmax)"
    end
    Bx = reshape(view(flat, :, 1:fx), 35, nx + 1, ny, nz)
    By = reshape(view(flat, :, 1:fy), 35, ny + 1, nx, nz)
    Bz = reshape(view(flat, :, 1:fz), 35, nz + 1, nx, ny)
    bc = cld(nx * ny * nz, threads)
    # Recon-var cache (35,nx,ny,nz): to_recon_vars is axis-independent, so precompute it ONCE per
    # cell here (one thread/cell) instead of ~12x across the per-face/per-axis fhat kernels. The
    # fhat kernels then read cached Vbuf for the 4 MUSCL stencil cells. Byte-identical; only used
    # when order>=2 (order-1 faces are cell means, no recon). Caller may supply a (35,nx,ny,nz)
    # scratch to avoid per-call alloc.
    if order >= 2
        if vbuf === nothing
            vbuf = CUDA.zeros(Float64, 35, nx, ny, nz)
        else
            @assert size(vbuf) == (35, nx, ny, nz) "vbuf must be (35,nx,ny,nz)"
        end
        @cuda threads=threads blocks=bc _recon_kernel!(vbuf, M, nx, ny, nz, prec)
    else
        vbuf = M  # unused by the fhat kernels at order==1; pass a valid CuArray to satisfy types
    end

    # opt-in θ-cache: precompute each cell's limiter θ once per axis (byte-identical)
    use_tc = LIMITER_THETA_CACHE[] && lim == 1 && order >= 2
    if use_tc
        if tbuf === nothing
            tbuf = CUDA.zeros(Float64, 3, nx, ny, nz)
        else
            @assert size(tbuf) == (3, nx, ny, nz) "tbuf must be (3,nx,ny,nz)"
        end
        @cuda threads=threads blocks=bc _theta_kernel!(tbuf, vbuf, nx, ny, nz, prec)
    end

    fill!(R, 0.0)
    if use_tc
        @cuda threads=threads blocks=cld(fx, threads) _fhat_x_g_tc!(Bx, M, vbuf, tbuf, nx, ny, nz, Maf, s3f, vacf, project_faces, order, proj_first_order, rs, lim, prec)
        @cuda threads=threads blocks=bc               _diff_x_g!(R, Bx, nx, ny, nz, dxf)
        @cuda threads=threads blocks=cld(fy, threads) _fhat_y_g_tc!(By, M, vbuf, tbuf, nx, ny, nz, Maf, s3f, vacf, project_faces, order, proj_first_order, rs, lim, prec)
        @cuda threads=threads blocks=bc               _diff_y_g!(R, By, nx, ny, nz, dxf)
        @cuda threads=threads blocks=cld(fz, threads) _fhat_z_g_tc!(Bz, M, vbuf, tbuf, nx, ny, nz, Maf, s3f, vacf, project_faces, order, proj_first_order, rs, lim, prec)
        @cuda threads=threads blocks=bc               _diff_z_g!(R, Bz, nx, ny, nz, dxf)
    else
        @cuda threads=threads blocks=cld(fx, threads) _fhat_x_g!(Bx, M, vbuf, nx, ny, nz, Maf, s3f, vacf, project_faces, order, proj_first_order, rs, lim, prec)
        @cuda threads=threads blocks=bc               _diff_x_g!(R, Bx, nx, ny, nz, dxf)
        @cuda threads=threads blocks=cld(fy, threads) _fhat_y_g!(By, M, vbuf, nx, ny, nz, Maf, s3f, vacf, project_faces, order, proj_first_order, rs, lim, prec)
        @cuda threads=threads blocks=bc               _diff_y_g!(R, By, nx, ny, nz, dxf)
        @cuda threads=threads blocks=cld(fz, threads) _fhat_z_g!(Bz, M, vbuf, nx, ny, nz, Maf, s3f, vacf, project_faces, order, proj_first_order, rs, lim, prec)
        @cuda threads=threads blocks=bc               _diff_z_g!(R, Bz, nx, ny, nz, dxf)
    end
    return nothing
end

"Host convenience: upload `(35,nx,ny,nz)`, compute the box residual, return `(35,nx,ny,nz)`."
function residual3d_box_gpu(M_host::Array{Float64,4}, nx::Int, ny::Int, nz::Int, dx::Real, Ma::Real;
                            vacuum_floor::Real=0.001, project_faces::Bool=true, order::Int=2, proj_first_order::Bool=false, riemann_solver::Symbol=:hll, limiter::Bool=false, threads::Int=128)
    @assert size(M_host) == (35, nx, ny, nz) "M_host must be (35,nx,ny,nz)"
    Md = CuArray(M_host); R = CUDA.zeros(Float64, 35, nx, ny, nz)
    residual3d_box_gpu!(R, Md, nx, ny, nz, dx, Ma;
                        vacuum_floor=vacuum_floor, project_faces=project_faces, order=order, proj_first_order=proj_first_order, riemann_solver=riemann_solver, limiter=limiter, threads=threads)
    CUDA.synchronize()
    return Array(R)
end

end # module

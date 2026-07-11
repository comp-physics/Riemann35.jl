"""
    residual3d_order3_gpu.jl — on-device 3D order-3 (WENO5 + θ*-IDP) HLL residual.

GPU analogue of the CPU `residual_ho_3d_order3!` (`src/numerics/highorder_3d.jl`).
Two passes, mirroring the CPU driver EXACTLY:

  Pass 1 (per axis): reconstruct the WENO5 high-order face flux `F_HO` and the
  first-order cell-mean HLL flux `F_LO` at every interface.  This composes the
  SHARED, single-source device functions VERBATIM (no math is reimplemented):
    recon_point_dev  (smooth5-gated deconv5 → recon-var POINT value)
    recon_avg_dev    (conv5 → recon-var cell AVERAGE)
    weno_faces_dev   (WENO5-Z L/R faces + continuous realizability scaling)
  then the HLL face flux (`_hll_states`, the exact tail of the order-2
  `_face_flux_core` with rs=0), matching CPU `_face_flux_tup`.

  Pass 2 (per cell): first-order anchor `Mlo`, six per-face θ via
  `theta_star_update_dev`, interface θ = min over the two adjacent cells, blend
  `F = F_LO + θ(F_HO − F_LO)`, residual `R = −Σ_axis (F_right − F_left)/ds`.

HALO / LAYOUT.  Unlike the order-1/2 GPU path (no stored halo, index clamp for
outflow), the order-3 stencil is ±4 wide and the CPU driver's boundary handling
(recon_point cell-average fallback within 2 of the array end, then Vavg / face
index clamps) is not reproducible by a bare index clamp.  So this path operates
on a FULLY-HALOED cube `G` (35, nfx, nfy, nfz) with nfx = nx+2g etc., built
host-side with the SAME ghost values the CPU driver sees (x/y from the stored
halo, z from outflow edge copies).  Interface `f` along an axis sits between cube
cells `il = g+f-1` and `il+1`, exactly as `residual_line3` indexes with halo g.
The interior residual `R` (35, nx, ny, nz) is then bit-comparable to the CPU
interior `R[g+1:g+nx, g+1:g+ny, 1:nz, :]`.

fp64 throughout.  No tuple-splat `f(x...)` on the device.  Pure addition under
`gpu/`; not wired into production; the order-1/2 paths are untouched.
"""
module Residual3DOrder3GPU

using CUDA

include(joinpath(@__DIR__, "wavespeed_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "flux_closure_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "realizability", "realize_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "riemann_flux_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "idp_limiter_dev.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics", "hiorder3_recon_dev.jl"))

using .RiemannFluxDev: riemann_flux_dev
using .WavespeedDev: realize_and_speed_Mr_dev
using .FluxClosureDev: flux_closure35_dev
using .RealizeDev: realizable_3D_M4_dev
using .ReconDev: to_recon_vars_tup
using .IdpLimiterDev: theta_star_update_dev, theta_star_update_closed
using .HiOrder3ReconDev: recon_point_dev, recon_avg_dev, weno_faces_dev

# --- kinetic-FVS anchor (F3): SINGLE-SOURCE flux + full-cone θ*, shared verbatim with
# the CPU order-3 march (src/numerics/highorder_3d.jl). The bare `kfvs_anchor_core.jl`
# is the one home for kfvs_face_flux_dev / theta_star_fullcone_bisect / marginal_regularized_dev.
include(joinpath(@__DIR__, "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev: chyqmom_nodes_3d_dev, chyqmom_nodes_3d_store_dev!
include(joinpath(@__DIR__, "kfvs_anchor_core.jl"))
include(joinpath(@__DIR__, "kfvs_blend_dev.jl"))
using .KFVSBlendDev: state_realizable_fullcone_dev

# Runtime θ* dispatch (single compiled kernel holds BOTH paths). `use_closed`
# is a plain Bool threaded from the host through the whole call chain (NOT a
# precompile-time const — that pattern freezes at precompile and silently
# no-ops on the package path). Default OFF ⇒ bisection ⇒ byte-identical.
@inline _theta_star(use_closed::Bool, Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64}) =
    use_closed ? theta_star_update_closed(Mlo, dM) : theta_star_update_dev(Mlo, dM)

export residual3d_order3_box_gpu!, residual3d_order3_gpu

@inline _cellG(M, i::Int, j::Int, k::Int) =
    ntuple(m -> @inbounds(M[m, i, j, k]), Val(35))

@inline _clamp(a::Int, n::Int) = a < 1 ? 1 : (a > n ? n : a)

# ---------------------------------------------------------------------------
# HLL flux from two explicit face states — the EXACT tail of the order-2
# `_face_flux_core` (residual3d_gpu.jl) with rs=0 (HLL) and project=true, which
# is byte-identical to the CPU order-3 `_face_flux_tup` (highorder_3d.jl):
#   realizable_3D_M4_dev → realize_and_speed_Mr_dev → flux_closure35_dev (StdClosure)
#   → riemann_flux_dev(0, axis, …).  No math is reimplemented here.
# ---------------------------------------------------------------------------
@inline function _hll_states(mL::NTuple{35,Float64}, mR::NTuple{35,Float64},
                             axis::Int, Ma::Float64, s3f::Float64)
    MLf = realizable_3D_M4_dev(
        mL[1],  mL[2],  mL[3],  mL[4],  mL[5],  mL[6],  mL[7],
        mL[8],  mL[9],  mL[10], mL[11], mL[12], mL[13], mL[14],
        mL[15], mL[16], mL[17], mL[18], mL[19], mL[20], mL[21],
        mL[22], mL[23], mL[24], mL[25], mL[26], mL[27], mL[28],
        mL[29], mL[30], mL[31], mL[32], mL[33], mL[34], mL[35], Ma, s3f)
    MRf = realizable_3D_M4_dev(
        mR[1],  mR[2],  mR[3],  mR[4],  mR[5],  mR[6],  mR[7],
        mR[8],  mR[9],  mR[10], mR[11], mR[12], mR[13], mR[14],
        mR[15], mR[16], mR[17], mR[18], mR[19], mR[20], mR[21],
        mR[22], mR[23], mR[24], mR[25], mR[26], mR[27], mR[28],
        mR[29], mR[30], mR[31], mR[32], mR[33], mR[34], mR[35], Ma, s3f)

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

    FLall = flux_closure35_dev(
        MLr[1],  MLr[2],  MLr[3],  MLr[4],  MLr[5],  MLr[6],  MLr[7],
        MLr[8],  MLr[9],  MLr[10], MLr[11], MLr[12], MLr[13], MLr[14],
        MLr[15], MLr[16], MLr[17], MLr[18], MLr[19], MLr[20], MLr[21],
        MLr[22], MLr[23], MLr[24], MLr[25], MLr[26], MLr[27], MLr[28],
        MLr[29], MLr[30], MLr[31], MLr[32], MLr[33], MLr[34], MLr[35])
    FRall = flux_closure35_dev(
        MRr[1],  MRr[2],  MRr[3],  MRr[4],  MRr[5],  MRr[6],  MRr[7],
        MRr[8],  MRr[9],  MRr[10], MRr[11], MRr[12], MRr[13], MRr[14],
        MRr[15], MRr[16], MRr[17], MRr[18], MRr[19], MRr[20], MRr[21],
        MRr[22], MRr[23], MRr[24], MRr[25], MRr[26], MRr[27], MRr[28],
        MRr[29], MRr[30], MRr[31], MRr[32], MRr[33], MRr[34], MRr[35])

    off = (axis - 1) * 35
    sL = min(lminL, lminR)
    sR = max(lmaxL, lmaxR)
    return riemann_flux_dev(0, axis, MLr, MRr,
                            ntuple(j -> FLall[off + j], Val(35)),
                            ntuple(j -> FRall[off + j], Val(35)), sL, sR)
end

# ===========================================================================
# PASS 1 — per axis: Ppt (recon-var point), Vavg (recon-var average), faces.
# ===========================================================================
# Step 1: recon-var POINT value from the 5-cell RAW stencil along the axis.
# Boundary fallback (within 2 of the array end along that axis) = cell average,
# matching residual_line3 (`k >= 3 && k <= n2g-2 ? recon_point : to_recon_vars`).
function _ppt_x!(P, G, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            Pv = (a >= 3 && a <= nfx - 2) ?
                recon_point_dev(_cellG(G, a-2, b, c), _cellG(G, a-1, b, c),
                                _cellG(G, a, b, c), _cellG(G, a+1, b, c), _cellG(G, a+2, b, c)) :
                to_recon_vars_tup(_cellG(G, a, b, c))
            for m in 1:35; P[m, a, b, c] = Pv[m]; end
        end
    end
    return nothing
end

function _ppt_y!(P, G, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            Pv = (b >= 3 && b <= nfy - 2) ?
                recon_point_dev(_cellG(G, a, b-2, c), _cellG(G, a, b-1, c),
                                _cellG(G, a, b, c), _cellG(G, a, b+1, c), _cellG(G, a, b+2, c)) :
                to_recon_vars_tup(_cellG(G, a, b, c))
            for m in 1:35; P[m, a, b, c] = Pv[m]; end
        end
    end
    return nothing
end

function _ppt_z!(P, G, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            Pv = (c >= 3 && c <= nfz - 2) ?
                recon_point_dev(_cellG(G, a, b, c-2), _cellG(G, a, b, c-1),
                                _cellG(G, a, b, c), _cellG(G, a, b, c+1), _cellG(G, a, b, c+2)) :
                to_recon_vars_tup(_cellG(G, a, b, c))
            for m in 1:35; P[m, a, b, c] = Pv[m]; end
        end
    end
    return nothing
end

# Step 2: conv5 of the 5-cell recon-var POINT stencil → recon-var cell AVERAGE.
# Stencil index clamped to [1,nf] along the axis (residual_line3 `_pp`).
function _vavg_x!(V, P, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            Vv = recon_avg_dev(_cellG(P, _clamp(a-2, nfx), b, c), _cellG(P, _clamp(a-1, nfx), b, c),
                               _cellG(P, a, b, c),
                               _cellG(P, _clamp(a+1, nfx), b, c), _cellG(P, _clamp(a+2, nfx), b, c))
            for m in 1:35; V[m, a, b, c] = Vv[m]; end
        end
    end
    return nothing
end

function _vavg_y!(V, P, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            Vv = recon_avg_dev(_cellG(P, a, _clamp(b-2, nfy), c), _cellG(P, a, _clamp(b-1, nfy), c),
                               _cellG(P, a, b, c),
                               _cellG(P, a, _clamp(b+1, nfy), c), _cellG(P, a, _clamp(b+2, nfy), c))
            for m in 1:35; V[m, a, b, c] = Vv[m]; end
        end
    end
    return nothing
end

function _vavg_z!(V, P, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nfy * nfz
        @inbounds begin
            a = (idx - 1) % nfx + 1; r = (idx - 1) ÷ nfx
            b = r % nfy + 1;         c = r ÷ nfy + 1
            Vv = recon_avg_dev(_cellG(P, a, b, _clamp(c-2, nfz)), _cellG(P, a, b, _clamp(c-1, nfz)),
                               _cellG(P, a, b, c),
                               _cellG(P, a, b, _clamp(c+1, nfz)), _cellG(P, a, b, _clamp(c+2, nfz)))
            for m in 1:35; V[m, a, b, c] = Vv[m]; end
        end
    end
    return nothing
end

# Step 3: per interface, WENO5 L/R faces + HLL → F_HO and F_LO.
# Interface f (1..nx+1) between cube cells il = g+f-1 and il+1 at interior (j,k).
# --- store-once node pass (fast path): invert every haloed cell ONCE per stage into
# node arrays, so the weno flux reads stored L/R nodes instead of inverting per face
# (which put the 255-reg inversion inline in the hot kernel and crashed occupancy). ---
@inline _lin(i::Int, j::Int, k::Int, nfx::Int, nfy::Int) = i + (j-1)*nfx + (k-1)*nfx*nfy
@inline function _store4_anchor!(NW, UX, UY, UZ, ci::Int, q::Int, w::Float64, ux::Float64, uy::Float64, uz::Float64)
    @inbounds begin; NW[q, ci] = w; UX[q, ci] = ux; UY[q, ci] = uy; UZ[q, ci] = uz; end
    return nothing
end
# store pass: one thread per cell in the ANCHOR REGION only — the interior plus one halo
# layer, cells [g, g+n+1] on each axis, which is exactly the set the weno faces read
# (il ∈ [g, g+n], il+1 ∈ [g+1, g+n+1]). This skips the 8-deep WENO halos, cutting the
# 255-reg inversions from (n+2g)^3 to (n+2)^3. NC stays 0 outside the region (never read).
function _anchor_store!(NW, UX, UY, UZ, NC, G, nfx::Int, nfy::Int, nx::Int, ny::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    mx = nx + 2; my = ny + 2; mz = nz + 2
    if idx <= mx * my * mz
        @inbounds begin
            ci = (idx - 1) % mx; r = (idx - 1) ÷ mx
            cj = r % my;         ck = r ÷ my
            i = g + ci; j = g + cj; k = g + ck          # cells [g, g+n+1]
            cell = _lin(i, j, k, nfx, nfy)
            m = _cellG(G, i, j, k)
            NC[cell] = chyqmom_nodes_3d_store_dev!(_store4_anchor!, NW, UX, UY, UZ, cell, m)
        end
    end
    return nothing
end
# read a cell's stored ≤27 nodes into an NTuple{27} (static-index array reads ⇒ GPU-safe).
@inline _read27(A, ci::Int) = ntuple(q -> @inbounds(A[q, ci]), Val(27))
# F3 anchor interface flux from STORED nodes; native device HLL when either cell is
# degenerate (node count 0). The upwind fold is the single-source kfvs_flux_from_nodes.
@inline function _anchor_face_flux_stored(NW, UX, UY, UZ, ciL::Int, ciR::Int, NCL, NCR,
                                          cL::NTuple{35,Float64}, cR::NTuple{35,Float64},
                                          axis::Int, Ma::Float64, s3f::Float64)
    (NCL >= 1 && NCR >= 1) || return _hll_states(cL, cR, axis, Ma, s3f)
    return kfvs_flux_from_nodes(_read27(NW, ciL), _read27(UX, ciL), _read27(UY, ciL), _read27(UZ, ciL),
                                _read27(NW, ciR), _read27(UX, ciR), _read27(UY, ciR), _read27(UZ, ciR), axis)
end
# F3 full-cone θ*: the shared bisection with the GPU predicate injected (inertia Δ2*
# `state_realizable_fullcone_dev` + the marginal regularization twin).
@inline function _theta_anchor(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64},
                               Ma::Float64, s3f::Float64)
    theta_star_fullcone_bisect(Mlo, dM,
        m -> (state_realizable_fullcone_dev(m) && marginal_regularized_dev(m, Ma, s3f)))
end

function _weno_flux_x!(FHO, FLO, G, V, nx::Int, ny::Int, nz::Int, g::Int, nfx::Int,
                       Ma::Float64, s3f::Float64, use_ka::Bool, NW, UX, UY, UZ, NC)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nx + 1
    if idx <= nf * ny * nz
        @inbounds begin
            f = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            j = r % ny + 1;         k = r ÷ ny + 1
            b = g + j; c = g + k; il = g + f - 1
            cL = _cellG(G, il, b, c); cR = _cellG(G, il + 1, b, c)
            W1 = _cellG(V, _clamp(il-2, nfx), b, c); W2 = _cellG(V, _clamp(il-1, nfx), b, c)
            W3 = _cellG(V, _clamp(il,   nfx), b, c); W4 = _cellG(V, _clamp(il+1, nfx), b, c)
            W5 = _cellG(V, _clamp(il+2, nfx), b, c); W6 = _cellG(V, _clamp(il+3, nfx), b, c)
            mL, mR = weno_faces_dev(W1, W2, W3, W4, W5, W6, cL, cR)
            FH = _hll_states(mL, mR, 1, Ma, s3f)
            ciL = _lin(il, b, c, nfx, nfx); ciR = _lin(il+1, b, c, nfx, nfx)
            FL = use_ka ? _anchor_face_flux_stored(NW, UX, UY, UZ, ciL, ciR, NC[ciL], NC[ciR], cL, cR, 1, Ma, s3f) :
                          _hll_states(cL, cR, 1, Ma, s3f)
            for m in 1:35; FHO[m, f, j, k] = FH[m]; FLO[m, f, j, k] = FL[m]; end
        end
    end
    return nothing
end

function _weno_flux_y!(FHO, FLO, G, V, nx::Int, ny::Int, nz::Int, g::Int, nfy::Int,
                       Ma::Float64, s3f::Float64, use_ka::Bool, NW, UX, UY, UZ, NC)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = ny + 1
    if idx <= nf * nx * nz
        @inbounds begin
            f = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         k = r ÷ nx + 1
            a = g + i; c = g + k; jl = g + f - 1
            cL = _cellG(G, a, jl, c); cR = _cellG(G, a, jl + 1, c)
            W1 = _cellG(V, a, _clamp(jl-2, nfy), c); W2 = _cellG(V, a, _clamp(jl-1, nfy), c)
            W3 = _cellG(V, a, _clamp(jl,   nfy), c); W4 = _cellG(V, a, _clamp(jl+1, nfy), c)
            W5 = _cellG(V, a, _clamp(jl+2, nfy), c); W6 = _cellG(V, a, _clamp(jl+3, nfy), c)
            mL, mR = weno_faces_dev(W1, W2, W3, W4, W5, W6, cL, cR)
            FH = _hll_states(mL, mR, 2, Ma, s3f)
            ciL = _lin(a, jl, c, nfy, nfy); ciR = _lin(a, jl+1, c, nfy, nfy)
            FL = use_ka ? _anchor_face_flux_stored(NW, UX, UY, UZ, ciL, ciR, NC[ciL], NC[ciR], cL, cR, 2, Ma, s3f) :
                          _hll_states(cL, cR, 2, Ma, s3f)
            for m in 1:35; FHO[m, i, f, k] = FH[m]; FLO[m, i, f, k] = FL[m]; end
        end
    end
    return nothing
end

function _weno_flux_z!(FHO, FLO, G, V, nx::Int, ny::Int, nz::Int, g::Int, nfz::Int,
                       Ma::Float64, s3f::Float64, use_ka::Bool, NW, UX, UY, UZ, NC)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nz + 1
    if idx <= nf * nx * ny
        @inbounds begin
            f = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         j = r ÷ nx + 1
            a = g + i; b = g + j; kl = g + f - 1
            cL = _cellG(G, a, b, kl); cR = _cellG(G, a, b, kl + 1)
            W1 = _cellG(V, a, b, _clamp(kl-2, nfz)); W2 = _cellG(V, a, b, _clamp(kl-1, nfz))
            W3 = _cellG(V, a, b, _clamp(kl,   nfz)); W4 = _cellG(V, a, b, _clamp(kl+1, nfz))
            W5 = _cellG(V, a, b, _clamp(kl+2, nfz)); W6 = _cellG(V, a, b, _clamp(kl+3, nfz))
            mL, mR = weno_faces_dev(W1, W2, W3, W4, W5, W6, cL, cR)
            FH = _hll_states(mL, mR, 3, Ma, s3f)
            ciL = _lin(a, b, kl, nfz, nfz); ciR = _lin(a, b, kl+1, nfz, nfz)
            FL = use_ka ? _anchor_face_flux_stored(NW, UX, UY, UZ, ciL, ciR, NC[ciL], NC[ciR], cL, cR, 3, Ma, s3f) :
                          _hll_states(cL, cR, 3, Ma, s3f)
            for m in 1:35; FHO[m, i, j, f] = FH[m]; FLO[m, i, j, f] = FL[m]; end
        end
    end
    return nothing
end

# ===========================================================================
# PASS 2a — per interior cell, the six per-face θ* into Th (6,nx,ny,nz):
#   row 1=x-right, 2=x-left, 3=y-right, 4=y-left, 5=z-right, 6=z-left.
# Mirrors highorder_3d.jl Pass-2a exactly (factor-6 bound, dt=0 short-circuit).
# ===========================================================================
@inline _face3(F, p::Int, q::Int, r::Int) =
    ntuple(m -> @inbounds(F[m, p, q, r]), Val(35))

function _theta_cell!(Th, G, FHOx, FLOx, FHOy, FLOy, FHOz, FLOz,
                      nx::Int, ny::Int, nz::Int, g::Int,
                      λx::Float64, λy::Float64, λz::Float64, use_closed::Bool,
                      use_ka::Bool, Ma::Float64, s3f::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            a = g + i; b = g + j; c = g + k
            Mc = _cellG(G, a, b, c)

            fxr1 = _face3(FLOx, i+1, j, k); fxl0 = _face3(FLOx, i, j, k)
            fyr1 = _face3(FLOy, i, j+1, k); fyl0 = _face3(FLOy, i, j, k)
            fzr1 = _face3(FLOz, i, j, k+1); fzl0 = _face3(FLOz, i, j, k)
            hxr1 = _face3(FHOx, i+1, j, k); hxl0 = _face3(FHOx, i, j, k)
            hyr1 = _face3(FHOy, i, j+1, k); hyl0 = _face3(FHOy, i, j, k)
            hzr1 = _face3(FHOz, i, j, k+1); hzl0 = _face3(FHOz, i, j, k)

            Mlo = (λx == 0.0 && λy == 0.0 && λz == 0.0) ? Mc :
                ntuple(q -> Mc[q] - λx*(fxr1[q]-fxl0[q])
                                  - λy*(fyr1[q]-fyl0[q])
                                  - λz*(fzr1[q]-fzl0[q]), Val(35))

            Gxr = ntuple(q -> hxr1[q] - fxr1[q], Val(35))
            Gxl = ntuple(q -> hxl0[q] - fxl0[q], Val(35))
            Gyr = ntuple(q -> hyr1[q] - fyr1[q], Val(35))
            Gyl = ntuple(q -> hyl0[q] - fyl0[q], Val(35))
            Gzr = ntuple(q -> hzr1[q] - fzr1[q], Val(35))
            Gzl = ntuple(q -> hzl0[q] - fzl0[q], Val(35))

            dx1 = ntuple(q -> -6λx * Gxr[q], Val(35)); dx2 = ntuple(q ->  6λx * Gxl[q], Val(35))
            dy1 = ntuple(q -> -6λy * Gyr[q], Val(35)); dy2 = ntuple(q ->  6λy * Gyl[q], Val(35))
            dz1 = ntuple(q -> -6λz * Gzr[q], Val(35)); dz2 = ntuple(q ->  6λz * Gzl[q], Val(35))
            Th[1, i, j, k] = use_ka ? _theta_anchor(Mlo, dx1, Ma, s3f) : _theta_star(use_closed, Mlo, dx1)
            Th[2, i, j, k] = use_ka ? _theta_anchor(Mlo, dx2, Ma, s3f) : _theta_star(use_closed, Mlo, dx2)
            Th[3, i, j, k] = use_ka ? _theta_anchor(Mlo, dy1, Ma, s3f) : _theta_star(use_closed, Mlo, dy1)
            Th[4, i, j, k] = use_ka ? _theta_anchor(Mlo, dy2, Ma, s3f) : _theta_star(use_closed, Mlo, dy2)
            Th[5, i, j, k] = use_ka ? _theta_anchor(Mlo, dz1, Ma, s3f) : _theta_star(use_closed, Mlo, dz1)
            Th[6, i, j, k] = use_ka ? _theta_anchor(Mlo, dz2, Ma, s3f) : _theta_star(use_closed, Mlo, dz2)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# First-order (HLL) six-face anchor of a HALO cell at cube position (px,py,pk),
# read directly from the haloed cube G. GPU analogue of the CPU `halo_cell_mlo`
# (highorder_3d.jl PASS 2a'): the SAME cheap anchor the neighbour rank evaluates
# for that (there interior) cell, so — with the g=8 halo exchange + outflow clamp
# giving bit-identical adjacent cells — the two ranks agree bit-for-bit.
# dt=0 (all λ zero) short-circuits to the raw cell state, exactly as CPU.
# Axis-generic (all six faces): used at whichever axis carries rank boundaries
# (z-slab on the GPU; the x/y call sites are DORMANT — see `_blend_residual!`).
# ---------------------------------------------------------------------------
@inline function _halo_cell_mlo(G, px::Int, py::Int, pk::Int,
                                 λx::Float64, λy::Float64, λz::Float64,
                                 Ma::Float64, s3f::Float64)
    C = _cellG(G, px, py, pk)
    (λx == 0.0 && λy == 0.0 && λz == 0.0) && return C
    FxL = _hll_states(_cellG(G, px-1, py, pk), C, 1, Ma, s3f)
    FxR = _hll_states(C, _cellG(G, px+1, py, pk), 1, Ma, s3f)
    FyD = _hll_states(_cellG(G, px, py-1, pk), C, 2, Ma, s3f)
    FyU = _hll_states(C, _cellG(G, px, py+1, pk), 2, Ma, s3f)
    FzB = _hll_states(_cellG(G, px, py, pk-1), C, 3, Ma, s3f)
    FzF = _hll_states(C, _cellG(G, px, py, pk+1), 3, Ma, s3f)
    return ntuple(q -> C[q] - λx*(FxR[q]-FxL[q]) - λy*(FyU[q]-FyD[q])
                            - λz*(FzF[q]-FzB[q]), Val(35))
end

# ---------------------------------------------------------------------------
# Rank-boundary interface θ for ONE face (axis-generic). Returns
#   min(Thf, θ_halo),  θ_halo = theta_star_update_dev(Mlo_halo, s6λ·G_shared),
# where Mlo_halo is the halo cell's first-order anchor at cube coords (px,py,pk),
# G_shared = F_HO−F_LO at the shared interface (fi,fj,fk of the FHOa/FLOa face
# array for that axis), and s6λ = ±6λ_axis. The ntuple closure lives HERE (not in
# the kernel) so its captured locals are fresh per call — sidestepping the Core.Box
# hazard when several axis branches would otherwise reuse names before a closure.
# ---------------------------------------------------------------------------
@inline function _rank_face_theta(G, Thf::Float64, px::Int, py::Int, pk::Int,
                                  FHOa, FLOa, fi::Int, fj::Int, fk::Int, s6λ::Float64,
                                  λx::Float64, λy::Float64, λz::Float64,
                                  Ma::Float64, s3f::Float64, use_closed::Bool)
    Mlo = _halo_cell_mlo(G, px, py, pk, λx, λy, λz, Ma, s3f)
    Gsh = _face3(FHOa, fi, fj, fk)
    Gsl = _face3(FLOa, fi, fj, fk)
    θh  = _theta_star(use_closed, Mlo, ntuple(q -> s6λ*(Gsh[q]-Gsl[q]), Val(35)))
    return min(Thf, θh)
end

# ===========================================================================
# PASS 2b — interface θ = min over the two adjacent cells; blend F; residual.
#
# RANK boundaries (axis-generic): at a shared interface facing a neighbour rank the
# θ = min(own interior cell θ, the neighbour HALO cell's θ). The halo cell's θ is
# computed via `_rank_face_theta` from its cheap first-order anchor (`_halo_cell_mlo`,
# read from the haloed cube) and the SHARED interface's G = F_HO−F_LO (identical
# across ranks). Both ranks arrive at the same min ⇒ conservative + rank-consistent
# + bit-identical to the single-GPU march (the wide g=8 halo makes every recon
# footprint real). GPU multi-GPU decomposes z only, so zlo/zhi carry the flags and
# the x/y branches are present but DORMANT (xlo/xhi/ylo/yhi always false here) —
# the code is axis-symmetric, not a per-axis copy. At GLOBAL boundaries (flag false)
# the own-cell θ is kept (the single-GPU cube march passes all six false ⇒ byte-
# identical to before).
# ===========================================================================
function _blend_residual!(R, Th, FHOx, FLOx, FHOy, FLOy, FHOz, FLOz,
                          nx::Int, ny::Int, nz::Int,
                          dx::Float64, dy::Float64, dz::Float64,
                          G, g::Int, λx::Float64, λy::Float64, λz::Float64,
                          Ma::Float64, s3f::Float64,
                          xlo::Bool, xhi::Bool, ylo::Bool, yhi::Bool, zlo::Bool, zhi::Bool,
                          use_closed::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1

            # x-right face: interior neighbour; else (hi rank boundary) halo cell at
            # cube px = g+nx+1, shared interface nx+1; else global-boundary own-θ.
            if i < nx
                θxr = min(Th[1,i,j,k], Th[2,i+1,j,k])
            elseif xhi
                θxr = _rank_face_theta(G, Th[1,i,j,k], g+nx+1, g+j, g+k,
                                       FHOx, FLOx, nx+1, j, k, 6.0*λx, λx, λy, λz, Ma, s3f, use_closed)
            else
                θxr = Th[1,i,j,k]
            end
            # x-left face: halo cell at cube px = g, shared interface 1.
            if i > 1
                θxl = min(Th[2,i,j,k], Th[1,i-1,j,k])
            elseif xlo
                θxl = _rank_face_theta(G, Th[2,i,j,k], g, g+j, g+k,
                                       FHOx, FLOx, 1, j, k, -6.0*λx, λx, λy, λz, Ma, s3f, use_closed)
            else
                θxl = Th[2,i,j,k]
            end
            # y-right face: halo cell at cube py = g+ny+1, shared interface ny+1.
            if j < ny
                θyr = min(Th[3,i,j,k], Th[4,i,j+1,k])
            elseif yhi
                θyr = _rank_face_theta(G, Th[3,i,j,k], g+i, g+ny+1, g+k,
                                       FHOy, FLOy, i, ny+1, k, 6.0*λy, λx, λy, λz, Ma, s3f, use_closed)
            else
                θyr = Th[3,i,j,k]
            end
            # y-left face: halo cell at cube py = g, shared interface 1.
            if j > 1
                θyl = min(Th[4,i,j,k], Th[3,i,j-1,k])
            elseif ylo
                θyl = _rank_face_theta(G, Th[4,i,j,k], g+i, g, g+k,
                                       FHOy, FLOy, i, 1, k, -6.0*λy, λx, λy, λz, Ma, s3f, use_closed)
            else
                θyl = Th[4,i,j,k]
            end
            # z-right face: halo cell just past interior nz at cube pk = g+nz+1.
            if k < nz
                θzr = min(Th[5,i,j,k], Th[6,i,j,k+1])
            elseif zhi
                θzr = _rank_face_theta(G, Th[5,i,j,k], g+i, g+j, g+nz+1,
                                       FHOz, FLOz, i, j, nz+1, 6.0*λz, λx, λy, λz, Ma, s3f, use_closed)
            else
                θzr = Th[5,i,j,k]
            end
            # z-left face: halo cell just before interior 1 at cube pk = g.
            if k > 1
                θzl = min(Th[6,i,j,k], Th[5,i,j,k-1])
            elseif zlo
                θzl = _rank_face_theta(G, Th[6,i,j,k], g+i, g+j, g,
                                       FHOz, FLOz, i, j, 1, -6.0*λz, λx, λy, λz, Ma, s3f, use_closed)
            else
                θzl = Th[6,i,j,k]
            end

            FHxr = _face3(FHOx, i+1, j, k); FLxr = _face3(FLOx, i+1, j, k)
            FHxl = _face3(FHOx, i,   j, k); FLxl = _face3(FLOx, i,   j, k)
            FHyr = _face3(FHOy, i, j+1, k); FLyr = _face3(FLOy, i, j+1, k)
            FHyl = _face3(FHOy, i, j,   k); FLyl = _face3(FLOy, i, j,   k)
            FHzr = _face3(FHOz, i, j, k+1); FLzr = _face3(FLOz, i, j, k+1)
            FHzl = _face3(FHOz, i, j, k  ); FLzl = _face3(FLOz, i, j, k  )

            for m in 1:35
                Fxr = FLxr[m] + θxr * (FHxr[m] - FLxr[m])
                Fxl = FLxl[m] + θxl * (FHxl[m] - FLxl[m])
                Fyr = FLyr[m] + θyr * (FHyr[m] - FLyr[m])
                Fyl = FLyl[m] + θyl * (FHyl[m] - FLyl[m])
                Fzr = FLzr[m] + θzr * (FHzr[m] - FLzr[m])
                Fzl = FLzl[m] + θzl * (FHzl[m] - FLzl[m])
                R[m, i, j, k] = -((Fxr-Fxl)/dx + (Fyr-Fyl)/dy + (Fzr-Fzl)/dz)
            end
        end
    end
    return nothing
end

# ===========================================================================
# Driver.  G is the FULLY-HALOED cube (35, nfx, nfy, nfz), nfx=nx+2g etc.
# R is the interior residual (35, nx, ny, nz).
# ===========================================================================
function residual3d_order3_box_gpu!(R::CuArray{Float64,4}, G::CuArray{Float64,4},
                                    nx::Int, ny::Int, nz::Int, g::Int,
                                    dx::Real, dy::Real, dz::Real, Ma::Real, dt::Real;
                                    s3max::Real = 40.0, threads::Int = 128,
                                    theta_closed::Bool = true, use_kfvs_anchor::Bool = false,
                                    rank_bnd = (xlo=false, xhi=false, ylo=false, yhi=false,
                                                zlo=false, zhi=false))
    nfx = nx + 2g; nfy = ny + 2g; nfz = nz + 2g
    @assert g >= 4 "order-3 residual requires halo g ≥ 4; got g=$g"
    @assert size(G) == (35, nfx, nfy, nfz) "G must be (35,nx+2g,ny+2g,nz+2g)"
    @assert size(R) == (35, nx, ny, nz) "R must be (35,nx,ny,nz)"
    Maf = Float64(Ma); s3f = Float64(s3max)
    dxf = Float64(dx); dyf = Float64(dy); dzf = Float64(dz)
    λx = Float64(dt) / dxf; λy = Float64(dt) / dyf; λz = Float64(dt) / dzf

    P = CUDA.zeros(Float64, 35, nfx, nfy, nfz)   # recon-var point scratch (reused per axis)
    V = CUDA.zeros(Float64, 35, nfx, nfy, nfz)   # recon-var average scratch (reused per axis)
    FHOx = CUDA.zeros(Float64, 35, nx+1, ny, nz); FLOx = CUDA.zeros(Float64, 35, nx+1, ny, nz)
    FHOy = CUDA.zeros(Float64, 35, nx, ny+1, nz); FLOy = CUDA.zeros(Float64, 35, nx, ny+1, nz)
    FHOz = CUDA.zeros(Float64, 35, nx, ny, nz+1); FLOz = CUDA.zeros(Float64, 35, nx, ny, nz+1)
    Th   = CUDA.zeros(Float64, 6, nx, ny, nz)

    ncube = nfx * nfy * nfz
    bcube = cld(ncube, threads)
    fx = (nx+1)*ny*nz; fy = (ny+1)*nx*nz; fz = (nz+1)*nx*ny
    bint = cld(nx*ny*nz, threads)

    # F3 anchor STORE-ONCE pass: invert every haloed cell ONCE into node arrays so the
    # weno flux kernels read stored L/R nodes (the 255-reg inversion leaves the hot
    # kernel ⇒ occupancy recovers). Tiny dummies on the default path (never indexed).
    if use_kfvs_anchor
        NW = CUDA.zeros(Float64, 27, ncube); UX = CUDA.zeros(Float64, 27, ncube)
        UY = CUDA.zeros(Float64, 27, ncube); UZ = CUDA.zeros(Float64, 27, ncube)
        NC = CUDA.zeros(Int32, ncube)
        bstore = cld((nx+2)*(ny+2)*(nz+2), threads)
        @cuda threads=threads blocks=bstore _anchor_store!(NW, UX, UY, UZ, NC, G, nfx, nfy, nx, ny, nz, g)
    else
        NW = CUDA.zeros(Float64, 1, 1); UX = NW; UY = NW; UZ = NW; NC = CUDA.zeros(Int32, 1)
    end

    # --- Pass 1: per axis Ppt → Vavg → faces ---
    @cuda threads=threads blocks=bcube _ppt_x!(P, G, nfx, nfy, nfz)
    @cuda threads=threads blocks=bcube _vavg_x!(V, P, nfx, nfy, nfz)
    @cuda threads=threads blocks=cld(fx, threads) _weno_flux_x!(FHOx, FLOx, G, V, nx, ny, nz, g, nfx, Maf, s3f, use_kfvs_anchor, NW, UX, UY, UZ, NC)

    @cuda threads=threads blocks=bcube _ppt_y!(P, G, nfx, nfy, nfz)
    @cuda threads=threads blocks=bcube _vavg_y!(V, P, nfx, nfy, nfz)
    @cuda threads=threads blocks=cld(fy, threads) _weno_flux_y!(FHOy, FLOy, G, V, nx, ny, nz, g, nfy, Maf, s3f, use_kfvs_anchor, NW, UX, UY, UZ, NC)

    @cuda threads=threads blocks=bcube _ppt_z!(P, G, nfx, nfy, nfz)
    @cuda threads=threads blocks=bcube _vavg_z!(V, P, nfx, nfy, nfz)
    @cuda threads=threads blocks=cld(fz, threads) _weno_flux_z!(FHOz, FLOz, G, V, nx, ny, nz, g, nfz, Maf, s3f, use_kfvs_anchor, NW, UX, UY, UZ, NC)

    # --- Pass 2: θ* per cell, then blend + residual ---
    @cuda threads=threads blocks=bint _theta_cell!(Th, G, FHOx, FLOx, FHOy, FLOy, FHOz, FLOz,
                                                   nx, ny, nz, g, λx, λy, λz, theta_closed,
                                                   use_kfvs_anchor, Maf, s3f)
    @cuda threads=threads blocks=bint _blend_residual!(R, Th, FHOx, FLOx, FHOy, FLOy, FHOz, FLOz,
                                                       nx, ny, nz, dxf, dyf, dzf,
                                                       G, g, λx, λy, λz, Maf, s3f,
                                                       Bool(rank_bnd.xlo), Bool(rank_bnd.xhi),
                                                       Bool(rank_bnd.ylo), Bool(rank_bnd.yhi),
                                                       Bool(rank_bnd.zlo), Bool(rank_bnd.zhi),
                                                       theta_closed)
    return nothing
end

"""
    residual3d_order3_gpu(G_host, nx, ny, nz, g, dx, dy, dz, Ma, dt; s3max=40.0)
        -> Array{Float64,4}

Host convenience: upload the haloed cube `(35, nx+2g, ny+2g, nz+2g)`, compute the
order-3 residual, return the interior `(35, nx, ny, nz)`.
"""
function residual3d_order3_gpu(G_host::Array{Float64,4}, nx::Int, ny::Int, nz::Int, g::Int,
                               dx::Real, dy::Real, dz::Real, Ma::Real, dt::Real;
                               s3max::Real = 40.0, threads::Int = 128, theta_closed::Bool = true,
                               use_kfvs_anchor::Bool = false)
    Gd = CuArray(G_host)
    R  = CUDA.zeros(Float64, 35, nx, ny, nz)
    residual3d_order3_box_gpu!(R, Gd, nx, ny, nz, g, dx, dy, dz, Ma, dt; s3max=s3max, threads=threads, theta_closed=theta_closed, use_kfvs_anchor=use_kfvs_anchor)
    CUDA.synchronize()
    return Array(R)
end

end # module

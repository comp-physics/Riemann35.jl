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

# Device kernels come from the package — ONE instance per process, shared with the
# CPU solver and every other GPU module. Do NOT `include` them here: `include`
# splices text, so each include site builds a separate module that is
# type-inferred, LLVM'd and ptxas'd from scratch. This module alone used to pull in
# nine of them, several already instantiated by Residual3DGPU. See the
# device-kernel block in src/Riemann35.jl and misc/04-gotchas.md.
using Riemann35.RiemannFluxDev: riemann_flux_dev
using Riemann35.WavespeedDev: realize_and_speed_Mr_dev
using Riemann35.FluxClosureDev: flux_closure35_dev, flux_closure35_central_dev

# Flux-closure path for the order-3 face flux, selected by dispatch on a singleton -- the same
# mechanism residual3d_gpu.jl already uses for order-2, mirrored here so the two paths agree.
#
#   StdClosure()     standardize -> 21 closures -> destandardize.
#   CentralClosure() central-direct. The standardization sigma factors cancel identically (a
#                    parity property of the HyQMOM closure), so the fifth-order central moments
#                    are rational in the lower ones: 2 sqrt and ~56 intermediates leave the
#                    per-face live set.
#
# MEASURED, order-3, A100, 48^3 (dsmc/reference/gpu_spill_attempt3.csv, companion repo):
#   residual 16.2452 -> 15.4123 ms on top of the unrolling above, 6.5% against the pre-pass
#   baseline of 16.4818 ms. Deviation from StdClosure over 3,870,720 residual entries:
#   8.6e-14 absolute, 7.4e-11 relative -- consistent with the 7e-14 this closure was validated to.
#
# DEFAULT IS Std, DELIBERATELY. The gain is real but it is NOT bit-identical, and every accuracy
# number this project has published came off the standardized path. Flipping this is a one-line
# opt-in for whoever is prepared to re-baseline; it must not change silently.
struct StdClosure end
struct CentralClosure end
const FLUX_CLOSURE = StdClosure()
@inline _flux35(::StdClosure, m::NTuple{35,Float64}) = flux_closure35_dev(
    m[1],  m[2],  m[3],  m[4],  m[5],  m[6],  m[7],  m[8],  m[9],  m[10], m[11], m[12],
    m[13], m[14], m[15], m[16], m[17], m[18], m[19], m[20], m[21], m[22], m[23], m[24],
    m[25], m[26], m[27], m[28], m[29], m[30], m[31], m[32], m[33], m[34], m[35])
@inline _flux35(::CentralClosure, m::NTuple{35,Float64}) = flux_closure35_central_dev(
    m[1],  m[2],  m[3],  m[4],  m[5],  m[6],  m[7],  m[8],  m[9],  m[10], m[11], m[12],
    m[13], m[14], m[15], m[16], m[17], m[18], m[19], m[20], m[21], m[22], m[23], m[24],
    m[25], m[26], m[27], m[28], m[29], m[30], m[31], m[32], m[33], m[34], m[35])
using Riemann35.RealizeDev: realizable_3D_M4_dev
using Riemann35.ReconDev: to_recon_vars_tup
using Riemann35.IdpLimiterDev: theta_star_update_dev, theta_star_update_closed
using Riemann35.KfvsWallDev: kfvs_wall_flux_dev
using Riemann35.HiOrder3ReconDev: recon_point_dev, recon_avg_dev, weno_faces_dev

# --- opt-in log-Jacobi marginal reconstruction (device pieces) ---
using Riemann35.Weno5Dev: weno5z, deconv5, conv5, smooth5
using Riemann35.LogJacobiReconDev: marg_m_to_J, marg_J_to_m, _affine_remap
# per-axis 5 marginal slot indices (= moment_indices.MARG_IDX; hardcoded to avoid a
# device include of the indices module; verified against IJK: x=(1..5), y=(0,1,2,3,4)_y,
# z=(0,1,2,3,4)_z).
const _MARG_IDX = ((1,2,3,4,5), (1,6,10,13,15), (1,16,20,23,25))
# Realizability-safe log-J override: affine-remap the whole state so its AX-marginal
# matches the log-J (rho,u,var) reconstruction jv (device copy of CPU residual_line3).
@inline function _lj_remap_dev(m::NTuple{35,Float64}, v::Val, jv::NTuple{5,Float64})
    ρ = jv[1]; u = jv[2] / ρ; vv = jv[3] / ρ - u * u
    _affine_remap(m, v, ρ, u, vv)
end
@inline _marg5(G, midx, a, b, c) = @inbounds (G[midx[1],a,b,c], G[midx[2],a,b,c], G[midx[3],a,b,c], G[midx[4],a,b,c], G[midx[5],a,b,c])
@inline _vj5(VJ, a, b, c) = @inbounds (VJ[1,a,b,c], VJ[2,a,b,c], VJ[3,a,b,c], VJ[4,a,b,c], VJ[5,a,b,c])

# Runtime θ* dispatch (single compiled kernel holds BOTH paths). `use_closed`
# is a plain Bool threaded from the host through the whole call chain (NOT a
# precompile-time const — that pattern freezes at precompile and silently
# no-ops on the package path). Default OFF ⇒ bisection ⇒ byte-identical.
@inline _theta_star(use_closed::Bool, Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64}) =
    use_closed ? theta_star_update_closed(Mlo, dM) : theta_star_update_dev(Mlo, dM)

export residual3d_order3_box_gpu!, residual3d_order3_gpu, Order3Scratch

@inline _cellG(M, i::Int, j::Int, k::Int) =
    ntuple(m -> @inbounds(M[m, i, j, k]), Val(35))

# ---------------------------------------------------------------------------
# WHY THESE STORES ARE UNROLLED, AND WHY IT IS NOT COSMETIC.
#
# `for m in 1:35; F[m,p,q,r] = t[m]; end` indexes an NTuple{35} with a LOOP VARIABLE. A tuple is
# an SSA value, so a dynamic index forces LLVM to give it an address: it emits
# `alloca [35 x double]` plus a getelementptr with a runtime index, and SROA can never promote
# that alloca back into registers. The tuple then lives in local memory BY CONSTRUCTION -- and
# local memory is exactly what the profiler reports as spill traffic.
#
# MEASURED, not assumed. The IR for _weno_flux_x carried 16 x [35 x double] allocas and 24
# getelementptrs into them with a runtime index; the ones reached by a loop induction variable
# traced to these store loops. Regenerate with gpu/bench/probe_hll_ir.jl.
#
# @nexprs substitutes a literal 1..35, so every index is compile-time and the value stays in
# registers. Identical arithmetic in identical order: bit-identical, and verified so across
# builds with gpu/bench/probe_residual_ref.jl.
@inline function _store35!(F, t::NTuple{35,Float64}, p::Int, q::Int, r::Int)
    Base.Cartesian.@nexprs 35 m -> @inbounds F[m, p, q, r] = t[m]
    return nothing
end

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

    FLall = _flux35(FLUX_CLOSURE, MLr)
    FRall = _flux35(FLUX_CLOSURE, MRr)

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
function _ppt_x!(P, G, nfx::Int, nfy::Int, nfz::Int, b0::Int, nb::Int, c0::Int, nc::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nb * nc
        @inbounds begin
            a = (idx - 1) % nfx + 1;  r = (idx - 1) ÷ nfx
            b = b0 + r % nb;          c = c0 + r ÷ nb
            Pv = (a >= 3 && a <= nfx - 2) ?
                recon_point_dev(_cellG(G, a-2, b, c), _cellG(G, a-1, b, c),
                                _cellG(G, a, b, c), _cellG(G, a+1, b, c), _cellG(G, a+2, b, c)) :
                to_recon_vars_tup(_cellG(G, a, b, c))
            _store35!(P, Pv, a, b, c)
        end
    end
    return nothing
end

# SUB-BOX. `_weno_flux_y!` indexes i in [1,nx], k in [1,nz] and reads V only at a = g+i,
# c = g+k -- interior extent in x and z, full halo in y. Computing the y pass over the whole
# haloed cube therefore produces (nfx*nfz)/(nx*nz) times more than is ever read: 9x at
# nx=nz=8, g=8. Profiling put _ppt_y + _vavg_y at 26.9% of the step, so this is not marginal.
# The restriction is EXACT: it computes what is read and nothing else.
function _ppt_y!(P, G, nfx::Int, nfy::Int, nfz::Int, a0::Int, na::Int, c0::Int, nc::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= na * nfy * nc
        @inbounds begin
            a = a0 + (idx - 1) % na;  r = (idx - 1) ÷ na
            b = r % nfy + 1;          c = c0 + r ÷ nfy
            Pv = (b >= 3 && b <= nfy - 2) ?
                recon_point_dev(_cellG(G, a, b-2, c), _cellG(G, a, b-1, c),
                                _cellG(G, a, b, c), _cellG(G, a, b+1, c), _cellG(G, a, b+2, c)) :
                to_recon_vars_tup(_cellG(G, a, b, c))
            _store35!(P, Pv, a, b, c)
        end
    end
    return nothing
end

function _ppt_z!(P, G, nfx::Int, nfy::Int, nfz::Int, a0::Int, na::Int, b0::Int, nb::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= na * nb * nfz
        @inbounds begin
            a = a0 + (idx - 1) % na;  r = (idx - 1) ÷ na
            b = b0 + r % nb;          c = r ÷ nb + 1
            Pv = (c >= 3 && c <= nfz - 2) ?
                recon_point_dev(_cellG(G, a, b, c-2), _cellG(G, a, b, c-1),
                                _cellG(G, a, b, c), _cellG(G, a, b, c+1), _cellG(G, a, b, c+2)) :
                to_recon_vars_tup(_cellG(G, a, b, c))
            _store35!(P, Pv, a, b, c)
        end
    end
    return nothing
end

# Step 2: conv5 of the 5-cell recon-var POINT stencil → recon-var cell AVERAGE.
# Stencil index clamped to [1,nf] along the axis (residual_line3 `_pp`).
function _vavg_x!(V, P, nfx::Int, nfy::Int, nfz::Int, b0::Int, nb::Int, c0::Int, nc::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nfx * nb * nc
        @inbounds begin
            a = (idx - 1) % nfx + 1;  r = (idx - 1) ÷ nfx
            b = b0 + r % nb;          c = c0 + r ÷ nb
            Vv = recon_avg_dev(_cellG(P, _clamp(a-2, nfx), b, c), _cellG(P, _clamp(a-1, nfx), b, c),
                               _cellG(P, a, b, c),
                               _cellG(P, _clamp(a+1, nfx), b, c), _cellG(P, _clamp(a+2, nfx), b, c))
            _store35!(V, Vv, a, b, c)
        end
    end
    return nothing
end

function _vavg_y!(V, P, nfx::Int, nfy::Int, nfz::Int, a0::Int, na::Int, c0::Int, nc::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= na * nfy * nc
        @inbounds begin
            a = a0 + (idx - 1) % na;  r = (idx - 1) ÷ na
            b = r % nfy + 1;          c = c0 + r ÷ nfy
            Vv = recon_avg_dev(_cellG(P, a, _clamp(b-2, nfy), c), _cellG(P, a, _clamp(b-1, nfy), c),
                               _cellG(P, a, b, c),
                               _cellG(P, a, _clamp(b+1, nfy), c), _cellG(P, a, _clamp(b+2, nfy), c))
            _store35!(V, Vv, a, b, c)
        end
    end
    return nothing
end

function _vavg_z!(V, P, nfx::Int, nfy::Int, nfz::Int, a0::Int, na::Int, b0::Int, nb::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= na * nb * nfz
        @inbounds begin
            a = a0 + (idx - 1) % na;  r = (idx - 1) ÷ na
            b = b0 + r % nb;          c = r ÷ nb + 1
            Vv = recon_avg_dev(_cellG(P, a, b, _clamp(c-2, nfz)), _cellG(P, a, b, _clamp(c-1, nfz)),
                               _cellG(P, a, b, c),
                               _cellG(P, a, b, _clamp(c+1, nfz)), _cellG(P, a, b, _clamp(c+2, nfz)))
            _store35!(V, Vv, a, b, c)
        end
    end
    return nothing
end

# Step 3: per interface, WENO5 L/R faces + HLL → F_HO and F_LO.
# Interface f (1..nx+1) between cube cells il = g+f-1 and il+1 at interior (j,k).
function _weno_flux_x!(FHO, FLO, G, V, nx::Int, ny::Int, nz::Int, g::Int, nfx::Int,
                       Ma::Float64, s3f::Float64, first_order::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nx + 1
    if idx <= nf * ny * nz
        @inbounds begin
            f = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            j = r % ny + 1;         k = r ÷ ny + 1
            b = g + j; c = g + k; il = g + f - 1
            cL = _cellG(G, il, b, c); cR = _cellG(G, il + 1, b, c)
            FL = _hll_states(cL, cR, 1, Ma, s3f)           # first-order HLL anchor (shared)
            FH = if first_order                            # first-order: FHO = FLO
                FL
            else                                           # order-3: WENO5 face + HLL
                W1 = _cellG(V, _clamp(il-2, nfx), b, c); W2 = _cellG(V, _clamp(il-1, nfx), b, c)
                W3 = _cellG(V, _clamp(il,   nfx), b, c); W4 = _cellG(V, _clamp(il+1, nfx), b, c)
                W5 = _cellG(V, _clamp(il+2, nfx), b, c); W6 = _cellG(V, _clamp(il+3, nfx), b, c)
                mL, mR = weno_faces_dev(W1, W2, W3, W4, W5, W6, cL, cR)
                _hll_states(mL, mR, 1, Ma, s3f)
            end
            _store35!(FHO, FH, f, j, k); _store35!(FLO, FL, f, j, k)
        end
    end
    return nothing
end

function _weno_flux_y!(FHO, FLO, G, V, nx::Int, ny::Int, nz::Int, g::Int, nfy::Int,
                       Ma::Float64, s3f::Float64, first_order::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = ny + 1
    if idx <= nf * nx * nz
        @inbounds begin
            f = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         k = r ÷ nx + 1
            a = g + i; c = g + k; jl = g + f - 1
            cL = _cellG(G, a, jl, c); cR = _cellG(G, a, jl + 1, c)
            FL = _hll_states(cL, cR, 2, Ma, s3f)
            FH = if first_order
                FL
            else
                W1 = _cellG(V, a, _clamp(jl-2, nfy), c); W2 = _cellG(V, a, _clamp(jl-1, nfy), c)
                W3 = _cellG(V, a, _clamp(jl,   nfy), c); W4 = _cellG(V, a, _clamp(jl+1, nfy), c)
                W5 = _cellG(V, a, _clamp(jl+2, nfy), c); W6 = _cellG(V, a, _clamp(jl+3, nfy), c)
                mL, mR = weno_faces_dev(W1, W2, W3, W4, W5, W6, cL, cR)
                _hll_states(mL, mR, 2, Ma, s3f)
            end
            _store35!(FHO, FH, i, f, k); _store35!(FLO, FL, i, f, k)
        end
    end
    return nothing
end

function _weno_flux_z!(FHO, FLO, G, V, nx::Int, ny::Int, nz::Int, g::Int, nfz::Int,
                       Ma::Float64, s3f::Float64, first_order::Bool)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    nf = nz + 1
    if idx <= nf * nx * ny
        @inbounds begin
            f = (idx - 1) % nf + 1; r = (idx - 1) ÷ nf
            i = r % nx + 1;         j = r ÷ nx + 1
            a = g + i; b = g + j; kl = g + f - 1
            cL = _cellG(G, a, b, kl); cR = _cellG(G, a, b, kl + 1)
            FL = _hll_states(cL, cR, 3, Ma, s3f)
            FH = if first_order
                FL
            else
                W1 = _cellG(V, a, b, _clamp(kl-2, nfz)); W2 = _cellG(V, a, b, _clamp(kl-1, nfz))
                W3 = _cellG(V, a, b, _clamp(kl,   nfz)); W4 = _cellG(V, a, b, _clamp(kl+1, nfz))
                W5 = _cellG(V, a, b, _clamp(kl+2, nfz)); W6 = _cellG(V, a, b, _clamp(kl+3, nfz))
                mL, mR = weno_faces_dev(W1, W2, W3, W4, W5, W6, cL, cR)
                _hll_states(mL, mR, 3, Ma, s3f)
            end
            _store35!(FHO, FH, i, j, f); _store35!(FLO, FL, i, j, f)
        end
    end
    return nothing
end

# ===========================================================================
# OPT-IN log-Jacobi marginal pipeline: a J-domain 3-pass on the face-normal
# marginal chain (m0..m4 at midx), parallel to the raw recon 3-pass above, mirroring
# CPU logjacobi_marginal_faces call-for-call. _ppt_marg: deconv5-gated marginal point
# -> marg_m_to_J (+ per-cell ok). _vavg_marg: conv5 -> J cell-average. _okline: AND ok
# over the whole line (all-or-nothing fallback). _weno_flux_lj: as _weno_flux but, when
# the line is ok, WENO5-Z in J -> marg_J_to_m -> override the marginal slots of mL/mR.
# ===========================================================================
function _ppt_marg_x!(PJ, OK, G, midx, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if idx <= nfx*nfy*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; r=(idx-1)÷nfx; b=r%nfy+1; c=r÷nfy+1
            mpt = (a>=3 && a<=nfx-2) ?
                (cm2=_marg5(G,midx,a-2,b,c); cm1=_marg5(G,midx,a-1,b,c); c0=_marg5(G,midx,a,b,c); cp1=_marg5(G,midx,a+1,b,c); cp2=_marg5(G,midx,a+2,b,c);
                 ntuple(q -> smooth5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) ? deconv5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) : c0[q], Val(5))) :
                _marg5(G,midx,a,b,c)
            ok, J = marg_m_to_J(mpt[1],mpt[2],mpt[3],mpt[4],mpt[5])
            PJ[1,a,b,c]=J[1]; PJ[2,a,b,c]=J[2]; PJ[3,a,b,c]=J[3]; PJ[4,a,b,c]=J[4]; PJ[5,a,b,c]=J[5]
            OK[a,b,c] = ok ? 1.0 : 0.0
        end
    end
    return nothing
end
function _ppt_marg_y!(PJ, OK, G, midx, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if idx <= nfx*nfy*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; r=(idx-1)÷nfx; b=r%nfy+1; c=r÷nfy+1
            mpt = (b>=3 && b<=nfy-2) ?
                (cm2=_marg5(G,midx,a,b-2,c); cm1=_marg5(G,midx,a,b-1,c); c0=_marg5(G,midx,a,b,c); cp1=_marg5(G,midx,a,b+1,c); cp2=_marg5(G,midx,a,b+2,c);
                 ntuple(q -> smooth5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) ? deconv5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) : c0[q], Val(5))) :
                _marg5(G,midx,a,b,c)
            ok, J = marg_m_to_J(mpt[1],mpt[2],mpt[3],mpt[4],mpt[5])
            PJ[1,a,b,c]=J[1]; PJ[2,a,b,c]=J[2]; PJ[3,a,b,c]=J[3]; PJ[4,a,b,c]=J[4]; PJ[5,a,b,c]=J[5]
            OK[a,b,c] = ok ? 1.0 : 0.0
        end
    end
    return nothing
end
function _ppt_marg_z!(PJ, OK, G, midx, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if idx <= nfx*nfy*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; r=(idx-1)÷nfx; b=r%nfy+1; c=r÷nfy+1
            mpt = (c>=3 && c<=nfz-2) ?
                (cm2=_marg5(G,midx,a,b,c-2); cm1=_marg5(G,midx,a,b,c-1); c0=_marg5(G,midx,a,b,c); cp1=_marg5(G,midx,a,b,c+1); cp2=_marg5(G,midx,a,b,c+2);
                 ntuple(q -> smooth5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) ? deconv5(cm2[q],cm1[q],c0[q],cp1[q],cp2[q]) : c0[q], Val(5))) :
                _marg5(G,midx,a,b,c)
            ok, J = marg_m_to_J(mpt[1],mpt[2],mpt[3],mpt[4],mpt[5])
            PJ[1,a,b,c]=J[1]; PJ[2,a,b,c]=J[2]; PJ[3,a,b,c]=J[3]; PJ[4,a,b,c]=J[4]; PJ[5,a,b,c]=J[5]
            OK[a,b,c] = ok ? 1.0 : 0.0
        end
    end
    return nothing
end
function _vavg_marg_x!(VJ, PJ, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if idx <= nfx*nfy*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; r=(idx-1)÷nfx; b=r%nfy+1; c=r÷nfy+1
            am2=_clamp(a-2,nfx); am1=_clamp(a-1,nfx); ap1=_clamp(a+1,nfx); ap2=_clamp(a+2,nfx)
            for q in 1:5; VJ[q,a,b,c]=conv5(PJ[q,am2,b,c],PJ[q,am1,b,c],PJ[q,a,b,c],PJ[q,ap1,b,c],PJ[q,ap2,b,c]); end
        end
    end
    return nothing
end
function _vavg_marg_y!(VJ, PJ, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if idx <= nfx*nfy*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; r=(idx-1)÷nfx; b=r%nfy+1; c=r÷nfy+1
            bm2=_clamp(b-2,nfy); bm1=_clamp(b-1,nfy); bp1=_clamp(b+1,nfy); bp2=_clamp(b+2,nfy)
            for q in 1:5; VJ[q,a,b,c]=conv5(PJ[q,a,bm2,c],PJ[q,a,bm1,c],PJ[q,a,b,c],PJ[q,a,bp1,c],PJ[q,a,bp2,c]); end
        end
    end
    return nothing
end
function _vavg_marg_z!(VJ, PJ, nfx::Int, nfy::Int, nfz::Int)
    idx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if idx <= nfx*nfy*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; r=(idx-1)÷nfx; b=r%nfy+1; c=r÷nfy+1
            cm2=_clamp(c-2,nfz); cm1=_clamp(c-1,nfz); cp1=_clamp(c+1,nfz); cp2=_clamp(c+2,nfz)
            for q in 1:5; VJ[q,a,b,c]=conv5(PJ[q,a,b,cm2],PJ[q,a,b,cm1],PJ[q,a,b,c],PJ[q,a,b,cp1],PJ[q,a,b,cp2]); end
        end
    end
    return nothing
end
function _okline_x!(OKL, OK, nfx::Int, nfy::Int, nfz::Int)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx <= nfy*nfz
        @inbounds begin
            b=(idx-1)%nfy+1; c=(idx-1)÷nfy+1; good=1.0
            for a in 1:nfx; if OK[a,b,c] < 0.5; good=0.0; end; end
            OKL[b,c]=good
        end
    end
    return nothing
end
function _okline_y!(OKL, OK, nfx::Int, nfy::Int, nfz::Int)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx <= nfx*nfz
        @inbounds begin
            a=(idx-1)%nfx+1; c=(idx-1)÷nfx+1; good=1.0
            for b in 1:nfy; if OK[a,b,c] < 0.5; good=0.0; end; end
            OKL[a,c]=good
        end
    end
    return nothing
end
function _okline_z!(OKL, OK, nfx::Int, nfy::Int, nfz::Int)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if idx <= nfx*nfy
        @inbounds begin
            a=(idx-1)%nfx+1; b=(idx-1)÷nfx+1; good=1.0
            for c in 1:nfz; if OK[a,b,c] < 0.5; good=0.0; end; end
            OKL[a,b]=good
        end
    end
    return nothing
end
@inline function _wenoJ5(J1,J2,J3,J4,J5)
    (weno5z(J1[1],J2[1],J3[1],J4[1],J5[1]), weno5z(J1[2],J2[2],J3[2],J4[2],J5[2]),
     weno5z(J1[3],J2[3],J3[3],J4[3],J5[3]), weno5z(J1[4],J2[4],J3[4],J4[4],J5[4]),
     weno5z(J1[5],J2[5],J3[5],J4[5],J5[5]))
end
function _weno_flux_lj_x!(FHO, FLO, G, V, VJ, OKL, midx, nx::Int,ny::Int,nz::Int,g::Int,nfx::Int, Ma::Float64, s3f::Float64)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nx+1
    if idx <= nf*ny*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; j=r%ny+1; k=r÷ny+1
            b=g+j; c=g+k; il=g+f-1
            cL=_cellG(G,il,b,c); cR=_cellG(G,il+1,b,c)
            W1=_cellG(V,_clamp(il-2,nfx),b,c); W2=_cellG(V,_clamp(il-1,nfx),b,c); W3=_cellG(V,_clamp(il,nfx),b,c)
            W4=_cellG(V,_clamp(il+1,nfx),b,c); W5=_cellG(V,_clamp(il+2,nfx),b,c); W6=_cellG(V,_clamp(il+3,nfx),b,c)
            mL, mR = weno_faces_dev(W1,W2,W3,W4,W5,W6,cL,cR)
            if OKL[b,c] > 0.5
                J1=_vj5(VJ,_clamp(il-2,nfx),b,c); J2=_vj5(VJ,_clamp(il-1,nfx),b,c); J3=_vj5(VJ,_clamp(il,nfx),b,c)
                J4=_vj5(VJ,_clamp(il+1,nfx),b,c); J5=_vj5(VJ,_clamp(il+2,nfx),b,c); J6=_vj5(VJ,_clamp(il+3,nfx),b,c)
                mL=_lj_remap_dev(mL,Val(1),marg_J_to_m(_wenoJ5(J1,J2,J3,J4,J5)))
                mR=_lj_remap_dev(mR,Val(1),marg_J_to_m(_wenoJ5(J6,J5,J4,J3,J2)))
            end
            FH=_hll_states(mL,mR,1,Ma,s3f); FL=_hll_states(cL,cR,1,Ma,s3f)
            _store35!(FHO, FH, f, j, k); _store35!(FLO, FL, f, j, k)
        end
    end
    return nothing
end
function _weno_flux_lj_y!(FHO, FLO, G, V, VJ, OKL, midx, nx::Int,ny::Int,nz::Int,g::Int,nfy::Int, Ma::Float64, s3f::Float64)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=ny+1
    if idx <= nf*nx*nz
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; i=r%nx+1; k=r÷nx+1
            a=g+i; c=g+k; jl=g+f-1
            cL=_cellG(G,a,jl,c); cR=_cellG(G,a,jl+1,c)
            W1=_cellG(V,a,_clamp(jl-2,nfy),c); W2=_cellG(V,a,_clamp(jl-1,nfy),c); W3=_cellG(V,a,_clamp(jl,nfy),c)
            W4=_cellG(V,a,_clamp(jl+1,nfy),c); W5=_cellG(V,a,_clamp(jl+2,nfy),c); W6=_cellG(V,a,_clamp(jl+3,nfy),c)
            mL, mR = weno_faces_dev(W1,W2,W3,W4,W5,W6,cL,cR)
            if OKL[a,c] > 0.5
                J1=_vj5(VJ,a,_clamp(jl-2,nfy),c); J2=_vj5(VJ,a,_clamp(jl-1,nfy),c); J3=_vj5(VJ,a,_clamp(jl,nfy),c)
                J4=_vj5(VJ,a,_clamp(jl+1,nfy),c); J5=_vj5(VJ,a,_clamp(jl+2,nfy),c); J6=_vj5(VJ,a,_clamp(jl+3,nfy),c)
                mL=_lj_remap_dev(mL,Val(2),marg_J_to_m(_wenoJ5(J1,J2,J3,J4,J5)))
                mR=_lj_remap_dev(mR,Val(2),marg_J_to_m(_wenoJ5(J6,J5,J4,J3,J2)))
            end
            FH=_hll_states(mL,mR,2,Ma,s3f); FL=_hll_states(cL,cR,2,Ma,s3f)
            _store35!(FHO, FH, i, f, k); _store35!(FLO, FL, i, f, k)
        end
    end
    return nothing
end
function _weno_flux_lj_z!(FHO, FLO, G, V, VJ, OKL, midx, nx::Int,ny::Int,nz::Int,g::Int,nfz::Int, Ma::Float64, s3f::Float64)
    idx=(blockIdx().x-1)*blockDim().x+threadIdx().x; nf=nz+1
    if idx <= nf*nx*ny
        @inbounds begin
            f=(idx-1)%nf+1; r=(idx-1)÷nf; i=r%nx+1; j=r÷nx+1
            a=g+i; b=g+j; kl=g+f-1
            cL=_cellG(G,a,b,kl); cR=_cellG(G,a,b,kl+1)
            W1=_cellG(V,a,b,_clamp(kl-2,nfz)); W2=_cellG(V,a,b,_clamp(kl-1,nfz)); W3=_cellG(V,a,b,_clamp(kl,nfz))
            W4=_cellG(V,a,b,_clamp(kl+1,nfz)); W5=_cellG(V,a,b,_clamp(kl+2,nfz)); W6=_cellG(V,a,b,_clamp(kl+3,nfz))
            mL, mR = weno_faces_dev(W1,W2,W3,W4,W5,W6,cL,cR)
            if OKL[a,b] > 0.5
                J1=_vj5(VJ,a,b,_clamp(kl-2,nfz)); J2=_vj5(VJ,a,b,_clamp(kl-1,nfz)); J3=_vj5(VJ,a,b,_clamp(kl,nfz))
                J4=_vj5(VJ,a,b,_clamp(kl+1,nfz)); J5=_vj5(VJ,a,b,_clamp(kl+2,nfz)); J6=_vj5(VJ,a,b,_clamp(kl+3,nfz))
                mL=_lj_remap_dev(mL,Val(3),marg_J_to_m(_wenoJ5(J1,J2,J3,J4,J5)))
                mR=_lj_remap_dev(mR,Val(3),marg_J_to_m(_wenoJ5(J6,J5,J4,J3,J2)))
            end
            FH=_hll_states(mL,mR,3,Ma,s3f); FL=_hll_states(cL,cR,3,Ma,s3f)
            _store35!(FHO, FH, i, j, f); _store35!(FLO, FL, i, j, f)
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
                      λx::Float64, λy::Float64, λz::Float64, use_closed::Bool)
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

            Th[1, i, j, k] = _theta_star(use_closed, Mlo, ntuple(q -> -6λx * Gxr[q], Val(35)))
            Th[2, i, j, k] = _theta_star(use_closed, Mlo, ntuple(q ->  6λx * Gxl[q], Val(35)))
            Th[3, i, j, k] = _theta_star(use_closed, Mlo, ntuple(q -> -6λy * Gyr[q], Val(35)))
            Th[4, i, j, k] = _theta_star(use_closed, Mlo, ntuple(q ->  6λy * Gyl[q], Val(35)))
            Th[5, i, j, k] = _theta_star(use_closed, Mlo, ntuple(q -> -6λz * Gzr[q], Val(35)))
            Th[6, i, j, k] = _theta_star(use_closed, Mlo, ntuple(q ->  6λz * Gzl[q], Val(35)))
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
# COLD-PATH wrapper around `_hll_states`, deliberately NOT inlined. `_halo_cell_mlo`
# calls the HLL body six times, and this is rank-boundary code: dormant entirely for
# x/y (the GPU decomposes z only) and reached only at slab edges for z. So six inlined
# copies of a ~14k-line body buy nothing at runtime and cost heavily at compile time.
# The hot WENO flux kernels keep calling `_hll_states` directly and stay fully inlined —
# that is the whole point of routing only this path through a separate function.
@noinline _hll_states_cold(mL::NTuple{35,Float64}, mR::NTuple{35,Float64},
                           ax::Int, Ma::Float64, s3f::Float64) =
    _hll_states(mL, mR, ax, Ma, s3f)

@inline function _halo_cell_mlo(G, px::Int, py::Int, pk::Int,
                                 λx::Float64, λy::Float64, λz::Float64,
                                 Ma::Float64, s3f::Float64)
    C = _cellG(G, px, py, pk)
    (λx == 0.0 && λy == 0.0 && λz == 0.0) && return C
    FxL = _hll_states_cold(_cellG(G, px-1, py, pk), C, 1, Ma, s3f)
    FxR = _hll_states_cold(C, _cellG(G, px+1, py, pk), 1, Ma, s3f)
    FyD = _hll_states_cold(_cellG(G, px, py-1, pk), C, 2, Ma, s3f)
    FyU = _hll_states_cold(C, _cellG(G, px, py+1, pk), 2, Ma, s3f)
    FzB = _hll_states_cold(_cellG(G, px, py, pk-1), C, 3, Ma, s3f)
    FzF = _hll_states_cold(C, _cellG(G, px, py, pk+1), 3, Ma, s3f)
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
#
# @noinline IS LOAD-BEARING FOR COMPILE TIME, and is the reason this file compiles in
# minutes rather than a quarter hour. `_blend_residual!` calls this once per face, six
# times. Inlined, each copy drags in `_halo_cell_mlo`, which itself inlines
# `_hll_states` six times — so the kernel carried THIRTY-SIX inlined copies of the HLL
# closure+eigenvalue body plus six of `_theta_star`. Measured on an A100 with
# CUDA.@device_code_ptx: `_blend_residual!` emitted 452,709 lines of PTX, 79.2% of ALL
# device code in the order-3 march and 15x the next-largest kernel (`_weno_flux_z!`,
# 29k). ptxas cost grows worse than linearly in function size, so that one kernel was
# essentially the entire compile.
#
# Do NOT "optimize" this back to @inline without re-measuring the PTX line count.
@noinline function _rank_face_theta(G, Thf::Float64, px::Int, py::Int, pk::Int,
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
                          use_closed::Bool,
                          # KFVS wall faces: which global boundaries are walls, and the wall
                          # state. All false => byte-identical to the ghost-cell path.
                          wxlo::Bool, wxhi::Bool, wylo::Bool, wyhi::Bool, wzlo::Bool, wzhi::Bool,
                          wTw::Float64, wTwH::Float64, wU1::Float64, wU2::Float64,
                          wAnti::Bool)
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

            # --- KFVS override at wall faces ------------------------------------------
            # A wall face gets the exact half-space flux instead of the ghost-cell HLL flux.
            # The wall's TANGENTIAL velocity is antisymmetric across the pair when wAnti is
            # set (that is what makes a Couette channel, not a translating one), and the hi
            # face may sit at a different temperature (wTwH) for a Fourier channel.
            uxl =  wAnti ? -wU1 : wU1;  ux2l =  wAnti ? -wU2 : wU2
            axr = (i == nx) && wxhi; axl = (i == 1) && wxlo
            ayr = (j == ny) && wyhi; ayl = (j == 1) && wylo
            azr = (k == nz) && wzhi; azl = (k == 1) && wzlo
            Wxr = axr ? _kfvs_face(G, g+nx, g+j, g+k, 1, +1.0, wTwH, wU1,  wU2,  nx) : FHxr
            Wxl = axl ? _kfvs_face(G, g+1,  g+j, g+k, 1, -1.0, wTw,  uxl,  ux2l, nx) : FHxl
            Wyr = ayr ? _kfvs_face(G, g+i, g+ny, g+k, 2, +1.0, wTwH, wU1,  wU2,  ny) : FHyr
            Wyl = ayl ? _kfvs_face(G, g+i, g+1,  g+k, 2, -1.0, wTw,  uxl,  ux2l, ny) : FHyl
            Wzr = azr ? _kfvs_face(G, g+i, g+j, g+nz, 3, +1.0, wTwH, wU1,  wU2,  nz) : FHzr
            Wzl = azl ? _kfvs_face(G, g+i, g+j, g+1,  3, -1.0, wTw,  uxl,  ux2l, nz) : FHzl

            # UNROLLED for the same reason as _store35!, and this is the worst case in the file:
            # written as `for m in 1:35`, this body indexes EIGHTEEN NTuple{35} values with a loop
            # variable (six faces x FH/FL/W), so all eighteen are forced into `alloca
            # [35 x double]` -- 630 doubles pinned to local memory in a kernel whose register
            # budget is 127 doubles. That matched this kernel's measured profile exactly: 255
            # registers, 12.05% occupancy, 67% DRAM-bound with most of that traffic being spill.
            #
            # Unrolling took its local traffic from 123.5M to 38.9M sectors, -68%, the largest
            # single win of this pass (dsmc/reference/gpu_spill_attempt3.csv in the companion
            # repo). Identical arithmetic in identical order; bit-identical.
            Base.Cartesian.@nexprs 35 m -> begin
                Fxr = axr ? Wxr[m] : FLxr[m] + θxr * (FHxr[m] - FLxr[m])
                Fxl = axl ? Wxl[m] : FLxl[m] + θxl * (FHxl[m] - FLxl[m])
                Fyr = ayr ? Wyr[m] : FLyr[m] + θyr * (FHyr[m] - FLyr[m])
                Fyl = ayl ? Wyl[m] : FLyl[m] + θyl * (FHyl[m] - FLyl[m])
                Fzr = azr ? Wzr[m] : FLzr[m] + θzr * (FHzr[m] - FLzr[m])
                Fzl = azl ? Wzl[m] : FLzl[m] + θzl * (FHzl[m] - FLzl[m])
                R[m, i, j, k] = -((Fxr-Fxl)/dx + (Fyr-Fyl)/dy + (Fzr-Fzl)/dz)
            end
        end
    end
    return nothing
end


# ---------------------------------------------------------------------------
# KFVS WALL FLUX at a boundary face. @noinline for the same reason
# `_rank_face_theta` is: this kernel already emits ~79% of the march's PTX, and
# inlining a 35-moment closure six times is how the order-3 compile once reached
# 758 s. Emitted once, called per wall face.
#
# Replaces the ghost-cell flux, which cannot represent a diffuse wall at all: f is
# DISCONTINUOUS at v.n = 0 and a full-line Riemann problem has no half-space split,
# so at u_n = 0 HLL leaves pure dissipation on the density jump and leaks mass
# (-3.9e-4 against the DVM's 1e-15, issue #36). Here rho_w is set FROM zero net mass
# flux, so conservation is exact by construction.
# ---------------------------------------------------------------------------
#
# THE STATE IT IS GIVEN IS EXTRAPOLATED TO THE FACE, not read from the cell centre.
# `kfvs_wall_flux_dev` reconstructs a Gaussian from whatever state it is handed and
# integrates it over the half-space AT THE WALL, so handing it the first cell's CENTRE
# value -- half a cell away -- makes the boundary flux first-order while the interior is
# WENO order-3. Interior faces never see this because they are fed reconstructed values.
#
# It matters far more than an order count suggests, because the near-wall curvature is
# exactly what the Knudsen layer is. Measured at Kn = 0.73 against DSMC, whole-profile
# L2 with the cell-centre state:
#
#     ny = 96   17.24%      <- vs 3.69% for the ghost cell it was meant to replace
#     ny = 192   7.34%
#     ny = 384   3.09%      <- vs 3.41%, i.e. only converged does it come out ahead
#
# so it was a REGRESSION at any resolution a user would actually reach for. Cell centres
# sit at dy/2, 3dy/2, 5dy/2 from the wall, so the one-sided quadratic to the face is
# (15 M1 - 10 M2 + 3 M3)/8 -- third-order, matching the interior.
#
# GUARDED, because componentwise extrapolation of a moment vector is not closed on the
# realizable set: an overshoot can hand back a state no distribution has. Only ten of the
# 35 reach the flux (rho, the three velocities, the six second moments), so the guard is
# exactly a positive density and a positive-definite covariance -- checked by leading
# minors, falling back to the cell centre when it fails. Fewer than 3 cells on the axis
# has no stencil, so it falls back there too.
@inline function _pd_ok(M::NTuple{35,Float64})
    rho = M[1]
    rho > 0.0 || return false
    ux = M[2]/rho; uy = M[6]/rho; uz = M[16]/rho
    a = M[3]/rho - ux*ux; b = M[10]/rho - uy*uy; c = M[20]/rho - uz*uz
    d = M[7]/rho - ux*uy; e = M[17]/rho - ux*uz; f = M[26]/rho - uy*uz
    a > 0.0 || return false
    (a*b - d*d) > 0.0 || return false
    (a*(b*c - f*f) - d*(d*c - f*e) + e*(d*f - b*e)) > 0.0
end

@noinline function _kfvs_face(G, px::Int, py::Int, pk::Int, axis::Int, outward::Float64,
                              Tw::Float64, uw1::Float64, uw2::Float64, n::Int)
    M1 = ntuple(m -> @inbounds(G[m, px, py, pk]), Val(35))
    if n >= 3
        s  = outward > 0 ? -1 : 1                       # step INWARD from the wall
        dx = axis == 1 ? s : 0; dy = axis == 2 ? s : 0; dz = axis == 3 ? s : 0
        M2 = ntuple(m -> @inbounds(G[m, px+dx,   py+dy,   pk+dz]),   Val(35))
        M3 = ntuple(m -> @inbounds(G[m, px+2dx,  py+2dy,  pk+2dz]),  Val(35))
        Mf = ntuple(m -> (15.0*M1[m] - 10.0*M2[m] + 3.0*M3[m])/8.0, Val(35))
        _pd_ok(Mf) && return kfvs_wall_flux_dev(Mf, axis, outward, Tw, uw1, uw2)
    end
    kfvs_wall_flux_dev(M1, axis, outward, Tw, uw1, uw2)
end

# ===========================================================================
# Driver.  G is the FULLY-HALOED cube (35, nfx, nfy, nfz), nfx=nx+2g etc.
# R is the interior residual (35, nx, ny, nz).
# ===========================================================================
"""
    Order3Scratch(nx, ny, nz, g)

Per-call scratch for [`residual3d_order3_box_gpu!`](@ref), hoisted so a caller can allocate it
ONCE instead of once per residual call.

WHY THIS EXISTS. The residual allocates two full `(35, nf, nf, nf)` cubes (`P`, `V`) plus six
face arrays and `Th` on every call, and CUDA.jl's pool does not release them between the three
RK stages of a step. The peak is therefore ~3x the scratch of a single call on top of the
resident state: measured 1208 MB at 48^3 against a 134 MB resident cube, and 2.6 GB at 64^3.
`residual3d_gpu!` (order 1/2) already takes an `Fbuf` from its caller for exactly this reason;
the order-3 path never got the equivalent.

BIT-IDENTITY IS PRESERVED BY ZEROING ON ENTRY, not by arguing that every element is written
before it is read. Fresh `CUDA.zeros` hands the kernels zeros; a reused buffer must therefore be
handed zeros too. That argument would be delicate to make element-wise -- `Th` is explicitly
relied upon to be zero in the first-order path, and the pass-1 reconstruction kernels were
restricted to a sub-box (1bbaa67) so parts of `P`/`V` are deliberately never written -- and a
`fill!` costs a memset against a residual that reads the same arrays many times over.
"""
struct Order3Scratch
    P::CuArray{Float64,4}
    V::CuArray{Float64,4}
    FHOx::CuArray{Float64,4}; FLOx::CuArray{Float64,4}
    FHOy::CuArray{Float64,4}; FLOy::CuArray{Float64,4}
    FHOz::CuArray{Float64,4}; FLOz::CuArray{Float64,4}
    Th::CuArray{Float64,4}
    dims::NTuple{4,Int}          # (nx, ny, nz, g) -- guards against reuse at the wrong size
end

function Order3Scratch(nx::Int, ny::Int, nz::Int, g::Int)
    nfx = nx + 2g; nfy = ny + 2g; nfz = nz + 2g
    Order3Scratch(CUDA.zeros(Float64, 35, nfx, nfy, nfz),
                  CUDA.zeros(Float64, 35, nfx, nfy, nfz),
                  CUDA.zeros(Float64, 35, nx+1, ny, nz), CUDA.zeros(Float64, 35, nx+1, ny, nz),
                  CUDA.zeros(Float64, 35, nx, ny+1, nz), CUDA.zeros(Float64, 35, nx, ny+1, nz),
                  CUDA.zeros(Float64, 35, nx, ny, nz+1), CUDA.zeros(Float64, 35, nx, ny, nz+1),
                  CUDA.zeros(Float64, 6, nx, ny, nz),
                  (nx, ny, nz, g))
end

"""Zero every buffer, so a reused scratch is indistinguishable from a freshly allocated one."""
function _reset!(w::Order3Scratch)
    for a in (w.P, w.V, w.FHOx, w.FLOx, w.FHOy, w.FLOy, w.FHOz, w.FLOz, w.Th)
        fill!(a, 0.0)
    end
    return w
end

function residual3d_order3_box_gpu!(R::CuArray{Float64,4}, G::CuArray{Float64,4},
                                    nx::Int, ny::Int, nz::Int, g::Int,
                                    dx::Real, dy::Real, dz::Real, Ma::Real, dt::Real;
                                    s3max::Real = 40.0, threads::Int = 128,
                                    theta_closed::Bool = true,
                                    use_logjacobi_recon::Bool = false,
                                    first_order::Bool = false,
                                    # KFVS wall faces; default all-false reproduces the
                                    # ghost-cell path byte-for-byte.
                                    kfvs_walls = (false,false,false,false,false,false),
                                    wall_Tw::Float64 = 1.0, wall_Tw_hi::Float64 = 1.0,
                                    wall_uw1::Float64 = 0.0, wall_uw2::Float64 = 0.0,
                                    wall_antisym::Bool = false,
                                    # DIAGNOSTIC ONLY. idp=false forces theta == 1, i.e. pure
                                    # WENO5 with the IDP blend switched OFF. It exists to
                                    # separate two candidate causes of the closure's
                                    # wrong-SIGN slip trend (zeta rises with Kn where truth
                                    # falls): fourth-order moment truncation, or the limiter
                                    # clamping near-wall fluxes harder as rarefaction pushes
                                    # the near-wall state toward the realizability boundary.
                                    # Both configurations otherwise run identically, so the
                                    # difference is attributable. NOT for production: with
                                    # theta == 1 there is no realizability guarantee at all.
                                    idp::Bool = true,
                                    rank_bnd = (xlo=false, xhi=false, ylo=false, yhi=false,
                                                zlo=false, zhi=false),
                                     active::NTuple{3,Bool} = (true, true, true),
                                    ws::Union{Nothing,Order3Scratch} = nothing)
    nfx = nx + 2g; nfy = ny + 2g; nfz = nz + 2g
    @assert g >= 4 "order-3 residual requires halo g ≥ 4; got g=$g"
    @assert size(G) == (35, nfx, nfy, nfz) "G must be (35,nx+2g,ny+2g,nz+2g)"
    @assert size(R) == (35, nx, ny, nz) "R must be (35,nx,ny,nz)"
    Maf = Float64(Ma); s3f = Float64(s3max)
    dxf = Float64(dx); dyf = Float64(dy); dzf = Float64(dz)
    λx = Float64(dt) / dxf; λy = Float64(dt) / dyf; λz = Float64(dt) / dzf

    # ws === nothing keeps the original per-call allocation, so every existing caller is
    # byte-identical and unchanged. With a scratch supplied the same buffers are reused, zeroed
    # on entry so the kernels see exactly what CUDA.zeros would have handed them.
    if ws === nothing
        P = CUDA.zeros(Float64, 35, nfx, nfy, nfz)   # recon-var point scratch (reused per axis)
        V = CUDA.zeros(Float64, 35, nfx, nfy, nfz)   # recon-var average scratch (reused per axis)
        FHOx = CUDA.zeros(Float64, 35, nx+1, ny, nz); FLOx = CUDA.zeros(Float64, 35, nx+1, ny, nz)
        FHOy = CUDA.zeros(Float64, 35, nx, ny+1, nz); FLOy = CUDA.zeros(Float64, 35, nx, ny+1, nz)
        FHOz = CUDA.zeros(Float64, 35, nx, ny, nz+1); FLOz = CUDA.zeros(Float64, 35, nx, ny, nz+1)
        Th   = CUDA.zeros(Float64, 6, nx, ny, nz)
    else
        ws.dims == (nx, ny, nz, g) ||
            error("Order3Scratch was built for $(ws.dims), called with $((nx, ny, nz, g))")
        _reset!(ws)
        P = ws.P; V = ws.V
        FHOx = ws.FHOx; FLOx = ws.FLOx
        FHOy = ws.FHOy; FLOy = ws.FLOy
        FHOz = ws.FHOz; FLOz = ws.FLOz
        Th   = ws.Th
    end

    ncube = nfx * nfy * nfz
    bcube = cld(ncube, threads)
    fx = (nx+1)*ny*nz; fy = (ny+1)*nx*nz; fz = (nz+1)*nx*ny
    bint = cld(nx*ny*nz, threads)

    # marginal-J scratch (opt-in log-Jacobi); PJ/VJ/OKc reused per axis
    if use_logjacobi_recon && !first_order
        PJ  = CUDA.zeros(Float64, 5, nfx, nfy, nfz)
        VJ  = CUDA.zeros(Float64, 5, nfx, nfy, nfz)
        OKc = CUDA.zeros(Float64, nfx, nfy, nfz)
    end

    if first_order
        # --- First-order HLL (order=1): no reconstruction, no θ*. Each flux kernel sets
        # FHO=FLO=HLL(cell,cell) (shares _hll_states), and _blend_residual! with Th≡0
        # (already zeroed) yields F=FLO. DRY: reuses the same flux + residual machinery. ---
        active[1] && @cuda threads=threads blocks=cld(fx, threads) _weno_flux_x!(FHOx, FLOx, G, V, nx, ny, nz, g, nfx, Maf, s3f, true)
        active[2] && @cuda threads=threads blocks=cld(fy, threads) _weno_flux_y!(FHOy, FLOy, G, V, nx, ny, nz, g, nfy, Maf, s3f, true)
        active[3] && @cuda threads=threads blocks=cld(fz, threads) _weno_flux_z!(FHOz, FLOz, G, V, nx, ny, nz, g, nfz, Maf, s3f, true)
    else
    # --- Pass 1: per axis Ppt → Vavg → faces (raw); log-Jacobi marginal override opt-in ---
    if active[1]
        # only the interior extent in y and z is ever read by _weno_flux_x!
        nxb = cld(nfx*ny*nz, threads)
        @cuda threads=threads blocks=nxb _ppt_x!(P, G, nfx, nfy, nfz, g+1, ny, g+1, nz)
        @cuda threads=threads blocks=nxb _vavg_x!(V, P, nfx, nfy, nfz, g+1, ny, g+1, nz)
        if use_logjacobi_recon
            OKLx = CUDA.zeros(Float64, nfy, nfz)
            @cuda threads=threads blocks=bcube _ppt_marg_x!(PJ, OKc, G, _MARG_IDX[1], nfx, nfy, nfz)
            @cuda threads=threads blocks=bcube _vavg_marg_x!(VJ, PJ, nfx, nfy, nfz)
            @cuda threads=threads blocks=cld(nfy*nfz, threads) _okline_x!(OKLx, OKc, nfx, nfy, nfz)
            @cuda threads=threads blocks=cld(fx, threads) _weno_flux_lj_x!(FHOx, FLOx, G, V, VJ, OKLx, _MARG_IDX[1], nx, ny, nz, g, nfx, Maf, s3f)
        else
            @cuda threads=threads blocks=cld(fx, threads) _weno_flux_x!(FHOx, FLOx, G, V, nx, ny, nz, g, nfx, Maf, s3f, false)
        end
    end

    if active[2]
        # only the interior extent in x and z is ever read by _weno_flux_y!
        nyb = cld(nx*nfy*nz, threads)
        @cuda threads=threads blocks=nyb _ppt_y!(P, G, nfx, nfy, nfz, g+1, nx, g+1, nz)
        @cuda threads=threads blocks=nyb _vavg_y!(V, P, nfx, nfy, nfz, g+1, nx, g+1, nz)
        if use_logjacobi_recon
            OKLy = CUDA.zeros(Float64, nfx, nfz)
            @cuda threads=threads blocks=bcube _ppt_marg_y!(PJ, OKc, G, _MARG_IDX[2], nfx, nfy, nfz)
            @cuda threads=threads blocks=bcube _vavg_marg_y!(VJ, PJ, nfx, nfy, nfz)
            @cuda threads=threads blocks=cld(nfx*nfz, threads) _okline_y!(OKLy, OKc, nfx, nfy, nfz)
            @cuda threads=threads blocks=cld(fy, threads) _weno_flux_lj_y!(FHOy, FLOy, G, V, VJ, OKLy, _MARG_IDX[2], nx, ny, nz, g, nfy, Maf, s3f)
        else
            @cuda threads=threads blocks=cld(fy, threads) _weno_flux_y!(FHOy, FLOy, G, V, nx, ny, nz, g, nfy, Maf, s3f, false)
        end
    end

    if active[3]
        # only the interior extent in x and y is ever read by _weno_flux_z!
        nzb = cld(nx*ny*nfz, threads)
        @cuda threads=threads blocks=nzb _ppt_z!(P, G, nfx, nfy, nfz, g+1, nx, g+1, ny)
        @cuda threads=threads blocks=nzb _vavg_z!(V, P, nfx, nfy, nfz, g+1, nx, g+1, ny)
        if use_logjacobi_recon
            OKLz = CUDA.zeros(Float64, nfx, nfy)
            @cuda threads=threads blocks=bcube _ppt_marg_z!(PJ, OKc, G, _MARG_IDX[3], nfx, nfy, nfz)
            @cuda threads=threads blocks=bcube _vavg_marg_z!(VJ, PJ, nfx, nfy, nfz)
            @cuda threads=threads blocks=cld(nfx*nfy, threads) _okline_z!(OKLz, OKc, nfx, nfy, nfz)
            @cuda threads=threads blocks=cld(fz, threads) _weno_flux_lj_z!(FHOz, FLOz, G, V, VJ, OKLz, _MARG_IDX[3], nx, ny, nz, g, nfz, Maf, s3f)
        else
            @cuda threads=threads blocks=cld(fz, threads) _weno_flux_z!(FHOz, FLOz, G, V, nx, ny, nz, g, nfz, Maf, s3f, false)
        end
    end
    end  # if first_order

    # --- Pass 2: θ* per cell (skipped for first-order: Th≡0), then blend + residual ---
    if !first_order
        if idp
            @cuda threads=threads blocks=bint _theta_cell!(Th, G, FHOx, FLOx, FHOy, FLOy, FHOz, FLOz,
                                                       nx, ny, nz, g, λx, λy, λz, theta_closed)
        else
            fill!(Th, 1.0)      # theta == 1 => F = FHO, pure WENO5, no IDP blend
        end
    end
    @cuda threads=threads blocks=bint _blend_residual!(R, Th, FHOx, FLOx, FHOy, FLOy, FHOz, FLOz,
                                                       nx, ny, nz, dxf, dyf, dzf,
                                                       G, g, λx, λy, λz, Maf, s3f,
                                                       Bool(rank_bnd.xlo), Bool(rank_bnd.xhi),
                                                       Bool(rank_bnd.ylo), Bool(rank_bnd.yhi),
                                                       Bool(rank_bnd.zlo), Bool(rank_bnd.zhi),
                                                       theta_closed,
                                                       Bool(kfvs_walls[1]), Bool(kfvs_walls[2]),
                                                       Bool(kfvs_walls[3]), Bool(kfvs_walls[4]),
                                                       Bool(kfvs_walls[5]), Bool(kfvs_walls[6]),
                                                       wall_Tw, wall_Tw_hi, wall_uw1, wall_uw2,
                                                       wall_antisym)
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
                               use_logjacobi_recon::Bool = false)
    Gd = CuArray(G_host)
    R  = CUDA.zeros(Float64, 35, nx, ny, nz)
    residual3d_order3_box_gpu!(R, Gd, nx, ny, nz, g, dx, dy, dz, Ma, dt; s3max=s3max, threads=threads, theta_closed=theta_closed, use_logjacobi_recon=use_logjacobi_recon)
    CUDA.synchronize()
    return Array(R)
end

end # module

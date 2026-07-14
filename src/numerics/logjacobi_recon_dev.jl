"""
    logjacobi_recon_dev.jl — OPT-IN log-Jacobi (J-chart) reconstruction of the
    per-axis MARGINAL moment chain, for the order-3 WENO5 path.

Contact-fidelity + order upgrade to the marginal reconstruction ONLY. This is
NOT a realizability fix: the 35-moment CROSS constraints have no 1D J-chart, so
cross moments reconstruct raw exactly as before, and the existing realizability
enforcement (θ*-IDP scaling + anchor/projection) is untouched. See the de-risk
report (roe-writeup-scripts/LOGJACOBI_DERISK_REPORT.md).

The face-normal marginal (m0..m4 of the axis) is a univariate moment sequence.
Its Jacobi recurrence coordinates
    J = (log m0, a1, log b2, a2, log b3)
are a bijection onto R^5 with realizability = positivity of b2,b3 (built in by
the log parametrization). At a uniform-p contact log p = log m0 + log b2 is
reconstructed EXACTLY constant (equal WENO smoothness indicators), so the
marginal contact is captured with ~machine accuracy instead of smeared.

Faithful port of roe1d.jl's :weno5j pipeline (the one that measured order 4.99
and contacts 4e-16): deconv5 (smoothness-gated) → m→J pointwise → conv5 in J →
WENO5-Z in J → J→m. The deconvolution is REQUIRED — a bare map-then-WENO caps at
2nd order (the nonlinear change-of-variables barrier). Verified 5.00/5.01 in 1D.

Pure NTuple / scalar Float64 arithmetic (no allocation, no throw): the 5-element
marginal maps are device-safe (log/exp are hardware intrinsics on the GPUs), so
this module is a straight CPU/GPU single source like hiorder3_recon_dev.jl. The
GPU port consumes these same maps.
"""
module LogJacobiReconDev

include(joinpath(@__DIR__, "weno5_dev.jl")); using .Weno5Dev: weno5z, deconv5, conv5, smooth5

export marg_m_to_J, marg_J_to_m, logjacobi_marginal_faces, affine_remap_axis, _affine_remap

# ---------------------------------------------------------------------------
# Realizability-safe marginal override via an AFFINE velocity remap.
#
# The naive override (swap the 5 axis-marginal raw moments, keep the 30 others)
# breaks joint realizability: a sharper marginal variance leaves the cross moments
# oversized, the correlation exceeds 1, and the wave-speed eig NaNs (root-caused
# 2026-07-13). Instead we apply log-J's marginal as an affine map of the axis
# velocity, u -> beta + alpha*u (alpha = sigma_new/sigma_raw), matching log-J's
# mean+variance while transforming EVERY moment (cross included) consistently.
# Realizability is invariant under an affine change of one velocity variable, so
# the result is realizable by construction (no gate). Density is matched by a
# uniform scale gamma. Skew/kurtosis stay at the raw (scaled) values (only 2 DOF
# in an affine map); the measured log-J fidelity gain is 2nd-order, so this keeps it.
#
# M'_{ijk} = gamma * sum_{p=0}^{e} C(e,p) alpha^p beta^{e-p} M_{sib(p)}   (e = axis power)
# ---------------------------------------------------------------------------
const _IJK35 = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
                (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
                (0,3,0),(1,3,0),(0,4,0),
                (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
                (0,0,3),(1,0,3),(0,0,4),
                (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
                (0,1,2),(1,1,2),(0,1,3),(0,2,2))
const _MARG_LJ = ((1,2,3,4,5),(1,6,10,13,15),(1,16,20,23,25))  # marginal slots per axis
# per-axis: axis power e_q, and sibling slots for p=0..e padded to length 5 (pad=1)
const _AXPOW, _SIB = let
    slot = Dict(t=>q for (q,t) in enumerate(_IJK35))
    axpow = ntuple(ax->ntuple(q->_IJK35[q][ax], 35), 3)
    sib = ntuple(ax->ntuple(q->begin
            e = _IJK35[q][ax]
            ntuple(pp->begin p = pp-1
                p <= e ? slot[ntuple(d->d==ax ? p : _IJK35[q][d], 3)] : 1
            end, 5)
        end, 35), 3)
    (axpow, sib)
end
# branch-based binomial (device-safe: no runtime tuple indexing) for e,p <= 4
@inline _binom4(e::Int, p::Int) =
    e == 1 ? 1.0 :
    e == 2 ? (p == 1 ? 2.0 : 1.0) :
    e == 3 ? (p == 1 || p == 2 ? 3.0 : 1.0) :
    e == 4 ? (p == 2 ? 6.0 : (p == 1 || p == 3 ? 4.0 : 1.0)) : 1.0
# exact integer power by repeated multiply (identical CPU/GPU; avoids the runtime
# `x^p` -> pow(x, Float64(p)) = exp(p*log x) path that diverges ~1e-7 on device)
@inline function _ipow(x::Float64, p::Int)
    r = 1.0
    @inbounds for _ in 1:p
        r *= x
    end
    r
end

# Device-safe core: AX is a compile-time axis (Val), so every table lookup and the
# per-slot loop bound fold to constants (required for GPU). Remaps the full 35-moment
# state `m` so its AX-marginal has (rho_new,u_new,var_new) via the affine velocity map.
# Guards: degenerate raw/target marginal (rho<=0, var<=0) -> return `m` unchanged.
@inline function _affine_remap(m::NTuple{35,Float64}, ::Val{AX},
                               rho_new::Float64, u_new::Float64, var_new::Float64) where {AX}
    mg = _MARG_LJ[AX]
    rho = m[mg[1]]
    (isfinite(rho) && rho > 0.0) || return m
    u_raw   = m[mg[2]] / rho
    var_raw = m[mg[3]] / rho - u_raw * u_raw
    (var_raw > 0.0 && var_new > 0.0 && rho_new > 0.0 && isfinite(var_new) && isfinite(u_new)) || return m
    α = sqrt(var_new / var_raw)
    β = u_new - α * u_raw
    γ = rho_new / rho
    axp = _AXPOW[AX]; sibs = _SIB[AX]
    ntuple(Val(35)) do q
        @inbounds e = axp[q]
        @inbounds sib = sibs[q]
        acc = 0.0
        @inbounds for pp in 0:e
            acc += _binom4(e, pp) * _ipow(α, pp) * _ipow(β, e - pp) * m[sib[pp+1]]
        end
        γ * acc
    end
end

# CPU convenience: runtime axis (branch into the compile-time core; keeps device parity).
@inline affine_remap_axis(m::NTuple{35,Float64}, ax::Int, rn::Float64, un::Float64, vn::Float64) =
    ax == 1 ? _affine_remap(m, Val(1), rn, un, vn) :
    ax == 2 ? _affine_remap(m, Val(2), rn, un, vn) :
              _affine_remap(m, Val(3), rn, un, vn)

# ---------------------------------------------------------------------------
# m -> J on a 5-element marginal (m0,m1,m2,m3,m4). Returns (ok, J::NTuple{5}).
# ok=false (leave marginal to the raw path) iff m0<=0 or b2<=0 or b3<=0 or any
# non-finite — i.e. the marginal itself is off the 1D cone. Device-safe: no throw.
# Central-moment recurrence identical to roe1d.rec_ab / the production closure.
# ---------------------------------------------------------------------------
@inline function marg_m_to_J(m0::Float64, m1::Float64, m2::Float64, m3::Float64, m4::Float64)
    if !(m0 > 0.0) || !isfinite(m0)
        return (false, (0.0, 0.0, 0.0, 0.0, 0.0))
    end
    a1 = m1 / m0
    c2 = m2 / m0 - a1 * a1
    c3 = m3 / m0 - 3.0 * a1 * (m2 / m0) + 2.0 * a1^3
    c4 = m4 / m0 - 4.0 * a1 * (m3 / m0) + 6.0 * a1 * a1 * (m2 / m0) - 3.0 * a1^4
    b2 = c2
    if !(b2 > 0.0)
        return (false, (0.0, 0.0, 0.0, 0.0, 0.0))
    end
    a2 = a1 + c3 / c2
    b3 = (c4 - c3 * c3 / c2) / c2 - c2
    if !(b3 > 0.0) || !isfinite(a1) || !isfinite(a2) || !isfinite(b3)
        return (false, (0.0, 0.0, 0.0, 0.0, 0.0))
    end
    (true, (log(m0), a1, log(b2), a2, log(b3)))
end

# ---------------------------------------------------------------------------
# J -> m: inverse map (verbatim from roe1d.J_to_m). Always realizable by
# construction (b2=exp>0, b3=exp>0). Returns the 5 raw marginal moments.
# ---------------------------------------------------------------------------
@inline function marg_J_to_m(J::NTuple{5,Float64})
    rho = exp(J[1]); u = J[2]; b2 = exp(J[3]); a2 = J[4]; b3 = exp(J[5])
    c2 = b2
    c3 = b2 * (a2 - u)
    c4 = b2 * b3 + b2 * b2 + b2 * (a2 - u)^2
    (rho,
     rho * u,
     rho * (c2 + u * u),
     rho * (c3 + 3.0 * u * c2 + u^3),
     rho * (c4 + 4.0 * u * c3 + 6.0 * u * u * c2 + u^4))
end

# scalar helper: apply the full deconv→(handled by caller)… see below. The J
# pipeline needs to interleave the per-cell m→J (a JOINT nonlinear map of the 5
# marginal slots) with the per-component deconv5/conv5/weno5z. We therefore work
# in three passes over the line, exactly mirroring residual_line3's structure.

# ---------------------------------------------------------------------------
# logjacobi_marginal_faces — full :weno5j marginal pipeline for ONE axis line.
#
# Inputs (all length n2g, indexed 1..n2g including g ghosts each end):
#   Mmarg :: Vector{NTuple{5,Float64}}   the 5 raw marginal moments per cell
#            (extracted at MARG_IDX[axis] by the caller).
#   g     :: Int  ghost count (>=4).
# Output:
#   (okline, Lface, Rface) where
#     okline :: Bool — false if ANY cell's marginal left the 1D cone (then the
#               caller keeps the raw-moment marginal reconstruction for the whole
#               line, a conservative all-or-nothing fallback like roe1d's ok_all).
#     Lface, Rface :: Vector{NTuple{5,Float64}} length n+1 — the reconstructed
#               LEFT-of-interface and RIGHT-of-interface raw marginal moments at
#               each of the n+1 interior-bounding interfaces (f=1..n+1).
#
# Pipeline (per component of J, componentwise WENO on J cell-averages):
#   1. per cell: deconv5-gated raw marginals -> point marginals -> m_to_J (point).
#   2. conv5 the J point values back to J cell-averages (per J component).
#   3. WENO5-Z the J averages to L/R faces; J_to_m back to raw marginals.
# Steps 1-2 mirror residual_line3 Steps 1-2 exactly (same deconv5/conv5, same
# boundary clamp), just carrying the 5-vector J instead of the 35-vector recon.
# ---------------------------------------------------------------------------
function logjacobi_marginal_faces(Mmarg::Vector{NTuple{5,Float64}}, g::Int)
    n2g = length(Mmarg)
    n = n2g - 2g

    # Step 1: deconv5-gated marginal point values, then map to J pointwise.
    Jpt = Vector{NTuple{5,Float64}}(undef, n2g)
    @inbounds for k in 1:n2g
        # per-component smooth5-gated deconv (mirror recon_point_dev's gate)
        mpt = if k >= 3 && k <= n2g - 2
            cm2 = Mmarg[k-2]; cm1 = Mmarg[k-1]; c0 = Mmarg[k]; cp1 = Mmarg[k+1]; cp2 = Mmarg[k+2]
            ntuple(q -> smooth5(cm2[q], cm1[q], c0[q], cp1[q], cp2[q]) ?
                        deconv5(cm2[q], cm1[q], c0[q], cp1[q], cp2[q]) : c0[q], Val(5))
        else
            Mmarg[k]                               # boundary/ghost: cell average
        end
        ok, J = marg_m_to_J(mpt[1], mpt[2], mpt[3], mpt[4], mpt[5])
        ok || return (false, NTuple{5,Float64}[], NTuple{5,Float64}[])
        Jpt[k] = J
    end

    # Step 2: forward-convolve J point values -> J cell-averages (per component).
    _jp(k) = @inbounds Jpt[clamp(k, 1, n2g)]
    Javg = Vector{NTuple{5,Float64}}(undef, n2g)
    @inbounds for k in 1:n2g
        pm2 = _jp(k-2); pm1 = _jp(k-1); p0 = _jp(k); pp1 = _jp(k+1); pp2 = _jp(k+2)
        Javg[k] = ntuple(q -> conv5(pm2[q], pm1[q], p0[q], pp1[q], pp2[q]), Val(5))
    end

    # Step 3: WENO5-Z in J to L/R faces, then J->m back to raw marginals.
    _jv(k) = @inbounds Javg[clamp(k, 1, n2g)]
    Lface = Vector{NTuple{5,Float64}}(undef, n + 1)
    Rface = Vector{NTuple{5,Float64}}(undef, n + 1)
    @inbounds for f in 1:n+1
        il = g + f - 1                              # left cell of interface f
        W1 = _jv(il-2); W2 = _jv(il-1); W3 = _jv(il); W4 = _jv(il+1); W5 = _jv(il+2); W6 = _jv(il+3)
        vL = ntuple(q -> weno5z(W1[q], W2[q], W3[q], W4[q], W5[q]), Val(5))   # right face of il
        vR = ntuple(q -> weno5z(W6[q], W5[q], W4[q], W3[q], W2[q]), Val(5))   # left  face of ir
        Lface[f] = marg_J_to_m(vL)
        Rface[f] = marg_J_to_m(vR)
    end
    (true, Lface, Rface)
end

end # module

"""
    roeps3_dev.jl — parity-split Roe flux (RoePS3), single-source CPU/GPU.

The RoePS3 dissipation of the 1D study (private research notes, July 2026),
lifted to the 35-moment system per direction:

  * the 5-moment DIRECTIONAL MARGINAL block gets the full 1D RoePS3 treatment:
    wave-resolved |λ| (Harten–Hyman-fixed) dissipation on the reflection-ODD
    part of the marginal jump, constant q(u) on the EVEN part, with the
    closed-form marginal spectrum (Q₂ quadratic + perturbed-R₃ symmetric-
    tridiagonal cubic) and standardized-frame Vandermonde solves;
  * the remaining 30 components get the scalar sector coefficients by the
    face-normal reflection parity of M_ijk (odd normal exponent → Rusanov a,
    even → constant q(u)).

Properties (validated in the 1D reference solver): uniform-pressure contacts
are preserved exactly at any contact speed (constant even-sector coefficient —
the parity theorem); shocks gain over HLL; a spectral-gap guard falls back to
scalar coefficients near the realizability boundary, where the marginal system
is weakly hyperbolic (eigenvalue gap ≈ 0.7·c₂·H).

Everything here is plain, allocation-light Julia (MVector workspaces), safe to
call from CUDA kernels and from the CPU flux path — the single source for both.
"""
module RoePS3Dev

using StaticArrays

export roeps3_diss_dev, MARG_IDX, ODD_MASK

# ---------------------------------------------------------------------------
# canonical exponent table (i,j,k) for the 35 moments — single source for the
# parity masks and marginal index sets.
# ---------------------------------------------------------------------------
const IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
             (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
             (0,3,0),(1,3,0),(0,4,0),
             (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
             (0,0,3),(1,0,3),(0,0,4),
             (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
             (0,1,2),(1,1,2),(0,1,3),(0,2,2))

# per-axis: is the face-normal exponent odd? (reflection parity of M_ijk)
const ODD_MASK = (ntuple(q -> isodd(IJK[q][1]), 35),
                  ntuple(q -> isodd(IJK[q][2]), 35),
                  ntuple(q -> isodd(IJK[q][3]), 35))

# per-axis marginal moment indices (m0..m4 of the face-normal marginal chain)
const MARG_IDX = ((1, 2, 3, 4, 5), (1, 6, 10, 13, 15), (1, 16, 20, 23, 25))

# ---------------------------------------------------------------------------
# closed-form spectrum of the 1D marginal Jacobian: Q2 roots (quadratic) plus
# perturbed-R3 roots (symmetric tridiagonal 3x3, trigonometric cubic — three
# real roots guaranteed). Recurrence identical to `closure5_dev`, including
# Fox's b3 floor.
# ---------------------------------------------------------------------------
@inline function _tridiag3_eigs(a1, b2, a2, b3, a3)
    # eigenvalues of [a1 √b2 0; √b2 a2 √b3; 0 √b3 a3] via the trig cubic
    p1 = b2 + b3
    qm = (a1 + a2 + a3) / 3
    p2 = (a1 - qm)^2 + (a2 - qm)^2 + (a3 - qm)^2 + 2 * p1
    p = sqrt(max(p2 / 6, 1e-300))
    # det(B) with B = (T − qm·I)/p
    da1 = (a1 - qm) / p; da2 = (a2 - qm) / p; da3 = (a3 - qm) / p
    r = (da1 * (da2 * da3 - b3 / p^2) - (b2 / p^2) * da3) / 2
    r = clamp(r, -1.0, 1.0)
    phi = acos(r) / 3
    e1 = qm + 2p * cos(phi)
    e3 = qm + 2p * cos(phi + 2π / 3)
    e2 = 3qm - e1 - e3
    # ascending
    lo = min(e1, min(e2, e3)); hi = max(e1, max(e2, e3))
    mid = e1 + e2 + e3 - lo - hi
    return lo, mid, hi
end

@inline function _marg_eigen5(w1, w2, w3, w4, w5)
    # recurrence (as closure5_dev), returns 5 ascending eigenvalues + (u, σ)
    a1 = w2 / w1
    s33 = w3 - a1 * w2
    s34 = w4 - a1 * w3
    s35 = w5 - a1 * w4
    a2 = s34 / s33 - w2 / w1
    b2 = s33 / w1
    s44 = s35 - a2 * s34 - b2 * w3
    b3 = s44 / s33
    if b3 < 0.0
        b3 = 1.0e-10
    end
    a3 = (a1 + a2) / 2
    b3p = b3 * 5.0 / 2.0
    # Q2 pair
    hm = (a1 + a2) / 2
    hd = sqrt(max(((a1 - a2) / 2)^2 + b2, 0.0))
    q1 = hm - hd; q2 = hm + hd
    r1, r2, r3 = _tridiag3_eigs(a1, b2, a2, b3p, a3)
    # 5-sorted merge (both lists ascending)
    l = MVector{5,Float64}(q1, q2, r1, r2, r3)
    # tiny insertion sort
    @inbounds for i in 2:5
        x = l[i]; j = i - 1
        while j >= 1 && l[j] > x
            l[j+1] = l[j]; j -= 1
        end
        l[j+1] = x
    end
    sig = sqrt(max(b2, 1e-300))
    return l[1], l[2], l[3], l[4], l[5], a1, sig
end

# ---------------------------------------------------------------------------
# Björck–Pereyra: solve V(x) α = y for Vandermonde V_{kj} = x_j^k, O(n²),
# division-free of any pivoting — exact for distinct nodes. n = 5, in registers.
# ---------------------------------------------------------------------------
@inline function _vandermonde_solve5!(α::MVector{5,Float64}, x::MVector{5,Float64})
    # primal Björck–Pereyra (Golub & Van Loan Alg. 4.6.2): solves V α = y for
    # V_{kj} = x_j^{k-1} in place, O(n²), exact for distinct nodes.
    @inbounds begin
        for k in 1:4, i in 5:-1:(k+1)
            α[i] = α[i] - x[k] * α[i-1]
        end
        for k in 4:-1:1
            for i in (k+1):5
                α[i] = α[i] / (x[i] - x[i-k])
            end
            for i in k:4
                α[i] = α[i] - α[i+1]
            end
        end
    end
    return α
end

@inline _hh_dev(l, del) = abs(l) < del ? (l * l + del * del) / (2 * del) : abs(l)

# ---------------------------------------------------------------------------
# marginal-block RoePS3 dissipation: returns the 5 dissipation components for
# the face-normal marginal chain, or `nothing`-signal (via ok=false) when the
# spectrum is (near-)defective — caller falls back to scalar coefficients.
# wL/wR are the marginal 5-vectors of the two face states.
# ---------------------------------------------------------------------------
@inline function _marg_roeps3(wL1, wL2, wL3, wL4, wL5, wR1, wR2, wR3, wR4, wR5,
                              qu, sl, sr; gap_tol = 1e-6, entropy_delta = 0.05)
    wb1 = (wL1 + wR1) / 2; wb2 = (wL2 + wR2) / 2; wb3 = (wL3 + wR3) / 2
    wb4 = (wL4 + wR4) / 2; wb5 = (wL5 + wR5) / 2
    (wb1 > 0 && wb3 > 0) || return false, 0.0, 0.0, 0.0, 0.0, 0.0
    l1, l2, l3, l4, l5, u, sig = _marg_eigen5(wb1, wb2, wb3, wb4, wb5)
    (isfinite(l1) && isfinite(l5) && sig > 0) || return false, 0.0, 0.0, 0.0, 0.0, 0.0
    spread = l5 - l1
    gap = min(min(l2 - l1, l3 - l2), min(l4 - l3, l5 - l4))
    (spread > 0 && gap > gap_tol * spread) || return false, 0.0, 0.0, 0.0, 0.0, 0.0
    # standardized central jump: ŵ_k = Σ_j C(k,j)(−u)^{k−j} Δw_j / σ^k
    d1 = wR1 - wL1; d2 = wR2 - wL2; d3 = wR3 - wL3; d4 = wR4 - wL4; d5 = wR5 - wL5
    s2 = sig * sig; s3 = s2 * sig; s4 = s3 * sig
    h0 = d1
    h1 = (d2 - u * d1) / sig
    h2 = (d3 - 2u * d2 + u^2 * d1) / s2
    h3 = (d4 - 3u * d3 + 3u^2 * d2 - u^3 * d1) / s3
    h4 = (d5 - 4u * d4 + 6u^2 * d3 - 4u^3 * d2 + u^4 * d1) / s4
    # parity split in the standardized central frame
    xh = MVector{5,Float64}((l1 - u) / sig, (l2 - u) / sig, (l3 - u) / sig,
                            (l4 - u) / sig, (l5 - u) / sig)
    aodd = MVector{5,Float64}(0.0, h1, 0.0, h3, 0.0)
    aeven = MVector{5,Float64}(h0, 0.0, h2, 0.0, h4)
    _vandermonde_solve5!(aodd, xh)
    _vandermonde_solve5!(aeven, xh)
    del = entropy_delta * sig * (xh[5] - xh[1])
    # D̂ = Σ_i (codd(λ_i) α_odd,i + q(u) α_even,i) · Vandermonde column(x̂_i)
    D0 = 0.0; D1 = 0.0; D2 = 0.0; D3 = 0.0; D4 = 0.0
    @inbounds for i in 1:5
        li = u + sig * xh[i]
        ci = _hh_dev(li, del) * aodd[i] + qu * aeven[i]
        x1 = xh[i]; x2 = x1 * x1
        D0 += ci
        D1 += ci * x1
        D2 += ci * x2
        D3 += ci * x2 * x1
        D4 += ci * x2 * x2
    end
    # back to raw moments: D_k = σ^k Σ_j C(k,j) u^{k−j} D̂_j
    r0 = D0
    r1 = u * D0 + sig * D1
    r2 = u^2 * D0 + 2u * sig * D1 + s2 * D2
    r3 = u^3 * D0 + 3u^2 * sig * D1 + 3u * s2 * D2 + s3 * D3
    r4 = u^4 * D0 + 4u^3 * sig * D1 + 6u^2 * s2 * D2 + 4u * s3 * D3 + s4 * D4
    return true, r0, r1, r2, r3, r4
end

"""
    roeps3_diss_dev(mL::NTuple{35}, mR::NTuple{35}, axis, sl, sr) -> NTuple{35}

The RoePS3 dissipation vector D such that F = (F_L + F_R)/2 − D/2. `sl`, `sr`
are the HLL wave-speed bounds already computed by the flux path.
"""
@inline function roeps3_diss_dev(mL::NTuple{35,Float64}, mR::NTuple{35,Float64},
                                 axis::Int, sl::Float64, sr::Float64)
    idx = MARG_IDX[axis]
    odd = ODD_MASK[axis]
    ubar = (mL[idx[2]] / mL[1] + mR[idx[2]] / mR[1]) / 2
    dsr = sr - sl
    qu = dsr > 0 ? ((sr + sl) * clamp(ubar, sl, sr) - 2 * sl * sr) / dsr :
                   max(abs(sl), abs(sr))
    a = max(abs(sl), abs(sr))
    ok, r0, r1, r2, r3, r4 = _marg_roeps3(
        mL[idx[1]], mL[idx[2]], mL[idx[3]], mL[idx[4]], mL[idx[5]],
        mR[idx[1]], mR[idx[2]], mR[idx[3]], mR[idx[4]], mR[idx[5]], qu, sl, sr)
    return ntuple(Val(35)) do q
        if ok && q == idx[1]
            r0
        elseif ok && q == idx[2]
            r1
        elseif ok && q == idx[3]
            r2
        elseif ok && q == idx[4]
            r3
        elseif ok && q == idx[5]
            r4
        else
            (odd[q] ? a : qu) * (mR[q] - mL[q])
        end
    end
end

end # module

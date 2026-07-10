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
using .RiemannFluxDev.RoePS3Dev.MomentIndices: MARG_IDX

export theta_star_update_dev, theta_star_update_closed

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

# ---------------------------------------------------------------------------
# Closed-form θ* (OPT-IN — default path above stays bisection, byte-identical).
#
# `_state_realizable` checks ρ>0 plus, for each of the three axes, the marginal
# Hamburger margin K − 1 − q̂² > 0 on the 5-moment chain (m0..m4). That margin is
# — up to a strictly positive factor — the determinant of the 3×3 raw-moment
# Hankel matrix
#       Hk = [ m0 m1 m2 ; m1 m2 m3 ; m2 m3 m4 ].
# Positive-definiteness of Hk is exactly marginal realizability (Sylvester):
#   D1 = m0 > 0,  D2 = m0 m2 − m1² > 0  (i.e. σ² > 0),  D3 = det Hk > 0
#   ( D3 = m0³·c2²·(K − 1 − q̂²) with c2 = σ², so sign(D3)=sign(K−1−q̂²) when
#     D1,D2>0 — the same test _marg_shape/_state_realizable applies, no floor. )
#
# Along the update line the marginal moments are affine in θ, so each leading
# minor is a polynomial in θ: D1 linear, D2 quadratic, D3 cubic. Starting PD at
# θ=0 (Mlo realizable), θ* is the smallest θ∈(0,1] at which any minor of any axis
# (or the global ρ>0 bound) first reaches 0; if none binds in (0,1], θ*=1.
#
# Device-safe: no allocations, no closures, plain scalar arithmetic + NTuple
# reads. Mirrors _marg_shape/MARG_IDX so it reads the identical marginal slots.
# ---------------------------------------------------------------------------

# Smallest root strictly in (0, tcap] of the affine-pencil minor a·θ² + b·θ + c,
# starting positive at θ=0 (c>0). Returns tcap if the minor stays > 0 on (0,tcap].
# Handles the degenerate near-linear case (|a| tiny) robustly.
@inline function _first_root_quad(a::Float64, b::Float64, c::Float64, tcap::Float64)
    # c = value at θ=0; if already ≤0 the state is (numerically) on/over the
    # boundary — bind immediately.
    c <= 0.0 && return 0.0
    # Effectively linear: a·θ² term negligible relative to b·θ.
    if abs(a) <= 1e-300 || abs(a) * tcap <= eps(Float64) * (abs(b) + abs(c))
        # b·θ + c, c>0: root at −c/b only if b<0.
        if b < 0.0
            t = -c / b
            return (t > 0.0 && t <= tcap) ? t : tcap
        end
        return tcap
    end
    disc = b * b - 4.0 * a * c
    disc < 0.0 && return tcap        # no real root ⇒ minor never hits 0 (a>0 arc)
    sq = sqrt(disc)
    # Numerically stable roots (avoid cancellation): q = −(b + sign(b)·sq)/2.
    q = b >= 0.0 ? -0.5 * (b + sq) : -0.5 * (b - sq)
    r1 = q / a
    r2 = (q == 0.0) ? Inf : c / q
    # smallest strictly-positive root ≤ tcap
    t = tcap
    (r1 > 0.0 && r1 < t) && (t = r1)
    (r2 > 0.0 && r2 < t) && (t = r2)
    return t
end

# One Newton step from t on p(θ)=((k3 θ+k2) θ+k1) θ+k0, guarded to stay in (0,tcap].
@inline function _polish_cubic(t::Float64, k3::Float64, k2::Float64, k1::Float64, k0::Float64, tcap::Float64)
    @inbounds for _ in 1:3
        f  = ((k3 * t + k2) * t + k1) * t + k0
        df = (3.0 * k3 * t + 2.0 * k2) * t + k1
        (df == 0.0) && break
        tn = t - f / df
        (isfinite(tn) && tn > 0.0 && tn <= tcap) || break
        (abs(tn - t) <= 1e-15 * (abs(t) + 1e-300)) && (t = tn; break)
        t = tn
    end
    t
end

# Smallest real root strictly in (0, tcap] of p(θ)=k3 θ³+k2 θ²+k1 θ+k0 with
# p(0)=k0>0 (state realizable at θ=0). Closed-form: analytic cubic roots via the
# depressed-cubic trigonometric/Cardano formulas, then pick the smallest positive
# root in (0,tcap] and polish with a couple of Newton steps for full accuracy.
# Returns tcap if no such root exists (minor stays > 0 on (0,tcap]).
@inline function _first_root_cubic(k3::Float64, k2::Float64, k1::Float64, k0::Float64, tcap::Float64)
    k0 <= 0.0 && return 0.0
    # Degenerate to quadratic if leading coeff negligible.
    if abs(k3) <= eps(Float64) * (abs(k2) + abs(k1) + abs(k0))
        return _first_root_quad(k2, k1, k0, tcap)
    end
    # Monic: θ³ + a θ² + b θ + c.
    inv3 = 1.0 / k3
    a = k2 * inv3; b = k1 * inv3; c = k0 * inv3
    # Depressed cubic y³ + p y + q via θ = y − a/3.
    a3 = a / 3.0
    p = b - a * a3
    q = c - a3 * b + 2.0 * a3 * a3 * a3
    # three candidate roots (θ); default to +Inf so "no root" sorts out.
    r1 = Inf; r2 = Inf; r3 = Inf
    disc = 0.25 * q * q + (p * p * p) / 27.0
    if disc > 0.0
        # one real root (Cardano)
        sq = sqrt(disc)
        u = cbrt(-0.5 * q + sq)
        v = cbrt(-0.5 * q - sq)
        r1 = u + v - a3
    elseif p < 0.0
        # three real roots (trigonometric)
        m = 2.0 * sqrt(-p / 3.0)
        arg = 3.0 * q / (p * m)
        arg = arg > 1.0 ? 1.0 : (arg < -1.0 ? -1.0 : arg)
        θt = acos(arg) / 3.0
        twopi3 = 2.0943951023931953  # 2π/3
        r1 = m * cos(θt) - a3
        r2 = m * cos(θt - twopi3) - a3
        r3 = m * cos(θt - 2.0 * twopi3) - a3
    else
        # p ≈ 0 (and disc ≤ 0 ⇒ q ≈ 0): triple root at y=0.
        r1 = -a3
    end
    # smallest root strictly in (0, tcap]
    t = tcap
    (r1 > 0.0 && r1 < t) && (t = r1)
    (r2 > 0.0 && r2 < t) && (t = r2)
    (r3 > 0.0 && r3 < t) && (t = r3)
    (t >= tcap) && return tcap
    return _polish_cubic(t, k3, k2, k1, k0, tcap)
end

# θ-bound from one axis's marginal-chain minors. m*0 = Mlo marginal moments,
# d* = dM marginal increments (both indexed via MARG_IDX). Returns the smallest
# θ∈(0,tcap] at which any leading minor of Hk(θ)=H0+θH1 first hits 0.
@inline function _axis_theta_bound(a0::Float64, a1::Float64, a2::Float64, a3::Float64, a4::Float64,
                                   b0::Float64, b1::Float64, b2::Float64, b3::Float64, b4::Float64,
                                   tcap::Float64)
    t = tcap
    # D1 = m0(θ) = a0 + θ b0 > 0
    if a0 <= 0.0
        return 0.0
    elseif b0 < 0.0
        tb = -a0 / b0
        (tb < t) && (t = tb)
    end
    # D2(θ) = m0 m2 − m1² , with m0=a0+θb0, m1=a1+θb1, m2=a2+θb2.
    #   = (a0 a2 − a1²) + θ(a0 b2 + b0 a2 − 2 a1 b1) + θ²(b0 b2 − b1²)
    q2a = b0 * b2 - b1 * b1
    q2b = a0 * b2 + b0 * a2 - 2.0 * a1 * b1
    q2c = a0 * a2 - a1 * a1
    t2 = _first_root_quad(q2a, q2b, q2c, t)
    (t2 < t) && (t = t2)
    # D3(θ) = det Hk(θ), Hk = [[m0,m1,m2],[m1,m2,m3],[m2,m3,m4]]; cubic in θ.
    #   det = m0(m2 m4 − m3²) − m1(m1 m4 − m2 m3) + m2(m1 m3 − m2²)
    # Expand each entry mi = ai + θ bi; collect powers of θ via nested products.
    # Coefficients computed from the three affine 3-vectors of Hk rows using the
    # standard det-of-affine-pencil expansion.
    # Row entries: r1=(m0,m1,m2), r2=(m1,m2,m3), r3=(m2,m3,m4).
    # We build det(θ) coefficients by symbolic accumulation of the 3×3 determinant.
    # det = Σ over the 6 permutations; each term is a product of three affine
    # factors → cubic. Accumulate k3,k2,k1,k0.
    # Entry (row,col) affine parts:
    #   (1,1)=a0,b0 (1,2)=a1,b1 (1,3)=a2,b2
    #   (2,1)=a1,b1 (2,2)=a2,b2 (2,3)=a3,b3
    #   (3,1)=a2,b2 (3,2)=a3,b3 (3,3)=a4,b4
    k3, k2, k1, k0 = _det3_affine_coeffs(
        a0, a1, a2,
        a1, a2, a3,
        a2, a3, a4,
        b0, b1, b2,
        b1, b2, b3,
        b2, b3, b4)
    t3 = _first_root_cubic(k3, k2, k1, k0, t)
    (t3 < t) && (t = t3)
    return t
end

# Coefficients (k3 θ³ + k2 θ² + k1 θ + k0) of det(A + θB) for symmetric 3×3
# affine pencils given entrywise (A11..A33, B11..B33), row-major. Plain scalar
# arithmetic — the six signed permutation products, each a product of three
# affine factors P(θ)=(Aij+θBij), accumulated by power of θ.
@inline function _det3_affine_coeffs(A11, A12, A13, A21, A22, A23, A31, A32, A33,
                                     B11, B12, B13, B21, B22, B23, B31, B32, B33)
    # Helper: product of three affine factors (a1+θb1)(a2+θb2)(a3+θb3) → (c3,c2,c1,c0)
    # inlined per term. det = +112233 −112332 −132231... standard permutation sum:
    # det = A11(A22 A33 − A23 A32) − A12(A21 A33 − A23 A31) + A13(A21 A32 − A22 A31)
    # Build as sum of six signed triple products.
    # term (i,j,k) sign s: s * (Ai + θBi)(Aj + θBj)(Ak + θBk)
    # +(11,22,33) −(11,23,32) −(12,21,33) +(12,23,31) +(13,21,32) −(13,22,31)
    c3 = 0.0; c2 = 0.0; c1 = 0.0; c0 = 0.0
    # +(11,22,33)
    c3 += B11 * B22 * B33
    c2 += A11 * B22 * B33 + B11 * A22 * B33 + B11 * B22 * A33
    c1 += A11 * A22 * B33 + A11 * B22 * A33 + B11 * A22 * A33
    c0 += A11 * A22 * A33
    # −(11,23,32)
    c3 -= B11 * B23 * B32
    c2 -= A11 * B23 * B32 + B11 * A23 * B32 + B11 * B23 * A32
    c1 -= A11 * A23 * B32 + A11 * B23 * A32 + B11 * A23 * A32
    c0 -= A11 * A23 * A32
    # −(12,21,33)
    c3 -= B12 * B21 * B33
    c2 -= A12 * B21 * B33 + B12 * A21 * B33 + B12 * B21 * A33
    c1 -= A12 * A21 * B33 + A12 * B21 * A33 + B12 * A21 * A33
    c0 -= A12 * A21 * A33
    # +(12,23,31)
    c3 += B12 * B23 * B31
    c2 += A12 * B23 * B31 + B12 * A23 * B31 + B12 * B23 * A31
    c1 += A12 * A23 * B31 + A12 * B23 * A31 + B12 * A23 * A31
    c0 += A12 * A23 * A31
    # +(13,21,32)
    c3 += B13 * B21 * B32
    c2 += A13 * B21 * B32 + B13 * A21 * B32 + B13 * B21 * A32
    c1 += A13 * A21 * B32 + A13 * B21 * A32 + B13 * A21 * A32
    c0 += A13 * A21 * A32
    # −(13,22,31)
    c3 -= B13 * B22 * B31
    c2 -= A13 * B22 * B31 + B13 * A22 * B31 + B13 * B22 * A31
    c1 -= A13 * A22 * B31 + A13 * B22 * A31 + B13 * A22 * A31
    c0 -= A13 * A22 * A31
    return c3, c2, c1, c0
end

"""
    theta_star_update_closed(Mlo, dM) -> Float64

Closed-form replacement for `theta_star_update_dev`'s bisection: the largest
θ∈[0,1] with `Mlo + θ·dM` realizable, computed analytically from the marginal
raw-moment Hankel pencils rather than by 24 bisection realizability evals.

Realizability (per `_state_realizable`) is ρ>0 and, on each axis, the marginal
Hamburger margin K−1−q̂²>0, which equals positive-definiteness of the 3×3
raw-moment Hankel Hk=[[m0,m1,m2],[m1,m2,m3],[m2,m3,m4]]. Since the marginal
moments are affine in θ, Hk(θ)=H0+θH1 is a symmetric affine pencil, PD at θ=0
(Mlo realizable). θ* is the smallest θ∈(0,1] where any leading minor (D1 linear,
D2 quadratic, D3 cubic) or the global ρ>0 bound first reaches 0; 1.0 if none
binds. Returns 0.0 if Mlo is already non-realizable (matching bisection).

Device-safe (no allocations/closures). OPT-IN: the shipped default path remains
`theta_star_update_dev` (bisection), byte-identical. This is validated to agree
with bisection to ~1e-6 (test/verify_theta_star_closed.jl) and to never
overshoot into non-realizable territory (it is an exact lower bound).
"""
@inline function theta_star_update_closed(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64})
    # Match bisection's Mlo-not-realizable behavior: it returns 0.0.
    _state_realizable(Mlo) || return 0.0
    t = 1.0
    @inbounds for ax in 1:3
        idx = MARG_IDX[ax]
        i1 = idx[1]; i2 = idx[2]; i3 = idx[3]; i4 = idx[4]; i5 = idx[5]
        tb = _axis_theta_bound(
            Mlo[i1], Mlo[i2], Mlo[i3], Mlo[i4], Mlo[i5],
            dM[i1],  dM[i2],  dM[i3],  dM[i4],  dM[i5],
            t)
        (tb < t) && (t = tb)
    end
    return t < 0.0 ? 0.0 : t
end

end # module

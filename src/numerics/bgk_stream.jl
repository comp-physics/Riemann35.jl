"""
    bgk_stream.jl — two-stream BGK coupling (3D upgrade of `bgk_step!`).

The ONLY coupling between the `+` and `-` streams. Per cell:

  * total M = M⁺ + M⁺ … i.e. M⁺ + M⁻ (element-wise); get (ρ, u⃗, Θ) of the total;
  * build the SHARED Maxwellian and split it into the two half-space (v_x ≷ c)
    targets with `split_maxwellian35`;
  * exact-exponential relaxation M± += (1 − exp(−dt/τ))·(target± − M±) with the
    production collision time τ = Kn/(ρ·√Θ·2). Unconditionally stable at any Kn.

`split_maxwellian35` uses erf/exp closed-form half-Gaussian x-moments (about the
gauge velocity c) and full-Gaussian y,z factors, so each target is separable:
M^±_{ijk} = A^±_i · B_j · C_k with A^±_i the half-space x-moments (carrying ρ and
the split), B_j / C_k the full raw Gaussian y / z moments. By construction
M⁺+M⁻ = the full isotropic Maxwellian (= `InitializeM4_35(ρ,u,v,w,Θ,0,0,Θ,0,Θ)`).

Matches `collision35`'s conventions (Θ from the covariance trace, floored 1e-14).
"""
module BGKStream

export split_maxwellian35, bgk_stream_relax, split_temperature

const _VAC = 1e-10
const _SQRT2PI = 2.5066282746310002   # √(2π)
const _INV_SQRT2PI = 0.3989422804014327
const _INV_SQRT2 = 0.7071067811865476

# ---------------------------------------------------------------------------
# erfc(x), full double precision (Cody 1969 rational-Chebyshev, CALERF). No dep.
# ---------------------------------------------------------------------------
@inline function _erfc(x::Float64)
    y = abs(x)
    if y <= 0.46875
        z = y * y
        xnum = 1.85777706184603153e-1 * z; xden = z
        @inbounds for (a, b) in ((3.16112374387056560e0, 2.36012909523441209e1),
                                 (1.13864154151050156e2, 2.44024637934444173e2),
                                 (3.77485237685302021e2, 1.28261652607737228e3))
            xnum = (xnum + a) * z; xden = (xden + b) * z
        end
        erf = x * (xnum + 3.20937758913846947e3) / (xden + 2.84423683343917062e3)
        return 1.0 - erf
    elseif y <= 4.0
        xnum = 2.15311535474403846e-8 * y; xden = y
        @inbounds for (c, d) in ((5.64188496988670089e-1, 1.57449261107098347e1),
                                 (8.88314979438837594e0, 1.17693950891312499e2),
                                 (6.61191906371416295e1, 5.37181101862009858e2),
                                 (2.98635138197400131e2, 1.62138957456669019e3),
                                 (8.81952221241769090e2, 3.29079923573345963e3),
                                 (1.71204761263407058e3, 4.36261909014324716e3),
                                 (2.05107837782607147e3, 3.43936767414372164e3))
            xnum = (xnum + c) * y; xden = (xden + d) * y
        end
        res = (xnum + 1.23033935479799725e3) / (xden + 1.23033935480374942e3)
        ysq = trunc(y * 16.0) / 16.0
        del = (y - ysq) * (y + ysq)
        res *= exp(-ysq * ysq) * exp(-del)
        return x < 0 ? 2.0 - res : res
    else
        z = 1.0 / (y * y)
        xnum = 1.63153871373020978e-2 * z; xden = z
        @inbounds for (p, q) in ((3.05326634961232344e-1, 2.56852019228982242e0),
                                 (3.60344899949804439e-1, 1.87295284992346047e0),
                                 (1.25781726111229246e-1, 5.27905102951428412e-1),
                                 (1.60837851487422766e-2, 6.05183413124413191e-2))
            xnum = (xnum + p) * z; xden = (xden + q) * z
        end
        res = z * (xnum + 6.58749161529837803e-4) / (xden + 2.33520497626869185e-3)
        res = (5.6418958354775628695e-1 - res) / y
        ysq = trunc(y * 16.0) / 16.0
        del = (y - ysq) * (y + ysq)
        res *= exp(-ysq * ysq) * exp(-del)
        return x < 0 ? 2.0 - res : res
    end
end

# ---------------------------------------------------------------------------
# Half-space raw x-moments A^±_0..A^±_4 for N(u,Θ)·ρ restricted to v ≷ c.
# Standardized split point a=(c−u)/σ; upper partial standard moments I_0..I_4 by
# recurrence; then v^i = Σ C(i,k) u^{i-k} σ^k t^k.  A^-_i = ρ·E[v^i] − A^+_i.
# ---------------------------------------------------------------------------
@inline function _halfx_moments(rho::Float64, u::Float64, Θ::Float64, c::Float64)
    σ = sqrt(max(Θ, 1e-14))
    a = (c - u) / σ
    φa = _INV_SQRT2PI * exp(-0.5 * a * a)
    I0 = 0.5 * _erfc(a * _INV_SQRT2)          # ∫_a^∞ φ
    I1 = φa
    I2 = a * φa + I0
    I3 = a * a * φa + 2.0 * I1
    I4 = a * a * a * φa + 3.0 * I2
    I = (I0, I1, I2, I3, I4)
    # binomial(i,k) for i,k ≤ 4
    Ap0 = rho * I0
    Ap1 = rho * (u * I[1] + σ * I[2])
    Ap2 = rho * (u^2 * I[1] + 2u * σ * I[2] + σ^2 * I[3])
    Ap3 = rho * (u^3 * I[1] + 3u^2 * σ * I[2] + 3u * σ^2 * I[3] + σ^3 * I[4])
    Ap4 = rho * (u^4 * I[1] + 4u^3 * σ * I[2] + 6u^2 * σ^2 * I[3] + 4u * σ^3 * I[4] + σ^4 * I[5])
    # full raw Gaussian moments ρ·E[v^i]
    E0 = rho; E1 = rho * u; E2 = rho * (u^2 + Θ)
    E3 = rho * (u^3 + 3u * Θ); E4 = rho * (u^4 + 6u^2 * Θ + 3Θ^2)
    Aplus = (Ap0, Ap1, Ap2, Ap3, Ap4)
    Aminus = (E0 - Ap0, E1 - Ap1, E2 - Ap2, E3 - Ap3, E4 - Ap4)
    return Aplus, Aminus
end

# full raw 1D Gaussian moments B_0..B_4 for N(μ,Θ) (dimensionless, B_0=1)
@inline function _gauss_moments(μ::Float64, Θ::Float64)
    return (1.0, μ, μ^2 + Θ, μ^3 + 3μ * Θ, μ^4 + 6μ^2 * Θ + 3Θ^2)
end

# assemble the 35-moment vector M_{ijk} = A_i·B_j·C_k in canonical M4 order.
@inline function _assemble35(A::NTuple{5,Float64}, B::NTuple{5,Float64}, C::NTuple{5,Float64})
    return (
        A[1]*B[1]*C[1], A[2]*B[1]*C[1], A[3]*B[1]*C[1], A[4]*B[1]*C[1], A[5]*B[1]*C[1],   # 000..400
        A[1]*B[2]*C[1], A[2]*B[2]*C[1], A[3]*B[2]*C[1], A[4]*B[2]*C[1],                    # 010 110 210 310
        A[1]*B[3]*C[1], A[2]*B[3]*C[1], A[3]*B[3]*C[1],                                    # 020 120 220
        A[1]*B[4]*C[1], A[2]*B[4]*C[1],                                                    # 030 130
        A[1]*B[5]*C[1],                                                                    # 040
        A[1]*B[1]*C[2], A[2]*B[1]*C[2], A[3]*B[1]*C[2], A[4]*B[1]*C[2],                    # 001 101 201 301
        A[1]*B[1]*C[3], A[2]*B[1]*C[3], A[3]*B[1]*C[3],                                    # 002 102 202
        A[1]*B[1]*C[4], A[2]*B[1]*C[4],                                                    # 003 103
        A[1]*B[1]*C[5],                                                                    # 004
        A[1]*B[2]*C[2], A[2]*B[2]*C[2], A[3]*B[2]*C[2],                                    # 011 111 211
        A[1]*B[3]*C[2], A[2]*B[3]*C[2],                                                    # 021 121
        A[1]*B[4]*C[2],                                                                    # 031
        A[1]*B[2]*C[3], A[2]*B[2]*C[3],                                                    # 012 112
        A[1]*B[2]*C[4],                                                                    # 013
        A[1]*B[3]*C[3])                                                                    # 022
end

# ---------------------------------------------------------------------------
# split_maxwellian35(ρ, u, v, w, Θ, c) -> (M⁺, M⁻) two 35-vectors.
# ---------------------------------------------------------------------------
@inline function split_maxwellian35(rho::Float64, u::Float64, v::Float64, w::Float64,
                                    Θ::Float64, c::Float64)
    Aplus, Aminus = _halfx_moments(rho, u, Θ, c)
    B = _gauss_moments(v, Θ)
    C = _gauss_moments(w, Θ)
    return _assemble35(Aplus, B, C), _assemble35(Aminus, B, C)
end

# temperature (per DOF) of a 35-moment total, matching collision35 (floored).
@inline function split_temperature(M::NTuple{35,Float64})
    rho = M[1]
    u = M[2] / rho; v = M[6] / rho; w = M[16] / rho
    C200 = M[3] / rho - u^2; C020 = M[10] / rho - v^2; C002 = M[20] / rho - w^2
    return max((C200 + C020 + C002) / 3, 1e-14)
end

# ---------------------------------------------------------------------------
# Exact-exponential two-stream BGK relaxation. Returns (M⁺', M⁻').
# Collisionless (Kn very large) or vacuum ⇒ streams unchanged.
# ---------------------------------------------------------------------------
@inline function bgk_stream_relax(Mp::NTuple{35,Float64}, Mm::NTuple{35,Float64},
                                  dt::Float64, Kn::Float64, c::Float64)
    Kn >= 1e5 && return Mp, Mm
    tot = ntuple(q -> Mp[q] + Mm[q], Val(35))
    rho = tot[1]
    rho <= _VAC && return Mp, Mm
    u = tot[2] / rho; v = tot[6] / rho; w = tot[16] / rho
    Θ = split_temperature(tot)
    tc = Kn / (rho * sqrt(Θ) * 2) + 1e-30
    e = 1.0 - exp(-dt / tc)
    tgtP, tgtM = split_maxwellian35(rho, u, v, w, Θ, c)
    Mp2 = ntuple(q -> Mp[q] + e * (tgtP[q] - Mp[q]), Val(35))
    Mm2 = ntuple(q -> Mm[q] + e * (tgtM[q] - Mm[q]), Val(35))
    return Mp2, Mm2
end

end # module

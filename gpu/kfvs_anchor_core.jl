# gpu/kfvs_anchor_core.jl — SINGLE-SOURCE anchor primitives (F3 flux + full-cone θ*).
#
# Bare functions (no module): textually `include`d into whichever module needs them, at
# a point where `chyqmom_nodes_3d_dev` is in scope. This is the DRY home for the
# kinetic-FVS interface flux and the full-cone θ* bisection: the CPU order-3 march
# (src/numerics/highorder_3d.jl) and the GPU order-3 residual both call these instead of
# maintaining separate copies. Device-safe: NTuple-based, no heap, no @fastmath, scalar
# arithmetic, and every node loop is a STATIC unroll over all NODEMAX nodes (via zip) so
# there is no runtime tuple indexing (illegal in a CUDA kernel). Unused nodes carry zero
# weight and zero abscissa, so processing all NODEMAX and gating on the abscissa sign is
# bit-identical to the CPU loop that ran to the live node count.

# 35-moment exponent triples (i,j,k), standard ordering (matches _CHYQ_TRIPLES).
const _KFVSA_TRIPLES = (
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))

# Accumulate one weighted node (w, ux, uy, uz) into a 35-moment NTuple, returning the
# updated tuple. Uses `x^i` (Base integer power) to be bit-identical to the CPU
# `_anchor_accum!` it replaces.
@inline function _kfvsa_accum(M::NTuple{35,Float64}, w::Float64, ux::Float64, uy::Float64, uz::Float64)
    return ntuple(Val(35)) do n
        @inbounds begin
            (i, j, k) = _KFVSA_TRIPLES[n]
            M[n] + w * ux^i * uy^j * uz^k
        end
    end
end

# All-finite check over a 35-tuple via static-index recursion (GPU-safe: no runtime
# tuple indexing; the recursion unrolls at compile time).
@inline function _kfvsa_allfinite(m::NTuple{35,Float64}, ::Val{Q}=Val(35)) where {Q}
    Q == 0 && return true
    @inbounds isfinite(m[Q]) || return false
    return _kfvsa_allfinite(m, Val(Q-1))
end

# Node-finiteness check over the 27 quadrature slots (static-index recursion).
@inline function _kfvsa_nodes_finite(ns::NTuple{27,Float64}, uxs::NTuple{27,Float64},
                                     uys::NTuple{27,Float64}, uzs::NTuple{27,Float64},
                                     ::Val{Q}=Val(27)) where {Q}
    Q == 0 && return true
    @inbounds (isfinite(ns[Q]) && isfinite(uxs[Q]) && isfinite(uys[Q]) && isfinite(uzs[Q])) || return false
    return _kfvsa_nodes_finite(ns, uxs, uys, uzs, Val(Q-1))
end

# Sequential static-index fold over the 27 quadrature nodes (GPU-safe unroll). Processes
# nodes 1..Q in order (⇒ bit-identical accumulation order to the CPU loop it replaces),
# keeping those whose axis abscissa has the wanted sign, accumulating (n·u_axis)·monomials.
# Unused slots carry zero abscissa, so the sign gate skips them exactly as the CPU loop
# that ran only to the live node count.
@inline function _kfvsa_upwind(F::NTuple{35,Float64}, ns::NTuple{27,Float64},
                               uxs::NTuple{27,Float64}, uys::NTuple{27,Float64}, uzs::NTuple{27,Float64},
                               axis::Int, want_pos::Bool, ::Val{Q}) where {Q}
    Q == 0 && return F
    F = _kfvsa_upwind(F, ns, uxs, uys, uzs, axis, want_pos, Val(Q-1))
    @inbounds begin
        ua = axis == 1 ? uxs[Q] : (axis == 2 ? uys[Q] : uzs[Q])
        keep = want_pos ? (ua > 0.0) : (ua < 0.0)
        return keep ? _kfvsa_accum(F, ns[Q]*ua, uxs[Q], uys[Q], uzs[Q]) : F
    end
end

# F3 kinetic-FVS interface flux (upwind quadrature): LEFT-cell nodes with u_axis>0 plus
# RIGHT-cell nodes with u_axis<0, each contributing (n·u_axis)·(ux^i uy^j uz^k). Returns
# the 35-moment flux NTuple, or `nothing` when either cell is degenerate (ρ≤0, non-finite
# moments, no nodes, or a non-finite node) — the caller then supplies its native HLL flux.
# axis ∈ {1=x, 2=y, 3=z}. Single-source core of the CPU `_kfvs_face_flux_tup`.
@noinline function kfvs_face_flux_dev(mL::NTuple{35,Float64}, mR::NTuple{35,Float64}, axis::Int)
    (mL[1] > 0.0 && _kfvsa_allfinite(mL)) || return nothing
    (mR[1] > 0.0 && _kfvsa_allfinite(mR)) || return nothing
    (nL, uxL, uyL, uzL, NL) = chyqmom_nodes_3d_dev(mL)
    (NL >= 1) || return nothing
    (nR, uxR, uyR, uzR, NR) = chyqmom_nodes_3d_dev(mR)
    (NR >= 1) || return nothing
    _kfvsa_nodes_finite(nL, uxL, uyL, uzL) || return nothing
    _kfvsa_nodes_finite(nR, uxR, uyR, uzR) || return nothing
    F = ntuple(_ -> 0.0, Val(35))
    F = _kfvsa_upwind(F, nL, uxL, uyL, uzL, axis, true,  Val(27))   # left cell: u_axis > 0
    F = _kfvsa_upwind(F, nR, uxR, uyR, uzR, axis, false, Val(27))   # right cell: u_axis < 0
    return F
end

# Device twin of the CPU `_marginal_regularized`: the marginal half of the θ* feasibility
# test (s3max skewness cap, per-axis Hankel/variance floors, and the 2nd-order-cross S2
# floor). Computes only the needed standardized moments from the raw moments with inline
# scalar central-moment formulas (the CPU path routes through the array-based, non-device
# M2CS4_35; these are math twins). NTuple in, no heap. Matches _marginal_regularized's
# floors (h2min=S2min=1e-6, tol=1e-9) and its sqrt(max(·,eps)) variance convention.
@inline function marginal_regularized_dev(M::NTuple{35,Float64}, Ma, s3max)
    ρ = M[1]
    (isfinite(ρ) && ρ > 0.0) || return false
    # normalized raw moments and means
    u = M[2]/ρ;  ax2 = M[3]/ρ;  ax3 = M[4]/ρ;  ax4 = M[5]/ρ
    v = M[6]/ρ;  ay2 = M[10]/ρ; ay3 = M[13]/ρ; ay4 = M[15]/ρ
    w = M[16]/ρ; az2 = M[20]/ρ; az3 = M[23]/ρ; az4 = M[25]/ρ
    # diagonal central moments
    c200 = ax2 - u*u
    c020 = ay2 - v*v
    c002 = az2 - w*w
    (c200 > 0.0 && c020 > 0.0 && c002 > 0.0 &&
     isfinite(c200) && isfinite(c020) && isfinite(c002)) || return false
    c300 = ax3 - 3.0*u*ax2 + 2.0*u*u*u
    c400 = ax4 - 4.0*u*ax3 + 6.0*u*u*ax2 - 3.0*u*u*u*u
    c030 = ay3 - 3.0*v*ay2 + 2.0*v*v*v
    c040 = ay4 - 4.0*v*ay3 + 6.0*v*v*ay2 - 3.0*v*v*v*v
    c003 = az3 - 3.0*w*az2 + 2.0*w*w*w
    c004 = az4 - 4.0*w*az3 + 6.0*w*w*az2 - 3.0*w*w*w*w
    # cross 2nd central moments
    c110 = M[7]/ρ  - u*v
    c101 = M[17]/ρ - u*w
    c011 = M[26]/ρ - v*w
    # standardized moments (sqrt(max(·,eps)) matches M2CS4_35)
    sx = sqrt(max(c200, eps())); sy = sqrt(max(c020, eps())); sz = sqrt(max(c002, eps()))
    S300 = c300/(sx*sx*sx); S400 = c400/(c200*c200)
    S030 = c030/(sy*sy*sy); S040 = c040/(c020*c020)
    S003 = c003/(sz*sz*sz); S004 = c004/(c002*c002)
    S110 = c110/(sx*sy);    S101 = c101/(sx*sz); S011 = c011/(sy*sz)
    (isfinite(S300) && isfinite(S400) && isfinite(S030) && isfinite(S040) &&
     isfinite(S003) && isfinite(S004) && isfinite(S110) && isfinite(S101) && isfinite(S011)) || return false
    h2min = 1.0e-6; S2min = 1.0e-6; tol = 1.0e-9
    (abs(S300) <= s3max + tol && abs(S030) <= s3max + tol && abs(S003) <= s3max + tol) || return false
    (S400 - S300*S300 - 1.0 >= h2min - tol) || return false
    (S040 - S030*S030 - 1.0 >= h2min - tol) || return false
    (S004 - S003*S003 - 1.0 >= h2min - tol) || return false
    S2 = 1.0 + 2.0*S110*S101*S011 - (S110*S110 + S101*S101 + S011*S011)
    (S2 >= S2min - tol) || return false
    return true
end

# Full-cone θ* bisection: largest θ∈[0,1] keeping Mlo + θ·dM realizable under the injected
# predicate `okfun` (Julia specializes on the function type ⇒ fully inlined, device-safe,
# no register bloat). The predicate differs by platform at its leaf (CPU: exact-eig
# is_realizable; GPU: inertia state_realizable_fullcone_dev) but the search is shared.
@inline function theta_star_fullcone_bisect(Mlo::NTuple{35,Float64}, dM::NTuple{35,Float64},
                                            okfun::FN; nb::Int = 24) where {FN}
    full = ntuple(j -> Mlo[j] + dM[j], Val(35))
    okfun(full) && return 1.0
    lo = 0.0; hi = 1.0
    for _ in 1:nb
        mid = 0.5 * (lo + hi)
        m = ntuple(j -> Mlo[j] + mid * dM[j], Val(35))
        okfun(m) ? (lo = mid) : (hi = mid)
    end
    return lo
end

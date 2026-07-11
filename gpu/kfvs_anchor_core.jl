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

@inline _kfvsa_allfinite(m::NTuple{35,Float64}) = all(isfinite, m)

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
    # node finiteness guard (matches CPU _anchor_quad); static unroll over the tuples
    for (w, ux, uy, uz) in zip(nL, uxL, uyL, uzL)
        (isfinite(w) && isfinite(ux) && isfinite(uy) && isfinite(uz)) || return nothing
    end
    for (w, ux, uy, uz) in zip(nR, uxR, uyR, uzR)
        (isfinite(w) && isfinite(ux) && isfinite(uy) && isfinite(uz)) || return nothing
    end
    F = ntuple(_ -> 0.0, Val(35))
    # left cell: outgoing nodes (u_axis > 0)
    for (w, ux, uy, uz) in zip(nL, uxL, uyL, uzL)
        ua = axis == 1 ? ux : (axis == 2 ? uy : uz)
        (ua > 0.0) || continue
        F = _kfvsa_accum(F, w * ua, ux, uy, uz)
    end
    # right cell: incoming nodes (u_axis < 0)
    for (w, ux, uy, uz) in zip(nR, uxR, uyR, uzR)
        ua = axis == 1 ? ux : (axis == 2 ? uy : uz)
        (ua < 0.0) || continue
        F = _kfvsa_accum(F, w * ua, ux, uy, uz)
    end
    return F
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

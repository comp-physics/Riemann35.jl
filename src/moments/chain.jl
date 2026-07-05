"""
    chain.jl — the "all-mean" Stieltjes chain (canonical-moment) coordinates for a
    1D half-line (x-support in [0,∞)) marginal.

Port of `zchain5` from `roe-writeup-scripts/verify_halfline_scheme.jl`, generalized
to order n ≤ 5. For the face-normal x-marginal moments m₀..m_{n−1} the chain returns
ζ₁..ζ_{n−1}:

    a₀ = m₁/m₀;  s₁ˡ = m_{l+2} − a₀ m_{l+1};  β₁ = s₁¹/m₀;  α₁ = s₁²/s₁¹ − a₀;
    s₂² = s₁³ − α₁ s₁² − β₁ m₂;  β₂ = s₂²/s₁¹;
    ζ₁ = a₀,  ζ₂ = β₁/ζ₁,  ζ₃ = α₁ − ζ₂,  ζ₄ = β₂/ζ₃.

Realizability of the stream (x-direction) ⇔ all ζ > 0. Repair is a coordinate-wise
clip `ζ = max(ζ, ZFLOOR)`; the number of activations is a key diagnostic (should be
~zero away from vacuum fronts).

Pure, allocation-free scalar functions (NTuple in / NTuple out), matching the
single-source-kernel style of `numerics/flux_closure_dev.jl`.
"""
module Chain

export chain, clip_chain, chain_realizable, ZFLOOR, VAC_CHAIN

const ZFLOOR = 1e-12   # coordinate-wise chain floor (repair)
const VAC_CHAIN = 1e-10  # density floor: m₀ ≤ VAC_CHAIN ⇒ treat cell as vacuum

# ---------------------------------------------------------------------------
# Raw chain: m₀..m_{n−1} (NTuple{n}) -> ζ₁..ζ_{n−1} (NTuple{n−1}). Order n ≤ 5.
# The lower orders are exact prefixes of the n=5 recurrence, so a single
# progressive evaluation covers n = 2..5 (type-stable per n).
# ---------------------------------------------------------------------------
@inline function chain(m::NTuple{N,T}) where {N,T}
    m0 = m[1]; a0 = m[2] / m0
    N == 2 && return (a0,)
    s1_1 = m[3] - a0 * m[2]; b1 = s1_1 / m0; z2 = b1 / a0
    N == 3 && return (a0, z2)
    s1_2 = m[4] - a0 * m[3]; a1 = s1_2 / s1_1 - a0; z3 = a1 - z2
    N == 4 && return (a0, z2, z3)
    s1_3 = m[5] - a0 * m[4]
    s22 = s1_3 - a1 * s1_2 - b1 * m[3]; b2 = s22 / s1_1; z4 = b2 / z3
    return (a0, z2, z3, z4)
end

"all ζ finite and strictly positive ⇒ the x-marginal is realizable"
@inline chain_realizable(z::NTuple{K,T}) where {K,T} =
    all(x -> isfinite(x) && x > ZFLOOR, z)

# ---------------------------------------------------------------------------
# Coordinate-wise repair. Returns (clipped ζ tuple, nclip) where nclip = number
# of coordinates that had to be floored (0 when the stream is realizable).
# ---------------------------------------------------------------------------
@inline _clip1(x) = (isfinite(x) && x > ZFLOOR) ? x : ZFLOOR
@inline _needs(x) = (isfinite(x) && x > ZFLOOR) ? 0 : 1

@inline function clip_chain(z::NTuple{K,T}) where {K,T}
    chain_realizable(z) && return z, 0
    zc = map(_clip1, z)
    nc = 0
    @inbounds for i in 1:K
        nc += _needs(z[i])
    end
    return zc, nc
end

end # module

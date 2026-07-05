"""
    halfline_closure.jl — the half-line ("all-mean chain") closure and its
    h-functional channel closures.

Ported from `halfline_flux`/`hseq`/`clos` in `roe-writeup-scripts/`
(`verify_halfline_scheme.jl`, `verify_halfspace_2d15.jl`). The theory is DONE and
proven (thm:mirror, prop:interlace): the closure is strictly hyperbolic with real,
positive wave speeds on the whole realizable cone at every order n = 1..5.

Given a stream's x-marginal moments m₀..m₄ (chain ζ₁..ζ₄ from `Chain.chain`):

  * marginal closure  m₅ = h₅  where h is the pseudo-moment sequence h₀..h₈ of the
    palindromic chain extension  ζ̂ = (ζ₁,ζ₂,ζ₃,ζ₄,ζ₅=mean(ζ₁..ζ₄), (5/4)ζ₄,ζ₃,ζ₂[,ζ₁]),
    built by the 5×5 NONSYMMETRIC Jacobi walk (diag α, superdiag 1, subdiag β — no
    sqrt). By construction h₀..h₄ are the marginal moments and h₅ is the marginal
    closure. (ζ̂₉=ζ₁ affects only α₅, hence only h₉₊, so the 8-entry palindrome
    already fixes h₀..h₈ exactly — matches `hseq` in verify_halfspace_2d15.jl.)

  * transverse channel of order n_r (fixed (j,k), n_r = 5−(j+k)): solve the
    n_r×n_r Hankel section H·c = u, H = [h_{a+b}], u = (M_{0jk}..M_{(n_r−1)jk}),
    and the closure is (h_{n_r}..h_{2n_r−1})·c. Each H is PD by theorem, so every
    channel block is the n_r-point h-Gauss Jacobi section (real positive nodes).
    n_r = 1 reduces to ζ₁·M_{0jk}.

Wave speeds of any channel block are eigenvalues of the leading symmetric Jacobi
truncation of ζ̂; by interlacing the marginal (order-5) nodes bound all channel
nodes, so the CFL needs only the marginal eig. `xspeed` returns that bound (the
largest marginal support node), matching `halfline_flux`'s `maximum(nodes)`.
"""
module HalflineClosure

using LinearAlgebra
using StaticArrays

export hseq, chan_closure, marg_closure, xspeed, channel_nodes, gauss_nodes

# ---------------------------------------------------------------------------
# Pseudo-moment sequence h₀..h₈ from the clipped chain (m₀, ζ₁..ζ₄).
# Nonsymmetric 5×5 Jacobi walk of the palindromic extension ζ̂ (verbatim `hseq`).
# ---------------------------------------------------------------------------
@inline function hseq(m0::T, z1::T, z2::T, z3::T, z4::T) where {T}
    z5 = (z1 + z2 + z3 + z4) / 4
    # palindrome ζ̂ (8 entries suffice for h₀..h₈):
    zc1, zc2, zc3, zc4, zc5, zc6, zc7, zc8 = z1, z2, z3, z4, z5, 5z4 / 4, z3, z2
    # nonsymmetric Jacobi bands: α₁..α₄, β₁..β₄ (α₅ unused for h≤8)
    a1 = zc1;        a2 = zc2 + zc3;  a3 = zc4 + zc5;  a4 = zc6 + zc7
    b1 = zc1 * zc2;  b2 = zc3 * zc4;  b3 = zc5 * zc6;  b4 = zc7 * zc8
    # walk w ← J·w on 5 scalars; hₖ = m₀·(J^k)[1,1] = m₀·w₁ after k steps
    w1 = one(T); w2 = zero(T); w3 = zero(T); w4 = zero(T); w5 = zero(T)
    h0 = m0
    h = MVector{9,T}(undef); @inbounds h[1] = h0
    @inbounds for k in 1:8
        n1 = a1 * w1 + w2
        n2 = b1 * w1 + a2 * w2 + w3
        n3 = b2 * w2 + a3 * w3 + w4
        n4 = b3 * w3 + a4 * w4 + w5
        n5 = b4 * w4            # α₅ = 0
        w1, w2, w3, w4, w5 = n1, n2, n3, n4, n5
        h[k+1] = m0 * w1
    end
    return NTuple{9,T}(h)
end

"convenience: build h directly from the (clipped) chain tuple"
@inline hseq(m0::T, z::NTuple{4,T}) where {T} = hseq(m0, z[1], z[2], z[3], z[4])

# ---------------------------------------------------------------------------
# Marginal closure m₅ = h₅ (the (⌊5/2⌋+1)=3-size Jacobi walk lives inside h).
# ---------------------------------------------------------------------------
@inline marg_closure(h::NTuple{9,T}) where {T} = h[6]   # h₅

# ---------------------------------------------------------------------------
# n_r-point Gauss nodes of the h-functional (leading symmetric Jacobi truncation
# of the palindrome ζ̂). Real, distinct, positive by hyperbolicity. Returned as an
# SVector{n_r} for a stack-allocated channel closure.
# ---------------------------------------------------------------------------
@inline function gauss_nodes(z1::T, z2::T, z3::T, z4::T, ::Val{nr}) where {nr,T}
    z5 = (z1 + z2 + z3 + z4) / 4
    zc = (z1, z2, z3, z4, z5, 5z4 / 4, z3, z2)
    al = (zc[1], zc[2] + zc[3], zc[4] + zc[5], zc[6] + zc[7])
    be = (zc[1] * zc[2], zc[3] * zc[4], zc[5] * zc[6], zc[7] * zc[8])
    J = SMatrix{nr,nr,T}(ntuple(Val(nr * nr)) do t
        r = (t - 1) % nr + 1; c = (t - 1) ÷ nr + 1
        r == c ? al[r] : (abs(r - c) == 1 ? sqrt(max(be[min(r, c)], zero(T))) : zero(T))
    end)
    return eigvals(Symmetric(J))
end

# ---------------------------------------------------------------------------
# h-functional channel closure of order n_r. u = (M_{0jk}..M_{(n_r−1)jk}).
#
# Mathematically closure = (h_{n_r}..h_{2n_r−1})·H⁻¹·u with H = [h_{a+b}] the Hankel
# section, but the Hankel form is exponentially ill-conditioned at high Ma (it can go
# singular). The equivalent NODE form is stable: with ξ_a the n_r-point Gauss nodes
# of the h-functional (`gauss_nodes`), solve the Vandermonde P y = u (P[b,a]=ξ_a^b,
# b=0..n_r−1) and take closure = Σ_a ξ_a^{n_r}·y_a. cond(P) ≈ √cond(H), so this
# recovers the machine-precision exactness the Hankel form loses.
# ---------------------------------------------------------------------------
@inline chan_closure(nodes::SVector{1,T}, u::NTuple{1,T}) where {T} = nodes[1] * u[1]

@inline function chan_closure(nodes::SVector{nr,T}, u::NTuple{nr,T}) where {nr,T}
    P = SMatrix{nr,nr,T}(ntuple(Val(nr * nr)) do t
        r = (t - 1) % nr + 1; c = (t - 1) ÷ nr + 1
        nodes[c]^(r - 1)                    # row = power b=r−1, col = node a=c
    end)
    y = P \ SVector{nr,T}(u)
    s = zero(T)
    @inbounds for a in 1:nr
        s += nodes[a]^nr * y[a]
    end
    return s
end

# ---------------------------------------------------------------------------
# CFL wave-speed bound: largest marginal support node = max eigenvalue of the
# symmetric 3×3 Jacobi (a₀,√β₁; a₁,√β₂; a₂), a₂ = ζ₄+ζ₅. Matches halfline_flux.
# ---------------------------------------------------------------------------
@inline function xspeed(z1::T, z2::T, z3::T, z4::T) where {T}
    z5 = (z1 + z2 + z3 + z4) / 4
    a0 = z1; b1 = z1 * z2; a1 = z2 + z3; b2 = z3 * z4; a2 = z4 + z5
    sb1 = sqrt(max(b1, zero(T))); sb2 = sqrt(max(b2, zero(T)))
    J = SMatrix{3,3,T}(a0, sb1, zero(T), sb1, a1, sb2, zero(T), sb2, a2)
    return maximum(eigvals(Symmetric(Matrix(J))))
end

# ---------------------------------------------------------------------------
# Predicted channel-block spectrum: n_r-point Gauss nodes of the h-functional
# (leading symmetric Jacobi truncation of ζ̂). For tests / interlacing checks.
# ---------------------------------------------------------------------------
function channel_nodes(m0::T, z1::T, z2::T, z3::T, z4::T, nr::Int) where {T}
    z5 = (z1 + z2 + z3 + z4) / 4
    zc = (z1, z2, z3, z4, z5, 5z4 / 4, z3, z2)
    al = (zc[1], zc[2] + zc[3], zc[4] + zc[5], zc[6] + zc[7])
    be = (zc[1] * zc[2], zc[3] * zc[4], zc[5] * zc[6], zc[7] * zc[8])
    J = SymTridiagonal(collect(al[1:nr]), [sqrt(max(be[i], zero(T))) for i in 1:nr-1])
    return eigvals(J)
end

end # module

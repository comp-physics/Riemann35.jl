# ---------------------------------------------------------------------------------------------
# multirate35.jl -- A COLLISION OPERATOR WITH ONE RATE PER IRREDUCIBLE MOMENT GROUP.
#
# WHY THIS EXISTS. `bgk_relax_tup` builds its update as Mout = (1-e)*MG + e*M, a SINGLE convex
# combination toward the Maxwellian, so BGK necessarily relaxes all thirty non-conserved moments
# at one rate. ES-BGK's anisotropic Gaussian target adds exactly one more, for the stress. That is
# two rates. Monatomic Maxwell molecules need five, and the exact values are known in closed form
# because the Boltzmann moment hierarchy closes for that interaction:
#
#     sigma_ij 1     q_i 2/3     m_ijk 3/2     Delta 2/3     R_ij 7/6     phi_ijkl 7/4
#
# (phi_ijkl is MEASURED here, not cited -- see MR_RATES_EXACT.)
#
# (arXiv:1903.00966 sec. 3.4 for d=3, e=1, cross-referencing Struchtrup 2005 and Gu & Emerson
# 2009.) MEASURED delivery, by diagonalising a finite-difference Jacobian of one relaxation step:
# BGK gives 30 modes at rate 1; ES-BGK(Pr=2/3) gives 5 at 1 and 25 at 2/3. So ES-BGK is exact at
# the two orders that set mu and k, and wrong at EVERY order above -- m_ijk by 9/4, R_ij by 7/4,
# phi_ijkl by 21/8, all too slow. Those are precisely the moments a 35-moment method exists to
# carry. See roe-hyqmom-research/dsmc/reference/relax_spectrum.csv.
#
# WHY A DIAGONAL OPERATOR SUFFICES. The same source gives nu*_Rsigma = nu*_phiq = 0 at e = 1, so
# the linearised Maxwell-molecule production terms are COMPLETELY DECOUPLED. One rate per group
# therefore reproduces the exact spectrum rather than approximating it.
#
# WHAT IS GIVEN UP. `bgk_relax_tup` preserves realizability BY CONVEXITY: both endpoints are
# realizable and the realizable set is convex. Per-group rates destroy that argument -- the result
# lies in the BOX spanned by M and MG, whose corners need not be realizable, which is also why
# Shakhov-type models can go non-realizable. MEASURED instead: no violation in 30 (state, dt/tau)
# pairs, in a sweep driving the input cone margin to 6.75e-07 (two near-delta beams, essentially
# ON the boundary), or over 3000 iterated applications. All six rates are strictly positive, so
# every mode contracts toward an interior target. That is an observation, NOT a proof, and the
# realizability projection still runs downstream exactly as before.
#
# HOW THE DECOMPOSITION IS BUILT, since a hand-written one is where a plausible-but-wrong operator
# would come from. R_ij is rank-2 trace-free exactly like sigma_ij and the two are separated only
# by a Sonine radial index. Nothing here is transcribed:
#   * H_l = ker(Laplacian) on degree-l polynomials. The Laplacian IS the tensor contraction, so
#     its kernel IS the trace-free part. Computed as a nullspace.
#   * V_d = H_d (+) r^2 H_{d-2} (+) r^4 H_{d-4}, and dim H_l = 2l+1, gives the groups and their
#     multiplicities 5/3/7/1/5/9 directly.
#   * Gram-Schmidt across k under the Gaussian inner product PRODUCES the Sonine separation.
# The composite operator depends only on the group PROJECTORS, which are unique, so it does not
# depend on the (arbitrary) basis Gram-Schmidt happens to pick inside a degenerate block.
# ---------------------------------------------------------------------------------------------
module MultiRate35

using LinearAlgebra
using ..MomentIndices: IJK, IJK_INDEX
using ..ReconDev: _rk3_stage_factor

export MR_RATES_EXACT, MR_RATES_BGK, MR_RATES_ESBGK, MR_RATES_QONLY,
       multirate_matrices, multirate_relax_tup

const ND = [sum(t) for t in IJK]

"Unit isotropic Gaussian moment <cx^a cy^b cz^c> at u = 0: a product of double factorials."
function _gmom(t::NTuple{3,Int})
    p = 1.0
    for e in t
        isodd(e) && return 0.0
        for m in 1:2:(e - 1); p *= m; end
    end
    p
end
_ip(p, q) = sum(p[a] * q[b] * _gmom(IJK[a] .+ IJK[b]) for a in 1:35, b in 1:35)

_degidx(d) = [n for n in 1:35 if ND[n] == d]

function _laplacian(d)
    src, dst = _degidx(d), _degidx(d - 2)
    L = zeros(length(dst), length(src))
    for (cj, n) in enumerate(src), ax in 1:3
        e = IJK[n][ax]; e < 2 && continue
        tt = ntuple(z -> z == ax ? IJK[n][z] - 2 : IJK[n][z], 3)
        L[findfirst(==(IJK_INDEX[tt]), dst), cj] += e * (e - 1)
    end
    L
end

function _harmonic(d)
    src = _degidx(d)
    Bm = d < 2 ? Matrix{Float64}(I, length(src), length(src)) : nullspace(_laplacian(d))
    [(v = zeros(35); for (cj, n) in enumerate(src); v[n] = Bm[cj, c]; end; v) for c in 1:size(Bm, 2)]
end

function _times_r2(p)
    q = zeros(35)
    for n in 1:35
        p[n] == 0 && continue
        for ax in 1:3
            t = ntuple(z -> z == ax ? IJK[n][z] + 2 : IJK[n][z], 3)
            sum(t) <= 4 && (q[IJK_INDEX[t]] += p[n])
        end
    end
    q
end

# (name, l, k, multiplicity). Conserved groups carry rate 0 in every rate set below.
const MR_GROUPS = [("rho", 0, 0, 1), ("momentum", 1, 0, 3), ("energy", 0, 1, 1),
                   ("sigma_ij", 2, 0, 5), ("q_i", 1, 1, 3), ("m_ijk", 3, 0, 7),
                   ("Delta", 0, 2, 1), ("R_ij", 2, 1, 5), ("phi_ijkl", 4, 0, 9)]

"Gaussian-orthonormal basis rows over the 35 monomials, plus each row's group name."
function _basis()
    rows = Vector{Float64}[]; labels = String[]
    for l in 0:4
        Hl = _harmonic(l); isempty(Hl) && continue
        prev = Vector{Float64}[]; k = 0
        while l + 2k <= 4
            blk = Vector{Float64}[]
            for h in Hl
                p = copy(h)
                for _ in 1:k; p = _times_r2(p); end
                for u in vcat(prev, blk); p .-= _ip(p, u) .* u; end
                n = sqrt(_ip(p, p))
                n < 1e-10 && error("degenerate basis vector at l=$l k=$k")
                push!(blk, p ./ n)
            end
            g = findfirst(G -> G[2] == l && G[3] == k, MR_GROUPS)
            length(blk) == MR_GROUPS[g][4] ||
                error("(l=$l,k=$k): built $(length(blk)), declared $(MR_GROUPS[g][4])")
            append!(rows, blk); append!(labels, fill(MR_GROUPS[g][1], length(blk)))
            append!(prev, blk); k += 1
        end
    end
    reduce(vcat, [r' for r in rows]), labels
end

const _B, _LABELS = _basis()
const _G = [_gmom(IJK[a] .+ IJK[b]) for a in 1:35, b in 1:35]
const _BINV = _G * _B'                    # B is Gaussian-orthonormal, so B*G*B' = I

# ---- named rate sets, per GROUP (conserved groups are pinned at 0) ----------------------------
_rates(d) = [get(d, _LABELS[a], 0.0) for a in 1:35]
"""
Exact monatomic Maxwell-molecule spectrum.

phi_ijkl = 7/4 IS MEASURED, NOT CITED, and the value first used here was wrong. arXiv:1903.00966
lists `nu*_phi = 1`, but its G29 system is G26 plus THREE moments, so its phi is a VECTOR (l=1,
r=2) -- confirmed by its stated zero coupling `nu*_phi_q = 0` being with q, another vector, exactly
as `nu*_R_sigma = 0` pairs the two rank-2 moments. The rank-4 deviator a 35-moment system carries
is not in G29 at all, so that source never gave its rate, and reading 1.0 off it was an error.

Measured instead, as a collision matrix element at equilibrium (dsmc/collision_matrix.py):
7/4 to 0.48 sigma over 1.28e8 pairs, with every other simple rational at 7 sigma or worse and the
1.0 originally used 347 sigma away. The same measurement independently reproduces the four values
that ARE citable -- q 0.6690, m 1.5006, Delta 0.6668, R 1.1690 against 2/3, 3/2, 2/3, 7/6 -- which
is what licenses believing it on the fifth.
"""
const MR_RATES_EXACT = _rates(Dict("sigma_ij" => 1.0, "q_i" => 2/3, "m_ijk" => 3/2,
                                   "Delta" => 2/3, "R_ij" => 7/6, "phi_ijkl" => 7/4))
"What plain BGK delivers: one rate for everything."
const MR_RATES_BGK   = _rates(Dict(g[1] => 1.0 for g in MR_GROUPS[4:end]))
"What ES-BGK(Pr=2/3) delivers: stress at 1, everything else at 2/3."
const MR_RATES_ESBGK = _rates(Dict("sigma_ij" => 1.0, "q_i" => 2/3, "m_ijk" => 2/3,
                                   "Delta" => 2/3, "R_ij" => 2/3, "phi_ijkl" => 2/3))
"""
THE DISCRIMINATOR, and the reason this module is worth having: the Prandtl fix WITHOUT ES-BGK's
side effect on the higher moments. ES-BGK cannot express this -- it has only one knob below the
stress. If this beats ES-BGK against DSMC, then the ES-BGK penalty the decomposition table
measures is the higher-moment damage and not the Prandtl correction.
"""
const MR_RATES_QONLY = _rates(Dict("sigma_ij" => 1.0, "q_i" => 2/3, "m_ijk" => 1.0,
                                   "Delta" => 1.0, "R_ij" => 1.0, "phi_ijkl" => 1.0))

"Host-side matrices for the device kernel: (B, Binv, rate-per-basis-row, degree-per-moment)."
multirate_matrices(rates::AbstractVector{Float64} = MR_RATES_EXACT) =
    (copy(_B), copy(_BINV), collect(Float64, rates), Float64.(ND))

# ---- raw <-> central shift, unrolled at compile time from the canonical table -----------------
# @generated so the (i,j,k) sums become straight-line code with STATIC tuple indices. Dynamic
# tuple indexing is what blocks SROA on the GPU (see recon_dev.jl and issue #66), so it is
# avoided rather than worked around.
@generated function _shift35(M::NTuple{35,Float64}, ux::Float64, uy::Float64, uz::Float64,
                             ::Val{SGN}) where {SGN}
    outs = Any[]
    for (a, b, c) in IJK
        terms = Any[]
        for i in 0:a, j in 0:b, k in 0:c
            co = binomial(a, i) * binomial(b, j) * binomial(c, k)
            f = Any[:(M[$(IJK_INDEX[(i, j, k)])])]
            co != 1 && pushfirst!(f, :($(Float64(co))))
            for _ in 1:(a - i); push!(f, SGN > 0 ? :ux : :(-ux)); end
            for _ in 1:(b - j); push!(f, SGN > 0 ? :uy : :(-uy)); end
            for _ in 1:(c - k); push!(f, SGN > 0 ? :uz : :(-uz)); end
            push!(terms, length(f) == 1 ? f[1] : Expr(:call, :*, f...))
        end
        push!(outs, length(terms) == 1 ? terms[1] : Expr(:call, :+, terms...))
    end
    Expr(:tuple, outs...)
end

@inline function _matvec35(A, x::NTuple{35,Float64})
    ntuple(i -> begin
        s = 0.0
        Base.Cartesian.@nexprs 35 j -> (s += A[i, j] * x[j])
        s
    end, Val(35))
end

"""
    multirate_relax_tup(M, dt, Kn, B, Binv, rates, nd) -> NTuple{35,Float64}

One exact-exponential relaxation step with a separate rate per irreducible group.

`tc = (Kn/2)*Theta^(omega-1)/rho` matches `bgk_relax_tup` exactly, so `rates == MR_RATES_BGK`
reproduces plain BGK and is the natural regression check. rho, u and the ISOTROPIC Theta are
recomputed from the state and re-applied unchanged, so mass, momentum and energy are conserved by
CONSTRUCTION rather than by cancellation.

The state is carried to standardized central moments about the local u and isotropic Theta --
the frame in which the Maxwell-molecule eigenfunctions are defined, and the one in which a
Maxwellian sits at exactly zero on all 34 non-constant basis coefficients, so relaxing toward it
is a plain per-mode rescale with no target to build.
"""
@inline function multirate_relax_tup(M::NTuple{35,Float64}, dt::Float64, Kn::Float64,
                                     B, Binv, rates, nd, rk3::Bool = false,
                                     omega::Float64 = 0.5)::NTuple{35,Float64}
    rho = M[1]
    rho > 0.0 || return M
    ux = M[2] / rho; uy = M[6] / rho; uz = M[16] / rho
    Th = ((M[3] / rho - ux * ux) + (M[10] / rho - uy * uy) + (M[20] / rho - uz * uz)) / 3
    Th = Th > 1e-14 ? Th : 1e-14
    # tau_ref = (Kn/2) * Theta^(omega-1) / rho, matching `_esbgk_relax_tup`. The omega == 0.5
    # branch is kept separate rather than left to the algebra: Theta^(-0.5) is NOT bitwise
    # 1/sqrt(Theta), and the default path must stay byte-identical to bgk_relax_tup.
    tc = omega == 0.5 ? Kn / (rho * sqrt(Th) * 2) : (Kn / 2) * Th^(omega - 1.0) / rho
    isfinite(tc) && tc > 0.0 || return M

    C = _shift35(M, ux, uy, uz, Val(-1))
    s = sqrt(Th)
    # S_n = C_n / (rho * s^deg(n)); powers built by repeated multiply, never `^`.
    sp = ntuple(n -> begin
        d = nd[n]; p = 1.0
        d >= 1 && (p *= s); d >= 2 && (p *= s); d >= 3 && (p *= s); d >= 4 && (p *= s)
        p
    end, Val(35))
    S = ntuple(n -> C[n] / (rho * sp[n]), Val(35))
    a = _matvec35(B, S)
    x = dt / tc
    # SSP-RK3 composite correction, PER GROUP. `stage_bgk` applies the collision once per stage
    # with the full dt, and three convex applications with per-stage factor s compose to
    # F(s) = s(s+1)(s+2)/6, so an uncorrected factor over-relaxes by F'(1) = 11/6. Each group
    # relaxes independently here, so the same inversion applies groupwise. Omitting it would make
    # the MR_RATES_BGK control disagree with the existing BGK runs and invalidate every
    # comparison against them. Single assignment to `e`: reassigning it boxes the closure-captured
    # local to Any and makes the call dynamic, which fails GPU compilation (see recon_dev.jl).
    ar = ntuple(n -> begin
        e0 = exp(-rates[n] * x)
        e = rk3 ? _rk3_stage_factor(e0) : e0
        a[n] * e
    end, Val(35))
    S2 = _matvec35(Binv, ar)
    C2 = ntuple(n -> S2[n] * rho * sp[n], Val(35))
    _shift35(C2, ux, uy, uz, Val(+1))
end

end # module

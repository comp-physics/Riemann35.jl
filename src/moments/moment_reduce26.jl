"""
    moment_reduce26.jl — R.O. Fox's 26-moment reduction map, reimplemented.

The 35-moment closure carries fifteen fourth-order moments. Fox's reduction keeps only
the six that are EVEN in every axis (S400, S040, S004, S220, S202, S022) as independent
state, and closes the nine that are odd in two axes algebraically:

    S_{3a 1b} = S_{1a 1b} S_{4a} + (3/2) S_{3a} ( S_{2a 1b} - S_{1a 1b} S_{3a} )
    S_{2a 1b 1c} = S_{1b 1c} + S_{3a} S_{111}

over the six ordered pairs (a,b) and the three choices of the doubled axis a. Written out,
the nine are S310, S301, S130, S031, S103, S013 and S211, S121, S112.

PROVENANCE AND CROSS-CHECK. This began as a reimplementation from the formulas printed
in sec:oblique-reduce26, because the scripts the notes cite (moment_reduce26.jl,
dvm_reduce26.jl, dvm_point.jl) were in no commit on any branch of either repository.
Fox's own implementation later landed as `src/numerics/moment_reduce26.jl` on the
`feat/reduce26` branch, and the two were compared directly: over 200 random realizable
states the nine closed moments agree to 7.9e-15, i.e. bit-identical up to roundoff. The
`(+perms)` expansion read here -- six ordered pairs (a,b) for the s310-type, three
choices of doubled axis for the s211-type -- is therefore confirmed correct.

This file is kept alongside Fox's rather than deleted because it works in STANDARDIZED
space and exposes the per-moment closure residual, which is what
`test/probe_reduce26_residual.jl` needs to separate the streamwise miss (S310, 0.7-2%)
from the wall-normal one (S130, up to 12%). Fox's version returns raw moments and is the
one wired into the solver via `REDUCE26[]`; this one is a diagnostic. Two independent
implementations agreeing to roundoff is also the only external check either has.

WHY THE LEADING TERM MATTERS. For a near-Maxwellian sheared state S400 -> 3, so

    S310 ≈ 3 S110,

i.e. the closure predicts the odd cross-moment is three times the standardized shear
stress. Any measurement finding |S310/S110| ≈ 3 is therefore NOT evidence that the
reduction loses information — it is evidence the reduction is right.
"""
module MomentReduce26

export reduce26_S, reduce26_residual, S_INDEX, DROPPED_KEYS

# The (i,j,k) -> slot map comes from the canonical table, NOT from a local copy of it.
# This module previously carried its own `S_KEYS` tuple, which was a
# character-for-character duplicate of `MomentIndices.IJK` — a second copy of the
# 35-moment ordering that nothing kept in sync with the first. `S_INDEX` is retained
# as the name here because it is what the probes import and what the notes cite, but
# it is now an alias, so there is exactly one ordering in the package.
using ..MomentIndices: IJK_INDEX
using ..Riemann35: nt35
const S_INDEX = IJK_INDEX

"The nine odd fourth-order cross-moments the reduction drops."
const DROPPED_KEYS = ((3,1,0),(3,0,1),(1,3,0),(0,3,1),(1,0,3),(0,1,3),(2,1,1),(1,2,1),(1,1,2))

@inline _e(a::Int) = a == 1 ? (1,0,0) : (a == 2 ? (0,1,0) : (0,0,1))
@inline _add(p,q) = (p[1]+q[1], p[2]+q[2], p[3]+q[3])
@inline _scale(p,n) = (p[1]*n, p[2]*n, p[3]*n)

"S value by (i,j,k) key from a standardized-moment vector"
@inline _S(S, k) = S[S_INDEX[k]]

"""
    reduce26_S(S) -> NTuple{35,Float64}

Apply the reduction: return a copy of the standardized-moment vector `S` with the nine
dropped entries overwritten by their algebraic closure. All other entries pass through
untouched, so `reduce26_S(S) - S` is supported exactly on those nine slots.
"""
function reduce26_S(S::AbstractVector{Float64})
    out = collect(Float64, S)
    # S_{3a 1b} for the six ordered pairs a != b
    for a in 1:3, b in 1:3
        a == b && continue
        k31 = _add(_scale(_e(a),3), _e(b))          # 3 in a, 1 in b
        k11 = _add(_e(a), _e(b))                     # S_{1a 1b}
        k4  = _scale(_e(a),4)                        # S_{4a}
        k3  = _scale(_e(a),3)                        # S_{3a}
        k21 = _add(_scale(_e(a),2), _e(b))           # S_{2a 1b}
        S11 = _S(S,k11); S4 = _S(S,k4); S3 = _S(S,k3); S21 = _S(S,k21)
        out[S_INDEX[k31]] = S11*S4 + 1.5*S3*(S21 - S11*S3)
    end
    # S_{2a 1b 1c} for the three choices of the doubled axis
    S111 = _S(S,(1,1,1))
    for a in 1:3
        b, c = (a % 3) + 1, ((a+1) % 3) + 1
        k211 = _add(_scale(_e(a),2), _add(_e(b), _e(c)))
        k11  = _add(_e(b), _e(c))
        k3   = _scale(_e(a),3)
        out[S_INDEX[k211]] = _S(S,k11) + _S(S,k3)*S111
    end
    nt35(out)
end

"""
    reduce26_residual(S) -> (rel, per_moment)

Relative closure residual `||R(S)-S|| / ||S||` restricted to the nine dropped slots, plus
the signed per-moment relative error `(R(S)-S)/S` for each. A small `rel` means the odd
cross-moments are SLAVED to the retained ones and the reduction loses nothing; a large one
means they carry independent content.
"""
function reduce26_residual(S::AbstractVector{Float64})
    R = reduce26_S(S)
    num = 0.0; den = 0.0
    per = Dict{NTuple{3,Int},NTuple{2,Float64}}()
    for k in DROPPED_KEYS
        i = S_INDEX[k]
        d = R[i] - S[i]
        num += d^2; den += S[i]^2
        per[k] = (S[i], abs(S[i]) > 0 ? d/S[i] : NaN)
    end
    (den > 0 ? sqrt(num/den) : 0.0), per
end

end # module

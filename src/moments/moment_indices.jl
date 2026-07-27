"""
    moment_indices.jl — THE canonical exponent table for the 35-moment set.

Single source of the moment ordering knowledge (M4 canonical order, matching
Fox's MATLAB header): the (i,j,k) velocity exponents per slot, and everything
derivable from them — per-axis marginal-chain index sets and face-normal
reflection-parity masks. Every site that previously hardcoded index tuples
(e.g. `[1,6,10,13,15]` for the y-marginal) should consume these.

Pure constants, no dependencies — safe to include from the package and from
the standalone GPU modules alike (each include creates an identical module
instance; the constants are what matters).
"""
module MomentIndices

export IJK, IJK_INDEX, MARG_IDX, MARG_VEC, ODD_MASK, momidx

# (i,j,k) exponents in canonical M4 order
const IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
             (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
             (0,3,0),(1,3,0),(0,4,0),
             (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
             (0,0,3),(1,0,3),(0,0,4),
             (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
             (0,1,2),(1,1,2),(0,1,3),(0,2,2))

"slot of moment M_ijk (throws if not in the 35-moment set)"
momidx(i, j, k) = findfirst(==((i, j, k)), IJK)::Int

# Reverse of IJK: (i,j,k) -> slot, as a Dict for use in non-hot diagnostic code
# where the key is a runtime tuple. `momidx` is the same lookup by linear scan and
# stays the right choice inside kernels; this is the right choice when the caller
# has a key in hand (`S[IJK_INDEX[(3,1,0)]]`) and would otherwise rebuild the table.
#
# It lives HERE, with the canonical ordering it inverts, rather than in whichever
# module first happened to need it. It was previously defined in
# `moments/moment_reduce26.jl` as `S_INDEX`, over a `S_KEYS` tuple that was a
# character-for-character duplicate of `IJK` above — two independent copies of the
# canonical moment ordering, either of which could have been edited without the
# other. That is the failure mode this file exists to prevent.
const IJK_INDEX = Dict{NTuple{3,Int},Int}(k => i for (i, k) in enumerate(IJK))

# per-axis marginal-chain slots (m0..m4 of the face-normal marginal), derived
const MARG_IDX = ntuple(ax -> ntuple(n -> momidx(ntuple(d -> d == ax ? n - 1 : 0, 3)...), 5), 3)
# Vector forms for CPU fancy indexing (Mr[MARG_VEC[axis]])
const MARG_VEC = ntuple(ax -> collect(MARG_IDX[ax]), 3)

# per-axis reflection parity: is the face-normal exponent odd?
const ODD_MASK = ntuple(ax -> ntuple(q -> isodd(IJK[q][ax]), 35), 3)

end # module

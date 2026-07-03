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

export IJK, MARG_IDX, MARG_VEC, ODD_MASK, momidx

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

# per-axis marginal-chain slots (m0..m4 of the face-normal marginal), derived
const MARG_IDX = ntuple(ax -> ntuple(n -> momidx(ntuple(d -> d == ax ? n - 1 : 0, 3)...), 5), 3)
# Vector forms for CPU fancy indexing (Mr[MARG_VEC[axis]])
const MARG_VEC = ntuple(ax -> collect(MARG_IDX[ax]), 3)

# per-axis reflection parity: is the face-normal exponent odd?
const ODD_MASK = ntuple(ax -> ntuple(q -> isodd(IJK[q][ax]), 35), 3)

end # module

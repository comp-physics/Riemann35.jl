module ProjectionDiagnostics
# projection_diagnostics.jl -- WHEN does the realizability projection fire, WHICH principal
# minor is negative, and HOW FAR does it move the moments.
#
# WHY THIS EXISTS. Two independent requests for the same artifact:
#
#   * R.O. Fox, 2026-08-02: "the current algorithm is more aggressive in changing cross moments
#     than it needs to be. For example, the bounds for 3rd-order moments are easy to apply, so
#     it should be possible to make 'small' modifications to s220, s202, s022 to get positive
#     eigenvalues. To get started, it might be best to collect some statistics on when
#     correction occur and which principal minor(s) is negative?"
#   * JCP Reviewer #3 on the 35-moment paper: the ratio of corrected cells to total, norms of
#     the moment change, and separate statistics for realizability vs hyperbolicity firings as
#     a function of Ma, Kn and mesh resolution.
#
# Plus a third, local reason: in the Fox-contact ladder, `reduce26` and full-35 separate at the
# 5e-3 level at Kn=0.8/Nx=144 even though all nine dropped moments vanish IDENTICALLY by
# vy/vz symmetry for that problem. Only a nonlinear operator can break that symmetry, and the
# projection is the only candidate in the path. This module is how that gets confirmed or ruled
# out rather than asserted.
#
# WHAT IS MEASURED. Realizability of a 35-moment state is positive-semidefiniteness of the 6x6
# matrix Delta2* built by `_delta2star_entries`. So "which principal minor is negative" is
# literally the leading principal minors D1..D6 of that matrix, and the FIRST non-positive one
# localises the failure: a small index means the failure is already present in a low-order
# sub-block, a large index means the state is only marginally unrealizable in the full 6x6.
# (Fox's own note of 2025-11-06 records that his MATLAB checked only to the 4th minor, "which
# was sufficient for 2-D", and that the 5th and 6th would eventually be needed -- so the index
# is a quantity he already reasons in.)
#
# COST. Host-side and diagnostic. This is deliberately NOT wired into the device hot path:
# instrumenting the kernel would need atomics per cell and would perturb the very runs being
# measured. Call it on a saved field instead.
using LinearAlgebra: det
using ..RealizeDev: _delta2star_entries, projection35_dev

export delta2star_matrix, leading_minors, projection_report, PROJ_S_IDX

"""
Indices into the 35-vector for the 28 standardized moments `projection35_dev` consumes, in its
argument order. Written out rather than derived so a reordering of the moment vector produces a
test failure instead of silently permuted diagnostics -- the IJK table has been mis-transcribed
once before with every total and trace still correct (issue #61).
"""
const PROJ_S_IDX = (4, 5, 7, 8, 9, 11, 12, 13, 14, 15,
                    17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 31, 32, 33, 34, 35)

# the three cross moments Fox singles out as the ones a minimal correction should touch
const CROSS_IDX = (12, 22, 35)          # S220, S202, S022 in 35-vector indexing
const _CROSS_IN_PROJ = (7, 15, 28)      # ...and their positions in the 28-tuple

"""
    delta2star_matrix(S28) -> 6x6 Symmetric

The realizability matrix whose PSD-ness *is* realizability. `S28` is the 28 standardized
moments in `projection35_dev` argument order.
"""
function delta2star_matrix(S28)
    e = _delta2star_entries(S28...)
    A = zeros(6, 6)
    k = 0
    for i in 1:6, j in i:6
        k += 1
        A[i, j] = e[k]; A[j, i] = e[k]
    end
    A
end

"""
    leading_minors(A) -> NTuple{6,Float64}

Leading principal minors `D1..D6` (determinants of the top-left k x k blocks). All strictly
positive iff `A` is positive definite (Sylvester).
"""
leading_minors(A) = ntuple(k -> det(@view A[1:k, 1:k]), 6)

"""
    projection_report(M35; Ma=0.0) -> NamedTuple

Everything the two requests above ask for, for ONE state.

Fields:
  `fired`      -- did the projection change anything
  `first_bad`  -- index of the first non-positive leading minor (0 if the state is realizable)
  `minors`     -- D1..D6, so the margin is visible and not just the sign
  `dS`         -- 2-norm of the change over all 28 standardized moments
  `dS_cross`   -- the part of that change carried by S220/S202/S022 alone
  `dS_other`   -- the rest. Fox's conjecture is that `dS_other` could be ~0; this is the
                  measurement that supports or refutes it, per state.
  `n_changed`  -- how many of the 28 moved by more than 1e-12
"""
function projection_report(M35; Ma::Real = 0.0, m2cs4 = nothing)
    # M2CS4_35 lives in the parent module, which is not loadable from here at include time
    # (this file is included before the parent finishes defining itself). Take it as an
    # argument, defaulting to the parent's binding resolved at call time.
    f = m2cs4 === nothing ? getfield(parentmodule(@__MODULE__), :M2CS4_35) : m2cs4
    _, S = f(collect(Float64, M35))
    S28 = ntuple(i -> S[PROJ_S_IDX[i]], 28)

    A = delta2star_matrix(S28)
    D = leading_minors(A)
    first_bad = something(findfirst(<=(0.0), collect(D)), 0)

    P28 = projection35_dev(S28...)
    d = ntuple(i -> P28[i] - S28[i], 28)
    dS = sqrt(sum(abs2, d))
    dS_cross = sqrt(sum(abs2, ntuple(i -> d[_CROSS_IN_PROJ[i]], 3)))
    dS_other = sqrt(max(dS^2 - dS_cross^2, 0.0))
    n_changed = count(x -> abs(x) > 1e-12, d)

    (fired = dS > 1e-12, first_bad = first_bad, minors = D,
     dS = dS, dS_cross = dS_cross, dS_other = dS_other, n_changed = n_changed)
end

end # module

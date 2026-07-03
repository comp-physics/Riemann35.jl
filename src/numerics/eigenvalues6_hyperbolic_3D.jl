"""
    eigenvalues6_hyperbolic_3D(M, axis, flag2D, Ma)

Eigenvalues of the 3D flux Jacobian in the x (axis=1) or y (axis=2) direction,
with a hyperbolicity correction. Direct port of `eigenvalues6x_hyperbolic_3D.m`
and `eigenvalues6y_hyperbolic_3D.m` (Code_Riemann_3D_35mom_july2026).

For each direction two transverse planes are checked via the 15-moment 2D flux
Jacobian `jacobian15`: the min/max eigenvalues come from its 3x3 block (rows
13:15) and 4x4 block (rows 6:9). If the 3x3 block has complex eigenvalues the
moments are projected to a hyperbolic set (zero the cross 3rd-order standardized
moments, floor S220/S202/S022 at 1/3) and the eigenvalues are recomputed.

Returns `(v6min, v6max, Mr)` where `Mr` is the (possibly corrected) 35-moment
vector. `flag2D`/`Ma` are accepted for caller compatibility (unused, matching
the MATLAB reference).
"""

# --- 15-moment slices for each transverse plane (MATLAB jacobian15 arg order) ---
@inline _plane_UV(M) = (M[1],M[6],M[10],M[13],M[15], M[2],M[7],M[11],M[14], M[3],M[8],M[12], M[4],M[9], M[5])
@inline _plane_UW(M) = (M[1],M[16],M[20],M[23],M[25], M[2],M[17],M[21],M[24], M[3],M[18],M[22], M[4],M[19], M[5])
@inline _plane_VU(M) = (M[1],M[2],M[3],M[4],M[5], M[6],M[7],M[8],M[9], M[10],M[11],M[12], M[13],M[14], M[15])
@inline _plane_VW(M) = (M[1],M[16],M[20],M[23],M[25], M[6],M[26],M[32],M[34], M[10],M[29],M[35], M[13],M[31], M[15])
@inline _plane_WU(M) = (M[1],M[2],M[3],M[4],M[5], M[16],M[17],M[18],M[19], M[20],M[21],M[22], M[23],M[24], M[25])
@inline _plane_WV(M) = (M[1],M[6],M[10],M[13],M[15], M[16],M[26],M[29],M[31], M[20],M[32],M[35], M[23],M[34], M[25])

"""
    _jac15_eig(m15)

Eigenvalue summary for one transverse plane: returns `(vmin, vmax, lam3)` where
`vmin/vmax` combine the 3x3 (13:15) and 4x4 (6:9) blocks of `jacobian15`, and
`lam3` are the 3x3-block eigenvalues (used for the complex/hyperbolicity check).
Matches MATLAB: `lam6a = eig(J6(13:15,13:15)); lam4 = eig(J6(6:9,6:9))`.
"""
function _jac15_eig(m15::NTuple{15,<:Real})
    J = jacobian15(m15...)
    if any(!isfinite, J)
        # matches the original: NaN eigenvalues, no hyperbolicity correction triggered
        return NaN, NaN, false
    end
    # 3x3 block J[13:15,13:15]: analytic eigenvalues (real parts + complex flag),
    # passing entries directly (no slice alloc). Replaces eigvals(J[13:15,13:15]).
    r3, has_complex = eig3_realparts(J[13,13], J[13,14], J[13,15],
                                     J[14,13], J[14,14], J[14,15],
                                     J[15,13], J[15,14], J[15,15])
    # 4x4 block J[6:9,6:9]: same LAPACK dgeev as eigvals, via reused-buffer direct
    # call (bit-identical, no slice/workspace/result allocation)
    e4lo, e4hi = jac4_realpart_minmax(J, 6, 6)
    vmin = min(r3[1], e4lo)
    vmax = max(r3[3], e4hi)
    return vmin, vmax, has_complex
end

# MATLAB ~isreal: any nonzero imaginary part (LAPACK returns exact 0 for real eigs)
@inline _has_complex(lam) = any(z -> imag(z) != 0, lam)

"""
    correct_moments_hyperbolic_3D(M)

UNIFIED (2026-07-03): delegates to the single-source `correct_moments_dev`
(src/numerics/moment_correction_dev.jl), shared verbatim with the GPU
wave-speed path. This changed the CPU result by ~1 ulp relative to the
previous autogen-conversion formulation (7e-14 over 200 states, pure
reassociation — verified by a 2000-state battery); the CI golden battery is
tolerance-based (GOLDEN_TOL = 1e-10), so no golden regeneration was needed.

Project the 35-moment vector onto a hyperbolic set (port of the correction in
`eigenvalues6{x,y,z}_hyperbolic_3D.m`): zero the cross 3rd-order standardized
moments, set S112=S110, S121=S101, S211=S011, floor S220/S202/S022 at 1/3,
rebuild the central moments and convert back to raw moments. No realizability
projection is applied here (matches the MATLAB reference).
"""
function correct_moments_hyperbolic_3D(M::AbstractVector)
    return collect(correct_moments_dev(M...))
end

function eigenvalues6_hyperbolic_3D(M::AbstractVector, axis::Int, flag2D::Int, Ma::Real; debug_output=false)
    if axis == 1
        pa, pb = _plane_UV(M), _plane_UW(M)
    else
        pa, pb = _plane_VU(M), _plane_VW(M)
    end
    va_min, va_max, hca = _jac15_eig(pa)
    vb_min, vb_max, hcb = _jac15_eig(pb)
    v6min = min(va_min, vb_min)
    v6max = max(va_max, vb_max)
    Mr = M

    if hca || hcb
        Mr = correct_moments_hyperbolic_3D(M)
        if axis == 1
            qa, qb = _plane_UV(Mr), _plane_UW(Mr)
        else
            qa, qb = _plane_VU(Mr), _plane_VW(Mr)
        end
        a_min, a_max, _ = _jac15_eig(qa)
        b_min, b_max, _ = _jac15_eig(qb)
        v6min = min(a_min, b_min)
        v6max = max(a_max, b_max)
    end

    return v6min, v6max, Mr
end

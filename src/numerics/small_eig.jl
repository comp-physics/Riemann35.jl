import LinearAlgebra.BLAS: @blasfunc, libblastrampoline, BlasInt

# Reused scratch for the 4x4 eigenvalue solve (values only). NOT thread-safe: the
# solver runs one thread per MPI rank, so module-level buffers are safe here. This
# calls the SAME LAPACK dgeev as `eigvals`, bit-for-bit, but with a fixed LWORK
# (skips the per-call workspace query) and no allocation — ~1.2x faster + zero GC.
const _EIG4_A    = Matrix{Float64}(undef, 4, 4)
const _EIG4_WR   = Vector{Float64}(undef, 4)
const _EIG4_WI   = Vector{Float64}(undef, 4)
const _EIG4_WORK = Vector{Float64}(undef, 136)   # LWORK >= 3N; 136 is generous for N=4
const _EIG4_V    = Vector{Float64}(undef, 1)

"""
    jac4_realpart_minmax(J, r0, c0) -> (rmin, rmax)

Min and max real part of the eigenvalues of the 4x4 block `J[r0:r0+3, c0:c0+3]`,
via a direct (reused-buffer, fixed-LWORK) LAPACK dgeev call. Bit-identical to
`sort(real(eigvals(J[r0:r0+3,c0:c0+3])))[[1,4]]`; avoids the slice allocation,
the workspace query, and the result allocation that `eigvals` incurs.
"""
@inline function jac4_realpart_minmax(J, r0::Int, c0::Int)
    @inbounds for jj in 1:4, ii in 1:4
        _EIG4_A[ii, jj] = J[r0+ii-1, c0+jj-1]
    end
    info = Ref{BlasInt}(0)
    ccall((@blasfunc(dgeev_), libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
         Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
        'N', 'N', 4, _EIG4_A, 4, _EIG4_WR, _EIG4_WI, _EIG4_V, 1, _EIG4_V, 1,
        _EIG4_WORK, 136, info, 1, 1)
    @inbounds begin
        lo = _EIG4_WR[1]; hi = _EIG4_WR[1]
        for k in 2:4
            w = _EIG4_WR[k]; lo = ifelse(w < lo, w, lo); hi = ifelse(w > hi, w, hi)
        end
    end
    return lo, hi
end

"""
    eig3_realparts(a11,a12,a13, a21,a22,a23, a31,a32,a33) -> (NTuple{3,Float64}, Bool)

Eigenvalues of a general (non-symmetric) real 3x3 matrix, returned as the sorted
real parts and a `has_complex` flag (true if the matrix has a complex-conjugate
eigenvalue pair). Analytic (Cardano/trigonometric) replacement for `eigvals` on
the small flux-Jacobian blocks, avoiding LAPACK `dgeev`'s tiny-matrix overhead.

The real/complex split is decided by the sign of the characteristic cubic's
discriminant, matching what LAPACK's real Schur form reports (1x1 vs 2x2 blocks)
away from the exact boundary. Used by `_jac15_eig`; see the golden-kernel
regression (debug/golden_kernels.jl), which gates this against the LAPACK path.
"""
@inline function eig3_realparts(a11,a12,a13, a21,a22,a23, a31,a32,a33)
    # DELEGATES to WavespeedDev.eig3_realparts_dev, which is the single source. The two were
    # byte-for-byte identical arithmetic maintained in two files (verified bitwise-equal over
    # 20000 random matrices before unification, including the has_complex discriminant). Only
    # the return PACKING differed: this one nests the roots, the device one returns them flat
    # because a nested tuple is awkward in a GPU kernel.
    #
    # Direction matters: the device form is the source and the host wraps it, never the
    # reverse. Device code cannot call host code (allocation, dynamic dispatch), so a host
    # source would have to be re-ported by hand -- which is exactly how these drifted apart in
    # the first place. See kfvs_wall_flux, where the same two-copy arrangement silently became
    # two different physical models.
    lo, mid, hi, has_complex = WavespeedDev.eig3_realparts_dev(a11,a12,a13, a21,a22,a23, a31,a32,a33)
    return (lo, mid, hi), has_complex
end

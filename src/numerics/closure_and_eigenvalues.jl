"""
    closure_and_eigenvalues(mom)

Compute moment closure and eigenvalue bounds using Chebyshev algorithm.

This function computes the (2N+1)-th order moment from moments of order 0 to 2N
using the Chebyshev algorithm, and also computes the min/max eigenvalues.

# Arguments
- `mom`: Vector of moments of order 0 to 2N (length 2N+1)

# Returns
- `Mp`: Moment of order 2N+1
- `vpmin`: Minimum eigenvalue
- `vpmax`: Maximum eigenvalue
"""
function closure_and_eigenvalues(mom)
    N = div(length(mom) - 1, 2)
    
    # Recurrence coefficients
    sig = zeros(N+2, 2*N+3)
    a = zeros(N+1)
    b = zeros(N+2)
    
    for i = 2:(2*N+2)
        sig[2,i] = mom[i-1]
    end
    
    a[1] = mom[2] / mom[1]
    
    for k = 3:(N+1)
        for l = k:(2*N-k+4)
            sig[k,l] = sig[k-1,l+1] - a[k-2]*sig[k-1,l] - b[k-2]*sig[k-2,l]
        end
        a[k-1] = sig[k,k+1]/sig[k,k] - sig[k-1,k]/sig[k-1,k-1]
        b[k-1] = sig[k,k] / sig[k-1,k-1]
    end
    
    k = N + 2
    sig[k,k] = sig[k-1,k+1] - a[k-2]*sig[k-1,k] - b[k-2]*sig[k-2,k]
    b[k-1] = sig[k,k] / sig[k-1,k-1]

    # Rodney Fox (2026-07): at large Ma, roundoff in the s_k -> m_k change of variables can
    # make the closure coefficient b[N+1] slightly NEGATIVE (unrealizable) although the
    # density is order 1. b[N+1]=0 is the two-delta-function limit, so floor a negative
    # b[N+1] to a tiny positive (~QMOM) rather than forming spurious complex abscissae in
    # the Jacobi matrix below. Matches the reset in his MATLAB closure_and_eigenvalues.m;
    # gives cleaner high-Ma results (RF, Ma=200). Keeps parity with the GPU `closure5_dev`.
    # Only `b[N+1]` (the Jacobi off-diagonal) is affected; the closure moment `Mp` is not.
    if b[N+1] < 0.0
        b[N+1] = 1.0e-10
    end

    # Closure
    a[N+1] = sum(a[1:N]) / N
    
    # Moment of order 2N+1
    sig[N+2,N+3] = sig[N+2,N+2] * (a[N+1] + sig[N+1,N+2]/sig[N+1,N+1])
    for k = (N+2):-1:3
        l = 2*N - k + 5
        sig[k-1,l+1] = sig[k,l] + a[k-2]*sig[k-1,l] + b[k-2]*sig[k-2,l]
    end
    Mp = sig[2,2*N+3]
    
    # Computation of the maximal and minimal values of the eigenvalues
    
    # Setup Jacobi matrix to find roots of R_{n+1}
    b[N+1] = b[N+1] * (2*N+1) / N
    # The Jacobi matrix is SYMMETRIC TRIDIAGONAL: diagonal a, off-diagonal sqrt(b).
    # It is built over ComplexF64 only to tolerate b < 0, which cannot occur for a
    # realizable measure (the b are Hankel/chain coefficients) but does occur on
    # near-vacuum and off-cone states, where the imaginary part is the signal that the
    # state is bad. So: take the real symmetric-tridiagonal path when every b >= 0, and
    # keep the complex path verbatim otherwise.
    #
    # This was the single largest allocation site in the solver (22.4% of all bytes, and
    # the source of the Memory{ComplexF64} at 14.5%): a dense complex eigensolve on a
    # 3x3 tridiagonal whose spectrum is provably real, with real(vp) taken immediately
    # afterwards. eigvals(::SymTridiagonal) allocates ~6.8x less and is backward-stable
    # for this structure.
    #
    # MEASURED BIT-IDENTICAL, which was not the expectation: 9000/9000 returned values
    # (Mp, lo, hi) over 3000 random realizable states compare bitwise-equal to the
    # complex path, max relative difference exactly 0. The prediction going in was that
    # a different LAPACK routine would move the wave speeds at roundoff; it does not, so
    # existing byte-identity baselines still reproduce. Re-run that comparison rather
    # than trusting this note if the LAPACK version changes.
    breal = true
    @inbounds for i = 1:N
        if !(b[i+1] >= 0)
            breal = false
            break
        end
    end

    if breal
        dv = Vector{Float64}(undef, N+1)
        ev = Vector{Float64}(undef, N)
        @inbounds for i = 1:N
            dv[i] = a[i]
            ev[i] = sqrt(b[i+1])
        end
        @inbounds dv[N+1] = a[N+1]
        (all(isfinite, dv) && all(isfinite, ev)) || return Mp, NaN, NaN
        vp = try
            eigvals(SymTridiagonal(dv, ev))
        catch err
            err isa LinearAlgebra.LAPACKException ? (return Mp, NaN, NaN) : rethrow(err)
        end
        return Mp, minimum(vp), maximum(vp)
    end

    z = zeros(ComplexF64, N+1, N+1)
    for i = 1:N
        z[i,i] = a[i]
        # MATLAB's sqrt returns complex for negative input, Julia needs explicit Complex()
        z[i,i+1] = sqrt(Complex(b[i+1]))
        z[i+1,i] = z[i,i+1]
    end
    z[N+1,N+1] = a[N+1]

    # Abscissas via eigenvalues. Degrade to NaN (rather than throwing) on non-finite
    # input or LAPACK non-convergence — both occur for extreme near-vacuum states and
    # are handled downstream like any NaN wave speed, matching the other eigen sites.
    if any(!isfinite, z)
        return Mp, NaN, NaN
    end
    vp = try
        eigvals(z)
    catch err
        err isa LinearAlgebra.LAPACKException ? (return Mp, NaN, NaN) : rethrow(err)
    end
    return Mp, minimum(real(vp)), maximum(real(vp))
end

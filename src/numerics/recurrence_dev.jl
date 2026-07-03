"""
    recurrence_dev.jl — the 5-moment Jacobi recurrence, single source.

Computes the three-term-recurrence coefficients (a1, a2, b2, b3) of the
orthogonal polynomials of a 1D 5-moment set (w1..w5, raw moments), with
R.O. Fox's b3 floor (b3 = c2*H < 0 can occur from roundoff at large Ma; the
floor is the two-delta/QMOM limit). This exact operation order is shared —
byte-identically — by the GPU wave-speed closure (`closure5_dev`) and the
RoePS3 marginal spectrum (`roeps3_dev.jl`). The CPU `closure_and_eigenvalues`
keeps its own MATLAB-parity sigma-algorithm formulation (bit-locked to the
golden battery); the two formulations agree mathematically but not bitwise.

Device-safe plain Julia; no dependencies.
"""
module RecurrenceDev

export recurrence5_dev

@inline function recurrence5_dev(w1, w2, w3, w4, w5)
    a1  = w2 / w1
    s33 = w3 - a1 * w2
    s34 = w4 - a1 * w3
    s35 = w5 - a1 * w4
    a2  = s34 / s33 - w2 / w1
    b2  = s33 / w1
    s44 = s35 - a2 * s34 - b2 * w3
    b3  = s44 / s33
    if b3 < 0.0
        b3 = 1.0e-10
    end
    return a1, a2, b2, b3
end

end # module

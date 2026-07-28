"""
    realizability_S2(S110, S101, S011)

Check and correct realizability of 2nd-order moments.

# Arguments
- S110, S101, S011: Second-order standardized cross-moments

# Returns
- S110r, S101r, S011r: Corrected second-order moments
- S2r: Determinant-like quantity (should be non-negative)

# Algorithm
Computes S2 = 1 + 2*S110*S101*S011 - (S110^2 + S101^2 + S011^2).
If S2 < 0, scales all moments by a factor xr in (0,1) found via root-finding
to make S2 = 0.
"""
function realizability_S2(S110, S101, S011)
    S2 = 1 + 2*S110*S101*S011 - (S110^2 + S101^2 + S011^2)
    xr = 1.0
    
    if S2 < 0
        # Find scaling factor that makes S2 = 0
        Y(x) = 1 + 2*S110*S101*S011*x^3 - (S110^2 + S101^2 + S011^2)*x^2
        
        # Use bisection to find root in [0, 1]
        xr = find_zero_bisection(Y, 0.0, 1.0)

        # Back off strictly inside the cone -- INSIDE the branch, because the bisection
        # lands ON S2 = 0 and only that case needs backing off. Unconditionally it shaved
        # 1e-4 off the velocity correlations on every cell every stage, an artificial
        # shear-stress dissipation accumulating as 1/dt (see realize_dev.jl).
        xr = 0.9999 * xr
    end
    
    S110r = xr * S110
    S101r = xr * S101
    S011r = xr * S011
    S2r = 1 + 2*S110r*S101r*S011r - (S110r^2 + S101r^2 + S011r^2)
    
    if S2r < 0
        @warn "S2 < 0 after correction in realizability_S2" S2 S2r
    end
    
    return S110r, S101r, S011r, S2r
end

# Simple bisection root finder
function find_zero_bisection(f, a, b; tol=1e-12, maxiter=100)
    fa = f(a)
    fb = f(b)
    
    for i in 1:maxiter
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol || (b - a) / 2 < tol
            return c
        end
        
        if sign(fc) == sign(fa)
            a = c
            fa = fc
        else
            b = c
            fb = fc
        end
    end
    
    return (a + b) / 2
end

"""
    collision35(M, dt, Kn)

Apply elastic BGK collision operator to moments.

Relaxes moments toward Maxwellian equilibrium using BGK model:
dM/dt = (MG - M)/tc

# Arguments
- `M`: 35-element moment vector
- `dt`: Time step
- `Kn`: Knudsen number
- `Pr`: Prandtl number (default `1.0` = plain BGK; `2/3` = monatomic, via ES-BGK)
- `omega`: VHS viscosity exponent, `mu ~ T^omega` (default `0.5` = hard spheres)

`Pr = 1.0, omega = 0.5` reproduces the historical operator bitwise. Anything else
delegates to the single-source `ReconDev.bgk_relax_tup`; see
`docs/design/esbgk-vhs-transport.md`.

# Returns
- `Mout`: Updated moments after collision
"""
function collision35(M, dt, Kn; Pr::Real = 1.0, omega::Real = 0.5, stage_bgk_exact::Bool = false)
    # ES-BGK / VHS is single-sourced in `ReconDev.bgk_relax_tup`; delegate rather than
    # duplicate the Isserlis target here. The default path below stays verbatim so the
    # golden (test/test_golden_files.jl) is bit-for-bit unaffected.
    if !(Pr == 1.0 && omega == 0.5 && !stage_bgk_exact)
        return collect(bgk_relax_tup(nt35(Float64.(M)),
                                     Float64(dt), Float64(Kn),
                                     Float64(Pr), Float64(omega), stage_bgk_exact))
    end
    # Extract conserved quantities and compute temperature
    rho = M[1]
    umean = M[2] / rho
    vmean = M[6] / rho
    wmean = M[16] / rho
    
    # Compute temperature from trace of covariance matrix
    C200 = M[3]/rho - umean^2
    C020 = M[10]/rho - vmean^2
    C002 = M[20]/rho - wmean^2
    Theta = (C200 + C020 + C002) / 3
    
    # Safeguard: ensure Theta is positive (numerical errors can make it slightly negative)
    if POSITIVITY_ENABLED[]
        Theta = max(Theta, 1e-14)
    end
    
    # Collision time scale
    tc = Kn / (rho * sqrt(Theta) * 2)
    
    # Maxwellian equilibrium (isotropic covariance)
    MG = InitializeM4_35(rho, umean, vmean, wmean, Theta, 0.0, 0.0, Theta, 0.0, Theta)
    
    # BGK relaxation: dM/dt = (MG - M)/tc
    Mout = MG - exp(-dt/tc) * (MG - M)
    
    return Mout
end

#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# POISEUILLE FLOW RATE vs KNUDSEN NUMBER — the Knudsen minimum.
#
# Force-driven planar channel between two stationary diffuse walls. The dimensionless
# flow rate
#     Q = (1/(H * G)) * integral_0^H u(y) dy ,     G = g*H/(2 R T)   the forcing scale
# is the canonical rarefied benchmark: it FALLS as the gas rarefies out of the continuum
# regime, reaches a MINIMUM near Kn ~ 1, and rises again (logarithmically) into the free
# molecular regime. That non-monotonicity is a genuinely kinetic effect -- Navier-Stokes
# with any slip coefficient gives a monotone curve -- and reproducing it is the standard
# credential for a rarefied method. Reference solutions: Ohwada-Sone-Aoki (linearized
# Boltzmann, hard sphere), Cercignani-Daneri (BGK), Sharipov's tabulations.
#
# We report Q for full-35 and for reduced-26 (the reduction applied per cell per step as
# an operator-split projection), so the same run answers both "does the closure get the
# Knudsen minimum" and "does the reduction change it".
#
# THE FORCE IS APPLIED EXACTLY. A uniform acceleration is a rigid translation in velocity
# space, so it shifts the mean velocity and leaves every central moment alone
# (src/numerics/body_force.jl). No source-term discretization error enters the flow rate,
# which matters here because Q is a small difference of an integral and would happily
# absorb an O(dt) force error and call it physics.
#
# GATE FIRST. Before any flow result, the force operator is checked on a free gas with no
# walls and no gradients, where the exact answer is known: u(t) = g*t and every central
# moment is frozen. If that fails, nothing downstream is meaningful.
#
# Env: PS_KNH, PS_CPL, PS_G, PS_ORDER, PS_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC
using Riemann35: reduce26_S, S_INDEX, DROPPED_KEYS, body_force_shift, apply_body_force!
# `using MPI` is EXPLICIT here. Every test file is included into the same Main, so a
# file that omits it still works as long as some earlier include did `using MPI` --
# an order-dependent coupling that breaks the moment the file is run on its own, or
# the include order changes. See issue #62.
using MPI
MPI.Initialized() || MPI.Init()

const KNH   = [parse(Float64, s) for s in split(envparam("PS_KNH", "0.05,0.1,0.2,0.5,1.0,2.0,5.0"), ",")]
const CPL   = parse(Float64, envparam("PS_CPL", "6.0"))
const NYMIN = parse(Int,     envparam("PS_NYMIN", "24"))
const GACC  = parse(Float64, envparam("PS_G", "0.02"))   # weak: keep the response linear
const ORDER = parse(Int,     envparam("PS_ORDER", "2"))
const TENDF = parse(Float64, envparam("PS_TEND", "0.4"))
# CFL scale factor. Splitting/stage-consistency error scales with dt; a genuine
# dynamical instability of the reduced system does not. Sweeping this separates them.
const CFLF  = parse(Float64, envparam("PS_CFL", "1.0"))

stdz(Mv) = (M2CS4_35(collect(Float64, Mv)))[2]

# ---------------------------------------------------------------------------
# GATE: the body force must be exact on a free gas.
# ---------------------------------------------------------------------------
function gate_body_force()
    M0 = collect(Float64, InitializeM4_35(1.3, 0.2, -0.1, 0.05, 1.1, 0.05, 0.0, 0.9, 0.0, 1.2))
    C0 = M4toC4_3D(M0...)
    g, dt, n = 0.7, 0.013, 37
    M = copy(M0)
    for _ in 1:n
        M = body_force_shift(M, g, 0.0, 0.0, dt)
    end
    C1 = M4toC4_3D(M...)
    du = M[2]/M[1] - (M0[2]/M0[1] + g*dt*n)          # mean must advance by exactly g*t
    dC = maximum(abs, C1 .- C0)                       # central moments must not move
    dm = abs(M[1] - M0[1])
    @printf("GATE body force: mean-velocity error %.3e, max central-moment drift %.3e, mass drift %.3e  %s\n",
            abs(du), dC, dm, (abs(du) < 1e-12 && dC < 1e-10 && dm < 1e-13) ? "PASS" : "FAIL")
    abs(du) < 1e-12 && dC < 1e-10 && dm < 1e-13
end

# ---------------------------------------------------------------------------
"""
    poiseuille(KnH; reduce) -> (Q, ny, drift, u_profile)

Force-driven channel at channel Knudsen number `KnH = lambda_mu/H`, grid refined to hold
`CPL` cells per mean free path. `reduce=true` applies the 26-moment reduction as an
operator-split per-cell projection each step.
"""
function poiseuille(KnH::Float64; reduce::Bool = false, cpl = CPL, order = ORDER,
                    g = GACC, tendf = TENDF, Pr = 2/3, omega = 0.81, alpha = 1.0)
    halo = order == 3 ? 8 : 2
    H    = 1.0
    lam  = KnH*H
    ny   = max(NYMIN, round(Int, cpl/KnH))
    nx   = halo + 2; nz = 1
    dy   = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    rho0, T0 = 1.0, 1.0

    WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=0.0, alpha=alpha),
                   yhi = (Tw=T0, uw1=0.0, uw2=0.0, alpha=alpha))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    tau_ref = lam*sqrt(2.0)
    kn_tau  = 2*tau_ref
    nu      = T0*tau_ref

    # Start from the Navier-Stokes parabola with a slip correction: near the steady answer
    # in the continuum end, harmless in the rarefied end.
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j - halo) - 0.5)*dy, 0.0, H)
        up = g/(2nu)*(yj*(H - yj)) + g*H*1.1*lam/(2nu)
        M[i,j,k,:] = InitializeM4_35(rho0, up, 0.0, 0.0, T0,0,0, T0,0, T0)
    end

    dt   = CFLF*0.2*dy/(5.0*sqrt(T0))
    nst  = ceil(Int, (tendf*H*H/nu)/dt)
    ubar_of(Mf) = mean(Mf[halo+1, halo+j, 1, 2]/Mf[halo+1, halo+j, 1, 1] for j in 1:ny)

    u_early = 0.0
    mark    = max(1, nst - nst÷10)
    for n in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order = order, stage_bgk_kn = kn_tau, stage_bgk_exact = true,
                           Pr = Pr, omega = omega)
        apply_body_force!(M, g, 0.0, 0.0, dt, nx, ny, nz, halo)
        if reduce
            # operator-split projection: overwrite the nine dropped moments by their
            # algebraic closure, per cell, every step
            for kk in 1:nz, jj in (halo+1):(halo+ny), ii in (halo+1):(halo+nx)
                Mv = M[ii,jj,kk,:]
                # M2CS4_35 returns 35-ELEMENT VECTORS of central and standardized
                # moments (not the 5x5x5 arrays that M4toC4_3D returns), so index them
                # through the canonical (i,j,k) -> slot map.
                C, S = M2CS4_35(collect(Float64, Mv))
                R  = reduce26_S(S)
                cg(i,j,k) = C[S_INDEX[(i,j,k)]]
                sx = sqrt(max(cg(2,0,0), eps()))
                sy = sqrt(max(cg(0,2,0), eps()))
                sz = sqrt(max(cg(0,0,2), eps()))
                for key in DROPPED_KEYS            # de-standardize the nine, in place
                    (a,b,c) = key
                    C[S_INDEX[key]] = R[S_INDEX[key]]*sx^a*sy^b*sz^c
                end
                rho = Mv[1]; u = Mv[2]/rho; v = Mv[6]/rho; w = Mv[16]/rho
                A = C4toM4_3D(rho, u, v, w,
                    cg(2,0,0), cg(1,1,0), cg(1,0,1), cg(0,2,0), cg(0,1,1), cg(0,0,2),
                    cg(3,0,0), cg(2,1,0), cg(2,0,1), cg(1,2,0), cg(1,1,1), cg(1,0,2),
                    cg(0,3,0), cg(0,2,1), cg(0,1,2), cg(0,0,3),
                    cg(4,0,0), cg(3,1,0), cg(3,0,1), cg(2,2,0), cg(2,1,1), cg(2,0,2),
                    cg(1,3,0), cg(1,2,1), cg(1,1,2), cg(1,0,3),
                    cg(0,4,0), cg(0,3,1), cg(0,2,2), cg(0,1,3), cg(0,0,4))
                idx = [1,2,3,4,5,6,7,8,9,11,12,13,16,17,21,26,27,28,29,51,52,53,
                       76,77,101,31,32,33,36,37,41,56,57,81,61]
                for q in 1:35; M[ii,jj,kk,q] = A[idx[q]]; end
            end
        end
        n == mark && (u_early = ubar_of(M))
    end
    u_late = ubar_of(M)
    drift  = abs(u_late) > 0 ? abs(u_late - u_early)/abs(u_late) : NaN

    # Dimensionless flow rate. G = g*H/(2 R T) with R T = Theta = 1 here.
    G = g*H/(2*T0)
    Q = u_late/(G*sqrt(2*T0))
    prof = [M[halo+1, halo+j, 1, 2]/M[halo+1, halo+j, 1, 1] for j in 1:ny]
    Q, ny, drift, prof
end

# Every ENV-configurable parameter is printed, because it was NOT: this header used to
# carry g, cells-per-mfp and the order but neither PS_TEND nor PS_CFL, and the published
# 26-moment flow-rate table was produced at PS_TEND=1.2 rather than the default 0.4. Two
# runs differing threefold in marched time emitted identical-looking logs, the parameter had
# to be found by search, and the number it produced was a transient at a tenth of its
# settling time. print_run_header prints exactly what envparam read, so the list cannot
# drift out of step with the code again.
print_run_header("POISEUILLE FLOW RATE vs KNUDSEN NUMBER (the Knudsen minimum)";
                 extra = ("collision" => "ES-BGK, Pr=2/3, omega=0.81", "walls" => "diffuse"))
println("Q should FALL, reach a MINIMUM near Kn_H ~ 1, then RISE. A monotone Q means the")
println("scheme is behaving like Navier-Stokes-with-slip and has missed the kinetic effect.")
println("="^92)
flush(stdout)

if !gate_body_force()
    println("\nBody-force gate FAILED -- flow-rate results below would be meaningless. Stopping.")
    exit(1)
end
println()

@printf("%7s %6s %12s %12s %10s %9s\n", "Kn_H", "ny", "Q full-35", "Q reduced-26", "rel diff", "drift")
flush(stdout)
function run()
    for KnH in KNH
        Qf, ny, drift, _ = poiseuille(KnH; reduce = false)
        Qr, _,  _,     _ = poiseuille(KnH; reduce = true)
        rel = abs(Qf) > 0 ? abs(Qr - Qf)/abs(Qf) : NaN
        @printf("%7.3f %6d %12.5f %12.5f %10.3e %9.1e\n", KnH, ny, Qf, Qr, rel, drift)
        flush(stdout)
    end
end
run()
println("="^100)
println("READING THIS. The Knudsen minimum is the pass/fail for the CLOSURE. The 'rel diff'")
println("column is the pass/fail for the REDUCTION: if it stays at the 1e-3 level the reduction")
println("is invisible in the engineering quantity even where the individual dropped moments")
println("disagree with their closure by 10-20 percent, which would mean the residual measured")
println("in probe_reduce26_residual.jl does not propagate to what anyone reports.")

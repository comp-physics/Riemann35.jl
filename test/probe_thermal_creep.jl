#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# THERMAL CREEP (thermal transpiration).
#
# Gas next to a wall whose temperature varies ALONG the wall slides from cold toward
# hot, with no pressure gradient and no moving surface driving it. There is no
# continuum analogue: Navier-Stokes with no-slip predicts exactly zero flow. It is
# therefore the sharpest single demonstration that a method is doing kinetic physics
# rather than fluid dynamics with a slip correction, and it is the effect behind
# Knudsen pumps -- devices with no moving parts, driven by heat alone.
#
# Classical result (Maxwell): the creep velocity at the wall is
#
#     u_creep = sigma_T * (mu / (rho * T)) * dT/dx ,     sigma_T ~ 0.75 - 1.2
#
# with sigma_T = 3/4 for Maxwell molecules in the simplest treatment. We MEASURE
# sigma_T rather than assume it, exactly as the velocity-slip coefficient was handled
# in validate_wall_slip.jl.
#
# WHY THE WALL TEMPERATURE IS SINUSOIDAL, NOT LINEAR. A linear T along the wall is
# incompatible with periodicity in x: the field would have to jump back at the period
# boundary. Imposing it would need inflow/outflow ends and would contaminate the
# measurement with entrance effects. A sinusoid
#
#     T_w(x) = T0 * (1 + eps*sin(2*pi*x/L))
#
# is periodic by construction, has a clean analytic gradient everywhere, and drives a
# creep flow that reverses sign with dT/dx -- which is itself a strong check, since a
# spurious flow from any asymmetry in the scheme would NOT reverse with the gradient.
# eps is kept small so the response stays linear and sigma_T is well defined.
#
# GATE FIRST: with eps = 0 (isothermal wall) the gas must stay at rest to roundoff. If
# a uniform-temperature wall drives flow, any creep measurement is meaningless.
#
# Env: TC_KNH, TC_NX, TC_NY, TC_EPS, TC_ORDER, TC_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC
MPI.Initialized() || MPI.Init()

const KNH   = parse(Float64, get(ENV, "TC_KNH", "0.1"))
const NX    = parse(Int,     get(ENV, "TC_NX", "32"))
const NY    = parse(Int,     get(ENV, "TC_NY", "24"))
const EPS   = parse(Float64, get(ENV, "TC_EPS", "0.1"))
const ORDER = parse(Int,     get(ENV, "TC_ORDER", "2"))
const TENDF = parse(Float64, get(ENV, "TC_TEND", "2.0"))

"""
    creep(eps) -> (sigma_T, u_max, dTdx_max, drift)

Channel of height H = 1 and length L = 2, walls at y = 0 and y = H, both with
`T_w(x) = T0*(1 + eps*sin(2 pi x / L))`. Periodic in x. No forcing, no wall motion.
"""
function creep(eps::Float64; knh = KNH, nx = NX, ny = NY, order = ORDER, tendf = TENDF)
    halo = order == 3 ? 8 : 2
    H = 1.0; L = 2.0
    lam = knh*H
    nz = 1
    dx = L/nx; dy = H/ny; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    T0 = 1.0; rho0 = 1.0

    # wall temperature per x cell, INCLUDING halo columns (the refill indexes the full
    # transverse extent, halo included)
    xs  = [ ((i - halo) - 0.5)*dx for i in 1:(nx + 2halo) ]
    Twv = [ T0*(1 + eps*sin(2pi*x/L)) for x in xs ]

    WALL_SPEC[] = (ylo = (Tw=Twv, uw1=0.0, uw2=0.0, alpha=1.0),
                   yhi = (Tw=Twv, uw1=0.0, uw2=0.0, alpha=1.0))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    tau_ref = lam*sqrt(2.0)
    nu      = T0*tau_ref

    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        T = Twv[i]                                  # start at the local wall temperature
        M[i,j,k,:] = InitializeM4_35(rho0*T0/T, 0.0, 0.0, 0.0, T,0,0, T,0, T)
    end

    dt  = 0.2*min(dx,dy)/(5.0*sqrt(T0*(1+eps)))
    nst = ceil(Int, (tendf*H*H/nu)/dt)
    ux_of(Mf) = [Mf[halo+i, halo+1, 1, 2]/Mf[halo+i, halo+1, 1, 1] for i in 1:nx]

    u_early = zeros(nx); mark = max(1, nst - nst÷10)
    for n in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order = order, stage_bgk_kn = 2*tau_ref,
                           stage_bgk_exact = true, Pr = 2/3, omega = 0.81)
        n == mark && (u_early = ux_of(M))
    end
    u_late = ux_of(M)
    drift  = maximum(abs, u_late .- u_early) / max(maximum(abs, u_late), 1e-30)

    # analytic wall gradient at each x, and the creep prediction
    dTdx = [ T0*eps*(2pi/L)*cos(2pi*((i-0.5)*dx)/L) for i in 1:nx ]
    # sigma_T from the least-squares slope of u_wall against nu*dTdx/T
    pred = [ nu*dTdx[i]/T0 for i in 1:nx ]
    sT   = sum(pred .* u_late) / max(sum(pred .^ 2), 1e-30)
    sT, maximum(abs, u_late), maximum(abs, dTdx), drift, u_late, pred
end

println("="^96)
println("THERMAL CREEP: flow driven by a wall temperature gradient alone")
@printf("Kn_H=%.2f  nx=%d  ny=%d  eps=%.3f  order=%d  ES-BGK Pr=2/3 omega=0.81\n",
        KNH, NX, NY, EPS, ORDER)
println("Maxwell: u_creep = sigma_T * nu * (dT/dx) / T,  sigma_T ~ 0.75-1.2.")
println("Navier-Stokes with no-slip predicts ZERO flow here, so any nonzero u is kinetic.")
println("="^96)

println("\nGATE: isothermal wall (eps = 0) must drive NO flow.")
let (sT0, umax0, _, _, _, _) = creep(0.0)
    @printf("  max|u| with a uniform wall = %.3e   %s\n", umax0,
            umax0 < 1e-12 ? "PASS" : "FAIL -- a uniform wall is driving flow")
end

println("\nMEASUREMENT")
@printf("%8s %14s %14s %12s %10s\n", "eps", "max|u_wall|", "max|dT/dx|", "sigma_T", "drift")
for eps in (EPS, EPS/2)
    sT, umax, gmax, drift, ul, pr = creep(eps)
    @printf("%8.3f %14.5e %14.5e %12.4f %10.1e\n", eps, umax, gmax, sT, drift)
    flush(stdout)
end

println("\n" * "="^96)
println("READING THIS. sigma_T in the 0.75-1.2 band means the moment wall reproduces")
println("thermal creep at the right strength -- a genuinely kinetic effect, with no")
println("continuum counterpart, captured without a velocity grid. sigma_T near zero means")
println("the closure cannot represent the wall heat-flux/stress coupling that drives it.")
println("Halving eps must leave sigma_T unchanged; if it does not, the response is not")
println("linear and the coefficient is not yet well defined.")

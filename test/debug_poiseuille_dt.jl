#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# WHY IS FULL-35 POISEUILLE NOT dt-CONVERGED?
#
# Observed: at fixed physical end time, the steady flow rate MOVES under timestep
# refinement, and the moves GROW (+2.1% then +4.2% for CFL 1.0 -> 0.5 -> 0.25). A
# converging scheme halves them. Growth under refinement is the signature of a
# per-step error that does NOT scale with dt: total error ~ nst * eps ~ eps*T/dt,
# which grows as dt shrinks.
#
# PRIME SUSPECT. The Maxwell wall ghost is only an approximate fixed point. Its
# equilibrium gate (test/test_wall_bc.jl W2) measures 3.3e-7 relative, not roundoff,
# because rho_w comes from a half-space mass balance evaluated with erfc_dev -- an
# Abramowitz-Stegun rational approximation good to ~1e-7 absolute. A gas at rest
# against a wall at its own temperature should be EXACTLY stationary; if instead each
# step injects ~1e-7 of momentum, then over nst steps the injection is ~nst*1e-7, and
# halving dt doubles nst and doubles the injected momentum. That is the observed sign
# and the observed scaling.
#
# THIS SCRIPT SEPARATES THE CANDIDATES by removing one ingredient at a time. Each case
# has an EXACT known answer, so any drift is unambiguous:
#
#   A. periodic everywhere + body force, no walls
#        exact: spatially uniform, u(t) = g*t, all central moments frozen.
#        isolates force + transport + collision. Drift here => not the wall.
#   B. walls + NO force, gas at rest at the wall temperature
#        exact: nothing happens, ever.
#        isolates the wall. Drift here, growing with step count => the wall is the leak.
#   C. same as B but with alpha = 0 (pure specular)
#        specular is an EXACT sign flip with no erfc and no rho_w, so if B drifts and
#        C does not, the leak is specifically the diffuse half-space closure, and
#        erfc_dev is the concrete suspect.
#
# Case B is run at three step counts at FIXED dt: a per-step leak gives drift growing
# LINEARLY in step count, which distinguishes it from a one-off transient.
#
# Env: DB_KNH, DB_NY, DB_ORDER
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC, apply_body_force!
MPI.Initialized() || MPI.Init()

const KNH   = parse(Float64, get(ENV, "DB_KNH", "1.0"))
const NY    = parse(Int,     get(ENV, "DB_NY", "24"))
const ORDER = parse(Int,     get(ENV, "DB_ORDER", "2"))

function setup(; walls::Bool, alpha = 1.0, order = ORDER, ny = NY)
    halo = order == 3 ? 8 : 2
    H = 1.0; lam = KNH*H
    nx = halo + 2; nz = 1
    dy = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    T0 = 1.0
    if walls
        WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=0.0, alpha=alpha),
                       yhi = (Tw=T0, uw1=0.0, uw2=0.0, alpha=alpha))
        BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)
    else
        WALL_SPEC[] = nothing
        BC = (xlo=:periodic, xhi=:periodic, ylo=:periodic, yhi=:periodic, zlo=:outflow, zhi=:outflow)
    end
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    M0 = InitializeM4_35(1.0, 0.0, 0.0, 0.0, T0,0,0, T0,0, T0)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo); M[i,j,k,:] = M0; end
    tau_ref = lam*sqrt(2.0)
    (M=M, decomp=decomp, BC=BC, nx=nx, ny=ny, nz=nz, halo=halo,
     dx=dx, dy=dy, dz=dz, kn_tau=2*tau_ref, dt=0.2*dy/5.0)
end

"mean x-momentum and mass over the interior"
function diag(s)
    px = 0.0; m = 0.0; mxu = 0.0
    for j in (s.halo+1):(s.halo+s.ny)
        rho = s.M[s.halo+1, j, 1, 1]; mom = s.M[s.halo+1, j, 1, 2]
        px += mom; m += rho; mxu = max(mxu, abs(mom/rho))
    end
    px/s.ny, m/s.ny, mxu
end

march!(s, nst; g = 0.0) = begin
    for _ in 1:nst
        step_highorder_3d!(s.M, s.dt, s.decomp, s.BC, s.nx, s.ny, s.nz, s.halo,
                           s.dx, s.dy, s.dz, 0.0; order = ORDER,
                           stage_bgk_kn = s.kn_tau, stage_bgk_exact = true,
                           Pr = 2/3, omega = 0.81)
        g != 0 && apply_body_force!(s.M, g, 0.0, 0.0, s.dt, s.nx, s.ny, s.nz, s.halo)
    end
end

println("="^100)
println("DIAGNOSING THE dt-NON-CONVERGENCE OF FULL-35 POISEUILLE")
@printf("Kn_H=%.2f  ny=%d  order=%d  dt=%.3e\n", KNH, NY, ORDER, setup(walls=false).dt)
println("="^100)

# ---- A: no walls, body force. Exact: u = g*t, uniform. ----
println("\nA. PERIODIC + BODY FORCE (no walls).  Exact: u(t) = g*t, uniform, moments frozen.")
@printf("%10s %16s %16s\n", "steps", "u error vs g*t", "max|u - mean u|")
for nst in (500, 2000, 8000)
    s = setup(walls=false); g = 0.02
    march!(s, nst; g = g)
    px, m, _ = diag(s)
    u = px/m
    spread = maximum(abs(s.M[s.halo+1,j,1,2]/s.M[s.halo+1,j,1,1] - u)
                     for j in (s.halo+1):(s.halo+s.ny))
    @printf("%10d %16.3e %16.3e\n", nst, abs(u - g*s.dt*nst), spread)
    flush(stdout)
end

# ---- B: walls, NO force, gas at rest. Exact: nothing happens. ----
println("\nB. DIFFUSE WALLS, NO FORCE, gas at rest at T_wall.  Exact: stationary forever.")
println("   A per-step leak gives drift growing LINEARLY in step count.")
@printf("%10s %16s %16s %16s\n", "steps", "mean px", "px per step", "mass drift")
for nst in (500, 2000, 8000)
    s = setup(walls=true, alpha=1.0)
    _, m0, _ = diag(s)
    march!(s, nst)
    px, m, mxu = diag(s)
    @printf("%10d %16.3e %16.3e %16.3e\n", nst, px, px/nst, abs(m - m0))
    flush(stdout)
end

# ---- C: same, pure specular. No erfc, no rho_w -> exact by construction. ----
println("\nC. SPECULAR WALLS (alpha=0), NO FORCE.  Specular is an exact sign flip:")
println("   no erfc, no rho_w. If B leaks and C does not, the diffuse closure is the cause.")
@printf("%10s %16s %16s %16s\n", "steps", "mean px", "px per step", "mass drift")
for nst in (500, 2000, 8000)
    s = setup(walls=true, alpha=0.0)
    _, m0, _ = diag(s)
    march!(s, nst)
    px, m, mxu = diag(s)
    @printf("%10d %16.3e %16.3e %16.3e\n", nst, px, px/nst, abs(m - m0))
    flush(stdout)
end

println("\n" * "="^100)
println("READING THIS.")
println("  A drifts            -> the leak is in force/transport/collision, not the wall.")
println("  B drifts ~ linearly in steps, C clean -> the DIFFUSE wall closure leaks per step;")
println("                         erfc_dev's ~1e-7 accuracy is the concrete suspect, and the")
println("                         fix is an accurate erfc on the CPU path.")
println("  B and C both drift  -> the leak is in the ghost-refill machinery common to both.")
println("  Nothing drifts      -> the dt-dependence is elsewhere; look at the force/wall")
println("                         INTERACTION, which only case C-with-force would expose.")

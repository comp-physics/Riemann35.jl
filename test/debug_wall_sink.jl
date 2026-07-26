#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# IS THE WALL'S MOMENTUM SINK O(dt) OR O(dt^2)?
#
# Poiseuille's steady flow rate satisfies Q ~ 1/nu_eff, and the measured Q obeys
# Q = Q0 + a/dt (fitted exponent p = -1 from CFL 1.0/0.5/0.25). So the effective
# momentum sink vanishes linearly with dt. The only sink in a force-driven channel is
# the wall, which points at the wall extracting O(dt^2) of momentum per step where it
# should extract O(dt).
#
# EVERYTHING ELSE IS ALREADY EXCLUDED:
#   * wall ghost at equilibrium is EXACT (0.0, not 3.3e-7 -- that number belongs to a
#     different test), so a state with no momentum to remove is a perfect fixed point;
#   * the skewness cap never fires (max|S300| = 3.7e-4 against a 4.0 cap);
#   * transport + collision + body force with NO walls reproduces u = g*t to 1e-16.
#
# THE TEST. No long march, no steady state, no convergence question. Take ONE state
# with real momentum in it, apply ONE step, and measure how much x-momentum the walls
# removed. Then halve dt and repeat. The extraction RATE dpx/dt must be dt-INDEPENDENT:
#
#     rate constant in dt      -> sink is O(dt), correct; the bug is elsewhere
#     rate falls in proportion -> sink is O(dt^2), the wall is one power of dt too weak,
#                                 which reproduces Q ~ 1/dt exactly
#
# A single step also removes every confound at once: no accumulation, no steady state,
# no splitting over many steps, and the initial condition is identical in every run.
#
# Env: WS_KNH, WS_NY, WS_ORDER
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC
MPI.Initialized() || MPI.Init()

const KNH   = parse(Float64, get(ENV, "WS_KNH", "1.0"))
const NY    = parse(Int,     get(ENV, "WS_NY", "24"))
const ORDER = parse(Int,     get(ENV, "WS_ORDER", "2"))

"identical initial state every time: a parabolic x-velocity profile between two walls"
function build(; walls::Bool, order = ORDER, ny = NY, u0 = 0.05)
    halo = order == 3 ? 8 : 2
    H = 1.0; lam = KNH*H
    nx = halo + 2; nz = 1
    dy = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    T0 = 1.0
    if walls
        WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=0.0, alpha=1.0),
                       yhi = (Tw=T0, uw1=0.0, uw2=0.0, alpha=1.0))
        BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)
    else
        WALL_SPEC[] = nothing
        BC = (xlo=:periodic, xhi=:periodic, ylo=:periodic, yhi=:periodic, zlo=:outflow, zhi=:outflow)
    end
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j - halo) - 0.5)*dy, 0.0, H)
        ux = 4*u0*yj*(H - yj)/H^2                 # parabola, zero at both walls
        M[i,j,k,:] = InitializeM4_35(1.0, ux, 0.0, 0.0, T0,0,0, T0,0, T0)
    end
    (M=M, decomp=decomp, BC=BC, nx=nx, ny=ny, nz=nz, halo=halo,
     dx=dx, dy=dy, dz=dz, kn_tau=2*lam*sqrt(2.0))
end

"total interior x-momentum"
totpx(s) = sum(s.M[s.halo+1, j, 1, 2] for j in (s.halo+1):(s.halo+s.ny))

function one_step(dt; walls)
    s = build(walls = walls)
    p0 = totpx(s)
    step_highorder_3d!(s.M, dt, s.decomp, s.BC, s.nx, s.ny, s.nz, s.halo,
                       s.dx, s.dy, s.dz, 0.0; order = ORDER,
                       stage_bgk_kn = s.kn_tau, stage_bgk_exact = true,
                       Pr = 2/3, omega = 0.81)
    p1 = totpx(s)
    p1 - p0
end

println("="^92)
println("WALL MOMENTUM SINK: is it O(dt) or O(dt^2)?")
@printf("Kn_H=%.2f  ny=%d  order=%d   parabolic profile, walls at rest, NO body force\n",
        KNH, NY, ORDER)
println("A correct sink removes momentum at a dt-INDEPENDENT RATE. dpx/dt must be flat.")
println("="^92)

dt0 = 0.2*(1.0/NY)/5.0
@printf("\n%12s %16s %16s %14s\n", "dt", "dpx (walls)", "rate dpx/dt", "rate ratio")
prev = NaN
for f in (1.0, 0.5, 0.25, 0.125)
    dt = f*dt0
    d  = one_step(dt; walls = true)
    r  = d/dt
    @printf("%12.4e %16.6e %16.6e %14s\n", dt, d, r,
            isnan(prev) ? "--" : @sprintf("%.4f", r/prev))
    global prev = r
    flush(stdout)
end

println("\nCONTROL: same profile, PERIODIC in y (no walls). Momentum is conserved exactly,")
println("so any dpx here is not a wall effect and would invalidate the test above.")
@printf("%12s %16s\n", "dt", "dpx (no walls)")
for f in (1.0, 0.25)
    dt = f*dt0
    @printf("%12.4e %16.6e\n", dt, one_step(dt; walls = false))
    flush(stdout)
end

println("\n" * "="^92)
println("READING THIS. 'rate ratio' is dpx/dt at this dt divided by the previous one.")
println("  ratio ~ 1.0  -> sink is O(dt): correct, and the Q ~ 1/dt divergence is NOT the wall.")
println("  ratio ~ 0.5  -> sink is O(dt^2): the wall weakens as dt falls, momentum piles up,")
println("                  and Q ~ 1/dt follows immediately. That is the bug.")

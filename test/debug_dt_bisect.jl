#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# LOCALIZE THE dt-DEPENDENT STEADY STATE IN FULL-35 WALL-BOUNDED FLOW.
#
# THE DEFECT. Force-driven planar Poiseuille reaches a steady state whose value depends
# on the timestep. At Kn_H = 1, ny = 24, both ends time-converged:
#     CFL 1.00 -> Q = 0.9953  (drift 1.4e-6 at TENDF 6)
#     CFL 0.25 -> Q = 1.0629  (drift 1.2e-4 at TENDF 6)
# 6.8% apart, and the gap grows as dt falls (fitted error ~ 1/dt). A consistent scheme
# converges to ONE steady state; this one does not.
#
# WHY THIS SCRIPT EXISTS RATHER THAN ANOTHER HYPOTHESIS. Five mechanisms have been
# proposed and each killed by direct test: the wall ghost is an exact fixed point (0.0),
# the skewness cap never fires (max|S300| = 3.7e-4 vs a 4.0 cap), transport + collision +
# body force with no walls reproduces u = g*t to 1e-16, the wall momentum sink scales
# correctly as O(dt) (rate ratios 0.990, 0.995, 0.997 -> 1), and the runs are genuinely
# time-converged. Guessing mechanisms is 0-for-5, so stop guessing: BISECT.
#
# METHOD. Run the same configuration at two timesteps, both marched to convergence, and
# report r = Q(CFL/4) / Q(CFL). r = 1 means the steady state is timestep-independent.
# Toggle ONE ingredient per row. The row where r collapses to 1 contains the defect.
#
# Row 0 is the most informative and is deliberately first: wall-DRIVEN Couette with no
# body force at all. If Couette is dt-clean, the defect needs the force, and the measured
# slip coefficients stand. If Couette is dt-dirty too, the defect is wall+shear and the
# slip results are implicated as well.
#
# Env: BI_KNH, BI_NY, BI_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC, apply_body_force!
MPI.Initialized() || MPI.Init()

const KNH  = parse(Float64, get(ENV, "BI_KNH", "1.0"))
const NY   = parse(Int,     get(ENV, "BI_NY",  "24"))
const TEND = parse(Float64, get(ENV, "BI_TEND","6.0"))

"""
Run one configuration to steady state and return the mean interior x-velocity.

`mode = :poiseuille` (stationary walls + body force) or `:couette` (walls at +/-Uw, no
force). Every other argument is a solver ingredient we can switch off.
"""
function run_case(cflf; mode = :poiseuille, order = 2, exact_bgk = true,
                  Pr = 2/3, omega = 0.81, force_split = :exact, g = 0.02, Uw = 0.1)
    halo = order == 3 ? 8 : 2
    H = 1.0; lam = KNH*H; ny = NY
    nx = halo + 2; nz = 1
    dy = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    T0 = 1.0; rho0 = 1.0
    uw = mode === :couette ? Uw : 0.0
    gg = mode === :couette ? 0.0 : g

    WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=-uw, alpha=1.0),
                   yhi = (Tw=T0, uw1=0.0, uw2=+uw, alpha=1.0))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    tau_ref = lam*sqrt(2.0); nu = T0*tau_ref
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j - halo) - 0.5)*dy, 0.0, H)
        u0 = mode === :couette ? uw*(2yj/H - 1)/(1 + 2*1.1*KNH) :
                                 gg/(2nu)*(yj*(H-yj)) + gg*H*1.1*lam/(2nu)
        M[i,j,k,:] = InitializeM4_35(rho0, u0, 0.0, 0.0, T0,0,0, T0,0, T0)
    end

    dt  = cflf*0.2*dy/(5.0*sqrt(T0))
    nst = ceil(Int, (TEND*H*H/nu)/dt)
    ubar(Mf) = mean(Mf[halo+1, halo+j, 1, 2]/Mf[halo+1, halo+j, 1, 1] for j in 1:ny)

    # explicit source form of the body force, for comparison against the exact
    # velocity-space translation: dM_ijk/dt += g * i * M_{(i-1)jk}. First order in dt
    # and NOT realizability-preserving, which is why the exact form is the default.
    idxm1 = Dict((i,j,k) => nothing for (i,j,k) in [(0,0,0)])
    function src_force!(Mf, gx, dtl)
        @inbounds for kk in 1:nz, jj in (halo+1):(halo+ny), ii in (halo+1):(halo+nx)
            # only x-moments gain: M_{ijk} += dt*g*i*M_{(i-1)jk}
            Mf[ii,jj,kk,2] += dtl*gx*1*Mf[ii,jj,kk,1]     # M100 += dt g M000
            Mf[ii,jj,kk,3] += dtl*gx*2*Mf[ii,jj,kk,2]     # M200 += dt g 2 M100
            Mf[ii,jj,kk,4] += dtl*gx*3*Mf[ii,jj,kk,3]     # M300
            Mf[ii,jj,kk,5] += dtl*gx*4*Mf[ii,jj,kk,4]     # M400
            Mf[ii,jj,kk,7] += dtl*gx*1*Mf[ii,jj,kk,6]     # M110
            Mf[ii,jj,kk,17] += dtl*gx*1*Mf[ii,jj,kk,16]   # M101
        end
    end

    d_early = 0.0; mark = max(1, nst - nst÷10)
    for n in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order = order, stage_bgk_kn = 2*tau_ref,
                           stage_bgk_exact = exact_bgk, Pr = Pr, omega = omega)
        if gg != 0
            force_split === :exact ? apply_body_force!(M, gg, 0.0, 0.0, dt, nx, ny, nz, halo) :
                                     src_force!(M, gg, dt)
        end
        n == mark && (d_early = ubar(M))
    end
    u = ubar(M)
    u, abs(u) > 0 ? abs(u - d_early)/abs(u) : NaN
end

println("="^104)
println("BISECTING THE dt-DEPENDENT STEADY STATE (full-35, wall-bounded)")
@printf("Kn_H=%.2f  ny=%d  TENDF=%.1f  --  r = u(CFL/4)/u(CFL); r = 1 means dt-INDEPENDENT\n",
        KNH, NY, TEND)
println("The row where r collapses to 1 contains the defect.")
println("="^104)
@printf("%-42s %12s %12s %8s %9s\n", "configuration", "u(CFL=1)", "u(CFL=0.25)", "r", "drift")

cases = [
 ("0. COUETTE (walls move, NO body force)", (mode=:couette,)),
 ("1. baseline Poiseuille (as measured)",   (mode=:poiseuille,)),
 ("2. + stage_bgk_exact OFF",               (mode=:poiseuille, exact_bgk=false)),
 ("3. + plain BGK (Pr=1, omega=1/2)",       (mode=:poiseuille, Pr=1.0, omega=0.5)),
 ("4. + order 1 reconstruction",            (mode=:poiseuille, order=1)),
 ("5. + force as explicit source",          (mode=:poiseuille, force_split=:source)),
]
function main()
    for (label, kw) in cases
        u1, d1 = run_case(1.0;  kw...)
        u4, d4 = run_case(0.25; kw...)
        r = abs(u1) > 0 ? u4/u1 : NaN
        @printf("%-42s %12.6f %12.6f %8.4f %9.1e\n", label, u1, u4, r, max(d1,d4))
        flush(stdout)
    end
end
main()
println("="^104)
println("Row 0 is the pivot: if Couette is dt-clean the defect needs the body force, and the")
println("measured slip coefficients are unaffected. If Couette is dirty too, it is wall+shear")
println("and the slip results are implicated as well.")

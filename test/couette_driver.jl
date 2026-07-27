# ---------------------------------------------------------------------------
# Shared planar-Couette driver for the wall-bounded micro-flow probes.
#
# The sweep variable is the CHANNEL Knudsen number Kn_H = lambda_mu/H, the quantity
# the micro-flow literature reports -- NOT the solver's tau-scaling parameter. With
# tau_ref = kn_tau/2 and Theta = 1,
#     lambda_mu = tau_ref*sqrt(Theta/2) = kn_tau/(2 sqrt 2),
# so a tau parameter of 2.0 in a channel of height 5 is Kn_H = 0.14 (early transition),
# not free-molecular. Sweeping kn_tau at a fixed grid also silently destroys resolution:
# cells per mean free path fall like kn_tau, burying the whole Knudsen layer inside one
# cell at the near-continuum end. Here the grid refines with Kn_H so cells-per-lambda is
# held fixed and every point in a sweep is equally resolved.
# ---------------------------------------------------------------------------
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC

const SIGP = 1.1   # Maxwell slip coefficient; used ONLY to pick a good initial guess

"""
    couette_field(KnH; ...) -> (column, ny, nst, want, drift, kn_tau, cells_per_lambda)

Planar Couette at channel Knudsen number `KnH = lambda_mu/H`, grid refined so that
`lambda/dy ~ cpl`. Length unit is the channel height (`H = 1`), hence `lambda = KnH`
and `ny = cpl/KnH`. Returns the interior column of raw 35-moment vectors.

`drift` is the relative change of the interior shear rate over the last tenth of the
march. It is REPORTED, never asserted.

WARNING -- `drift` IS A WEAK CONVERGENCE INDICATOR AND HAS ALREADY MISLED ONCE. The
approach to steady state here is diffusive and monotone, so the change over the last
tenth of a march can be small while the solution is still far from the answer. A
Poiseuille run at Kn_H = 1 reported drift = 1.8e-2 and was 22% away from its converged
flow rate; the reduction difference read 1.4% and was really 17%. Use `drift` only to
catch gross non-convergence, and establish march length by an actual sweep in `tendf`
(see the convergence study in probe_poiseuille.jl) before quoting any number.
"""
function couette_field(KnH::Float64; cpl = 6.0, nymin = 24, Uw = 0.1, order = 2,
                       alpha = 1.0, tendf = 0.3, maxst = 400_000,
                       Pr = 2/3, omega = 0.81, cflf = 1.0)
    halo = order == 3 ? 8 : 2
    H    = 1.0
    lam  = KnH*H
    ny   = max(nymin, round(Int, cpl/KnH))
    nx   = halo + 2; nz = 1
    dy   = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    rho0, T0 = 1.0, 1.0

    WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=-Uw, alpha=alpha),
                   yhi = (Tw=T0, uw1=0.0, uw2=+Uw, alpha=alpha))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    tau_ref = lam*sqrt(2.0)
    kn_tau  = 2*tau_ref
    nu      = T0*tau_ref

    # Slip-corrected linear initial guess. The steady state differs from it only by the
    # Knudsen layer, so the transient is short; a no-slip or uniform start would need a
    # full diffusion time to relax the global slope change and cost ~1e6 steps at the
    # near-continuum end.
    A = Uw/(1 + 2*SIGP*KnH)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = ((j - halo) - 0.5)*dy
        ux = A*(2*clamp(yj/H, 0.0, 1.0) - 1.0)
        M[i,j,k,:] = InitializeM4_35(rho0, ux, 0.0, 0.0, T0,0,0, T0,0, T0)
    end

    # cflf scales the timestep: splitting/consistency error scales with dt, a genuine
    # dt-dependent steady state does not. Sweeping it separates the two.
    dt   = cflf*0.2*dy/(5.0*sqrt(T0))
    want = ceil(Int, (tendf*H*H/nu)/dt)
    nst  = min(maxst, want)

    dudy_of(Mf) = begin
        u = [Mf[halo+1, halo+j, 1, 2]/Mf[halo+1, halo+j, 1, 1] for j in 1:ny]
        lo = max(1, ny÷4); hi = min(ny, 3*ny÷4)
        (u[hi] - u[lo])/((hi-lo)*dy)
    end

    d_early = 0.0
    mark    = max(1, nst - nst÷10)
    for n in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order = order, stage_bgk_kn = kn_tau, stage_bgk_exact = true,
                           Pr = Pr, omega = omega)
        n == mark && (d_early = dudy_of(M))
    end
    d_late = dudy_of(M)
    drift  = abs(d_late) > 0 ? abs(d_late - d_early)/abs(d_late) : NaN

    [M[halo+1, halo+j, 1, :] for j in 1:ny], ny, nst, want, drift, kn_tau, lam/dy
end

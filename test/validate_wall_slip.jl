#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# MEASURES the velocity-slip coefficient produced by the wall ghost BC.
#
# WHY THIS IS A MEASUREMENT, NOT A PASS/FAIL. The wall ghost is exact for specular
# reflection (a sign flip on odd-normal-order moments) but approximates the DIFFUSE half
# by assuming the interior distribution is Gaussian when computing the incoming mass
# flux. A 35-moment vector cannot encode a half-space split, so that assumption is the
# one real approximation in the design (see src/numerics/wall_ghost_dev.jl).
#
# Maxwell slip theory gives, for accommodation alpha,
#     u_slip = zeta * lambda * (du/dy)|_wall ,    zeta = ((2-alpha)/alpha) * sigma_p
# with sigma_p ~ 1.0-1.1 for a fully diffuse wall. Whether THIS closure reproduces that
# coefficient is an open question the design deliberately left to measurement: if it is
# badly off, the escalation is a prescribed wall flux (KFVS-at-wall), which would have to
# touch the residual and the theta*-IDP anchor.
#
# So: report zeta, do not assert it. The only hard gates here are the ones that must hold
# regardless -- alpha=0 must give ZERO slip resistance (free slip), and the interior
# profile must be linear.
#
# lambda uses the SHARED convention  lambda_mu = mu/(rho sqrt(2RT)) = tau_ref*sqrt(Theta/2),
# the same one bridged to SPARTA in ../dsmc, so the number is comparable across codes.
#
# Env: WS_KN, WS_NY, WS_H, WS_UW, WS_ALPHA, WS_ORDER, WS_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC
# `using MPI` is EXPLICIT here. Every test file is included into the same Main, so a
# file that omits it still works as long as some earlier include did `using MPI` --
# an order-dependent coupling that breaks the moment the file is run on its own, or
# the include order changes. See issue #62.
using MPI
MPI.Initialized() || MPI.Init()

const KN    = parse(Float64, get(ENV, "WS_KN", "1.0"))
const NY    = parse(Int,     get(ENV, "WS_NY", "40"))
const HH    = parse(Float64, get(ENV, "WS_H", "5.0"))
const UW    = parse(Float64, get(ENV, "WS_UW", "0.2"))
const ORDER = parse(Int,     get(ENV, "WS_ORDER", "2"))
const TENDF = parse(Float64, get(ENV, "WS_TEND", "3.0"))

"run Couette to steady state; return (y, u_x, dudy_interior, u_wall_gas)"
function couette(alpha::Float64; Kn = KN, ny = NY, H = HH, Uw = UW, order = ORDER)
    halo = order == 3 ? 8 : 2
    nx = max(halo + 2, 8); nz = 1
    dy = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    rho0, T0 = 1.0, 1.0
    # walls on y (axis 2): for that axis the tangential slots are (uw1,uw2) = (z, x),
    # so the x-directed wall motion goes in uw2.
    WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=-Uw, alpha=alpha),
                   yhi = (Tw=T0, uw1=0.0, uw2=+Uw, alpha=alpha))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    M = zeros(nx+2halo, ny+2halo, nz, 35)
    M0 = InitializeM4_35(rho0, 0.0, 0.0, 0.0, T0,0,0, T0,0, T0)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo); M[i,j,k,:] = M0; end

    tau_ref = (Kn/2) / rho0                    # omega=0.5, Theta=1
    nu      = T0*tau_ref
    dt      = 0.2*dy/(5.0*sqrt(T0))
    nst     = ceil(Int, (TENDF*H*H/nu)/dt)     # a few viscous diffusion times
    for _ in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order = order, stage_bgk_kn = Kn, stage_bgk_exact = true)
    end
    yc = [(j-0.5)*dy for j in 1:ny]
    ux = [M[halo+1, halo+j, 1, 2]/M[halo+1, halo+j, 1, 1] for j in 1:ny]
    lo = ny÷4; hi = 3*ny÷4                     # interior only
    ys = yc[lo:hi]; us = ux[lo:hi]; m = length(ys)
    sx = sum(ys); sy = sum(us)
    dudy = (m*sum(a*b for (a,b) in zip(ys,us)) - sx*sy)/(m*sum(a*a for a in ys) - sx^2)
    # linearity of the interior fit (R^2)
    ub = sy/m; pred = [ub + dudy*(yy - sx/m) for yy in ys]
    ss_res = sum((a-b)^2 for (a,b) in zip(us,pred)); ss_tot = sum((a-ub)^2 for a in us)
    (yc, ux, dudy, ux[end], ux[1], 1 - ss_res/max(ss_tot,1e-30), nst, dt, tau_ref)
end

println("="^88)
println("WALL SLIP COEFFICIENT (measurement, not a pass/fail)")
@printf("Kn=%.3f  H=%.1f  ny=%d  Uw=%.3f  order=%d\n", KN, HH, NY, UW, ORDER)
lam = (KN/2)*sqrt(1.0/2)          # lambda_mu = tau_ref*sqrt(Theta/2), Theta=1
@printf("lambda_mu = tau_ref*sqrt(Theta/2) = %.4f   lambda/H = %.4f\n", lam, lam/HH)
println("Maxwell slip: u_slip = zeta*lambda*(du/dy)|wall,  zeta=((2-a)/a)*sigma_p, sigma_p~1.0-1.1")
println("="^88)
@printf("%-7s %12s %12s %10s %12s %10s\n", "alpha", "du/dy", "u_gas(wall)", "R^2", "u_slip", "zeta")
for alpha in (1.0, 0.75, 0.5, 0.0)
    yc, ux, dudy, uhi, ulo, r2, nst, dt, tref = couette(alpha)
    slip = UW - uhi                       # wall moves at +Uw; gas lags by the slip
    zeta = abs(dudy) > 0 ? slip/(lam*abs(dudy)) : NaN
    @printf("%-7.2f %12.5e %12.5e %10.6f %12.5e %10.4f\n", alpha, dudy, uhi, r2, slip, zeta)
end
println("="^88)
println("HARD GATES (must hold regardless of the slip coefficient):")
let (_, ux, dudy, uhi, ulo, r2) = couette(0.0)
    freeslip = abs(dudy) < 1e-8 * UW / HH
    @printf("  alpha=0 gives FREE SLIP (du/dy ~ 0): %s   (du/dy=%.3e)\n",
            freeslip ? "PASS" : "FAIL", dudy)
    @printf("  specular wall drives no shear, so the gas stays at rest: max|u|=%.3e\n",
            maximum(abs, ux))
end

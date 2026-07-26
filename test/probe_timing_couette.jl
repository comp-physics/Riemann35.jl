#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# WALL-CLOCK of the moment solver on the SAME Couette problem SPARTA is timed on.
#
# The standing claim behind this whole program is that the method is "an accurate and
# FAST alternative to DSMC." Accuracy has been measured (shock thickness, transport
# coefficients, slip). Cost has never been measured at all. This is that measurement.
#
# MATCHED SETUP: channel Knudsen number Kn_H = lambda/H = 0.1, ~6 cells per mean free
# path (ny = 60, the same grid SPARTA gets), fully diffuse isothermal walls at +/- Uw,
# Uw/sqrt(2 R T) ~ 0.148 to match the deck's 50 m/s against argon's 337 m/s thermal
# speed. Single core on both sides.
#
# WHY LOW SPEED IS THE HONEST PLACE TO COMPARE, AND ALSO THE FAVOURABLE ONE. DSMC's
# statistical error falls as 1/sqrt(N) while the signal it must resolve falls with the
# Mach number, so the sample count needed for a fixed RELATIVE precision grows like
# 1/Ma^2. A moment method has no such penalty -- its cost is Mach-independent. So the
# fair way to state a speedup is not "seconds vs seconds" but
#
#     time DSMC needs to reach the moment solver's accuracy   vs   the moment solver's time
#
# which is what the companion SPARTA runs (two seeds, same everything) are for: the
# seed-to-seed scatter gives DSMC's actual noise floor at a known cost, and noise
# scales as 1/sqrt(t), so the time to any target precision follows.
#
# This script deliberately reports its OWN accuracy too (the steady-state drift), so a
# fast number that has not converged cannot be mistaken for a win.
#
# Env: TC_KNH, TC_CPL, TC_UW, TC_ORDER, TC_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
include(joinpath(@__DIR__, "couette_driver.jl"))
MPI.Initialized() || MPI.Init()

const KNH   = parse(Float64, get(ENV, "TC_KNH", "0.1"))
const CPL   = parse(Float64, get(ENV, "TC_CPL", "6.0"))
const UW    = parse(Float64, get(ENV, "TC_UW", "0.21"))   # matches 50 m/s / 337 m/s
const ORDER = parse(Int,     get(ENV, "TC_ORDER", "2"))
const TENDF = parse(Float64, get(ENV, "TC_TEND", "0.3"))

println("="^96)
println("MOMENT-SOLVER WALL CLOCK on the SPARTA Couette problem")
@printf("Kn_H=%.3f, ~%.0f cells/lambda, Uw/sqrt(2RT)=%.3f, order=%d, ES-BGK Pr=2/3 omega=0.81\n",
        KNH, CPL, UW/sqrt(2.0), ORDER)
println("="^96)

# warm up so JIT compilation is not counted as solver cost (SPARTA has no equivalent,
# so including it would be a straightforward unfairness)
couette_field(1.0; cpl = 4.0, nymin = 12, Uw = UW, order = ORDER, tendf = 0.002)

t0 = time()
col, ny, nst, want, drift, kn_tau, cpl_act =
    couette_field(KNH; cpl = CPL, Uw = UW, order = ORDER, tendf = TENDF)
el = time() - t0

dy = 1.0/ny
u  = [col[j][2]/col[j][1] for j in 1:ny]
lo = max(1, ny÷4); hi = min(ny, 3*ny÷4)
ys = [(j-0.5)*dy for j in lo:hi]; us = u[lo:hi]; m = length(ys)
sx = sum(ys); sy = sum(us)
dudy = (m*sum(a*b for (a,b) in zip(ys,us)) - sx*sy)/(m*sum(a*a for a in ys) - sx^2)
slip = UW - u[end]

"""
Cells the solver actually touches. The driver builds a 3D array with `nx = halo+2`
interior cells in x plus `2*halo` ghosts, and `ny + 2*halo` in y. For a y-only problem
EVERY x cell is redundant, so the x-extent is pure harness overhead -- at order 2 that
is a factor of 8. Dividing wall time by `nst*ny` (the useful cells) silently attributes
that overhead to the per-cell cost and inflates it 8x; dividing by the true count is the
honest per-cell-step figure, and the ratio between them is the harness tax.
"""
total_cells(order, ny) = (h = order == 3 ? 8 : 2; (h + 2 + 2h) * (ny + 2h))
tc = total_cells(ORDER, ny)
@printf("grid           : ny=%d  (%.1f cells per lambda)\n", ny, cpl_act)
@printf("cells touched  : %d  (%d useful in y; %.1fx harness overhead from the idle x-extent)\n",
        tc, ny, tc/ny)
@printf("steps          : %d of %d requested\n", nst, want)
@printf("WALLCLOCK_SEC  : %.2f\n", el)
@printf("per cell-step  : %.2f us  (true); %.2f us if the idle x-cells are charged to it\n",
        1e6*el/(nst*tc), 1e6*el/(nst*ny))
@printf("1D-equivalent  : %.1f s  <- cost if the harness did not carry the idle x-extent\n",
        el*ny/tc)
@printf("du/dy interior : %.6e\n", dudy)
@printf("slip velocity  : %.6e   (%.2f%% of Uw)\n", slip, 100*slip/UW)
@printf("steady drift   : %.2e   <- the solver's OWN error bar; a fast unconverged\n", drift)
println("                              answer is not a win, so this is reported first-class.")
println("="^96)
println("To compare: DSMC noise falls as 1/sqrt(t), so from the seed-to-seed scatter s at")
println("cost t_dsmc, the time to reach relative precision p is t_dsmc*(s/p)^2. Put the")
println("moment solver's own drift in for p to get the like-for-like ratio.")

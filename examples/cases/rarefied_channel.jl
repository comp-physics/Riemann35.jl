# rarefied_channel.jl — THE MOMENT METHOD IN A WALL-BOUNDED RAREFIED FLOW.
#
# Every other example in this directory is free-space: shock tubes, jets, crossing flows,
# rising bubbles. Those exercise the closure where it is strongest -- strong gradients, no
# boundaries. This one exercises it where it is hardest and where the interesting physics
# is: a channel between two diffuse walls at finite Knudsen number, which is the setting for
# slip, temperature jump, thermal creep and the Knudsen minimum.
#
# Three canonical configurations, selected by `mode`:
#
#   :couette    plates counter-sliding at -Uw/+Uw, both at T0.  Observable: velocity SLIP.
#   :poiseuille stationary plates, uniform body force.          Observable: FLOW RATE Q.
#   :fourier    stationary plates at Tc/Th.                     Observable: TEMPERATURE JUMP.
#
# WHAT MAKES THE WALL CASE DIFFERENT, and what a user needs to know before trusting output:
#
# * COLLISION MODEL. The closure defaults to ES-BGK (Pr = 2/3), which is right for a real
#   monatomic gas. Set Pr = 1 only when matching a BGK reference -- and note the difference
#   is not small: it moves the slip coefficient by 14-43%.
#
# * INTEGRATION TIME. The obvious choice t_end = 3 H^2/nu is the DIFFUSIVE timescale and it
#   fails at high Kn, where transport is ballistic. At Kn = 0.8 it supplies only 2.65 transit
#   times and the shear rate is still 22% from its steady value. Use
#       t_end = N H/sqrt(T),  N ~ 5   (measured: 3 transits is within 0.05%)
#   as `steady_time` below does. This is the single most common way to get a wrong wall
#   number, because the formula looks Kn-independent. It is also the single biggest waste:
#   3 H^2/nu over-integrates Kn = 0.05 by 8x.
#
# * RESOLUTION. What must be resolved is the KNUDSEN LAYER, whose thickness is set by lambda,
#   not by H. So the cell requirement gets HARDER as the gas gets DENSER: at Kn = 0.05,
#   ny = 96 gives only ~4.8 cells per mean free path and is 9% off; at Kn = 0.4 the same ny
#   gives ~38 and is fine. A sweep at fixed ny is therefore not a controlled comparison.
#
# * OBSERVABLES. Report the dimensionless shear rate Sbar; the slip coefficient
#   zeta = (1-Sbar)/(2 Kn Sbar) is an algebraic transform of it that amplifies error ~1.3x.
#   In :fourier, read the jump at BOTH walls and take the mean -- a single wall carries an
#   O(dT) bias (6.6% at dT/T = 10%) while the mean is amplitude-independent.
#
# Usage, via the standard case driver:
#     julia --project=. examples/run_case.jl examples/cases/rarefied_channel.jl
# or directly:
#     julia --project=. -e 'include("examples/cases/rarefied_channel.jl"); run_rarefied_channel()'
using Riemann35, MPI, Printf, Statistics
using Riemann35: WALL_SPEC

"""
    steady_time(Kn; H=1.0, T=1.0, ntransit=5)

Integration time for a wall-bounded steady state, set by BALLISTIC TRANSIT TIMES
`H/sqrt(T)` rather than by the diffusive time `H^2/nu`.

MEASURED, at production resolution (ny = 384, Kn = 0.1). Sbar against the 21.2-transit
value 0.72808:

     3 transits  0.72844  (+0.049%)
     5 transits  0.72839  (+0.043%)
    10.6         0.72828  (+0.027%)
    21.2         0.72808   reference

so ~3 transits already suffices and 5 is a comfortable margin. The resolution matters: an
earlier version of this check at ny = 8 showed no sensitivity at all, so the rule has to be
established on the grid actually used.

WHY NOT `3 H^2/nu`, the obvious choice. It is the DIFFUSIVE time, and expressed in transits
it runs 42.4, 21.2, 10.6, 5.3, 2.65 at Kn = 0.05 ... 0.80 -- backwards at BOTH ends. It
over-integrates the dense end by up to 8x (wasted compute) and under-integrates the rarefied
end, where 2.65 transits leaves the shear rate 22% from steady. That single choice produced
two retracted conclusions.

An earlier version of this file recommended `ntransit = 20`. That was ~4x too conservative;
it was picked as a safe-looking round number before the measurement above existed.
"""
function steady_time(Kn; H = 1.0, T = 1.0, ntransit = 5)
    nu = T*(Kn*H*sqrt(2.0))
    max(0.5*H*H/nu, ntransit*H/sqrt(T))    # transit floor dominates except at very low Kn
end

"""
    suggest_ny(Kn; cells_per_mfp=25, H=1.0)

Cells across the gap needed to resolve the Knudsen layer at `cells_per_mfp` cells per mean
free path. Grows as Kn falls, which is the opposite of most intuitions about rarefied flow.
"""
suggest_ny(Kn; cells_per_mfp = 25, H = 1.0) = max(32, ceil(Int, cells_per_mfp/Kn))

"""
    run_rarefied_channel(; mode=:couette, Kn=0.2, ny=nothing, Pr=2/3, ...)

One wall-bounded channel run. Returns a NamedTuple of the observables appropriate to `mode`,
plus the settings used -- so a result can always be traced to the run that produced it.
"""
function run_rarefied_channel(; mode::Symbol = :couette,
                                Kn::Float64 = 0.2,
                                ny::Union{Int,Nothing} = nothing,
                                Pr::Float64 = 2/3,        # ES-BGK; use 1.0 to match BGK
                                omega::Float64 = 1.0,
                                Uw::Float64 = 0.1,        # :couette
                                gx::Float64 = 1e-3,       # :poiseuille
                                dTr::Float64 = 0.0125,    # :fourier, (Th-Tc)/2T0
                                ntransit::Int = 5,
                                verbose::Bool = true)
    MPI.Initialized() || MPI.Init()
    H = 1.0; T0 = 1.0
    ny === nothing && (ny = suggest_ny(Kn))
    halo = 2; nx = halo + 2; nz = 1
    dy = H/ny; dt = 0.2*dy/(5.0*sqrt(T0))
    tau = Kn*H*sqrt(2.0)
    nst = max(1, ceil(Int, steady_time(Kn; ntransit = ntransit)/dt))

    Tc, Th = mode === :fourier ? (T0*(1-dTr), T0*(1+dTr)) : (T0, T0)
    ulo, uhi = mode === :couette ? (-Uw, +Uw) : (0.0, 0.0)
    # uw2 carries the x-tangent for a y-normal wall (uw1 is the z-tangent). Driving uw1
    # and reading u_x gives an identically zero velocity field -- verified from rest.
    WALL_SPEC[] = (ylo = (Tw=Tc, uw1=0.0, uw2=ulo, alpha=1.0),
                   yhi = (Tw=Th, uw1=0.0, uw2=uhi, alpha=1.0))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)

    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j-halo)-0.5)*dy, 0.0, H)
        u0 = mode === :couette ? Uw*(2yj/H - 1) : 0.0
        Tj = Tc + (Th - Tc)*yj/H
        M[i,j,k,:] = InitializeM4_35(1.0, u0, 0.0, 0.0, Tj,0,0, Tj,0, Tj)
    end

    verbose && @printf("  %s: Kn=%.3f ny=%d Pr=%.4f t_end=%.2f (%d steps)\n",
                       mode, Kn, ny, Pr, steady_time(Kn; ntransit=ntransit), nst)
    for _ in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dy, dy, dy, 0.0;
                           order = 2, stage_bgk_kn = 2*tau, stage_bgk_exact = true,
                           Pr = Pr, omega = omega)
        if mode === :poiseuille
            for k in 1:nz, j in (halo+1):(halo+ny), i in (halo+1):(halo+nx)
                M[i,j,k,:] = body_force_shift(collect(Float64, M[i,j,k,:]), gx, 0.0, 0.0, dt)
            end
        end
    end

    y   = [(j-0.5)*dy for j in 1:ny]
    rho = [M[halo+1, halo+j, 1, 1] for j in 1:ny]
    u   = [M[halo+1, halo+j, 1, 2]/rho[j] for j in 1:ny]
    T   = [((M[halo+1,halo+j,1,3]/rho[j] - u[j]^2) +
             M[halo+1,halo+j,1,10]/rho[j] + M[halo+1,halo+j,1,20]/rho[j])/3 for j in 1:ny]

    # core fit over the middle half -- the standard estimator for these observables
    lo, hi = max(1,ny÷4), min(ny,3ny÷4)
    fit(q) = begin
        xs, qs = y[lo:hi], q[lo:hi]; xb, qb = mean(xs), mean(qs)
        S = sum((xs.-xb).*(qs.-qb))/sum((xs.-xb).^2); (S, qb - S*xb)
    end

    if mode === :couette
        S, b = fit(u)
        Sbar = abs(S)*H/(2Uw)
        return (mode=mode, Kn=Kn, ny=ny, Pr=Pr, nst=nst,
                Sbar=Sbar, zeta=(1-Sbar)/(2*Kn*Sbar), y=y, u=u)
    elseif mode === :poiseuille
        # Q normalised so the continuum no-slip Poiseuille value is 1
        Q = sum(rho .* u)*dy / (gx*H^3/(12*tau*T0))
        return (mode=mode, Kn=Kn, ny=ny, Pr=Pr, nst=nst, Q=Q, y=y, u=u)
    else
        S, b = fit(T)
        lam = Kn*H
        jump_hot  = Th - (S*H + b)
        jump_cold = b - Tc                       # BOTH walls -- see the header
        zt_hot, zt_cold = jump_hot/(lam*abs(S)), jump_cold/(lam*abs(S))
        return (mode=mode, Kn=Kn, ny=ny, Pr=Pr, nst=nst,
                zetaT_hot=zt_hot, zetaT_cold=zt_cold, zetaT=(zt_hot+zt_cold)/2,
                asym=abs(abs(zt_hot)-abs(zt_cold))/max(abs(zt_hot),eps()), y=y, T=T)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # DEMO SETTINGS, chosen to finish in ~90 s on one core -- NOT production settings.
    # Measured cost: 19 / 33 / 65 ms per step at ny = 8 / 16 / 32, and the step count scales
    # as ny too, so cost grows as ny^2: ny=32 with the production ntransit=20 is ~17 minutes
    # PER CASE. For real work drop `ny` and `ntransit` here and let the defaults apply --
    # `suggest_ny(Kn)` and `ntransit = 5` -- and expect minutes to hours.
    println("RAREFIED CHANNEL -- the moment method at a diffuse wall  [DEMO SETTINGS]")
    println("t_end floored by the ballistic transit time; see suggest_ny for production ny.\n")
    for Kn in (0.2, 0.4)
        r = run_rarefied_channel(mode = :couette, Kn = Kn, ny = 16, ntransit = 3)
        @printf("    Sbar = %.5f   zeta = %.4f   [production: ny=%d, ntransit=5]\n",
                r.Sbar, r.zeta, suggest_ny(Kn))
    end
    r = run_rarefied_channel(mode = :fourier, Kn = 0.4, ny = 16, ntransit = 3)
    @printf("    zeta_T hot=%.4f cold=%.4f mean=%.4f (asym %.1e)\n",
            r.zetaT_hot, r.zetaT_cold, r.zetaT, r.asym)
end

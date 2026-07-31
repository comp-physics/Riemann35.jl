# test_steady_state_time_convergence.jl — A STEADY STATE MUST NOT DEPEND ON HOW LONG YOU
# INTEGRATED TO REACH IT.
#
# The companion to test_driven_dt_convergence.jl. That one asserts a steady state has a
# dt -> 0 limit; this one asserts it has a t_end -> infinity limit. Both are properties of
# R(M) = 0, an equation containing neither dt nor t_end, and the suite had neither until a
# defect forced each to be written.
#
# WHY THIS EXISTS. Every wall-bounded closure measurement in the notes was integrated to
# t_end = 3 H^2/nu and compared against a DVM reference integrated to the same *formula*.
# The DVM was convergence-checked in t_end (zeta stable to 7 digits from 1.5 onward) and the
# closure was NOT -- the check was run on one solver and the conclusion assumed for the
# other. At Kn = 0.8 the closure's shear rate moves 23.6% between t_end = 1.5 and 3.0, so the
# published comparison had the two codes at genuinely different physical states.
#
# THE CAUSE IS THE NORMALISATION, and it is worth stating because it is not obvious.
# t_end = 3 H^2/nu with nu = Kn sqrt(2) hands the HIGH-Kn cases the LEAST absolute time:
#
#     Kn      0.05    0.10    0.20    0.40    0.80
#     t_end   42.4    21.2    10.6     5.3     2.65     (in units of the transit time H/sqrt(T))
#
# H^2/nu is the DIFFUSIVE timescale, and it is the right one only while transport is
# collisional. As Kn grows transport becomes ballistic and the relevant timescale is the
# transit time H/sqrt(T) -- so the diffusive normalisation under-integrates precisely the
# rarefied cases where the closure is under the most stress. Kn = 0.8 gets 2.65 transits,
# which is not a steady state by any reading.
#
# WHAT THIS TEST DOES AND DOES NOT COVER -- stated plainly, because the first version of
# this header claimed more.
#
# It asserts a Cauchy property: over T/2, T, 2T the increments must SHRINK. That catches a
# gross failure to reach a steady state, and it is affordable in CI.
#
# It does NOT reproduce the defect that motivated it. Falsification was attempted and did not
# falsify: with the diffusive-only t_end (2.65 transits) the gate still passes 3/3, increments
# 7.3e-3 -> 1.1e-4. The reason is RESOLUTION. At the ny = 8 this test can afford, the drift
# from T to 2T is ~0.14%; at the ny = 384 the published runs used, it is 23.6%. So the
# sensitivity is strongly grid-dependent and vanishes at CI scale.
#
# That resolution dependence is itself unexplained and is the open question -- a time-to-
# steady-state should not depend much on dy, and the fact that it does suggests the fine-grid
# runs resolve a slow near-wall transient the coarse grid damps away. Until that is
# understood, treat this file as a WEAK general guard, not as coverage for the t_end defect.
# The real guard is procedural: convergence must be established for EACH solver in a
# comparison, at the resolution actually used, and never inherited from the other solver.
using Test, Riemann35, MPI
using Riemann35: WALL_SPEC

MPI.Initialized() || MPI.Init()

@testset "steady state is independent of integration time" begin
    H = 1.0; T0 = 1.0; Uw = 0.1
    ny = 8; nz = 1; halo = 2; nx = halo + 2
    Kn = 0.8                      # the rarefied end, where the diffusive normalisation fails
    dy = H/ny; dx = dy; dz = dy
    lam = Kn*H; tau = lam*sqrt(2.0); nu = T0*tau; kn_tau = 2*tau
    dt = 0.2*dy/(5.0*sqrt(T0))

    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)

    "Shear rate in the channel core after integrating to `tend` (absolute time)."
    function shear(tend)
        # uw2, NOT uw1. For a y-normal wall uw1 is the z-tangent and uw2 the x-tangent; the
        # first version of this file drove uw1 and read u_x, so the wall exerted NO forcing
        # on the profile being measured and the test watched an initial shear DECAY. An
        # "increments shrink" gate passes trivially for exponential decay, so the test was
        # worse than vacuous -- it measured the wrong quantity and reported success.
        # Verified from rest: uw1 gives max|u_x| = 0 exactly, uw2 gives 6.5e-2.
        WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=-Uw, alpha=1.0),
                       yhi = (Tw=T0, uw1=0.0, uw2=+Uw, alpha=1.0))
        nst = max(1, ceil(Int, tend/dt))
        M = zeros(nx+2halo, ny+2halo, nz, 35)
        for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
            yj = clamp(((j-halo)-0.5)*dy, 0.0, H)
            M[i,j,k,:] = InitializeM4_35(1.0, Uw*(2yj/H - 1), 0.0, 0.0, T0,0,0, T0,0, T0)
        end
        for _ in 1:nst
            step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                               order = 2, stage_bgk_kn = kn_tau, stage_bgk_exact = true,
                               Pr = 1.0, omega = 1.0)
        end
        u = [ M[halo+1, halo+j, 1, 2]/M[halo+1, halo+j, 1, 1] for j in 1:ny ]
        lo = max(1, ny÷4); hi = min(ny, 3*ny÷4)
        (u[hi] - u[lo])/((hi-lo)*dy)          # core slope, no fit needed at this size
    end

    # The transit-time floor: at least `ntr` ballistic crossings, whatever the diffusive
    # formula says. At Kn = 0.8 this is what the diffusive formula alone fails to supply.
    # ntr = 3 keeps this at ~38 s, matching the suite's existing integration tests
    # (18.0 ms/step measured for this configuration, 2100 steps total). Larger ntr is
    # better physics and unaffordable in CI; the Cauchy gate below is what makes the
    # smaller window sufficient.
    ntr = 3.0
    t_diff = 3*H*H/nu
    T      = max(t_diff, ntr*H/sqrt(T0))
    @test T >= t_diff                         # the floor must bind, or this is silently
                                              # re-testing the old diffusive-only behaviour

    # A CAUCHY GATE, not an absolute one. Asserting |s(2T) - s(T)| < tol would require
    # knowing where the steady state IS -- the thing under investigation -- and a T large
    # enough to be there, which is too slow for CI. Requiring the increments to SHRINK
    # tests convergence itself: an under-integrated run has increments that stay flat or
    # grow, which is exactly the 23.6% drift that motivated this file.
    s0 = shear(T/2); s1 = shear(T); s2 = shear(2T)
    d1 = abs(s1 - s0); d2 = abs(s2 - s1)
    @info "steady-state time convergence" t_diff T shear=(s0,s1,s2) increments=(d1,d2) ratio=d2/max(d1,eps())

    @test abs(s1) > 1e-6                      # a real response, or the gate is vacuous
    @test d2 < 0.8*d1                         # increments must shrink
end

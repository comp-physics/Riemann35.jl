# test_wall_conservation.jl — a wall is impermeable, so it must pass ZERO net mass.
#
# This test exists because that was violated for a long time without anything noticing.
# The wall density was set from the continuum half-space balance,
#     rho_w = influx / sqrt(Tw/2pi)   =>   rho_w = rho*sqrt(T/Tw)  at u_n = 0,
# which is correct physics but not something the ghost-cell discretisation can impose: the
# wall face solves a FULL-LINE Riemann problem, and at u_n = 0 the HLL flux reduces to pure
# upwind dissipation acting on the density jump rho_w - rho. A physically-correct rho_w
# therefore manufactures a mass flux through an impermeable wall.
#
# It was invisible to every existing check. The equilibrium-wall gate uses T = Tw, where
# rho_w = rho and the bug vanishes identically. The slip and Couette diagnostics read a
# SHEAR RATE, which sat flat at 0.137 across two march-length doublings (changing 1.6% then
# 0.61% -- any convergence criterion accepts that) while the mass error underneath grew
# exponentially. Interior mass reached 1.9e11 by t = 6, and the growth rate was timestep
# independent (e-folding 0.1723 at dt vs 0.1717 at dt/2), i.e. an eigenvalue of the
# semi-discrete operator with positive real part.
#
# THE LESSON THIS TEST ENCODES: check a CONSERVED quantity, not the observable of interest.
# Viscous heating makes T > Tw permanent in any sheared wall-bounded flow, so the source is
# permanent and self-amplifying, but nothing that looks at velocity will see it early.

using Test, Printf
using Riemann35
using Riemann35.WallGhostDev: wall_ghost_tup
# `using MPI` is EXPLICIT here. Every test file is included into the same Main, so a
# file that omits it still works as long as some earlier include did `using MPI` --
# an order-dependent coupling that breaks the moment the file is run on its own, or
# the include order changes. See issue #62.
using MPI
MPI.Initialized() || MPI.Init()

@testset "wall conservation (impermeability)" begin

    # ---------------------------------------------------------------------------
    # Unit level: the mass flux across a wall face must be identically zero.
    # T != Tw is the case that used to fail; alpha = 0 and T = Tw are the cases that
    # always passed, kept here so a regression can be localised rather than just seen.
    # ---------------------------------------------------------------------------
    @testset "zero mass flux across the wall face" begin
        for axis in 1:3, outward in (-1.0, 1.0)
            for T in (0.5, 0.8, 1.0, 1.1, 1.5, 2.5), alpha in (0.0, 0.1, 0.5, 1.0)
                Tw = 1.0
                M  = InitializeM4_35(1.0, 0.0, 0.0, 0.0, T,0,0, T,0, T)
                Mt = NTuple{35,Float64}(M)
                G  = wall_ghost_tup(Mt, axis, outward, Tw, 0.0, 0.0, alpha)
                F  = face_flux_1d(collect(Mt), collect(G), axis, 1.0)
                @test abs(F[1]) < 1e-13
            end
        end
    end

    # Tangential wall motion must not pump mass either: it is parallel to the wall.
    @testset "tangential wall motion passes no mass" begin
        for axis in 1:3, alpha in (0.0, 0.5, 1.0)
            M  = InitializeM4_35(1.0, 0.0, 0.0, 0.0, 1.0,0,0, 1.0,0, 1.0)
            Mt = NTuple{35,Float64}(M)
            G  = wall_ghost_tup(Mt, axis, 1.0, 1.0, 0.15, -0.2, alpha)
            F  = face_flux_1d(collect(Mt), collect(G), axis, 1.0)
            @test abs(F[1]) < 1e-13
        end
    end

    # ---------------------------------------------------------------------------
    # The specular limit is measure-preserving, so it conserves mass EXACTLY for any
    # interior state -- including strongly non-equilibrium ones where the Gaussian
    # reasoning behind the old wall density does not apply at all. This is the control
    # that localised the bug to the diffuse half: it holds at alpha = 0 no matter what.
    # ---------------------------------------------------------------------------
    @testset "specular limit is exact for non-equilibrium states" begin
        for axis in 1:3
            M = InitializeM4_35(1.3, 0.05, -0.03, 0.02, 1.4,0.1,0.0, 0.9,-0.05, 1.1)
            Mt = NTuple{35,Float64}(M)
            G  = wall_ghost_tup(Mt, axis, 1.0, 2.0, 0.0, 0.0, 0.0)   # alpha=0, Tw ignored
            F  = face_flux_1d(collect(Mt), collect(G), axis, 1.0)
            @test abs(F[1]) < 1e-13
        end
    end

    # ---------------------------------------------------------------------------
    # Integrated: the sheared channel must not have an UNSTABLE mass mode.
    #
    # Note carefully what is and is not asserted. Exact conservation is NOT achievable in
    # this formulation and asserting it would be dishonest: with rho_w = rho the dissipative
    # term vanishes, but when the near-wall normal velocity u_n is nonzero the physical HLL
    # flux terms leave a residual ~ [sR + sL(1-alpha)]*rho*u_n/(sR - sL). The ghost-cell form
    # reflects only part of the incident normal stream once alpha > 0, and no choice of
    # ghost DENSITY can cancel that; it needs a genuine half-space (KFVS-at-wall) flux.
    # Measured residual over 200 steps: ~1.2e-6 at alpha=0.25, ~1.8e-5 at alpha=1.
    #
    # What broke the solver was not that leak, it was that the leak GREW EXPONENTIALLY:
    # viscous heating held T > Tw, which held rho_w > rho, which injected mass, which
    # heated further -- a semi-discrete eigenvalue with positive real part (lambda ~ 5.8 in
    # H^2/nu, e-folding 0.172, timestep independent). So the property to regress on is the
    # GROWTH LAW, not the magnitude: doubling the march must roughly double the drift
    # (linear accumulation), not multiply it by e^(lambda*dt).
    #
    # A pure threshold would be the wrong test twice over: it would have to be loose enough
    # to admit the residual, and once loose it would pass for a long time during an
    # exponential blowup -- which is exactly how this defect survived.
    # ---------------------------------------------------------------------------
    @testset "no exponentially growing wall mass mode" begin
        halo = 2; ny = 24; nx = halo + 2; nz = 1
        H = 1.0; dy = H/ny; dx = dy; dz = dy
        decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
        rho0 = 1.0; T0 = 1.0; Uw = 0.2          # strong shear => strong viscous heating

        function drift_after(alpha, nsteps)
            Riemann35.WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=-Uw, alpha=alpha),
                                     yhi = (Tw=T0, uw1=0.0, uw2=+Uw, alpha=alpha))
            BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall,
                  zlo=:outflow, zhi=:outflow)
            M = zeros(nx+2halo, ny+2halo, nz, 35)
            for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
                yj = ((j - halo) - 0.5)*dy
                ux = Uw*(2*clamp(yj/H, 0.0, 1.0) - 1.0)
                M[i,j,k,:] = InitializeM4_35(rho0, ux, 0.0, 0.0, T0,0,0, T0,0, T0)
            end
            mass(Mf) = sum(Mf[halo+1, halo+j, 1, 1] for j in 1:ny)
            m0 = mass(M)
            dt = 0.2*dy/(5.0*sqrt(T0))
            for _ in 1:nsteps
                step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                                   order=2, stage_bgk_kn=0.5, stage_bgk_exact=true,
                                   Pr=2/3, omega=0.81)
            end
            abs(mass(M) - m0)/m0
        end

        # A NOTE ON WHAT THIS CAN AND CANNOT CHECK. The first version of this test compared
        # drift at 200 vs 400 steps and demanded a ratio near 2 (linear accumulation). That
        # was a bad test: 200 steps here is tendf ~ 0.08, entirely inside the startup
        # transient, where the profile is still developing and u_n at the wall is large. The
        # measured ratio is 7.4 (alpha=0.25) and 8.3 (alpha=1) even with the defect fixed --
        # transient growth, not an unstable mode. Asymptotically the drift IS linear, but
        # reaching the asymptotic regime takes tendf ~ 1, i.e. thousands of steps, which is
        # too slow for CI.
        #
        # So the strong, exact check here is the UNIT one above: the wall mass flux is
        # identically zero at T != Tw, which is the defect itself. This integrated case is a
        # bounded-drift SMOKE TEST. It is deliberately not the primary evidence, because a
        # magnitude threshold loose enough to admit the transient would also pass for a long
        # time during an exponential blowup -- exactly how the original defect survived.
        #
        # The stability evidence proper is the march-length convergence study, which lives
        # in the probe scripts because it takes ~13 min: post-fix the Couette shear rate
        # converges monotonically (0.13924, 0.13694, 0.13704, 0.13700 at tendf = 0.3, 0.6,
        # 1.2, 2.4; successive changes 11.3%, 1.7%, 0.074%, 0.028%), where pre-fix the same
        # sequence ended at 0.29155, a 52.7% jump, on its way to mass 1.9e11 by tendf = 6.
        for alpha in (0.25, 1.0)
            d = drift_after(alpha, 400)
            @info "wall mass drift (smoke)" alpha d
            @test d < 1e-3
        end
    end
end

# test_dvm_reference.jl — THE DVM REFERENCE MUST BE TRUSTWORTHY, BECAUSE EVERYTHING IS
# MEASURED AGAINST IT.
#
# The DVM-BGK solver is the ground truth for the closure: the slip coefficient, the
# temperature jump, the reduced-26 comparisons and the shock profiles are all quoted as
# closure-versus-DVM. It spent its life as a script in the research repo with no tests, and
# that is exactly how it acquired a defect that a single assertion would have caught:
# `transport!` accepted a `bc` keyword and IGNORED it. The wrap-around branch did not exist,
# so `bc=:periodic` silently produced zero-gradient.
#
# No published result was affected -- every caller used the default, and reduce26_shear.jl
# needed periodicity, noticed, and wrote its own wrapping transport rather than trusting the
# kwarg. But the failure mode is the dangerous one: the code did something reasonable and
# WRONG, silently, for anyone who did trust it. A GPU port later compared true-periodic
# against zero-gradient and read a 7.5e-2 discrepancy, which looks exactly like a broken port.
#
# The tests below are properties of the physics, not recorded outputs. Behaviour-pinning
# goldens are what let the realizability defect survive for the life of the project -- twelve
# of them recorded the buggy values as correct and their response to the fix was to fail.
using Test, Riemann35, LinearAlgebra
using Riemann35.DVMBGK: VGrid, moments35, maxwellian!, discrete_maxwellian!,
                        collide_cell!, transport!, rho_u_T

@testset "DVM-BGK reference solver" begin
    # A coarse grid: these are exact identities, not convergence claims, so resolution only
    # needs to be enough for the quadrature to represent a Maxwellian at T = 1 (vmax = 5 is
    # 5 thermal speeds).
    g = VGrid(5.0, 12)

    # ---- the quadrature recovers the moments of a Maxwellian ---------------------------
    # rho, u and T must come back out of moments35 to quadrature accuracy. If this fails
    # nothing downstream means anything.
    @testset "moment recovery" begin
        for (rho, ux, uy, T) in ((1.0, 0.0, 0.0, 1.0), (2.0, 0.3, -0.2, 0.8))
            f = zeros(g.n, g.n, g.n)
            maxwellian!(f, rho, ux, uy, 0.0, T, g)
            r, x, y, z, Tm, _ = rho_u_T(f, g)
            @test r  ≈ rho atol=1e-6
            @test x  ≈ ux  atol=1e-6
            @test y  ≈ uy  atol=1e-6
            @test Tm ≈ T   atol=1e-6
        end
    end

    # ---- the DISCRETE Maxwellian matches its invariants EXACTLY on the grid -------------
    # This is the entire reason the Mieussens construction is used instead of sampling the
    # continuous Maxwellian: on a finite velocity grid the sampled version does NOT conserve
    # rho, rho*u and rho*E, so the collision leaks them every step. "Exactly" means to the
    # Newton tolerance, far tighter than the 1e-6 quadrature error above.
    @testset "discrete Maxwellian is exactly conservative" begin
        rho, ux, uy, uz, T = 1.3, 0.2, -0.15, 0.05, 0.9
        f = zeros(g.n, g.n, g.n)
        discrete_maxwellian!(f, rho, ux, uy, uz, T, g)
        M = moments35(f, g)
        E = 1.5*T + 0.5*(ux^2 + uy^2 + uz^2)
        @test M[1]  ≈ rho      atol=1e-12
        @test M[2]  ≈ rho*ux   atol=1e-12
        @test M[6]  ≈ rho*uy   atol=1e-12
        @test M[16] ≈ rho*uz   atol=1e-12
        @test (M[3] + M[10] + M[20]) ≈ rho*2E atol=1e-11
    end

    # ---- the collision conserves its invariants and relaxes at the right rate ----------
    @testset "collision" begin
        # A non-equilibrium state: a Maxwellian sheared in velocity space by hand.
        f0 = zeros(g.n, g.n, g.n)
        discrete_maxwellian!(f0, 1.0, 0.0, 0.0, 0.0, 1.0, g)
        for k in 1:g.n, j in 1:g.n, i in 1:g.n
            f0[i,j,k] *= 1 + 0.2*g.v[i]*g.v[j]/(1 + g.v[i]^2 + g.v[j]^2)
        end
        M0 = moments35(f0, g)

        f = copy(f0); tau = 0.7; dt = 0.1
        collide_cell!(f, dt, tau, g)
        M1 = moments35(f, g)

        # invariants: untouched
        @test M1[1]  ≈ M0[1]  atol=1e-12
        @test M1[2]  ≈ M0[2]  atol=1e-12
        @test M1[6]  ≈ M0[6]  atol=1e-12
        @test M1[16] ≈ M0[16] atol=1e-12
        @test (M1[3]+M1[10]+M1[20]) ≈ (M0[3]+M0[10]+M0[20]) atol=1e-11

        # a NON-invariant must decay by exactly exp(-dt/tau). The exact-exponential
        # collision f = feq + (f-feq)e is linear, so this holds to roundoff, not just
        # to leading order -- which makes it a real assertion rather than a trend check.
        # m110 = <vx vy> is the shear stress the perturbation above put in.
        feq = zeros(g.n, g.n, g.n)
        r, ux, uy, uz, T, _ = rho_u_T(f0, g)
        discrete_maxwellian!(feq, r, ux, uy, uz, T, g)
        Me = moments35(feq, g)
        pred = Me[7] + (M0[7] - Me[7])*exp(-dt/tau)
        @test M1[7] ≈ pred atol=1e-12
    end

    # ---- TRANSPORT: the bc keyword must do what it says --------------------------------
    # THE REGRESSION TEST. `bc=:periodic` used to silently give zero-gradient.
    @testset "transport honours bc" begin
        Nx = 8
        mk() = begin
            f = zeros(Nx, g.n, g.n, g.n)
            for i in 1:Nx
                # a bump localised at one end, so the two boundary conditions cannot agree
                discrete_maxwellian!(@view(f[i,:,:,:]), i == 1 ? 2.0 : 1.0,
                                     0.0, 0.0, 0.0, 1.0, g)
            end
            f
        end
        dx = 1.0/Nx; dt = 0.2*dx/5.0

        # PERIODIC: total mass is conserved to roundoff, because nothing leaves the domain.
        fp = mk(); m0 = sum(moments35(@view(fp[i,:,:,:]), g)[1] for i in 1:Nx)
        for _ in 1:40; transport!(fp, dt, dx, g; bc=:periodic); end
        mp = sum(moments35(@view(fp[i,:,:,:]), g)[1] for i in 1:Nx)
        @test abs(mp - m0)/m0 < 1e-13

        # ZERO-GRADIENT: mass is NOT conserved -- the upwind stencil at i=1 duplicates cell 1
        # for v>0 and at i=Nx duplicates cell Nx for v<0, which injects and removes mass.
        # Asserting the two DIFFER is what makes the periodic test above non-vacuous: before
        # the fix both branches ran the same code and this assertion would have failed.
        fc = mk()
        for _ in 1:40; transport!(fc, dt, dx, g; bc=:copy); end
        mc = sum(moments35(@view(fc[i,:,:,:]), g)[1] for i in 1:Nx)
        @test !isapprox(mc, mp; rtol=1e-9)

        # and the fields themselves must differ, not merely their totals
        @test maximum(abs.(fc .- fp)) > 1e-6
    end

    # ---- free streaming translates moments, with no collision -------------------------
    # A pure advection check: with tau -> Inf the density profile moves at the mean velocity.
    # Periodic so the check is not confounded by outflow at the boundary.
    @testset "free streaming is conservative" begin
        Nx = 16; dx = 1.0/Nx; dt = 0.2*dx/5.0
        f = zeros(Nx, g.n, g.n, g.n)
        for i in 1:Nx
            discrete_maxwellian!(@view(f[i,:,:,:]),
                                 1.0 + 0.2*sin(2pi*(i-0.5)/Nx), 0.0, 0.0, 0.0, 1.0, g)
        end
        m0 = sum(moments35(@view(f[i,:,:,:]), g)[1] for i in 1:Nx)
        p0 = sum(moments35(@view(f[i,:,:,:]), g)[2] for i in 1:Nx)
        for _ in 1:60; transport!(f, dt, dx, g; bc=:periodic); end
        m1 = sum(moments35(@view(f[i,:,:,:]), g)[1] for i in 1:Nx)
        p1 = sum(moments35(@view(f[i,:,:,:]), g)[2] for i in 1:Nx)
        @test abs(m1 - m0)/m0 < 1e-13
        @test abs(p1 - p0) < 1e-13*max(abs(p0), 1.0)   # x-momentum: no force, so unchanged
    end
end

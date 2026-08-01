# test_kfvs_wall.jl — the half-space wall flux must conserve mass EXACTLY and reproduce the
# equilibrium wall density the ghost cell had to abandon.
#
# Issue #36: the Maxwell-accommodating ghost wall passes a net mass flux through an
# impermeable boundary (-3.9e-4 relative over a Couette run, against the DVM's 1e-15). No
# choice of ghost density can cancel it -- at u_n = 0 the HLL formula leaves pure upwind
# dissipation on the density jump, so a physically correct rho_w = rho*sqrt(T/Tw) MANUFACTURES
# a mass source (measured: interior mass 30 -> 1.9e11, growth rate timestep-independent, i.e.
# an eigenvalue). `wall_ghost_dev.jl` therefore settled for rho_w = rho, trading real physics
# for stability. In Couette the gas is viscously heated so T > Tw permanently -- the discarded
# adjustment is always active.
#
# The half-space flux sets rho_w FROM the zero-net-mass-flux condition instead, so conservation
# is a construction rather than a hope, and the equilibrium limit comes back for free.
using Test, Riemann35
using Riemann35.KfvsWall: halfline_gauss_moments

@testset "KFVS half-space wall flux" begin
    # ---- mass conservation is exact, by construction -----------------------------------
    for (T, Tw, un) in ((1.0,1.0,0.0), (1.3,1.0,0.0), (1.0,1.3,0.0),
                        (1.5,1.0,0.05), (0.8,1.2,0.0), (1.0,1.0,-0.10))
        M = collect(Float64, InitializeM4_35(1.0, 0.0, un, 0.0, T,0,0, T,0, T))
        _, _, mdot = kfvs_wall_flux(M, 2, +1.0, Tw, 0.0, 0.0)
        @test abs(mdot) < 1e-12
    end

    # ---- the equilibrium wall density is recovered EXACTLY -------------------------------
    # rho_w = rho sqrt(T/Tw) at rest: correct physics, and precisely what the ghost cell
    # could not impose without diverging.
    for (T, Tw) in ((1.0,1.0), (1.3,1.0), (1.0,1.3), (0.8,1.2), (2.0,0.5))
        M = collect(Float64, InitializeM4_35(1.0, 0.0, 0.0, 0.0, T,0,0, T,0, T))
        _, rho_w, _ = kfvs_wall_flux(M, 2, +1.0, Tw, 0.0, 0.0)
        @test rho_w ≈ sqrt(T/Tw) rtol=1e-10
    end

    # ---- with a normal velocity, rho_w follows the half-space outflux --------------------
    for (T, Tw, un) in ((1.5,1.0,0.05), (1.0,1.0,0.10), (1.0,1.0,-0.10))
        M = collect(Float64, InitializeM4_35(1.0, 0.0, un, 0.0, T,0,0, T,0, T))
        _, rho_w, _ = kfvs_wall_flux(M, 2, +1.0, Tw, 0.0, 0.0)
        Io = halfline_gauss_moments(1, un, T,  +1)      # interior half, leaving
        Ii = halfline_gauss_moments(1, 0.0, Tw, -1)     # wall half, entering
        @test rho_w ≈ -Io[2]/Ii[2] rtol=1e-10
    end

    # ---- a moving wall transmits TANGENTIAL momentum but no mass -------------------------
    # For a y-normal wall the cyclic tangents are t1 = z, t2 = x, so uw2 carries u_x. That
    # matches the convention measured independently in the march (uw1 -> u_z, uw2 -> u_x),
    # which is worth pinning: driving the wrong component silently produces a zero field.
    M = collect(Float64, InitializeM4_35(1.0,0.0,0.0,0.0,1.0,0,0,1.0,0,1.0))
    F0, _, _ = kfvs_wall_flux(M, 2, +1.0, 1.0, 0.0, 0.0)
    @test F0[2] == 0.0 && F0[16] == 0.0                 # wall at rest: no tangential flux
    Fz, _, mz = kfvs_wall_flux(M, 2, +1.0, 1.0, 0.1, 0.0)
    @test abs(mz) < 1e-12 && Fz[16] != 0.0 && Fz[2] == 0.0     # uw1 drives z
    Fx, rw, mx = kfvs_wall_flux(M, 2, +1.0, 1.0, 0.0, 0.1)
    @test abs(mx) < 1e-12 && Fx[2]  != 0.0 && Fx[16] == 0.0    # uw2 drives x
    @test Fx[2] ≈ rw*0.1*(-sqrt(1.0/(2pi))) rtol=1e-8          # exactly rho_w uw <v_n>_in
end

@testset "KFVS wall: CPU and device agree, all axes and both faces" begin
    # THE TESTS ABOVE MISSED TWO BUGS, both found only by cross-checking the two
    # implementations against each other. Recording the shape of each so they cannot return:
    #
    #   (a) OUT OF BOUNDS. The flux carries an extra v_n beyond the moment's own normal
    #       power, so psi = v^4 needs the p = 5 half-line moment. Requesting only p <= 4 read
    #       past the array; @inbounds suppressed the check and returned a denormal
    #       (1.9e-309) for moment (0,4,0). The original assertions never touched it.
    #
    #   (b) SIGN ON LO FACES. rho_w was formed from abs(outflux). On a lo face the leaving
    #       half has v_n < 0, so abs() flips the ratio and yields a NEGATIVE wall density and
    #       a large spurious mass flux. The original assertions only used outward = +1.
    #
    # Both were caught because two independent implementations disagreed -- not by any bound,
    # and not by inspection. That is the argument for keeping both.
    for ax in 1:3, ow in (-1.0, 1.0), (T, Tw, un) in ((1.2,0.9,0.0), (0.8,1.3,0.05), (2.0,0.5,-0.03))
        M = collect(Float64, InitializeM4_35(1.0, 0.02, un, 0.01, T,0,0, T,0, T))
        Fc, rho_w, mdot = kfvs_wall_flux(M, ax, ow, Tw, 0.05, -0.05)
        Fd = kfvs_wall_flux_dev(ntuple(i -> M[i], Val(35)), ax, ow, Tw, 0.05, -0.05)
        # The device version now reconstructs the interior half from the FULL anisotropic
        # Gaussian (10 moments) while this CPU reference still uses the DIAGONAL form (7).
        # They agree only when the off-diagonal second moments vanish, which is the case for
        # the isotropic states built above -- and that agreement is still a real regression
        # test: the correlated form MUST reduce to the diagonal one when C_nt = 0.
        @test maximum(abs.(collect(Fc) .- collect(Fd))) < 1e-12*max(maximum(abs.(collect(Fc))), 1.0)
        @test abs(mdot) < 1e-12          # (b): lo faces must conserve too
        @test rho_w > 0.0                # (b): a wall density is never negative
        @test isfinite(Fc[15])           # (a): the (0,4,0) moment, which needs p = 5
        @test abs(Fc[15]) > 1e-300       #      and is not a denormal
    end
end

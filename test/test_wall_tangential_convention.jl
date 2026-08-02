# test_wall_tangential_convention.jl -- WHICH VELOCITY COMPONENT DOES uw1 MOVE?
#
# `kfvs_wall_flux(M, axis, outward, Tw, uw1, uw2)` names its two tangential wall speeds
# `uw1`/`uw2` rather than by component, and resolves them internally as
#
#     t1 = axis % 3 + 1;   t2 = t1 % 3 + 1
#
# so for the :channel wall normal (axis = 2) the mapping is uw1 -> z and uw2 -> x. That is
# NOT the ordering most readers assume from the names -- the natural guess for a y-normal
# wall is uw1 -> x -- and NOTHING pinned it, so a driver could set the wrong one and still
# produce a plausible, converged, entirely wrong answer.
#
# That is not hypothetical. A wall-free shear study in this project ran the DVM on the
# LONGITUDINAL mode while the closure ran the TRANSVERSE one, and the swap survived a long
# time because every downstream number stayed finite and smooth; it was retracted only after
# the decay rates refused to match an exact dispersion relation. A Couette driver that sets
# `wall_uw2` while ramping `u_x` in its initial condition is making exactly this assumption,
# and the wall shear stress it reports is meaningless if the assumption is wrong.
#
# These tests fix the mapping BEHAVIOURALLY -- by which momentum flux actually moves -- not
# by restating the `axis % 3 + 1` formula, which would pass just as happily if the formula
# and the physics disagreed.
using Test, Riemann35

@testset "wall tangential convention (uw1 -> t1, uw2 -> t2)" begin
    # gas at rest, wall sliding: the ONLY momentum the wall can inject is tangential, so the
    # component that picks it up identifies the mapping unambiguously.
    T0 = 1.0
    Mrest = collect(Float64, InitializeM4_35(1.0, 0.0, 0.0, 0.0, T0,0,0, T0,0, T0))
    mom = (2, 6, 16)                        # indices of (1,0,0), (0,1,0), (0,0,1)

    for axis in 1:3
        t1 = axis % 3 + 1
        t2 = t1 % 3 + 1

        # -- uw1 alone drives t1 and NOTHING else -------------------------------------------
        F1, _, _ = kfvs_wall_flux(Mrest, axis, +1.0, T0, 0.3, 0.0)
        @test abs(F1[mom[t1]]) > 1e-6                     # t1 receives the momentum
        @test abs(F1[mom[t2]]) < 1e-12                    # t2 receives none

        # -- uw2 alone drives t2 and NOTHING else -------------------------------------------
        F2, _, _ = kfvs_wall_flux(Mrest, axis, +1.0, T0, 0.0, 0.3)
        @test abs(F2[mom[t2]]) > 1e-6
        @test abs(F2[mom[t1]]) < 1e-12

        # -- the two are independent: driving both is the sum of driving each ---------------
        # (linearity in the wall speeds is not the claim -- this is a CROSS-TALK check, that
        # uw1 does not leak into t2 once uw2 is also nonzero.)
        Fb, _, _ = kfvs_wall_flux(Mrest, axis, +1.0, T0, 0.3, 0.0)
        Fc, _, _ = kfvs_wall_flux(Mrest, axis, +1.0, T0, 0.3, 0.5)
        @test abs(Fc[mom[t1]] - Fb[mom[t1]]) < 1e-12      # uw2 did not disturb t1

        # -- a sliding wall injects NO normal momentum beyond the static case ---------------
        F0, _, _ = kfvs_wall_flux(Mrest, axis, +1.0, T0, 0.0, 0.0)
        @test abs(F1[mom[axis]] - F0[mom[axis]]) < 1e-12

        # -- and it still passes zero net mass, sliding or not ------------------------------
        for (a, b) in ((0.0,0.0), (0.3,0.0), (0.0,0.3), (0.3,0.5))
            _, _, mdot = kfvs_wall_flux(Mrest, axis, +1.0, T0, a, b)
            @test abs(mdot) < 1e-12
        end
    end

    # -- the concrete fact a Couette driver depends on ---------------------------------------
    # :channel walls are the y faces (axis = 2). Spelled out because this is the one line a
    # driver author actually needs, and deriving it from `axis % 3 + 1` in one's head is
    # exactly the step that goes wrong.
    @test (2 % 3 + 1) == 3                  # uw1 -> z
    @test ((2 % 3 + 1) % 3 + 1) == 1        # uw2 -> x   <- Couette drives THIS one
    Fy, _, _ = kfvs_wall_flux(Mrest, 2, +1.0, T0, 0.0, 0.3)
    @test abs(Fy[2]) > 1e-6                 # x-momentum flux is nonzero ...
    @test abs(Fy[16]) < 1e-12               # ... and z-momentum flux is not

    # The wall shear stress such a driver reads is P_xy = M[7] - rho ux uy, i.e. the
    # (normal, tangential) = (y, x) component. Confirm index 7 is (1,1,0) so that the
    # stress expression cannot silently drift if the moment ordering is ever renumbered.
    @test Riemann35.IJK[7] == (1, 1, 0)
    @test Riemann35.momidx(1, 1, 0) == 7
    @test Riemann35.momidx(1, 0, 0) == 2
    @test Riemann35.momidx(0, 1, 0) == 6
end

@testset "host and device wall flux agree" begin
    # `kfvs_wall_flux` (host) and `kfvs_wall_flux_dev` (device) are SEPARATE implementations
    # of the same half-space integral -- the device one is written for GPU register pressure
    # and does not share a line of code with the host one. Every existing wall test exercises
    # the HOST path, while every production Couette/Poiseuille run goes through the DEVICE
    # path, so agreement between them has been assumed rather than checked. If they drifted,
    # the tests above would keep passing while the numbers we actually publish came from an
    # untested function.
    #
    # Both resolve tangents with the same `t1 = axis % 3 + 1` rule (verified by reading, and
    # now by this test), but the host integrates the interior half from `halfline_gauss_moments`
    # and the device from an inlined series, so exact equality is not expected -- agreement to
    # tight tolerance is.
    for axis in 1:3, (T, Tw, uw1, uw2) in ((1.0, 1.0, 0.3, 0.0), (1.0, 1.0, 0.0, 0.3),
                                           (1.3, 1.0, 0.2, 0.4), (0.8, 1.2, -0.3, 0.1))
        M = collect(Float64, InitializeM4_35(1.0, 0.05, -0.02, 0.03, T,0,0, T,0, T))
        Fh, _, _ = kfvs_wall_flux(M, axis, +1.0, Tw, uw1, uw2)
        Fd = Riemann35.kfvs_wall_flux_dev(NTuple{35,Float64}(M), axis, +1.0, Tw, uw1, uw2)
        scale = maximum(abs, Fh) + 1e-30
        @test maximum(abs, Fh .- Fd) / scale < 1e-10
    end
end

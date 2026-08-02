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
using Test, Riemann35, Random

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

@testset "wall flux agreement must be probed WITH cross-moments" begin
    # THE GUARD THAT FAILED. Two separate agreement testsets -- this file's original one and
    # the 90-assertion set in test_kfvs_wall.jl -- both passed while host and device
    # implemented DIFFERENT PHYSICAL MODELS: the host factorised the interior half with a
    # diagonal covariance and dropped cxy/cxz/cyz entirely. Both testsets missed it because
    # both built states with InitializeM4_35(..., T,0,0, T,0, T) -- every off-diagonal
    # exactly zero, which is precisely the subspace where the two models coincide.
    #
    # Measured: 5e-16 agreement with diagonal covariance, up to 44% with off-diagonals. A
    # test that only probes the degenerate subspace is not a weak test, it is a test of a
    # different function. So: states here carry NONZERO cxy/cxz/cyz, and the diagonal model
    # is kept available under a private name so the difference is demonstrated, not asserted.
    using Riemann35.KfvsWall: _kfvs_wall_flux_diagonal
    Random.seed!(20260802)
    worst_diag = 0.0
    for _ in 1:400
        r = 0.5 + rand(); T = 0.6 + 0.8rand()
        M = collect(Float64, InitializeM4_35(r, 0.3randn(), 0.3randn(), 0.3randn(),
                                             T, 0.05randn(), 0.05randn(), T, 0.05randn(), T))
        axis = rand(1:3); ow = rand(Bool) ? 1.0 : -1.0
        Tw = 0.6 + 0.8rand(); uw1 = 0.3randn(); uw2 = 0.3randn()

        Fh, _, _ = kfvs_wall_flux(M, axis, ow, Tw, uw1, uw2)
        Fd = Riemann35.kfvs_wall_flux_dev(NTuple{35,Float64}(M), axis, ow, Tw, uw1, uw2)
        scale = maximum(abs, Fh) + 1e-30
        # host now DELEGATES, so this is exact -- not "close"
        @test maximum(abs, collect(Fh) .- collect(Fd)) == 0.0

        Fdiag, _, _ = _kfvs_wall_flux_diagonal(M, axis, ow, Tw, uw1, uw2)
        worst_diag = max(worst_diag, maximum(abs, collect(Fh) .- collect(Fdiag))/scale)
    end
    # The retired diagonal model must remain VISIBLY different on these states; if this ever
    # drops to roundoff the states have lost their off-diagonals and the test is asleep again.
    @test worst_diag > 1e-3
    @info "diagonal-covariance wall model differs from the correlated one by" worst_diag
end

@testset "host and device wall flux agree" begin
    # Retained after the host was made a delegating wrapper. It no longer proves two
    # implementations agree -- there is only one now -- but it pins the WRAPPER: that
    # kfvs_wall_flux still forwards axis/outward/Tw/uw1/uw2 in the right order and still
    # returns (F, rho_w, mdot) with F matching the device tuple. A wrapper that silently
    # transposed two arguments would produce finite, plausible output and break nothing else.
    for axis in 1:3, (T, Tw, uw1, uw2) in ((1.0, 1.0, 0.3, 0.0), (1.0, 1.0, 0.0, 0.3),
                                           (1.3, 1.0, 0.2, 0.4), (0.8, 1.2, -0.3, 0.1))
        M = collect(Float64, InitializeM4_35(1.0, 0.05, -0.02, 0.03, T,0,0, T,0, T))
        Fh, _, _ = kfvs_wall_flux(M, axis, +1.0, Tw, uw1, uw2)
        Fd = Riemann35.kfvs_wall_flux_dev(NTuple{35,Float64}(M), axis, +1.0, Tw, uw1, uw2)
        scale = maximum(abs, Fh) + 1e-30
        @test maximum(abs, Fh .- Fd) / scale < 1e-10
    end
end

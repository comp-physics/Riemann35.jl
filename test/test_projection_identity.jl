# test_projection_identity.jl — THE REALIZABILITY PROJECTION MUST NOT MOVE A STATE THAT IS
# ALREADY REALIZABLE.
#
# This is the test that was missing, and its absence cost a day. `realizability_S2` applied
# its `xr = 0.9999*xr` back-off OUTSIDE the `if S2 < 0` guard, so it shrank the velocity
# correlations by 1 part in 1e4 on every input -- including inputs comfortably inside the
# cone, where the correct answer is the input unchanged. Applied once per RK stage to every
# cell that is an artificial dissipation on SHEAR STRESS whose total scales with the NUMBER
# of stages, i.e. as 1/dt, and it meant no driven steady state had a dt -> 0 limit.
#
# The suite had 7,600 tests and did not catch it. Twelve of them were actively DEFENDING it:
# the MATLAB-derived goldens recorded the shrunk values as correct, so the suite's response
# to the fix was to fail. A behaviour-pinning test can tell you something changed; it cannot
# tell you the original was wrong.
#
# So this asserts a PROPERTY instead: idempotence on the admissible set. It is one line of
# mathematics -- a projection onto a set must fix every point already in that set -- and it
# would have caught the defect in milliseconds.
#
#     P(x) = x   for all x in the realizable set
#
# Checked at three levels: the S2 correlation kernel where the bug lived, the full 35-moment
# projection built on it, and the device port that the GPU march actually calls.
using Test, Riemann35, LinearAlgebra, Random

@testset "projection is the identity on realizable states" begin
    # ---- level 1: the S2 correlation kernel, where the defect was ----------------------
    # A triple (S110, S101, S011) is realizable when
    #     S2 = 1 + 2 S110 S101 S011 - (S110^2 + S101^2 + S011^2) >= 0.
    # For every such triple the projection must return it untouched.
    s2_of(a,b,c) = 1 + 2a*b*c - (a^2 + b^2 + c^2)

    # hand-picked interior points, including the three the MATLAB goldens use
    for (a,b,c) in ((0.99,0.5,0.6), (-0.5,0.4,-0.3), (0.5,0.3,0.4),
                    (0.0,0.0,0.0), (0.2,-0.1,0.05), (-0.7,0.1,0.2))
        S2 = s2_of(a,b,c)
        @test S2 >= 0                                     # the case is what we think it is
        ar, br, cr, S2r = realizability(:S2, a, b, c)
        @test ar ≈ a atol=1e-14
        @test br ≈ b atol=1e-14
        @test cr ≈ c atol=1e-14
        @test S2r ≈ S2 atol=1e-14
    end

    # randomized: rejection-sample the admissible set so the property is exercised over the
    # interior generally, not just at points someone thought to write down.
    rng = MersenneTwister(20260728)
    n = 0
    while n < 300
        a, b, c = 2rand(rng)-1, 2rand(rng)-1, 2rand(rng)-1
        s2_of(a,b,c) < 0 && continue                       # out of cone: not this test's job
        n += 1
        ar, br, cr, _ = realizability(:S2, a, b, c)
        @test max(abs(ar-a), abs(br-b), abs(cr-c)) < 1e-14
    end

    # ---- level 2: OUT-OF-CONE inputs must still be corrected ---------------------------
    # The property above must not be bought by disabling the projection. A state outside the
    # cone has to come back inside it.
    # S2 computed, not assumed: (0.95,0.9,0.85) looks extreme but is S2 = +0.0185, i.e.
    # INSIDE the cone, and was rejected by this very assertion when first written.
    for (a,b,c) in ((0.99,0.99,-0.99), (0.9,0.9,0.0), (-0.98,0.97,0.96), (0.99,0.0,0.99))
        @test s2_of(a,b,c) < 0                             # genuinely out of cone
        ar, br, cr, S2r = realizability(:S2, a, b, c)
        @test S2r >= 0                                     # brought back in
        @test max(abs(ar), abs(br), abs(cr)) < max(abs(a), abs(b), abs(c)) + 1e-12
    end

    # ---- level 3: the full 35-moment projection, and the device port -------------------
    # realizable_3D_M4 is what the CPU march calls; realizable_3D_M4_dev is what the GPU
    # kernels call. A Maxwellian and mild perturbations of one are deep inside the cone, so
    # both must return them unchanged to roundoff. The device path is the one that ran
    # 3 times per step on every cell.
    for scale in (1.0, 0.5, 2.0), T in (1.0, 0.7, 1.4)
        M = collect(Float64, InitializeM4_35(scale, 0.05, -0.02, 0.01, T,0,0, T,0, T))
        P = realizable_3D_M4(M, 1.0, 40.0)
        @test norm(P .- M)/norm(M) < 1e-13

        D = collect(Float64, Riemann35.RealizeDev.realizable_3D_M4_dev(M..., 1.0, 40.0))
        @test norm(D .- M)/norm(M) < 1e-13
    end
end

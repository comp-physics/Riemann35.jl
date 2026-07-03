# test_moment_correction.jl — the unified hyperbolicity moment correction
# (src/numerics/moment_correction_dev.jl; CPU correct_moments_hyperbolic_3D
# delegates to it since 2026-07-03; ~1 ulp vs the old autogen formulation,
# verified pure reassociation over 2000 states, within GOLDEN_TOL = 1e-10).
using Test
using Riemann35
using Riemann35: correct_moments_hyperbolic_3D, InitializeM4_35, M2CS4_35
using Riemann35.MomentCorrectionDev: correct_moments_dev

@testset "moment correction (unified CPU/GPU)" begin
    M = InitializeM4_35(2.0, 0.4, -0.3, 0.2, 1.2, 0.3, -0.2, 0.9, 0.25, 1.1)
    M[4] *= 1.4; M[27] += 0.15                     # activate the correction

    @testset "CPU wrapper delegates bitwise to the shared device function" begin
        @test correct_moments_hyperbolic_3D(M) == collect(correct_moments_dev(M...))
    end

    @testset "correction properties" begin
        Mc = correct_moments_hyperbolic_3D(M)
        _, S = M2CS4_35(Mc)
        # cross third-order standardized moments zeroed
        for idx in (8, 11, 18, 21, 29, 32)          # S210,S120,S201,S102,S021,S012
            @test abs(S[idx]) < 1e-13
        end
        # S220/S202/S022 floored at 1/3
        @test S[12] >= 1/3 - 1e-13
        @test S[22] >= 1/3 - 1e-13
        @test S[35] >= 1/3 - 1e-13
        # conserved quantities untouched (rho, means, variances)
        for idx in (1, 2, 6, 16, 3, 10, 20)
            @test Mc[idx] ≈ M[idx] rtol = 1e-14
        end
        # idempotence (a corrected state is a fixed point to roundoff)
        Mcc = correct_moments_hyperbolic_3D(Mc)
        @test maximum(abs.(Mcc .- Mc) ./ max.(abs.(Mc), 1e-300)) < 1e-12
    end
end

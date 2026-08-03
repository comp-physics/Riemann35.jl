# test_projection_diagnostics.jl -- the realizability projection's firing diagnostics.
#
# These exist to answer two standing requests (R.O. Fox 2026-08-02; JCP Reviewer #3): when does
# the projection fire, which principal minor is negative, and how far does it move the moments.
# The tests below pin the properties those answers depend on, so a refactor cannot silently
# change what the statistics mean.
using Test, Riemann35, LinearAlgebra

@testset "projection diagnostics" begin

    # A Gaussian is realizable by construction, correlated or not. If the projection fires here
    # the diagnostic is measuring the projection's own noise floor, not physics.
    @testset "realizable states do not fire" begin
        for (c110, c101, c011) in ((0.0,0.0,0.0), (0.2,0.1,-0.15), (0.4,-0.3,0.25))
            M = InitializeM4_35(1.0, 0.1,-0.05,0.02, 1.0,c110,c101, 1.0,c011, 1.0)
            r = projection_report(M)
            @test !r.fired
            @test r.first_bad == 0
            @test r.dS < 1e-12
            @test all(>(0.0), r.minors)      # Sylvester: PD <=> every leading minor > 0
        end
    end

    # Sylvester's criterion is the definition of the thing being reported, so tie the two
    # together directly: `first_bad == 0` must mean exactly "positive definite".
    @testset "first_bad agrees with Sylvester" begin
        M0 = collect(Float64, InitializeM4_35(1.0, 0.0,0.0,0.0, 1.0,0.15,0.1, 1.0,-0.1, 1.0))
        for f in (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0)
            M = copy(M0); M[12] *= f                 # M220
            r = projection_report(M)
            pd = all(>(0.0), r.minors)
            @test (r.first_bad == 0) == pd
            if !pd
                # the reported index must BE the first violation, not merely some violation
                @test r.minors[r.first_bad] <= 0.0
                @test all(>(0.0), r.minors[1:r.first_bad-1])
            end
        end
    end

    # The measurement that carries Fox's question. Perturbing ONE moment must not require
    # touching most of the vector; if it does, that is the aggressiveness he is asking about and
    # the number belongs in a test so it cannot drift unnoticed.
    @testset "collateral change is real and violation-independent" begin
        M0 = collect(Float64, InitializeM4_35(1.0, 0.1,-0.05,0.02, 1.0,0.2,0.1, 1.0,-0.15, 1.0))
        rs = map((3.0, 8.0)) do f
            M = copy(M0); M[12] *= f
            projection_report(M)
        end
        for r in rs
            @test r.fired
            @test r.n_changed > 10          # measured 15 of 28 from a single-moment violation
            @test r.dS_other > 1e-3         # ...and the collateral part is not roundoff
        end
        # The collateral change does not scale with how badly the state violates realizability:
        # measured 0.6295 at both 3x and 8x. That invariance is the specific evidence that it is
        # structural rather than proportional response, so pin it.
        @test isapprox(rs[1].dS_other, rs[2].dS_other; rtol = 1e-6)
        # ...while the part carried by the violated cross moment DOES scale.
        @test rs[2].dS_cross > 2 * rs[1].dS_cross
    end

    # The matrix and the minors must be self-consistent, independent of the projection.
    @testset "delta2star_matrix and leading_minors" begin
        M = InitializeM4_35(1.0, 0.0,0.0,0.0, 1.0,0.2,0.1, 1.0,-0.15, 1.0)
        _, S = M2CS4_35(collect(Float64, M))
        S28 = ntuple(i -> S[Riemann35.ProjectionDiagnostics.PROJ_S_IDX[i]], 28)
        A = delta2star_matrix(S28)
        @test size(A) == (6, 6)
        @test A ≈ A'                                  # symmetric by construction
        D = leading_minors(A)
        @test length(D) == 6
        @test D[1] ≈ A[1,1]
        @test D[2] ≈ A[1,1]*A[2,2] - A[1,2]^2
        @test D[6] ≈ det(A)
        # a realizable state is PD, so min eigenvalue > 0 and every minor > 0 must agree
        @test (minimum(eigvals(Symmetric(A))) > 0) == all(>(0.0), D)
    end
end

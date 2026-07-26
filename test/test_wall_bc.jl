# ---------------------------------------------------------------------------
# Maxwell-accommodating wall ghost state (src/numerics/wall_ghost_dev.jl).
#
# The two EXACTNESS GATES come first, because they can falsify the design:
#   W1  specular (alpha=0) is exact -- conserves mass and energy to machine precision,
#       reverses normal momentum exactly, leaves tangential momentum untouched, and must
#       ignore T_w entirely.
#   W2  equilibrium wall -- a Maxwellian at (rho, 0, T) against a wall with T_w = T,
#       u_w = 0 must be an EXACT fixed point: the ghost equals the interior state. This is
#       the wall analogue of the stationary-contact test the repo uses as its sharpest
#       gate, and it is the one that catches a wrong rho_w normalization.
#
# W3 checks the half-space influx closed form against numerical quadrature -- an
# independent computation, the same tactic used to validate the Isserlis table.
# ---------------------------------------------------------------------------
using Test, LinearAlgebra
using Riemann35
using Riemann35.WallGhostDev: wall_ghost_tup, halfspace_influx, erfc_dev

maxw(rho,u,v,w,T) = NTuple{35,Float64}(InitializeM4_35(rho,u,v,w,T,0,0,T,0,T))
mass(M)=M[1]; momx(M)=M[2]; momy(M)=M[6]; momz(M)=M[16]
energy(M)=M[3]+M[10]+M[20]

@testset "wall BC (Maxwell accommodation)" begin

    @testset "W1 specular (alpha=0) is EXACT" begin
        for axis in 1:3
            M = maxw(1.3, 0.25, -0.15, 0.08, 1.1)
            for outward in (-1.0, 1.0), Tw in (1.0, 5.0, 0.2)   # Tw must be ignored
                G = wall_ghost_tup(M, axis, outward, Tw, 0.3, -0.2, 0.0)
                @test mass(G)   ≈ mass(M)   rtol=1e-15
                @test energy(G) ≈ energy(M) rtol=1e-15
                # normal momentum reversed, tangential untouched
                mom = (momx(G), momy(G), momz(G)); mom0 = (momx(M), momy(M), momz(M))
                for d in 1:3
                    if d == axis
                        @test mom[d] ≈ -mom0[d] rtol=1e-15
                    else
                        @test mom[d] ≈  mom0[d] rtol=1e-15
                    end
                end
            end
        end
    end

    @testset "W2 equilibrium wall is an EXACT fixed point" begin
        # Maxwellian at rest against a wall at the same temperature: nothing should happen.
        worst = 0.0
        for axis in 1:3, outward in (-1.0, 1.0), alpha in (0.25, 0.5, 1.0)
            rho, T = 1.0, 1.0
            M = maxw(rho, 0.0, 0.0, 0.0, T)
            G = wall_ghost_tup(M, axis, outward, T, 0.0, 0.0, alpha)
            worst = max(worst, maximum(abs, collect(G) .- collect(M)) / maximum(abs, collect(M)))
        end
        @info "W2 equilibrium-wall fixed point: worst rel deviation" worst
        @test worst < 1e-12
    end

    @testset "W3 half-space influx vs numerical quadrature" begin
        # integral_{v<0} |v| N(un,sig^2) dv, by direct midpoint quadrature
        function influx_quad(rho, un, sig; n=400_000, span=12.0)
            lo = min(-span*sig + un, -span*sig); hi = 0.0
            h = (hi-lo)/n; s = 0.0
            for i in 1:n
                v = lo + (i-0.5)*h
                s += abs(v) * exp(-0.5*((v-un)/sig)^2) * h
            end
            rho * s / (sig*sqrt(2pi))
        end
        worst = 0.0
        for un in (-1.5,-0.5,0.0,0.5,1.5), sig in (0.5,1.0,2.0)
            a = halfspace_influx(1.0, un, sig); b = influx_quad(1.0, un, sig)
            worst = max(worst, abs(a-b)/max(b,1e-12))
        end
        @info "W3 influx closed form vs quadrature: worst rel error" worst
        @test worst < 2e-3          # erfc approximation is ~1e-7 absolute; quadrature is the floor
    end

    @testset "W4 realizability and finiteness" begin
        for axis in 1:3, alpha in (0.0, 0.5, 1.0), outward in (-1.0, 1.0)
            for M in (maxw(1.0,0,0,0,1.0), maxw(0.02, 2.5,-1.0,0.4, 0.05),
                      maxw(5.0, -0.3,0.1,0.0, 3.0))
                G = wall_ghost_tup(M, axis, outward, 1.7, 0.4, -0.25, alpha)
                @test all(isfinite, G)
                @test G[1] > 0                       # positive density
                @test G[3] > 0 && G[10] > 0 && G[20] > 0
            end
        end
        # rho <= 0 short-circuits
        Mz = NTuple{35,Float64}(vcat(0.0, collect(maxw(1.0,0,0,0,1.0))[2:35]))
        @test wall_ghost_tup(Mz, 1, 1.0, 1.0, 0.0, 0.0, 1.0) === Mz
    end

    @testset "W5 erfc_dev accuracy" begin
        # against the relation erfc(x) + erfc(-x) == 2 and known values
        @test erfc_dev(0.0) ≈ 1.0 atol=1e-6
        for x in (-3.0,-1.0,-0.3,0.0,0.3,1.0,3.0)
            @test erfc_dev(x) + erfc_dev(-x) ≈ 2.0 atol=1e-6
        end
        @test erfc_dev(1.0) ≈ 0.15729920705028513 atol=1e-6
        @test erfc_dev(2.0) ≈ 0.004677734981047266 atol=1e-6
    end
end

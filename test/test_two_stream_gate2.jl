# test_two_stream_gate2.jl — Gate 2 (1D pipeline) for the two-stream half-space mode.
#
# Reproduces verify_halfline_scheme.jl using the PORTED 35-moment kernels run in 1D
# (ny=nz=1). Each cell carries two 35-moment streams (+ / −) split at the gauge
# velocity c=0. Donor-cell x-update (the + stream takes inflow from the left, the −
# stream from the right) with the half-space x-flux; exact-exponential BGK coupling.
#
# Targets (from the verified prototype / spec):
#   * collisionless contact drift maxu converges with resolution toward the exact
#     free-molecular value 1.03e-1: nx=100..800 gave 9.7e-3, 2.5e-2, 7.6e-2, 1.28e-1;
#   * crossing Ma=100 survives with ZERO chain clips away from vacuum fronts;
#   * Kn=1e-3 contact held to <= ~1e-2 via BGK.

using LinearAlgebra, Printf, Test

const SRC = get(ENV, "RIEMANN35_SRC", joinpath(@__DIR__, "..", "src"))
if !isdefined(Main, :Chain)
    include(joinpath(SRC, "moments", "chain.jl"));             using .Chain
    include(joinpath(SRC, "numerics", "flux_closure_dev.jl")); using .FluxClosureDev
    include(joinpath(SRC, "numerics", "halfline_closure.jl")); using .HalflineClosure
    include(joinpath(SRC, "numerics", "flux_halfspace35.jl")); using .FluxHalfspace35
    include(joinpath(SRC, "numerics", "bgk_stream.jl"));       using .BGKStream
end

const VAC = 1e-10

# H margin of an x-marginal (m0..m4): c4/c2^2 - (c3/c2^1.5)^2 - 1. Reflection-invariant
# (c3^2), so it applies to either stream's marginal directly. A Maxwellian gives H=2;
# a symmetric two-beam total gives H=0 (the realizability boundary).
function margin_H(m)
    m[1] <= VAC && return NaN
    u = m[2]/m[1]; c2 = m[3]/m[1]-u^2; c2 <= 0 && return NaN
    c3 = m[4]/m[1]-3u*m[3]/m[1]+2u^3
    c4 = m[5]/m[1]-4u*m[4]/m[1]+6u^2*m[3]/m[1]-3u^4
    return c4/c2^2 - (c3/c2^1.5)^2 - 1
end

# is a stream's x-marginal chain-realizable? (mirror the − stream to [0,∞) first)
function stream_ok(M, isplus)
    marg = isplus ? (M[1], M[2], M[3], M[4], M[5]) : (M[1], -M[2], M[3], -M[4], M[5])
    chain_realizable(chain(marg))
end

# one 1D two-stream run; returns (maxu, minH, blew, nclip)
function run1d(statefn; nx=200, tmax=3e-3, cfl=0.3, Kn=1e9, c=0.0)
    xs = range(-1, 1; length=nx); dx = 2/(nx-1)
    Mp = Vector{NTuple{35,Float64}}(undef, nx)
    Mm = Vector{NTuple{35,Float64}}(undef, nx)
    vacs = ntuple(q -> q==1 ? VAC : 0.0, 35)
    for (i, x) in enumerate(xs)
        ρ, u, T = statefn(x)
        a, b = split_maxwellian35(ρ, u, 0.0, 0.0, T, c)
        Mp[i] = a[1] > VAC ? a : vacs
        Mm[i] = b[1] > VAC ? b : vacs
    end
    reset_chain_clips!()
    t = 0.0; maxu = 0.0; minH = Inf; bulkH = Inf; blew = false; bulk_clips = 0
    Fp = Vector{NTuple{35,Float64}}(undef, nx); Fm = similar(Fp)
    while t < tmax
        # wave-speed CFL from both streams (marginal support node)
        sp = 1e-12
        for i in 1:nx
            sp = max(sp, stream_xspeed(Mp[i]), stream_xspeed(Mm[i]))
        end
        dt = cfl*dx/sp
        for i in 1:nx
            Fp[i] = xflux_plus35(Mp[i]...)
            Fm[i] = xflux_minus35(Mm[i])
        end
        Np = copy(Mp); Nm = copy(Mm)
        for i in 2:nx-1
            Np[i] = ntuple(q -> Mp[i][q] - (dt/dx)*(Fp[i][q]-Fp[i-1][q]), 35)   # + from left
            Nm[i] = ntuple(q -> Mm[i][q] - (dt/dx)*(Fm[i+1][q]-Fm[i][q]), 35)   # − from right
        end
        for i in 1:nx; Mp[i] = Np[i]; Mm[i] = Nm[i]; end
        # BGK relaxation
        if Kn < 1e5
            for i in 2:nx-1
                Mp[i], Mm[i] = bgk_stream_relax(Mp[i], Mm[i], dt, Kn, c)
            end
        end
        # diagnostics — per-stream (away from vacuum fronts, mass-gated at 0.1)
        for i in 2:nx-1
            (all(isfinite, Mp[i]) && all(isfinite, Mm[i])) || (blew = true; break)
            tot = ntuple(q -> Mp[i][q]+Mm[i][q], 35)
            tot[1] > 0.1 && (maxu = max(maxu, abs(tot[2]/tot[1])))
            if Mp[i][1] > 0.1
                h = margin_H(Mp[i]); isnan(h) || (minH = min(minH, h))
                stream_ok(Mp[i], true) || (bulk_clips += 1)
            end
            if Mm[i][1] > 0.1
                h = margin_H(Mm[i]); isnan(h) || (minH = min(minH, h))
                stream_ok(Mm[i], false) || (bulk_clips += 1)
            end
        end
        blew && break
        t += dt
    end
    # settled-stream H on the FINAL state, measured in the UNDISTURBED cores (the +
    # stream deep-left, the − stream deep-right — far from the advancing fronts that
    # first-order upwind smears). These are the established Maxwellian streams; the
    # smeared fronts are numerical diffusion, not a realizability failure.
    for i in 2:nx-1
        x = xs[i]
        if -0.9 <= x <= -0.5 && Mp[i][1] > 0.5
            h = margin_H(Mp[i]); isnan(h) || (bulkH = min(bulkH, h))
        end
        if 0.5 <= x <= 0.9 && Mm[i][1] > 0.5
            h = margin_H(Mm[i]); isnan(h) || (bulkH = min(bulkH, h))
        end
    end
    return maxu, minH, bulkH, blew, bulk_clips, chain_clips()
end

contact    = x -> x < 0 ? (1000.0, 0.0, 1e-3) : (1.0, 0.0, 1.0)
crossing   = x -> x < 0 ? (1.0, 100.0, 1.0) : (1.0, -100.0, 1.0)

@testset "Gate 2: 1D two-stream pipeline" begin
    println("--- collisionless contact drift convergence toward free-molecular 1.03e-1 ---")
    us = Float64[]
    for nx in (100, 200, 400, 800)
        u, _, _, blew, bc, nc = run1d(contact; nx=nx)
        push!(us, u)
        @printf("  nx=%4d : maxu=%.3e  bulk_clips=%d  (flux clips=%d)  %s\n",
                nx, u, bc, nc, blew ? "BLEW" : "ok")
        @test !blew
        @test bc == 0
    end
    # monotone increase toward ~1.03e-1 (kinetic free-molecular value)
    @test issorted(us)
    @test us[end] > 5e-2 && us[end] < 1.6e-1

    println("--- crossing Ma=100: per-stream survival, H≈2, zero bulk clips ---")
    u, minH, bulkH, blew, bc, nc = run1d(crossing; nx=200)
    @printf("  Ma=100 : maxu=%.3e  settled-stream H=%+.3f  (front-transient minH=%+.3e)  bulk_clips=%d  (vacuum-front flux clips=%d)  %s\n",
            u, bulkH, minH, bc, nc, blew ? "BLEW" : "ok")
    @test !blew
    @test bc == 0                       # ZERO chain clips away from vacuum fronts
    @test minH > 0.0                    # every mass-bearing stream cell stays realizable
    @test bulkH > 1.5                   # settled streams ≈ Maxwellian, H ≈ 2
    @test u > 90                        # streams retain the Ma=100 velocity

    println("--- Kn=1e-3 contact held via BGK ---")
    u, _, _, blew, bc, nc = run1d(contact; nx=200, Kn=1e-3)
    @printf("  Kn=1e-3 contact : maxu=%.3e  bulk_clips=%d  %s\n", u, bc, blew ? "BLEW" : "ok")
    @test !blew
    @test u <= 2e-2                     # drift suppressed to <= ~1e-2
end

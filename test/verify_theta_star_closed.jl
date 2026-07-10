"""
verify_theta_star_closed.jl — validate the closed-form θ* limiter
(`theta_star_update_closed`) against the shipped bisection
(`theta_star_update_dev`) over a large synthetic set of realizable anchor
states Mlo and candidate updates dM.

Checks:
  (1) agreement |θ_closed − θ_bisect| — bisection resolves θ only to 2^-24 ≈
      6e-8, so agreement to ~1e-6 is expected (the closed form is the more
      accurate one). Reports max and 99.9th-percentile abs diff.
  (2) valid lower bound — Mlo + θ_closed·dM must be realizable (never overshoot).
  (3) coverage of the actual production update forms ±6λ·G.

Run:
  module load julia/1.11.3; export JULIA_DEPOT_PATH=...; \
    julia --project=<worktree> test/verify_theta_star_closed.jl
"""

include(joinpath(@__DIR__, "..", "src", "numerics", "idp_limiter_dev.jl"))
using .IdpLimiterDev: theta_star_update_dev, theta_star_update_closed
using .IdpLimiterDev.RiemannFluxDev: _state_realizable
using .IdpLimiterDev.RiemannFluxDev.RoePS3Dev.MomentIndices: MARG_IDX
using Printf, Random, Statistics

# ---------------------------------------------------------------------------
# Synthesize a realizable 35-moment state from independent per-axis marginals
# with prescribed mean/variance/skewness/kurtosis (Hamburger-realizable when
# K > 1 + q̂²), and cross moments from a product (independent-axes) ansatz,
# which keeps the three marginal chains — the only thing _state_realizable
# reads — realizable. Cross slots use the independent-marginal product rule
# M_ijk = ρ · Mx_i * My_j * Mz_k (central-moment builder below).
# ---------------------------------------------------------------------------

# raw moments m0..m4 of a 1D distribution with mean u, variance v>0,
# standardized skew q̂, kurtosis K (K>1+q̂² for realizability).
function raw5(rho, u, v, qh, K)
    s = sqrt(v)
    c2 = v
    c3 = qh * s^3
    c4 = K * v^2
    m0 = rho
    m1 = rho * u
    m2 = rho * (c2 + u^2)
    m3 = rho * (c3 + 3u*c2 + u^3)
    m4 = rho * (c4 + 4u*c3 + 6u^2*c2 + u^4)
    return (m0, m1, m2, m3, m4)
end

# central-moment sequences c0..c4 per axis (c0=1,c1=0)
function central5(v, qh, K)
    s = sqrt(v)
    (1.0, 0.0, v, qh*s^3, K*v^2)
end

# Build a full 35-moment tuple from per-axis (u,v,qh,K) via the product ansatz:
# M_ijk = ρ * μx_i * μy_j * μz_k, where μa_n is the n-th RAW moment of axis a's
# marginal (mean ua, variance va, ...). Independent axes ⇒ each marginal chain is
# exactly (raw5 of that axis), hence realizable iff K>1+q̂².
const IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
             (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
             (0,3,0),(1,3,0),(0,4,0),
             (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
             (0,0,3),(1,0,3),(0,0,4),
             (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
             (0,1,2),(1,1,2),(0,1,3),(0,2,2))

# raw 1D moments up to order 4 (m0..m4) given mean/var/skew/kurt (as fractions of ρ=1)
function rawmoms1d(u, v, qh, K)
    m0, m1, m2, m3, m4 = raw5(1.0, u, v, qh, K)
    (m0, m1, m2, m3, m4)  # ρ=1 factored out
end

function build35(rho, ux, vx, qx, Kx, uy, vy, qy, Ky, uz, vz, qz, Kz)
    mx = rawmoms1d(ux, vx, qx, Kx)
    my = rawmoms1d(uy, vy, qy, Ky)
    mz = rawmoms1d(uz, vz, qz, Kz)
    ntuple(35) do q
        (i,j,k) = IJK[q]
        rho * mx[i+1] * my[j+1] * mz[k+1]
    end
end

rng = MersenneTwister(20260709)

# random realizable state generator
function rand_state(rng)
    rho = exp(2.0*(rand(rng)-0.5))          # ρ in ~[0.37,2.7]
    function axis()
        u  = 4.0*(rand(rng)-0.5)            # mean in [-2,2]
        v  = exp(2.0*(rand(rng)-0.5))       # variance
        qh = 3.0*(rand(rng)-0.5)            # skew in [-1.5,1.5]
        # K must exceed 1+q̂²; draw a positive margin
        K  = 1.0 + qh*qh + exp(2.0*(rand(rng)-0.5))
        (u, v, qh, K)
    end
    ux,vx,qx,Kx = axis(); uy,vy,qy,Ky = axis(); uz,vz,qz,Kz = axis()
    build35(rho, ux,vx,qx,Kx, uy,vy,qy,Ky, uz,vz,qz,Kz)
end

# ---------------------------------------------------------------------------
# Draw update dM. Mix of:
#   (a) small random full-tuple perturbations (mostly interior, θ*≈1)
#   (b) large random perturbations (frequently boundary-crossing)
#   (c) difference toward another realizable state (Mlo2 - Mlo), scaled
#   (d) the production one-sided forms ±6λ·G with G a random flux-gradient
#       proxy — here we emulate G as (Mrand - Mlo) so dM = ±6λ (Mrand - Mlo).
# ---------------------------------------------------------------------------

function verify(N)
    maxdiff = 0.0; diffs = Float64[]; sizehint!(diffs, N)
    overshoots = 0; worst_overshoot = 0.0
    n_bind = 0; n_free = 0
    worst = nothing
    for _ in 1:N
        Mlo = rand_state(rng)
        _state_realizable(Mlo) || continue
        mode = rand(rng, 1:6)
        dM = if mode == 1
            # small random perturbation
            scale = 1e-2
            ntuple(q -> scale * (rand(rng)-0.5) * (abs(Mlo[q]) + 1e-3), 35)
        elseif mode == 2
            # large random perturbation
            scale = 2.0
            ntuple(q -> scale * (rand(rng)-0.5) * (abs(Mlo[q]) + 1e-3), 35)
        elseif mode == 3 || mode == 4
            # difference toward another realizable state
            M2 = rand_state(rng)
            s = mode == 3 ? 1.0 : 6.0
            ntuple(q -> s * (M2[q] - Mlo[q]), 35)
        else
            # production one-sided forms ±6λ·G, G ≈ (Mrand - Mlo)
            M2 = rand_state(rng)
            λ  = exp(1.5*(rand(rng)-0.5))
            sgn = mode == 5 ? -1.0 : 1.0
            ntuple(q -> sgn * 6.0 * λ * (M2[q] - Mlo[q]), 35)
        end

        tb = theta_star_update_dev(Mlo, dM)      # bisection (24 iters)
        tc = theta_star_update_closed(Mlo, dM)   # closed form

        d = abs(tc - tb)
        push!(diffs, d)
        if d > maxdiff
            maxdiff = d
            worst = (Mlo, dM, tb, tc)
        end
        (tc < 1.0 - 1e-12) ? (n_bind += 1) : (n_free += 1)

        # valid lower bound: Mlo + tc*dM realizable (allow tiny tolerance).
        # Test slightly BELOW tc to avoid penalizing the exact-boundary tie:
        # a true θ* sits on the realizability boundary, so evaluate at a point
        # backed off by a hair; overshoot = state at tc is non-realizable AND
        # backing off does not recover it.
        if tc > 0.0
            Mt = ntuple(q -> Mlo[q] + tc * dM[q], 35)
            if !_state_realizable(Mt)
                # back off by relative 1e-7 (bisection's own resolution)
                tcb = tc * (1.0 - 1e-7)
                Mtb = ntuple(q -> Mlo[q] + tcb * dM[q], 35)
                if !_state_realizable(Mtb)
                    overshoots += 1
                    worst_overshoot = max(worst_overshoot, tc)
                end
            end
        end
    end
    sort!(diffs)
    p999 = diffs[clamp(round(Int, 0.999*length(diffs)), 1, length(diffs))]
    return maxdiff, p999, diffs, overshoots, worst_overshoot, worst, n_bind, n_free
end

N = 200_000
println("Drawing $N (Mlo, dM) pairs...")
@time maxdiff, p999, diffs, overshoots, worst_overshoot, worst, n_bind, n_free = verify(N)

@printf("\n== Agreement closed vs bisection (24-iter) ==\n")
@printf("  samples (θ<1 binding):   %d\n", n_bind)
@printf("  samples (θ=1 free):      %d\n", n_free)
@printf("  max  |Δθ|:               %.3e\n", maxdiff)
@printf("  p99.9|Δθ|:               %.3e\n", p999)
@printf("  mean |Δθ|:               %.3e\n", mean(diffs))
@printf("  frac > 1e-5:             %.3e\n", count(>(1e-5), diffs)/length(diffs))
@printf("  frac > 1e-6:             %.3e\n", count(>(1e-6), diffs)/length(diffs))

@printf("\n== Valid lower bound (no overshoot into non-realizable) ==\n")
@printf("  overshoots:              %d / %d\n", overshoots, N)
@printf("  worst overshoot θ:       %.3e\n", worst_overshoot)

if worst !== nothing
    Mlo, dM, tb, tc = worst
    @printf("\n== Worst-disagreement pair ==\n")
    @printf("  θ_bisect=%.10f θ_closed=%.10f  |Δ|=%.3e\n", tb, tc, abs(tc-tb))
    # is the closed one on the boundary?
    for (lbl,t) in (("bisect",tb),("closed",tc))
        Mt = ntuple(q -> Mlo[q] + t*dM[q], 35)
        @printf("    at θ_%s: realizable=%s\n", lbl, _state_realizable(Mt))
    end
end

# ---------------------------------------------------------------------------
# Benchmark: ns/call closed vs bisection over a fixed array of pairs.
# ---------------------------------------------------------------------------
function sumloop_bisect(prs)
    s = 0.0
    @inbounds for i in eachindex(prs)
        s += theta_star_update_dev(prs[i][1], prs[i][2])
    end
    s
end
function sumloop_closed(prs)
    s = 0.0
    @inbounds for i in eachindex(prs)
        s += theta_star_update_closed(prs[i][1], prs[i][2])
    end
    s
end
function benchmark(rng)
    NB = 50_000
    prs = Vector{Tuple{NTuple{35,Float64},NTuple{35,Float64}}}(undef, NB)
    for i in 1:NB
        Mlo = rand_state(rng)
        M2 = rand_state(rng)
        λ = exp(1.5*(rand(rng)-0.5)); sgn = isodd(i) ? -1.0 : 1.0
        dM = ntuple(q -> sgn*6.0*λ*(M2[q]-Mlo[q]), 35)
        prs[i] = (Mlo, dM)
    end
    s1 = sumloop_bisect(prs); s2 = sumloop_closed(prs)   # warmup
    reps = 20
    t1 = @elapsed for _ in 1:reps; s1 = sumloop_bisect(prs); end
    t2 = @elapsed for _ in 1:reps; s2 = sumloop_closed(prs); end
    nb = t1/(reps*NB)*1e9
    nc = t2/(reps*NB)*1e9
    @printf("  bisection (24-iter):  %8.1f ns/call  (checksum %.6f)\n", nb, s1)
    @printf("  closed form:          %8.1f ns/call  (checksum %.6f)\n", nc, s2)
    @printf("  speedup:              %6.2fx\n", nb/nc)
end
println("\n== Benchmark (fixed 50k pairs) ==")
benchmark(rng)

# Assertions (fail loudly for CI)
@assert maxdiff < 1e-4 "closed-form disagreement too large: $maxdiff"
@assert overshoots == 0 "closed form overshot into non-realizable territory: $overshoots cases"
println("\nOK: closed form agrees with bisection and never overshoots.")

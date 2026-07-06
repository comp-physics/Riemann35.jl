# two_stream_escape_3beam.jl — the asymmetric/multi-beam ESCAPE test.
#
# The symmetric 2-beam crossing is single-stream's BEST case (a symmetric two-beam is
# exactly 2 nodes → 5-moment HYQMOM is exact, H=0). The one scenario that could still
# favor two-stream is a state single-stream's 2-node quadrature CANNOT represent: a
# symmetric 3-BEAM. Populations at {−U, 0, +U} have exact center H≈0.5 (>0), but a
# symmetric 2-node closure is FORCED to c4=c2² → H=0. So single-stream must miss the
# 3-beam kurtosis, while two-stream (split at c=0 → each half carries {0-half, ±U},
# up to 4 effective nodes) can hold it.
#
# IC: three spatial zones  x<−δ : u=+U ,  |x|<δ : u=0 ,  x>+δ : u=−U  (all ρ=1, T=1).
# The +U and −U beams stream into the central stationary background → a 3-beam forms at
# the center. We report center total-field H vs the EXACT 3-beam H, plus proj/clip counts.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, Printf, LinearAlgebra

const VAC = 1e-10
const N35 = NTuple{35,Float64}
@inline tup(M, i) = ntuple(q -> @inbounds(M[i, q]), 35)

function margin_H(m)
    m[1] <= VAC && return NaN
    u = m[2]/m[1]; c2 = m[3]/m[1]-u^2; c2 <= 0 && return NaN
    c3 = m[4]/m[1]-3u*m[3]/m[1]+2u^3
    c4 = m[5]/m[1]-4u*m[4]/m[1]+6u^2*m[3]/m[1]-3u^4
    return c4/c2^2 - (c3/c2^1.5)^2 - 1
end

# exact H of an equal-weight 3-beam {−U,0,+U}, each Maxwellian T (mean 0, symmetric).
function exactH_3beam(U, T)
    c2 = (2*(U^2+T) + T)/3
    c4 = (2*(U^4+6U^2*T+3T^2) + 3T^2)/3
    return c4/c2^2 - 1
end

three_zone(U; nx=300, δ=0.3) = begin
    xs = range(-1, 1; length=nx); dx = 2/(nx-1)
    M = Array{Float64}(undef, nx, 35)
    for (i, x) in enumerate(xs)
        u = x < -δ ? U : (x > δ ? -U : 0.0)
        @views M[i, :] .= collect(InitializeM4_35(1.0, u, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
    end
    return M, collect(xs), dx
end

@inline wsp(m) = (ρ=m[1]; ρ<=VAC ? 0.0 : abs(m[2]/ρ) + 3*sqrt(max(m[3]/ρ-(m[2]/ρ)^2, 1e-12)))

function run_production(U; nx=300, nsteps=200, cfl=0.2, δ=0.3, Ma=U)
    M, xs, dx = three_zone(U; nx=nx, δ=δ)
    F = Array{Float64}(undef, nx, 35); vmin = zeros(nx); vmax = zeros(nx); proj = 0
    for step in 1:nsteps
        α = 1e-12; for i in 1:nx; α = max(α, wsp(tup(M, i))); end
        dt = cfl*dx/α
        for i in 1:nx
            m = collect(tup(M, i))
            Fx, _, _, _ = Flux_closure35_and_realizable_3D(m, 0, Ma)
            @views F[i, :] .= Fx
            lo, hi, _ = eigenvalues6_hyperbolic_3D(m, 1, 0, Ma)
            vmin[i] = isfinite(lo) ? lo : -wsp(m); vmax[i] = isfinite(hi) ? hi : wsp(m)
        end
        M .= pas_HLL(M, F, dt, dx, vmin, vmax)
        for i in 1:nx
            m = collect(tup(M, i))
            if m[1] > 0.05 && realizability_margin(m) < 0
                proj += 1; m = realizable_3D_M4(m, Ma)
            end
            @views M[i, :] .= m
        end
    end
    return M, xs, proj
end

function run_twostream(U; nx=300, nsteps=200, cfl=0.2, δ=0.3, Kn=1e9, c=0.0)
    M0, xs, dx = three_zone(U; nx=nx, δ=δ)
    Mp = Vector{N35}(undef, nx); Mm = Vector{N35}(undef, nx)
    vacs = ntuple(q -> q==1 ? VAC : 0.0, 35)
    for i in 1:nx
        m = tup(M0, i); ρ=m[1]; u=m[2]/ρ; T=max(m[3]/ρ-u^2, 1e-12)
        a, b = split_maxwellian35(ρ, u, 0.0, 0.0, T, c)
        Mp[i] = a[1]>VAC ? a : vacs; Mm[i] = b[1]>VAC ? b : vacs
    end
    reset_chain_clips!()
    Fp = Vector{N35}(undef, nx); Fm = similar(Fp)
    for step in 1:nsteps
        sp = 1e-12; for i in 1:nx; sp = max(sp, stream_xspeed(Mp[i]), stream_xspeed(Mm[i])); end
        dt = cfl*dx/sp
        for i in 1:nx; Fp[i] = xflux_plus35(Mp[i]...); Fm[i] = xflux_minus35(Mm[i]); end
        Np = copy(Mp); Nm = copy(Mm)
        for i in 2:nx-1
            Np[i] = ntuple(q -> Mp[i][q]-(dt/dx)*(Fp[i][q]-Fp[i-1][q]), 35)
            Nm[i] = ntuple(q -> Mm[i][q]-(dt/dx)*(Fm[i+1][q]-Fm[i][q]), 35)
        end
        for i in 1:nx; Mp[i]=Np[i]; Mm[i]=Nm[i]; end
        if Kn < 1e5
            for i in 2:nx-1; Mp[i], Mm[i] = bgk_stream_relax(Mp[i], Mm[i], dt, Kn, c); end
        end
    end
    Mt = Array{Float64}(undef, nx, 35)
    for i in 1:nx, q in 1:35; Mt[i, q] = Mp[i][q] + Mm[i][q]; end
    return Mt, Mp, Mm, xs, chain_clips()
end

centerH(M, xs; band=0.12) = begin
    s = 0.0; n = 0
    for i in 1:length(xs)
        abs(xs[i]) <= band || continue
        M[i,1] > 0.1 || continue
        v = margin_H(tup(M, i)); isnan(v) || (s += v; n += 1)
    end
    n == 0 ? NaN : s/n
end

function main()
    println("=== 1D 3-beam ESCAPE test: {−U,0,+U}, does two-stream beat single-stream's 2-node? ===")
    println("    center H: exact 3-beam ~0.5; single-stream symmetric 2-node FORCED to 0 (=fails); two-stream can hold >0.")
    @printf("%-6s | %-9s | %-11s %-9s | %-11s %-9s\n",
            "Ma", "exact Hc", "PROD Hc", "PROD proj", "2STR Hc", "2STR clip")
    for U in (5.0, 10.0, 30.0, 100.0)
        He = exactH_3beam(U, 1.0)
        Mp_prod, xs, proj = run_production(U; nsteps=200, cfl=0.2)
        Hp = centerH(Mp_prod, xs)
        Mt, _, _, xs2, clip = run_twostream(U; nsteps=200, cfl=0.2)
        Ht = centerH(Mt, xs2)
        @printf("%-6g | %+.3e | %+.3e %-9d | %+.3e %-9d\n", U, He, Hp, proj, Ht, clip)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

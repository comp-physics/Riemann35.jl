# two_stream_fidelity_1d.jl — the fidelity payoff: does the two-stream half-space
# scheme resolve counter-streaming beams that single-stream 5-moment HYQMOM cannot?
#
# Problem (exact, collisionless): two beams, left half u=+U, right half u=−U, meet at
# x=0. The exact kinetic solution INTERPENETRATES — each beam passes through undisturbed,
# so at the center the velocity distribution is bimodal (peaks at +U and −U). Single-
# stream 5-moment HYQMOM CANNOT represent a bimodal marginal (it would need H<0, outside
# the realizable set), so at the crossing it is projected to the H=0 boundary and
# spuriously THERMALIZES the beams into one hot blob (no interpenetration). The two-stream
# split carries +U in the + stream and −U in the − stream, each unimodal (H=2), so donor-
# cell advection interpenetrates them exactly.
#
# We run BOTH schemes on the same IC for the same physical time and report, per Ma:
#   * maxu retention  (bulk beam speed kept),
#   * H at the center x≈0  (two-stream: per-stream H≈2; production: driven to ~0),
#   * projection / chain-clip fires  (production must project hard at the crossing),
#   * an interpenetration scalar P = Σρ|u| / (U Σρ) over the central band: ~1 if the beams
#     keep speed U through the crossing (interpenetration), →0 if they thermalize (u→0).
#
# `using Riemann35` provides BOTH the production kernels and the ported two-stream modules.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, Printf, LinearAlgebra

const VAC = 1e-10
const N35 = NTuple{35,Float64}

@inline tup(M, i) = ntuple(q -> @inbounds(M[i, q]), 35)

# H margin of an x-marginal m0..m4 (Maxwellian→2, symmetric two-beam→0 boundary).
function margin_H(m)
    m[1] <= VAC && return NaN
    u = m[2]/m[1]; c2 = m[3]/m[1]-u^2; c2 <= 0 && return NaN
    c3 = m[4]/m[1]-3u*m[3]/m[1]+2u^3
    c4 = m[5]/m[1]-4u*m[4]/m[1]+6u^2*m[3]/m[1]-3u^4
    return c4/c2^2 - (c3/c2^1.5)^2 - 1
end

# ---- counter-streaming IC: left half +U, right half −U, uniform density & T ------
crossing_ic1d(U; nx=200) = begin
    xs = range(-1, 1; length=nx); dx = 2/(nx-1)
    M = Array{Float64}(undef, nx, 35)
    for (i, x) in enumerate(xs)
        u = x < 0 ? U : -U
        @views M[i, :] .= collect(InitializeM4_35(1.0, u, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
    end
    return M, collect(xs), dx
end

@inline wsp(m) = (ρ = m[1]; ρ<=VAC ? 0.0 : abs(m[2]/ρ) + 3*sqrt(max(m[3]/ρ-(m[2]/ρ)^2, 1e-12)))

# ---------- production single-stream 1D (HLL flux + realizability projection) -----
function run_production(U; nx=200, nsteps=60, cfl=0.2, Ma=U)
    M, xs, dx = crossing_ic1d(U; nx=nx)
    F = Array{Float64}(undef, nx, 35); vmin = zeros(nx); vmax = zeros(nx)
    proj = 0
    for step in 1:nsteps
        α = 1e-12
        for i in 1:nx; α = max(α, wsp(tup(M, i))); end
        dt = cfl*dx/α
        for i in 1:nx
            m = collect(tup(M, i))
            Fx, _, _, _ = Flux_closure35_and_realizable_3D(m, 0, Ma)
            @views F[i, :] .= Fx
            lo, hi, _ = eigenvalues6_hyperbolic_3D(m, 1, 0, Ma)
            vmin[i] = isfinite(lo) ? lo : -wsp(m); vmax[i] = isfinite(hi) ? hi : wsp(m)
        end
        Mnew = pas_HLL(M, F, dt, dx, vmin, vmax)
        M .= Mnew
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

# ---------- two-stream 1D (donor-cell half-space x-flux, no projection) ------------
function run_twostream(U; nx=200, nsteps=60, cfl=0.2, Kn=1e9, c=0.0)
    M0, xs, dx = crossing_ic1d(U; nx=nx)
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
        sp = 1e-12
        for i in 1:nx; sp = max(sp, stream_xspeed(Mp[i]), stream_xspeed(Mm[i])); end
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
    # fold to a single total field for uniform reporting
    Mt = Array{Float64}(undef, nx, 35)
    for i in 1:nx, q in 1:35; Mt[i, q] = Mp[i][q] + Mm[i][q]; end
    return Mt, Mp, Mm, xs, chain_clips()
end

# central effective temperature ratio T_eff/U^2 over |x|<=band (mass-weighted). Both
# schemes conserve energy, so this ~1 for BOTH — a control confirming the fidelity
# difference is distribution SHAPE (kurtosis), not energy.
function centerTratio(M, xs, U; band=0.25)
    num = 0.0; den = 0.0
    for i in 1:length(xs)
        abs(xs[i]) <= band || continue
        ρ = M[i, 1]; ρ > 0.05 || continue
        u = M[i,2]/ρ; T = M[i,3]/ρ - u^2
        num += ρ*max(T, 0.0); den += ρ
    end
    den <= 0 && return NaN
    return num/(den*U^2)
end

# center H (mean over |x|<=band, mass-gated). Exact bimodal answer ~0; thermalized ~2.
function centerH(M, xs; band=0.25)
    s = 0.0; n = 0
    for i in 1:length(xs)
        abs(xs[i]) <= band || continue
        M[i,1] > 0.1 || continue
        v = margin_H(tup(M, i)); isnan(v) || (s += v; n += 1)
    end
    return n == 0 ? NaN : s/n
end

function main()
    println("=== 1D counter-streaming fidelity: two-stream vs production single-stream ===")
    println("    Hc(tot) = center total-field H-margin: exact bimodal answer ~0; thermalized (wrong) ~2.")
    println("    Tr = center T_eff/U^2 (energy control, ~1 for both). Hc(±) = per-stream H (2STR streams stay Maxwellian ~2).")
    @printf("%-6s | %-11s %-10s %-9s | %-11s %-11s %-10s %-9s\n",
            "Ma", "PROD Hc(tot)", "PROD Tr", "PROD proj", "2STR Hc(tot)", "2STR Hc(±)", "2STR Tr", "2STR clip")
    for U in (5.0, 10.0, 30.0, 100.0)
        Mp_prod, xs, proj = run_production(U; nx=200, nsteps=250, cfl=0.2)
        Pp = centerTratio(Mp_prod, xs, U); Hp = centerH(Mp_prod, xs)
        Mt, Mplus, Mminus, xs2, clip = run_twostream(U; nx=200, nsteps=250, cfl=0.2)
        Pt = centerTratio(Mt, xs2, U); Ht = centerH(Mt, xs2)
        # per-stream center H (the streams themselves, which stay unimodal)
        hpm = Inf
        for i in 1:length(xs2)
            abs(xs2[i]) <= 0.25 || continue
            if Mplus[i][1] > 0.1;  v = margin_H(Mplus[i]);  isnan(v) || (hpm = min(hpm, v)); end
            if Mminus[i][1] > 0.1; v = margin_H(Mminus[i]); isnan(v) || (hpm = min(hpm, v)); end
        end
        @printf("%-6g | %+.3e %+.3e %-9d | %+.3e %+.3e %+.3e %-9d\n",
                U, Hp, Pp, proj, Ht, hpm, Pt, clip)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

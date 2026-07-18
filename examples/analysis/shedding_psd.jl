# shedding_psd.jl — vortex-shedding diagnostics for a bubble2d_shedding run.
#
# Extracts a lift-like transverse signal from a JLD2 snapshot series and computes
# its power spectral density, reporting the dominant frequency in Strouhal units
# f*D/U (D=U=1 here) for comparison with McMullen & Gallis (SAND2024-13841J):
# the primary shedding peak at f_s*D/U ~ 0.12-0.145 and the Brillouin acoustic
# peaks at f*D/U ~ n/(10*Ma) = n/3 (Ma=0.3).
#
# Signals (both antisymmetric — zero for a symmetric wake, oscillatory if shedding):
#   Lwake(t) = sum over the wake (x > x_bubble) of rho*v  (net transverse momentum)
#   vprobe(t)= v at a point ~2D downstream on the centerline
#
# Uniform-dt snapshots (the GPU driver / fixed-dt runs) give a clean uniform time
# series, so a plain DFT over a scanned frequency band suffices (no FFT dependency).
#
# Usage:  julia --project=. examples/analysis/shedding_psd.jl <run.jld2> [x_bubble] [D_downstream]
using JLD2, Printf

function load_series(f)
    jf = jldopen(f, "r")
    p  = jf["meta/params"]
    ns = 0; while haskey(jf, "snapshots/" * lpad(ns + 1, 6, "0") * "/M"); ns += 1; end
    ts = Float64[]; Ms = Array{Float64,4}[]
    for s in 1:ns
        key = "snapshots/" * lpad(s, 6, "0")
        push!(ts, jf[key * "/t"]); push!(Ms, jf[key * "/M"])
    end
    close(jf)
    return p, ts, Ms
end

# domain geometry from params (falls back to the bubble2d_shedding defaults)
function geom(p, Nx, Ny)
    xmin = get(p, :xmin, 0.0); xmax = get(p, :xmax, 15.0)
    ymin = get(p, :ymin, 0.0); ymax = get(p, :ymax, 10.0)
    xc(i) = xmin + (i - 0.5) * (xmax - xmin) / Nx
    yc(j) = ymin + (j - 0.5) * (ymax - ymin) / Ny
    return xc, yc
end

# single-sided DFT power over a scanned Strouhal band; returns (freqs, power).
function psd_scan(t, y; fmin = 0.03, fmax = 1.6, nf = 600)
    y0 = y .- sum(y) / length(y)                    # de-mean
    w  = [0.5 * (1 - cos(2π * (k - 1) / (length(y) - 1))) for k in 1:length(y)]  # Hann
    yw = y0 .* w
    fs = range(fmin, fmax; length = nf)
    P  = similar(collect(fs))
    @inbounds for (m, f) in enumerate(fs)
        re = 0.0; im = 0.0
        for k in eachindex(t)
            ph = 2π * f * t[k]
            re += yw[k] * cos(ph); im -= yw[k] * sin(ph)
        end
        P[m] = re^2 + im^2
    end
    return collect(fs), P
end

function top_peaks(fs, P; n = 3)
    idx = Int[]
    for m in 2:length(P)-1
        (P[m] > P[m-1] && P[m] >= P[m+1]) && push!(idx, m)
    end
    sort!(idx; by = m -> -P[m])
    return [(fs[m], P[m]) for m in idx[1:min(n, length(idx))]]
end

function main()
    isempty(ARGS) && error("usage: shedding_psd.jl <run.jld2> [x_bubble=5] [D_down=2]")
    f = ARGS[1]
    xb = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5.0
    Dd = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 2.0
    p, ts, Ms = load_series(f)
    Nx, Ny = size(Ms[1], 1), size(Ms[1], 2)
    xc, yc = geom(p, Nx, Ny)
    kz = 1
    jc = argmin(abs.([yc(j) for j in 1:Ny] .- (get(p, :ymin, 0.0) + get(p, :ymax, 10.0)) / 2))
    ip = argmin(abs.([xc(i) for i in 1:Nx] .- (xb + Dd)))   # probe column ~Dd behind bubble
    wake_i = [i for i in 1:Nx if xc(i) > xb]

    Lwake = Float64[]; vprobe = Float64[]
    for M in Ms
        rho = @view M[:, :, kz, 1]; rv = @view M[:, :, kz, 6]   # M010 = rho*v
        push!(Lwake, sum(@view rv[wake_i, :]))
        push!(vprobe, M[ip, jc, kz, 6] / M[ip, jc, kz, 1])
    end

    tspan = ts[end] - ts[1]
    @printf("run: %s\n", f)
    @printf("snapshots=%d  t in [%.3f, %.3f]  (%.2f D/U)  dt_snap~%.4f\n",
            length(ts), ts[1], ts[end], tspan, length(ts) > 1 ? ts[2] - ts[1] : 0.0)
    @printf("Lwake(t): mean=%.3e  std=%.3e  min=%.3e max=%.3e\n",
            sum(Lwake)/length(Lwake), std_(Lwake), minimum(Lwake), maximum(Lwake))
    @printf("vprobe(t) @x=%.2f,y=centerline: mean=%.3e std=%.3e\n",
            xc(ip), sum(vprobe)/length(vprobe), std_(vprobe))

    # drop the first ~20% as startup transient before spectral analysis
    i0 = max(1, round(Int, 0.2 * length(ts)))
    for (nm, sig) in (("Lwake", Lwake), ("vprobe", vprobe))
        t = ts[i0:end]; y = sig[i0:end]
        s = std_(y)
        if s < 1e-12 * (abs(sum(y)/length(y)) + 1)
            @printf("[%s] essentially flat (std=%.2e) — NO oscillation detected\n", nm, s)
            continue
        end
        fs, P = psd_scan(t, y)
        pk = top_peaks(fs, P; n = 3)
        @printf("[%s] top peaks (f*D/U, rel power):", nm)
        pmax = maximum(P)
        for (fq, pw) in pk; @printf("  (%.3f, %.2f)", fq, pw / pmax); end
        @printf("\n         Strouhal ref: shedding 0.12-0.145 | Brillouin n/3 = 0.33,0.67,1.0\n")
    end
end

std_(x) = (m = sum(x)/length(x); sqrt(sum((xi-m)^2 for xi in x)/length(x)))

main()

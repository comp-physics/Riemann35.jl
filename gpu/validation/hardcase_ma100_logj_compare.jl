# hardcase_ma100_logj_compare.jl — the real hard 3D case: Ma=100 :crossing_matlab jets
# marched with GPU order-3 (WENO5 + fixed closed-form θ*-IDP), run TWICE per grid:
# baseline (raw-moment recon) vs log-Jacobi marginal recon. Reports, per grid & mode:
# survival (finite & ρ>0), ρ range, mass drift, wall, throughput; and the log-J-vs-
# baseline solution change (max |Δρ|, max |Δp|). This is the production evidence for
# whether log-Jacobi should be promoted to default: stability + throughput cost + effect.
#
# Run under gpuenv2 (see run_hiorder3_ma100_gpu.jl header for env).
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
g = 4

# ensure the crossing vectors exist (main-env dump, same as run_hiorder3_ma100_gpu.jl)
need = ["r3d_cross_ma100.f64", "r3d_cross_ma100.meta"]
if any(f -> !isfile(joinpath(DATA, f)), need)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_march.jl")
    println("[setup] crossing vectors missing → running dump in main env …")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end
cmeta = split(strip(read(joinpath(DATA, "r3d_cross_ma100.meta"), String)), '\n')
MaB = parse(Float64, cmeta[1])
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3B = max(40.0, 4.0 + abs(MaB)/2.0)

function build_crossing(N::Int, g::Int)
    nf = N + 2g; G = zeros(35, nf, nf, nf)
    Csize = floor(Int, 0.1 * N)
    Minb = div(N,2) - Csize; Maxb = div(N,2)
    Mnt  = div(N,2) + 1;     Maxt = div(N,2) + 1 + Csize
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        vec = bg
        if Minb <= i <= Maxb && Minb <= j <= Maxb && Minb <= k <= Maxb; vec = Mb; end
        if Mnt  <= i <= Maxt && Mnt  <= j <= Maxt && Mnt  <= k <= Maxt; vec = Mt; end
        @views G[:, i+g, j+g, k+g] .= vec
    end
    return G
end

# pressure proxy: trace of the 2nd central-moment block via marginal C200+C020+C002 raw
# slots (indices 5,10,15 are m200/m020/m002 in the 35-vec? use density-normalized 2nd raw)
pmap(Gi) = begin
    ρ = @view Gi[1, :, :, :]
    (@view Gi[5, :, :, :]) ./ max.(ρ, 1e-300)  # m200/ρ, a directional pressure proxy
end

@printf("Ma=%.0f crossing jets — order-3 (fixed θ*), baseline vs log-Jacobi\n", MaB)
@printf("%-6s %-8s %-9s %-24s %-12s %-9s %-13s\n",
        "grid", "mode", "survived", "rho range", "mass drift", "wall(s)", "Mcell·step/s")
# warm-up compile (small) so the reported wall times exclude JIT
let Gw = CuArray(build_crossing(16, g))
    try; march3d_order3_gpu!(Gw, 1.0/16, MaB, 1; s3max=s3B); catch; end
    try; march3d_order3_gpu!(Gw, 1.0/16, MaB, 1; s3max=s3B, use_logjacobi_recon=true); catch; end
    CUDA.synchronize()
end

for N in (32, 64, 128)
    nstep = 20; dxN = 1.0 / N
    Gh = build_crossing(N, g)
    mass0 = sum(@view Gh[1, g+1:g+N, g+1:g+N, g+1:g+N])
    results = Dict{String,Any}()
    for (mode, ljflag) in (("base", false), ("logJ", true))
        local G = CuArray(copy(Gh))
        local surv=false; local rmin=NaN; local rmax=NaN; local drift=NaN; local wall=NaN; local thru=NaN; local crashed=""
        try
            CUDA.synchronize(); local t0 = time()
            march3d_order3_gpu!(G, dxN, MaB, nstep; s3max=s3B, use_logjacobi_recon=ljflag)
            CUDA.synchronize(); wall = time() - t0
            local Gi = Array(G)[:, g+1:g+N, g+1:g+N, g+1:g+N]
            local ρ = Gi[1, :, :, :]
            surv = all(isfinite, Gi) && minimum(ρ) > 0.0
            rmin = minimum(ρ); rmax = maximum(ρ)
            drift = abs(sum(ρ) - mass0) / mass0
            thru = (N^3 * nstep) / wall / 1e6
            results[mode] = Gi
        catch e
            crashed = sprint(showerror, e)[1:min(end,70)]
        end
        if crashed == ""
            @printf("%-6s %-8s %-9s [%.3e, %.3e] %-12.3e %-9.2f %-13.2f\n",
                    "$(N)³", mode, surv ? "YES" : "NO", rmin, rmax, drift, wall, thru)
        else
            @printf("%-6s %-8s %-9s CRASHED: %s\n", "$(N)³", mode, "NO", crashed)
        end
    end
    if haskey(results,"base") && haskey(results,"logJ")
        local dρ = maximum(abs.(results["logJ"][1,:,:,:] .- results["base"][1,:,:,:]))
        local dp = maximum(abs.(pmap(results["logJ"]) .- pmap(results["base"])))
        local rref = maximum(abs.(results["base"][1,:,:,:]))
        @printf("       log-J vs base: max|Δρ|=%.3e (rel %.2e)  max|Δ(m200/ρ)|=%.3e\n",
                dρ, dρ/max(rref,1e-300), dp)
    end
end
println("\nDone. Headline: both modes should SURVIVE; throughput ratio = log-J cost; Δ = log-J effect at scale.")

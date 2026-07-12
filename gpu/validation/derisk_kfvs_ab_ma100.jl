# derisk_kfvs_ab_ma100.jl  de-risk diagnostic (hard Ma=100 crossing A/B)
# OFF-vs-ON de-risk of the F3 kinetic-FVS flux anchor on the HARD Ma=100
# crossing-jets IC (r3d_cross_ma100 vectors: jets at ±Ma/√3), GPU order-3
# (WENO5 + θ*-IDP) SSP-RK3 march. Same order-3 path, single flag swap:
#   OFF = current default (use_kfvs_anchor=false)
#   ON  = F3 kinetic-FVS flux anchor (use_kfvs_anchor=true)
#
# Headline metrics per grid, per branch:
#   survived (finite & rho>0), rho range, MASS drift, ENERGY drift
#   (energy = Σ M200+M020+M002 = trace of the stress = conserved total energy),
#   steps, wall(s), throughput.  The energy drift is the discriminator: F3 was
#   built because the per-cell STATE form drifted energy to 9.7e-1 at Ma=100
#   (notes §1319); does the flux form hold like the default projection path?
#
# Run under gpuenv2:
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   $JULIA --project=gpu/gpuenv2 gpu/validation/derisk_kfvs_ab_ma100.jl
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
g = Timestep3DOrder3GPU.HALO3      # order-3 needs g=8 (was hardcoded 4 = halo mismatch)
NSTEP = 20
GRIDS = (32, 64)

# --- ensure the crossing IC vectors exist (main-env CPU dump if missing) ------
need = ["r3d_cross_ma100.f64", "r3d_cross_ma100.meta"]
if any(f -> !isfile(joinpath(DATA, f)), need)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_march.jl")
    println("[setup] crossing IC vectors missing → running CPU dump in main env ($repo) …")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end

cmeta = split(strip(read(joinpath(DATA, "r3d_cross_ma100.meta"), String)), '\n')
MaB = parse(Float64, cmeta[1])
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3B = max(40.0, 4.0 + abs(MaB) / 2.0)

# interior->haloed cube layout identical to run_hiorder3_ma100_gpu.jl Gate B
function build_crossing(N::Int, g::Int)
    nf = N + 2g
    G = zeros(35, nf, nf, nf)
    Csize = floor(Int, 0.1 * N)
    Minb = div(N, 2) - Csize; Maxb = div(N, 2)
    Mnt  = div(N, 2) + 1;      Maxt = div(N, 2) + 1 + Csize
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        vec = bg
        if Minb <= i <= Maxb && Minb <= j <= Maxb && Minb <= k <= Maxb; vec = Mb; end
        if Mnt  <= i <= Maxt && Mnt  <= j <= Maxt && Mnt  <= k <= Maxt; vec = Mt; end
        @views G[:, i+g, j+g, k+g] .= vec
    end
    return G
end

# energy proxy = Σ (M200 + M020 + M002) over interior  (indices 3,10,20)
interior(GiB) = GiB
en_of(GiB) = sum(@view GiB[3, :, :, :]) + sum(@view GiB[10, :, :, :]) + sum(@view GiB[20, :, :, :])

# sched=nothing => adaptive (returns the schedule); pass a saved schedule to REPLAY it so
# OFF and ON march on IDENTICAL timesteps (fair conservation/perf comparison).
function run_branch(N::Int, use_anchor::Bool; sched=nothing)
    Ghost = build_crossing(N, g)
    G = CuArray(Ghost)
    dxN = 1.0 / N
    Gi0 = Ghost[:, g+1:g+N, g+1:g+N, g+1:g+N]
    mass0 = sum(@view Gi0[1, :, :, :]); en0 = en_of(Gi0)

    t0 = time()
    dts = march3d_order3_gpu!(G, dxN, MaB, NSTEP; dts=sched, s3max=s3B, use_kfvs_anchor=use_anchor)
    CUDA.synchronize()
    wall = time() - t0

    GiB = Array(G)[:, g+1:g+N, g+1:g+N, g+1:g+N]
    rho = GiB[1, :, :, :]
    survived = all(isfinite, GiB) && minimum(rho) > 0.0
    mass1 = sum(rho); en1 = en_of(GiB)
    mdrift = abs(mass1 - mass0) / mass0
    edrift = abs(en1 - en0) / max(abs(en0), 1e-300)
    thru = (N^3 * NSTEP) / wall / 1e6
    return (; survived, rmin=minimum(rho), rmax=maximum(rho), mdrift, edrift,
            wall, thru, treached=sum(dts), sched=Array(dts))
end

prow(tag, r) = @printf("%-6s %-4s %-9s [%.3e,%.3e]  %-11.3e %-11.3e %-9.2f %-11.2f\n",
        r.N, tag, r.survived ? "YES" : "NO", r.rmin, r.rmax, r.mdrift, r.edrift, r.wall, r.thru)

@printf("=== F3 kinetic-FVS anchor de-risk: hard Ma=%.0f crossing, order-3 (g=HALO3=%d), %d steps ===\n", MaB, g, NSTEP)
@printf("(OFF adaptive schedule is SAVED and REPLAYED for ON — identical timesteps; ON failure is caught)\n")
@printf("%-6s %-4s %-9s %-24s %-11s %-11s %-9s %-11s\n",
        "grid", "br", "survived", "rho range", "mass drift", "en drift", "wall(s)", "Mcell·st/s")
for N in GRIDS
    roff = run_branch(N, false)
    prow("OFF", merge(roff, (; N="$(N)³")))
    try
        ron = run_branch(N, true; sched=roff.sched)   # REPLAY OFF's saved schedule
        prow("ON ", merge(ron, (; N="$(N)³")))
    catch e
        @printf("%-6s %-4s FAILED — %s  (OFF result stands; comparison not aborted)\n",
                "$(N)³", "ON", sprint(showerror, e)[1:min(end,70)])
    end
    @printf("       t_reached=%.4e over %d steps (identical schedule both branches)\n", roff.treached, NSTEP)
end
println("\nDone.  Discriminator = energy drift ON vs OFF on IDENTICAL timesteps (single flag swap, g=HALO3).")

# run_hiorder3_ma100_gpu.jl — GPU order-3 (WENO5 + θ*-IDP) SSP-RK3 march driver.
#
#   Gate A (parity):  short-time GPU-vs-CPU agreement.  A small Ma=2 realizable box
#                     marched a few steps with matched dt; compares interior density
#                     against the CPU `step_highorder_3d!` reference (the exact
#                     operator the GPU march reproduces).  Reports max abs rel diff.
#   Gate B (payoff):  Ma=100 :crossing_matlab survival at 32³/64³/128³.  Reports per
#                     grid: survived (finite & rho>0), rho range, mass drift, steps,
#                     wall time, throughput (cells·steps/s).
#
# Run under gpuenv2:
#   export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   $JULIA --project=gpu/gpuenv2 gpu/validation/run_hiorder3_ma100_gpu.jl
#
# The CPU reference / crossing vectors (dump_cpu_hiorder3_march.jl) are produced
# automatically in the MAIN package env if missing.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
g = 4

# --- ensure the CPU reference + crossing vectors exist (main-env dump) --------
need = ["r3d_march.meta", "r3d_march_M0.f64", "r3d_march_MK.f64",
        "r3d_cross_ma100.f64", "r3d_cross_ma100.meta"]
if any(f -> !isfile(joinpath(DATA, f)), need)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_march.jl")
    println("[gate] CPU reference missing → running dump in main env ($repo) …")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end

# build an outflow-haloed cube (35,nf,nf,nf) from an interior (35,n,n,n): clamp.
function haloed_cube(Mint::Array{Float64,4}, n::Int, g::Int)
    nf = n + 2g
    G = zeros(35, nf, nf, nf)
    cl(a) = a < 1 ? 1 : (a > n ? n : a)
    @inbounds for c in 1:nf, b in 1:nf, a in 1:nf
        @views G[:, a, b, c] .= Mint[:, cl(a-g), cl(b-g), cl(c-g)]
    end
    return G
end

# ===========================================================================
# GATE A — GPU march vs CPU step_highorder_3d! (order=3), matched dt, K steps.
# ===========================================================================
println("=== Gate A: GPU order-3 march vs CPU step_highorder_3d! (parity) ===")
meta = split(strip(read(joinpath(DATA, "r3d_march.meta"), String)), '\n')
nA = parse(Int, meta[1]); dxA = parse(Float64, meta[2]); MaA = parse(Float64, meta[3])
gA = parse(Int, meta[4]); dtA = parse(Float64, meta[5]); K = parse(Int, meta[6]); s3A = parse(Float64, meta[7])
@assert gA == g
M0 = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_march_M0.f64")))), 35, nA, nA, nA)
MK = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_march_MK.f64")))), 35, nA, nA, nA)

GA = CuArray(haloed_cube(M0, nA, g))
march3d_order3_gpu!(GA, dxA, MaA, K; dts=fill(dtA, K), s3max=s3A)
Gi = Array(GA)[:, g+1:g+nA, g+1:g+nA, g+1:g+nA]

rho_gpu = Gi[1, :, :, :]; rho_cpu = MK[1, :, :, :]
relrho = maximum(abs.(rho_gpu .- rho_cpu) ./ max.(abs.(rho_cpu), 1e-300))
absall = maximum(abs.(Gi .- MK))
@printf("  n=%d Ma=%.0f dt=%.3e K=%d:  max abs rel diff (density) = %.3e   max abs diff (all moments) = %.3e\n",
        nA, MaA, dtA, K, relrho, absall)
gateA = relrho <= 1e-7
@printf("  Gate A: %s (short-time parity vs step_highorder_3d!; ~ulp·steps·eig-floor)\n",
        gateA ? "PASS" : "REVIEW")

# ===========================================================================
# GATE B — Ma=100 :crossing_matlab survival at 32³ / 64³ / 128³.
# ===========================================================================
println("\n=== Gate B: Ma=100 :crossing_matlab survival at scale ===")
cmeta = split(strip(read(joinpath(DATA, "r3d_cross_ma100.meta"), String)), '\n')
MaB = parse(Float64, cmeta[1])
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3B = max(40.0, 4.0 + abs(MaB)/2.0)

function build_crossing(N::Int, g::Int)
    nf = N + 2g
    G = zeros(35, nf, nf, nf)
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

@printf("%-6s %-9s %-26s %-13s %-7s %-11s %-13s\n",
        "grid", "survived", "rho range", "mass drift", "steps", "wall(s)", "Mcell·step/s")
for N in (32, 64, 128)
    nstep = 20
    Ghost = build_crossing(N, g)
    local G = CuArray(Ghost)
    local dxN = 1.0 / N
    local mass0 = sum(@view Ghost[1, g+1:g+N, g+1:g+N, g+1:g+N])

    local t0 = time()
    local dts = march3d_order3_gpu!(G, dxN, MaB, nstep; s3max=s3B)
    CUDA.synchronize()
    local wall = time() - t0

    local GiB = Array(G)[:, g+1:g+N, g+1:g+N, g+1:g+N]
    local rho = GiB[1, :, :, :]
    local survived = all(isfinite, GiB) && minimum(rho) > 0.0
    local mass1 = sum(rho)
    local drift = abs(mass1 - mass0) / mass0
    local thru = (N^3 * nstep) / wall / 1e6
    @printf("%-6s %-9s [%.4e, %.4e]  %-13.3e %-7d %-11.3f %-13.2f\n",
            "$(N)³", survived ? "YES" : "NO", minimum(rho), maximum(rho),
            drift, nstep, wall, thru)
    @printf("       t_reached=%.4e  (sum dt over %d steps)\n", sum(dts), nstep)
end

println("\nDone.  (Gate A is diagnostic parity; Gate B survival is the headline.)")

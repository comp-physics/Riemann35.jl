# validate_run_gpu_order3.jl — order-3 through the STANDARD single-GPU run path.
#
# Proves that `run_gpu_3d(order=3, ...)` (gpu/gpu_run.jl) drives the full order-3
# WENO5 + θ*-IDP march (gpu/timestep3d_order3_gpu.jl `march3d_order3_gpu!`) end to
# end — CFL dt, snapshot dumps, realizability — and that routing through the run
# driver is the SAME compute as calling the march directly.
#
#   Gate 1 (survival):   run_gpu_3d(order=3) on a Ma=100 crossing-jets IC, a few
#                        steps + snapshots, single GPU. Snapshot must be finite,
#                        rho>0, and stored in the standard (Nx,Ny,Nz,35) layout.
#   Gate 2 (equivalence): the run-driver interior == a direct march3d_order3_gpu!
#                        from the same IC/dt (build_haloed_cube → march → extract).
#                        Reports max abs rel diff (expected ~machine / exact).
#
# Run under gpuenv2:
#   export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   $JULIA --project=gpu/gpuenv2 gpu/validation/validate_run_gpu_order3.jl
using CUDA, JLD2, Printf

include(joinpath(@__DIR__, "..", "gpu_run.jl"))
using .GPURun: run_gpu_3d
using .GPURun.Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))

# --- Ma=100 crossing-jets IC (same construction the order-3 payoff gate uses) ---
cfile = joinpath(DATA, "r3d_cross_ma100.f64"); mfile = joinpath(DATA, "r3d_cross_ma100.meta")
(isfile(cfile) && isfile(mfile)) ||
    error("missing $cfile / $mfile — run gpu/validation/run_hiorder3_ma100_gpu.jl once to generate the crossing vectors")
Ma = parse(Float64, split(strip(read(mfile, String)), '\n')[1])
cross = reshape(collect(reinterpret(Float64, read(cfile))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3 = max(40.0, 4.0 + abs(Ma) / 2.0)

# interior (35,N,N,N) crossing field (two counter-set jet blocks on a background).
function crossing_interior(N::Int)
    M = zeros(35, N, N, N)
    Csize = floor(Int, 0.1 * N)
    Minb = div(N, 2) - Csize; Maxb = div(N, 2)
    Mnt  = div(N, 2) + 1;     Maxt = div(N, 2) + 1 + Csize
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        vec = bg
        if Minb <= i <= Maxb && Minb <= j <= Maxb && Minb <= k <= Maxb; vec = Mb; end
        if Mnt  <= i <= Maxt && Mnt  <= j <= Maxt && Mnt  <= k <= Maxt; vec = Mt; end
        @views M[:, i, j, k] .= vec
    end
    return M
end

N = 32; nstep = 8; dx = 1.0 / N
M0 = crossing_interior(N)
tmp = mktempdir()

# ===========================================================================
# GATE 1 — run_gpu_3d(order=3) survives + writes the standard snapshot layout.
# ===========================================================================
println("=== Gate 1: run_gpu_3d(order=3) end-to-end survival + snapshots ===")
snap1 = joinpath(tmp, "order3_gate1.jld2")
run_gpu_3d(copy(M0), dx, Ma, nstep; snapshot_interval=4, snapshot_filename=snap1,
           order=3, s3max=s3)

Msnap, nsnap = jldopen(snap1, "r") do jf
    ns = jf["meta/n_snapshots"]
    jf["snapshots/$(lpad(ns, 6, '0'))/M"], ns
end
rho1 = @view Msnap[:, :, :, 1]
finite1 = all(isfinite, Msnap); rhomin = minimum(rho1); rhomax = maximum(rho1)
layout_ok = size(Msnap) == (N, N, N, 35)
mass0 = sum(@view M0[1, :, :, :]); drift = abs(sum(rho1) - mass0) / mass0
@printf("  snapshots=%d  layout=%s (%s)  finite=%s  rho∈[%.4e, %.4e]  mass drift=%.3e\n",
        nsnap, layout_ok ? "OK" : "BAD", string(size(Msnap)), finite1, rhomin, rhomax, drift)
gate1 = layout_ok && finite1 && rhomin > 0.0
@printf("  Gate 1: %s\n", gate1 ? "PASS" : "FAIL")

# ===========================================================================
# GATE 2 — run-driver interior == direct march3d_order3_gpu! (same compute).
# ===========================================================================
println("\n=== Gate 2: run_gpu_3d(order=3) == direct march3d_order3_gpu! ===")
# Run driver with a DIFFERENT snapshot cadence (forces multiple march segments:
# nstep=8 → 3,3,2) to prove segmentation does not change the compute.
snap2 = joinpath(tmp, "order3_gate2.jld2")
run_gpu_3d(copy(M0), dx, Ma, nstep; snapshot_interval=3, snapshot_filename=snap2,
           order=3, s3max=s3)
Mrun = jldopen(snap2, "r") do jf
    jf["snapshots/$(lpad(jf["meta/n_snapshots"], 6, '0'))/M"]  # (N,N,N,35)
end
run_int = permutedims(Mrun, (4, 1, 2, 3))                        # -> (35,N,N,N)

# direct: build the shared haloed cube, march nstep in one shot, extract interior.
G = build_haloed_cube(CuArray(copy(M0)))
march3d_order3_gpu!(G, dx, Ma, nstep; s3max=s3)
Mi = CUDA.zeros(Float64, 35, N, N, N); interior_from_cube!(Mi, G)
dir_int = Array(Mi)

absdiff = maximum(abs.(run_int .- dir_int))
reldiff = maximum(abs.(run_int .- dir_int) ./ max.(abs.(dir_int), 1e-300))
@printf("  max abs diff (all moments) = %.3e   max rel diff = %.3e\n", absdiff, reldiff)
gate2 = reldiff <= 1e-12
@printf("  Gate 2: %s (threshold rel diff ≤ 1e-12)\n", gate2 ? "PASS" : "FAIL")

println()
@printf("SUMMARY: Gate 1 (survival) %s | Gate 2 (equivalence) %s\n",
        gate1 ? "PASS" : "FAIL", gate2 ? "PASS" : "FAIL")
(gate1 && gate2) || error("validation FAILED")
println("All gates passed.")

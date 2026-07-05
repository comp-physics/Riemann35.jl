# validate_slab_order3_gpu.jl — multi-GPU z-slab order-3 (WENO5 + θ*-IDP) consistency.
#
# Proves the z-slab order-3 march (march3d_slab_gpu!(order=3), gpu/timestep3d_gpu.jl →
# march3d_slab_order3_gpu!) is RANK-CONSISTENT: a 2-rank z-slab run reproduces the
# single-GPU march (march3d_order3_gpu!) bit-for-bit, and total mass is identical across
# ranks. This exercises the g=8 z-halo exchange + the z rank-boundary θ layer (the shared
# z-interface θ = min(own θ, z-neighbour halo-cell θ), single-valued across ranks).
#
#   Gate 1 (consistency):  2-rank z-slab final field == single-GPU final field (max abs
#                          interior diff → ~0 / bit-identical).
#   Gate 2 (conservation): total mass (Σ rho) identical, 1-rank vs gathered 2-rank.
#
# DEVICE BINDING: each rank binds device `rank % CUDA.ndevices()`. On a 2-GPU allocation
# the two ranks land on distinct physical GPUs (a true 2-device run); on a 1-GPU node they
# share device 0 (validating the EXCHANGE + rank-θ LOGIC). The z-slab exchange is
# device-count-agnostic; the SUMMARY line reports the ACTUAL per-rank device (name + short
# UUID) so the message is truthful either way.
#
# Run under gpuenv2:
#   export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
#   srun --mpi=pmix -n 2 --overlap $JULIA --project=gpu/gpuenv2 gpu/validation/validate_slab_order3_gpu.jl
using MPI, CUDA, Printf
H = joinpath(@__DIR__, "..")
include(joinpath(H, "timestep3d_gpu.jl")); using .Timestep3DGPU
using .Timestep3DGPU.Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!

DATA = get(ENV, "RIEMANN35_DATA", joinpath(H, "..", "data"))

MPI.Init(); comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
CUDA.device!(rank % CUDA.ndevices())

# --- Ma=100 crossing-jets IC (same construction as validate_run_gpu_order3.jl) ---
cfile = joinpath(DATA, "r3d_cross_ma100.f64"); mfile = joinpath(DATA, "r3d_cross_ma100.meta")
(isfile(cfile) && isfile(mfile)) || error("missing $cfile — run gpu/validation/run_hiorder3_ma100_gpu.jl once")
Ma = parse(Float64, split(strip(read(mfile, String)), '\n')[1])
cross = reshape(collect(reinterpret(Float64, read(cfile))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3 = max(40.0, 4.0 + abs(Ma) / 2.0)

function crossing_interior(N::Int)
    M = zeros(35, N, N, N)
    Csize = floor(Int, 0.1 * N)
    Minb = div(N, 2) - Csize; Maxb = div(N, 2)
    Mnt  = div(N, 2) + 1;     Maxt = div(N, 2) + 1 + Csize
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        v = bg
        if Minb <= i <= Maxb && Minb <= j <= Maxb && Minb <= k <= Maxb; v = Mb; end
        if Mnt  <= i <= Maxt && Mnt  <= j <= Maxt && Mnt  <= k <= Maxt; v = Mt; end
        @views M[:, i, j, k] .= v
    end
    return M
end

N = 16; nstep = 6; dx = 1.0 / N
@assert N % nranks == 0 "need N divisible by nranks"
nzloc = div(N, nranks)
Mfull = crossing_interior(N)

# --- single-GPU order-3 reference (rank 0), on-device global CFL ---
usedref = Float64[]; Mref_host = zeros(0)
if rank == 0
    G = build_haloed_cube(CuArray(copy(Mfull)))
    usedref = march3d_order3_gpu!(G, dx, Ma, nstep; s3max=s3)
    Mi = CUDA.zeros(Float64, 35, N, N, N); interior_from_cube!(Mi, G)
    Mref_host = Array(Mi)
end

# --- 2-rank z-slab order-3 march, on-device global CFL (Allreduce(max) == single-GPU) ---
z0 = rank * nzloc
Mslab = CuArray(Array(@view Mfull[:, :, :, z0+1:z0+nzloc]))
used = Timestep3DGPU.march3d_slab_gpu!(Mslab, dx, Ma, nstep, comm; order=3, s3max=s3)

# --- gather final slabs to rank 0 (rank order == z order) ---
sb = vec(Array(Mslab)); counts = fill(35 * N * N * nzloc, nranks)
rbuf = rank == 0 ? Vector{Float64}(undef, 35 * N^3) : Float64[]
MPI.Gatherv!(sb, rank == 0 ? MPI.VBuffer(rbuf, counts) : nothing, comm; root=0)

# --- gather each rank's ACTUAL bound CUDA device (name + short UUID) to rank 0,
#     so the summary reports the true hardware whether run on 1 or 2 physical GPUs ---
const _DEVW = 80
mydev  = string("rank", rank, "→", CUDA.name(CUDA.device()), " ",
                first(string(CUDA.uuid(CUDA.device())), 8))
devbuf = fill(UInt8(' '), _DEVW)
let b = codeunits(mydev); devbuf[1:min(_DEVW, length(b))] .= @view b[1:min(_DEVW, length(b))]; end
alldev = rank == 0 ? Vector{UInt8}(undef, _DEVW * nranks) : UInt8[]
MPI.Gather!(devbuf, rank == 0 ? MPI.UBuffer(alldev, _DEVW) : nothing, comm; root=0)

if rank == 0
    Mmulti = reshape(rbuf, 35, N, N, N)
    df  = maximum(abs.(Mmulti .- Mref_host))
    ddt = maximum(abs.(used .- usedref))
    mass_ref   = sum(@view Mref_host[1, :, :, :])
    mass_multi = sum(@view Mmulti[1, :, :, :])
    dmass = abs(mass_multi - mass_ref)
    finite = all(isfinite, Mmulti)
    @printf("=== multi-GPU z-slab order-3 vs single-GPU (nranks=%d, N=%d, nzloc=%d, nstep=%d, Ma=%.0f) ===\n",
            nranks, N, nzloc, nstep, Ma)
    @printf("  dt-sequence   max abs diff = %.3e\n", ddt)
    @printf("  final field   max abs diff = %.3e\n", df)
    @printf("  final field   finite = %s   rho range [%.4e, %.4e]\n",
            finite, extrema(@view Mmulti[1, :, :, :])...)
    gate1 = (df <= 1e-10) && (ddt == 0.0) && finite
    @printf("  Gate 1 (consistency)  = %s (field ≤ 1e-10, dt exact)\n", gate1 ? "PASS" : "FAIL")
    @printf("  total mass: 1-rank = %.15e  2-rank = %.15e  |Δ| = %.3e\n", mass_ref, mass_multi, dmass)
    gate2 = dmass <= 1e-9 * max(abs(mass_ref), 1.0)
    @printf("  Gate 2 (conservation) = %s\n", gate2 ? "PASS" : "FAIL")
    ok = gate1 && gate2
    devs  = [strip(String(@view alldev[(r-1)*_DEVW+1 : r*_DEVW])) for r in 1:nranks]
    ndist = length(unique(split(d)[end] for d in devs))   # distinct UUID prefixes
    @printf("SUMMARY: %s  (%d rank(s): %s; %d distinct physical GPU%s)\n",
            ok ? "ALL PASS" : "FAIL", nranks, join(devs, ", "),
            ndist, ndist == 1 ? "" : "s")
    MPI.Barrier(comm); MPI.Finalize()
    ok || error("z-slab order-3 validation FAILED")
    println("All gates passed.")
else
    MPI.Barrier(comm); MPI.Finalize()
end

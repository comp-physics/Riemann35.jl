# validate_gpu_snapshots.jl — GPU run + snapshot dump in the canonical JLD2 schema.
# Runs a small GPU sim, streams snapshots, reopens the file and checks the schema the
# existing readers/viz expect (meta/n_snapshots, snapshots/NNNNNN/M as (Nx,Ny,Nz,35)).
# Works for -n 1 (single-GPU) and -n>1 (multi-GPU z-slab, gathered on rank 0).
#   srun --mpi=pmix -n 1 --gpus=1 julia --project=gpu/gpuenv2 gpu/validate_gpu_snapshots.jl
#   srun --mpi=pmix -n 2 --gpus=2 julia --project=gpu/gpuenv2 gpu/validate_gpu_snapshots.jl
# Needs proj_M.f64 in $RIEMANN35_DATA (default <repo>/data) for a realizable IC.
using CUDA, MPI, JLD2, Printf
include(joinpath(joinpath(@__DIR__, ".."), "gpu_run.jl")); using .GPURun
DATA = get(ENV, "RIEMANN35_DATA", joinpath(joinpath(@__DIR__, ".."), "..", "data"))

MPI.Init(); comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
CUDA.device!(rank % CUDA.ndevices())

n = 16; nz = 8; dx = 1.0/n; Ma = 2.0; nstep = 6; snap_int = 2
@assert nz % nranks == 0; nzloc = div(nz, nranks)

# realizable global IC (35,n,n,nz) from tiled proj_M states (identical on all ranks)
nb = parse(Int, strip(read(joinpath(DATA, "proj.meta"), String)))
src = reshape(reinterpret(Float64, read(joinpath(DATA, "proj_M.f64"))), 35, nb)
idx = round.(Int, range(1, nb, length=n*n*nz))
Mglob = Array{Float64}(undef, 35, n, n, nz)
for c in 1:n*n*nz; Mglob[:, (c-1)%n+1, ((c-1)÷n)%n+1, (c-1)÷(n*n)+1] .= src[:, idx[c]]; end

# each rank's slab interior (z-decomposition); single-rank => whole field
z0 = rank*nzloc
M0 = Array(@view Mglob[:, :, :, z0+1:z0+nzloc])

fname = joinpath(DATA, "gpu_snapshots_test_$(nranks)rank.jld2")
out = run_gpu_3d(M0, dx, Ma, nstep;
                 snapshot_interval=snap_int, snapshot_filename=fname,
                 comm=(nranks > 1 ? comm : nothing),
                 params=Dict("Nx"=>n, "Ny"=>n, "Nz"=>nz, "Ma"=>Ma, "dx"=>dx, "source"=>"gpu"))

if rank == 0
    jf = jldopen(out, "r")
    ns = jf["meta/n_snapshots"]; si = jf["meta/snapshot_interval"]
    key1 = lpad(1, 6, '0'); M1 = jf["snapshots/$key1/M"]
    keylast = lpad(ns, 6, '0'); tlast = jf["snapshots/$keylast/t"]; steplast = jf["snapshots/$keylast/step"]
    finite = all(isfinite, M1)
    close(jf)
    exp_ns = 1 + cld(nstep, snap_int)   # initial + one per interval
    ok = (size(M1) == (n, n, nz, 35)) && (ns == exp_ns) && finite && (steplast == nstep)
    @printf("GPU snapshot dump (%d rank(s)) -> %s\n", nranks, basename(out))
    @printf("  n_snapshots=%d (expected %d)  interval=%d\n", ns, exp_ns, si)
    @printf("  snapshot M shape=%s (expected (%d,%d,%d,35))  finite=%s\n", size(M1), n, n, nz, finite)
    @printf("  last snapshot: step=%d (expected %d)  t=%.4g\n", steplast, nstep, tlast)
    println(ok ? "GPU SNAPSHOT DUMP PASS (schema matches simulation_runner)" : "FAIL")
end
MPI.Barrier(comm); MPI.Finalize()

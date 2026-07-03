# Rodney 2D on GPU, stage 1/2 (main package env, single rank): extract the IC
# and dt sequence from the SAME case definition the CPU driver uses
# (rodney2d_setup.jl; the runner's :bubble branch builds the field at tmax=0)
# and stage them as raw f64 for the gpuenv2 march (rodney_validation_2d_gpu.jl).
# No case setup is duplicated — this script only moves bytes between the two
# Julia environments.
#
#   RODNEY_NP=512 julia --project=. examples/rodney_validation_2d_gpu_prep.jl
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
include(joinpath(@__DIR__, "rodney2d_setup.jl"))
@assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "prep runs single-rank"

k = rodney2d_knobs()
p = rodney2d_params(; k..., tmax = 0.0)   # zero steps -> the gathered IC
M, _, _, _ = simulation_runner(p)         # (Nx,Ny,Nz,35)
M0 = permutedims(M, (4, 1, 2, 3))         # (35,nx,ny,nz) device layout

# Rodney's dt cap: constant sequence with a trimmed final step (his own code
# takes dt = min(CFL dt, dtmax); the GPU driver rebuilds the sequence if the
# cap turns out not to be binding at fine resolution).
dt = rodney2d_dtmax(k.Kn)
dts = rodney2d_dts(k.tmax, dt)

dir = get(ENV, "RODNEY_GPU_DATA", "output/rodney2d_gpu")
mkpath(dir)
rodney2d_write_meta(joinpath(dir, "meta.txt");
    Np = k.Np, nz = size(M0, 4), nstep = length(dts),
    dx = 1.0 / k.Np,                                 # domain is [-0.5,0.5]
    Ma = k.Ma, Kn = k.Kn, tmax = k.tmax,
    snap_interval = rodney2d_snapshot_interval(; k...),
    tag = rodney2d_tag(; k...))
write(joinpath(dir, "M0.f64"), M0)
write(joinpath(dir, "dts.f64"), dts)
println("prep done: $dir  (Np=$(k.Np), nstep=$(length(dts)), dt=$dt)")

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

# Rodney's dt cap is binding for this case, so the dt sequence is constant with
# a trimmed final step (his own code takes dt = min(CFL dt, dtmax) and lands on
# dtmax too). The GPU driver asserts the cap really does sit below the CFL dt.
dt = rodney2d_dtmax(k.Kn)
nstep = ceil(Int, k.tmax / dt)
dts = fill(dt, nstep)
dts[end] = k.tmax - (nstep - 1) * dt

dir = get(ENV, "RODNEY_GPU_DATA", "output/rodney2d_gpu")
mkpath(dir)
open(joinpath(dir, "meta.txt"), "w") do io
    println(io, k.Np)
    println(io, size(M0, 4))
    println(io, nstep)
    println(io, 1.0 / k.Np)                          # dx (domain is [-0.5,0.5])
    println(io, k.Ma)
    println(io, k.Kn)
    println(io, k.tmax)
    println(io, rodney2d_snapshot_interval(; k...))
    println(io, rodney2d_tag(; k...))
end
write(joinpath(dir, "M0.f64"), M0)
write(joinpath(dir, "dts.f64"), dts)
println("prep done: $dir  (Np=$(k.Np), nstep=$nstep, dt=$dt)")

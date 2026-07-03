# run_case.jl — device-agnostic driver: run any case file (examples/cases/*.jl)
# on the CPU (default, serial or MPI) or on the GPU (--gpu).
#
# Usage:
#   julia --project=. examples/run_case.jl examples/cases/riemann1d.jl
#   CASE_NP=512 mpiexec -n 4 julia --project=. examples/run_case.jl examples/cases/bubble2d.jl
#   CASE_NP=512 julia --project=. examples/run_case.jl examples/cases/bubble2d.jl --gpu
# Output: output/runs/<tag>[_gpu].jld2 + browseable bundle in output/viz/ (./serve.sh).
#
# --gpu stages the runner-built IC (gpu/staging_common.jl) and spawns the march
# in the CUDA project (gpu/run_staged.jl, gpuenv2) as a subprocess — one
# command, same case file, no duplicated setup. Requires a single rank and a
# visible GPU (see docs for the CUDA env recipe on HPC nodes).
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
include(joinpath(@__DIR__, "..", "gpu", "staging_common.jl"))

args  = filter(a -> !startswith(a, "--"), ARGS)
gpu   = "--gpu" in ARGS
isempty(args) && error("usage: run_case.jl <case file> [--gpu]  (see examples/cases/)")
include(abspath(args[1]))
c = case()
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0
rank0 && mkpath("output/runs")

if gpu
    MPI.Comm_size(MPI.COMM_WORLD) == 1 ||
        error("--gpu stages single-rank and marches on one GPU; run without mpiexec")
    dir = stage_case(simulation_runner, c)
    gpuenv = joinpath(@__DIR__, "..", "gpu", "gpuenv2")
    script = joinpath(@__DIR__, "..", "gpu", "run_staged.jl")
    run(`$(Base.julia_cmd()) --project=$gpuenv $script $dir`)
    println("done: $(c.tag) → output/runs/$(c.tag)_gpu.jld2 (browse: output/viz/serve.sh)")
else
    params = (; c.params...,
        snapshot_interval = c.snap_interval,
        snapshot_filename = "output/runs/$(c.tag).jld2",
        web_dir = "output")
    simulation_runner(params)
    rank0 && println("done: $(c.tag) → output/runs/$(c.tag).jld2 (browse: output/viz/serve.sh)")
end

# stage_case.jl — thin CLI over staging_common.stage_case: stage a case on a
# machine that will NOT run the march (e.g. stage on a login node, march via a
# batch job with gpu/run_staged.jl). For the one-command path on a GPU node use
#   julia --project=. examples/run_case.jl <case file> --gpu
#
# Usage (main package env, single rank):
#   julia --project=. gpu/stage_case.jl examples/cases/bubble2d.jl [stage dir]
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
include(joinpath(@__DIR__, "staging_common.jl"))

isempty(ARGS) && error("usage: stage_case.jl <case file> [stage dir]")
include(abspath(ARGS[1]))
c = case()
@assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "staging runs single-rank"
dir = length(ARGS) >= 2 ? stage_case(simulation_runner, c; dir = ARGS[2]) :
                          stage_case(simulation_runner, c)
println("staged: $dir  (tag $(c.tag); march: julia --project=gpu/gpuenv2 gpu/run_staged.jl $dir)")

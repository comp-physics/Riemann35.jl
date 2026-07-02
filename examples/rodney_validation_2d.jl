# Rodney Fox's 2D uniform-pressure dense-bubble case — CPU/MPI driver.
# Case definition lives in rodney2d_setup.jl (single source, shared with the
# GPU driver pair). Quasi-2D flow past an effectively rigid cold cylinder with
# heat transfer; copy BCs act as crude in/outflow — keep tmax small enough that
# disturbances stay interior.
#
# Usage:
#   julia --project=. examples/rodney_validation_2d.jl
#   RODNEY_NP=512 RODNEY_MA=1.0 RODNEY_KN=0.001 mpiexec -n 4 julia --project=. examples/rodney_validation_2d.jl
# Output: output/runs/<tag>.jld2 + browseable bundle in output/viz/ (./serve.sh).
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
include(joinpath(@__DIR__, "rodney2d_setup.jl"))

k = rodney2d_knobs()
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0
rank0 && mkpath("output/runs")
tag = rodney2d_tag(; k...)

params = rodney2d_params(; k...,
    snapshot_interval = rodney2d_snapshot_interval(; k...),
    snapshot_filename = "output/runs/$tag.jld2",
    web_dir = "output",
)
result = simulation_runner(params)
rank0 && println("done: $tag → output/runs/$tag.jld2 (browse: output/viz/serve.sh)")

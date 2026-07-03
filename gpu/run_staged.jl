# run_staged.jl — march a staged case (gpu/stage_case.jl) on the GPU with
# `run_gpu_3d` and the recommended scheme, writing the same JLD2 snapshot
# schema as the CPU runner plus the web-viewer bundle. vacuum_floor=0 matches
# the CPU runner's ho_vacuum_floor default (the GPU-side default is 0.001).
#
# dt policy: constant sequence at min(dtcap, 0.9 x probed CFL dt) — the same
# dt = min(CFL dt, dtmax) rule the CPU runner applies each step; the probe is
# one adaptive step on a scratch copy of the IC, and the 0.9 absorbs transient
# wave-speed growth the IC-time probe can't see.
#
# Usage (gpuenv2):
#   julia --project=gpu/gpuenv2 gpu/run_staged.jl <stage dir>
using CUDA, Printf
include(joinpath(@__DIR__, "gpu_run.jl"))
using .GPURun
include(joinpath(@__DIR__, "staging_common.jl"))

isempty(ARGS) && error("usage: run_staged.jl <stage dir>  (from gpu/stage_case.jl)")
dir = ARGS[1]
m = read_stage_meta(joinpath(dir, "meta.txt"))
nx = parse(Int, m["nx"]); ny = parse(Int, m["ny"]); nz = parse(Int, m["nz"])
dx = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"]); Kn = parse(Float64, m["Kn"])
tmax = parse(Float64, m["tmax"]); dtcap = parse(Float64, m["dtcap"])
snap_int = parse(Int, m["snap_interval"]); tag = m["tag"]

M0 = reshape(collect(reinterpret(Float64, read(joinpath(dir, "M0.f64")))), 35, nx, ny, nz)

probe = GPURun.Timestep3DGPU.march3d_gpu!(CuArray(M0), dx, Ma, 1; order = 2, vacuum_floor = 0.0)
dt = min(dtcap, 0.9 * probe[1])
dts = constant_dts(tmax, dt)
@printf("CFL dt = %.3e, dtcap = %.3e -> dt = %.3e (%d steps)  [%s]\n",
        probe[1], dtcap, dt, length(dts), CUDA.name(CUDA.device()))

mkpath("output/runs")
out = "output/runs/$(tag)_gpu.jld2"
t0 = time()
run_gpu_3d(M0, dx, Ma, length(dts);
    snapshot_interval = snap_int, snapshot_filename = out,
    dts = dts, Kn = Kn, scheme = :recommended, order = 2, vacuum_floor = 0.0,
    params = Dict{String,Any}("case" => tag, "Ma" => Ma, "Kn" => Kn, "tmax" => tmax,
                              "scheme" => "recommended",
                              "device" => CUDA.name(CUDA.device())),
    web_dir = "output")
el = time() - t0
@printf("done: %s  (%d steps, %.1f s wall, %.3f s/step)\n", out, length(dts), el, el / length(dts))

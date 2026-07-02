# Rodney 2D on GPU, stage 2/2 (gpuenv2): march the exact IC bytes staged by
# rodney_validation_2d_gpu_prep.jl with run_gpu_3d (scheme=:recommended =
# pressure recon + stage BGK), writing the same JLD2 snapshot schema as the CPU
# driver plus the web-viewer bundle. vacuum_floor=0 matches the CPU runner's
# ho_vacuum_floor default (the GPU-side default is 0.001).
#
#   julia --project=gpu/gpuenv2 examples/rodney_validation_2d_gpu.jl
using CUDA, Printf
include(joinpath(@__DIR__, "..", "gpu", "gpu_run.jl"))
using .GPURun

dir = get(ENV, "RODNEY_GPU_DATA", "output/rodney2d_gpu")
meta  = readlines(joinpath(dir, "meta.txt"))
Np    = parse(Int, meta[1]);     nz = parse(Int, meta[2]); nstep = parse(Int, meta[3])
dx    = parse(Float64, meta[4]); Ma = parse(Float64, meta[5]); Kn = parse(Float64, meta[6])
tmax  = parse(Float64, meta[7]); snap_int = parse(Int, meta[8]); tag = meta[9]

rd(f) = collect(reinterpret(Float64, read(joinpath(dir, f))))
M0  = reshape(rd("M0.f64"), 35, Np, Np, nz)
dts = rd("dts.f64")
@assert length(dts) == nstep

# CFL probe: one adaptive step on a scratch copy returns the local-CFL dt; the
# fixed collision-cap dt must sit below it or the fixed sequence is invalid.
probe = GPURun.Timestep3DGPU.march3d_gpu!(CuArray(M0), dx, Ma, 1; order = 2, vacuum_floor = 0.0)
@printf("CFL dt = %.3e, fixed dt = %.3e (margin %.1fx)  [%s]\n",
        probe[1], dts[1], probe[1] / dts[1], CUDA.name(CUDA.device()))
@assert dts[1] <= probe[1] "fixed dt violates CFL"

mkpath("output/runs")
out = "output/runs/$(tag)_gpu.jld2"
t0 = time()
run_gpu_3d(M0, dx, Ma, nstep;
    snapshot_interval = snap_int, snapshot_filename = out,
    dts = dts, Kn = Kn, scheme = :recommended, order = 2, vacuum_floor = 0.0,
    params = Dict{String,Any}("case" => "rodney2d", "Np" => Np, "Ma" => Ma, "Kn" => Kn,
                              "tmax" => tmax, "scheme" => "recommended",
                              "device" => CUDA.name(CUDA.device())),
    web_dir = "output")
el = time() - t0
@printf("done: %s  (%d steps, %.1f s wall, %.3f s/step)\n", out, nstep, el, el / nstep)

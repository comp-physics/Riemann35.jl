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
include(joinpath(@__DIR__, "rodney2d_setup.jl"))   # shared meta schema + dt helper (pure Base)

dir = get(ENV, "RODNEY_GPU_DATA", "output/rodney2d_gpu")
m     = rodney2d_read_meta(joinpath(dir, "meta.txt"))
Np    = parse(Int, m["Np"]);     nz = parse(Int, m["nz"]); nstep = parse(Int, m["nstep"])
dx    = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"]); Kn = parse(Float64, m["Kn"])
tmax  = parse(Float64, m["tmax"]); snap_int = parse(Int, m["snap_interval"]); tag = m["tag"]

rd(f) = collect(reinterpret(Float64, read(joinpath(dir, f))))
M0  = reshape(rd("M0.f64"), 35, Np, Np, nz)
dts = rd("dts.f64")
@assert length(dts) == nstep

# CFL probe: one adaptive step on a scratch copy returns the local-CFL dt.
# At coarse grids Rodney's collision cap is binding; at fine grids (~1024^2+)
# the CFL dt drops below the cap — rebuild the constant sequence at 0.9x the
# probed CFL dt (his own stepping is dt = min(CFL dt, dtmax) too; the 0.9
# absorbs transient wave-speed growth the IC-time probe can't see).
probe = GPURun.Timestep3DGPU.march3d_gpu!(CuArray(M0), dx, Ma, 1; order = 2, vacuum_floor = 0.0)
@printf("CFL dt = %.3e, staged dt = %.3e (margin %.2fx)  [%s]\n",
        probe[1], dts[1], probe[1] / dts[1], CUDA.name(CUDA.device()))
if dts[1] > 0.9 * probe[1]
    dts = rodney2d_dts(tmax, 0.9 * probe[1])
    nstep = length(dts)
    @printf("dt CFL-limited: rebuilt sequence at dt = %.3e (%d steps)\n", dts[1], nstep)
end
@assert dts[1] <= probe[1] "staged dt violates CFL"

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

# profile_order3.jl — CUDA.@profile the order-3 crossflow+obstacle march to see
# the per-kernel time breakdown (residual vs projection vs BGK vs refill vs recon).
# Answers "where does the time actually go" instead of guessing.
#
# Usage (gpuenv2, cuda module + project depot):
#   julia --project=gpu/gpuenv2 gpu/profile_order3.jl <stage_dir> [nsteps=20]
using CUDA, Printf
include(joinpath(@__DIR__, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
include(joinpath(@__DIR__, "staging_common.jl"))

dir = ARGS[1]; nstep = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
m = read_stage_meta(joinpath(dir, "meta.txt"))
nx = parse(Int, m["nx"]); ny = parse(Int, m["ny"]); nz = parse(Int, m["nz"])
dx = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"]); Kn = parse(Float64, m["Kn"])
s3max = haskey(m, "s3max") ? parse(Float64, m["s3max"]) : max(40.0, 4.0 + abs(Ma)/2.0)
M0 = reshape(collect(reinterpret(Float64, read(joinpath(dir, "M0.f64")))), 35, nx, ny, nz)
inlet = M0[:, 1, 1, 1]
ost = nothing; ocx = 0.0; ocy = 0.0; or2 = 0.0
if haskey(m, "obst_cx")
    ocx = parse(Float64, m["obst_cx"]); ocy = parse(Float64, m["obst_cy"])
    rc = parse(Float64, m["obst_r_cells"]); or2 = rc*rc
    ost = M0[:, clamp(round(Int,ocx),1,nx), clamp(round(Int,ocy),1,ny), 1]
end
@printf("grid %dx%dx%d  Kn=%.4g  device=%s\n", nx, ny, nz, Kn, CUDA.name(CUDA.device()))

G = build_haloed_cube(CuArray(M0))
mk() = march3d_order3_gpu!(G, dx, Ma, 1; s3max=s3max, stage_bgk=true, Kn=Kn,
        bc=:crossflow, inlet=inlet, obst_state=ost, obst_cx=ocx, obst_cy=ocy, obst_r2=or2)
mk(); CUDA.synchronize()                       # warmup / compile
# wall-clock per step
t0 = time(); for _ in 1:nstep; mk(); end; CUDA.synchronize()
@printf("wall: %.4f s/step over %d steps\n", (time()-t0)/nstep, nstep)
# device-side per-kernel breakdown
CUDA.@profile march3d_order3_gpu!(G, dx, Ma, nstep; s3max=s3max, stage_bgk=true, Kn=Kn,
        bc=:crossflow, inlet=inlet, obst_state=ost, obst_cx=ocx, obst_cy=ocy, obst_r2=or2)

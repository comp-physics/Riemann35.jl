# validate_rank_boundary_residual.jl — exercise the RANK-BOUNDARY path of the order-3
# residual WITHOUT MPI, and dump the residual for bit-comparison across a code change.
#
# WHY THIS EXISTS. The single-GPU march passes rank_bnd = all-false, so
# `_rank_face_theta` -- the function whose @inline -> @noinline change cut compile time
# 12x -- is COMPILED BUT NEVER EXECUTED. A byte-identical single-GPU field therefore
# says nothing about that function. The repo's MPI validator
# (gpu/validation/validate_slab_order3_gpu.jl) does cover it, but its input
# data/r3d_cross_ma100.f64 is absent from the checkout, so it cannot be run here.
#
# `rank_bnd` is a plain kwarg on residual3d_order3_box_gpu!, so the path can be driven
# directly with no MPI at all. This sets ALL SIX faces true -- broader coverage than the
# MPI validator, which is z-slab only and so leaves the x/y branches dormant.
#
# Usage: julia [-g0] --project=gpu/gpuenv2 \
#          gpu/validation/validate_rank_boundary_residual.jl <stage_dir> <out.f64>
# Stage a case first with gpu/stage_case.jl. Set R35_GPUDIR to point at another
# checkout's gpu/ dir to produce the "before" side from a git worktree.
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", normpath(joinpath(@__DIR__, "..")))
include(joinpath(GPUDIR, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
using .Timestep3DOrder3GPU.Residual3DOrder3GPU: residual3d_order3_box_gpu!
include(joinpath(GPUDIR, "staging_common.jl"))

dir = ARGS[1]; outf = ARGS[2]
m = read_stage_meta(joinpath(dir, "meta.txt"))
nx = parse(Int, m["nx"]); ny = parse(Int, m["ny"]); nz = parse(Int, m["nz"])
dx = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"])
s3max = haskey(m, "s3max") ? parse(Float64, m["s3max"]) : max(40.0, 4.0 + abs(Ma)/2.0)
M0 = reshape(collect(reinterpret(Float64, read(joinpath(dir, "M0.f64")))), 35, nx, ny, nz)

G = build_haloed_cube(CuArray(M0))
g = (size(G,2) - nx) ÷ 2
R = CUDA.zeros(Float64, 35, nx, ny, nz)
@printf("debug_level=%d  grid %dx%dx%d  halo g=%d  dev=%s\n", Base.JLOptions().debug_level,
        nx, ny, nz, g, CUDA.name(CUDA.device())); flush(stdout)

# dt must be NONZERO: _halo_cell_mlo short-circuits to the raw cell state when all the
# lambdas are zero, which would skip the six _hll_states calls this test exists to cover.
dt = 0.2*dx/(5.0*sqrt(1.0))
RB = (xlo=true, xhi=true, ylo=true, yhi=true, zlo=true, zhi=true)
residual3d_order3_box_gpu!(R, G, nx, ny, nz, g, dx, dx, dx, Ma, dt;
                           s3max=s3max, rank_bnd=RB)
CUDA.synchronize()
H = Array(R)
write(outf, reinterpret(UInt8, vec(H)))
@printf("ALL SIX rank flags true, dt=%.6e  ->  %s\n", dt, outf)
@printf("  sum=%.17e  min=%.17e  max=%.17e  nonfinite=%d\n",
        sum(H), minimum(H), maximum(H), count(!isfinite, H)); flush(stdout)

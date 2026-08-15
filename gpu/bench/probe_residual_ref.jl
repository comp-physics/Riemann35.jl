#!/usr/bin/env julia
# Compute the order-3 residual on a fixed state, write it to a file, and time it.
#
# WHY A FILE. The unrolling change (@nexprs in _store35! and _blend_residual!) has to be proved
# bit-identical, but it changes the ONLY build in the tree -- there is no second code path to
# compare against inside one run, the way probe_split_x.jl could compare fused against split. So
# the reference is written to disk from one build and compared against the other:
#
#   git stash && julia ... probe_residual_ref.jl before.bin && git stash pop
#   julia ... probe_residual_ref.jl after.bin
#   cmp before.bin after.bin        # must be identical, byte for byte
#
# The state is deterministic (no RNG) and the residual is a pure function of it, so any
# difference is the change and nothing else.
#
# Usage: julia --project=gpu/gpuenv2 gpu/bench/probe_residual_ref.jl <out.bin> [N]
using CUDA, Printf
using Riemann35: InitializeM4_35

include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl"))
using .Residual3DOrder3GPU: residual3d_order3_box_gpu!

const OUT = length(ARGS) >= 1 ? ARGS[1] : "residual_ref.bin"
const N   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 48
const G_HALO = 4

function build_state(nx, ny, nz, g)
    nfx, nfy, nfz = nx + 2g, ny + 2g, nz + 2g
    A = zeros(Float64, 35, nfx, nfy, nfz)
    for k in 1:nfz, j in 1:nfy, i in 1:nfx
        ph = 2pi * ((i - 0.5) / nfx); th = 2pi * ((j - 0.5) / nfy)
        A[:, i, j, k] .= collect(Float64,
            InitializeM4_35(1.0 * (1 + 0.2sin(ph) * cos(th)), 0.2cos(ph), 0.05sin(th), 0.0,
                            1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
    end
    CuArray(A)
end

function main()
    nx = ny = nz = N; g = G_HALO
    dx = 1.0 / nx; dt = 0.2 * dx / 5.0
    G = build_state(nx, ny, nz, g)
    R = CUDA.zeros(Float64, 35, nx, ny, nz)

    call() = residual3d_order3_box_gpu!(R, G, nx, ny, nz, g, dx, dx, dx, 1.0, dt; s3max = 40.0)

    call(); CUDA.synchronize()                      # warm: first call pays JIT
    open(OUT, "w") do fh; write(fh, Array(R)); end

    reps = 40
    CUDA.synchronize(); t0 = time()
    for _ in 1:reps; call(); end
    CUDA.synchronize()
    ms = 1e3 * (time() - t0) / reps

    # x axis alone, the kernel the last two attempts targeted
    axcall() = residual3d_order3_box_gpu!(R, G, nx, ny, nz, g, dx, dx, dx, 1.0, dt;
                                          s3max = 40.0, active = (true, false, false))
    axcall(); CUDA.synchronize(); t1 = time()
    for _ in 1:reps; axcall(); end
    CUDA.synchronize()
    msx = 1e3 * (time() - t1) / reps

    @printf("%s at %d^3, %s\n", OUT, N, CUDA.name(CUDA.device()))
    @printf("residual %.4f ms   x-axis only %.4f ms\n", ms, msx)
    @printf("wrote %d bytes\n", filesize(OUT))
    return 0
end

exit(main())

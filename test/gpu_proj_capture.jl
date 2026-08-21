#!/usr/bin/env julia
# Controls for the GPU realizability-projection CAPTURE buffer.
#
# The capture exists to answer a question the bare counter cannot: which stage of the correction
# did the work, and which principal minor of the 6x6 delta2star was negative on entry. It records
# the projection's INPUT state so that classification can happen on the host, against the shipped
# routine rather than against a device-side copy of its logic that could drift.
#
# Three things have to hold, and each has already failed once somewhere in this project:
#   1. capture must be INERT -- a captured march reproduces an uncaptured one bit-for-bit;
#   2. what it captures must be the PRE-projection state, not the corrected one. That is the
#      whole point and it is easy to get backwards, because the natural place to write is the
#      loop that stores P. A captured state that is already realizable means the write happened
#      after the overwrite, and the diagnostic would report "nothing was wrong" on every cell;
#   3. the number captured must agree with the counter on the same run.
#
# Run: julia --project=<gpuenv> test/gpu_proj_capture.jl
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", joinpath(@__DIR__, "..", "gpu"))
include(joinpath(GPUDIR, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
using Riemann35

const N = 8
const DX = 1.0 / N
const MA = 1.0

function maxwellian_cube()
    M = zeros(Float64, 35, N, N, N)
    for k in 1:N, j in 1:N, i in 1:N
        M[:, i, j, k] .= Riemann35.InitializeM4_35(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    end
    M
end

fails = String[]

# an unrealizable field: S220 driven far past its bound on half the cells
Mbad = maxwellian_cube()
for k in 1:N, j in 1:N, i in 1:2:N
    Mbad[12, i, j, k] *= 5.0
end

# ---- capture alongside the counter -----------------------------------------------------------
G = build_haloed_cube(CuArray(Mbad))
c = gpu_proj_counter()
buf, idx = gpu_proj_capture(4096)
march3d_order3_gpu!(G, DX, MA, 2; dts = fill(1e-4, 2), s3max = 40.0, stage_bgk = false,
                    proj_counter = c, proj_capture = buf, proj_capture_idx = idx)
nfire = gpu_proj_count(c)
states, nattempt = gpu_proj_captured(buf, idx)

nfire > 0 || push!(fails, "positive control produced 0 firings; nothing to capture")
nattempt == nfire || push!(fails, "capture attempted $nattempt writes against $nfire firings; " *
                                  "the two diagnostics disagree on the same run")

# ---- 2: what came back must be the PRE-projection state --------------------------------------
# Every captured state must differ from its own correction. If capture ran after the overwrite,
# each row would already be a fixed point and this fraction would be zero.
nchanged = 0
for col in 1:size(states, 2)
    global nchanged
    v = ntuple(m -> states[m, col], 35)
    p = Riemann35.realizable_3D_M4(collect(v), MA, 40.0)
    if any(abs(p[m] - v[m]) > 1e-10 * max(abs(v[m]), abs(p[m])) + 1e-13 for m in 1:35)
        nchanged += 1
    end
end
nchanged == size(states, 2) ||
    push!(fails, "$(size(states,2) - nchanged) of $(size(states,2)) captured states are already " *
                 "fixed points of the correction; capture is recording the OUTPUT, not the input")

# ---- 1: capture must be inert ----------------------------------------------------------------
G1 = build_haloed_cube(CuArray(Mbad))
march3d_order3_gpu!(G1, DX, MA, 3; dts = fill(1e-4, 3), s3max = 40.0)
G2 = build_haloed_cube(CuArray(Mbad))
b2, i2 = gpu_proj_capture(64)
march3d_order3_gpu!(G2, DX, MA, 3; dts = fill(1e-4, 3), s3max = 40.0,
                    proj_capture = b2, proj_capture_idx = i2)
inert = Array(G1) == Array(G2)
inert || push!(fails, "capturing changed the solution; the diagnostic is not inert")

# ---- truncation is reported, not hidden ------------------------------------------------------
_, n2 = gpu_proj_captured(b2, i2)
n2 > 64 || push!(fails, "the small buffer did not report truncation; a prefix that looks " *
                        "complete is worse than one that says it is a prefix")

println("="^80)
println("GPU REALIZABILITY-PROJECTION CAPTURE -- CONTROLS")
println("="^80)
@printf("  firings / capture attempts                 : %d / %d   %s\n", nfire, nattempt,
        nfire == nattempt ? "OK" : "FAIL -- diagnostics disagree")
@printf("  captured states are PRE-projection          : %d/%d   %s\n", nchanged, size(states,2),
        nchanged == size(states,2) ? "OK" : "FAIL -- capturing the corrected state")
@printf("  captured march is bit-identical to plain   : %s\n", inert ? "OK" : "FAIL")
@printf("  truncation reported (%d attempts, 64 slots) : %s\n", n2, n2 > 64 ? "OK" : "FAIL")
if isempty(fails)
    println("\nall controls pass"); exit(0)
end
println("\nFAILURES:"); for f in fails; println("  ", f); end
exit(1)

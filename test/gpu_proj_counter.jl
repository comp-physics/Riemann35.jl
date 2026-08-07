#!/usr/bin/env julia
# Positive and negative controls for the GPU realizability-projection counter.
#
# WHY THIS FILE IS THE POINT OF THE FEATURE, NOT AN EXTRA. The counter it guards replaces one
# that silently reported zero on every GPU run, because the increment lived in the CPU
# `_project_interior!` while the GPU marcher runs its own device kernel. That failure is
# invisible from the outside: "the projection never fires" and "the counter is not wired up"
# produce identical output. The only way to tell them apart is to feed in a state that MUST fire
# and check that the count moves.
#
# Run: julia --project=<gpuenv> test/gpu_proj_counter.jl
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", joinpath(@__DIR__, "..", "gpu"))
include(joinpath(GPUDIR, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
using Riemann35

const N = 8
const DX = 1.0 / N
const MA = 1.0

"Maxwellian moments at rest, rho = T = 1: the realizable reference state."
function maxwellian_cube()
    M = zeros(Float64, 35, N, N, N)
    for k in 1:N, j in 1:N, i in 1:N
        M[:, i, j, k] .= Riemann35.InitializeM4_35(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    end
    M
end

function count_for(M0; nstep = 2)
    G = build_haloed_cube(CuArray(M0))
    c = gpu_proj_counter()
    march3d_order3_gpu!(G, DX, MA, nstep; dts = fill(1e-4, nstep), s3max = 40.0,
                        stage_bgk = false, proj_counter = c)
    gpu_proj_count(c)
end

fails = String[]

# ---- NEGATIVE CONTROL 1: a uniform Maxwellian at rest -----------------------------------------
M = maxwellian_cube()
n_eq = count_for(M)
n_eq == 0 || push!(fails, "equilibrium fired $n_eq times; a uniform Maxwellian is realizable")

# ---- NEGATIVE CONTROL 2: a NON-TRIVIAL realizable state, and this is the one that matters -----
# A Maxwellian at rest is the degenerate case where the projection's round-trip through central
# and standardised moments happens to be exact in floating point, so it passes even under a
# broken exact-equality criterion -- which is how the first version of this counter shipped
# reporting 1.000 firings per (cell,stage) on Taylor-Green. A sheared, moving, anisotropic but
# still realizable field exercises the round-trip properly and must also give zero.
Msm = maxwellian_cube()
for k in 1:N, j in 1:N, i in 1:N
    x, y, z = (i-0.5)/N, (j-0.5)/N, (k-0.5)/N
    Msm[:, i, j, k] .= Riemann35.InitializeM4_35(
        1.0 + 0.03*sin(2pi*x), 0.30*sin(2pi*x)*cos(2pi*y), -0.30*cos(2pi*x)*sin(2pi*y), 0.0,
        1.0 + 0.02*cos(2pi*y), 0.01*sin(2pi*z), -0.02*sin(2pi*x),
        1.0 - 0.02*cos(2pi*y), 0.008*cos(2pi*z), 1.0 + 0.01*sin(2pi*z))
end
n_sm = count_for(Msm)
n_sm == 0 || push!(fails, "smooth realizable field fired $n_sm times; round-trip noise is " *
                          "being counted as a firing")

# ---- POSITIVE CONTROL: a deliberately unrealizable state MUST fire ---------------------------
# S220 driven far above its bound, the same construction the CPU-side control uses. If the
# counter is not reaching the device this stays 0 and the test fails, which is the whole point.
Mbad = maxwellian_cube()
for k in 1:N, j in 1:N, i in 1:2:N
    Mbad[12, i, j, k] *= 5.0        # M220: cross fourth-order, pushed out of the moment cone
end
n_bad = count_for(Mbad)
n_bad > 0 || push!(fails, "POSITIVE CONTROL FAILED: an unrealizable state produced 0 firings -- " *
                          "the counter is not reaching the device kernel")

# ---- the counter must not perturb the answer -------------------------------------------------
# Counting changes only whether an atomic is incremented, so a counted march must reproduce an
# uncounted one bit-for-bit. If it does not, the diagnostic is altering what it measures.
G1 = build_haloed_cube(CuArray(M)); march3d_order3_gpu!(G1, DX, MA, 3; dts = fill(1e-4, 3), s3max = 40.0)
G2 = build_haloed_cube(CuArray(M)); c2 = gpu_proj_counter()
march3d_order3_gpu!(G2, DX, MA, 3; dts = fill(1e-4, 3), s3max = 40.0, proj_counter = c2)
Array(G1) == Array(G2) || push!(fails, "counting changed the solution; the diagnostic is not inert")

# ---- report ----------------------------------------------------------------------------------
println("="^80)
println("GPU REALIZABILITY-PROJECTION COUNTER -- CONTROLS")
println("="^80)
@printf("  negative control 1 (Maxwellian at rest)    : %d firings   %s\n", n_eq,
        n_eq == 0 ? "OK" : "FAIL")
@printf("  negative control 2 (smooth anisotropic)     : %d firings   %s\n", n_sm,
        n_sm == 0 ? "OK" : "FAIL -- round-trip noise counted as firing")
@printf("  positive control (S220 x5, half the cells) : %d firings   %s\n", n_bad,
        n_bad > 0 ? "OK" : "FAIL -- COUNTER NOT WIRED TO THE DEVICE")
@printf("  counted march is bit-identical to uncounted: %s\n",
        Array(G1) == Array(G2) ? "OK" : "FAIL")
if isempty(fails)
    println("\nall controls pass")
    exit(0)
end
println("\nFAILURES:"); for f in fails; println("  ", f); end
exit(1)

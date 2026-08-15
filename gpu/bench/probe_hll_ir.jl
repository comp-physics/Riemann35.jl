#!/usr/bin/env julia
# Dump the LLVM IR for the fused x flux kernel and locate the spilled values.
#
# WHY IR RATHER THAN ANOTHER TIMED RUN. Two attempts have now been refuted by measuring
# wall-clock and register counts after the fact. Both told us where the spill is NOT. The IR
# says directly what is held in addressable memory and how big each object is, so it names the
# live set instead of bounding it.
#
# WHAT TO LOOK FOR. `alloca [N x double]` is a value the compiler could not keep in registers.
# A 105-element alloca would mean the full three-axis flux tuple from flux_closure35_dev is
# materialised even though _hll_states uses only 35 of it; a run of 35-element allocas points at
# the NTuple{35} intermediates instead.
#
# Usage: julia --project=gpu/gpuenv2 gpu/bench/probe_hll_ir.jl
using CUDA, Printf
using Riemann35: InitializeM4_35

include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl"))
using .Residual3DOrder3GPU

const R3 = Residual3DOrder3GPU

nx = ny = nz = 8; g = 4
nfx, nfy, nfz = nx+2g, ny+2g, nz+2g
A = zeros(Float64, 35, nfx, nfy, nfz)
for k in 1:nfz, j in 1:nfy, i in 1:nfx
    A[:, i, j, k] .= collect(Float64, InitializeM4_35(1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 1.0, 0, 1.0))
end
G = CuArray(A); V = CuArray(copy(A))
FHO = CUDA.zeros(Float64, 35, nx+1, ny, nz); FLO = CUDA.zeros(Float64, 35, nx+1, ny, nz)

io = IOBuffer()
CUDA.@device_code_llvm io = io debuginfo = :none begin
    @cuda threads=64 blocks=1 R3._weno_flux_x!(FHO, FLO, G, V, nx, ny, nz, g, nfx, 1.0, 40.0, false)
end
ir = String(take!(io))

open(joinpath(@__DIR__, "weno_flux_x.ll"), "w") do fh; write(fh, ir); end
@printf("IR: %d lines, %.1f KB -> gpu/bench/weno_flux_x.ll\n",
        count(==('\n'), ir), sizeof(ir)/1024)

allocas = Dict{String,Int}()
for m in eachmatch(r"alloca\s+\[(\d+)\s+x\s+([^\]]+)\]", ir)
    allocas["[$(m[1]) x $(m[2])]"] = get(allocas, "[$(m[1]) x $(m[2])]", 0) + 1
end
for m in eachmatch(r"alloca\s+(double|float|i\d+)\s*(?:,|$)", ir)
    allocas[String(m[1])] = get(allocas, String(m[1]), 0) + 1
end
function report(allocas)
    println("\nalloca inventory (count x shape), largest first:")
    tot = 0
    for (k, v) in sort(collect(allocas), by = x -> -x[2])
        sz = match(r"\[(\d+) x", k)
        n = sz === nothing ? 1 : parse(Int, sz[1])
        tot += n * v
        @printf("  %4d x %-20s = %6d doubles\n", v, k, n * v)
    end
    @printf("  %-27s   %6d doubles total in addressable memory\n", "TOTAL", tot)
end
report(allocas)
println("\n(255 registers = 127 doubles. Anything above that must spill.)")

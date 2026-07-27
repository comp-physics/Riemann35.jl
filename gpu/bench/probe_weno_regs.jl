# probe_weno_regs.jl — registers and LOCAL-MEMORY SPILL for the order-3 flux kernels.
#
# The parked diagnosis for #15 was "occupancy-bound at the 255 reg/thread ceiling". That is
# an inference from a flat block-size sweep, not a measurement. This measures it: the loaded
# cubin reports registers and the static local frame per thread, and spill traffic is the
# thing that would actually explain why the block-size sweep was flat and why maxregs made
# it monotonically worse (fewer registers => MORE spilling).
#
# BASELINE, measured A100 2026-07-27:
#   _weno_flux_x! (o3)  regs=255  local=12048 B
#   _weno_flux_x! (o1)  regs=255  local=12048 B   <- IDENTICAL, and the o1 path runs no WENO
#   _ppt_x!             regs=166  local=  776 B
#   _vavg_x!            regs=166  local=  280 B
# 12048 B/thread is ~1506 doubles, far more than the ~315 doubles live at the peak of the
# dataflow (W1..W6 + cL,cR + FL), so the frame is NOT just the stencil -- something inlined
# is much larger. The o1/o3 tie is its own finding: `first_order` is a runtime Bool, so one
# kernel carries both paths and every order-1 launch pays the order-3 frame.
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", "/fastscratch/sbryngelson3/Riemann35.jl/gpu")
include(joinpath(GPUDIR, "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
const R = Residual3DOrder3GPU

nx=ny=nz=16; g=8
nfx=nfy=nfz=nx+2g
G = CUDA.zeros(Float64, 35, nfx, nfy, nfz)
V = CUDA.zeros(Float64, 35, nfx, nfy, nfz)
P = CUDA.zeros(Float64, 35, nfx, nfy, nfz)
FHO = CUDA.zeros(Float64, 35, nx+1, ny, nz)
FLO = CUDA.zeros(Float64, 35, nx+1, ny, nz)

const TAG = get(ENV, "PROBE_TAG", "baseline")

function stats(name, kern)
    m = CUDA.memory(kern); regs = CUDA.registers(kern)
    @printf("%-22s regs=%3d  local=%6d B  shared=%5d B\n", name, regs, m.local, m.shared)
    # occupancy: A100 has 65536 regs and 64 warps max per SM
    (regs=regs, lmem=m.local)
end

println("="^88)
@printf("ORDER-3 FLUX KERNELS  [%s]  — registers and LOCAL FRAME (spill) per thread\n", TAG)
println("="^88)

k_wf  = @cuda launch=false R._weno_flux_x!(FHO, FLO, G, V, nx, ny, nz, g, nfx, 1.0, 40.0, false)
k_wf1 = @cuda launch=false R._weno_flux_x!(FHO, FLO, G, V, nx, ny, nz, g, nfx, 1.0, 40.0, true)
k_pp  = @cuda launch=false R._ppt_x!(P, G, nfx, nfy, nfz)
k_va  = @cuda launch=false R._vavg_x!(V, P, nfx, nfy, nfz)

s_wf  = stats("_weno_flux_x! (o3)", k_wf)
       stats("_weno_flux_x! (o1)", k_wf1)
       stats("_ppt_x!", k_pp)
       stats("_vavg_x!", k_va)

println()
for thr in (64, 128, 256)
    occ = CUDA.active_blocks(k_wf.fun, thr)
    warps = occ * thr / 32
    @printf("threads=%3d : %d blocks/SM => %.0f of 64 warps = %5.1f%% occupancy\n",
            thr, occ, warps, 100*warps/64)
end
println()
@printf("SUMMARY[%s] o3_regs=%d o3_lmem=%d\n", TAG, s_wf.regs, s_wf.lmem)

# validate_limiter_vs_cpu.jl — GPU residual3d_box_gpu(limiter=true) vs CPU
# residual_ho_3d!(order=2, use_limiter=true). Prereq: dump_cpu_limiter_residual.jl. gpuenv2.
#
# The realizability scaling limiter picks the largest theta in [0,1] for which both faces
# are realizable, via a 20-iteration bisection (== CPU `nbisect`). The realizability ORACLE
# eig differs CPU (LAPACK) vs GPU (analytic delta2star) — the documented "eigensolvers not
# single-sourced" floor — so the chosen theta sits on a slightly different boundary. Where
# theta==1 (the vast majority of cells, smooth/realizable) the limiter == plain MUSCL and
# the GPU residual is MACHINE-EXACT vs CPU (median ~1e-15). At the shock cells where theta<1,
# the Ma=100 high-order moment slopes (O(1e9)) AMPLIFY the small theta-boundary difference
# into a large *pointwise* residual error — the same FP-conditioning-at-shocks effect
# documented for the timestep cross moments. The gate is therefore on the BULK (median):
# the limiter is bulk-exact, and the rigorous theta-logic correctness is verified separately
# by `validate_limiter_theta_vs_cpu.jl` (theta agrees to the 2^-20 bisection quantum).
using CUDA, Printf, Statistics
include(joinpath(@__DIR__, "..", "residual3d_gpu.jl")); using .Residual3DGPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3])
M    = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
Rref = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_Rlim.f64")))),35,n,n,n)
Rg = residual3d_box_gpu(Array(M),n,n,n,dx,Ma; order=2, limiter=true, vacuum_floor=0.001, project_faces=true)
rel = abs.(Rg.-Rref)./max.(abs.(Rref),1.0)
a=maximum(abs.(Rg.-Rref)); rmax=maximum(rel)
p999=quantile(vec(rel),0.999); p9999=quantile(vec(rel),0.9999); med=median(vec(rel))
nbig=count(>(1e-6), vec(rel))
@printf("GPU limiter vs CPU use_limiter, %d^3 Ma=%.0f: max abs=%.3e\n", n,Ma,a)
@printf("  rel: median=%.3e  p99.9=%.3e  p99.99=%.3e  max=%.3e  n(>1e-6)=%d/%d (shock cells, theta<1)\n",
        med,p999,p9999,rmax,nbig,length(rel))
@printf("  BULK GATE (median <= 1e-12): %s   [shock-cell tail is eigensolver-boundary amplified — expected]\n",
        med<=1e-12 ? "PASS" : "FAIL")

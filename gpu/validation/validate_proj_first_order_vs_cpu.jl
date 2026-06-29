# validate_proj_first_order_vs_cpu.jl — GPU proj_first_order residual vs CPU
# residual_ho_3d!(order=2, use_proj_recon=true). Prereq: dump_cpu_proj_residual.jl. gpuenv2.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_gpu.jl")); using .Residual3DGPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3])
M    = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
Rref = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_Rproj.f64")))),35,n,n,n)
Rg = residual3d_box_gpu(Array(M),n,n,n,dx,Ma; order=2, proj_first_order=true, vacuum_floor=0.001, project_faces=true)
r=maximum(abs.(Rg.-Rref)./max.(abs.(Rref),1.0)); a=maximum(abs.(Rg.-Rref))
@printf("GPU proj_first_order vs CPU use_proj_recon, %d^3 Ma=%.0f: max abs=%.3e max rel=%.3e GATE=%s\n",
        n,Ma,a,r, r<=1e-6 ? "PASS" : "FAIL")

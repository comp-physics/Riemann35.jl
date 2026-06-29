# validate_rusanov_vs_cpu.jl — GPU residual3d_box_gpu(riemann_solver=:rusanov) vs CPU
# residual_ho_3d!(order=2) with RIEMANN_SOLVER[]=:rusanov. Prereq: dump_cpu_rusanov_residual.jl. gpuenv2.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_gpu.jl")); using .Residual3DGPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3])
M    = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
Rref = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_Rrus.f64")))),35,n,n,n)
Rg = residual3d_box_gpu(Array(M),n,n,n,dx,Ma; order=2, riemann_solver=:rusanov, vacuum_floor=0.001, project_faces=true)
r=maximum(abs.(Rg.-Rref)./max.(abs.(Rref),1.0)); a=maximum(abs.(Rg.-Rref))
@printf("GPU :rusanov vs CPU :rusanov, %d^3 Ma=%.0f: max abs=%.3e max rel=%.3e GATE=%s\n",
        n,Ma,a,r, r<=1e-6 ? "PASS" : "FAIL")

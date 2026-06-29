# validate_order1_vs_cpu.jl — GPU first-order (order=1) residual vs CPU residual_ho_3d!(order=1).
# Prereq: dump_cpu_order1_residual.jl (main env) writes r3d_R1.f64. Run under gpuenv2.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_gpu.jl")); using .Residual3DGPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3])
M    = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
R1ref= reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_R1.f64")))),35,n,n,n)
Rg1  = residual3d_box_gpu(Array(M),n,n,n,dx,Ma; order=1, vacuum_floor=0.001, project_faces=true)
r=maximum(abs.(Rg1.-R1ref)./max.(abs.(R1ref),1.0)); a=maximum(abs.(Rg1.-R1ref))
@printf("GPU order=1 vs CPU residual_ho_3d!(order=1), %d^3 Ma=%.0f: max abs=%.3e max rel=%.3e GATE=%s\n",
        n,Ma,a,r, r<=1e-6 ? "PASS" : "FAIL")

# validate_theta_diag.jl — isolate the dt=dtN parity gap: is it the CLOSED-FORM θ* (GPU vs
# CPU) or something shared? Compares GPU(theta_closed=true) vs closed golden AND
# GPU(theta_closed=false) vs bisection golden, at dt=dtN. If bisection PASSES and closed
# FAILS -> the closed-form θ* GPU path is the bug.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
repo = normpath(joinpath(@__DIR__, "..", ".."))
isfile(joinpath(DATA,"r3d_ho3_RN.f64"))     || run(setenv(`$(Base.julia_cmd()) --project=$repo $(joinpath(@__DIR__,"dump_cpu_hiorder3_residual.jl"))`, ENV))
isfile(joinpath(DATA,"r3d_ho3_bis_RN.f64")) || run(setenv(`$(Base.julia_cmd()) --project=$repo $(joinpath(@__DIR__,"dump_cpu_hiorder3_bisection.jl"))`, ENV))

meta = split(strip(read(joinpath(DATA,"r3d_ho3.meta"), String)), '\n')
n=parse(Int,meta[1]); dx=parse(Float64,meta[2]); Ma=parse(Float64,meta[3]); g=parse(Int,meta[4]); dtN=parse(Float64,meta[5]); s3max=parse(Float64,meta[6])
ld(f)=reshape(collect(reinterpret(Float64, read(joinpath(DATA,f)))), 35, n, n, n)
Mint=ld("r3d_ho3_M.f64"); RN_cl=ld("r3d_ho3_RN.f64"); RN_bis=ld("r3d_ho3_bis_RN.f64")
nf=n+2g; G=zeros(35,nf,nf,nf); cl(a)= a<1 ? 1 : (a>n ? n : a)
for c in 1:nf, b in 1:nf, a in 1:nf; @views G[:,a,b,c] .= Mint[:, cl(a-g), cl(b-g), cl(c-g)]; end
rel(A,B)=maximum(abs.(A .- B) ./ max.(abs.(B), 1.0))

Rg_cl  = residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,dtN; s3max=s3max, theta_closed=true)
Rg_bis = residual3d_order3_gpu(G,n,n,n,g,dx,dx,dx,Ma,dtN; s3max=s3max, theta_closed=false)
@printf("dt=dtN  GPU(closed)   vs CPU(closed)   : max rel=%.3e  %s\n", rel(Rg_cl,RN_cl),  rel(Rg_cl,RN_cl)<=1e-10  ? "PASS" : "FAIL")
@printf("dt=dtN  GPU(bisection)vs CPU(bisection): max rel=%.3e  %s\n", rel(Rg_bis,RN_bis),rel(Rg_bis,RN_bis)<=1e-10 ? "PASS" : "FAIL")
@printf("closed-vs-bisection (CPU goldens) diff : max abs=%.3e  (expected ~1e-6, the analytic approx)\n", maximum(abs.(RN_cl .- RN_bis)))

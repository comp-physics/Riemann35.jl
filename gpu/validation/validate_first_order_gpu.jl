# validate_first_order_gpu.jl — GPU first_order=true must (a) run/survive, (b) be MORE
# diffusive than order-3 base, (c) match the CPU order=1 ladder numbers on the same
# 24³ moving contact (HLL-o1: u@15=7.572e-4 p@15=8.211e-2 ... u@100=2.440e-2 p@100=6.102e-1).
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
include(joinpath(@__DIR__, "..", "..", "src", "Riemann35.jl")); using .Riemann35: InitializeM4_35
g=4
build_contact(N,g,u0,p0,ratio)=begin
    nf=N+2g; G=zeros(35,nf,nf,nf)
    for k in 1:N,j in 1:N,i in 1:N
        rho = i<=N÷2 ? 1.0 : ratio; T=p0/rho
        @views G[:,i+g,j+g,k+g] .= InitializeM4_35(rho,u0,0.0,0.0,T,0.0,0.0,T,0.0,T)
    end; G
end
function cerr(Gi,N,u0,p0)
    um=0.0;pm=0.0;ok=true
    for k in 1:N,j in 1:N,i in 1:N
        r=Gi[1,i,j,k]; (r>0 && isfinite(r)) || (ok=false;continue)
        um=max(um,abs(Gi[2,i,j,k]/r-u0)); pm=max(pm,abs((Gi[3,i,j,k]-Gi[2,i,j,k]^2/r)-p0))
    end; (ok,um,pm)
end
# CPU order=1 reference (from the 3-rung ladder, same case)
cpu_o1 = Dict(15=>(7.572e-4,8.211e-2),30=>(2.618e-3,1.855e-1),50=>(6.958e-3,3.044e-1),
              75=>(1.461e-2,4.476e-1),100=>(2.440e-2,6.102e-1))
N=24; u0=0.5; p0=1.0; ratio=1000.0; Ma=1.0; dx=1.0/N; dt=0.2*dx/(u0+4.0*sqrt(p0))
G0=build_contact(N,g,u0,p0,ratio)
cps=(15,30,50,75,100)
@printf("GPU %s, 24³ moving contact\n","first_order vs base vs CPU order=1")
for (nm,fo) in (("base",false),("firstO",true))
    G=CuArray(copy(G0)); step=0
    for target in cps
        march3d_order3_gpu!(G,dx,Ma,target-step;dts=fill(dt,target-step),s3max=40.0,first_order=fo); step=target
        Gi=Array(G)[:,g+1:g+N,g+1:g+N,g+1:g+N]; ok,um,pm=cerr(Gi,N,u0,p0)
        if fo
            cu,cp=cpu_o1[target]
            @printf("  firstO @%3d u=%.4e p=%.4e  | CPU-o1 u=%.4e p=%.4e  | Δu/rel=%.2e Δp/rel=%.2e %s\n",
                    target,um,pm,cu,cp, abs(um-cu)/cu, abs(pm-cp)/cp, ok ? "" : "<NaN>")
        else
            @printf("  base   @%3d u=%.4e p=%.4e %s\n", target,um,pm, ok ? "" : "<NaN>")
        end
    end
end
println("PASS if firstO matches CPU-o1 to a few % (same scheme, GPU-march vs CPU-step details differ ~1e-2).")

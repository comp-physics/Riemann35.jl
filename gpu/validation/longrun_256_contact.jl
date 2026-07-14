# longrun_256_contact.jl — 256^3 moving-contact accuracy, WENO5(base) vs log-J, on GPU.
# First a 32^3 SANITY that must reproduce the prior run's numbers (confirms the first_order
# refactor left order-3 byte-neutral), then the 256^3 comparison with memory freed between
# schemes (54 GB each; both at once would OOM the 80 GB H100).
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
run_scheme(N,lj,nchunk,perchunk,u0,p0,ratio,Ma) = begin
    dx=1.0/N; dt=0.2*dx/(u0+4.0*sqrt(p0))
    G=CuArray(build_contact(N,g,u0,p0,ratio))
    mass0=sum(@view Array(G)[1,g+1:g+N,g+1:g+N,g+1:g+N])
    for ch in 1:nchunk
        march3d_order3_gpu!(G,dx,Ma,perchunk;dts=fill(dt,perchunk),s3max=40.0,use_logjacobi_recon=lj)
        Gi=Array(G)[:,g+1:g+N,g+1:g+N,g+1:g+N]
        ok,um,pm=cerr(Gi,N,u0,p0); dr=abs(sum(Gi[1,:,:,:])-mass0)/mass0
        @printf("%-6s %-5s %-4d u_err=%.4e p_err=%.4e mass_drift=%.2e%s\n",
                "$(N)³", lj ? "logJ" : "base", ch*perchunk, um, pm, dr, ok ? "" : "  <NaN/rho<=0>")
        ok || break
    end
    G=nothing; GC.gc(); CUDA.reclaim()
end

println("=== SANITY 32³ (must match prior: base@25 u=1.30e-2 p=4.23e-1 ; logJ@25 u=1.88e-2 p=2.88e-1) ===")
run_scheme(32,false,1,25,0.5,1.0,1000.0,1.0)
run_scheme(32,true ,1,25,0.5,1.0,1000.0,1.0)

println("\n=== 256³ moving contact (Ma=1), WENO5 vs log-J, 80 steps ===")
run_scheme(256,false,4,20,0.5,1.0,1000.0,1.0)
run_scheme(256,true ,4,20,0.5,1.0,1000.0,1.0)
println("\nDone.")

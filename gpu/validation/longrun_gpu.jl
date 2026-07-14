# longrun_gpu.jl — long GPU marches, base (WENO5+HLL) vs log-J, with checkpoints.
#  A) moving contact (Ma=1): ACCURACY vs exact (max|u-u0|,|p-p0|) + stability over time.
#  B) Ma=100 crossing jets: STABILITY over ~150 steps + conservation + base-vs-logJ diff.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
include(joinpath(@__DIR__, "..", "..", "src", "Riemann35.jl")); using .Riemann35: InitializeM4_35
DATA = get(ENV,"RIEMANN35_DATA", joinpath(@__DIR__,"..","..","data"))
g=4

# ---- A) moving contact accuracy ----
function build_contact(N,g,u0,p0,ratio)
    nf=N+2g; G=zeros(35,nf,nf,nf)
    for k in 1:N,j in 1:N,i in 1:N
        rho = i<=N÷2 ? 1.0 : ratio; T=p0/rho
        @views G[:,i+g,j+g,k+g] .= InitializeM4_35(rho,u0,0.0,0.0,T,0.0,0.0,T,0.0,T)
    end
    G
end
function contact_err(Gi,N,u0,p0)
    um=0.0;pm=0.0;ok=true
    ρ=Gi[1,:,:,:]
    for k in 1:N,j in 1:N,i in 1:N
        r=Gi[1,i,j,k]; (r>0 && isfinite(r)) || (ok=false;continue)
        um=max(um,abs(Gi[2,i,j,k]/r-u0)); pm=max(pm,abs((Gi[3,i,j,k]-Gi[2,i,j,k]^2/r)-p0))
    end
    (ok,um,pm)
end
println("=== A) moving contact (Ma=1), ACCURACY vs exact, base vs log-J ===")
let N=32, u0=0.5, p0=1.0, ratio=1000.0, Ma=1.0, dx=1.0/N, s3=40.0
    G0=build_contact(N,g,u0,p0,ratio)
    dt=0.2*dx/(u0+4.0*sqrt(p0))
    @printf("%-10s %-7s %-11s %-11s %-11s\n","mode","step","u_err","p_err","mass_drift")
    for (nm,lj) in (("base",false),("logJ",true))
        G=CuArray(copy(G0)); mass0=sum(@view Array(G)[1,g+1:g+N,g+1:g+N,g+1:g+N])
        done=0
        for chunk in 1:4          # 4×25 = 100 steps
            march3d_order3_gpu!(G,dx,Ma,25;dts=fill(dt,25),s3max=s3,use_logjacobi_recon=lj)
            Gi=Array(G)[:,g+1:g+N,g+1:g+N,g+1:g+N]
            ok,um,pm=contact_err(Gi,N,u0,p0); dr=abs(sum(Gi[1,:,:,:])-mass0)/mass0
            @printf("%-10s %-7d %-11.3e %-11.3e %-11.2e\n", nm, chunk*25, um, pm, dr)
            ok || (println("   ^ NaN/rho<=0"); break)
        end
    end
end

# ---- B) Ma=100 crossing jets stability ----
println("\n=== B) Ma=100 crossing jets, STABILITY (150 steps), base vs log-J ===")
need=["r3d_cross_ma100.f64","r3d_cross_ma100.meta"]
if any(f->!isfile(joinpath(DATA,f)),need)
    repo=normpath(joinpath(@__DIR__,"..",".."))
    run(setenv(`$(Base.julia_cmd()) --project=$repo $(joinpath(@__DIR__,"dump_cpu_hiorder3_march.jl"))`,ENV))
end
cmeta=split(strip(read(joinpath(DATA,"r3d_cross_ma100.meta"),String)),'\n'); MaB=parse(Float64,cmeta[1])
cross=reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_cross_ma100.f64")))),35,3)
bg=cross[:,1];Mt=cross[:,2];Mb=cross[:,3]; s3B=max(40.0,4.0+abs(MaB)/2.0)
function build_crossing(N,g)
    nf=N+2g;G=zeros(35,nf,nf,nf);Cs=floor(Int,0.1N)
    Minb=div(N,2)-Cs;Maxb=div(N,2);Mnt=div(N,2)+1;Maxt=div(N,2)+1+Cs
    for k in 1:N,j in 1:N,i in 1:N
        v=bg
        if Minb<=i<=Maxb&&Minb<=j<=Maxb&&Minb<=k<=Maxb;v=Mb;end
        if Mnt<=i<=Maxt&&Mnt<=j<=Maxt&&Mnt<=k<=Maxt;v=Mt;end
        @views G[:,i+g,j+g,k+g].=v
    end;G
end
let N=64
    Gh=build_crossing(N,g); mass0=sum(@view Gh[1,g+1:g+N,g+1:g+N,g+1:g+N])
    @printf("%-10s %-7s %-24s %-11s %-11s\n","mode","step","rho range","mass_drift","M200_drift")
    E0=sum(@view Gh[3,g+1:g+N,g+1:g+N,g+1:g+N])
    for (nm,lj) in (("base",false),("logJ",true))
        G=CuArray(copy(Gh))
        for chunk in 1:6         # 6×25 = 150 steps
            march3d_order3_gpu!(G,1.0/N,MaB,25;s3max=s3B,use_logjacobi_recon=lj)
            Gi=Array(G)[:,g+1:g+N,g+1:g+N,g+1:g+N]; ρ=Gi[1,:,:,:]
            ok=all(isfinite,Gi)&&minimum(ρ)>0
            dr=abs(sum(ρ)-mass0)/mass0; dE=abs(sum(Gi[3,:,:,:])-E0)/abs(E0)
            @printf("%-10s %-7d [%.3e, %.3e]  %-11.2e %-11.2e\n", nm, chunk*25, minimum(ρ),maximum(ρ),dr,dE)
            ok || (println("   ^ CRASHED (NaN/rho<=0)"); break)
        end
    end
end
println("\nDone.")

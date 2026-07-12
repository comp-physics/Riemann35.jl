# kfvs_classA_seeds.jl — dump the CHyQMOM seed representation for every class-A (and A/B) cell so the
# continuous-defect HOMOTOPY can run in a solver-only env without Riemann35. Per cell we record:
#   - ρ, μ=(ux,uy,uz), σ=(σx,σy,σz)      (standardization of the TRUE moment vector M = center_state)
#   - sstd[35]                            (standardized target M, i.e. homotopy endpoint t=1)
#   - seed atoms {w_a, X_a^std}           (CHyQMOM's own 27 nodes; they represent M̃, the t=0 start)
#   - seed physical speeds                (for the CFL sanity check: λ Σ_d |μ_d+σ_d X_ad| must be <=1)
# Env: r35env (Riemann35 for chyqmom_nodes_3d_dev). Output: flat Float64 arrays keyed by column.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, JLD2, Printf, LinearAlgebra
include(joinpath(@__DIR__,"..","kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev.KFVSInversionDev: chyqmom_nodes_3d_dev
setprecision(BigFloat, 512)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)=(k<0||k>n) ? big(0) : big(binomial(n,k))
ce=load(joinpath(@__DIR__,"kfvs_defect_counterexample.jld2")); Cst=ce["center_state"]; cls=ce["class"]; λ=ce["lam"][1]

function standardize(Mraw)
    M=big.(Mraw); ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
    σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
    s=zeros(BigFloat,35)
    for n in 1:35; (i,j,k)=TRIP[n]; acc=big(0.0)
        for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
            acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
        s[n]=acc/(σx^i*σy^j*σz^k); end
    (s,(Float64(ux),Float64(uy),Float64(uz)),(Float64(σx),Float64(σy),Float64(σz)))
end
function chy(m); nn,nx,ny,nz,Nn=chyqmom_nodes_3d_dev(m); w=Float64[];U=NTuple{3,Float64}[]
    for q in 1:Nn; nn[q]>0 && (push!(w,nn[q]);push!(U,(nx[q],ny[q],nz[q]))); end; (w,U); end

which = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 1     # 1=class A
targets=findall(==(which),cls)
@printf("class-%d cells: %d  (dumping CHyQMOM seeds)\n", which, length(targets)); flush(stdout)
out=Dict{String,Any}("cols"=>targets, "lam"=>Float64(λ))
for ccol in targets
    Mraw=Cst[:,ccol]; ρ=Mraw[1]
    sstd,mu,sig=standardize(Mraw)
    w,U=chy(ntuple(q->Mraw[q],Val(35)))             # CHyQMOM physical nodes (weights sum to ρ)
    Na=length(w)
    Xstd=zeros(3,Na); wstd=zeros(Na); speed=zeros(Na)
    for a in 1:Na
        Xstd[1,a]=(U[a][1]-mu[1])/sig[1]; Xstd[2,a]=(U[a][2]-mu[2])/sig[2]; Xstd[3,a]=(U[a][3]-mu[3])/sig[3]
        wstd[a]=w[a]/ρ
        speed[a]=Float64(λ)*(abs(U[a][1])+abs(U[a][2])+abs(U[a][3]))
    end
    # M̃^std : standardized moments actually reproduced by the seed nodes
    Mtil=zeros(35)
    for n in 1:35; (i,j,k)=TRIP[n]; Mtil[n]=sum(wstd[a]*Xstd[1,a]^i*Xstd[2,a]^j*Xstd[3,a]^k for a in 1:Na); end
    defect=Float64.(sstd).-Mtil
    out["sstd_$ccol"]=Float64.(sstd); out["mtil_$ccol"]=Mtil; out["defect_$ccol"]=defect
    out["X_$ccol"]=Xstd; out["w_$ccol"]=wstd; out["mu_$ccol"]=collect(mu); out["sig_$ccol"]=collect(sig)
    out["rho_$ccol"]=Float64(ρ); out["speed_$ccol"]=speed
    @printf("  col%-4d Na=%2d  seed max CFL speed=%.4f  |defect|_2=%.3e  |defect|_inf=%.3e (moment #%d)\n",
            ccol, Na, maximum(speed), norm(defect), maximum(abs,defect), argmax(abs.(defect))); flush(stdout)
end
save(joinpath(@__DIR__,"kfvs_classA_seeds.jld2"), out)
@printf("saved %d seeds -> kfvs_classA_seeds.jld2\n", length(targets))

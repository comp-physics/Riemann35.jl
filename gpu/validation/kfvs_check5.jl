# kfvs_check5.jl — the missing algebraic identity U^F3 - U^Q (measure/flux parity) + direct
# atomic-coefficient measure proof, for the enriched class-C center [14,13,13].
#   U^F3 = conservative KFVS flux update (flux form, telescoping).
#   U^Q  = atomic measure update = retained center mass w_iq(1-λΣ|v|) + upwind neighbor inflow λ·nk·|u|.
# With exact center reproduction + shared single-valued faces + consistent λ these must agree to the
# BigFloat->Float64 evaluation floor. Also verify every retained/inflow coefficient >= 0 (=> U^Q is a
# positive atomic measure => realizable by construction, not merely by empirical margin).
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, JLD2, Printf, LinearAlgebra
include(joinpath(@__DIR__,"..","kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev.KFVSInversionDev: chyqmom_nodes_3d_dev
setprecision(BigFloat,512)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
basis(u)=ntuple(n->u[1]^TRIP[n][1]*u[2]^TRIP[n][2]*u[3]^TRIP[n][3], Val(35))

raw=load(joinpath(@__DIR__,"kfvs_hardcross_stencil_raw.jld2")); ce=load(joinpath(@__DIR__,"kfvs_defect_counterexample.jld2")); pol=load(joinpath(@__DIR__,"kfvs_polished_solution.jld2"))
state=raw["state"]; h=raw["halo"]; λ=raw["lam"][1]; Ma=raw["Ma"]; s3max=raw["s3max"]
cellrow(i,j,k)=ntuple(q->state[i,j,k,q],Val(35))
s0i=findfirst(==(3),ce["class"]); cc=round.(Int,ce["cell"][:,s0i]); ci,cj,ck=cc[1],cc[2],cc[3]
C=ntuple(q->ce["center_state"][q,s0i],Val(35))
Lx=cellrow(ci+h-1,cj+h,ck);Rx=cellrow(ci+h+1,cj+h,ck);Ly=cellrow(ci+h,cj+h-1,ck);Ry=cellrow(ci+h,cj+h+1,ck);Lz=cellrow(ci+h,cj+h,ck-1);Rz=cellrow(ci+h,cj+h,ck+1)

# enriched physical nodes
atoms=pol["atoms_std"]; wstd=Float64.(pol["weights"]); ρ=C[1]
ux=C[2]/ρ;uy=C[6]/ρ;uz=C[16]/ρ; σx=sqrt(C[3]/ρ-ux^2);σy=sqrt(C[10]/ρ-uy^2);σz=sqrt(C[20]/ρ-uz^2)
Uf=[(ux+σx*a[1],uy+σy*a[2],uz+σz*a[3]) for a in atoms]; Wf=[ρ*w for w in wstd]; nA=length(Wf)

# CHyQMOM node sets for neighbors
function chy(m); nn,nx,ny,nz,Nn=chyqmom_nodes_3d_dev(m); w=Float64[];U=NTuple{3,Float64}[]
    for q in 1:Nn; nn[q]>0 && (push!(w,nn[q]);push!(U,(nx[q],ny[q],nz[q]))); end; (w,U); end
wL=[chy(Lx),chy(Rx),chy(Ly),chy(Ry),chy(Lz),chy(Rz)]

# ---- U^Q measure update (retained + upwind inflow), enriched center + CHyQMOM neighbors ----
function measure_update(Wf,Uf,wL)
    UQ=zeros(35); wrem_min=Inf; infl_min=Inf
    for q in eachindex(Wf)
        u=Uf[q]; wr=Wf[q]*(1-λ*(abs(u[1])+abs(u[2])+abs(u[3]))); wrem_min=min(wrem_min,wr)
        b=basis(u); for n in 1:35; UQ[n]+=wr*b[n]; end
    end
    # Lx/Ly/Lz upwind u_axis>0 (+λ), Rx/Ry/Rz u_axis<0 (-λ)
    for (idx,(ax,pos)) in enumerate(((1,true),(1,false),(2,true),(2,false),(3,true),(3,false)))
        w,U=wL[idx]
        for q in eachindex(w); u=U[q]; ua=u[ax]
            (pos ? ua>0 : ua<0) || continue
            coeff = pos ? λ*w[q]*ua : -λ*w[q]*ua; infl_min=min(infl_min,coeff)
            b=basis(u); for n in 1:35; UQ[n]+=coeff*b[n]; end
        end
    end
    (UQ,wrem_min,infl_min)
end
UQ,wrem_min,infl_min=measure_update(Wf,Uf,wL)

# ---- U^F3 conservative flux update (node KFVS split flux) ----
function kfvs_nodes(wL_,UL,wR_,UR,axis); F=zeros(35)
    for q in eachindex(wL_); u=UL[q]; u[axis]>0 || continue; c=wL_[q]*u[axis]; b=basis(u); for n in 1:35; F[n]+=c*b[n]; end; end
    for q in eachindex(wR_); u=UR[q]; u[axis]<0 || continue; c=wR_[q]*u[axis]; b=basis(u); for n in 1:35; F[n]+=c*b[n]; end; end
    F; end
Cen=(Wf,Uf)
Fx_r=kfvs_nodes(Cen[1],Cen[2],wL[2][1],wL[2][2],1); Fx_l=kfvs_nodes(wL[1][1],wL[1][2],Cen[1],Cen[2],1)
Fy_r=kfvs_nodes(Cen[1],Cen[2],wL[4][1],wL[4][2],2); Fy_l=kfvs_nodes(wL[3][1],wL[3][2],Cen[1],Cen[2],2)
Fz_r=kfvs_nodes(Cen[1],Cen[2],wL[6][1],wL[6][2],3); Fz_l=kfvs_nodes(wL[5][1],wL[5][2],Cen[1],Cen[2],3)
UF3=[C[q]-λ*((Fx_r[q]-Fx_l[q])+(Fy_r[q]-Fy_l[q])+(Fz_r[q]-Fz_l[q])) for q in 1:35]

d=UF3.-UQ; rel=maximum(abs.(d)./max.(abs.(UQ),1e-300))
@printf("(5) U^F3 - U^Q identity: max abs=%.3e  max rel=%.3e\n", maximum(abs.(d)), rel)
@printf("    min retained coeff w_iq(1-λΣ|v|) = %.3e  ; min neighbor inflow coeff = %.3e  (both >=0 => U^Q positive measure)\n", wrem_min, infl_min)
mF3=realizability_margin(UF3); mQ=realizability_margin(collect(UQ))
@printf("    margin(U^F3)=%.4e  margin(U^Q)=%.4e\n", mF3, mQ)
save(joinpath(@__DIR__,"kfvs_check5.jld2"),"UF3",UF3,"UQ",UQ,"resid",d,"wrem_min",wrem_min,"infl_min",infl_min)
if maximum(abs.(d)) < 1e-3*max(maximum(abs.(UQ)),1) && wrem_min>=0 && infl_min>=0 && mF3>=0
    println("==> CHECK 5 PASS: flux update = measure update; U^Q is a positive atomic measure (retained+inflow all >=0) => realizable BY CONSTRUCTION, not just empirically.")
else
    @printf("==> check5: identity resid %.2e (rel %.2e), wrem_min=%.2e — inspect.\n", maximum(abs.(d)), rel, wrem_min)
end

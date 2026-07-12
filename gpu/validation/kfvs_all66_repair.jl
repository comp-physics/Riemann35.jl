# kfvs_all66_repair.jl — PHASE 2: donor-repair test for every cleanly-constructed class-C cell.
# For each OK cubature, substitute it for the cell's defective CHyQMOM center and check whether the
# F3 anchor margin goes from <0 (baseline) to >=0 (enriched), with finite next inversion. Reports the
# distribution of repaired margins. Env: r35env (Riemann35 + JLD2).
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, JLD2, Printf, LinearAlgebra
include(joinpath(@__DIR__,"..","kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev.KFVSInversionDev: chyqmom_nodes_3d_dev
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
basis(u)=ntuple(n->u[1]^TRIP[n][1]*u[2]^TRIP[n][2]*u[3]^TRIP[n][3], Val(35))
raw=load(joinpath(@__DIR__,"kfvs_hardcross_stencil_raw.jld2")); ce=load(joinpath(@__DIR__,"kfvs_defect_counterexample.jld2")); cub=load(joinpath(@__DIR__,"kfvs_all66_cubatures_flat.jld2"))
state=raw["state"]; h=raw["halo"]; λ=raw["lam"][1]; Ma=raw["Ma"]; s3max=raw["s3max"]
cellrow(i,j,k)=ntuple(q->state[i,j,k,q],Val(35))
oks=cub["cols"]
function chy(m); nn,nx,ny,nz,Nn=chyqmom_nodes_3d_dev(m); w=Float64[];U=NTuple{3,Float64}[]
    for q in 1:Nn; nn[q]>0 && (push!(w,nn[q]);push!(U,(nx[q],ny[q],nz[q]))); end; (w,U); end
function kfvs_nodes(wL,UL,wR,UR,axis); F=zeros(35)
    for q in eachindex(wL); u=UL[q]; u[axis]>0 || continue; c=wL[q]*u[axis]; b=basis(u); for n in 1:35; F[n]+=c*b[n]; end; end
    for q in eachindex(wR); u=UR[q]; u[axis]<0 || continue; c=wR[q]*u[axis]; b=basis(u); for n in 1:35; F[n]+=c*b[n]; end; end
    F; end
function mlo(Cen,Ln,Rn,C)
    Fx_r=kfvs_nodes(Cen[1],Cen[2],Rn[1][1],Rn[1][2],1); Fx_l=kfvs_nodes(Ln[1][1],Ln[1][2],Cen[1],Cen[2],1)
    Fy_r=kfvs_nodes(Cen[1],Cen[2],Rn[2][1],Rn[2][2],2); Fy_l=kfvs_nodes(Ln[2][1],Ln[2][2],Cen[1],Cen[2],2)
    Fz_r=kfvs_nodes(Cen[1],Cen[2],Rn[3][1],Rn[3][2],3); Fz_l=kfvs_nodes(Ln[3][1],Ln[3][2],Cen[1],Cen[2],3)
    [C[q]-λ*((Fx_r[q]-Fx_l[q])+(Fy_r[q]-Fy_l[q])+(Fz_r[q]-Fz_l[q])) for q in 1:35]
end
@printf("phase-2 donor repair over %d cleanly-constructed class-C cells:\n", length(oks))
rep=0; res=[]
for ccol in oks
    cc=round.(Int,ce["cell"][:,ccol]); ci,cj,ck=cc[1],cc[2],cc[3]
    C=ntuple(q->ce["center_state"][q,ccol],Val(35))
    Lx=cellrow(ci+h-1,cj+h,ck);Rx=cellrow(ci+h+1,cj+h,ck);Ly=cellrow(ci+h,cj+h-1,ck);Ry=cellrow(ci+h,cj+h+1,ck);Lz=cellrow(ci+h,cj+h,ck-1);Rz=cellrow(ci+h,cj+h,ck+1)
    Ln=(chy(Lx),chy(Ly),chy(Lz)); Rn=(chy(Rx),chy(Ry),chy(Rz))
    A=cub["atoms_$ccol"]; atoms=[(A[1,j],A[2,j],A[3,j]) for j in 1:size(A,2)]; wstd=cub["w_$ccol"]; mu=cub["mu_$ccol"]; sig=cub["sig_$ccol"]; ρ=C[1]
    Uf=[(mu[1]+sig[1]*a[1],mu[2]+sig[2]*a[2],mu[3]+sig[3]*a[3]) for a in atoms]; Wf=[ρ*w for w in wstd]
    Cch=chy(C)
    mlo_base=mlo((Cch[1],Cch[2]),Ln,Rn,C); mlo_enr=mlo((Wf,Uf),Ln,Rn,C)
    mb= all(isfinite,mlo_base) ? realizability_margin(collect(mlo_base)) : -Inf
    me= all(isfinite,mlo_enr) ? realizability_margin(collect(mlo_enr)) : -Inf
    invok = try; nn,nx,ny,nz,Nn=chyqmom_nodes_3d_dev(ntuple(q->mlo_enr[q],Val(35))); all(isfinite,nn) catch; false end
    repaired = (mb<0) && (me>=0) && invok
    repaired && (global rep+=1); push!(res,(ccol,mb,me,invok,repaired))
    @printf("  col%-4d baseline margin=%-11.3e enriched=%-11.3e inv=%-5s %s\n", ccol, mb, me, invok, repaired ? "REPAIRED" : (me>=0 ? "ok(base already>=0?)" : "still<0"))
end
mes=[r[3] for r in res if r[5]]
@printf("\n=== PHASE 2: %d/%d cleanly-constructed cells REPAIRED (baseline<0 -> enriched>=0, inversion finite) ===\n", rep, length(oks))
isempty(mes) || @printf("repaired enriched-margin: min=%.3e median=%.3e max=%.3e\n", minimum(mes), sort(mes)[cld(length(mes),2)], maximum(mes))
save(joinpath(@__DIR__,"kfvs_all66_repair.jld2"),"cols",[r[1] for r in res],"m_base",[r[2] for r in res],"m_enr",[r[3] for r in res],"repaired",[r[5] for r in res])

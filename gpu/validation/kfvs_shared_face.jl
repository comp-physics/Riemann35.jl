# kfvs_shared_face.jl — THREAD 2: conservative shared-face full-stage recomputation.
# For each repaired class-C center, enrich its donor cubature, recompute its 6 incident faces with a
# SINGLE-VALUED flux (identical to both adjacent cells => exact conservation), reconstruct every
# affected endpoint, and GATE the neighborhood. If a neighbor endpoint fails, expand the enriched
# donor set (only cells with an available cubature) and repeat. Record whether the enriched region
# STABILIZES locally, CASCADES, or is BLOCKED by an unavailable donor cubature.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, JLD2, Printf, LinearAlgebra
include(joinpath(@__DIR__,"..","kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev.KFVSInversionDev: chyqmom_nodes_3d_dev
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
basis(u)=ntuple(n->u[1]^TRIP[n][1]*u[2]^TRIP[n][2]*u[3]^TRIP[n][3], Val(35))
raw=load(joinpath(@__DIR__,"kfvs_hardcross_stencil_raw.jld2")); ce=load(joinpath(@__DIR__,"kfvs_defect_counterexample.jld2")); cub=load(joinpath(@__DIR__,"kfvs_all66_cubatures_flat.jld2"))
state=raw["state"]; h=raw["halo"]; N=raw["N"]; λ=raw["lam"][1]
nfx,nfy,nfz=size(state,1),size(state,2),size(state,3)
cellrow(i,j,k)=ntuple(q->state[i,j,k,q],Val(35))
oks=cub["cols"]; allcells=ce["cell"]

# enriched physical node set per OK cell, keyed by unhaloed (i,j,k)
cubmap=Dict{NTuple{3,Int},Tuple{Vector{Float64},Vector{NTuple{3,Float64}}}}()
cellcol=Dict{NTuple{3,Int},Int}()
for ccol in oks
    ijk=Tuple(round.(Int,allcells[:,ccol])); C=cellrow(ijk[1]+h,ijk[2]+h,ijk[3]); ρ=C[1]
    A=cub["atoms_$ccol"]; w=cub["w_$ccol"]; mu=cub["mu_$ccol"]; sig=cub["sig_$ccol"]
    Uf=[(mu[1]+sig[1]*A[1,j],mu[2]+sig[2]*A[2,j],mu[3]+sig[3]*A[3,j]) for j in 1:size(A,2)]
    cubmap[ijk]=([ρ*x for x in w],Uf)
end
for ccol in 1:size(allcells,2); cellcol[Tuple(round.(Int,allcells[:,ccol]))]=ccol; end
isCcell(ijk)=haskey(cellcol,ijk) && ce["class"][cellcol[ijk]]==3

function chy(m); nn,nx,ny,nz,Nn=chyqmom_nodes_3d_dev(m); w=Float64[];U=NTuple{3,Float64}[]
    for q in 1:Nn; nn[q]>0 && (push!(w,nn[q]);push!(U,(nx[q],ny[q],nz[q]))); end; (w,U); end
function fluxN(Ln,Rn,axis); F=zeros(35)
    for q in eachindex(Ln[1]); u=Ln[2][q]; u[axis]>0||continue; c=Ln[1][q]*u[axis]; b=basis(u); for n in 1:35; F[n]+=c*b[n]; end; end
    for q in eachindex(Rn[1]); u=Rn[2][q]; u[axis]<0||continue; c=Rn[1][q]*u[axis]; b=basis(u); for n in 1:35; F[n]+=c*b[n]; end; end
    F; end
inb(i,j,k)= 1<=i<=nfx && 1<=j<=nfy && 1<=k<=nfz
cubOf(ijk,D)= (ijk in D && haskey(cubmap,ijk)) ? cubmap[ijk] : chy(cellrow(ijk[1]+h,ijk[2]+h,ijk[3]))
function endpoint(ijk,D)
    C=cellrow(ijk[1]+h,ijk[2]+h,ijk[3]); cN=cubOf(ijk,D)
    nb(di,dj,dk)=cubOf((ijk[1]+di,ijk[2]+dj,ijk[3]+dk),D)
    Fx=fluxN(cN,nb(1,0,0),1).-fluxN(nb(-1,0,0),cN,1)
    Fy=fluxN(cN,nb(0,1,0),2).-fluxN(nb(0,-1,0),cN,2)
    Fz=fluxN(cN,nb(0,0,1),3).-fluxN(nb(0,0,-1),cN,3)
    [C[q]-λ*(Fx[q]+Fy[q]+Fz[q]) for q in 1:35]
end
mg(ijk,D)= (e=endpoint(ijk,D); all(isfinite,e) ? realizability_margin(e) : -Inf)

# iterative shared-face enrichment for one center
function run_center(c0; MAXIT=8, capsize=200)
    D=Set([c0])
    ring(c)= [(c[1]+d[1],c[2]+d[2],c[3]+d[3]) for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))]
    affected=Set([c0]); for r in ring(c0); (2<=r[3]<=N-1)&&push!(affected,r); end
    iters=0; unavail=Set{NTuple{3,Int}}()
    for it in 1:MAXIT; iters=it
        fails=[c for c in affected if mg(c,D)<0]
        isempty(fails) && return (:STABILIZED,iters,length(D),collect(unavail),affected)
        new=[c for c in fails if haskey(cubmap,c) && !(c in D)]
        for c in fails; (haskey(cubmap,c)||push!(unavail,c)); end
        isempty(new) && return (:BLOCKED,iters,length(D),collect(unavail),affected)
        for c in new; push!(D,c); for r in ring(c); (2<=r[3]<=N-1)&&push!(affected,r); end; end
        length(affected)>capsize && return (:CASCADE,iters,length(D),collect(unavail),affected)
    end
    (:MAXIT,iters,length(D),collect(unavail),affected)
end

# classify a cell: constructed class-C / unconstructed class-C / class A/B / passing (not a recorded fail)
function classify(ijk)
    haskey(cubmap,ijk) && return :C_constructed
    if haskey(cellcol,ijk); cl=ce["class"][cellcol[ijk]]; return cl==3 ? :C_unconstructed : :AB_fail; end
    :passing_broke
end
@printf("shared-face gating over %d repaired centers (affected = center + 1-ring, expand on failure):\n",length(oks))
tally=Dict(:STABILIZED=>0,:BLOCKED=>0,:CASCADE=>0,:MAXIT=>0)
allunav=Set{NTuple{3,Int}}()
for ccol in oks
    c0=Tuple(round.(Int,allcells[:,ccol]))
    base_center=mg(c0,Set{NTuple{3,Int}}())
    st,iters,nD,unav,aff=run_center(c0)
    tally[st]+=1; union!(allunav,unav)
    @printf("  col%-4d %-11s iters=%d |D_enriched|=%d |affected|=%d unavailable_donors=%d  base_center_margin=%.2e\n",
            ccol, st, iters, nD, length(aff), length(unav), base_center)
end
cbreak=Dict(:C_constructed=>0,:C_unconstructed=>0,:AB_fail=>0,:passing_broke=>0)
for c in allunav; cbreak[classify(c)]+=1; end
@printf("\nblocking (unavailable) cells across all centers: total=%d | C_unconstructed=%d  AB_fail=%d  passing_broke=%d  C_constructed=%d\n",
        length(allunav), cbreak[:C_unconstructed], cbreak[:AB_fail], cbreak[:passing_broke], cbreak[:C_constructed])
# diagnose the passing_broke cells: are they in the gated population (i,j in 1:N, k in 2:N-1) or x/y ghost cells?
inpop(c)= 1<=c[1]<=N && 1<=c[2]<=N && 2<=c[3]<=N-1
pb=[c for c in allunav if classify(c)==:passing_broke]
@printf("passing_broke detail (%d): %s\n", length(pb),
        join([@sprintf("(%d,%d,%d)%s",c[1],c[2],c[3], inpop(c) ? "-INPOP" : "-GHOST") for c in pb], " "))
@printf("  => %d in-population (real perturbed passing cells) ; %d ghost/halo (bookkeeping, not gated)\n",
        count(inpop,pb), count(!inpop,pb))
# discriminate: baseline (all-CHyQMOM) margin. >=0 => passing at baseline, broke only under neighbor
# enrichment (real perturbation, uniform-enrich would repair). <0 => my recompute disagrees w/ solver label.
empty=Set{NTuple{3,Int}}()
@printf("passing_broke baseline (all-CHyQMOM) margins:\n")
for c in pb; @printf("   (%d,%d,%d) baseline=%.3e  in_failset=%s\n", c[1],c[2],c[3], mg(c,empty), haskey(cellcol,c)); end
@printf("\n=== SHARED-FACE RESULT: STABILIZED=%d  BLOCKED(unavail donor)=%d  CASCADE=%d  MAXIT=%d  (of %d) ===\n",
        tally[:STABILIZED],tally[:BLOCKED],tally[:CASCADE],tally[:MAXIT],length(oks))
println("STABILIZED = enriched region closes with all margins >=0 (conservative local repair).")
println("BLOCKED    = a failing neighbor has NO available cubature => construction coverage is the blocker.")
println("CASCADE    = enriched region kept growing (cell-local success didn't localize).")

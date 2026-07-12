# kfvs_continuous_refine.jl — turn the grid WITNESS into an exact bounded CFL-safe cubature
# for ONE class-C target: grid FISTA → Carathéodory compression (≤35 atoms) → continuous
# constrained refinement (Levenberg–Marquardt on nodes+weights) driving the STANDARDIZED
# residual to ~1e-12, subject to w≥0 and |v|_CFL ≤ 1−δ. Then the required validation:
# componentwise 4th-moment errors, min weight, effective atom count, max CFL, cone margin,
# Jacobian conditioning. This tests EXACT bounded-support reproduction, not an ε-witness.
using Riemann35, JLD2, Printf, LinearAlgebra, Statistics
d=load("gpu/validation/kfvs_defect_counterexample.jld2"); λ=d["lam"][1]; C=d["center_state"]; cls=d["class"]
TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
cfl(v)=λ*(abs(v[1])+abs(v[2])+abs(v[3]))
pmono(v)=ntuple(n->(i=TRIP[n]; v[1]^i[1]*v[2]^i[2]*v[3]^i[3]), 35)
D4=Dict("m400"=>5,"m040"=>15,"m004"=>25,"m220"=>12,"m202"=>22,"m022"=>35)

s=findfirst(==(3),cls); M=collect(C[:,s]); ρ=M[1]
ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
σ(a,u)=sqrt(max(M[a]/ρ-u^2,1e-12)); σx=σ(3,ux);σy=σ(10,uy);σz=σ(20,uz)
S=[ρ*σx^TRIP[n][1]*σy^TRIP[n][2]*σz^TRIP[n][3] for n in 1:35]     # natural (standardized) moment scale
@printf("class-C target: mean(%.2f,%.2f,%.2f) σ(%.2g,%.2g,%.2g)  |ū|_CFL=%.3f  CHyQMOM node speed≈3079\n",ux,uy,uz,σx,σy,σz,cfl((ux,uy,uz)))

# --- 1. grid FISTA witness ---------------------------------------------------------
K=6.0; ng=13; gs=range(-K,K;length=ng)
grid=[[ux+σx*a,uy+σy*b,uz+σz*c] for a in gs for b in gs for c in gs if cfl((ux+σx*a,uy+σy*b,uz+σz*c))≤1.0]
Φ=reduce(hcat,[collect(pmono(v)) for v in grid]); Φs=Φ./S; bs=M./S
function gridwitness(Φs,bs;iters=20000)
    w=zeros(size(Φs,2)); wp=copy(w); t=1.0
    vv=randn(length(w)); for _ in 1:25; vv=Φs'*(Φs*vv); vv./=norm(vv); end; η=1/norm(Φs*(vv./norm(vv)))^2
    for _ in 1:iters; y=w.+((t-1)/t).*(w.-wp); wp=copy(w); w=max.(y.-η.*(Φs'*(Φs*y.-bs)),0.0); t=(1+sqrt(1+4t^2))/2; end
    w
end
w=gridwitness(Φs,bs)
@printf("grid witness: %d atoms, residual %.2e\n", count(>(1e-10*ρ),w), norm(Φs*w.-bs)/norm(bs))

# --- 2. seed: top-40 atoms by weight (pragmatic compression) -----------------------
ord=sortperm(w;rev=true)[1:min(40,length(w))]
nodes=[copy(grid[i]) for i in ord]; wt=[w[i] for i in ord]
Kn=length(nodes)

# --- 3. continuous constrained refinement (Levenberg–Marquardt) --------------------
function resid(nodes,wt)
    m=zeros(35); for b in 1:length(wt); p=pmono(nodes[b]); for n in 1:35; m[n]+=wt[b]*p[n]; end; end
    (m.-M)./S, m
end
function jac(nodes,wt)   # 35 × 4Kn : columns [w_1..w_Kn, vx.., vy.., vz..]
    Kn=length(wt); J=zeros(35,4Kn)
    for b in 1:Kn
        v=nodes[b]
        for n in 1:35
            (i,j,k)=TRIP[n]
            J[n,b]=(v[1]^i*v[2]^j*v[3]^k)/S[n]                                   # d/dw
            J[n,Kn+b]  = i==0 ? 0.0 : wt[b]*i*v[1]^(i-1)*v[2]^j*v[3]^k /S[n]     # d/dvx
            J[n,2Kn+b] = j==0 ? 0.0 : wt[b]*j*v[1]^i*v[2]^(j-1)*v[3]^k /S[n]     # d/dvy
            J[n,3Kn+b] = k==0 ? 0.0 : wt[b]*k*v[1]^i*v[2]^j*v[3]^(k-1) /S[n]     # d/dvz
        end
    end
    J
end
function lm_refine(nodes,wt;δ=1e-3,iters=400)
    Kn=length(wt); μ=1e-3; r,_=resid(nodes,wt)
    for it in 1:iters
        J=jac(nodes,wt); step=-(J'J+μ*I)\(J'*r)
        nw=[max(wt[b]+step[b],0.0) for b in 1:Kn]
        nn=[nodes[b].+[step[Kn+b],step[2Kn+b],step[3Kn+b]] for b in 1:Kn]
        for b in 1:Kn; c=cfl(nn[b]); c>1-δ && (nn[b].*= (1-δ)/c); end   # project into CFL polytope
        rn,_=resid(nn,nw)
        if norm(rn)<norm(r); nodes=nn; wt=nw; r=rn; μ=max(μ/3,1e-14) else; μ=min(μ*3,1e6) end
        norm(r,Inf)<1e-13 && break
    end
    nodes,wt
end
nodes,wt=lm_refine(nodes,wt)

# --- 4. validation -----------------------------------------------------------------
r,mrec=resid(nodes,wt)
act=findall(>(1e-12*ρ),wt)
@printf("\nREFINED cubature: effective atoms=%d  |r|_std_∞=%.2e  |r|_std_2=%.2e\n", length(act), norm(r,Inf), norm(r))
@printf("  raw rel residual = %.2e   min weight/ρ = %.2e   max CFL coord = %.3f (limit 1)\n",
        maximum(abs.(mrec.-M)./max.(abs.(M),1e-30)), minimum(wt[act])/ρ, maximum(cfl(nodes[b]) for b in act))
@printf("  componentwise standardized 4th-moment errors:\n")
for (nm,idx) in sort(collect(D4);by=x->x[2]); @printf("     %s: %.2e\n", nm, abs(mrec[idx]-M[idx])/S[idx]); end
mgn=realizability_margin(mrec)
@printf("  reproduced-state cone margin = %.3e (%s);  Jacobian cond = %.1e\n", mgn, mgn>=0 ? "in-cone" : "OUT", cond(jac(nodes,wt)))
@printf("VERDICT: %s\n", norm(r,Inf)<1e-10 && maximum(cfl(nodes[b]) for b in act)<1 && minimum(wt[act])>1e-10*ρ ?
        "EXACT bounded CFL-safe cubature (machine-precision, no tiny-weight cancellation)" :
        norm(r,Inf)<1e-6 ? "bounded CFL-safe but not yet machine (refine more / better seed)" : "did not converge")

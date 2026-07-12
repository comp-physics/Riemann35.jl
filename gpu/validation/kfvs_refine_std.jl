# kfvs_refine_std.jl — exact bounded CFL-safe cubature for ONE class-C target, done RIGHT:
# work in CENTERED, variance-scaled coordinates ũ=(v−mean)/σ (targets = standardized central
# moments s, O(1)–O(650) but cancellation-free); grid NNLS witness → Carathéodory compression
# to ≤35 atoms → Levenberg–Marquardt in ũ to machine precision; physical CFL constraint on
# v=mean+σũ. Then the full validation battery.
using Riemann35, JLD2, Printf, LinearAlgebra
d=load("gpu/validation/kfvs_defect_counterexample.jld2"); λ=d["lam"][1]; C=d["center_state"]; cls=d["class"]
TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
IDX=Dict(TRIP[n]=>n for n in 1:35)
cfl(v)=λ*(abs(v[1])+abs(v[2])+abs(v[3]))
bin(n,k)= k<0||k>n ? 0.0 : Float64(binomial(n,k))
pmono(u)=ntuple(n->(i=TRIP[n]; u[1]^i[1]*u[2]^i[2]*u[3]^i[3]),35)
D4=("m400"=>5,"m040"=>15,"m004"=>25,"m220"=>12,"m202"=>22,"m022"=>35)

s=findfirst(==(3),cls); M=collect(C[:,s]); ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
# central per-unit-mass moments μ via binomial shift of raw m/ρ
μ=zeros(35)
for n in 1:35; (i,j,k)=TRIP[n]; acc=0.0
    for p in 0:i, q in 0:j, r in 0:k
        haskey(IDX,(p,q,r)) || continue
        acc+= bin(i,p)*bin(j,q)*bin(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end; μ[n]=acc
end
σx=sqrt(max(μ[3],1e-14));σy=sqrt(max(μ[10],1e-14));σz=sqrt(max(μ[20],1e-14))
sstd=[μ[n]/(σx^TRIP[n][1]*σy^TRIP[n][2]*σz^TRIP[n][3]) for n in 1:35]     # standardized target (s_000=1,s_200=1)
Sc=[max(abs(sstd[n]),1.0) for n in 1:35]
@printf("class-C: mean(%.1f,%.1f,%.1f) σ(%.2g,%.2g,%.2g); std kurtosis(%.0f,%.0f,%.0f) → support ~%.1fσ (cfl of that ≈%.3f)\n",
        ux,uy,uz,σx,σy,σz, sstd[5],sstd[15],sstd[25], sstd[5]^0.25, cfl((ux+σx*sstd[5]^0.25,uy,uz)))

# --- grid witness in ũ (reproduce standardized moments) ---
K=7.0; ng=13; gs=range(-K,K;length=ng)
grid=[[a,b,c] for a in gs for b in gs for c in gs if cfl((ux+σx*a,uy+σy*b,uz+σz*c))≤1.0]
Φ=reduce(hcat,[collect(pmono(u)) for u in grid])./Sc; tgt=sstd./Sc
function fista(Φ,b;it=20000)
    w=zeros(size(Φ,2));wp=copy(w);t=1.0;v=randn(length(w));for _ in 1:25;v=Φ'*(Φ*v);v./=norm(v);end;η=1/norm(Φ*(v./norm(v)))^2
    for _ in 1:it;y=w.+((t-1)/t).*(w.-wp);wp=copy(w);w=max.(y.-η.*(Φ'*(Φ*y.-b)),0.0);t=(1+sqrt(1+4t^2))/2;end;w
end
w=fista(Φ,tgt)
@printf("grid witness (std): %d atoms, residual %.2e\n", count(>(1e-9),w), norm(Φ*w.-tgt)/norm(tgt))

# --- Carathéodory compression to ≤35 atoms (exact; preserves Φw) ---
function caratheodory(nodes,w)
    keep=findall(>(1e-9),w); nodes=nodes[keep]; w=w[keep]
    while length(w)>35
        A=reduce(hcat,[collect(pmono(u)) for u in nodes])   # 35×n
        z=nullspace(A); isempty(z) && break; z=z[:,1]
        # move w along z to zero an atom, keep w≥0
        α=Inf; kill=0
        for b in eachindex(w); z[b]>1e-14 && (r=w[b]/z[b]; r<α && (α=r;kill=b)); z[b]<-1e-14 && (r=-w[b]/z[b]; r<α && (α=r;kill=b)); end
        kill==0 && break
        w=w.-α.*z; keep2=setdiff(1:length(w),kill); nodes=nodes[keep2]; w=w[keep2]; w=max.(w,0.0)
    end
    nodes,w
end
nodes,wt=caratheodory(grid,w)
@printf("after Carathéodory: %d atoms\n", length(wt))

# --- LM refine in ũ to machine precision ---
resid(nd,w)=(m=zeros(35);for b in eachindex(w),n in 1:35;m[n]+=w[b]*pmono(nd[b])[n];end;((m.-sstd)./Sc,m))
function jac(nd,w); Kn=length(w);J=zeros(35,4Kn)
    for b in 1:Kn,n in 1:35;(i,j,k)=TRIP[n];u=nd[b]
        J[n,b]=(u[1]^i*u[2]^j*u[3]^k)/Sc[n]
        J[n,Kn+b]  = i==0 ? 0.0 : w[b]*i*u[1]^(i-1)*u[2]^j*u[3]^k/Sc[n]
        J[n,2Kn+b] = j==0 ? 0.0 : w[b]*j*u[1]^i*u[2]^(j-1)*u[3]^k/Sc[n]
        J[n,3Kn+b] = k==0 ? 0.0 : w[b]*k*u[1]^i*u[2]^j*u[3]^(k-1)/Sc[n]
    end;J
end
function lm(nodes,wt;δ=1e-3,it=600); Kn=length(wt);μd=1e-3;r,_=resid(nodes,wt)
    for _ in 1:it; J=jac(nodes,wt);step=-(J'J+μd*I)\(J'*r)
        nw=[max(wt[b]+step[b],0.0) for b in 1:Kn]
        nn=[nodes[b].+[step[Kn+b],step[2Kn+b],step[3Kn+b]] for b in 1:Kn]
        for b in 1:Kn; c=cfl((ux+σx*nn[b][1],uy+σy*nn[b][2],uz+σz*nn[b][3])); c>1-δ && (nn[b].*=(1-δ)/c); end
        rn,_=resid(nn,nw); norm(rn)<norm(r) ? (nodes=nn;wt=nw;r=rn;μd=max(μd/3,1e-15)) : (μd=min(μd*3,1e8))
        norm(r,Inf)<1e-13 && break
    end; nodes,wt
end
nodes,wt=lm(nodes,wt)

# --- validation (map back to physical + raw moments) ---
r,srec=resid(nodes,wt); act=findall(>(1e-12),wt)
vphys=[[ux+σx*nodes[b][1],uy+σy*nodes[b][2],uz+σz*nodes[b][3]] for b in act]
mraw=zeros(35); for (bi,b) in enumerate(act),n in 1:35; mraw[n]+=ρ*wt[b]*pmono(vphys[bi])[n]; end
@printf("\nREFINED (standardized): atoms=%d  |r_std|_∞=%.2e  raw rel resid=%.2e\n",
        length(act), norm(r,Inf), maximum(abs.(mraw.-M)./max.(abs.(M),1e-30)))
@printf("  min weight = %.2e (Σw=1)   max CFL coord = %.3f (limit 1)\n", minimum(wt[act]), maximum(cfl(v) for v in vphys))
@printf("  standardized 4th-moment errors:"); for (nm,idx) in D4; @printf(" %s=%.1e",nm,abs(srec[idx]-sstd[idx])/Sc[idx]); end; println()
mgn=realizability_margin(mraw); @printf("  reproduced raw-state cone margin=%.3e (%s)  Jac cond=%.1e\n", mgn, mgn>=0 ? "in-cone":"OUT", cond(jac(nodes,wt)))
@printf("VERDICT: %s\n", norm(r,Inf)<1e-10 && maximum(cfl(v) for v in vphys)<1 && minimum(wt[act])>1e-6 ?
        "EXACT bounded CFL-safe cubature (machine, no tiny-weight cancellation)" :
        norm(r,Inf)<1e-6 ? "bounded CFL-safe, near-machine (tighten)" : "did not converge")

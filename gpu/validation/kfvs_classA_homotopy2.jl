# kfvs_classA_homotopy2.jl — EXPERIMENT 1 (alternating continuation). Morph the CHyQMOM seed cubature
# to represent the TRUE moments M along s(t)=F0 + t*(sstd-F0), t:0->1, by ALTERNATING:
#   (W) weights  = NNLS(Phi(X), s(t))                      -> stays >=0, well-conditioned
#   (X) positions = trust-region min-norm Gauss-Newton step, then RADIAL-PROJECT each atom onto the
#                   kinetic CFL ball  λ(|ux|+|uy|+|uz|)<=1   (so nodes move continuously but never leave CFL)
# Accept a t-step only when the inner loop drives the row-scaled residual below tol AND the config is
# feasible; else shrink h. Success = reach t=1, BigFloat-verified <1e-6, w>=0, CFL<=1. Env: sdp_env.
using JLD2, Printf, LinearAlgebra, NonNegLeastSquares
setprecision(BigFloat, 512)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
mono(X,n)= X[1]^TRIP[n][1]*X[2]^TRIP[n][2]*X[3]^TRIP[n][3]
function gmono(X,n); (i,j,k)=TRIP[n]
    (i==0 ? 0.0 : i*X[1]^(i-1)*X[2]^j*X[3]^k, j==0 ? 0.0 : j*X[1]^i*X[2]^(j-1)*X[3]^k, k==0 ? 0.0 : k*X[1]^i*X[2]^j*X[3]^(k-1)); end
cfl(X,mu,sig,λ)=λ*(abs(mu[1]+sig[1]*X[1])+abs(mu[2]+sig[2]*X[2])+abs(mu[3]+sig[3]*X[3]))
function cfl_rmax(e,mu,sig,λ); g(r)=cfl((r*e[1],r*e[2],r*e[3]),mu,sig,λ); g(0.0)>1.0 && return 0.0
    hi=1.0; while g(hi)<=1.0 && hi<1e7; hi*=2; end; lo=0.0; for _ in 1:64; m=(lo+hi)/2;(g(m)<=1.0 ? lo=m : hi=m);end; lo; end
Phi(X)= [mono(X[a],n) for n in 1:35, a in eachindex(X)]
Fmom(w,X)=[sum(w[a]*mono(X[a],n) for a in eachindex(w)) for n in 1:35]
# radial projection of atom X onto CFL ball (scale toward origin; origin=mean is assumed inside CFL)
function proj_cfl(X,mu,sig,λ)
    cfl(X,mu,sig,λ)<=1.0 && return X
    lo=0.0; hi=1.0                      # find largest α with cfl(αX)<=1
    for _ in 1:60; m=(lo+hi)/2; (cfl((m*X[1],m*X[2],m*X[3]),mu,sig,λ)<=1.0 ? lo=m : hi=m); end
    (lo*X[1],lo*X[2],lo*X[3])
end

function homotopy2(sd,ccol; K=25)
    mu=sd["mu_$ccol"];sig=sd["sig_$ccol"];λ=sd["lam"];sstd=sd["sstd_$ccol"]
    Xs=sd["X_$ccol"];ws=sd["w_$ccol"];Na0=length(ws)
    # augmentation: one atom per direction at 0.6*rmax (positions will move under GN)
    DIRS=[[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.],[1,1,1.],[1,1,-1.],[1,-1,1.],[-1,1,1.],[1,-1,-1.],[-1,1,-1.],[-1,-1,1.],[-1,-1,-1.],[1,1,0.],[1,-1,0.],[1,0,1.],[1,0,-1.],[0,1,1.],[0,1,-1.]]
    X=[(Xs[1,a],Xs[2,a],Xs[3,a]) for a in 1:Na0]; w=copy(ws)
    for d in DIRS; e=d./norm(d); rm=cfl_rmax(e,mu,sig,λ); rm<=1e-9 && continue
        push!(X,(0.6rm*e[1],0.6rm*e[2],0.6rm*e[3])); push!(w,1e-9); end
    Na=length(w)
    rsc=[1.0/max(abs(sstd[n]),1.0) for n in 1:35]
    F0=Fmom(w,X); d=sstd.-F0
    tgt(t)=F0.+t.*d
    function nnls_w(X,s)
        Φ=Phi(X).*rsc; b=s.*rsc
        cn=[max(norm(@view Φ[:,j]),1e-12) for j in 1:size(Φ,2)]; Φn=Φ./cn'
        wn=nonneg_lsq(Φn,b;alg=:nnls)[:,1]; wn./cn
    end
    function posstep!(w,X,s,tr)
        r=rsc.*(Fmom(w,X).-s)
        J=zeros(35,3Na); for a in 1:Na; g=[gmono(X[a],n) for n in 1:35]
            for n in 1:35; J[n,3(a-1)+1]=rsc[n]*w[a]*g[n][1];J[n,3(a-1)+2]=rsc[n]*w[a]*g[n][2];J[n,3(a-1)+3]=rsc[n]*w[a]*g[n][3]; end; end
        cs=[1.0/max(norm(@view J[:,j]),1e-12) for j in 1:3Na]; Js=J.*cs'
        δ=cs.*(Js'*((Js*Js'+1e-8*I)\(-r)))
        for a in 1:Na; v=(δ[3(a-1)+1],δ[3(a-1)+2],δ[3(a-1)+3]); nv=norm(v); s_=nv>tr ? tr/nv : 1.0
            Xn=(X[a][1]+s_*v[1],X[a][2]+s_*v[2],X[a][3]+s_*v[3]); X[a]=proj_cfl(Xn,mu,sig,λ); end
    end
    t=0.0; h=0.04; tr=0.4; nstep=0
    while t<1.0-1e-12 && nstep<4000
        nstep+=1; tn=min(t+h,1.0); Xsav=copy(X); wsav=copy(w); s=tgt(tn); conv=false; best=Inf
        for it in 1:K
            w=nnls_w(X,s); posstep!(w,X,s,tr)
            w=nnls_w(X,s); rr=norm(rsc.*(Fmom(w,X).-s),Inf); best=min(best,rr)
            if rr<5e-10; conv=true; break; end
        end
        if conv
            t=tn; h=min(h*1.35,0.15); tr=min(tr*1.1,0.6)
        else
            X=Xsav; w=wsav; h*=0.5; tr=max(tr*0.6,0.02); h<1e-6 && break
        end
    end
    w=nnls_w(X,sstd.*(t>=1-1e-9 ? 1.0 : 0.0).+tgt(t).*(t>=1-1e-9 ? 0.0 : 1.0))  # final weights at reached t
    # BigFloat verify at reached t vs its own target
    star=tgt(t); wB=big.(max.(w,0.0))
    mx=big(0.0); for n in 1:35;(i,j,k)=TRIP[n]; val=sum(wB[a]*big(X[a][1])^i*big(X[a][2])^j*big(X[a][3])^k for a in 1:Na)
        mx=max(mx,abs(val-star[n])*rsc[n]); end
    # verify vs the TRUE sstd (only meaningful if t~1)
    mxT=big(0.0); for n in 1:35;(i,j,k)=TRIP[n]; val=sum(wB[a]*big(X[a][1])^i*big(X[a][2])^j*big(X[a][3])^k for a in 1:Na)
        mxT=max(mxT,abs(val-sstd[n])*rsc[n]); end
    (t, Float64(mxT), Na, minimum(w), maximum(cfl(X[a],mu,sig,λ) for a in 1:Na), count(>(1e-9),w))
end

sd=load(joinpath(@__DIR__,"kfvs_classA_seeds.jld2")); cols=sd["cols"]
smalls=[c for c in cols if norm(sd["defect_$c"])<1.0]; larges=[c for c in cols if norm(sd["defect_$c"])>=1.0]
sel = length(ARGS)>=1 ? (ARGS[1]=="all" ? cols : parse.(Int,ARGS)) : vcat(smalls[1:3], larges[1:3])
@printf("class-A alternating homotopy on %d cells:\n", length(sel)); ns=0; oks=0;okl=0
for c in sel
    reg = norm(sd["defect_$c"])<1.0 ? "small" : "LARGE"
    t,mxT,Na,minw,maxc,nnz=homotopy2(sd,c)
    good = (t>=1.0-1e-9 && mxT<1e-6 && minw>=-1e-12 && maxc<=1.0+1e-9)
    good && (ns+=1; reg=="small" ? (global oks+=1) : (global okl+=1))
    @printf("  col%-4d [%-5s] %-16s t=%.4f verifyM=%.2e minw=%.2e maxCFL=%.4f nnz=%d/%d\n",
            c,reg, good ? "SUCCESS" : (t>=1-1e-9 ? "reached-inexact" : "UNREACHED"), t,mxT,minw,maxc,nnz,Na); flush(stdout)
end
@printf("\n=== EXPERIMENT 1 (class-A alternating homotopy): %d/%d SUCCESS", ns, length(sel))
length(ARGS)>=1 && ARGS[1]=="all" && @printf(" (small %d/%d, large %d/%d)", oks,length(smalls),okl,length(larges))
@printf(" ===\n")

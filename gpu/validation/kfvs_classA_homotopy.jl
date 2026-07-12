# kfvs_classA_homotopy.jl — EXPERIMENT 1 (reviewer two-shot): continuous defect homotopy for class A.
# The CHyQMOM seed nodes exactly represent their own reconstructed moments M̃ (t=0). We track
#   s(t) = F(y0) + t*(sstd - F(y0)),  t:0->1,
# moving BOTH weights and node positions by a Levenberg min-norm Newton predictor-corrector, while
# enforcing w>=0 and the 8-fold kinetic CFL box  λ(|ux|+|uy|+|uz|) <= 1  on every physical node.
# Augmentation: extra atoms seeded (tiny weight) along principal directions, each clamped to its
# CFL-feasible radius — so the homotopy can only build tails that FIT inside the CFL ball. If the
# required tail exceeds the CFL ball, the step stalls and the cell is reported UNREACHED (a decisive
# negative). Success is BigFloat-verified at t=1. No Riemann35 needed. Env: sdp_env (LinearAlgebra+JLD2).
using JLD2, Printf, LinearAlgebra
setprecision(BigFloat, 512)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
mono(X,n)= X[1]^TRIP[n][1]*X[2]^TRIP[n][2]*X[3]^TRIP[n][3]
function gmono(X,n)  # gradient of monomial n wrt (X1,X2,X3)
    (i,j,k)=TRIP[n]
    gx = i==0 ? 0.0 : i*X[1]^(i-1)*X[2]^j*X[3]^k
    gy = j==0 ? 0.0 : j*X[1]^i*X[2]^(j-1)*X[3]^k
    gz = k==0 ? 0.0 : k*X[1]^i*X[2]^j*X[3]^(k-1)
    (gx,gy,gz)
end
Fmom(w,X)= [sum(w[a]*mono(X[a],n) for a in eachindex(w)) for n in 1:35]
cfl(X,mu,sig,λ)=λ*(abs(mu[1]+sig[1]*X[1])+abs(mu[2]+sig[2]*X[2])+abs(mu[3]+sig[3]*X[3]))
# largest radius r>=0 along unit std-direction e with cfl(r*e)<=1 (bisection; g increases ~linearly in r)
function cfl_rmax(e,mu,sig,λ)
    g(r)=cfl((r*e[1],r*e[2],r*e[3]),mu,sig,λ)
    g(0.0)>1.0 && return 0.0
    hi=1.0; while g(hi)<=1.0 && hi<1e6; hi*=2; end
    lo=0.0; for _ in 1:60; mid=(lo+hi)/2; (g(mid)<=1.0 ? lo=mid : hi=mid); end; lo
end

function build_aug(mu,sig,λ,supp)
    dirs=[[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.],
          [1,1,0.],[1,-1,0.],[-1,1,0.],[1,0,1.],[1,0,-1.],[-1,0,1.],[0,1,1.],[0,1,-1.],[0,-1,1.],
          [1,1,1.],[1,1,-1.],[1,-1,1.],[-1,1,1.],[1,-1,-1.],[-1,1,-1.],[-1,-1,1.],[-1,-1,-1.]]
    Xaug=NTuple{3,Float64}[]
    for d in dirs; e=d./norm(d); rmax=cfl_rmax(e,mu,sig,λ); rmax<=1e-9 && continue
        for f in (0.45,0.8); r=min(f*rmax, f*supp*1.5)   # place inside CFL ball, scaled toward the needed support
            push!(Xaug,(r*e[1],r*e[2],r*e[3])); end
    end
    Xaug
end

# Levenberg min-norm Newton continuation for one cell
function homotopy_cell(sd, ccol)
    mu=sd["mu_$ccol"]; sig=sd["sig_$ccol"]; λ=sd["lam"]; sstd=sd["sstd_$ccol"]
    Xs=sd["X_$ccol"]; ws=sd["w_$ccol"]; Na0=length(ws)
    supp=sqrt(max(sstd[25],sstd[15],sstd[5],1.0))
    Xaug=build_aug(mu,sig,λ,supp); eps0=1e-8
    X=vcat([ (Xs[1,a],Xs[2,a],Xs[3,a]) for a in 1:Na0], Xaug)
    w=vcat(copy(ws), fill(eps0,length(Xaug)))
    Na=length(w)
    y0F=Fmom(w,X); target=sstd; d=target.-y0F           # s(t)=y0F + t*d ; s(0)=F(y0) exactly
    rsc=[1.0/max(abs(target[n]),1.0) for n in 1:35]      # row (moment) scaling
    pack(w,X)=vcat(w,[X[a][c] for a in 1:Na for c in 1:3])
    function unpack(y); w=y[1:Na]; X=[(y[Na+3(a-1)+1],y[Na+3(a-1)+2],y[Na+3(a-1)+3]) for a in 1:Na]; (w,X); end
    function jac(w,X)   # 35 x 4Na  (weights block | positions block), row-scaled
        J=zeros(35,4Na)
        for a in 1:Na, n in 1:35; J[n,a]=rsc[n]*mono(X[a],n); end
        for a in 1:Na; for n in 1:35; g=gmono(X[a],n)
            J[n,Na+3(a-1)+1]=rsc[n]*w[a]*g[1]; J[n,Na+3(a-1)+2]=rsc[n]*w[a]*g[2]; J[n,Na+3(a-1)+3]=rsc[n]*w[a]*g[3]; end; end
        J
    end
    resid(w,X,t)= rsc.*(Fmom(w,X).-(y0F.+t.*d))
    feasible(w,X)= all(>=(-1e-13),w) && all(cfl(X[a],mu,sig,λ)<=1.0+1e-9 for a in 1:Na)
    # column-scaled Levenberg-Marquardt corrector to the t=tn target; returns (best_y, best_feasible_resid)
    function correct(ystart,tn; tol=1e-9, maxit=80)
        y=copy(ystart); best=copy(y); bestr=Inf; μ=1e-6
        for it in 1:maxit
            w,X=unpack(y); r=resid(w,X,tn); nr=norm(r,Inf)
            if feasible(w,X) && nr<bestr; bestr=nr; best=copy(y); end
            nr<tol && break
            J=jac(w,X); cs=[1.0/max(norm(@view J[:,j]),1e-12) for j in 1:size(J,2)]
            Js=J.*cs'
            δ=cs.*(Js'*((Js*Js'+μ*I)\(-r)))
            wt,Xt=unpack(y.+δ); nrt=norm(resid(wt,Xt,tn),Inf)
            if nrt<nr; y.+=δ; μ=max(μ*0.4,1e-11); else; μ=min(μ*5,1e8); end
        end
        (best,bestr)
    end
    t=0.0; h=0.03; y=pack(w,X); nstep=0
    while t<1.0-1e-12 && nstep<6000
        nstep+=1; tn=min(t+h,1.0); yprev=copy(y)
        w,X=unpack(y); J=jac(w,X)
        cs=[1.0/max(norm(@view J[:,j]),1e-12) for j in 1:size(J,2)]; Js=J.*cs'
        δ=cs.*(Js'*((Js*Js'+1e-8*I)\(rsc.*(h.*d)))); y.+=δ     # column-scaled predictor
        y,br=correct(y,tn)
        w,X=unpack(y)
        if br<1e-8 && feasible(w,X)
            t=tn; h=min(h*1.4,0.15)
        else
            y=yprev; h*=0.5
            h<1e-6 && break
        end
    end
    w,X=unpack(y)
    # BigFloat verify at reached t (should be ~1)
    wB=big.(max.(w,0.0)); XB=[big.(collect(X[a])) for a in 1:Na]
    mx=big(0.0)
    for n in 1:35; (i,j,k)=TRIP[n]; val=sum(wB[a]*XB[a][1]^i*XB[a][2]^j*XB[a][3]^k for a in 1:Na)
        mx=max(mx, abs(val-target[n])*rsc[n]); end
    minw=minimum(w); maxc=maximum(cfl(X[a],mu,sig,λ) for a in 1:Na)
    (t, Float64(mx), Na, minw, maxc, count(>(1e-9),w))
end

sd=load(joinpath(@__DIR__,"kfvs_classA_seeds.jld2"))
# choose cells: reviewer wants difficult failures; span the bimodality (small-defect + large z-kurtosis)
cols=sd["cols"]
smalls=[c for c in cols if norm(sd["defect_$c"])<1.0]
larges=[c for c in cols if norm(sd["defect_$c"])>=1.0]
sel = length(ARGS)>=1 ? parse.(Int,ARGS) : vcat(smalls[1:2], larges[1:3])
@printf("class-A homotopy on %d cells (small-defect + large z-kurtosis mix):\n", length(sel))
res=[]
for c in sel
    reg = norm(sd["defect_$c"])<1.0 ? "small" : "LARGE"
    t,mx,Na,minw,maxc,nnz=homotopy_cell(sd,c)
    st = (t>=1.0-1e-9 && mx<1e-6 && minw>=-1e-12) ? "SUCCESS" : (t>=1.0-1e-9 ? "reached-inexact" : "UNREACHED")
    push!(res,(c,reg,st,t,mx,minw,maxc,nnz))
    @printf("  col%-4d [%-5s] %-16s t_final=%.4f verify=%.2e minw=%.2e maxCFL=%.4f nnz=%d/%d\n",
            c,reg,st,t,mx,minw,maxc,nnz,Na); flush(stdout)
end
ns=count(r->r[3]=="SUCCESS",res)
@printf("\n=== EXPERIMENT 1 (class-A homotopy): %d/%d SUCCESS (t=1, BigFloat-verified, positive, CFL-safe) ===\n", ns, length(res))

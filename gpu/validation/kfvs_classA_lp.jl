# kfvs_classA_lp.jl — EXPERIMENT 1 (robust variant): weights-only positive-cubature LP for class A on a
# candidate cloud placed along the ACTUAL defect axes at CFL-clamped radii (dense radial sampling ~
# "continuous placement" approximated). Distinct from the failed uniform grid-LP: candidates are seed
# CHyQMOM nodes UNION rays {r*e} for e in axes/diagonals, r in fractions of that ray's CFL-max radius.
# LP: min sum(u) s.t. Phi_std*u = sstd, u>=0  (HiGHS simplex, exact). BigFloat-verified. Env: sdp_env.
using JLD2, Printf, LinearAlgebra, JuMP, HiGHS
setprecision(BigFloat, 512)
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
mono(X,n)= X[1]^TRIP[n][1]*X[2]^TRIP[n][2]*X[3]^TRIP[n][3]
cfl(X,mu,sig,λ)=λ*(abs(mu[1]+sig[1]*X[1])+abs(mu[2]+sig[2]*X[2])+abs(mu[3]+sig[3]*X[3]))
function cfl_rmax(e,mu,sig,λ)
    g(r)=cfl((r*e[1],r*e[2],r*e[3]),mu,sig,λ); g(0.0)>1.0 && return 0.0
    hi=1.0; while g(hi)<=1.0 && hi<1e7; hi*=2; end
    lo=0.0; for _ in 1:64; mid=(lo+hi)/2; (g(mid)<=1.0 ? lo=mid : hi=mid); end; lo
end
const DIRS = begin
    D=[[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.],
       [1,1,0.],[1,-1,0.],[-1,1,0.],[-1,-1,0.],[1,0,1.],[1,0,-1.],[-1,0,1.],[-1,0,-1.],[0,1,1.],[0,1,-1.],[0,-1,1.],[0,-1,-1.],
       [1,1,1.],[1,1,-1.],[1,-1,1.],[-1,1,1.],[1,-1,-1.],[-1,1,-1.],[-1,-1,1.],[-1,-1,-1.]]
    [d./norm(d) for d in D]
end
const FRACS=(0.12,0.24,0.36,0.48,0.6,0.72,0.84,0.93)

function candidates(mu,sig,λ; seedX=nothing)
    C=NTuple{3,Float64}[]; push!(C,(0.0,0.0,0.0))
    if seedX!==nothing; for a in 1:size(seedX,2); push!(C,(seedX[1,a],seedX[2,a],seedX[3,a])); end; end
    for e in DIRS; rmax=cfl_rmax(e,mu,sig,λ); rmax<=1e-9 && continue
        for f in FRACS; r=f*rmax; push!(C,(r*e[1],r*e[2],r*e[3])); end
    end
    unique(C)
end
function solve_lp(cands,sstd)
    Nn=length(cands)
    Φ=Array{Float64}(undef,35,Nn); for (jn,X) in enumerate(cands),n in 1:35; Φ[n,jn]=mono(X,n); end
    rowsc=[1.0/max(abs(sstd[n]),1.0) for n in 1:35]; Φr=Φ.*rowsc; br=sstd.*rowsc
    cn=[max(norm(@view Φr[:,j]),1e-300) for j in 1:Nn]; Φn=Φr./cn'
    lp=Model(HiGHS.Optimizer); set_silent(lp); set_attribute(lp,"solver","simplex")
    @variable(lp,u[1:Nn]>=0); @constraint(lp,Φn*u.==br); @objective(lp,Min,sum(u)); optimize!(lp)
    has_values(lp) || return (false,Float64[],NTuple{3,Float64}[])
    w=max.(value.(u),0.0)./cn; act=findall(>(1e-14),w)
    (true, w[act], cands[act])
end
function verifyBF(w,X,sstd)
    wB=big.(w); mx=big(0.0)
    for n in 1:35;(i,j,k)=TRIP[n]; val=sum(wB[a]*big(X[a][1])^i*big(X[a][2])^j*big(X[a][3])^k for a in eachindex(w))
        mx=max(mx,abs(val-sstd[n])/max(abs(sstd[n]),1.0)); end
    Float64(mx)
end

sd=load(joinpath(@__DIR__,"kfvs_classA_seeds.jld2")); cols=sd["cols"]; λ=sd["lam"]
mode = length(ARGS)>=1 && ARGS[1]=="all" ? :all : :sample
smalls=[c for c in cols if norm(sd["defect_$c"])<1.0]; larges=[c for c in cols if norm(sd["defect_$c"])>=1.0]
sel = mode==:all ? cols : vcat(smalls[1:3], larges[1:4])
@printf("class-A weights-LP on %d cells (candidate rays along defect axes, CFL-clamped):\n", length(sel))
ok_small=0;ok_large=0;ns=0
for c in sel
    reg = norm(sd["defect_$c"])<1.0 ? "small" : "LARGE"
    cands=candidates(sd["mu_$c"],sd["sig_$c"],λ; seedX=sd["X_$c"])
    feas,w,X=solve_lp(cands,sd["sstd_$c"])
    v = feas ? verifyBF(w,X,sd["sstd_$c"]) : Inf
    good = feas && v<1e-6 && all(>=(0.0),w)
    good && (ns+=1; reg=="small" ? (global ok_small+=1) : (global ok_large+=1))
    maxc = feas && !isempty(X) ? maximum(cfl(x,sd["mu_$c"],sd["sig_$c"],λ) for x in X) : NaN
    mode==:all || @printf("  col%-4d [%-5s] %-8s natoms=%3d verify=%.2e maxCFL=%.4f ncand=%d\n",
            c,reg, good ? "OK" : (feas ? "inexact" : "infeas"), length(w), v, maxc, length(cands))
    good || (mode==:all && @printf("  col%-4d [%-5s] FAIL verify=%.2e\n",c,reg,v))
    flush(stdout)
end
@printf("\n=== EXPERIMENT 1 (class-A weights-LP): %d/%d OK", ns, length(sel))
mode==:all && @printf("  (small %d/%d, large %d/%d)", ok_small,length(smalls),ok_large,length(larges))
@printf(" ===\n")

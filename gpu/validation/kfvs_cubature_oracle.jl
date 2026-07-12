# kfvs_cubature_oracle.jl — the narrow question: does each pre-anchor realizable state admit
# an EXACT positive atomic representation of its 35 moments with all nodes inside the kinetic
# CFL polytope?  Decisive output = the DISTRIBUTION of R*/R_CFL by class (not binary feasibility).
#
# Exact CFL coordinate of a node v (isotropic grid dx=dy=dz): c(v) = λ(|vx|+|vy|+|vz|), λ=dt/dx.
# The measure update weight is n·(1 − c(v)); nonneg ⇔ c(v) ≤ 1, so R_CFL = 1 (c_SSP=1).
# R* = inf{ R : the 35 moments have a positive representing measure supported in {c(v) ≤ R} }.
# We UPPER-bound R* by an R-ladder: the smallest clip radius R at which a candidate grid clipped
# to {c(v) ≤ R} reproduces the moments (NNLS residual ≈ 0) — a witnessed feasible measure ⊂ K_R.
# (A fixed grid proves feasibility, not infeasibility; R*>1 here is a suggestive obstruction to
#  confirm later with continuous optimization + an SOS/compact-support dual.)
# Solve centered + variance-scaled; verify raw-moment residual on the returned nodes.
using JLD2, Printf, LinearAlgebra, Statistics
d=load("gpu/validation/kfvs_defect_counterexample.jld2")
λ=d["lam"][1]; C=d["center_state"]; cls=d["class"]; NS=size(C,2)
TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),
 (2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),
 (1,1,2),(0,1,3),(0,2,2))
cfl(vx,vy,vz)=λ*(abs(vx)+abs(vy)+abs(vz))

function nnls(Φ,b;iters=25000)   # FISTA (accelerated projected gradient), Φ matvecs only
    n=size(Φ,2); w=zeros(n); wp=copy(w); t=1.0
    v=randn(n); for _ in 1:25; v=Φ'*(Φ*v); v./=norm(v); end
    L=norm(Φ*(v./norm(v)))^2; η=1.0/max(L,1e-30)
    for _ in 1:iters
        y=w.+((t-1)/t).*(w.-wp); wp=copy(w)
        w=max.(y .- η.*(Φ'*(Φ*y .- b)), 0.0); t=(1+sqrt(1+4t^2))/2
    end
    w
end

# One state: reproduce on a generous CFL-clipped grid (centered, variance-scaled with the
# NATURAL moment scale ρ·σ^α). Return (R* = maxCFL of active atoms, residual, active, minw).
function Rstar(M; K=7.0, ng=17)
    ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
    σ(a2,u)=sqrt(max(M[a2]/ρ-u^2,1e-10)); σx=σ(3,ux);σy=σ(10,uy);σz=σ(20,uz)
    gs=range(-K,K;length=ng)
    keep=[(ux+σx*a,uy+σy*b,uz+σz*c) for a in gs for b in gs for c in gs if cfl(ux+σx*a,uy+σy*b,uz+σz*c) ≤ 1.0]
    isempty(keep) && return (Inf,NaN,0,0.0)
    Φ=Array{Float64}(undef,35,length(keep))
    for (bi,(a,b,c)) in enumerate(keep), n in 1:35; (i,j,k)=TRIP[n]; Φ[n,bi]=a^i*b^j*c^k; end
    scale=[ρ*σx^TRIP[n][1]*σy^TRIP[n][2]*σz^TRIP[n][3] for n in 1:35]
    Φs=Φ./scale; w=nnls(Φs, M./scale)
    resid=norm(Φs*w .- M./scale)/max(norm(M./scale),1e-30)
    act=findall(>(1e-10*ρ),w)
    Rst=isempty(act) ? Inf : maximum(cfl(keep[bi]...) for bi in act)
    (Rst, resid, length(act), isempty(act) ? 0.0 : minimum(w[act]))
end

@printf("=== bounded positive-cubature oracle over %d states (R_CFL=1; c(v)=λΣ|v|, λ=%.4g) ===\n", NS, λ)
Rs=Dict(1=>Float64[],2=>Float64[],3=>Float64[]); rr=Dict(1=>Float64[],2=>Float64[],3=>Float64[]); at=Dict(1=>Int[],2=>Int[],3=>Int[])
t0=time()
for s in 1:NS
    R,res,na,mw = Rstar(C[:,s])
    push!(Rs[cls[s]], R); push!(rr[cls[s]], res); push!(at[cls[s]], na)
    s%20==0 && (@printf("  ...%d/%d (%.0fs)\n",s,NS,time()-t0); flush(stdout))
end
@printf("(reproduced = NNLS residual < 1e-3 on the CFL-clipped grid; R* = maxCFL of active atoms, R_CFL=1)\n")
for (c,nm) in ((1,"A"),(2,"B"),(3,"C"))
    R=Rs[c]; isempty(R) && continue
    repro=findall(<(1e-3), rr[c])                      # states the grid actually reproduced
    Rok=[R[i] for i in repro]; cflsafe=count(<(1.0),Rok)
    @printf("class %s (n=%d): reproduced %d/%d (residual<1e-3);  of those, R*<1 (CFL-safe) %d/%d\n",
            nm, length(R), length(repro), length(R), cflsafe, length(repro))
    isempty(Rok) || @printf("   R*/R_CFL over reproduced: median %.3f  p90 %.3f  max %.3f   active-atoms median %d   residual median %.1e\n",
            median(Rok), quantile(Rok,0.9), maximum(Rok), Int(round(median(at[c][repro]))), median(rr[c][repro]))
end

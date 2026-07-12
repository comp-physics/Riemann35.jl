# kfvs_enrichment_oracle.jl — source-level repair feasibility: does a BOUNDED positive
# quadrature reproduce the same 35 moments as the pathological CHyQMOM cloud?
#
# For the saved class-A/C representative cell states, build a scale-aware candidate grid
# (nodes at mean ± K·σ per axis — the physical velocity scale, NOT CHyQMOM's far atoms),
# row-scale the 35 moment equations (kills the u⁴~1e7 conditioning), and solve a
# non-negative least squares  min‖Φw − M‖ , w ≥ 0  (projected gradient, dependency-free).
# Report: residual (feasibility), active-atom count, max node Σ|u| (achieved bounded speed)
# vs the CFL limit 1/λ and CHyQMOM's node speed. Residual≈0 with Σ|u|<CFL ⇒ enrichment viable.
using JLD2, Printf, LinearAlgebra
d=load("gpu/validation/kfvs_defect_counterexample.jld2")
λ=d["lam"][1]; cfl=1/λ
TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),
 (2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),
 (1,1,2),(0,1,3),(0,2,2))

function nnls_pg(Φ, b; iters=200000, tol=1e-14)
    n=size(Φ,2); w=zeros(n); ΦtΦ=Φ'Φ; Φtb=Φ'b
    L=opnorm(ΦtΦ); η=1.0/L
    for it in 1:iters
        g=ΦtΦ*w .- Φtb
        w=max.(w .- η.*g, 0.0)
        it%5000==0 && (norm(g[w.>0])<tol && break)
    end
    w
end

function oracle(nm; K=6.0, ngrid=13)
    M=d["rep_$(nm)_center"]; ρ=M[1]
    ux=M[2]/ρ; uy=M[6]/ρ; uz=M[16]/ρ
    σx=sqrt(max(M[3]/ρ-ux^2,1e-12)); σy=sqrt(max(M[10]/ρ-uy^2,1e-12)); σz=sqrt(max(M[20]/ρ-uz^2,1e-12))
    gs=range(-K,K;length=ngrid)
    nodes=[(ux+σx*a, uy+σy*b, uz+σz*c) for a in gs for b in gs for c in gs]
    Nn=length(nodes)
    Φ=Array{Float64}(undef,35,Nn)
    for (bi,(a,b,c)) in enumerate(nodes), n in 1:35
        (i,j,k)=TRIP[n]; Φ[n,bi]=a^i*b^j*c^k
    end
    scale=[max(abs(M[n]),1e-30) for n in 1:35]     # row-scale each moment equation
    Φs=Φ./scale; Ms=M./scale
    w=nnls_pg(Φs, Ms)
    resid=norm(Φs*w .- Ms)/max(norm(Ms),1e-30)
    act=findall(>(1e-10*ρ), w)
    maxsu=isempty(act) ? 0.0 : maximum(abs(nodes[bi][1])+abs(nodes[bi][2])+abs(nodes[bi][3]) for bi in act)
    @printf("class %s: grid %d^3=%d cand, K=%.0f  σ=(%.2g,%.2g,%.2g)\n", nm, ngrid, Nn, K, σx,σy,σz)
    @printf("   NNLS rel residual = %.2e  (≈0 ⇒ bounded positive quadrature EXISTS)\n", resid)
    @printf("   active atoms = %d   max node Σ|u| = %.1f   (CFL limit %.0f, CHyQMOM ~%s)\n",
            length(act), maxsu, cfl, nm=="C" ? "3079" : "173")
    @printf("   VERDICT: %s\n", resid<1e-6 && maxsu<cfl ? "FEASIBLE — bounded CFL-safe positive quadrature reproduces the 35 moments" :
            resid<1e-6 ? "reproduces moments but exceeds CFL" : "grid cannot reproduce (refine/expand or genuinely infeasible)")
end

@printf("=== bounded positive-quadrature oracle (CFL limit Σ|u| ≤ %.0f) ===\n", cfl)
oracle("A")
oracle("C")

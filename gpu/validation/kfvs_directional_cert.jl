# kfvs_directional_cert.jl — cheap RIGOROUS obstruction test for one class-C target.
# Localizing-matrix directional lower bound on required support:
#   z=(1,X1,X2,X3); H0=E[zzᵀ]; H_ℓ=E[(ℓᵀX)² zzᵀ];  any measure with |ℓᵀX|≤R obeys R²H0−H_ℓ⪰0,
#   so R_min(ℓ)² = λ_max(H_ℓ,H0)  (rigorous, uses ALL degree-4 moments incl. cross-4th).
# R_CFL(ℓ) = max|ℓᵀX| over the physical CFL polytope (X=(v−mean)/σ, Σ|v|≤1/λ).
# If R_min(ℓ) > R_CFL(ℓ) for ANY ℓ ⇒ bounded CFL enrichment impossible (clean N5 obstruction).
# All standardized moments in BigFloat.
using JLD2, Printf, LinearAlgebra
setprecision(BigFloat,256)
d=load("gpu/validation/kfvs_defect_counterexample.jld2"); λ=d["lam"][1]; C=d["center_state"]; cls=d["class"]
TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)= k<0||k>n ? big(0) : big(binomial(n,k))

s=findfirst(==(3),cls); M=big.(collect(C[:,s])); ρ=M[1]; ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
# standardized central moments (BigFloat), s_000=1, s_200=1
sstd=Dict{NTuple{3,Int},BigFloat}()
for n in 1:35;(i,j,k)=TRIP[n];acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
    sstd[TRIP[n]]=acc/(σx^i*σy^j*σz^k)
end
sm(t)=get(sstd,t,big(0.0))                 # standardized moment of exponent triple t
add(a...)=ntuple(d->sum(x[d] for x in a),3)
E=((0,0,0),(1,0,0),(0,1,0),(0,0,1))         # z-vector component exponents (index 1..4)

function Rmin(ℓ)                             # ℓ = 3-vector for (X1,X2,X3)
    H0=[Float64(sm(add(E[i],E[j]))) for i in 1:4, j in 1:4]
    Hl=zeros(4,4)
    for i in 1:4, j in 1:4, a in 2:4, b in 2:4
        Hl[i,j]+=ℓ[a-1]*ℓ[b-1]*Float64(sm(add(E[a],E[b],E[i],E[j])))
    end
    μ=eigvals(Symmetric(Hl),Symmetric(H0))  # generalized eigenvalues
    sqrt(max(maximum(real,μ),0.0))
end
mean=[Float64(ux),Float64(uy),Float64(uz)]; σ=[Float64(σx),Float64(σy),Float64(σz)]
Rcfl(ℓ)= (1/λ)*maximum(abs(ℓ[i]/σ[i]) for i in 1:3) + abs(sum(ℓ[i]*mean[i]/σ[i] for i in 1:3))

@printf("class-C target: σ≈%.2g  std kurtosis≈%.0f\n", σ[1], Float64(sm((4,0,0))))
@printf("%-16s %-10s %-10s %-8s\n","direction ℓ","R_min","R_CFL","ratio")
dirs=Vector{Vector{Float64}}()
for v in ([1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]); push!(dirs,v./norm(v)); end
# dense sphere sample
import Random; Random.seed!(1)
for _ in 1:4000; g=randn(3); push!(dirs, g./norm(g)); end
worst=(-Inf,zeros(3),0.0,0.0)
for ℓ in dirs
    rm=Rmin(ℓ); rc=Rcfl(ℓ); rt=rm/rc
    rt>worst[1] && (global worst=(rt,ℓ,rm,rc))
end
# print the named directions + the worst
for v in ([1,0,0],[1,1,0],[1,1,1],[1,-1,0])
    ℓ=v./norm(v); @printf("%-16s %-10.3f %-10.1f %-8.4f\n", string(v), Rmin(ℓ), Rcfl(ℓ), Rmin(ℓ)/Rcfl(ℓ))
end
@printf("WORST over %d directions: ℓ≈(%.2f,%.2f,%.2f)  R_min=%.3f  R_CFL=%.1f  ratio=%.4f\n",
        length(dirs), worst[2]..., worst[3], worst[4], worst[1])
@printf("VERDICT: %s\n", worst[1]>1.0 ?
        "OBSTRUCTION — R_min > R_CFL in some direction ⇒ bounded CFL enrichment PROVABLY IMPOSSIBLE (clean N5)" :
        "no obstruction found (max ratio < 1) ⇒ feasibility OPEN; the LP/SQP solver is justified")

# kfvs_polish.jl — BigFloat minimum-norm weight polish with nodes FIXED.
# Solve A dw = m - A w  (35 x 75, underdetermined) for the least-norm dw, then check w+dw>0.
# Drives the 9.65e-11 standardized-moment residual toward ~1e-13 without moving nodes (CFL untouched).
using JLD2, Printf, LinearAlgebra
setprecision(BigFloat, 512)
const D=load("gpu/validation/kfvs_defect_counterexample.jld2")
const C=D["center_state"]; const CLS=D["class"]
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)=(k<0||k>n) ? big(0) : big(binomial(n,k))
s0i=findfirst(==(3),CLS); M=big.(collect(C[:,s0i])); ρ=M[1]
ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
sstd=zeros(BigFloat,35)
for n in 1:35
    (i,j,k)=TRIP[n]; acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
    sstd[n]=acc/(σx^i*σy^j*σz^k)
end
sol=load("gpu/validation/kfvs_assembled_solution.jld2"); atoms=sol["atoms_std"]; w=big.(sol["weights"]); nA=length(w)
# moment matrix A (35 x n), BigFloat
A=Array{BigFloat}(undef,35,nA)
for j in 1:nA, n in 1:35; (i,a,b)=TRIP[n]; A[n,j]=big(atoms[j][1])^i*big(atoms[j][2])^a*big(atoms[j][3])^b; end
r0=sstd .- A*w
@printf("pre-polish: max |A w - m| = %.3e   min w = %.3e\n", Float64(maximum(abs,r0)), Float64(minimum(w)))
# min-norm exact correction dw = A' (A A')^{-1} r0
G=A*A'; dw=A'*(G\r0); wnew=w.+dw
r1=sstd .- A*wnew
@printf("post-polish: max |A w - m| = %.3e   min w = %.3e   max|dw|=%.3e\n",
        Float64(maximum(abs,r1)), Float64(minimum(wnew)), Float64(maximum(abs,dw)))
if minimum(wnew) > 0 && maximum(abs,r1) < maximum(abs,r0)
    save("gpu/validation/kfvs_polished_solution.jld2","atoms_std",atoms,"weights",Float64.(wnew),
         "weightsBF",wnew,"maxerr",Float64(maximum(abs,r1)))
    @printf("==> polished weights saved (all positive, residual %.2e)\n", Float64(maximum(abs,r1)))
else
    println("==> polish introduced a negative weight or no improvement; keep original (run bounded QP).")
end

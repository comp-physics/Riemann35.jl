# kfvs_jet_audit.jl — reconcile the S=-47 / kurtosis inconsistency (reviewer step 1).
# Save exact E[S^k] k=0..4 (BigFloat), the affine def of S, implied physical velocity, support bound.
# Then build an exact positive 3-atom Gauss-Radau S-quadrature with a PRESCRIBED tail node (swept),
# matching m0..m4, requiring positive weights + CFL-admissible nodes.
using JLD2, Printf, LinearAlgebra
setprecision(BigFloat, 512)
const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM=D["lam"][1]; const C=D["center_state"]; const CLS=D["class"]
const TRIP=((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX=Dict(TRIP[n]=>n for n in 1:35); bino(n,k)=(k<0||k>n) ? big(0) : big(binomial(n,k))
s0i=findfirst(==(3),CLS); M=big.(collect(C[:,s0i])); ρ=M[1]
ux=M[2]/ρ;uy=M[6]/ρ;uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2);σy=sqrt(M[10]/ρ-uy^2);σz=sqrt(M[20]/ρ-uz^2)
sm=Dict{NTuple{3,Int},BigFloat}()
for n in 1:35
    (i,j,k)=TRIP[n]; acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k; haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ); end
    sm[TRIP[n]]=acc/(σx^i*σy^j*σz^k)
end
smf(t)=get(sm,t,big(0.0))
# E[S^deg], S=(X1+X2+X3)/sqrt3
s3=sqrt(big(3)); c=(1/s3,1/s3,1/s3)
function ES(deg)
    tot=big(0.0)
    for a in 0:deg,b in 0:(deg-a); cc=deg-a-b
        mult=factorial(big(deg))÷(factorial(big(a))*factorial(big(b))*factorial(big(cc)))
        tot+=mult*c[1]^a*c[2]^b*c[3]^cc*smf((a,b,cc)); end
    tot
end
mS=[ES(k) for k in 0:4]
@printf("=== JET-AXIS AUDIT  S=(X1+X2+X3)/sqrt3, X_i=(v_i-mu_i)/sigma_i ===\n")
@printf("physical mu=(%.5f,%.5f,%.5f) sigma=(%.5f,%.5f,%.5f)\n",Float64(ux),Float64(uy),Float64(uz),Float64(σx),Float64(σy),Float64(σz))
for k in 0:4; @printf("  E[S^%d] = %.6f\n",k,Float64(mS[k+1])); end
varS=mS[3]-mS[2]^2; skew=(mS[4]-3*mS[2]*varS-mS[2]^3)/varS^big(1.5); kurt=mS[5]/mS[3]^2
@printf("  var_S=E[S^2]-E[S]^2 = %.6f   (E[S]=%.3e)\n",Float64(varS),Float64(mS[2]))
@printf("  RAW kurtosis E[S^4]/E[S^2]^2 = %.2f ; central kurt = %.2f\n",Float64(mS[5]/mS[3]^2),Float64((mS[5])/varS^2))
supp=sqrt(mS[5]/mS[3])
@printf("  SUPPORT BOUND  max|S| >= sqrt(E[S^4]/E[S^2]) = %.3f  (NOT sqrt(3174)=56.3; matches dir-cert (1,1,1) R_min=65.1)\n",Float64(supp))
# physical velocity of an S=s, T=0 node: v_i = mu_i + sigma_i*(s/sqrt3)
vof(s)= (Float64(ux+σx*(s/s3)), Float64(uy+σy*(s/s3)), Float64(uz+σz*(s/s3)))
cflv(s)= Float64(LAM)*sum(abs,vof(s))
@printf("  node at S=-65: v=(%.2f,%.2f,%.2f) lambda*sum|v|=%.3f  |  node at S=-47: v=(%.2f,%.2f,%.2f) l*s|v|=%.3f\n",
        vof(-65.0)...,cflv(-65.0),vof(-47.0)...,cflv(-47.0))

# 2-atom Gauss (m0..m3) — SHOW it undershoots m4 (why -47 was an artifact)
m0,m1,m2,m3,m4=mS
a0=m1/m0; b1=m2/m0-a0^2; a1=(m3-2a0*m2+a0^2*m1)/(b1*m0)
J2=[Float64(a0) sqrt(Float64(b1));sqrt(Float64(b1)) Float64(a1)]; Eg=eigen(Symmetric(J2))
n2=Eg.values; w2=[Float64(m0)*Eg.vectors[1,j]^2 for j in 1:2]
@printf("\n2-atom Gauss (m0..m3): S=%.3f(w=%.5f), S=%.3f(w=%.5f) ; predicts E[S^4]=%.1f vs actual %.1f (UNDERSHOOT => tail farther out)\n",
        n2[1],w2[1],n2[2],w2[2], sum(w2[j]*n2[j]^4 for j in 1:2), Float64(m4))

# existence: 1D S-Hankel M_2 = [m_{i+j}]_{i,j=0..2} must be PSD for ANY representing S-measure
H = [Float64(mS[i+j+1]) for i in 0:2, j in 0:2]
evH = sort(eigvals(Symmetric(H)); rev=true)
@printf("\n1D S-Hankel M_2 eigs: %s  (PSD => S-measure exists; rank=%d)\n",
        join((@sprintf("%.3e",e) for e in evH)," "), count(>(1e-10*evH[1]),evH))

# 3-atom family matching m0..m4 parametrized by free m5 (Prony): solve cubic coeffs, roots=nodes.
function threeatom(m5)
    A=[Float64(mS[1]) Float64(mS[2]) Float64(mS[3]);
       Float64(mS[2]) Float64(mS[3]) Float64(mS[4]);
       Float64(mS[3]) Float64(mS[4]) Float64(mS[5])]
    rhs=[Float64(mS[4]),Float64(mS[5]),m5]
    c=A\rhs                                   # nodes are roots of x^3 - c3 x^2 - c2 x - c1
    comp=[0.0 0.0 c[1]; 1.0 0.0 c[2]; 0.0 1.0 c[3]]
    rt=eigvals(comp); (any(abs.(imag.(rt)).>1e-6*maximum(abs,real.(rt).+1e-9))) && return nothing
    nodes=sort(real.(rt))
    V=[nodes[j]^k for k in 0:2, j in 1:3]; w=V\[Float64(mS[1]),Float64(mS[2]),Float64(mS[3])]
    (nodes,w)
end
@printf("\n3-atom family sweep over free m5 (find POSITIVE + CFL member):\n")
best=nothing
for m5 in range(-6e5, -1e4; length=120)
    r=threeatom(m5); r===nothing && continue
    nodes,w=r
    if all(>(-1e-9), w) && all(s->cflv(s)<=1.0, nodes)
        # verify moments
        err=maximum(abs(sum(w[j]*big(nodes[j])^k for j in 1:3)-mS[k+1]) for k in 0:4)
        (best===nothing) && (global best=(m5,nodes,w,Float64(err)))
        @printf("  m5=%.2e: nodes=%s w=%s minw=%.2e CFLmax=%.3f momErr=%.1e <--OK\n",
                m5, join((@sprintf("%.2f",x) for x in nodes),","), join((@sprintf("%.5f",x) for x in w),","),
                minimum(w), maximum(cflv.(nodes)), Float64(err))
    end
end
best===nothing && println("  no positive+CFL 3-atom member found in swept m5 range (try 5-node / wider sweep)")
save("gpu/validation/kfvs_jet_moments.jld2","ES",Float64.(mS),"support",Float64(supp),
     "mu",[Float64(ux),Float64(uy),Float64(uz)],"sigma",[Float64(σx),Float64(σy),Float64(σz)])
println("\nsaved E[S^k] + geometry -> kfvs_jet_moments.jld2")

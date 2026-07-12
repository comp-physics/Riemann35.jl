# kfvs_jet_marginal.jl — 1D jet-marginal truncated moment problem (reviewer step 3).
# Rotate to principal jet axis S=(X1+X2+X3)/sqrt3 (mean is along (1,1,1)); compute S-marginal
# standardized moments m_k=E[S^k], k=0..4 (all available from degree<=4 data), then:
#   - 2-atom Gauss quadrature from m0..m3 (well-conditioned, 1D);
#   - compare predicted m4 to actual -> the tail/counter-stream signature;
#   - Hankel positivity/rank to decide 2 vs 3 atoms.
# Also report transverse (T1,T2) second/fourth structure to guide the 3D lift. All BigFloat.

using JLD2, Printf, LinearAlgebra
setprecision(BigFloat, 256)
const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const C = D["center_state"]; const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))

s0 = findfirst(==(3), CLS)
M = big.(collect(C[:, s0])); ρ = M[1]
ux=M[2]/ρ; uy=M[6]/ρ; uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2); σy=sqrt(M[10]/ρ-uy^2); σz=sqrt(M[20]/ρ-uz^2)
sstd = Dict{NTuple{3,Int},BigFloat}()
for n in 1:35
    (i,j,k)=TRIP[n]; acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k
        haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end
    sstd[TRIP[n]]=acc/(σx^i*σy^j*σz^k)
end
sm(t) = get(sstd, t, big(0.0))
@printf("physical mean=(%.4f,%.4f,%.4f)  sigma=(%.4f,%.4f,%.4f)\n",
        Float64(ux),Float64(uy),Float64(uz),Float64(σx),Float64(σy),Float64(σz))

# ---- moment of a linear form L = c1 X1 + c2 X2 + c3 X3:  E[L^deg]  (deg<=4) ----
function form_moment(c, deg)
    tot = big(0.0)
    for a in 0:deg, b in 0:(deg-a)
        cc = deg - a - b
        mult = factorial(big(deg)) ÷ (factorial(big(a))*factorial(big(b))*factorial(big(cc)))
        tot += mult * c[1]^a * c[2]^b * c[3]^cc * sm((a,b,cc))
    end
    tot
end

s3 = sqrt(big(3.0))
eS = (1/s3, 1/s3, 1/s3)                       # jet axis (1,1,1)/sqrt3
eT1 = (1/sqrt(big(2.0)), -1/sqrt(big(2.0)), big(0.0))   # transverse 1
eT2 = (1/sqrt(big(6.0)), 1/sqrt(big(6.0)), -2/sqrt(big(6.0)))  # transverse 2

mS = [form_moment(eS, k) for k in 0:4]
@printf("\nJET marginal S=(X1+X2+X3)/sqrt3 standardized moments:\n")
@printf("  m0=%.4f m1=%.4e m2=%.4f m3=%.4e m4=%.4f\n",
        Float64(mS[1]),Float64(mS[2]),Float64(mS[3]),Float64(mS[4]),Float64(mS[5]))
@printf("  (var_S=%.4f  skew=%.4f  kurt=%.2f)  vs transverse var: T1=%.4f T2=%.4f\n",
        Float64(mS[3]), Float64(mS[4]/mS[3]^1.5), Float64(mS[5]/mS[3]^2),
        Float64(form_moment(eT1,2)), Float64(form_moment(eT2,2)))
@printf("  transverse kurt: T1 m4=%.2f  T2 m4=%.2f\n",
        Float64(form_moment(eT1,4)), Float64(form_moment(eT2,4)))

# ---- 2-atom Gauss quadrature of S from m0..m3 (Golub-Welsch via 2x2 Jacobi) ----
# recurrence: alpha0=m1/m0 ; beta1=m2/m0-alpha0^2 ; alpha1=(m3 - 2 alpha0 m2 + alpha0^2 m1)/(beta1 m0) - alpha0 ... use Hankel directly
m0,m1,m2,m3,m4 = mS
# nodes = eigenvalues of Jacobi J = [[a0, sqrt(b1)],[sqrt(b1), a1]]
a0 = m1/m0
b1 = m2/m0 - a0^2
# a1 from moment m3: for orthonormal poly p1=(x-a0), <x p1,p1>/<p1,p1>
#   <p1,p1> = b1*m0 ; <x p1, p1> = E[x (x-a0)^2] = m3 - 2 a0 m2 + a0^2 m1
a1 = (m3 - 2*a0*m2 + a0^2*m1)/(b1*m0)
J2 = [Float64(a0) sqrt(Float64(b1)); sqrt(Float64(b1)) Float64(a1)]
Eg = eigen(Symmetric(J2)); nodesS = Eg.values
wS = [Float64(m0)*Eg.vectors[1,j]^2 for j in 1:2]   # Golub-Welsch weights = m0 * (first comp)^2
predm4 = sum(wS[j]*nodesS[j]^4 for j in 1:2)
@printf("\n2-atom Gauss (from m0..m3):\n")
for j in 1:2
    @printf("  atom %d: S=%.4f  weight=%.4f  (physical along jet: |v|~%.1f from mean)\n",
            j, nodesS[j], wS[j], abs(nodesS[j])*Float64((σx+σy+σz)/3)*sqrt(3))
end
@printf("  predicted m4 (2-atom) = %.4f   vs ACTUAL m4 = %.4f   EXCESS = %.4f\n",
        predm4, Float64(m4), Float64(m4)-predm4)
# Hankel 3x3 positivity/rank -> 2 vs 3 atoms
H3 = [Float64(mS[i+j+1]) for i in 0:2, j in 0:2]
ev = sort(eigvals(Symmetric(H3)); rev=true)
@printf("  Hankel3 eigs: %s  -> rank=%d (%s)\n",
        join((@sprintf("%.3e",e) for e in ev), " "), count(>(1e-8*ev[1]), ev),
        count(>(1e-8*ev[1]),ev) >= 3 ? "needs >=3 S-atoms (bulk + counter-stream tail)" : "2 S-atoms suffice")


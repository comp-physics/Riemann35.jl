# kfvs_structured_cubature.jl — constructive CFL-safe positive cubature from a JET-ALIGNED
# structured node cloud (reviewer step 3->4 lift), NNLS for positive weights, BigFloat-verified.
#
# Geometry learned from the 1D jet marginal:
#   * near-Gaussian bulk (transverse var~0.83, jet var~1.3) at the mean;
#   * ONE-SIDED counter-stream tail along -S = -(X1+X2+X3)/sqrt3, at S~-47 (skew -40.6).
# Build: bulk tensor set in (S,T1,T2) + a tail cluster along -S (asymmetric, NOT ±symmetric).
# Rotate each (S,T1,T2) node to standardized X, keep CFL-feasible, NNLS-fit the 35 standardized
# moments (degree-row-scaled + column-normalized), verify every moment in BigFloat.
#
# usage: julia ... kfvs_structured_cubature.jl [nb=6] [Stail_max=70] [ntail=8]

using JLD2, Printf, LinearAlgebra, JuMP, HiGHS
setprecision(BigFloat, 256)
NB    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6      # bulk nodes per axis
STAILMAX = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 70.0
NTAIL = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8      # tail S-levels along -S

const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM = D["lam"][1]; const C = D["center_state"]; const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))

s0 = findfirst(==(3), CLS)
M = big.(collect(C[:, s0])); ρ = M[1]
ux=M[2]/ρ; uy=M[6]/ρ; uz=M[16]/ρ
σx=sqrt(M[3]/ρ-ux^2); σy=sqrt(M[10]/ρ-uy^2); σz=sqrt(M[20]/ρ-uz^2)
sstdB = zeros(BigFloat,35)
for n in 1:35
    (i,j,k)=TRIP[n]; acc=big(0.0)
    for p in 0:i,q in 0:j,r in 0:k
        haskey(IDX,(p,q,r))||continue
        acc+=bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end
    sstdB[n]=acc/(σx^i*σy^j*σz^k)
end
sstd = Float64.(sstdB)
R = 1/LAM; mean=(Float64(ux),Float64(uy),Float64(uz)); sig=(Float64(σx),Float64(σy),Float64(σz))

# rotation: standardized X = S*eS + T1*eT1 + T2*eT2
eS=(1/sqrt(3),1/sqrt(3),1/sqrt(3)); eT1=(1/sqrt(2),-1/sqrt(2),0.0); eT2=(1/sqrt(6),1/sqrt(6),-2/sqrt(6))
Xof(S,T1,T2) = (S*eS[1]+T1*eT1[1]+T2*eT2[1], S*eS[2]+T1*eT1[2]+T2*eT2[2], S*eS[3]+T1*eT1[3]+T2*eT2[3])
cflok(X) = LAM*(abs(mean[1]+sig[1]*X[1])+abs(mean[2]+sig[2]*X[2])+abs(mean[3]+sig[3]*X[3])) <= 1.0
σS = sqrt(1.336); σT = sqrt(0.832)

# ---- structured node cloud ----
nodes = NTuple{3,Float64}[]
# bulk: tensor in (S,T1,T2), centered at mean, spanning ~±3σ
bs = range(-3.0, 3.0; length=NB)
for hs in bs, h1 in bs, h2 in bs
    X = Xof(σS*hs, σT*h1, σT*h2)
    cflok(X) && push!(nodes, X)
end
nbulk = length(nodes)
# tail: ONE-SIDED along -S, each S-level lifted into a transverse cluster (conditional cov / cross-4th).
# Dense sampling across the counter-stream region [-STAILMAX,-6] (marginal puts the atom at S~-47),
# transverse 5x5 at ±2σT. NTAIL levels -> fine S-resolution so a positive combination can land.
ttv = (-2.0,-1.0,0.0,1.0,2.0)
for Sl in range(-STAILMAX, -6.0; length=NTAIL), t1 in ttv, t2 in ttv
    X = Xof(Sl, σT*t1, σT*t2)
    cflok(X) && push!(nodes, X)
end
Nn = length(nodes)
@printf("nodes: bulk=%d tail=%d total=%d  (σS=%.3f σT=%.3f)\n", nbulk, Nn-nbulk, Nn, σS, σT)

# ---- degree-row-scaled + column-normalized NNLS ----
Φ = Array{Float64}(undef, 35, Nn)
for (jn,X) in enumerate(nodes), n in 1:35
    (i,jj,k)=TRIP[n]; Φ[n,jn] = X[1]^i * X[2]^jj * X[3]^k
end
Ls = sqrt(sqrt(sstd[5]))                        # ~support scale for degree row-scaling
rowsc = [1.0/Ls^sum(TRIP[n]) for n in 1:35]
Φr = Φ .* rowsc; br = sstd .* rowsc             # row-scaled equalities (all O(1))
cn = [max(norm(@view Φr[:,j]),1e-300) for j in 1:Nn]   # column-normalize (far X^4 cols ~1e6 else IPM stalls)
Φn = Φr ./ cn'
# EXACT LP in normalized coords: find u>=0 with Φn u = br; recover w = u ./ cn.
lp = Model(HiGHS.Optimizer); set_silent(lp)
set_attribute(lp, "solver", "simplex")          # exact basic feasible vertex, or Farkas cert
set_attribute(lp, "presolve", "on")
@variable(lp, uv[1:Nn] >= 0)
@constraint(lp, Φn * uv .== br)
@objective(lp, Min, sum(uv))
optimize!(lp)
stat = termination_status(lp)
if !has_values(lp)
    @printf("LP status=%s : node cone does NOT contain target (infeasible) -> need column generation.\n", stat)
    w = zeros(Nn)
else
    w = max.(value.(uv), 0.0) ./ cn             # clamp any tiny IPM negativity, then unscale
    relmax = maximum(abs(dot(Φ[n,:],w)-sstd[n])*rowsc[n] for n in 1:35)
    @printf("LP status=%s  active=%d  max per-moment rel err(F64)=%.3e  sum w=%.6f  minw=%.3e\n",
            stat, count(>(1e-12), w), relmax, sum(w), minimum(w))
end
act = findall(>(1e-12), w)
# DEBUG: m400 (n=5) computed three ways to locate the Float64-vs-BigFloat discrepancy
let n=5
    f64_all = dot(Φ[n,:], w)
    bf_all  = Float64(sum(big(w[j])*big(nodes[j][1])^big(4) for j in 1:Nn if w[j]>0))
    bf_act  = Float64(sum(big(w[j])*big(nodes[j][1])^big(4) for j in act))
    @printf("DEBUG m400: Float64(all)=%.4f  BigFloat(all w>0)=%.4f  BigFloat(act>1e-12)=%.4f  target=%.4f  #(w>0)=%d #act=%d\n",
            f64_all, bf_all, bf_act, sstd[n], count(>(0.0), w), length(act))
end

# ---- BigFloat verification ----
function verify(w, act, nodes)
    wB=big.(w); maxerr=big(0.0); worst=0
    for n in 1:35
        (i,jj,k)=TRIP[n]; val=big(0.0)
        for j in act; X=nodes[j]; val+=wB[j]*big(X[1])^i*big(X[2])^jj*big(X[3])^k; end
        e=abs(val-sstdB[n]); e>maxerr && (maxerr=e; worst=n)
    end
    (Float64(maxerr), worst)
end
maxerr, worst = verify(w, act, nodes)
cflmax = maximum(LAM*(abs(mean[1]+sig[1]*nodes[j][1])+abs(mean[2]+sig[2]*nodes[j][2])+abs(mean[3]+sig[3]*nodes[j][3])) for j in act)
@printf("VERIFY(BigFloat): max std-moment resid=%.3e (worst %s) | CFL λΣ|v|max=%.4f | minw=%.3e\n",
        maxerr, TRIP[worst], cflmax, minimum(w[act]))
if maxerr < 1e-4 && cflmax <= 1.0+1e-9
    println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (jet-aligned structured cloud).")
    save("gpu/validation/kfvs_structured_solution.jld2",
         "nodes_std",[collect(nodes[j]) for j in act],"weights",w[act],
         "mean",collect(mean),"sig",collect(sig),"R",R,"maxerr",maxerr)
else
    @printf("==> residual %.2e on %s — adjust tail placement/density.\n", maxerr, TRIP[worst])
end

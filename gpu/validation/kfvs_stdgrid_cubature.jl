# kfvs_stdgrid_cubature.jl — constructive CFL-safe positive cubature via a WIDE STANDARDIZED grid.
#
# Fixes the bug that made the old ±7sigma physical-coord oracle a mean-shift-cancellation artifact:
#   the class-C measure needs support out to ~sqrt(m400) ~ 25.6 sigma, so a ±7sigma grid CANNOT
#   place mass where it belongs and fakes the RAW moments by cancellation.  Here we grid in
#   STANDARDIZED coords X=(v-mu)/sigma out to ±K sigma (K~30), keep only CFL-feasible nodes
#   (physical sum|v|<=R), and NNLS-fit the 35 STANDARDIZED moments (all O(1), well-conditioned).
#   Verify the recovered cubature reproduces every standardized moment in BigFloat.
#
# usage: julia ... kfvs_stdgrid_cubature.jl [K=30] [ngrid=41]

using JLD2, Printf, LinearAlgebra
setprecision(BigFloat, 256)
K   = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 30.0
NG  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 41

const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM = D["lam"][1]; const C = D["center_state"]; const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))

s0 = findfirst(==(3), CLS)
M = big.(collect(C[:, s0])); ρ = M[1]
ux = M[2]/ρ; uy = M[6]/ρ; uz = M[16]/ρ
σx = sqrt(M[3]/ρ - ux^2); σy = sqrt(M[10]/ρ - uy^2); σz = sqrt(M[20]/ρ - uz^2)
sstdB = zeros(BigFloat, 35)
for n in 1:35
    (i,j,k) = TRIP[n]; acc = big(0.0)
    for p in 0:i, q in 0:j, r in 0:k
        haskey(IDX,(p,q,r)) || continue
        acc += bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end
    sstdB[n] = acc/(σx^i*σy^j*σz^k)
end
sstd = Float64.(sstdB)
R = 1/LAM
mean = (Float64(ux),Float64(uy),Float64(uz)); sig = (Float64(σx),Float64(σy),Float64(σz))
@printf("class-C: mean=(%.3g,%.3g,%.3g) sig=(%.3g,%.3g,%.3g) R=%.1f  grid K=±%.0fσ ng=%d\n",
        mean..., sig..., R, K, NG)
@printf("target std: var=%.4f m400=%.1f m220=%.1f => support ~%.1fσ\n",
        sstd[3], sstd[5], sstd[12], sqrt(sstd[5]))

# ---- build standardized grid, keep CFL-feasible nodes ----
gs = range(-K, K; length=NG)
cfl(X) = LAM*(abs(mean[1]+sig[1]*X[1]) + abs(mean[2]+sig[2]*X[2]) + abs(mean[3]+sig[3]*X[3]))
nodes = NTuple{3,Float64}[]
for a in gs, b in gs, c in gs
    cfl((a,b,c)) <= 1.0 && push!(nodes, (a,b,c))
end
Nn = length(nodes)
@printf("nodes in CFL diamond: %d / %d\n", Nn, NG^3)

# ---- Φ (35 x Nn) standardized monomials; NNLS fit to sstd (y0=1 => mass normalized) ----
Φ = Array{Float64}(undef, 35, Nn)
for (j,X) in enumerate(nodes), n in 1:35
    (i,jj,k) = TRIP[n]; Φ[n,j] = X[1]^i * X[2]^jj * X[3]^k
end
# FISTA projected-gradient NNLS: min ||Φw - sstd||^2 s.t. w>=0
function nnls_fista(A, b; iters=40000)
    w = zeros(size(A,2)); wp = copy(w); t = 1.0
    v = randn(size(A,2)); for _ in 1:30; v = A'*(A*v); v ./= norm(v); end
    η = 1 / norm(A*(v./norm(v)))^2
    for _ in 1:iters
        y = w .+ ((t-1)/t).*(w .- wp); wp = copy(w)
        w = max.(y .- η.*(A'*(A*y .- b)), 0.0)
        t = (1 + sqrt(1+4t^2))/2
    end
    w
end
# (1) degree-based row scale so all moment targets are O(1) [support scale Ls]; monomial X^a has
#     natural magnitude Ls^|a|, so scale row n by 1/Ls^deg(n).  (2) column-normalize Φ (fixes the
#     far-node domination that stalls NNLS).  Solve for u, unscale w = u ./ colnorm.
Ls = sqrt(sqrt(sstd[5]))                       # ~ support scale (m400^{1/4} ≈ 5.06 std units)... use 26σ box
Ls = max(Ls, sqrt(sstd[5]))                    # =25.6, the true support radius
rowsc = [1.0 / Ls^sum(TRIP[n]) for n in 1:35]
Φr = Φ .* rowsc; br = sstd .* rowsc
cn = [max(norm(@view Φr[:,j]), 1e-300) for j in 1:Nn]
Φhat = Φr ./ cn'
u = nnls_fista(Φhat, br)
w = u ./ cn
res = norm(Φ*w .- sstd) / norm(sstd)
relmoments = maximum(abs(dot(Φ[n,:], w) - sstd[n]) * rowsc[n] for n in 1:35)
act = findall(>(1e-12), w)
@printf("NNLS: active nodes=%d  rel-L2 resid=%.3e  max per-moment rel err=%.3e  sum w=%.6f\n",
        length(act), res, relmoments, sum(w))

# ---- BigFloat verification of ALL 35 standardized moments ----
function verify_bigfloat(w, act, nodes)
    wB = big.(w); maxerr = big(0.0); worst = 0
    for n in 1:35
        (i,jj,k) = TRIP[n]; val = big(0.0)
        for j in act; X = nodes[j]; val += wB[j]*big(X[1])^i*big(X[2])^jj*big(X[3])^k; end
        e = abs(val - sstdB[n]); e > maxerr && (maxerr = e; worst = n)
    end
    (maxerr, worst)
end
maxerr, worst = verify_bigfloat(w, act, nodes)
cflmax = maximum(LAM*(abs(mean[1]+sig[1]*nodes[j][1])+abs(mean[2]+sig[2]*nodes[j][2])+abs(mean[3]+sig[3]*nodes[j][3])) for j in act)
@printf("VERIFY (BigFloat): max |std-moment residual| = %.3e (worst %s)\n", Float64(maxerr), TRIP[worst])
@printf("CFL: max λ·sum|v| over active nodes = %.4f (<=1 required)\n", cflmax)
@printf("POSITIVITY: min weight = %.3e  (>=0 by construction)\n", minimum(w[act]))
if Float64(maxerr) < 1e-4 && cflmax <= 1.0 + 1e-9
    println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (standardized moments matched).")
    println("    Confirms the SDP no-obstruction result with explicit nodes+weights.")
    save("gpu/validation/kfvs_stdgrid_solution.jld2",
         "nodes_std", [collect(nodes[j]) for j in act], "weights", w[act],
         "mean", collect(mean), "sig", collect(sig), "R", R, "maxerr", Float64(maxerr))
else
    @printf("==> residual %.2e — increase K or ng (support may exceed ±%.0fσ) .\n", Float64(maxerr), K)
end

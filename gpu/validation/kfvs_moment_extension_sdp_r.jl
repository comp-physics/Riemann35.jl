# kfvs_moment_extension_sdp_r.jl — general-order moment-extension SDP + flat-extension scan
# + constructive atom extraction (Henrion-Lasserre) on ONE class-C state.
#
# Order-r Lasserre relaxation of the truncated K-moment problem on the CFL diamond
# K = { X : sum_i |mu_i+sigma_i X_i| <= R }, 8 affine facets g_s.
#   unknown moments y_alpha, |alpha| <= 2r ; deg<=4 FIXED to measured std moments; rest free.
#   M_r(y) >= 0                 (moment matrix, monomials deg<=r)
#   M_{r-1}(g_s . y) >= 0 all s (localizing, monomials deg<=r-1)
# Feasibility with random objective -> vertex-ish solution; scan flatness rank M_t == rank M_{t-1}.
# If flat at level t: extract atoms via commuting multiplication matrices, then VERIFY the
# recovered cubature reproduces ALL 35 standardized moments in BigFloat to ~1e-10.
#
# usage: julia ... kfvs_moment_extension_sdp_r.jl <order>     (default 4)

using JLD2, Printf, LinearAlgebra, JuMP, Hypatia, Random
setprecision(BigFloat, 256)
ORDER = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
Random.seed!(20260711)

const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM = D["lam"][1]; const C = D["center_state"]; const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))

s0 = findfirst(==(3), CLS)
M = big.(collect(C[:, s0])); ρ = M[1]
ux = M[2]/ρ; uy = M[6]/ρ; uz = M[16]/ρ
σx = sqrt(M[3]/ρ - ux^2); σy = sqrt(M[10]/ρ - uy^2); σz = sqrt(M[20]/ρ - uz^2)
sstdB = Dict{NTuple{3,Int},BigFloat}()
for n in 1:35
    (i,j,k) = TRIP[n]; acc = big(0.0)
    for p in 0:i, q in 0:j, r in 0:k
        haskey(IDX,(p,q,r)) || continue
        acc += bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(M[IDX[(p,q,r)]]/ρ)
    end
    sstdB[TRIP[n]] = acc/(σx^i*σy^j*σz^k)
end
# WORK IN RESCALED coords Y = X / L so the ~25sigma support is O(1) (Float64-conditioning fix).
# fixed moment in Y = sstd[alpha] / L^|alpha|;  de-scale atoms (X=L*Y) before final verification.
const L = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : (Float64(sstdB[(4,0,0)]))^(1/4)  # ~5.06 => deg-4 O(1)
sstd = Dict(k => Float64(v) / L^sum(k) for (k,v) in sstdB)   # rescaled fixed data
R = 1/LAM
meanv = (Float64(ux),Float64(uy),Float64(uz)); sigv = (Float64(σx),Float64(σy),Float64(σz))
@printf("class-C: mean=(%.3g,%.3g,%.3g) sig=(%.3g,%.3g,%.3g) R=%.1f  ORDER=%d  scale L=%.3f\n",
        meanv..., sigv..., R, ORDER, L)

mons(dmax) = sort([(i,j,k) for i in 0:dmax for j in 0:dmax for k in 0:dmax if i+j+k <= dmax];
                  by = m -> (sum(m), m))
addt(a,b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3])
Br = mons(ORDER); Brm1 = mons(ORDER-1)

model = Model(Hypatia.Optimizer); set_silent(model)
yvar = Dict{NTuple{3,Int},Any}()
frees = NTuple{3,Int}[]
for a in mons(2*ORDER)
    if sum(a) <= 4
        yvar[a] = sstd[a]
    else
        yvar[a] = @variable(model, base_name="y$(a[1])$(a[2])$(a[3])"); push!(frees, a)
    end
end
ym(a) = convert(AffExpr, yvar[a])
psd(mat) = @constraint(model, Symmetric(mat) in PSDCone())
# moment matrix
psd([ym(addt(Br[a],Br[b])) for a in eachindex(Br), b in eachindex(Br)])
# 8 diamond localizing (affine facets g_s), in Y coords: sigma -> sigma*L
for sgn in Iterators.product((-1,1),(-1,1),(-1,1))
    c0 = R - sum(sgn[i]*meanv[i] for i in 1:3); cc = ntuple(i -> -sgn[i]*sigv[i]*L, 3)
    Ls = [ c0*ym(addt(Brm1[a],Brm1[b])) +
          cc[1]*ym(addt(addt(Brm1[a],Brm1[b]),(1,0,0))) +
          cc[2]*ym(addt(addt(Brm1[a],Brm1[b]),(0,1,0))) +
          cc[3]*ym(addt(addt(Brm1[a],Brm1[b]),(0,0,1)))
          for a in eachindex(Brm1), b in eachindex(Brm1) ]
    psd(Ls)
end
# 3 box localizing g_i(Y)=Ri^2 - Y_i^2 >= 0 ; Y_i bound = X_i bound / L
Ri2 = ntuple(i -> (max(abs((R - meanv[i])/sigv[i]), abs((-R - meanv[i])/sigv[i]))/L)^2, 3)
for i in 1:3
    e2 = ntuple(d -> d==i ? 2 : 0, 3)
    Lb = [ Ri2[i]*ym(addt(Brm1[a],Brm1[b])) - ym(addt(addt(Brm1[a],Brm1[b]), e2))
           for a in eachindex(Brm1), b in eachindex(Brm1) ]
    psd(Lb)
end
# min trace(M_r): bounded below by 0 (PSD) -> well-posed rank surrogate (random obj is unbounded here)
@objective(model, Min, sum(ym(addt(Br[a],Br[a])) for a in eachindex(Br)))
@printf("[SDP] order=%d  M_r=%dx%d  8x M_{r-1}=%dx%d  free=%d ...\n",
        ORDER, length(Br), length(Br), length(Brm1), length(Brm1), length(frees))
optimize!(model)
@printf("  status=%s\n", termination_status(model))
yval = Dict(a => value(ym(a)) for a in mons(2*ORDER))

# flatness scan: rank M_t for t=1..ORDER
momrank(t) = begin
    Bt = mons(t); Mt = [yval[addt(Bt[a],Bt[b])] for a in eachindex(Bt), b in eachindex(Bt)]
    ev = sort(eigvals(Symmetric(Mt)); rev=true); (count(>(1e-8*maximum(abs,ev)), ev), ev, Bt, Mt)
end
ranks = [momrank(t) for t in 1:ORDER]
@printf("  ranks M_t: %s\n", join(("M$t=$(ranks[t][1])" for t in 1:ORDER), "  "))
flatlvl = 0
for t in 2:ORDER
    if ranks[t][1] == ranks[t-1][1]; global flatlvl = t; break; end
end

function extract_atoms(flatlvl, ranks)
    r = ranks[flatlvl][1]
    @printf("  VERDICT: FLAT at level %d (rank M_%d = rank M_%d = %d) => %d-atom measure EXISTS. Extracting...\n",
            flatlvl, flatlvl, flatlvl-1, r, r)
    _, _, Bt, Mt = ranks[flatlvl]
    Es = eigen(Symmetric(Mt)); ord = sortperm(Es.values; rev=true)
    V = Es.vectors[:, ord[1:r]] * Diagonal(sqrt.(Es.values[ord[1:r]]))  # Mt = V V'  (n x r)
    F = qr(Matrix(V'), ColumnNorm())
    pivots = sort(F.p[1:r])                 # r independent monomials (indices into Bt)
    basis = Bt[pivots]
    @printf("  monomial basis (%d): %s\n", r, join(string.(basis), ", "))
    Vb = V[pivots, :]                        # r x r, invertible
    Nmats = Vector{Matrix{Float64}}(undef, 3)
    ek = ((1,0,0),(0,1,0),(0,0,1))
    for k in 1:3
        rowsB = Array{Float64}(undef, r, r)
        for (bi, b) in enumerate(basis)
            sm = addt(b, ek[k]); idx = findfirst(==(sm), Bt)
            idx === nothing && (println("  extraction: shifted monomial $sm outside M_t — raise ORDER."); return)
            rowsB[bi, :] = V[idx, :]
        end
        Nmats[k] = rowsB / Vb                # multiplication matrix in basis coords
    end
    # common eigenvalues via random combination (Corless): schur of a random combo, read each N_k
    rc = randn(3); Ncomb = rc[1]*Nmats[1] + rc[2]*Nmats[2] + rc[3]*Nmats[3]
    Q = schur(Ncomb).Z
    atoms = Array{Float64}(undef, r, 3)
    for k in 1:3
        atoms[:, k] = real.(diag(Q' * Nmats[k] * Q))    # standardized-coord node values
    end
    Bw = mons(2); Phi = [prod(atoms[n,d]^Bw[m][d] for d in 1:3) for m in eachindex(Bw), n in 1:r]
    w = Phi \ [sstd[Bw[m]] for m in eachindex(Bw)]   # weights scale-free (solved in Y coords)
    @printf("  weights: min=%.3e  sum=%.4f (target 1)  #neg=%d\n", minimum(w), sum(w), count(<(-1e-9), w))
    atomsX = atoms .* L                              # de-scale Y -> X (standardized) for verification
    atB = big.(atomsX); wB = big.(w); maxerr = big(0.0); worst = (0,0,0)
    for n in 1:35
        (i,j,k) = TRIP[n]
        val = sum(wB[a]*atB[a,1]^i*atB[a,2]^j*atB[a,3]^k for a in 1:r)
        e = abs(val - sstdB[TRIP[n]]); e > maxerr && (maxerr = e; worst = TRIP[n])
    end
    cflmax = maximum(sum(abs(meanv[i] + sigv[i]*atomsX[a,i]) for i in 1:3) for a in 1:r)
    @printf("  VERIFY: max |std-moment residual| = %.3e (worst %s)\n", Float64(maxerr), worst)
    @printf("  CFL: max sum|v| over nodes = %.2f  (bound R=%.1f)  %s\n",
            cflmax, R, cflmax <= R*(1+1e-9) ? "OK all nodes CFL-feasible" : "VIOLATION")
    @printf("  POSITIVITY: %s\n", minimum(w) >= -1e-8 ? "all weights >= 0" : "has negative weight")
    if Float64(maxerr) < 1e-8 && minimum(w) >= -1e-8 && cflmax <= R*(1+1e-9)
        println("  ==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (moments matched to <1e-8).")
        save("gpu/validation/kfvs_cubature_solution.jld2",
             "atoms_std", atomsX, "weights", w, "mean", collect(meanv), "sig", collect(sigv),
             "R", R, "maxerr", Float64(maxerr), "cflmax", cflmax)
    else
        println("  ==> extraction imperfect (residual/positivity/CFL) — raise ORDER or refine.")
    end
end

if flatlvl == 0
    println("  VERDICT: FEASIBLE but not flat up to order $ORDER (no finite-atom certificate yet).")
    println("           => no compact-support obstruction; extraction needs higher order or grid fallback.")
else
    extract_atoms(flatlvl, ranks)
end

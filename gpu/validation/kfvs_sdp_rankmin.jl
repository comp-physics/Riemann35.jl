# kfvs_sdp_rankmin.jl — rescaled moment-extension SDP with Fazel reweighted-trace (log-det)
# rank minimization to drive toward a FLAT (atomic) extension, then constructive atom extraction
# + BigFloat verification of all 35 standardized moments. One class-C state, CFL-diamond support.
#
# Working coords Y = X/L (L ~ (m400)^1/4) so the ~25sigma support is O(1) (Float64 conditioning).
# Iterate:  min <W_k, M_r(y)>   with  W_0 = I,  W_{k+1} = (M_r(y_k) + delta I)^{-1}
#   -> converges toward a minimum-rank moment matrix; check flatness rank M_t == rank M_{t-1}.
#
# usage: julia ... kfvs_sdp_rankmin.jl [order=4] [scaleL] [iters=8] [delta=1e-3]

using JLD2, Printf, LinearAlgebra, JuMP, Hypatia
setprecision(BigFloat, 256)
ORDER = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
ITERS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
DELTA = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-3

const D = load("gpu/validation/kfvs_defect_counterexample.jld2")
const LAM = D["lam"][1]; const C = D["center_state"]; const CLS = D["class"]
const TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
const IDX = Dict(TRIP[n] => n for n in 1:35)
bino(n,k) = (k < 0 || k > n) ? big(0) : big(binomial(n,k))

s0 = findfirst(==(3), CLS)
Mst = big.(collect(C[:, s0])); ρ = Mst[1]
ux = Mst[2]/ρ; uy = Mst[6]/ρ; uz = Mst[16]/ρ
σx = sqrt(Mst[3]/ρ - ux^2); σy = sqrt(Mst[10]/ρ - uy^2); σz = sqrt(Mst[20]/ρ - uz^2)
sstdB = Dict{NTuple{3,Int},BigFloat}()
for n in 1:35
    (i,j,k) = TRIP[n]; acc = big(0.0)
    for p in 0:i, q in 0:j, r in 0:k
        haskey(IDX,(p,q,r)) || continue
        acc += bino(i,p)*bino(j,q)*bino(k,r)*(-ux)^(i-p)*(-uy)^(j-q)*(-uz)^(k-r)*(Mst[IDX[(p,q,r)]]/ρ)
    end
    sstdB[TRIP[n]] = acc/(σx^i*σy^j*σz^k)
end
const Lsc = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : (Float64(sstdB[(4,0,0)]))^(1/4)
sstd = Dict(k => Float64(v)/Lsc^sum(k) for (k,v) in sstdB)
R = 1/LAM
meanv = (Float64(ux),Float64(uy),Float64(uz)); sigv = (Float64(σx),Float64(σy),Float64(σz))
@printf("class-C: mean=(%.3g,%.3g,%.3g) sig=(%.3g,%.3g,%.3g) R=%.1f ORDER=%d L=%.3f iters=%d\n",
        meanv..., sigv..., R, ORDER, Lsc, ITERS)

mons(dmax) = sort([(i,j,k) for i in 0:dmax for j in 0:dmax for k in 0:dmax if i+j+k <= dmax]; by = m->(sum(m),m))
addt(a,b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3])
Br = mons(ORDER); Brm1 = mons(ORDER-1)

model = Model(Hypatia.Optimizer); set_silent(model)
yvar = Dict{NTuple{3,Int},Any}()
for a in mons(2*ORDER)
    yvar[a] = sum(a) <= 4 ? sstd[a] : @variable(model, base_name="y$(a[1])$(a[2])$(a[3])")
end
ym(a) = convert(AffExpr, yvar[a])
psd(mat) = @constraint(model, Symmetric(mat) in PSDCone())
Mrexpr = [ym(addt(Br[a],Br[b])) for a in eachindex(Br), b in eachindex(Br)]
psd(Mrexpr)
for sgn in Iterators.product((-1,1),(-1,1),(-1,1))
    c0 = R - sum(sgn[i]*meanv[i] for i in 1:3); cc = ntuple(i -> -sgn[i]*sigv[i]*Lsc, 3)
    Ls = [ c0*ym(addt(Brm1[a],Brm1[b])) +
           cc[1]*ym(addt(addt(Brm1[a],Brm1[b]),(1,0,0))) +
           cc[2]*ym(addt(addt(Brm1[a],Brm1[b]),(0,1,0))) +
           cc[3]*ym(addt(addt(Brm1[a],Brm1[b]),(0,0,1)))
           for a in eachindex(Brm1), b in eachindex(Brm1) ]
    psd(Ls)
end
Ri2 = ntuple(i -> (max(abs((R-meanv[i])/sigv[i]), abs((-R-meanv[i])/sigv[i]))/Lsc)^2, 3)
for i in 1:3
    e2 = ntuple(d -> d==i ? 2 : 0, 3)
    psd([ Ri2[i]*ym(addt(Brm1[a],Brm1[b])) - ym(addt(addt(Brm1[a],Brm1[b]), e2))
          for a in eachindex(Brm1), b in eachindex(Brm1) ])
end

rankof(Mv, tol=1e-7) = (ev = sort(eigvals(Symmetric(Mv)); rev=true); count(>(tol*maximum(abs,ev)), ev))
Mval_of(yv, t) = (Bt = mons(t); [yv[addt(Bt[a],Bt[b])] for a in eachindex(Bt), b in eachindex(Bt)])

function try_extract(yv, flatlvl)
    # flat at level flatlvl (<= ORDER-1): pick basis from monomials of degree <= flatlvl-1 so
    # multiplication-shifted monomials (degree <= flatlvl) stay inside the M_flatlvl index set.
    Bfull = mons(flatlvl); Mt = Mval_of(yv, flatlvl); r = rankof(Mt)
    Es = eigen(Symmetric(Mt)); ord = sortperm(Es.values; rev=true)
    V = Es.vectors[:, ord[1:r]] * Diagonal(sqrt.(abs.(Es.values[ord[1:r]])))   # |Bfull| x r
    lowidx = [i for i in eachindex(Bfull) if sum(Bfull[i]) <= flatlvl-1]
    piv_local = sort(qr(Matrix(V[lowidx,:]'), ColumnNorm()).p[1:r])
    pivots = lowidx[piv_local]; basis = Bfull[pivots]
    Vb = V[pivots, :]; ek = ((1,0,0),(0,1,0),(0,0,1)); Nmats = Vector{Matrix{Float64}}(undef,3)
    for k in 1:3
        rowsB = Array{Float64}(undef, r, r)
        for (bi,b) in enumerate(basis)
            sm = addt(b, ek[k]); idx = findfirst(==(sm), Bfull)
            idx === nothing && return (false, nothing, nothing, r)
            rowsB[bi,:] = V[idx,:]
        end
        Nmats[k] = rowsB / Vb
    end
    rc = randn(3); Q = schur(rc[1]*Nmats[1]+rc[2]*Nmats[2]+rc[3]*Nmats[3]).Z
    atomsY = Array{Float64}(undef, r, 3)
    for k in 1:3; atomsY[:,k] = real.(diag(Q'*Nmats[k]*Q)); end
    Bw = mons(2); Phi = [prod(atomsY[n,d]^Bw[m][d] for d in 1:3) for m in eachindex(Bw), n in 1:r]
    w = Phi \ [sstd[Bw[m]] for m in eachindex(Bw)]
    atomsX = atomsY .* Lsc
    atB = big.(atomsX); wB = big.(w); maxerr = big(0.0); worst=(0,0,0)
    for n in 1:35
        (i,j,k)=TRIP[n]; val = sum(wB[a]*atB[a,1]^i*atB[a,2]^j*atB[a,3]^k for a in 1:r)
        e = abs(val - sstdB[TRIP[n]]); e>maxerr && (maxerr=e; worst=TRIP[n])
    end
    cflmax = maximum(sum(abs(meanv[i]+sigv[i]*atomsX[a,i]) for i in 1:3) for a in 1:r)
    (true, (atomsX=atomsX, w=w, maxerr=Float64(maxerr), worst=worst, cflmax=cflmax, r=r), basis, r)
end

# ---- reweighted-trace (log-det) rank-min loop, targeting M_{ORDER-1} (drives it to rank M_2) ----
RW = ORDER - 1                        # reweight/level we push toward flatness (M_3 for ORDER=4)
Brw = mons(RW)
W = Matrix{Float64}(I, length(Brw), length(Brw))
local yv
for it in 0:ITERS
    @objective(model, Min, sum(W[a,b]*ym(addt(Brw[a],Brw[b])) for a in eachindex(Brw), b in eachindex(Brw)))
    optimize!(model)
    global yv = Dict(a => value(ym(a)) for a in mons(2*ORDER))
    rks = [rankof(Mval_of(yv,t)) for t in 1:ORDER]
    @printf("[it %d] status=%s  ranks M_t: %s\n", it, termination_status(model),
            join(("M$t=$(rks[t])" for t in 1:ORDER), " "))
    flat = 0
    for t in 2:ORDER-1; if rks[t]==rks[t-1]; flat=t; break; end; end   # extractable levels only
    if flat > 0
        @printf("  FLAT at level %d (rank=%d). extracting...\n", flat, rks[flat])
        ok, sol, basis, r = try_extract(yv, flat)
        if ok
            @printf("  weights: min=%.3e sum=%.4f #neg=%d | max std-moment resid=%.3e (worst %s)\n",
                    minimum(sol.w), sum(sol.w), count(<(-1e-9),sol.w), sol.maxerr, sol.worst)
            @printf("  CFL max sum|v|=%.2f (R=%.1f) %s | positivity %s\n", sol.cflmax, R,
                    sol.cflmax<=R*(1+1e-9) ? "OK" : "VIOL", minimum(sol.w)>=-1e-8 ? "OK" : "NEG")
            if sol.maxerr<1e-6 && minimum(sol.w)>=-1e-8 && sol.cflmax<=R*(1+1e-9)
                println("  ==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND.")
                save("gpu/validation/kfvs_cubature_solution.jld2",
                     "atoms_std", sol.atomsX, "weights", sol.w, "mean", collect(meanv),
                     "sig", collect(sigv), "R", R, "maxerr", sol.maxerr, "cflmax", sol.cflmax)
                break
            else
                println("  (flat but extraction not clean — continue reweighting)")
            end
        else
            println("  (flat but shifted monomial outside index set — continue)")
        end
    end
    # reweight: W = (M_{RW} + delta I)^{-1}, normalized  (log-det surrogate on the target level)
    Mrw = Mval_of(yv, RW); Wn = inv(Symmetric(Mrw) + DELTA*I)
    global W = Wn ./ opnorm(Wn)
end
println("done.")

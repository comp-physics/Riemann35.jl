# kfvs_sdp_order5.jl — higher-order (Lasserre-5) flat-extension search with GENERIC objectives.
#
# Per reviewer guidance: stop forcing a level-3 rank-10/12 flat. Build the degree-10 extension,
# use SEVERAL generic random linear objectives (NOT trace — trace distorts the high-degree block),
# and look for flatness rank M_t == rank M_{t-1} at any extractable level (2..ORDER), especially
# the plateau rank M_5 == rank M_4 (~20). Report FULL spectra; rank is relative to lambda_max.
#
# Rescaled coords Y=X/L (L~m400^{1/4}) for conditioning. Support-search box |X_i|<=XBOX*sigma
# (generous vs the proven ~25.6σ support) compactifies the relaxation so random objectives are
# bounded; any found measure is VERIFIED in BigFloat standardized monomials, so the box is only
# scaffolding for the search.
#
# usage: julia ... kfvs_sdp_order5.jl [order=5] [nobj=4] [XBOX=40]

using JLD2, Printf, LinearAlgebra, JuMP, Hypatia, Random
setprecision(BigFloat, 256)
ORDER = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
NOBJ  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
XBOX  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 40.0
TOL   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-7   # looser solve = faster; rank detection is robust

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
const Lsc = (Float64(sstdB[(4,0,0)]))^(1/4)
sstd = Dict(k => Float64(v)/Lsc^sum(k) for (k,v) in sstdB)
R = 1/LAM
meanv = (Float64(ux),Float64(uy),Float64(uz)); sigv = (Float64(σx),Float64(σy),Float64(σz))
@printf("class-C: mean=(%.3g,%.3g,%.3g) sig=(%.3g,%.3g,%.3g) R=%.1f ORDER=%d L=%.3f XBOX=%.0fσ nobj=%d\n",
        meanv..., sigv..., R, ORDER, Lsc, XBOX, NOBJ)

mons(dmax) = sort([(i,j,k) for i in 0:dmax for j in 0:dmax for k in 0:dmax if i+j+k <= dmax]; by=m->(sum(m),m))
addt(a,b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3])
Br = mons(ORDER); Brm1 = mons(ORDER-1)
@printf("sizes: M_%d = %dx%d ; localizing M_%d = %dx%d (x11) ; free moments = %d\n",
        ORDER, length(Br), length(Br), ORDER-1, length(Brm1), length(Brm1), length(mons(2*ORDER))-35)

function build_model()
    m = Model(Hypatia.Optimizer); set_silent(m)
    set_optimizer_attribute(m, "tol_rel_opt", TOL)
    set_optimizer_attribute(m, "tol_abs_opt", TOL)
    set_optimizer_attribute(m, "tol_feas", TOL)
    yv = Dict{NTuple{3,Int},Any}()
    for a in mons(2*ORDER)
        yv[a] = sum(a) <= 4 ? sstd[a] : @variable(m, base_name="y$(a[1])$(a[2])$(a[3])")
    end
    ymf(a) = convert(AffExpr, yv[a])
    psd(mat) = @constraint(m, Symmetric(mat) in PSDCone())
    psd([ymf(addt(Br[a],Br[b])) for a in eachindex(Br), b in eachindex(Br)])
    for sgn in Iterators.product((-1,1),(-1,1),(-1,1))       # 8 CFL facets (affine), Y coords
        c0 = R - sum(sgn[i]*meanv[i] for i in 1:3); cc = ntuple(i -> -sgn[i]*sigv[i]*Lsc, 3)
        psd([ c0*ymf(addt(Brm1[a],Brm1[b])) +
              cc[1]*ymf(addt(addt(Brm1[a],Brm1[b]),(1,0,0))) +
              cc[2]*ymf(addt(addt(Brm1[a],Brm1[b]),(0,1,0))) +
              cc[3]*ymf(addt(addt(Brm1[a],Brm1[b]),(0,0,1)))
              for a in eachindex(Brm1), b in eachindex(Brm1) ])
    end
    Yb2 = (XBOX/Lsc)^2                                        # search box |Y_i|<=XBOX/L
    for i in 1:3
        e2 = ntuple(d -> d==i ? 2 : 0, 3)
        psd([ Yb2*ymf(addt(Brm1[a],Brm1[b])) - ymf(addt(addt(Brm1[a],Brm1[b]), e2))
              for a in eachindex(Brm1), b in eachindex(Brm1) ])
    end
    m, yv, ymf
end

rankrel(Mv; rtol=1e-6) = (ev = sort(eigvals(Symmetric(Mv)); rev=true); (count(>(rtol*ev[1]), ev), ev))
Mval_of(yval, t) = (Bt = mons(t); [yval[addt(Bt[a],Bt[b])] for a in eachindex(Bt), b in eachindex(Bt)])

function try_extract(yval, flatlvl)
    Bfull = mons(flatlvl); Mt = Mval_of(yval, flatlvl); r, _ = rankrel(Mt)
    Es = eigen(Symmetric(Mt)); ord = sortperm(Es.values; rev=true)
    V = Es.vectors[:, ord[1:r]] * Diagonal(sqrt.(abs.(Es.values[ord[1:r]])))
    lowidx = [i for i in eachindex(Bfull) if sum(Bfull[i]) <= flatlvl-1]
    piv_local = sort(qr(Matrix(V[lowidx,:]'), ColumnNorm()).p[1:r])
    pivots = lowidx[piv_local]; basis = Bfull[pivots]; Vb = V[pivots, :]
    ek = ((1,0,0),(0,1,0),(0,0,1)); Nmats = Vector{Matrix{Float64}}(undef,3)
    for k in 1:3
        rowsB = Array{Float64}(undef, r, r)
        for (bi,b) in enumerate(basis)
            sm = addt(b, ek[k]); idx = findfirst(==(sm), Bfull)
            idx === nothing && return nothing
            rowsB[bi,:] = V[idx,:]
        end
        Nmats[k] = rowsB / Vb
    end
    rc = randn(3); Q = schur(rc[1]*Nmats[1]+rc[2]*Nmats[2]+rc[3]*Nmats[3]).Z
    atomsY = Array{Float64}(undef, r, 3)
    for k in 1:3; atomsY[:,k] = real.(diag(Q'*Nmats[k]*Q)); end
    Bw = mons(2); Phi = [prod(atomsY[n,d]^Bw[mm][d] for d in 1:3) for mm in eachindex(Bw), n in 1:r]
    w = Phi \ [sstd[Bw[mm]] for mm in eachindex(Bw)]
    atomsX = atomsY .* Lsc
    atB = big.(atomsX); wB = big.(w); maxerr = big(0.0); worst=(0,0,0)
    for n in 1:35
        (i,j,k)=TRIP[n]; val = sum(wB[a]*atB[a,1]^i*atB[a,2]^j*atB[a,3]^k for a in 1:r)
        e = abs(val - sstdB[TRIP[n]]); e>maxerr && (maxerr=e; worst=TRIP[n])
    end
    cflmax = maximum(sum(abs(meanv[i]+sigv[i]*atomsX[a,i]) for i in 1:3) for a in 1:r)
    (atomsX=atomsX, w=w, maxerr=Float64(maxerr), worst=worst, cflmax=cflmax, r=r)
end

best = nothing
for seed in 1:NOBJ
    Random.seed!(1000+seed)
    m, yv, ymf = build_model()
    cobj = Dict(a => randn() for a in mons(2*ORDER) if sum(a) > 4)   # generic objective, free moments
    @objective(m, Min, sum(c*ymf(a) for (a,c) in cobj))
    optimize!(m)
    stat = termination_status(m)
    if !has_values(m); @printf("[obj %d] status=%s (no solution)\n", seed, stat); continue; end
    yval = Dict(a => value(ymf(a)) for a in mons(2*ORDER))
    rk = [rankrel(Mval_of(yval,t)) for t in 1:ORDER]
    @printf("[obj %d] status=%s  ranks(rel1e-6): %s\n", seed, stat,
            join(("M$t=$(rk[t][1])" for t in 1:ORDER), " ")); flush(stdout)
    for t in 2:ORDER
        ev = rk[t][2]
        @printf("    M_%d spectrum: %s\n", t, join((@sprintf("%.1e",e) for e in ev[1:min(10,length(ev))]), " "))
    end
    flat = 0
    for t in 2:ORDER; if rk[t][1] == rk[t-1][1]; flat=t; break; end; end
    if flat > 0
        @printf("  >>> FLAT at level %d (rank=%d). extracting...\n", flat, rk[flat][1])
        sol = try_extract(yval, flat)
        if sol !== nothing
            @printf("  weights min=%.3e sum=%.4f #neg=%d | std-moment maxerr=%.3e (worst %s) | CFL sum|v|max=%.1f/%.0f | %s\n",
                    minimum(sol.w), sum(sol.w), count(<(-1e-9),sol.w), sol.maxerr, sol.worst, sol.cflmax, R,
                    (sol.maxerr<1e-8 && minimum(sol.w)>=-1e-8 && sol.cflmax<=R*(1+1e-9)) ? "CLEAN" : "imperfect")
            if best === nothing || sol.maxerr < best.maxerr; global best = sol; end
        else
            println("  (extraction: shifted monomial outside index set)")
        end
    end
end

if best !== nothing
    @printf("\nBEST extraction: %d atoms, std-moment maxerr=%.3e (worst %s), CFL max=%.1f/%.0f, minw=%.3e\n",
            best.r, best.maxerr, best.worst, best.cflmax, R, minimum(best.w))
    if best.maxerr < 1e-8 && minimum(best.w) >= -1e-8 && best.cflmax <= R*(1+1e-9)
        println("==> CONSTRUCTIVE CFL-SAFE POSITIVE CUBATURE FOUND (BigFloat-verified).")
        save("gpu/validation/kfvs_cubature_solution.jld2",
             "atoms_std", best.atomsX, "weights", best.w, "mean", collect(meanv),
             "sig", collect(sigv), "R", R, "maxerr", best.maxerr, "cflmax", best.cflmax, "natoms", best.r)
    else
        println("==> flat found but extraction not yet BigFloat-clean — continuation solve next.")
    end
else
    println("\nNo flat extension at any extractable level for the tried objectives.")
    println("=> raise ORDER or move to jet-marginal continuation solve.")
end

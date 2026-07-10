# fix1_gate_probe.jl — verify FIX 1 theory (GPU-light, CPU-only).
#
# The CPU condition gate is _design_cond(B) = svdvals(B)[1]/svdvals(B)[end] on the
# TALL weighted design B = sqrt(pw) .* Phi, rejecting a candidate column when
# κ₂(B) ≥ 1e4. The prototype device gate used a Cholesky-pivot-ratio proxy on the
# Gram G = BᵀB, which UNDER-estimates κ(G) → 0.18% spurious extra columns.
#
# FIX 1 theory: G is SPD and symmetric, so σᵢ(B) = sqrt(λᵢ(G)). Therefore
#   κ₂(B) = σmax(B)/σmin(B) = sqrt(λmax(G)/λmin(G)).
# So the EXACT device replacement for the CPU svdvals gate is: compute the extreme
# eigenvalues of the small (≤9×9) SPD Gram G and gate on sqrt(λmax/λmin) ≥ 1e4
# (equivalently λmax/λmin ≥ 1e8). This probe confirms, on real column decisions
# harvested from real cells, that:
#   (1) sqrt(eig-ratio of G) reproduces svdvals(B) to ~machine precision, and
#   (2) it reproduces the CPU accept/reject decision on ~100% of column tests,
#       vs the pivot-ratio proxy's ~99.8%.
#
# GPU-light: pure CPU LinearAlgebra. Run FIRST (GPU compiles are slow).

using Riemann35
using JLD2, Printf, Random, LinearAlgebra

# Reach into the CPU reference internals via its module path.
const R = Riemann35

# We re-implement the CPU _gram_fit column-selection loop here but INSTRUMENT it:
# at every candidate-column test, record (B_try, cpu_decision). We replicate the
# exact CPU logic from src/moments/chyqmom_nodes_3d.jl (_gram_fit / _design_cond).

_monomial(coords, p, e) = begin
    v = 1.0
    @inbounds for d in eachindex(e)
        ed = e[d]
        ed == 0 || (v *= coords[p, d]^ed)
    end
    v
end

# candidate-decision harvester: mirrors CPU _gram_fit exactly, but logs each
# tentative column's Gram G_try (leading (k)x(k)) and the CPU accept/reject.
function harvest_gram_decisions!(store, pw, coords, targets)
    Np = length(pw)
    nt = length(targets)
    order = sortperm([sum(targets[m][1]) for m in 1:nt])
    sw = sqrt.(max.(pw, 0.0))
    condmax = 1e4
    sel = Int[]
    B = Matrix{Float64}(undef, Np, 0)
    @inbounds for m in order
        col = Float64[sw[p] * _monomial(coords, p, targets[m][1]) for p in 1:Np]
        all(iszero, col) && continue
        Btry = hcat(B, col)
        # CPU decision
        s = svdvals(Btry)
        condB = (isempty(s) || s[end] <= 0) ? Inf : s[1]/s[end]
        cpu_accept = (size(Btry,2) <= Np) && (condB < condmax)
        # Gram of the trial (SPD, k×k)
        Gtry = Btry' * Btry
        push!(store, (Gtry, condB, cpu_accept))
        if cpu_accept
            B = Btry
            push!(sel, m)
            length(sel) == Np && break
        end
    end
    return nothing
end

# extreme eigenvalues of small SPD G → κ(B) = sqrt(λmax/λmin)
function kappaB_from_G(G)
    ev = eigvals(Symmetric(G))
    lo = minimum(ev); hi = maximum(ev)
    lo <= 0 && return Inf
    return sqrt(hi/lo)
end

# the prototype's pivot/scale proxy on G (max(max_diag,max_pivot)/min_pivot),
# reproduced here to measure its mismatch. Returns proxy for κ(G) (≈ κ(B)²).
function pivot_proxy_condG(G)
    n = size(G,1)
    L = zeros(n,n)
    dmin = Inf; dmax = -Inf; gdmax = -Inf
    for i in 1:n
        gdmax = max(gdmax, G[i,i])
        s = G[i,i]
        for k in 1:i-1
            s -= L[i,k]^2
        end
        s <= 0 && return Inf
        dmin = min(dmin, s); dmax = max(dmax, s)
        L[i,i] = sqrt(s)
        for j in i+1:n
            t = G[j,i]
            for k in 1:i-1
                t -= L[j,k]*L[i,k]
            end
            L[j,i] = t / L[i,i]
        end
    end
    return max(gdmax, dmax)/dmin
end

const TRIPLES = R.chyqmom_nodes_3d isa Function ? [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)] : nothing

# central-moment helper matching CPU _chyq_central
function cm_all(M)
    Mraw = zeros(5,5,5)
    for n in 1:35
        (i,j,k)=TRIPLES[n]; Mraw[i+1,j+1,k+1]=M[n]
    end
    rho=Mraw[1,1,1]; bu=Mraw[2,1,1]/rho; bv=Mraw[1,2,1]/rho; bw=Mraw[1,1,2]/rho
    cm(i,j,k) = begin
        s=0.0
        for a in 0:i, b in 0:j, d in 0:k
            s += binomial(i,a)*binomial(j,b)*binomial(k,d)*(-bu)^(i-a)*(-bv)^(j-b)*(-bw)^(k-d)*Mraw[a+1,b+1,d+1]
        end
        s/rho
    end
    return cm, bu, bv, bw, rho
end

function main()
    Random.seed!(2026)
    files = [
        "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2",
        "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma10_o1.jld2",
    ]
    cells = NTuple{35,Float64}[]
    for f in files
        isfile(f) || continue
        data = jldopen(f,"r") do jf
            arr=nothing
            for k in keys(jf)
                v=jf[k]
                if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; arr=v; break; end
            end
            arr
        end
        data === nothing && continue
        nx,ny,nz,_=size(data); nc=nx*ny*nz
        take=min(nc, 12000)
        for lin in randperm(nc)[1:take]
            k=(lin-1)÷(nx*ny)+1; r=(lin-1)%(nx*ny); j=r÷nx+1; i=r%nx+1
            m=ntuple(t->Float64(data[i,j,k,t]),Val(35))
            m[1]>0 || continue
            push!(cells, m)
        end
    end
    println("Harvesting Gram decisions from ", length(cells), " real cells...")

    # For each cell, reproduce the x/y/z levels via the CPU reference to get the
    # actual parent geometry, then harvest z-mean & z-var column decisions (the
    # ones the report flags). We call the CPU chyqmom to build the parent nodes,
    # but re-run the fit-level harvest ourselves.
    store = Tuple{Matrix{Float64},Float64,Bool}[]
    nthrew = 0
    for M in cells
        Mv = collect(M)
        local ncpu, Ucpu
        try
            ncpu, Ucpu = R.chyqmom_nodes_3d(Mv)
        catch
            nthrew += 1; continue
        end
        # rebuild parent (x,y) nodes: we approximate the parent set as the UNIQUE
        # (Ux,Uy) pairs of the returned quadrature with summed weights — this is the
        # exact z-level parent cloud the CPU z-fit saw.
        cm, bu, bv, bw, rho = cm_all(M)
        # group nodes by (ux,uy)
        keys_xy = Dict{Tuple{Float64,Float64},Float64}()
        for q in eachindex(ncpu)
            key=(Ucpu[q,1],Ucpu[q,2])
            keys_xy[key]=get(keys_xy,key,0.0)+ncpu[q]/rho
        end
        pw=Float64[]; cx=Float64[]; cy=Float64[]
        for (key,w) in keys_xy
            push!(pw,w); push!(cx,key[1]-bu); push!(cy,key[2]-bv)
        end
        Np=length(pw)
        coords = hcat(cx,cy)
        # z-mean targets c_{ij1}, i+j<=3
        mean_t = Tuple{NTuple{2,Int},Float64}[]
        for i in 0:3, j in 0:(3-i); push!(mean_t, ((i,j), cm(i,j,1))); end
        harvest_gram_decisions!(store, pw, coords, mean_t)
        # z-var targets c_{ij2}, i+j<=2
        var_t = Tuple{NTuple{2,Int},Float64}[]
        for i in 0:2, j in 0:(2-i); push!(var_t, ((i,j), cm(i,j,2))); end
        harvest_gram_decisions!(store, pw, coords, var_t)
    end

    println("Total column decisions harvested: ", length(store), " (CPU threw on ", nthrew, " cells)")

    # Compare the three gates.
    condmax = 1e4
    n_eig_mismatch = 0
    n_proxy_mismatch = 0
    max_svd_vs_eig_rel = 0.0
    n_accept = 0; n_reject = 0
    for (G, condB_cpu, cpu_accept) in store
        cpu_accept ? (n_accept+=1) : (n_reject+=1)
        # eig gate
        kB = kappaB_from_G(G)
        Np_ok = true # count handled by CPU already; here focus on cond decision
        eig_accept = kB < condmax
        # svd(B) vs sqrt(eig(G)): condB_cpu should equal kB up to rounding
        if isfinite(condB_cpu) && isfinite(kB) && condB_cpu > 0
            max_svd_vs_eig_rel = max(max_svd_vs_eig_rel, abs(kB-condB_cpu)/condB_cpu)
        end
        # NOTE: cpu_accept also embeds the size<=Np constraint; when condB<condmax
        # but size>Np the CPU rejects. Compare gate-only where that's not the cause.
        # We compare the CONDITION decision: eig_accept vs (condB_cpu<condmax).
        cpu_cond_accept = condB_cpu < condmax
        (eig_accept != cpu_cond_accept) && (n_eig_mismatch += 1)
        # proxy gate at tuned threshold 3e6 on κ(G) proxy
        proxy = pivot_proxy_condG(G)
        proxy_accept = proxy < 3.0e6
        (proxy_accept != cpu_cond_accept) && (n_proxy_mismatch += 1)
    end
    N = length(store)
    println("\n============== FIX 1 GATE PROBE ==============")
    @printf("Column decisions            : %d  (%d CPU-accept, %d CPU-reject)\n", N, n_accept, n_reject)
    @printf("max rel err sqrt(eig(G)) vs svdvals(B) : %.3e  (should be ~machine)\n", max_svd_vs_eig_rel)
    @printf("EIG gate   accept/reject mismatches vs CPU : %d / %d  (%.4f%%)\n",
            n_eig_mismatch, N, 100*n_eig_mismatch/N)
    @printf("PROXY gate accept/reject mismatches vs CPU : %d / %d  (%.4f%%)\n",
            n_proxy_mismatch, N, 100*n_proxy_mismatch/N)
    println("==============================================")
end

main()

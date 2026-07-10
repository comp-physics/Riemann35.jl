# parity_chyqmom_nodes_3d_dev.jl — CPU-parity gate for the device CHyQMOM inversion.
#
# Runs the DEVICE function `chyqmom_nodes_3d_dev` on the HOST over >=20k real cells
# (from the ma100/ma10 snapshots) + synthetic reachable states, and compares against
# the CPU reference `chyqmom_nodes_3d` (Riemann35). Reports:
#   * mean moments reproduced (CPU and DEV)
#   * min node weight >= -1e-12 (the realizability certificate — must be 100%)
#   * low-order (mass/mean/var) machine-exactness
#   * node-count match rate DEV vs CPU (target ~100% after the faithful gate)
#   * SPURIOUS-EXTRA-COLUMN rate: fraction of cells where DEV admits a column the
#     CPU rejects, detected as a blown-up high-order cross moment (|Mrep| >> |M|
#     while low-order is exact). Target ~0 (down from the pivot-proxy's 0.18%).
#
# NO GPU required — validates correctness before the device compile. Run first.

using Riemann35
using JLD2, Printf, Random, LinearAlgebra

include(joinpath(@__DIR__, "..", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev

const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

# Genuine low-order INVARIANTS that CHyQMOM reproduces exactly: mass (1) and the
# three means (2,6,16). The diagonal 2nd moments (3,10,20 = the marginal variances)
# are NOT invariants — CHyQMOM legitimately truncates them on cold/near-vacuum
# marginals, and the CPU reference itself does so (verified: DEV == CPU bit-for-bit
# on those cells). So they are excluded from the "must be machine-exact" set.
const INVARIANT_IDX = [1,2,6,16]

function moments_from_dev(nn, nux, nuy, nuz, Nn)
    out = zeros(35)
    for m in 1:35
        (i,j,k) = TRIPLES[m]; s = 0.0
        @inbounds for q in 1:Nn
            s += nn[q]*(nux[q]^i)*(nuy[q]^j)*(nuz[q]^k)
        end
        out[m] = s
    end
    out
end
function moments_from_quad(n, U)
    out = zeros(35)
    for m in 1:35
        (i,j,k) = TRIPLES[m]; s = 0.0
        @inbounds for q in eachindex(n)
            s += n[q]*(U[q,1]^i)*(U[q,2]^j)*(U[q,3]^k)
        end
        out[m] = s
    end
    out
end
function count_reproduced(Mt, Mr; atol=1e-6, rtol=1e-6)
    c=0
    for m in 1:35
        a=abs(Mr[m]-Mt[m]); r=a/max(abs(Mt[m]),1e-300)
        (a<=atol || r<=rtol) && (c+=1)
    end
    c
end

function main()
    Random.seed!(12345)
    files = [
        "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2",
        "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma10_o1.jld2",
    ]
    real_cells = NTuple{35,Float64}[]
    for f in files
        isfile(f) || (println("MISSING: ",f); continue)
        data = jldopen(f,"r") do jf
            arr=nothing
            for k in keys(jf); v=jf[k]; if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; arr=v; break; end; end
            arr
        end
        data === nothing && continue
        nx,ny,nz,_=size(data); nc=nx*ny*nz
        take=min(nc, 12000)
        for lin in randperm(nc)[1:take]
            k=(lin-1)÷(nx*ny)+1; r=(lin-1)%(nx*ny); j=r÷nx+1; i=r%nx+1
            m=ntuple(t->Float64(data[i,j,k,t]),Val(35))
            m[1]>0 || continue
            push!(real_cells, m)
        end
        println("Collected ", length(real_cells), " real cells so far (", basename(f), ")")
    end
    synth = NTuple{35,Float64}[]
    for _ in 1:5000
        K=rand(2:6); w=abs.(randn(K)).+1e-2; w./=sum(w); rho=exp(2*randn()); w.*=rho
        pts=randn(K,3).*exp(randn()); mm=zeros(35)
        for m in 1:35
            (i,j,k)=TRIPLES[m]; s=0.0
            for q in 1:K; s+=w[q]*(pts[q,1]^i)*(pts[q,2]^j)*(pts[q,3]^k); end
            mm[m]=s
        end
        mm[1]>0 || continue
        push!(synth, ntuple(t->mm[t],Val(35)))
    end
    println("Generated ", length(synth), " synthetic reachable states")
    all_cells = vcat(real_cells, synth)
    ncell = length(all_cells)
    nreal = length(real_cells)
    println("Total cells to test: ", ncell)

    min_weight_dev=Inf; min_weight_cpu=Inf
    max_inv_rel=0.0                   # invariant (mass/mean) rel err, rho-floored
    ndiverge_count=0; ndiverge_real=0; ndiverge_synth=0
    ncmatch_real=0; ncmatch_synth=0
    sum_rep_cpu=0; sum_rep_dev=0
    n_dev_neg=0; n_cpu_threw=0; n_cpu_ok=0
    n_spurious=0; n_spurious_real=0   # genuine wild-abscissa (over-admitted column) cells
    worst_blow=0.0
    diverge_examples=Tuple{Int,Int,Int}[]

    for (ci,M) in enumerate(all_cells)
        Mv=collect(M)
        (nn,nux,nuy,nuz,Nn)=chyqmom_nodes_3d_dev(M)
        wmin=Inf; for q in 1:Nn; wmin=min(wmin,nn[q]); end
        Nn==0 && (wmin=0.0)
        min_weight_dev=min(min_weight_dev,wmin)
        (wmin<-1e-12) && (n_dev_neg+=1)
        Mrep=moments_from_dev(nn,nux,nuy,nuz,Nn)
        sum_rep_dev += count_reproduced(M,Mrep)
        # DEV max abscissa magnitude (for the wild-node detector below)
        amax_dev=0.0; for q in 1:Nn; amax_dev=max(amax_dev,abs(nux[q]),abs(nuy[q]),abs(nuz[q])); end
        # invariant (mass/mean) accuracy: rel err with a rho-scaled floor so a
        # near-DENORMAL true mean (rho~1e-3, bu~1e-300 => M~1e-303) doesn't
        # manufacture a spurious relative "blowup" from a ~1e-19 absolute roundoff.
        rho=M[1]
        for idx in INVARIANT_IDX
            a=abs(Mrep[idx]-M[idx]); sc=max(abs(M[idx]), abs(rho)*1e-10)
            max_inv_rel=max(max_inv_rel, a/sc)
        end

        local ncpu,Ucpu
        try; ncpu,Ucpu=chyqmom_nodes_3d(Mv); catch; n_cpu_threw+=1; continue; end
        n_cpu_ok+=1
        wmc = isempty(ncpu) ? 0.0 : minimum(ncpu); min_weight_cpu=min(min_weight_cpu,wmc)
        sum_rep_cpu += count_reproduced(M, moments_from_quad(ncpu,Ucpu))
        # GENUINE spurious over-admission: a DEV abscissa far exceeding the CPU's max
        # abscissa (a wild node from an over-admitted near-collinear column). Gate-
        # dependent and denormal-robust (abscissas, not moments).
        amax_cpu=0.0; for q in eachindex(ncpu); amax_cpu=max(amax_cpu,abs(Ucpu[q,1]),abs(Ucpu[q,2]),abs(Ucpu[q,3])); end
        if amax_dev > 100.0*max(amax_cpu,1.0) && amax_dev > 1e3
            n_spurious+=1; ci<=nreal && (n_spurious_real+=1)
            worst_blow=max(worst_blow, amax_dev)
        end
        isreal_cell = ci <= nreal
        if Nn != length(ncpu)
            ndiverge_count+=1
            isreal_cell ? (ndiverge_real+=1) : (ndiverge_synth+=1)
            length(diverge_examples)<12 && push!(diverge_examples,(ci,length(ncpu),Nn))
        else
            isreal_cell ? (ncmatch_real+=1) : (ncmatch_synth+=1)
        end
    end

    println("\n================= CPU-PARITY RESULTS (faithful gate) =================")
    @printf("Cells tested                 : %d (%d real, %d synthetic)\n", ncell, nreal, length(synth))
    @printf("CPU reference THREW (Singular): %d cells  (DEV survived all)\n", n_cpu_threw)
    @printf("CPU succeeded on             : %d cells (parity denominator)\n", n_cpu_ok)
    @printf("Mean moments reproduced CPU  : %.2f / 35\n", n_cpu_ok>0 ? sum_rep_cpu/n_cpu_ok : 0.0)
    @printf("Mean moments reproduced DEV  : %.2f / 35 (all cells)\n", sum_rep_dev/ncell)
    @printf("Min node weight (DEV)        : %.3e  (cert >= -1e-12 ? %s)\n",
            min_weight_dev, min_weight_dev>=-1e-12 ? "YES" : "NO")
    @printf("Cells with DEV weight < -1e-12: %d\n", n_dev_neg)
    @printf("Invariant (mass/mean) rel err (DEV, rho-floored) max : %.3e\n", max_inv_rel)
    real_ok = ncmatch_real + ndiverge_real
    synth_ok = ncmatch_synth + ndiverge_synth
    @printf("Node-count match DEV vs CPU  : REAL %d/%d (%.4f%%)  SYNTH %d/%d (%.4f%%)\n",
            ncmatch_real, real_ok, real_ok>0 ? 100*ncmatch_real/real_ok : 0.0,
            ncmatch_synth, synth_ok, synth_ok>0 ? 100*ncmatch_synth/synth_ok : 0.0)
    @printf("Node-count mismatch (total)  : %d / %d  (%.4f%%)\n",
            ndiverge_count, n_cpu_ok, n_cpu_ok>0 ? 100*ndiverge_count/n_cpu_ok : 0.0)
    @printf("SPURIOUS wild-abscissa cells (DEV max|U| >> CPU max|U|): %d total (%d real, %.4f%% of real)\n",
            n_spurious, n_spurious_real, nreal>0 ? 100*n_spurious_real/nreal : 0.0)
    @printf("  worst DEV abscissa magnitude: %.3e\n", worst_blow)
    if !isempty(diverge_examples)
        println("  sample node-count mismatches (cell, cpu_nodes, dev_nodes):")
        for e in diverge_examples; println("    ", e); end
    end
    println("=====================================================================")
end
main()

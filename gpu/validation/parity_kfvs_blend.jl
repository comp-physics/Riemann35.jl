# parity_kfvs_blend.jl — validate the full-cone θ* blend (increment D) on REAL
# data. NO GPU required (runs the device functions on the host). Run first.
#
# Reproduces verify_kfvs.jl Test C on real interior stencils:
#   Ua  = measure_update_3d_dev  (the anchor — full-cone realizable BY CONSTRUCTION)
#   Uho = a raw high-order update = MC - λ*(F_iface(C,R±) - F_iface(L±,C)) summed
#         over the 3 axes (kinetic-FVS interface fluxes; CAN exit the cone)
#   U(θ) = (1-θ)Ua + θ Uho ; θ* = largest θ keeping U(θ) FULL-CONE realizable.
#
# Reports:
#   (a) mean θ* and fraction of cells with θ*=1 (target ~0.97-0.99 mean)
#   (b) U(θ*) full-cone realizable on ~100% of cells (0 cone exits)
#   (c) THE KEY DEMO: the SAME blend with the MARGINAL-only θ* (Track-2) lets a
#       NONZERO fraction of blended states exit the CROSS-moment cone — quantified.
#   plus predicate correctness vs CPU is_realizable.

using Riemann35
using JLD2, Printf, Random, LinearAlgebra
R = Riemann35

include(joinpath(@__DIR__, "..", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev
include(joinpath(@__DIR__, "..", "kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev
include(joinpath(@__DIR__, "..", "kfvs_blend_dev.jl"))
using .KFVSBlendDev

const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

# store a device quadrature into a per-cell (27,4) buffer; returns node count
function invert_store(M)
    S = zeros(27,4)
    st!(NW,UX,UY,UZ,ci,q,w,ux,uy,uz)=(S[q,1]=w; S[q,2]=ux; S[q,3]=uy; S[q,4]=uz; nothing)
    Nn = chyqmom_nodes_3d_store_dev!(st!, nothing,nothing,nothing,nothing, 1, M)
    return S, Nn
end

# kinetic-FVS split flux of an axis: F±_n = Σ_{k: ±U_axis>0} n_k U_axis * u^triple
# axis ∈ (1=x,2=y,3=z); returns (Fp, Fm) as length-35 vectors.
function kfvs_split_axis(S, Nn, axis)
    Fp = zeros(35); Fm = zeros(35)
    for k in 1:Nn
        nk = S[k,1]; ua = S[k,1+axis]
        for m in 1:35
            (i,j,l) = TRIPLES[m]
            val = nk * ua * (S[k,2]^i) * (S[k,3]^j) * (S[k,4]^l)
            ua >= 0 ? (Fp[m]+=val) : (Fm[m]+=val)
        end
    end
    Fp, Fm
end
iface(SL,NL, SR,NR, axis) = kfvs_split_axis(SL,NL,axis)[1] .+ kfvs_split_axis(SR,NR,axis)[2]

function main()
    Random.seed!(2028)
    f = "/storage/project/r-sbryngelson3-0/sbryngelson3/debug/ma100_np128_ma100_o1.jld2"
    data = jldopen(f,"r") do jf
        arr=nothing; for k in keys(jf); v=jf[k]; if v isa AbstractArray && ndims(v)==4 && size(v,4)==35; arr=v; break; end; end; arr
    end
    nx,ny,nz,_ = size(data)
    B=48; i0=40; j0=40; k0=40
    sub = Float64.(data[i0+1:i0+B, j0+1:j0+B, k0+1:k0+B, :])
    println("Subvolume: ", size(sub))

    # ---- (0) predicate correctness vs CPU is_realizable on this block ----
    npc=0; nmis=0
    for i in 1:B, j in 1:B, k in 1:B
        M=ntuple(t->sub[i,j,k,t],Val(35)); M[1]>0||continue
        (R.is_realizable(collect(M);lam_min=0.0) != state_realizable_fullcone_dev(M)) && (nmis+=1)
        npc+=1
    end
    @printf("predicate vs CPU is_realizable: %d/%d agree (%.4f%% mismatch, cone-boundary eig floor)\n",
            npc-nmis, npc, 100*nmis/max(npc,1))

    # ---- pre-invert all cells (device store) ----
    Sarr = Array{Matrix{Float64}}(undef, B,B,B)
    Narr = zeros(Int, B,B,B)
    for i in 1:B, j in 1:B, k in 1:B
        M=ntuple(t->sub[i,j,k,t],Val(35))
        if M[1] > 0.0; Sarr[i,j,k],Narr[i,j,k] = invert_store(M); else; Sarr[i,j,k]=zeros(27,4); Narr[i,j,k]=0; end
    end

    CFL=0.4
    nsten=0
    θf_sum=0.0; nθf1=0; nfull_exit=0
    θm_sum=0.0; nθm1=0; nmarg_crossexit=0    # marginal-θ* blended states OUTSIDE the cross-moment cone
    nmarg_ge_full=0                          # marginal θ* >= full-cone θ* (marginal too permissive)
    θf_hist=Float64[]
    for i in 2:B-1, j in 2:B-1, k in 2:B-1
        cells = ((i,j,k),(i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1))
        all(c->Narr[c...]>0, cells) || continue
        SC=Sarr[i,j,k]; NC=Narr[i,j,k]
        # 3D CFL over the stencil
        smax=1e-300
        for c in cells; S=Sarr[c...]; for q in 1:Narr[c...]; smax=max(smax, abs(S[q,2])+abs(S[q,3])+abs(S[q,4])); end; end
        λ = CFL/smax
        # anchor via device measure_update_3d
        slots = cells
        cnt(s)=Narr[slots[s]...]
        gw(s,q)=Sarr[slots[s]...][q,1]; gx(s,q)=Sarr[slots[s]...][q,2]
        gy(s,q)=Sarr[slots[s]...][q,3]; gz(s,q)=Sarr[slots[s]...][q,4]
        (Ua, minw) = measure_update_3d_dev(gw,gx,gy,gz,cnt,λ)
        minw < -1e-12 && continue   # anchor CFL guard (should not fire at 0.4)
        # high-order candidate: MC - λ*Σ_axis (F_iface(C,R) - F_iface(L,C))
        MC = ntuple(t->sub[i,j,k,t],Val(35))
        dU = zeros(35)
        for (axis, (cl, cr)) in ((1,((i-1,j,k),(i+1,j,k))), (2,((i,j-1,k),(i,j+1,k))), (3,((i,j,k-1),(i,j,k+1))))
            SL=Sarr[cl...]; NL=Narr[cl...]; SR=Sarr[cr...]; NR=Narr[cr...]
            Fr = iface(SC,NC, SR,NR, axis)
            Fl = iface(SL,NL, SC,NC, axis)
            dU .+= (Fr .- Fl)
        end
        Uho = ntuple(t-> MC[t] - λ*dU[t], Val(35))

        (θf, Uf) = theta_star_blend_fullcone_dev(Ua, Uho)
        (θm, Um) = theta_star_blend_marginal_dev(Ua, Uho)
        nsten += 1
        θf_sum += θf; θf >= 1-1e-9 && (nθf1+=1)
        length(θf_hist)<200000 && push!(θf_hist, θf)
        state_realizable_fullcone_dev(Uf) || (nfull_exit += 1)
        θm_sum += θm; θm >= 1-1e-9 && (nθm1+=1)
        # KEY: does the MARGINAL-θ* blended state exit the CROSS-moment cone?
        state_realizable_fullcone_dev(Um) || (nmarg_crossexit += 1)
        θm > θf + 1e-9 && (nmarg_ge_full += 1)
    end

    @printf("\n=============== FULL-CONE θ* BLEND (real interior stencils, CFL=%.2f) ===============\n", CFL)
    @printf("stencils evaluated                       : %d\n", nsten)
    @printf("(a) mean full-cone θ*                    : %.4f   (θ*=1 unlimited on %d = %.2f%%)\n",
            θf_sum/max(nsten,1), nθf1, 100*nθf1/max(nsten,1))
    @printf("(b) U(θ*) full-cone realizable           : %d/%d (%.4f%%)  [cone EXITS: %d]\n",
            nsten-nfull_exit, nsten, 100*(nsten-nfull_exit)/max(nsten,1), nfull_exit)
    if !isempty(θf_hist)
        sort!(θf_hist); q(p)=θf_hist[clamp(round(Int,p*length(θf_hist)),1,length(θf_hist))]
        @printf("    full-cone θ* percentiles             : p1=%.3f p10=%.3f p50=%.3f p90=%.3f\n", q(0.01),q(0.10),q(0.50),q(0.90))
    end
    @printf("\n(c) KEY DEMONSTRATION — marginal-only θ* (Track-2) is INSUFFICIENT:\n")
    @printf("    mean marginal θ*                     : %.4f   (θ*=1 on %d = %.2f%%)\n",
            θm_sum/max(nsten,1), nθm1, 100*nθm1/max(nsten,1))
    @printf("    marginal θ* > full-cone θ* (too permissive) : %d/%d (%.4f%%)\n",
            nmarg_ge_full, nsten, 100*nmarg_ge_full/max(nsten,1))
    @printf("    >>> marginal-θ* blended states OUTSIDE the CROSS-moment cone : %d/%d (%.4f%%)\n",
            nmarg_crossexit, nsten, 100*nmarg_crossexit/max(nsten,1))
    @printf("        (the full-cone limiter drives this to %d — the justification for full-cone)\n", nfull_exit)
    println("====================================================================================")
end
main()

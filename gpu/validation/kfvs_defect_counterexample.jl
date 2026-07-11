# kfvs_defect_counterexample.jl — Phase I.3–I.4: the hard-crossing anchor-exit
# counterexamples and their cone spectra, over ALL captured stage-2 failing cells.
#
# Loads the raw failing stencils (kfvs_hardcross_stencil_raw.jld2). For EACH failing
# cell, using only its 7 raw-moment cells:
#   U^Q  = nonneg-measure KFVS update (thm:kfvs-idp);  U^F3 = captured Mlo (flux form);
#   D = U^F3 - U^Q;  cone spectrum margin(U^Q + sD), exit fraction s*;  defect channel
#   shares (transverse-4th S_D={m202,m112,m022,m004} vs degree-2 {m200,m020,m002});
#   center-cell quadrature reproduction count.
# Reports the DISTRIBUTION (what decides Route A) and writes both the full arrays and a
# single representative counterexample to kfvs_defect_counterexample.jld2 (§7.1 artifact).
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, Printf, JLD2, Statistics
include(joinpath(@__DIR__, "..", "kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev: measure_update_3d_dev, accum35_node
using .KFVSMeasureUpdateDev.KFVSInversionDev: chyqmom_nodes_3d_dev

raw = load(joinpath(@__DIR__, "kfvs_hardcross_stencil_raw.jld2"))
ncap = raw["ncap"]; λ = raw["lam"][1]
TRIP = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),
 (2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),
 (1,1,2),(0,1,3),(0,2,2))
lab(n) = "m$(TRIP[n][1])$(TRIP[n][2])$(TRIP[n][3])"
SD = (22,33,35,25); DEG2 = (3,10,20)            # S_D transverse-4th; degree-2 covariance
col(A,c) = A[:,c]

sstar_all = fill(NaN, ncap); mQ_all = zeros(ncap); mF_all = zeros(ncap)
sdshare_all = zeros(ncap); deg2share_all = zeros(ncap); repro_all = zeros(Int, ncap)
minw_all = zeros(ncap); idmax_all = zeros(ncap)
UQs = zeros(35, ncap); Ds = zeros(35, ncap)

for c in 1:ncap
    cells = (col(raw["C"],c), col(raw["Lx"],c), col(raw["Rx"],c), col(raw["Ly"],c),
             col(raw["Ry"],c), col(raw["Lz"],c), col(raw["Rz"],c))
    nodes = map(x -> chyqmom_nodes_3d_dev(ntuple(q->x[q],Val(35))), cells)
    gw(s,q)=nodes[s][1][q]; gx(s,q)=nodes[s][2][q]; gy(s,q)=nodes[s][3][q]
    gz(s,q)=nodes[s][4][q]; cn(s)=nodes[s][5]
    UQt, minw = measure_update_3d_dev(gw,gx,gy,gz,cn, λ)
    UQ = collect(UQt); Mlo = col(raw["Mlos"], c); D = Mlo .- UQ
    UQs[:,c] = UQ; Ds[:,c] = D; minw_all[c] = minw
    idmax_all[c] = maximum(abs.(Mlo .- (UQ .+ D)))
    mQ_all[c] = realizability_margin(UQ); mF_all[c] = realizability_margin(Mlo)
    # cone exit fraction
    prev = mQ_all[c]
    for si in 1:200
        s = si/200; m = realizability_margin(UQ .+ s .* D)
        if prev >= 0 && m < 0
            sstar_all[c] = (si-1)/200 + (prev/(prev-m))/200; break
        end
        prev = m
    end
    Dn = abs.(D) ./ max(UQ[1], 1e-300); tot = sum(Dn)
    sdshare_all[c] = sum(Dn[n] for n in SD)/tot; deg2share_all[c] = sum(Dn[n] for n in DEG2)/tot
    # reproduction count at center cell
    nnC,nx,ny,nz,NnC = nodes[1]
    Ct = ntuple(_->0.0,Val(35)); for q in 1:NnC; Ct = accum35_node(Ct,nnC[q],nx[q],ny[q],nz[q]); end
    Cc = cells[1]; rel = abs.(collect(Ct).-Cc)./max.(abs.(Cc),1e-300)
    repro_all[c] = count(rel .< 1e-8)
end

pct(x)=100x
fin(v)=v[isfinite.(v)]
qq(v,p)= (w=fin(v); isempty(w) ? NaN : quantile(w,p))
finite_s = sstar_all[.!isnan.(sstar_all)]
nUQfin = count(c->all(isfinite, UQs[:,c]), 1:ncap)
@printf("=== hard-crossing anchor-exit DISTRIBUTION over %d captured stage-2 cells ===\n", ncap)
@printf("(capture restricted to realizable-INPUT cells: margin(Mc)>=0 -> anchor update exits)\n")
@printf("identity max|U^F3-(U^Q+D)|:     max %.1e\n", maximum(fin(idmax_all)))
@printf("MEASURE U^Q realizable-by-construction?  in-cone %d/%d, minw min %.3e  => nonneg-measure holds: %s\n",
        count(mQ_all .>= 0), ncap, minimum(fin(minw_all)), all(fin(minw_all) .>= -1e-12) ? "YES" : "NO (CFL bound violated)")
@printf("  measure U^Q margin: median %.3e   finite-U^Q cells %d/%d\n", median(fin(mQ_all)), nUQfin, ncap)
@printf("FLUX U^F3 margin:    out %d/%d  (worst %.3e, median %.3e)\n",
        count(mF_all .< 0), ncap, minimum(fin(mF_all)), median(fin(mF_all)))
@printf("cone exit s* (U^Q+sD): median %.4f  IQR [%.4f, %.4f]  min %.4f  (n_cross=%d/%d)\n",
        median(finite_s), qq(finite_s,0.25), qq(finite_s,0.75), minimum(finite_s), length(finite_s), ncap)
@printf("defect share transverse-4th S_D: median %.1f%%  IQR [%.1f%%, %.1f%%]  min %.1f%%\n",
        pct(qq(sdshare_all,0.5)), pct(qq(sdshare_all,0.25)), pct(qq(sdshare_all,0.75)), pct(minimum(fin(sdshare_all))))
@printf("defect share degree-2:           median %.1f%%  max %.1f%%\n",
        pct(qq(deg2share_all,0.5)), pct(maximum(fin(deg2share_all))))
@printf("quadrature reproduction count:   median %d/35  range [%d, %d]\n",
        Int(round(qq(repro_all .|> Float64, 0.5))), minimum(repro_all), maximum(repro_all))
# dominant defect channel frequency (manual count, no StatsBase dep)
domch = [argmax(abs.(Ds[:,c])) for c in 1:ncap]
freq = Dict{Int,Int}(); for n in domch; freq[n] = get(freq,n,0)+1; end
cm = sort(collect(freq); by=x->-x[2])
@printf("dominant defect channel (per cell): %s\n", join(["$(lab(n))×$(v)" for (n,v) in cm[1:min(5,end)]], "  "))

# representative = worst-margin cell
wc = argmin(mF_all)
out = joinpath(@__DIR__, "kfvs_defect_counterexample.jld2")
jldsave(out;
    ncap=ncap, lam=raw["lam"], dt=raw["dt"], dx=raw["dx"], halo=raw["halo"], Ma=raw["Ma"], s3max=raw["s3max"], rk_stage=2,
    cells=raw["cells"], margins_flux=mF_all, margins_measure=mQ_all, measure_minw=minw_all,
    U_measure=UQs, U_flux=raw["Mlos"], defect=Ds, s_star=sstar_all,
    sd_share=sdshare_all, deg2_share=deg2share_all, repro_count=repro_all,
    # representative single cell (worst margin) + its raw stencil for N1/N4 offline solves
    rep_cell=raw["cells"][:,wc], rep_C=raw["C"][:,wc], rep_Lx=raw["Lx"][:,wc], rep_Rx=raw["Rx"][:,wc],
    rep_Ly=raw["Ly"][:,wc], rep_Ry=raw["Ry"][:,wc], rep_Lz=raw["Lz"][:,wc], rep_Rz=raw["Rz"][:,wc],
    ic_bg=raw["ic_bg"], ic_Mt=raw["ic_Mt"], ic_Mb=raw["ic_Mb"])
@printf("\nwrote %s  (§7.1 artifact: %d cells + worst-margin representative)\n", out, ncap)

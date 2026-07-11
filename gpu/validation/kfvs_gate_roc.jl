# kfvs_gate_roc.jl — Phase I: the STAGED EXACT GATE over the full stage-2 population
# (passing AND failing cells), from the snapshotted stage-2 input state.
#
# For every realizable-input interior cell of the captured stage-2 state:
#   Mlo   = F3 anchor update, reconstructed via the solver's own _kfvs_face_flux_tup
#           (validated: matches the 278 captured Mlo to <1e-10 rel);
#   c_CFL = 1 − λ·max_center-node Σ|u|  (cheap certificate coordinate);
#   U^Q   = measure update (computed for FAILS, to route them).
# Staged exact gate (per your recommendation — no predicted s_pred):
#   (1) margin(Mlo) ≥ 0                    -> PASS: plain F3            [exact, cheap]
#   (2) else margin(U^Q) ≥ 0              -> COMPLETE/REDISTRIBUTE      [class A/B]
#   (3) else                              -> ENRICH / projection35      [class C]
# Reports the operating envelope (what fraction of the domain hits each branch), the
# false-positive rate on passing cells, and whether c_CFL alone would mis-detect.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, Printf, JLD2, Statistics
include(joinpath(@__DIR__, "..", "kfvs_measure_update_dev.jl"))
using .KFVSMeasureUpdateDev: measure_update_3d_dev
using .KFVSMeasureUpdateDev.KFVSInversionDev: chyqmom_nodes_3d_dev

raw = load(joinpath(@__DIR__, "kfvs_hardcross_stencil_raw.jld2"))
M = raw["state"]; N = raw["N"]; h = raw["halo"]; λ = raw["lam"][1]; Ma = raw["Ma"]; s3max = raw["s3max"]
capMlo = raw["Mlos"]; capcell = raw["cells"]; ncap = raw["ncap"]
nfx, nfy, nfz = size(M,1), size(M,2), size(M,3)
cellrow(i,j,k) = ntuple(q->M[i,j,k,q], Val(35))
kfvs = Riemann35._kfvs_face_flux_tup

# Mlo from a 7-cell stencil (C + 6 face neighbors), matching the residual's first-order
# KFVS anchor update: Mlo = C - λ[(F(C,Rx)-F(Lx,C)) + (F(C,Ry)-F(Ly,C)) + (F(C,Rz)-F(Lz,C))].
mlo_stencil(C,Lx,Rx,Ly,Ry,Lz,Rz) = ntuple(q -> C[q]
    - λ*(kfvs(C,Rx,1,Ma,s3max)[q]-kfvs(Lx,C,1,Ma,s3max)[q])
    - λ*(kfvs(C,Ry,2,Ma,s3max)[q]-kfvs(Ly,C,2,Ma,s3max)[q])
    - λ*(kfvs(C,Rz,3,Ma,s3max)[q]-kfvs(Lz,C,3,Ma,s3max)[q]), Val(35))
stencil(i,j,k) = (cellrow(i,j,k),cellrow(i-1,j,k),cellrow(i+1,j,k),cellrow(i,j-1,k),cellrow(i,j+1,k),cellrow(i,j,k-1),cellrow(i,j,k+1))

# --- validate reconstruction against the 278 captured Mlo (from THEIR stencils) ----
function validate()
    vmax = 0.0
    for c in 1:ncap
        C=raw["C"][:,c]; Lx=raw["Lx"][:,c]; Rx=raw["Rx"][:,c]; Ly=raw["Ly"][:,c]
        Ry=raw["Ry"][:,c]; Lz=raw["Lz"][:,c]; Rz=raw["Rz"][:,c]
        rec = collect(mlo_stencil(ntuple(q->C[q],Val(35)),ntuple(q->Lx[q],Val(35)),ntuple(q->Rx[q],Val(35)),
            ntuple(q->Ly[q],Val(35)),ntuple(q->Ry[q],Val(35)),ntuple(q->Lz[q],Val(35)),ntuple(q->Rz[q],Val(35))))
        cap = capMlo[:,c]
        vmax = max(vmax, maximum(abs.(rec.-cap))/max(maximum(abs.(cap)),1e-300))
    end
    vmax
end
vmax = validate()
@printf("Mlo reconstruction vs 278 captured: max rel diff = %.2e  (%s)\n", vmax, vmax<1e-8 ? "VALIDATED" : "MISMATCH — flux formula off")

# --- full population: realizable-input interior cells (z restricted to interior) ---
function cCFL_st(st)
    nn,nx,ny,nz,Nn = chyqmom_nodes_3d_dev(st[1]); smax=0.0
    for q in 1:Nn; nn[q]>0 && (smax=max(smax,abs(nx[q])+abs(ny[q])+abs(nz[q]))); end
    1 - λ*smax
end
function UQ_st(st)
    nd = map(chyqmom_nodes_3d_dev, st)
    gw(s,q)=nd[s][1][q]; gx(s,q)=nd[s][2][q]; gy(s,q)=nd[s][3][q]; gz(s,q)=nd[s][4][q]; cn(s)=nd[s][5]
    collect(measure_update_3d_dev(gw,gx,gy,gz,cn,λ)[1])
end
SD=(22,33,35,25); DEG2=(3,10,20)
function sweep()
    nin=0; npass=0; fp=0; fn=0
    F=Dict{Symbol,Vector}(:cell=>Vector{Vector{Int}}(), :mF=>Float64[], :mQ=>Float64[], :cCFL=>Float64[],
        :nspeed=>Float64[], :cls=>Int[], :sd=>Float64[], :UQ=>Vector{Vector{Float64}}(), :Mlo=>Vector{Vector{Float64}}(),
        :Cst=>Vector{Vector{Float64}}())
    for k in 2:N-1, j in 1:N, i in 1:N            # z interior (un-haloed); x,y haloed
        ih,jh = i+h, j+h
        st = stencil(ih,jh,k)
        realizability_margin(collect(st[1])) >= 0 || continue     # realizable input only
        nin += 1
        mlo = collect(mlo_stencil(st...))
        mf = all(isfinite,mlo) ? realizability_margin(mlo) : -Inf
        cc = cCFL_st(st)
        if mf >= 0
            npass += 1; cc<0 && (fp += 1)
        else
            cc>=0 && (fn += 1)
            UQ = UQ_st(st); mq = realizability_margin(UQ)
            cls = (cc>=0 && mq>=0) ? 1 : (cc<0 && mq>=0) ? 2 : 3
            nn,nx,ny,nz,Nn = chyqmom_nodes_3d_dev(st[1]); sm=0.0
            for q in 1:Nn; nn[q]>0 && (sm=max(sm,abs(nx[q])+abs(ny[q])+abs(nz[q]))); end
            D=mlo.-UQ; Dn=abs.(D)./max(UQ[1],1e-300)
            push!(F[:cell],[i,j,k]); push!(F[:mF],mf); push!(F[:mQ],mq); push!(F[:cCFL],cc)
            push!(F[:nspeed],sm); push!(F[:cls],cls); push!(F[:sd],sum(Dn[n] for n in SD)/sum(Dn))
            push!(F[:UQ],UQ); push!(F[:Mlo],mlo); push!(F[:Cst],collect(st[1]))
        end
    end
    (nin,npass,fp,fn,F)
end
nin,npass,fp_cCFL_pass,fn_cCFL_fail,F = sweep()
nfail=length(F[:mF]); nb2=count(<(3),F[:cls]); nb3=count(==(3),F[:cls])

@printf("\n=== staged-gate operating envelope over %d realizable-input z-interior cells ===\n", nin)
@printf("PASS (branch 1, plain F3):          %5d  (%.1f%%)\n", npass, 100npass/nin)
@printf("FAIL total:                          %5d  (%.1f%%)\n", nfail, 100nfail/nin)
@printf("  branch 2 COMPLETE (U^Q in-cone):   %5d  (%.1f%% of fails)  [class A+B]\n", nb2, 100nb2/max(nfail,1))
@printf("  branch 3 ENRICH   (U^Q out):       %5d  (%.1f%% of fails)  [class C]\n", nb3, 100nb3/max(nfail,1))
@printf("false-positive if c_CFL<0 alone gated PASS cells: %d/%d (%.1f%%) — c_CFL over-flags good cells\n",
        fp_cCFL_pass, npass, 100fp_cCFL_pass/max(npass,1))
@printf("mis-detect if c_CFL>=0 alone passed FAIL cells:   %d/%d (%.1f%%) — c_CFL alone MISSES these (need endpoint check)\n",
        fn_cCFL_fail, nfail, 100fn_cCFL_fail/max(nfail,1))
@printf("captured fails=%d (contaminated by stage-3); CLEAN stage-2 z-interior fails=%d\n", ncap, nfail)
@printf("class split (clean): A(cc>=0,UQ in)=%d  B(cc<0,UQ in)=%d  C(UQ out)=%d\n",
        count(==(1),F[:cls]),count(==(2),F[:cls]),count(==(3),F[:cls]))
using Statistics
for cl in 1:3
    m = F[:cls].==cl; any(m) || continue
    @printf("  class %d (n=%d): transverse-4th share median %.1f%%  node-speed median %.0f\n",
            cl,count(m),100median(F[:sd][m]),median(F[:nspeed][m]))
end

# --- write the CLEAN decision-grade artifact (from the state; retires the 278 path) --
mat(v)= isempty(v) ? zeros(35,0) : reduce(hcat,v)
worst(cl)= (idx=findall(==(cl),F[:cls]); isempty(idx) ? 0 : idx[argmin(F[:mF][idx])])
out=joinpath(@__DIR__,"kfvs_defect_counterexample.jld2")
base=Dict{String,Any}("nfail"=>nfail,"n_realizable_input"=>nin,"n_pass"=>npass,"lam"=>raw["lam"],
    "dt"=>raw["dt"],"dx"=>raw["dx"],"halo"=>h,"Ma"=>Ma,"s3max"=>s3max,"rk_stage"=>2,"source"=>"stage-2 state snapshot (clean)",
    "cell"=>mat([Float64.(c) for c in F[:cell]]),"margin_flux"=>F[:mF],"margin_measure"=>F[:mQ],"c_CFL"=>F[:cCFL],
    "node_speed"=>F[:nspeed],"class"=>F[:cls],"sd_share"=>F[:sd],"U_measure"=>mat(F[:UQ]),"U_flux"=>mat(F[:Mlo]),
    "center_state"=>mat(F[:Cst]),
    "gate_pass"=>npass,"gate_complete"=>nb2,"gate_enrich"=>nb3,"cCFL_falsepos_pass"=>fp_cCFL_pass,"cCFL_missed_fail"=>fn_cCFL_fail)
reps=Dict{String,Any}()
for (nm,cl) in (("A",1),("B",2),("C",3))
    wi=worst(cl); wi==0 && continue
    reps["rep_$(nm)_center"]=F[:Cst][wi]; reps["rep_$(nm)_Uflux"]=F[:Mlo][wi]; reps["rep_$(nm)_Umeasure"]=F[:UQ][wi]
    reps["rep_$(nm)_cell"]=F[:cell][wi]; reps["rep_$(nm)_margin_flux"]=F[:mF][wi]
end
jldopen(out,"w") do f; for (k,v) in base; f[k]=v; end; for (k,v) in reps; f[k]=v; end; end
@printf("wrote CLEAN %s (%d fails from state snapshot; A/B/C representatives)\n", out, nfail)

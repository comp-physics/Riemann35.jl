# kfvs_defect_counterexample.jl — Phase I.2–I.5 (decision-grade): the hard-crossing
# anchor-exit distribution, the three-way certificate/cone classification, and the
# two-coordinate N2 sensor (c_CFL, s_pred).
#
# For each captured realizable-input stage-2 exit cell, using only its 7 raw-moment cells:
#   U^Q  = nonneg-measure KFVS update (retained w = n·(1−λΣ|u|)); minw, margin(U^Q).
#   D    = U^F3 − U^Q (the EJECTING defect / true segment); cross-checked NON-tautologically
#          against the independently-computed reproduction defect M − M̃ (they agree to
#          ~1e-13 RELATIVE on all three classes ⇒ the reproduction defect IS the ejecting one).
#   U^F3 = captured Mlo (flux form); margin(U^F3).
#   c_CFL = 1 − λ·max_{center nodes}(|ux|+|uy|+|uz|)   (node-speed CFL margin; <0 ⇔ minw<0).
#   s*    = exact root of margin(U^Q+sD)=0;  s_pred = −margin(U^Q)/(dmargin/ds|_0)
#           (Hellmann–Feynman first-order eigenvalue crossing — no eigenvector extraction).
# Three-way class:  A = certificate holds & U^Q in-cone;  B = certificate fails, U^Q in-cone;
#                   C = certificate fails & U^Q out-of-cone.
# Writes a decision-grade kfvs_defect_counterexample.jld2: all per-cell arrays + one raw
# stencil + center node cloud from EACH class + the worst-CFL stencil (§7.1).
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
lab(n)="m$(TRIP[n][1])$(TRIP[n][2])$(TRIP[n][3])"
SD=(22,33,35,25); DEG2=(3,10,20)
col(A,c)=A[:,c]; tup(v)=ntuple(q->v[q],Val(35))

mQ=zeros(ncap); mF=zeros(ncap); minw=zeros(ncap); cCFL=zeros(ncap); nspeed=zeros(ncap)
sstar=fill(NaN,ncap); spred=fill(NaN,ncap); sdsh=fill(NaN,ncap); deg2sh=fill(NaN,ncap)
repro=zeros(Int,ncap); idresid=fill(NaN,ncap); cls=zeros(Int,ncap)
UQs=zeros(35,ncap); Ds=zeros(35,ncap)
ε=1e-3
for c in 1:ncap
    cells=(col(raw["C"],c),col(raw["Lx"],c),col(raw["Rx"],c),col(raw["Ly"],c),col(raw["Ry"],c),col(raw["Lz"],c),col(raw["Rz"],c))
    nodes=map(x->chyqmom_nodes_3d_dev(tup(x)),cells)
    gw(s,q)=nodes[s][1][q]; gx(s,q)=nodes[s][2][q]; gy(s,q)=nodes[s][3][q]; gz(s,q)=nodes[s][4][q]; cn(s)=nodes[s][5]
    UQt,mw=measure_update_3d_dev(gw,gx,gy,gz,cn,λ); UQ=collect(UQt)
    Mlo=col(raw["Mlos"],c)
    # independent reproduction defect D = C - C̃ (center requad)
    nn,nx,ny,nz,NnC=nodes[1]; Ct=ntuple(_->0.0,Val(35))
    for q in 1:NnC; Ct=accum35_node(Ct,nn[q],nx[q],ny[q],nz[q]); end
    Dindep=cells[1].-collect(Ct)
    Dupd=Mlo.-UQ
    # relative agreement of reproduction defect (M-M̃) vs the ejecting update defect
    idresid[c]=maximum(abs.(Dindep.-Dupd))/max(maximum(abs.(Dupd)),1e-300)
    D=Dupd; UQs[:,c]=UQ; Ds[:,c]=D          # EJECTING defect = U^F3 - U^Q (the true segment)
    minw[c]=mw; mQ[c]=realizability_margin(UQ); mF[c]=realizability_margin(Mlo)
    # center-node CFL speed
    smax=0.0; for q in 1:NnC; nn[q]>0 && (smax=max(smax,abs(nx[q])+abs(ny[q])+abs(nz[q]))); end
    nspeed[c]=smax; cCFL[c]=1-λ*smax
    # true s* and s_pred (only meaningful from an in-cone base)
    if mQ[c]>=0
        prev=mQ[c]
        for si in 1:400
            s=si/400; m=realizability_margin(UQ.+s.*D)
            if prev>=0 && m<0; sstar[c]=(si-1)/400+(prev/(prev-m))/400; break; end
            prev=m
        end
        mε=realizability_margin(UQ.+ε.*D); slope=(mε-mQ[c])/ε
        spred[c]= slope<0 ? -mQ[c]/slope : Inf
    end
    Dn=abs.(D)./max(UQ[1],1e-300); tot=sum(Dn)
    sdsh[c]=sum(Dn[n] for n in SD)/tot; deg2sh[c]=sum(Dn[n] for n in DEG2)/tot
    rel=abs.(collect(Ct).-cells[1])./max.(abs.(cells[1]),1e-300); repro[c]=count(rel.<1e-8)
    cls[c]= (minw[c]>=-1e-12 && mQ[c]>=0) ? 1 : (minw[c]<-1e-12 && mQ[c]>=0) ? 2 : (mQ[c]<0) ? 3 : 0
end

fin(v)=v[isfinite.(v)]; qq(v,p)=(w=fin(v); isempty(w) ? NaN : quantile(w,p)); pc(x)=100x
nA=count(==(1),cls); nB=count(==(2),cls); nC=count(==(3),cls); n0=count(==(0),cls)
@printf("=== hard-crossing anchor-exit: DECISION-GRADE distribution over %d cells (Ma=100 32³) ===\n",ncap)
@printf("THREE-WAY CLASS:  A(cert✓,in-cone)=%d   B(cert✗,in-cone)=%d   C(cert✗,out)=%d   other=%d\n",nA,nB,nC,n0)
@printf("reproduction defect (M-M̃) vs ejecting defect (U^F3-U^Q), rel agreement by class:\n")
for (nm,mask) in (("A",cls.==1),("B",cls.==2),("C",cls.==3))
    any(mask) && @printf("  class %s: median rel diff %.2e  (=0 => reproduction defect IS the ejecting defect)\n",nm,median(idresid[mask]))
end
@printf("  CFL certificate (minw>=0) holds on %d/%d; c_CFL<0 on %d/%d (consistency: %s)\n",
        count(minw.>=-1e-12),ncap,count(cCFL.<0),ncap, count((minw.< -1e-12).==(cCFL.<0))==ncap ? "EXACT" : "approx")
@printf("NODE-SPEED (center max Σ|u|):  median %.1f  p90 %.1f  MAX %.1f   (λ·max_C = %.0f, pathological near-boundary inversion)\n",
        median(nspeed),qq(nspeed,0.9),maximum(nspeed),λ*maximum(nspeed))
@printf("FLUX U^F3 out %d/%d (worst %.2e).  s*: median %.3f min %.3f (n=%d)\n",
        count(mF.<0),ncap,minimum(fin(mF)),median(fin(sstar)),minimum(fin(sstar)),count(!isnan,sstar))
for (nm,mask) in (("A",cls.==1),("B",cls.==2),("C",cls.==3))
    any(mask) || continue
    @printf("  class %s (n=%d): transverse-4th share median %.1f%%   degree-2 median %.1f%%   repro median %d/35   node-speed median %.0f\n",
            nm,count(mask),pc(qq(sdsh[mask],0.5)),pc(qq(deg2sh[mask],0.5)),Int(round(qq(Float64.(repro[mask]),0.5))),median(nspeed[mask]))
end
# N2 sensor: does s_pred predict s*? (in-cone cells with a finite crossing)
ok = (.!isnan.(sstar)) .& isfinite.(spred)
if count(ok)>2
    r=cor(spred[ok],sstar[ok])
    @printf("N2 sensor s_pred vs true s*: Pearson r=%.3f over %d in-cone cells; s_pred median %.3f (true s* median %.3f)\n",
            r,count(ok),median(spred[ok]),median(sstar[ok]))
end
domch=[argmax(abs.(Ds[:,c])) for c in 1:ncap]; fr=Dict{Int,Int}(); for n in domch; fr[n]=get(fr,n,0)+1; end
@printf("dominant defect channel: %s\n",join(["$(lab(n))×$(v)" for (n,v) in sort(collect(fr);by=x->-x[2])[1:min(5,end)]]," "))

# representatives: one per class + worst-CFL; save raw stencil + center node cloud
repcloud(c)=begin nd=chyqmom_nodes_3d_dev(tup(col(raw["C"],c))); (collect(nd[1]),collect(nd[2]),collect(nd[3]),collect(nd[4]),nd[5]) end
pick(mask)= any(mask) ? findmax(ifelse.(mask,-mF,-Inf))[2] : 0   # worst-flux-margin member of class
rA=pick(cls.==1); rB=pick(cls.==2); rC=pick(cls.==3); rCFL=argmin(cCFL)
reps=Dict{String,Any}()
for (nm,ci) in (("A",rA),("B",rB),("C",rC),("worstCFL",rCFL))
    ci==0 && continue
    cl=repcloud(ci)
    reps["rep_$(nm)_stencil"]=hcat(col(raw["C"],ci),col(raw["Lx"],ci),col(raw["Rx"],ci),col(raw["Ly"],ci),col(raw["Ry"],ci),col(raw["Lz"],ci),col(raw["Rz"],ci))
    reps["rep_$(nm)_nodes"]=hcat(cl[1],cl[2],cl[3],cl[4]); reps["rep_$(nm)_ncount"]=cl[5]
    reps["rep_$(nm)_cell"]=raw["cells"][:,ci]
end
out=joinpath(@__DIR__,"kfvs_defect_counterexample.jld2")
base=Dict{String,Any}("ncap"=>ncap,"lam"=>raw["lam"],"dt"=>raw["dt"],"dx"=>raw["dx"],"halo"=>raw["halo"],
    "Ma"=>raw["Ma"],"s3max"=>raw["s3max"],"rk_stage"=>2,"cells"=>raw["cells"],"class"=>cls,
    "margin_measure"=>mQ,"margin_flux"=>mF,"measure_minw"=>minw,"c_CFL"=>cCFL,"node_speed"=>nspeed,
    "s_star"=>sstar,"s_pred"=>spred,"sd_share"=>sdsh,"deg2_share"=>deg2sh,"repro_count"=>repro,
    "defect_eject"=>Ds,"U_measure"=>UQs,"U_flux"=>raw["Mlos"],"repro_vs_eject_reldiff"=>idresid)
jldopen(out,"w") do f
    for (k,v) in base; f[k]=v; end
    for (k,v) in reps; f[k]=v; end
end
@printf("\nwrote %s  (decision-grade: %d cells + A/B/C/worstCFL representatives w/ node clouds)\n",out,ncap)

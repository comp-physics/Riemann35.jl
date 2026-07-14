# logjacobi_contact_diagnostic.jl — DIAGNOSTIC (not a feature): WHY does the 3D
# log-Jacobi contact gain cap at ~2-3x instead of the 1D 2.4e6x?
#
# Isolates mechanism (E) un-linearized cross-moment smearing vs (G) guard-stack
# re-smear (per-cell realizable_3D_M4 + theta*-IDP + BGK on the full state) on the
# 3D moving uniform-p x-contact from logjacobi_recon_validation.jl.
#
# Run: julia --project=<wt-logjacobi> gpu/validation/logjacobi_contact_diagnostic.jl
using Riemann35
using Riemann35: residual_ho_3d_order3!
using Riemann35.MomentIndices: IJK
using Printf, LinearAlgebra

# ---- haloed field builders (outflow x/y) ----
function haloed(n,g,ic)
    M=zeros(n+2g,n+2g,n,35)
    for k in 1:n,j in 1:n,i in 1:n; @views M[i+g,j+g,k,:].=ic(i,j,k); end
    refill!(M,n,g); M
end
function refill!(M,n,g)
    for k in 1:n
        for j in 1:n+2g,hh in 1:g; @views M[hh,j,k,:].=M[g+1,j,k,:]; @views M[n+g+hh,j,k,:].=M[n+g,j,k,:]; end
        for i in 1:n+2g,hh in 1:g; @views M[i,hh,k,:].=M[i,g+1,k,:]; @views M[i,n+g+hh,k,:].=M[i,n+g,k,:]; end
    end
end
proj!(M,n,g,Ma,s3max)=(for k in 1:n,j in 1:n,i in 1:n; M[i+g,j+g,k,:]=realizable_3D_M4(M[i+g,j+g,k,:],Ma,s3max); end)

# one SSP-RK3 step; project=false DISABLES the per-stage realizability projection (G-off)
function step!(M,n,g,dx,Ma,dt,s3max,flag;project=true)
    R=zeros(size(M));M0=copy(M);intr=(g+1:g+n,g+1:g+n,1:n,:)
    st!(Mc)=(refill!(Mc,n,g);residual_ho_3d_order3!(R,Mc,n,n,n,g,dx,dx,dx,Ma,dt;s3max=s3max,use_logjacobi_recon=flag))
    st!(M);@views M[intr...].=M0[intr...].+dt.*R[intr...];project&&proj!(M,n,g,Ma,s3max)
    st!(M);@views M[intr...].=0.75.*M0[intr...].+0.25.*(M[intr...].+dt.*R[intr...]);project&&proj!(M,n,g,Ma,s3max)
    st!(M);@views M[intr...].=(1/3).*M0[intr...].+(2/3).*(M[intr...].+dt.*R[intr...]);project&&proj!(M,n,g,Ma,s3max)
    refill!(M,n,g)
end
function metrics(M,n,g,u0,p0)
    um=0.0;pm=0.0
    for k in 1:n,j in 1:n,i in 1:n
        m=@view M[i+g,j+g,k,:];rho=m[1];rho>0||continue
        um=max(um,abs(m[2]/rho-u0));pm=max(pm,abs((m[3]-m[2]^2/rho)-p0))
    end
    (um,pm)
end

# contact IC
n=24;g=8;Ma=1.0;u0=0.5;p0=1.0;ratio=1000.0;s3max=40.0;dx=1.0/n
ic(i,j,k)=(rho=i<=n÷2 ? 1.0 : ratio;T=p0/rho;InitializeM4_35(rho,u0,0.0,0.0,T,0.0,0.0,T,0.0,T))
dt=0.2*dx/(u0+4.0*sqrt(p0));nsteps=15

println("###### log-Jacobi 3D contact cap diagnostic (E vs G) ######\n")

# ---- 0. Does this contact even HAVE nonzero cross moments? (test mechanism E's premise) ----
function cross_content(M)
    println("=== 0. cross-moment content of the field (premise for mechanism E) ===")
    crossmax=0.0; crosslist=String[]
    for q in 1:35
        (i,j,k)=IJK[q]
        ncross = ((i>0)&&(j>0)) || ((i>0)&&(k>0)) || ((j>0)&&(k>0))   # mixed (cross) moment
        if ncross
            mx=maximum(abs.(M[g+1:g+n,g+1:g+n,:,q]))
            crossmax=max(crossmax,mx)
            mx>1e-14 && push!(crosslist,"M$i$j$k=$(round(mx,sigdigits=3))")
        end
    end
    @printf("   max |cross raw moment| over the field = %.3e\n", crossmax)
    println("   nonzero cross moments: ", isempty(crosslist) ? "NONE (all cross moments identically 0)" : join(crosslist,", "))
    println("   => if NONE, mechanism E (cross smearing) CANNOT be the cap for THIS contact.\n")
    crossmax
end
cross_content(haloed(n,g,ic))

# ---- 1. BASELINE: ON vs OFF, projection ON (the shipped config) ----
println("=== 1. BASELINE (projection ON, shipped config) ===")
res=Dict{String,Tuple{Float64,Float64}}()
for (nm,flag) in (("OFF",false),("ON ",true))
    M=haloed(n,g,ic); for _ in 1:nsteps; step!(M,n,g,dx,Ma,dt,s3max,flag;project=true); end
    res[nm]=metrics(M,n,g,u0,p0)
    @printf("   %s  u=%.3e  p=%.3e\n",nm,res[nm]...)
end
@printf("   -> gain: u %.2fx   p %.2fx\n\n", res["OFF"][1]/res["ON "][1], res["OFF"][2]/res["ON "][2])

# ---- 2. VARIANT G-off: projection DISABLED, ON vs OFF ----
println("=== 2. VARIANT (projection OFF = guard-stack G disabled) ===")
res2=Dict{String,Tuple{Float64,Float64}}(); finite=true
for (nm,flag) in (("OFF",false),("ON ",true))
    M=haloed(n,g,ic); ok=true
    for _ in 1:nsteps; step!(M,n,g,dx,Ma,dt,s3max,flag;project=false); if any(!isfinite,M);ok=false;break;end; end
    if ok
        res2[nm]=metrics(M,n,g,u0,p0)
        @printf("   %s  u=%.3e  p=%.3e\n",nm,res2[nm]...)
    else
        @printf("   %s  BLEW UP (projection-off unstable)\n",nm); global finite=false
    end
end
if finite
    @printf("   -> gain (proj OFF): u %.2fx   p %.2fx\n", res2["OFF"][1]/res2["ON "][1], res2["OFF"][2]/res2["ON "][2])
    @printf("   -> ON improvement vs shipped: u %.3e->%.3e   p %.3e->%.3e\n",
            res["ON "][1],res2["ON "][1],res["ON "][2],res2["ON "][2])
    println("   INTERPRET: if ON p-error DROPS a lot with projection off => guard stack (G) was capping it.")
    println("              if ON p-error ~ same => G is NOT the cap.")
end
println()

# ---- 3. VARIANT theta*-IDP off: reconstruct faces with theta=1 (pure WENO5). ----
# We drive the residual with dt_theta=0 (=> theta_star returns 1, no limiting) but
# still advance with the physical dt. Isolates the theta*-IDP limiter as a cap.
function step_noθ!(M,n,g,dx,Ma,dt,s3max,flag;project=true)
    R=zeros(size(M));M0=copy(M);intr=(g+1:g+n,g+1:g+n,1:n,:)
    # dt=0 in the residual => all theta*=1 (pure WENO5 fluxes); advance with real dt.
    st!(Mc)=(refill!(Mc,n,g);residual_ho_3d_order3!(R,Mc,n,n,n,g,dx,dx,dx,Ma,0.0;s3max=s3max,use_logjacobi_recon=flag))
    st!(M);@views M[intr...].=M0[intr...].+dt.*R[intr...];project&&proj!(M,n,g,Ma,s3max)
    st!(M);@views M[intr...].=0.75.*M0[intr...].+0.25.*(M[intr...].+dt.*R[intr...]);project&&proj!(M,n,g,Ma,s3max)
    st!(M);@views M[intr...].=(1/3).*M0[intr...].+(2/3).*(M[intr...].+dt.*R[intr...]);project&&proj!(M,n,g,Ma,s3max)
    refill!(M,n,g)
end
function run_noθ()
    println("=== 3. VARIANT (theta*-IDP OFF: pure WENO5 faces, theta=1) ===")
    r=Dict{String,Tuple{Float64,Float64}}()
    for (nm,flag) in (("OFF",false),("ON ",true))
        M=haloed(n,g,ic); ok=true
        for _ in 1:nsteps; step_noθ!(M,n,g,dx,Ma,dt,s3max,flag;project=false); any(!isfinite,M)&&(ok=false;break); end
        if ok; r[nm]=metrics(M,n,g,u0,p0); @printf("   %s  u=%.3e  p=%.3e\n",nm,r[nm]...)
        else @printf("   %s BLEW UP\n",nm); end
    end
    if haskey(r,"OFF")&&haskey(r,"ON ")
        @printf("   -> gain (theta off + proj off): u %.2fx   p %.2fx\n",r["OFF"][1]/r["ON "][1],r["OFF"][2]/r["ON "][2])
        println("   INTERPRET: if ON p-error now COLLAPSES => theta*-IDP was the cap (part of G).")
        println("              if still ~1e-1 => neither projection nor theta* caps it.\n")
    end
end
run_noθ()

# ---- 4. PURE RECONSTRUCTION probe (no marching, no guards): the quantity the J
# theorem is about. Reconstruct the x-faces of the INITIAL contact directly and
# measure the face pressure error, raw vs logJ. This is the 1D-analogue number:
# if logJ face-p is ~machine-exact here but the marched p is 1e-1, the loss is
# entirely downstream (evolution/guard), NOT the reconstruction. ----
using Riemann35: realizable_3D_M4, realize_and_speed
using Riemann35.LogJacobiReconDev: logjacobi_marginal_faces, marg_m_to_J, marg_J_to_m
using Riemann35.MomentIndices: MARG_IDX
using Riemann35.HiOrder3ReconDev: recon_point_dev, recon_avg_dev, weno_faces_dev
function pure_recon_probe()
    println("=== 4. PURE face reconstruction of the IC (no march, no guards) ===")
    M=haloed(n,g,ic)
    # take the central y,z line; reconstruct x-faces via the residual's own pipeline
    j=n÷2; k=n÷2; jh=j+g
    Mext=M[:,jh,k,:]                       # (n+2g,35) x-line through the contact
    # replicate residual_line3 Step 1-3 for both raw and logJ, read face pressure
    n2g=size(Mext,1)
    _cell(kk)=ntuple(q->Mext[kk,q],Val(35))
    Ppt=[ (kk>=3 && kk<=n2g-2) ? recon_point_dev(_cell(kk-2),_cell(kk-1),_cell(kk),_cell(kk+1),_cell(kk+2)) :
          Riemann35.ReconDev.to_recon_vars_tup(_cell(kk)) for kk in 1:n2g]
    _pp(kk)=Ppt[clamp(kk,1,n2g)]
    Vavg=[recon_avg_dev(_pp(kk-2),_pp(kk-1),_pp(kk),_pp(kk+1),_pp(kk+2)) for kk in 1:n2g]
    _vv(kk)=Vavg[clamp(kk,1,n2g)]
    # logJ marginal faces
    midx=MARG_IDX[1]
    Mmarg=[(Mext[kk,midx[1]],Mext[kk,midx[2]],Mext[kk,midx[3]],Mext[kk,midx[4]],Mext[kk,midx[5]]) for kk in 1:n2g]
    oklj,ljL,ljR=logjacobi_marginal_faces(Mmarg,g)
    pval(m)=m[3]-m[2]^2/m[1]
    praw=0.0;plj=0.0
    for f in 1:n+1
        il=g+f-1; cL=_cell(il); cR=_cell(il+1)
        mL,mR=weno_faces_dev(_vv(il-2),_vv(il-1),_vv(il),_vv(il+1),_vv(il+2),_vv(il+3),cL,cR)
        praw=max(praw,abs(pval(mL)-p0),abs(pval(mR)-p0))
        if oklj
            jL=ljL[f];jR=ljR[f]
            mLj=ntuple(q->q==midx[1] ? jL[1] : q==midx[2] ? jL[2] : q==midx[3] ? jL[3] : q==midx[4] ? jL[4] : q==midx[5] ? jL[5] : mL[q],Val(35))
            mRj=ntuple(q->q==midx[1] ? jR[1] : q==midx[2] ? jR[2] : q==midx[3] ? jR[3] : q==midx[4] ? jR[4] : q==midx[5] ? jR[5] : mR[q],Val(35))
            plj=max(plj,abs(pval(mLj)-p0),abs(pval(mRj)-p0))
        end
    end
    @printf("   RAW  face-pressure error at contact = %.3e\n", praw)
    @printf("   logJ face-pressure error at contact = %.3e   (oklj=%s)\n", plj, oklj)
    @printf("   -> pure-reconstruction pressure gain = %.1fx\n", praw/max(plj,1e-300))
    println("   INTERPRET: this is the reconstruction operator ALONE (the 1D-analogue).")
    println("   If logJ~machine-zero here but marched-p~1e-1 => loss is EVOLUTION not recon.")
    println("   If logJ~1e-1 here too => the 3D face reconstruction itself isn't contact-exact.\n")
end
pure_recon_probe()

# ---- 5. CROSS-MOMENT reconstruction error at the contact (mechanism E direct) ----
# For the same x-line, measure the RAW-WENO reconstruction error of each nonzero
# cross moment vs its exact (cell-constant-per-side) value at the contact faces,
# and whether that cross error feeds the pressure. Pressure p=M200-M100^2/M000 uses
# NO cross moment — so cross smearing cannot touch the SCALAR x-pressure directly;
# this quantifies whether cross moments are smeared at all (they'd matter for the
# full pressure TENSOR / transverse fluxes, not the x-pressure metric).
function cross_recon_probe()
    println("=== 5. cross-moment reconstruction error at the contact (mechanism E) ===")
    M=haloed(n,g,ic); j=n÷2;k=n÷2;jh=j+g
    Mext=M[:,jh,k,:]; n2g=size(Mext,1)
    _cell(kk)=ntuple(q->Mext[kk,q],Val(35))
    Ppt=[ (kk>=3 && kk<=n2g-2) ? recon_point_dev(_cell(kk-2),_cell(kk-1),_cell(kk),_cell(kk+1),_cell(kk+2)) :
          Riemann35.ReconDev.to_recon_vars_tup(_cell(kk)) for kk in 1:n2g]
    _pp(kk)=Ppt[clamp(kk,1,n2g)]
    Vavg=[recon_avg_dev(_pp(kk-2),_pp(kk-1),_pp(kk),_pp(kk+1),_pp(kk+2)) for kk in 1:n2g]
    _vv(kk)=Vavg[clamp(kk,1,n2g)]
    # exact per-side cross values: left side rho=1, right rho=1000 => cross moments differ per side.
    # Use the cell values straddling the contact (i=n/2, n/2+1) as the exact endpoints; a
    # contact-exact scheme reproduces one side's value at the face (no overshoot between them).
    worstover=zeros(35)
    for f in 1:n+1
        il=g+f-1; cL=_cell(il); cR=_cell(il+1)
        mL,mR=weno_faces_dev(_vv(il-2),_vv(il-1),_vv(il),_vv(il+1),_vv(il+2),_vv(il+3),cL,cR)
        for q in 1:35
            (i2,j2,k2)=IJK[q]; iscross=((i2>0)&&(j2>0))||((i2>0)&&(k2>0))||((j2>0)&&(k2>0)); iscross||continue
            lo=min(cL[q],cR[q]); hi=max(cL[q],cR[q])
            # overshoot beyond the [min,max] of the two adjacent cells = Gibbs smear
            ovL=max(mL[q]-hi,lo-mL[q],0.0); ovR=max(mR[q]-hi,lo-mR[q],0.0)
            worstover[q]=max(worstover[q],ovL,ovR)
        end
    end
    tot=maximum(worstover)
    println("   worst RAW-WENO cross-moment OVERSHOOT (Gibbs) at the contact faces:")
    for q in 1:35
        (i2,j2,k2)=IJK[q]; worstover[q]>1e-12 && @printf("     M%d%d%d : %.3e\n",i2,j2,k2,worstover[q])
    end
    @printf("   max cross overshoot = %.3e\n", tot)
    println("   NOTE: the scalar x-pressure metric p=M200-M100^2/M000 uses NO cross moment,")
    println("   so this cross smear does NOT enter the section-1 p-number; it would matter for")
    println("   the pressure TENSOR / transverse-flux fidelity (mechanism E's real domain).\n")
end
cross_recon_probe()

println("###### done ######")

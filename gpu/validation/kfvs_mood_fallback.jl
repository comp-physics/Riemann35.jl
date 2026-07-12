# kfvs_mood_fallback.jl — the decisive Phase-I experiment: sparse MOOD-style shared-face
# HLL fallback for F3 (standalone experimental residual; does NOT touch the shipped path).
#
# Per RK stage, faithfully reproducing the production order-3 residual:
#   Pass1  FHO/FLO (KFVS) per face via residual_line3;  HLL parachute per face = _face_flux_tup(cellmeans).
#   Pass2  Mlo per cell, per-face θ* (_theta_star_fullcone); face-shared θ (min); F3 blended face flux.
#   MOOD   mask=∅; U = M − λΣ(mask?HLL:F3); failed={margin(U)<0}; mark faces of failed; expand until fixed.
#   Conservative by construction: each face flux single-valued (F3 or HLL, same for both adjacent cells).
# Self-validation: with mask forced ∅, the update must equal the production F3 update (rel<1e-12).
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()
const R35 = Riemann35
rl3   = R35.residual_line3
θfc   = R35._theta_star_fullcone
hll   = R35._face_flux_tup            # HLL of two 35-tuples

Ma=100.0; vj=Ma/sqrt(3); N=32; halo=4; g=halo; dx=1.0/N; dy=dx; dz=dx; s3max=max(40.0,4.0+Ma/2); dt=0.12*dx/(Ma/2+5); λ=dt/dx
bg=InitializeM4_35(0.05,0.,0.,0.,1.,0.,0.,1.,0.,1.)
Mt=InitializeM4_35(1.0,-vj,-vj,-vj,1.,0.,0.,1.,0.,1.); Mb=InitializeM4_35(1.0,vj,vj,vj,1.,0.,0.,1.,0.,1.)
function build()
    M=zeros(Float64,N+2halo,N+2halo,N,35); Cs=floor(Int,0.1N); lo=div(N,2)-Cs;hi=div(N,2);lo2=div(N,2)+1;hi2=div(N,2)+1+Cs
    for k in 1:N,j in 1:N,i in 1:N
        v=bg;(lo<=i<=hi&&lo<=j<=hi&&lo<=k<=hi)&&(v=Mb);(lo2<=i<=hi2&&lo2<=j<=hi2&&lo2<=k<=hi2)&&(v=Mt)
        @views M[i+halo,j+halo,k,:].=v
    end; M
end
tup(v)=ntuple(q->v[q],Val(35))
crow(M,i,j,k)=ntuple(q->M[i,j,k,q],Val(35))

# Compute per-face F3 blended flux + HLL parachute + the θ machinery, returning arrays.
function fluxes(M)
    nx=ny=N; nz=N
    FHOx=Array{NTuple{35,Float64}}(undef,nx+1,ny,nz); FLOx=similar(FHOx); HLLx=similar(FHOx)
    FHOy=Array{NTuple{35,Float64}}(undef,nx,ny+1,nz); FLOy=similar(FHOy); HLLy=similar(FHOy)
    FHOz=Array{NTuple{35,Float64}}(undef,nx,ny,nz+1); FLOz=similar(FHOz); HLLz=similar(FHOz)
    for k in 1:nz,j in 1:ny
        jh=j+halo; Fho,Flo=rl3(@view(M[:,jh,k,:]),dx,1,Ma;g=g,s3max=s3max,use_kfvs_anchor=true)
        for f in 1:nx+1; FHOx[f,j,k]=Fho[f]; FLOx[f,j,k]=Flo[f]; HLLx[f,j,k]=hll(crow(M,f-1+halo,jh,k),crow(M,f+halo,jh,k),1,Ma,s3max); end
    end
    for k in 1:nz,i in 1:nx
        ih=i+halo; Fho,Flo=rl3(@view(M[ih,:,k,:]),dy,2,Ma;g=g,s3max=s3max,use_kfvs_anchor=true)
        for f in 1:ny+1; FHOy[i,f,k]=Fho[f]; FLOy[i,f,k]=Flo[f]; HLLy[i,f,k]=hll(crow(M,ih,f-1+halo,k),crow(M,ih,f+halo,k),2,Ma,s3max); end
    end
    for i in 1:nx,j in 1:ny
        ih=i+halo;jh=j+halo; col=M[ih,jh,:,:]; Mext=vcat(repeat(col[1:1,:],g,1),col,repeat(col[nz:nz,:],g,1))
        Fho,Flo=rl3(Mext,dz,3,Ma;g=g,s3max=s3max,use_kfvs_anchor=true)
        for f in 1:nz+1
            FHOz[i,j,f]=Fho[f]; FLOz[i,j,f]=Flo[f]
            zb= f>1 ? crow(M,ih,jh,f-1) : crow(M,ih,jh,1); zf= f<=nz ? crow(M,ih,jh,f) : crow(M,ih,jh,nz)
            HLLz[i,j,f]=hll(zb,zf,3,Ma,s3max)
        end
    end
    # Pass2: per-cell per-face θ (six), then face-shared min
    Θxr=ones(nx,ny,nz);Θxl=ones(nx,ny,nz);Θyr=ones(nx,ny,nz);Θyl=ones(nx,ny,nz);Θzr=ones(nx,ny,nz);Θzl=ones(nx,ny,nz)
    for k in 1:nz,j in 1:ny,i in 1:nx
        Mc=crow(M,i+halo,j+halo,k)
        Mlo=ntuple(q->Mc[q]-λ*(FLOx[i+1,j,k][q]-FLOx[i,j,k][q])-λ*(FLOy[i,j+1,k][q]-FLOy[i,j,k][q])-λ*(FLOz[i,j,k+1][q]-FLOz[i,j,k][q]),Val(35))
        G(FH,FL)=ntuple(q->FH[q]-FL[q],Val(35))
        Θxr[i,j,k]=θfc(Mlo,ntuple(q->-6λ*G(FHOx[i+1,j,k],FLOx[i+1,j,k])[q],Val(35)),Ma,s3max)
        Θxl[i,j,k]=θfc(Mlo,ntuple(q-> 6λ*G(FHOx[i,j,k],FLOx[i,j,k])[q],Val(35)),Ma,s3max)
        Θyr[i,j,k]=θfc(Mlo,ntuple(q->-6λ*G(FHOy[i,j+1,k],FLOy[i,j+1,k])[q],Val(35)),Ma,s3max)
        Θyl[i,j,k]=θfc(Mlo,ntuple(q-> 6λ*G(FHOy[i,j,k],FLOy[i,j,k])[q],Val(35)),Ma,s3max)
        Θzr[i,j,k]=θfc(Mlo,ntuple(q->-6λ*G(FHOz[i,j,k+1],FLOz[i,j,k+1])[q],Val(35)),Ma,s3max)
        Θzl[i,j,k]=θfc(Mlo,ntuple(q-> 6λ*G(FHOz[i,j,k],FLOz[i,j,k])[q],Val(35)),Ma,s3max)
    end
    blend(θ,FH,FL)=ntuple(q->FL[q]+θ*(FH[q]-FL[q]),Val(35))
    # face-shared F3 flux arrays (single-valued per interior face)
    F3x=Array{NTuple{35,Float64}}(undef,nx+1,ny,nz);F3y=Array{NTuple{35,Float64}}(undef,nx,ny+1,nz);F3z=Array{NTuple{35,Float64}}(undef,nx,ny,nz+1)
    for k in 1:nz,j in 1:ny,i in 1:nx+1
        θ = i==1 ? Θxl[1,j,k] : i==nx+1 ? Θxr[nx,j,k] : min(Θxr[i-1,j,k],Θxl[i,j,k]); F3x[i,j,k]=blend(θ,FHOx[i,j,k],FLOx[i,j,k])
    end
    for k in 1:nz,j in 1:ny+1,i in 1:nx
        θ = j==1 ? Θyl[i,1,k] : j==ny+1 ? Θyr[i,ny,k] : min(Θyr[i,j-1,k],Θyl[i,j,k]); F3y[i,j,k]=blend(θ,FHOy[i,j,k],FLOy[i,j,k])
    end
    for k in 1:nz+1,j in 1:ny,i in 1:nx
        θ = k==1 ? Θzl[i,j,1] : k==nz+1 ? Θzr[i,j,nz] : min(Θzr[i,j,k-1],Θzl[i,j,k]); F3z[i,j,k]=blend(θ,FHOz[i,j,k],FLOz[i,j,k])
    end
    (F3x,F3y,F3z,HLLx,HLLy,HLLz)
end

# one MOOD stage: returns (Uint interior (nx,ny,nz,35), n_failed0, n_faces_masked, n_iter, n_final_bad)
function mood_stage(M; force_no_mask=false)
    nx=ny=N;nz=N
    F3x,F3y,F3z,HLLx,HLLy,HLLz=fluxes(M)
    mx=falses(nx+1,ny,nz);my=falses(nx,ny+1,nz);mz=falses(nx,ny,nz+1)
    fx(i,j,k)= mx[i,j,k] ? HLLx[i,j,k] : F3x[i,j,k]
    fy(i,j,k)= my[i,j,k] ? HLLy[i,j,k] : F3y[i,j,k]
    fz(i,j,k)= mz[i,j,k] ? HLLz[i,j,k] : F3z[i,j,k]
    U=Array{Float64}(undef,nx,ny,nz,35)
    function upd()
        for k in 1:nz,j in 1:ny,i in 1:nx
            Mc=crow(M,i+halo,j+halo,k)
            for q in 1:35; U[i,j,k,q]=Mc[q]-λ*((fx(i+1,j,k)[q]-fx(i,j,k)[q])+(fy(i,j+1,k)[q]-fy(i,j,k)[q])+(fz(i,j,k+1)[q]-fz(i,j,k)[q])); end
        end
    end
    nbadcount()= count(<(0),[realizability_margin(@view U[i,j,k,:]) for i in 1:nx,j in 1:ny,k in 1:nz])
    upd(); nfail0=nbadcount()
    force_no_mask && return (U,nfail0,0,0,nfail0)
    niter=0
    while true
        niter+=1; newmark=false
        for k in 1:nz,j in 1:ny,i in 1:nx
            realizability_margin(@view U[i,j,k,:])<0 || continue
            mx[i,j,k]   || (mx[i,j,k]=true;   newmark=true)
            mx[i+1,j,k] || (mx[i+1,j,k]=true; newmark=true)
            my[i,j,k]   || (my[i,j,k]=true;   newmark=true)
            my[i,j+1,k] || (my[i,j+1,k]=true; newmark=true)
            mz[i,j,k]   || (mz[i,j,k]=true;   newmark=true)
            mz[i,j,k+1] || (mz[i,j,k+1]=true; newmark=true)
        end
        (!newmark || niter>30) && break
        upd()
    end
    (U,nfail0,count(mx)+count(my)+count(mz),niter,nbadcount())
end

# ---- self-validation: standalone F3 update (no mask) vs production residual ----
M=build(); comm=MPI.COMM_WORLD; decomp=setup_mpi_cartesian_3d(N,N,N,halo,comm); bc=:copy
halo_exchange_3d!(M,decomp,bc)
Rprod=similar(M)
R35.residual_ho_3d_order3!(Rprod,M,N,N,N,halo,dx,dx,dx,Ma,dt;s3max=s3max,use_kfvs_anchor=true)
Uf3=M[halo+1:halo+N,halo+1:halo+N,1:N,:].+dt.*Rprod[halo+1:halo+N,halo+1:halo+N,1:N,:]
Umood,_,_,_,_=mood_stage(M;force_no_mask=true)
vmax=maximum(abs.(Umood.-Uf3))/max(maximum(abs.(Uf3)),1e-300)
@printf("self-validation (standalone F3 update vs production): max rel diff = %.2e  (%s)\n", vmax, vmax<1e-10 ? "VALIDATED" : "MISMATCH"); flush(stdout)
vmax<1e-10 || error("standalone residual does not match production — fix before trusting MOOD")

# ---- full SSP-RK3 MOOD march on the hard crossing (failures appear at stage 2+) ----
D4=(5,9,12,14,15,19,28,30,31,22,33,35,24,34,25)
function invs(M)
    mass=0.;px=0.;py=0.;pz=0.;en=0.;m4=0.
    for k in 1:N,j in 1:N,i in 1:N; c=@view M[i+halo,j+halo,k,:]
        mass+=c[1];px+=c[2];py+=c[6];pz+=c[16];en+=c[3]+c[10]+c[20];for q in D4;m4+=c[q];end; end
    (mass,px,py,pz,en,m4)
end
putback!(M,U)= for k in 1:N,j in 1:N,i in 1:N,q in 1:35; M[i+halo,j+halo,k,q]=U[i,j,k,q]; end

function march(nsteps)
    M=build(); halo_exchange_3d!(M,decomp,bc); i0=invs(M)
    maxfail=0; totmask=0; maxiter=0; maxbad=0; survived=true
    for s in 1:nsteps
        M0=copy(M)
        stg(lbl,combine)=begin
            t=time(); U,nf,nm,ni,nb=mood_stage(M); dtw=time()-t
            maxfail=max(maxfail,nf);totmask+=nm;maxiter=max(maxiter,ni);maxbad=max(maxbad,nb)
            @printf("  step %d %s: failed=%d masked-faces=%d iters=%d remaining-bad=%d  (%.1fs)\n",s,lbl,nf,nm,ni,nb,dtw); flush(stdout)
            combine(U)
        end
        stg("s1",U->(putback!(M,U)))
        halo_exchange_3d!(M,decomp,bc)
        stg("s2",U->for k in 1:N,j in 1:N,i in 1:N,q in 1:35; M[i+halo,j+halo,k,q]=0.75*M0[i+halo,j+halo,k,q]+0.25*U[i,j,k,q]; end)
        halo_exchange_3d!(M,decomp,bc)
        stg("s3",U->for k in 1:N,j in 1:N,i in 1:N,q in 1:35; M[i+halo,j+halo,k,q]=(1/3)*M0[i+halo,j+halo,k,q]+(2/3)*U[i,j,k,q]; end)
        halo_exchange_3d!(M,decomp,bc)
        nfz=0;for k in 1:N,j in 1:N,i in 1:N,q in 1:35; isfinite(M[i+halo,j+halo,k,q])||(nfz+=1);end
        nfz>0 && (@printf("  nonfinite at step %d\n",s);survived=false;break)
    end
    i1=invs(M); rel(a,b)=abs(a-b)/max(abs(b),1e-300)
    @printf("MOOD march %d steps: survived=%s  max-failed/step=%d (%.2f%%)  max-faces-masked=%d  max-iter=%d  max-remaining-bad=%d\n",
            nsteps, survived, maxfail, 100maxfail/N^3, totmask, maxiter, maxbad)
    survived && @printf("  conservation: mass %.2e  |mom| %.2e  energy %.2e  4th-moment %.2e\n",
            rel(i1[1],i0[1]), abs(i1[2]-i0[2])+abs(i1[3]-i0[3])+abs(i1[4]-i0[4]), rel(i1[5],i0[5]), rel(i1[6],i0[6]))
end
@printf("\n=== MOOD HLL face-fallback march: hard Ma=100 crossing, 32³ ===\n")
flush(stdout); march(3)

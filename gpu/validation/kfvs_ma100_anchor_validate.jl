# kfvs_ma100_anchor_validate.jl — F2 validation: the anchor+marginal-guard at
# Ma=100 on a WELL-POSED crossing-jets case (OFF/projection path is itself stable),
# so it is a FAIR comparison (not the pathological raw-stress block).
#
# Reports, OFF (projection) vs ON (anchor→marginal-regularized full-cone θ*-blend):
#   (a) finite? rho>0? 0 unrealizable throughout
#   (b) conservation (mass / energy drift) vs the projection path
#   (c) projection retired / marginal-clamp engagement (would-fire) rate
#   (d) mean θ* at Ma=100 (high-order retention)

ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"; ENV["HYQMOM_KFVS_THETA_STATS"]="1"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()

# well-posed crossing-jets: proper Maxwellian cells (InitializeM4_35), jet speeds
# a modest fraction of Ma so the state stays comfortably realizable, warm background.
function build_ic(Np, halo, Ma)
    C = 1.0               # unit temperature (variance)
    vj = 0.15 * Ma        # jet speed (well inside the realizable set at s3max)
    bg = InitializeM4_35(0.05, 0.0, 0.0, 0.0, C, 0.0, 0.0, C, 0.0, C)   # warm background
    l  = InitializeM4_35(1.0,  vj,  vj/2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    r  = InitializeM4_35(1.0, -vj, -vj/2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    nx=ny=nz=Np; M=zeros(Float64, nx+2halo, ny+2halo, nz, 35)
    Cs=max(1,floor(Int,0.2*Np)); lo=div(Np,2)-Cs;hi=div(Np,2);lo2=div(Np,2)+1;hi2=div(Np,2)+1+Cs
    for k in 1:nz,i in 1:nx,j in 1:ny
        Mr=bg; (lo<=i<=hi&&lo<=j<=hi)&&(Mr=l); (lo2<=i<=hi2&&lo2<=j<=hi2)&&(Mr=r)
        M[i+halo,j+halo,k,:]=Mr
    end
    M,nx,ny,nz
end

function run(USE, Np, halo, Ma, nsteps, dt)
    M,nx,ny,nz = build_ic(Np,halo,Ma)
    comm=MPI.COMM_WORLD; decomp=setup_mpi_cartesian_3d(Np,Np,Np,halo,comm); bc=:copy
    halo_exchange_3d!(M,decomp,bc)
    dx=1.0/Np; s3max=max(40.0, 4.0+abs(Ma)/2.0)   # = 54 at Ma=100 (production value)
    mass0=0.0; en0=0.0
    for k in 1:nz,j in 1:ny,i in 1:nx; mass0+=M[i+halo,j+halo,k,1]; en0+=M[i+halo,j+halo,k,3]+M[i+halo,j+halo,k,10]+M[i+halo,j+halo,k,20]; end
    if USE; reset_kfvs_theta_stats!() else; ENV["HYQMOM_PROJ_COUNT"]="1"; reset_proj_counter!() end
    worst_nonfinite=0
    for s in 1:nsteps
        step_highorder_3d!(M,dt,decomp,bc,nx,ny,nz,halo,dx,dx,dx,Ma; order=3,s3max=s3max,use_kfvs_anchor=USE)
        nf=0; for k in 1:nz,j in 1:ny,i in 1:nx, q in 1:35; isfinite(M[i+halo,j+halo,k,q])||(nf+=1); end
        worst_nonfinite=max(worst_nonfinite,nf)
        nf>0 && break
    end
    nfin=0;nrho=0;nun=0;mass1=0.0;en1=0.0;worst_margin=0.0
    for k in 1:nz,j in 1:ny,i in 1:nx
        c=@view M[i+halo,j+halo,k,:]
        for q in 1:35; isfinite(c[q])||(nfin+=1); end
        (c[1]>0.0)||(nrho+=1); mass1+=c[1]; en1+=c[3]+c[10]+c[20]
        mgn = realizability_margin(c)
        mgn < -1e-8 && (nun+=1; worst_margin = min(worst_margin, mgn))
    end
    tag = USE ? "ON " : "OFF"
    @printf("%s(Ma=%g,%dstep,s3max=%g) nonfinite=%d rho<=0=%d margin<-1e-8=%d(worst=%.2e) massdrift=%.3e endrift=%.3e",
            tag,Ma,nsteps,s3max,nfin,nrho,nun,worst_margin, abs(mass1-mass0)/mass0, abs(en1-en0)/max(abs(en0),1e-300))
    if USE
        st=kfvs_theta_stats()   # F3 FLUX-level theta (the active anchor path)
        @printf(" mean_flux_theta=%.4f faces=%d theta_lt1=%d\n", st.mean_theta, st.faces, st.theta_lt1)
    else
        @printf(" proj-fired=%d\n", proj_correction_count())
    end
end

function main()
    Ma=100.0; Np=16; halo=4; dx=1.0/Np
    dt = 0.12*dx/(Ma/2 + 5)   # CFL-safe for Ma=100
    for nsteps in (8, 20, 40)
        run(false, Np, halo, Ma, nsteps, dt)
        run(true,  Np, halo, Ma, nsteps, dt)
    end
end
main()

# kfvs_sparse_projection_debt.jl — Phase I step "sparse intervention": does projecting
# ONLY the thin flagged set (the anchor's ~0.55% exit cells) conserve as well as the
# default (projection35 on EVERY cell), while keeping F3 exact on the 99.4% pass set?
#
# Runs the HARD Ma=100 crossing (in-memory IC, jets ±Ma/√3), order-3, 20 steps, in 3 configs:
#   DEFAULT  = projection35 on all cells      (use_kfvs_anchor=false)      — reference
#   F3+SPARSE= project ONLY flagged (margin<0)(use_kfvs_anchor, sparse_project)
#   F3-RAW   = unguarded F3                    (use_kfvs_anchor, nothing)   — expected crash
# Reports conservation debt vs t0: mass, |net momentum|, energy (2nd trace), and the
# degree-4 moment total (the non-invariant "high-moment debt"), + projection counts.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()

Ma=100.0; vj=Ma/sqrt(3); N=32; h=4; dx=1.0/N; s3max=max(40.0,4.0+Ma/2); dt=0.12*dx/(Ma/2+5)
bg=InitializeM4_35(0.05,0.,0.,0., 1.,0.,0.,1.,0.,1.)
Mt=InitializeM4_35(1.0,-vj,-vj,-vj, 1.,0.,0.,1.,0.,1.); Mb=InitializeM4_35(1.0,vj,vj,vj, 1.,0.,0.,1.,0.,1.)
function build()
    M=zeros(Float64,N+2h,N+2h,N,35); Cs=floor(Int,0.1N); lo=div(N,2)-Cs;hi=div(N,2);lo2=div(N,2)+1;hi2=div(N,2)+1+Cs
    for k in 1:N,j in 1:N,i in 1:N
        v=bg;(lo<=i<=hi&&lo<=j<=hi&&lo<=k<=hi)&&(v=Mb);(lo2<=i<=hi2&&lo2<=j<=hi2&&lo2<=k<=hi2)&&(v=Mt)
        @views M[i+h,j+h,k,:].=v
    end; M
end
const D4=(5,9,12,14,15,19,28,30,31,22,33,35,24,34,25)   # degree-4 moment indices
function invariants(M)
    mass=0.0;px=0.0;py=0.0;pz=0.0;en=0.0;m4=0.0
    for k in 1:N,j in 1:N,i in 1:N
        c=@view M[i+h,j+h,k,:]
        mass+=c[1];px+=c[2];py+=c[6];pz+=c[16];en+=c[3]+c[10]+c[20]
        for q in D4; m4+=c[q]; end
    end
    (mass,px,py,pz,en,m4)
end

function run(cfg)
    M=build(); comm=MPI.COMM_WORLD; decomp=setup_mpi_cartesian_3d(N,N,N,h,comm); bc=:copy
    halo_exchange_3d!(M,decomp,bc)
    use_anchor = cfg!=:default; sparse = cfg==:sparse; reproj=false
    Riemann35._KFVS_SPARSE_NPROJ[]=0
    m0,px0,py0,pz0,en0,m40=invariants(M)
    ok=true
    for s in 1:20
        try
            step_highorder_3d!(M,dt,decomp,bc,N,N,N,h,dx,dx,dx,Ma; order=3,s3max=s3max,
                               use_kfvs_anchor=use_anchor,sparse_project=sparse,anchor_reproject=reproj)
        catch e; @printf("  %-9s FAILED at step %d: %s\n",String(cfg),s,typeof(e)); ok=false; break; end
        nf=0; for k in 1:N,j in 1:N,i in 1:N, q in 1:35; isfinite(M[i+h,j+h,k,q])||(nf+=1); end
        nf>0 && (@printf("  %-9s nonfinite at step %d\n",String(cfg),s); ok=false; break)
    end
    ok || return
    m1,px1,py1,pz1,en1,m41=invariants(M)
    rel(a,b)=abs(a-b)/max(abs(b),1e-300)
    @printf("  %-9s mass %.2e  |mom| %.2e  energy %.2e  4th-moment %.2e   proj-cells=%d\n",
        String(cfg), rel(m1,m0), abs(px1-px0)+abs(py1-py0)+abs(pz1-pz0), rel(en1,en0), rel(m41,m40),
        Riemann35._KFVS_SPARSE_NPROJ[])
end

@printf("=== sparse-projection conservation debt: hard Ma=100 crossing, order-3, 20 steps ===\n")
@printf("(conservation debt vs t0; DEFAULT projects ALL cells, SPARSE projects only margin<0)\n")
run(:default)
run(:sparse)
run(:raw)
println("Done. Question: does SPARSE match DEFAULT on mass/mom/energy while projecting far fewer cells?")

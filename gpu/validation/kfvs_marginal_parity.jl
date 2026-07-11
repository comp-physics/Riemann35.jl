# kfvs_marginal_parity.jl — math-parity of the device twin marginal_regularized_dev
# against the CPU _marginal_regularized (M2CS4_35 path) on real reachable states.
# The two must agree on the true/false verdict for ~all cells (they are math twins;
# the device version uses inline scalar central-moment formulas instead of the array
# M2CS4_35). Reports the disagreement count and the worst near-threshold cases.
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()
const RM = Riemann35

function build_ic(Np,halo,Ma)
    C=1.0; vj=0.15*Ma
    bg=InitializeM4_35(0.05,0.0,0.0,0.0,C,0.0,0.0,C,0.0,C)
    l =InitializeM4_35(1.0, vj, vj/2,0.0,C,0.0,0.0,C,0.0,C)
    r =InitializeM4_35(1.0,-vj,-vj/2,0.0,C,0.0,0.0,C,0.0,C)
    nx=ny=nz=Np; M=zeros(Float64,nx+2halo,ny+2halo,nz,35)
    Cs=max(1,floor(Int,0.2*Np)); lo=div(Np,2)-Cs;hi=div(Np,2);lo2=div(Np,2)+1;hi2=div(Np,2)+1+Cs
    for k in 1:nz,i in 1:nx,j in 1:ny
        Mr=bg; (lo<=i<=hi&&lo<=j<=hi)&&(Mr=l); (lo2<=i<=hi2&&lo2<=j<=hi2)&&(Mr=r)
        M[i+halo,j+halo,k,:]=Mr
    end
    M,nx,ny,nz
end

function main()
    Ma=100.0; Np=16; halo=4; dx=1.0/Np; dt=0.12*dx/(Ma/2+5); s3max=max(40.0,4.0+abs(Ma)/2.0)
    M,nx,ny,nz=build_ic(Np,halo,Ma)
    comm=MPI.COMM_WORLD; decomp=setup_mpi_cartesian_3d(Np,Np,Np,halo,comm); bc=:copy
    halo_exchange_3d!(M,decomp,bc)
    for _ in 1:20
        step_highorder_3d!(M,dt,decomp,bc,nx,ny,nz,halo,dx,dx,dx,Ma; order=3,s3max=s3max,use_kfvs_anchor=true)
    end
    n=0; disagree=0; both_true=0
    for k in 1:nz,j in 1:ny,i in 1:nx
        c=Vector{Float64}(M[i+halo,j+halo,k,:]); ct=NTuple{35,Float64}(c)
        cpu = RM._marginal_regularized(c, Ma, s3max)
        dev = RM.marginal_regularized_dev(ct, Ma, s3max)
        n+=1; (cpu && dev) && (both_true+=1)
        if cpu != dev
            disagree+=1
            if disagree<=8; @printf("  DISAGREE cell(%d,%d,%d): cpu=%s dev=%s\n",i,j,k,cpu,dev); end
        end
    end
    @printf("\ninterior cells=%d  both_true=%d  DISAGREE=%d  (%.4f%%)\n", n, both_true, disagree, 100*disagree/n)
    println(disagree==0 ? "PARITY OK (identical verdicts)." : "check disagreements above (near-threshold expected, gross = bug).")
end
main()

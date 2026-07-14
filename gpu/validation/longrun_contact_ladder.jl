# longrun_contact_ladder.jl — long moving-contact march, error-vs-exact over time,
# for the 3-rung ladder: first-order HLL (order=1), WENO5+HLL (order=3 logJ off),
# log-J+HLL (order=3 logJ on). Measures STABILITY (survival) + ACCURACY (max |u-u0|,
# |p-p0| vs the exact uniform-pressure moving contact) at checkpoints.
using Riemann35, Printf, MPI
MPI.Initialized() || MPI.Init()
const R35 = Riemann35
n=24; g=8; Ma=1.0; u0=0.5; p0=1.0; ratio=1000.0; s3max=40.0; dx=1.0/n
ic(i,j,k)=(rho = i<=n÷2 ? 1.0 : ratio; T=p0/rho; InitializeM4_35(rho,u0,0.0,0.0,T,0.0,0.0,T,0.0,T))
dt=0.2*dx/(u0+4.0*sqrt(p0)); nsteps=100; checkpts=(15,30,50,75,100)

function haloed(n,g)
    M=zeros(n+2g,n+2g,n,35)
    for k in 1:n,j in 1:n,i in 1:n; @views M[i+g,j+g,k,:].=ic(i,j,k); end
    M
end
function metrics(M,n,g)
    um=0.0; pm=0.0; ok=true
    for k in 1:n,j in 1:n,i in 1:n
        m=@view M[i+g,j+g,k,:]; r=m[1]
        (r>0 && all(isfinite,m)) || (ok=false; continue)
        um=max(um,abs(m[2]/r-u0)); pm=max(pm,abs((m[3]-m[2]^2/r)-p0))
    end
    (ok,um,pm)
end
mass(M,n,g)=sum(@view M[g+1:g+n,g+1:g+n,:,1])

@printf("Long moving-contact march: HLL(o1) vs WENO5+HLL(o3) vs logJ+HLL(o3), %d steps\n\n", nsteps)
for (name,order,lj) in (("HLL-o1",1,false),("WENO5+HLL",3,false),("logJ+HLL",3,true))
    M=haloed(n,g); decomp=R35.setup_mpi_cartesian_3d(n,n,n,g,MPI.COMM_WORLD)
    m0=mass(M,n,g); dead=0
    @printf("--- %s ---\n", name)
    for s in 1:nsteps
        try
            step_highorder_3d!(M,dt,decomp,:copy,n,n,n,g,dx,dx,dx,Ma; order=order,s3max=s3max,use_logjacobi_recon=lj)
        catch e
            @printf("   CRASHED @step %d: %s\n", s, sprint(showerror,e)[1:min(end,45)]); dead=s; break
        end
        ok,_,_=metrics(M,n,g); ok || (@printf("   NaN/rho<=0 @step %d\n",s); dead=s; break)
        if s in checkpts
            _,um,pm=metrics(M,n,g); dr=abs(mass(M,n,g)-m0)/m0
            @printf("   step %3d:  u_err=%.3e  p_err=%.3e  mass_drift=%.2e\n", s, um, pm, dr)
        end
    end
    dead==0 && @printf("   SURVIVED all %d steps\n", nsteps)
    println()
end

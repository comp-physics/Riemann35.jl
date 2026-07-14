# cpu_logj_ma100_march.jl — discriminator: does the CPU order-3 march survive Ma=100
# crossing jets with log-Jacobi ON? GPU crashed (NaN@32³, DomainError@64³). If CPU-on
# also NaNs => genuine log-J high-Ma robustness loss. If CPU-on survives => GPU-path bug.
using Riemann35, Printf, MPI
MPI.Initialized() || MPI.Init()
const R35 = Riemann35
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))

cmeta = split(strip(read(joinpath(DATA, "r3d_cross_ma100.meta"), String)), '\n')
Ma = parse(Float64, cmeta[1])
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:,1]; Mt = cross[:,2]; Mb = cross[:,3]
s3max = max(40.0, 4.0 + abs(Ma)/2.0)
n = 24; g = 4; dx = 1.0/n; nstep = 20

function build_U(n)
    U = zeros(35, n, n, n); Cs = floor(Int, 0.1n)
    Minb = div(n,2)-Cs; Maxb = div(n,2); Mnt = div(n,2)+1; Maxt = div(n,2)+1+Cs
    for k in 1:n, j in 1:n, i in 1:n
        v = bg
        if Minb<=i<=Maxb && Minb<=j<=Maxb && Minb<=k<=Maxb; v = Mb; end
        if Mnt <=i<=Maxt && Mnt <=j<=Maxt && Mnt <=k<=Maxt; v = Mt; end
        @views U[:,i,j,k] .= v
    end
    U
end
function cfl_dt(M, n, halo, dx)
    vmax = 0.0
    for k in 1:n, j in 1:n, i in 1:n
        r = M[i+halo,j+halo,k,1]; r<=0 && return NaN
        ui=M[i+halo,j+halo,k,2]/r; vi=M[i+halo,j+halo,k,6]/r; wi=M[i+halo,j+halo,k,16]/r
        cx=max(M[i+halo,j+halo,k,3]/r-ui*ui,0.0); cy=max(M[i+halo,j+halo,k,10]/r-vi*vi,0.0); cz=max(M[i+halo,j+halo,k,20]/r-wi*wi,0.0)
        sp=max(abs(ui),abs(vi),abs(wi))+4.0*2.334*sqrt(max(cx,cy,cz)+1e-12)
        vmax=max(vmax,sp)
    end
    (1.0/3.0)*dx/max(vmax,1e-12)
end

@printf("CPU order-3 march, Ma=%.0f crossing jets, n=%d, %d steps\n", Ma, n, nstep)
for (mode, lj) in (("base",false), ("logJ",true))
    U = build_U(n)
    halo = g; M = zeros(n+2halo, n+2halo, n, 35)
    for k in 1:n, j in 1:n, i in 1:n; @views M[i+halo,j+halo,k,:] .= U[:,i,j,k]; end
    decomp = R35.setup_mpi_cartesian_3d(n, n, n, halo, MPI.COMM_WORLD)
    failed = ""; last_ok = 0
    for s in 1:nstep
        dt = cfl_dt(M, n, halo, dx)
        (isnan(dt) || dt<=0) && (failed = "bad dt (rho<=0) at step $s"; break)
        try
            step_highorder_3d!(M, dt, decomp, :copy, n,n,n, halo, dx,dx,dx, Ma;
                               order=3, s3max=s3max, use_logjacobi_recon=lj)
        catch e
            failed = "exception step $s: " * sprint(showerror,e)[1:min(end,50)]; break
        end
        anynan = any(isnan, @view M[halo+1:halo+n, halo+1:halo+n, :, :])
        anynan && (failed = "NaN at step $s"; break)
        last_ok = s
    end
    if failed == ""
        rmin = minimum(M[halo+1:halo+n, halo+1:halo+n, :, 1])
        @printf("  %-5s SURVIVED %d steps  min(rho)=%.3e\n", mode, nstep, rmin)
    else
        @printf("  %-5s FAILED after %d ok steps: %s\n", mode, last_ok, failed)
    end
end

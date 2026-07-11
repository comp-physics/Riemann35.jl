# derisk_repro_cpu_hardcross.jl (SCRATCH) — reproduce the anchor-ON DomainError on
# the CPU order-3 path using the SAME hard crossing IC (r3d_cross_ma100 vectors,
# jets at ±Ma/√3) the GPU driver used. CPU throws a FULL native stacktrace ⇒ pinpoints
# the unguarded sqrt with no ptxas / no -g2. If it does NOT crash, the bug is
# GPU-store-variant-specific (informative either way).
ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()

DATA = joinpath(@__DIR__, "..", "..", "data")
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA,"r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:,1]; Mt = cross[:,2]; Mb = cross[:,3]
Ma = 100.0

# hard crossing IC in CPU layout M[nx+2h, ny+2h, nz, 35] (x,y haloed, z slab)
function build_hardcross(N, h)
    nx=ny=nz=N
    M = zeros(Float64, nx+2h, ny+2h, nz, 35)
    Cs = floor(Int, 0.1*N)
    lo = div(N,2)-Cs; hi = div(N,2); lo2 = div(N,2)+1; hi2 = div(N,2)+1+Cs
    for k in 1:nz, j in 1:ny, i in 1:nx
        v = bg
        (lo<=i<=hi && lo<=j<=hi && lo<=k<=hi)   && (v = Mb)
        (lo2<=i<=hi2 && lo2<=j<=hi2 && lo2<=k<=hi2) && (v = Mt)
        @views M[i+h, j+h, k, :] .= v
    end
    M, nx, ny, nz
end

function run(N, nsteps, mode)
    h = 4; dx = 1.0/N
    s3max = max(40.0, 4.0+abs(Ma)/2.0)
    dt = 0.12*dx/(Ma/2 + 5)          # CFL-safe (matches kfvs_ma100_anchor_validate)
    anchor  = mode != :default
    reproj  = mode == :f4b
    M,nx,ny,nz = build_hardcross(N, h)
    comm = MPI.COMM_WORLD; decomp = setup_mpi_cartesian_3d(N,N,N,h,comm); bc=:copy
    halo_exchange_3d!(M, decomp, bc)
    tag = mode == :default ? "DEFAULT (projection35, CONTROL)" :
          mode == :f4b     ? "F4b (anchor_reproject=true)" : "F3  (anchor_reproject=false)"
    @printf("=== CPU hard-crossing %s: N=%d Ma=%.0f dt=%.3e s3max=%g, up to %d steps ===\n",
            tag, N, Ma, dt, s3max, nsteps)
    survived = true
    for s in 1:nsteps
        step_highorder_3d!(M, dt, decomp, bc, nx,ny,nz, h, dx,dx,dx, Ma;
                           order=3, s3max=s3max, use_kfvs_anchor=anchor, anchor_reproject=reproj)
        nf=0; rmin=Inf
        for k in 1:nz, j in 1:ny, i in 1:nx
            r = M[i+h,j+h,k,1]; rmin=min(rmin,r)
            for q in 1:35; isfinite(M[i+h,j+h,k,q]) || (nf+=1); end
        end
        @printf("  step %2d: nonfinite=%d  rho_min=%.4e\n", s, nf, rmin)
        nf>0 && (println("  -> nonfinite appeared; stopping"); survived=false; break)
    end
    survived && @printf("  -> SURVIVED %d steps (rho>0, all finite)\n", nsteps)
end

run(32, 20, :default)   # CONTROL: does the shipped projection35 path survive this exact harness/dt?
println()
run(32, 20, :f3)        # F3 alone (what the GPU port runs)
println()
run(32, 20, :f4b)       # F4b conservative re-projection (CPU-only; the sanctioned completion)

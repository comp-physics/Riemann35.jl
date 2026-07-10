# kfvs_cpu_order3_baseline.jl — small CPU order-3 crossing-jets march, used as the
# BYTE-IDENTITY gate for increment E. Runs step_highorder_3d! (order=3) for a few
# steps on a 16^3 crossing-jets IC and prints an exact L2 checksum of the final
# interior field. Run it (a) on the pristine tree and (b) after plumbing with
# use_kfvs_anchor=false; the two checksums MUST be bit-identical.
#
# Optional env KFVS_ANCHOR=1 turns the flag ON (flag-on path); default OFF.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()

const USE_ANCHOR = get(ENV, "KFVS_ANCHOR", "0") != "0"

function build_ic(Np, halo)
    C200=1.0; C020=1.0; C002=1.0; C110=0.0; C101=0.0; C011=0.0
    Mr_bg   = InitializeM4_35(0.001, 0.0,0.0,0.0, C200,C110,C101,C020,C011,C002)
    # crossing jets: two cubes with opposite-ish velocity to excite cross moments
    Mr_l = InitializeM4_35(1.0,  1.0, 0.5, 0.0, C200,C110,C101,C020,C011,C002)
    Mr_r = InitializeM4_35(1.0, -1.0,-0.5, 0.0, C200,C110,C101,C020,C011,C002)
    nx=ny=nz=Np
    M = zeros(Float64, nx+2halo, ny+2halo, nz, 35)
    Csize = max(1, floor(Int, 0.2*Np))
    lo = div(Np,2)-Csize; hi = div(Np,2)
    lo2 = div(Np,2)+1; hi2 = div(Np,2)+1+Csize
    for k in 1:nz, i in 1:nx, j in 1:ny
        Mr = Mr_bg
        (lo<=i<=hi && lo<=j<=hi) && (Mr = Mr_l)
        (lo2<=i<=hi2 && lo2<=j<=hi2) && (Mr = Mr_r)
        M[i+halo, j+halo, k, :] = Mr
    end
    return M, nx, ny, nz
end

function main()
    Np = 16; halo = 4   # order-3 needs halo>=4
    Ma = 10.0
    M, nx, ny, nz = build_ic(Np, halo)
    comm = MPI.COMM_WORLD
    decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
    bc = :copy
    halo_exchange_3d!(M, decomp, bc)
    dx = 1.0/Np; dy=dx; dz=dx
    dt = 0.1 * dx / 12.0     # conservative CFL for Ma=10 crossing
    nsteps = 4
    s3max = max(40.0, 4.0 + abs(Ma)/2.0)
    for s in 1:nsteps
        step_highorder_3d!(M, dt, decomp, bc, nx, ny, nz, halo, dx, dy, dz, Ma;
                           order=3, s3max=s3max, use_kfvs_anchor=USE_ANCHOR)
    end
    # exact checksum of the interior
    acc = 0.0; s1 = 0.0; nfin = 0; nrho = 0; nz_neg = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        for q in 1:35
            v = M[i+halo, j+halo, k, q]
            acc += v*v; s1 += v
            isfinite(v) || (nfin += 1)
        end
        (M[i+halo,j+halo,k,1] > 0.0) || (nrho += 1)
    end
    @printf("anchor=%s  L2=%.17e  sum=%.17e  nonfinite=%d  rho<=0=%d\n",
            USE_ANCHOR, sqrt(acc), s1, nfin, nrho)
end
main()

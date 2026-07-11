# kfvs_f4_validate.jl — VALIDATION of the F4 conservative re-projection (Route B).
#
# F4 = per-stage `_anchor_reproject_interior!` on out-of-cone cells (opt-in
# anchor_reproject): measure update M̃ + variant-B invariant restore + θ-limited tail.
# Make-or-break: F4-ON must (i) drive the multi-step Δ2* cross-cone drift to ~0 AND
# (ii) keep energy/mass conservation near the F3 face-flux level (7e-10), NOT the naive
# re-quad level (4e-4). It conserves invariants PER CELL on the ~90% variant-B cells;
# the only leak is the θ-limited ~7% of out-of-cone cells.
#
# Compares, at 8/20/40 steps (Ma=100, 16^3):
#   F3  (use_kfvs_anchor=true, anchor_reproject=false)  — baseline; also confirms the F3
#       path is byte-unchanged by the F4 addition (should match the known 224/704 drift).
#   F4  (use_kfvs_anchor=true, anchor_reproject=true)
# Reports drifted cells, worst margin, mass/energy relative drift.
#
# CPU-only.  HYQMOM_F4V_STEPS default "8,20,40".

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra
MPI.Initialized() || MPI.Init()

function build_ic(Np, halo, Ma)
    C = 1.0; vj = 0.15 * Ma
    bg = InitializeM4_35(0.05, 0.0, 0.0, 0.0, C, 0.0, 0.0, C, 0.0, C)
    l  = InitializeM4_35(1.0,  vj,  vj / 2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    r  = InitializeM4_35(1.0, -vj, -vj / 2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    nx = ny = nz = Np; M = zeros(Float64, nx + 2halo, ny + 2halo, nz, 35)
    Cs = max(1, floor(Int, 0.2 * Np))
    lo = div(Np, 2) - Cs; hi = div(Np, 2); lo2 = div(Np, 2) + 1; hi2 = div(Np, 2) + 1 + Cs
    for k in 1:nz, i in 1:nx, j in 1:ny
        Mr = bg
        (lo <= i <= hi && lo <= j <= hi)     && (Mr = l)
        (lo2 <= i <= hi2 && lo2 <= j <= hi2) && (Mr = r)
        M[i + halo, j + halo, k, :] = Mr
    end
    M, nx, ny, nz
end

reldrift(a, b) = abs(a - b) / max(abs(b), 1e-300)

function run(Np, halo, Ma, s3max, nsteps, dt, reproject::Bool)
    M, nx, ny, nz = build_ic(Np, halo, Ma)
    comm = MPI.COMM_WORLD; bc = :copy
    decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
    halo_exchange_3d!(M, decomp, bc)
    mass0 = 0.0; en0 = 0.0
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = @view M[i+halo, j+halo, k, :]; mass0 += c[1]; en0 += c[3] + c[10] + c[20]
    end
    for _ in 1:nsteps
        step_highorder_3d!(M, dt, decomp, bc, nx, ny, nz, halo, 1.0/Np, 1.0/Np, 1.0/Np, Ma;
                           order = 3, s3max = s3max, use_kfvs_anchor = true, anchor_reproject = reproject)
    end
    nneg = 0; worst = 0.0; mass1 = 0.0; en1 = 0.0; nfin = 0
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = Vector{Float64}(M[i+halo, j+halo, k, :])
        for qq in 1:35; isfinite(c[qq]) || (nfin += 1); end
        mg = realizability_margin(c); (mg < 0) && (nneg += 1; worst = min(worst, mg))
        mass1 += c[1]; en1 += c[3] + c[10] + c[20]
    end
    (nneg=nneg, worst=worst, mass=reldrift(mass1, mass0), en=reldrift(en1, en0), nfin=nfin)
end

function main()
    Ma = 100.0; Np = 16; halo = 4
    dt = 0.12 * (1.0/Np) / (Ma / 2 + 5)
    s3max = max(40.0, 4.0 + abs(Ma) / 2.0)
    set_kfvs_xfloor!(0.0)
    steps = parse.(Int, split(get(ENV, "HYQMOM_F4V_STEPS", "8,20,40"), ","))
    println("KFVS F4 VALIDATION  (Ma=$Ma, Np=$Np)")
    println("="^76)
    @printf("%-22s %6s %12s | %10s %10s %6s\n", "variant / steps", "drift", "worst", "mass", "energy", "nonfin")
    for ns in steps
        f3 = run(Np, halo, Ma, s3max, ns, dt, false)
        @printf("%-22s %6d %12.3e | %10.2e %10.2e %6d\n", "F3  $(ns)step", f3.nneg, f3.worst, f3.mass, f3.en, f3.nfin)
        f4 = run(Np, halo, Ma, s3max, ns, dt, true)
        @printf("%-22s %6d %12.3e | %10.2e %10.2e %6d\n", "F4  $(ns)step", f4.nneg, f4.worst, f4.mass, f4.en, f4.nfin)
    end
    println("\nPASS if F4 drift ≈ 0 AND F4 energy drift ≪ 4e-4 (naive re-quad) and near F3's ~1e-9.")
    println("F3 rows should reproduce the known 224/704/... drift (confirms F4 addition left F3 unchanged).")
    println("DONE.")
end
main()

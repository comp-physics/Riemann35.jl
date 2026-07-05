"""
stress_hiorder3_ma100.jl — HARD 3D stress test for the order=3 WENO5+θ*-IDP scheme.

The canonical killer: Ma=100 crossing-jets (`:crossing_matlab`) — two cubes of jet
fluid slam together at Uc = Ma/√3 ≈ 57.7 per axis into a near-vacuum background.
This is where straight high-order reconstruction loses realizability and NaNs.

Head-to-head at fixed Ma=100 / grid: order 1 (robust, diffusive), order 2 (production
MUSCL), order 3 (WENO5 + θ*-IDP). Report for each: survival (all finite & ρ>0),
ρ range, steps, wall, and global conservation drift. The claim under test: order 3
SURVIVES the Ma=100 collision at high order.

Run:
    source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
    \$JULIA --project=. test/stress_hiorder3_ma100.jl [Nx] [tmax] [Ma]
"""

using MPI
MPI.Initialized() || MPI.Init()
using Riemann35
using Printf

rank = MPI.Comm_rank(MPI.COMM_WORLD)

# CLI overrides (defaults chosen for a ~10-20 min CPU run that reaches the collision)
Nx   = length(ARGS) >= 1 ? parse(Int,     ARGS[1]) : 32
tmax = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.006
Ma   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 100.0

base = (
    Nx = Nx, Ny = Nx, Nz = Nx,
    tmax   = tmax,
    Kn     = 0.0,
    Ma     = Ma,
    flag2D = 0,
    CFL    = 1.0/3.0,
    Nmom   = 35,
    nnmax  = 100000,
    dtmax  = 1000.0,
    rhol   = 1.0,      # jet-cube density
    rhor   = 0.05,     # near-vacuum background (the hard part)
    T      = 1.0,
    r110 = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000,
    homogeneous_z = false,
    debug_output  = false,
    snapshot_interval = 0,
    ic_type       = :crossing_matlab,
    scheme        = :legacy,
    track_corrections = true,
)

function run_one(order)
    params = merge(base, (spatial_order = order,))
    t0 = time()
    M, t, steps, _ = simulation_runner(params)
    wall = time() - t0
    finite  = all(isfinite, M)
    posrho  = finite && all(M[:,:,:,1] .> 0)
    rmin    = finite ? minimum(M[:,:,:,1]) : NaN
    rmax    = finite ? maximum(M[:,:,:,1]) : NaN
    diag    = Riemann35.CORRECTION_DIAG[]
    # Mass (index 1) is the meaningful conserved quantity. The momentum components
    # have ≈0 initial net (symmetric ±Uc jets) so their RELATIVE drift divides by
    # ~0 and is not meaningful — report mass drift specifically.
    mdrift  = diag === nothing ? NaN : diag.conserved_rel_drift[1]
    fhyp    = diag === nothing ? NaN : diag.frac_hyperbolicity
    (order=order, survived=(finite && posrho), t=t, steps=steps,
     rmin=rmin, rmax=rmax, drift=mdrift, fhyp=fhyp, wall=wall)
end

if rank == 0
    println("="^78)
    @printf("HARD 3D STRESS: Ma=%.0f crossing jets, %d^3, rhol=%.2f rhor=%.2f, tmax=%.4f\n",
            Ma, Nx, base.rhol, base.rhor, tmax)
    println("="^78)
end

results = NamedTuple[]
for order in (1, 2, 3)
    if rank == 0
        println("\n--- spatial_order = $order ---")
    end
    local r
    try
        r = run_one(order)
    catch e
        if rank == 0
            @printf("  order=%d THREW: %s\n", order, sprint(showerror, e))
        end
        r = (order=order, survived=false, t=NaN, steps=-1,
             rmin=NaN, rmax=NaN, drift=NaN, fhyp=NaN, wall=NaN)
    end
    push!(results, r)
    if rank == 0
        @printf("  survived=%s  t=%.5f steps=%d  rho∈[%.3e,%.3e]  drift=%.2e  f_hyp=%.2e  wall=%.1fs\n",
                r.survived ? "YES" : "NO", r.t, r.steps, r.rmin, r.rmax, r.drift, r.fhyp, r.wall)
    end
end

if rank == 0
    println("\n" * "="^78)
    println("SUMMARY  (Ma=$(Int(Ma)) crossing jets, $(Nx)^3)")
    println("="^78)
    @printf("  %-6s %-9s %-8s %-24s %-11s %s\n",
            "order", "survived", "steps", "rho_range", "cons_drift", "wall")
    for r in results
        @printf("  %-6d %-9s %-8d [%.2e, %.2e]  %-11.2e %.1fs\n",
                r.order, r.survived ? "YES" : "NO", r.steps, r.rmin, r.rmax, r.drift, r.wall)
    end
    o3 = results[end]
    println()
    if o3.survived
        println("HEADLINE: order=3 (WENO5+θ*-IDP) SURVIVED the Ma=$(Int(Ma)) 3D collision.")
    else
        println("HEADLINE: order=3 did NOT survive — investigate before scaling.")
    end
end

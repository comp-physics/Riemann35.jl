# cpu_crosscheck_ma100.jl — REQUIRED trustworthiness cross-check for the GPU
# fixed-T sharpness sweep.  The two GPU marches use different halo BCs (order-2
# interior+clamp vs order-3 fully-haloed cube).  To confirm the peak RANKING is
# physics and not a BC artifact, we run BOTH orders through the CPU
# `simulation_runner` (main env) at 32^3 to the SAME fixed T and report peak rho.
#
#   $JULIA --project=. gpu/validation/cpu_crosscheck_ma100.jl [T] [N]
using Riemann35
using MPI
using Printf
MPI.Initialized() || MPI.Init()

Tfix = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 8.0e-4
N    = length(ARGS) >= 2 ? parse(Int, ARGS[2])     : 32
Ma   = 100.0
CFL  = 1.0/3.0

function run_order(order::Int)
    dx = 1.0 / N
    dtmax = CFL * dx
    nnmax = ceil(Int, Tfix / dtmax) + 100000
    params = (
        Nx = N, Ny = N, Nz = N,
        tmax = Tfix, Kn = 1.0, Ma = Ma, flag2D = 0, CFL = CFL,
        dx = dx, dy = dx, dz = dx,
        Nmom = 35, nnmax = nnmax, dtmax = dtmax,
        rhol = 1.0, rhor = 0.05, T = 1.0,
        r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 1_000_000,
        homogeneous_z = false,
        enable_memory_tracking = false,
        debug_output = false,
        ic_type = :crossing_matlab,
        spatial_order = order,
        scheme = :legacy,
    )
    M_final, final_time, time_steps, _ = simulation_runner(params)
    rho = M_final[:, :, :, 1]
    peak = maximum(rho); mn = minimum(rho)
    surv = all(isfinite, M_final) && mn > 0.0
    return (peak=peak, mn=mn, surv=surv, t=final_time, steps=time_steps)
end

@printf("=== CPU cross-check: Ma=%.0f :crossing_matlab, N=%d^3, fixed T=%.4e, scheme=:legacy ===\n",
        Ma, N, Tfix)
r2 = run_order(2)
r3 = run_order(3)
@printf("\n order-2 (spatial_order=2): peak_rho=%.6e min=%.4e surv=%s  t=%.4e steps=%d\n",
        r2.peak, r2.mn, r2.surv ? "Y" : "N", r2.t, r2.steps)
@printf(" order-3 (spatial_order=3): peak_rho=%.6e min=%.4e surv=%s  t=%.4e steps=%d\n",
        r3.peak, r3.mn, r3.surv ? "Y" : "N", r3.t, r3.steps)
higher = r2.peak > r3.peak ? "order-2 (MUSCL)" : "order-3 (WENO+IDP)"
@printf("\n CPU ranking: %s has the HIGHER peak (o2/o3 peak ratio = %.4f)\n",
        higher, r2.peak / r3.peak)

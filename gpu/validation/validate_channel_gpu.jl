# validate_channel_gpu.jl — does the GPU order-3 march run a wall-bounded CHANNEL correctly?
#
# WHY. Every wall/channel sweep so far ran single-threaded CPU at ~500 us/cell/step, which
# is why one properly-resolved sigma_p point costs ~28 h. The GPU runs 0.74 us/cell/step.
# Moving the sweeps there is the difference between "a day per point" and "a routine sweep",
# and it is the only thing that makes the resolution + low-Kn study for sigma_p affordable.
#
# WHAT IS NEW AND THEREFORE UNTRUSTED:
#   * bc = :channel      -- face codes (2,2,3,3,0,0): periodic-x, WALL-y, outflow-z.
#   * wall_uw_antisym    -- one scalar -> -Uw at the lo wall, +Uw at the hi wall (Couette).
#   * gx/gy/gz           -- uniform body force (Poiseuille), via _body_force_interior!.
# `wall_ghost_tup` itself is SHARED with the CPU and already fixed, but the GPU's
# mirror-index halo refill at g=8 is entirely separate machinery and has never been
# exercised for a channel. That surrounding code is exactly where the CPU wall bug lived.
#
# GATES:
#   1. it compiles and runs at all;
#   2. MASS is conserved the way the CPU's is -- linear drift, not exponential. This is the
#      check that caught the original defect, and the only one that would catch a repeat.
#   3. Couette antisymmetry: with +/-Uw walls the mean x-momentum must stay at machine zero
#      (it is zero by symmetry, exactly as on the CPU).
#   4. Poiseuille: a body force must produce POSITIVE mean flow and conserve mass.
#
# RESULT when this was written (A100, ny=30, 400 steps):
#   Couette    x-momentum 1.10e-16 (machine zero), mass rel -1.99e-05
#   Poiseuille mean x-mom  8.52e-03 (positive),     mass rel -2.14e-08
# and against the CPU order-3 channel at matched physical time, ny=30:
#   t=0.60  shear 0.12729821 both, mass drift -2.085e-04 both  (all 8 digits)
#   t=1.20  shear 0.12840051 CPU vs 0.12842784 GPU  (2.1e-04 relative)
# at 8.18 ms/step on GPU against ~250-288 ms/step on CPU -- about 31-35x.
#
# Usage: julia -g0 --project=gpu/gpuenv2 gpu/validation/validate_channel_gpu.jl [ny=30] [nsteps=400]
using CUDA, Printf
const GPUDIR = get(ENV, "R35_GPUDIR", normpath(joinpath(@__DIR__, "..")))
include(joinpath(GPUDIR, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
using Riemann35

ny    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
nstep = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400
KNH = 0.2; H = 1.0; T0 = 1.0; rho0 = 1.0; UW = 0.1; GACC = 0.02
nx = 8; nz = 8                      # periodic-x / outflow-z; g=8 needs room
dy = H/ny; dx = dy
lam = KNH*H; tau_ref = lam*sqrt(2.0); kn_tau = 2*tau_ref
dt = 0.2*dy/(5.0*sqrt(T0))

# uniform channel initial state (walls do the driving)
M0 = zeros(35, nx, ny, nz)
Mc = InitializeM4_35(rho0, 0.0, 0.0, 0.0, T0,0,0, T0,0, T0)
for k in 1:nz, j in 1:ny, i in 1:nx, m in 1:35; M0[m,i,j,k] = Mc[m]; end

@printf("dev=%s  grid %dx%dx%d  dt=%.3e  nsteps=%d  Kn_H=%.2f\n",
        CUDA.name(CUDA.device()), nx, ny, nz, dt, nstep, KNH); flush(stdout)

mass(H4)  = sum(@view H4[1, :, :, :])
xmom(H4)  = sum(@view H4[2, :, :, :])

function run_case(name; uw2, antisym, gx)
    G = build_haloed_cube(CuArray(M0))
    Mi = CUDA.zeros(Float64, 35, nx, ny, nz)
    interior_from_cube!(Mi, G); H0 = Array(Mi); m0 = mass(H0)
    tfirst = @elapsed begin
        march3d_order3_gpu!(G, dx, 1.0, 1; dts=fill(dt,1), s3max=40.0, stage_bgk=true,
            Kn=kn_tau, Pr=2/3, omega=0.81, stage_bgk_exact=true,
            bc=:channel, wall_Tw=T0, wall_uw1=0.0, wall_uw2=uw2,
            wall_alpha=1.0, wall_uw_antisym=antisym, gx=gx)
        CUDA.synchronize()
    end
    # sample mass along the march to see the GROWTH LAW, not just an endpoint
    hist = Float64[]
    nchunk = 8; per = max(1, nstep ÷ nchunk)
    trun = @elapsed for _ in 1:nchunk
        march3d_order3_gpu!(G, dx, 1.0, per; dts=fill(dt,per), s3max=40.0, stage_bgk=true,
            Kn=kn_tau, Pr=2/3, omega=0.81, stage_bgk_exact=true,
            bc=:channel, wall_Tw=T0, wall_uw1=0.0, wall_uw2=uw2,
            wall_alpha=1.0, wall_uw_antisym=antisym, gx=gx)
        CUDA.synchronize()
        interior_from_cube!(Mi, G); push!(hist, mass(Array(Mi)))
    end
    interior_from_cube!(Mi, G); Hf = Array(Mi)
    incs = [hist[i]-hist[i-1] for i in 2:length(hist)]
    @printf("\n%s\n", name)
    @printf("  compile+first step %.1f s ; %d steps in %.2f s = %.4f ms/step\n",
            tfirst, nchunk*per, trun, trun/(nchunk*per)*1e3)
    @printf("  mass  %.10f -> %.10f   rel %+.3e\n", m0, mass(Hf), (mass(Hf)-m0)/m0)
    @printf("  mass increments: %s\n", join((@sprintf("%+.2e", x) for x in incs), " "))
    @printf("  mean x-momentum: %.6e\n", xmom(Hf)/(nx*ny*nz))
    flush(stdout)
    (m0=m0, m1=mass(Hf), incs=incs, xmom=xmom(Hf)/(nx*ny*nz), ms=trun/(nchunk*per)*1e3)
end

println("="^96)
println("GPU CHANNEL VALIDATION — bc=:channel, wall code 3, g=8 mirror halo")
println("="^96)
c = run_case("COUETTE  (walls +/-Uw, antisym, no body force)"; uw2=UW, antisym=true, gx=0.0)
p = run_case("POISEUILLE (walls at rest, uniform body force)"; uw2=0.0, antisym=false, gx=GACC)

println()
println("READ: mass increments roughly CONSTANT => linear drift, same as CPU (gate PASS).")
println("      Couette mean x-momentum must be ~machine zero by the +/-Uw antisymmetry.")
println("      Poiseuille mean x-momentum must be clearly POSITIVE (the force does work).")

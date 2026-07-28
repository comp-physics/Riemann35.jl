# validate_thermal_creep_gpu.jl — a wall temperature GRADIENT must drive flow; a uniform
# wall must not.
#
# Thermal creep (transpiration) is gas set in motion along a wall by a tangential
# temperature gradient, from cold toward hot, with no pressure gradient anywhere. It is a
# purely kinetic effect -- Navier-Stokes with no-slip predicts exactly nothing -- so it is
# one of the sharper tests of a wall closure, and it is the reason the per-cell Tw path
# exists at all.
#
# GATES, in order of what they rule out:
#   1. NULL: uniform Tw must give zero mean tangential velocity, to machine zero. If the
#      machinery leaks momentum on its own, everything below is meaningless.
#   2. SIGN: with Tw increasing along +x, the gas must move toward the HOT end (+x).
#      Getting the magnitude wrong is a closure question; getting the sign wrong is a bug.
#   3. LINEARITY: doubling the gradient must roughly double the creep velocity. Thermal
#      creep is linear in dT/dx for small gradients, so a strongly non-linear response
#      means the response is not creep.
#   4. Mass must not run away -- the same conservation gate every wall case gets here,
#      because a channel that is losing mass is not a measurement.
#
# NOTE the convention, which matches the CPU (halo_exchange_3d.jl): for a y-normal wall the
# profile varies along X and spans the full haloed cube extent nfx = nx + 2g.
#
# THE PROFILE MUST BE PERIODIC. x is periodic in the :channel BC, so a linear ramp in Tw is
# a SAWTOOTH -- cold at one edge, hot at the other, discontinuous across the wrap. A first
# version of this test used a ramp and measured creep flowing the WRONG WAY, because the gas
# responds to the jump rather than to the gradient. A sinusoid is the honest test on a
# periodic domain: Tw = T0(1 + A sin(2 pi x/L)) drives u proportional to dT/dx = cos, so the
# creep signal is the projection of u(x) onto cos, and the mean of u over a period is zero
# by construction -- which is why the mean was the wrong statistic too.
# And the period must be nx (the wrap), NOT nfx (the haloed extent): prof[a] == prof[a+nx] is
# what makes the wall temperature a consistent function of physical position at all.
#
# Usage: julia -g0 --project=gpu/gpuenv2 gpu/validation/validate_thermal_creep_gpu.jl [ny=24]
using CUDA, Printf, Statistics
const GPUDIR = get(ENV, "R35_GPUDIR", normpath(joinpath(@__DIR__, "..")))
include(joinpath(GPUDIR, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
using Riemann35

ny    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 24
# Amplitude ladder. Creep is a COEFFICIENT only in the linear regime, so the sweep has to
# reach small enough A to show creep/A approaching a constant; a two-point check at
# A = 0.05, 0.10 cannot distinguish "linear" from "monotone" and this file previously
# claimed the former on that evidence.
AMPS  = length(ARGS) >= 3 ? [parse(Float64,x) for x in split(ARGS[3], ",")] : [0.0125, 0.025, 0.05, 0.10]
nsteps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6000   # tendf ~ 2.8; 600 is NOT converged
KNH = 0.2; H = 1.0; T0 = 1.0; rho0 = 1.0
nx = 24; nz = 8                     # x is the gradient direction, so give it extent
g  = 8
dy = H/ny; dx = dy
lam = KNH*H; tau_ref = lam*sqrt(2.0); kn_tau = 2*tau_ref
dt  = 0.2*dy/(5.0*sqrt(T0))
nfx = nx + 2g

M0 = zeros(35, nx, ny, nz)
Mc = InitializeM4_35(rho0, 0.0, 0.0, 0.0, T0,0,0, T0,0, T0)
for k in 1:nz, j in 1:ny, i in 1:nx, m in 1:35; M0[m,i,j,k] = Mc[m]; end

# Creep signal = projection of u(x) onto cos(2 pi x/L), i.e. onto dT/dx. The plain MEAN of
# u vanishes by symmetry on a periodic domain and would show nothing however large the creep.
# THE SIGNAL IS AT THE WALL, NOT THE y-AVERAGE. In a closed periodic channel creep drives
# the near-wall gas cold -> hot, and mass conservation forces a RETURN FLOW through the core
# going hot -> cold; the net flux is zero. Averaging u over y therefore cancels the two and
# leaves a residual of arbitrary sign -- an earlier version of this test did exactly that and
# read the creep backwards. Project the NEAR-WALL layer, and report the core separately so
# the counterflow is visible rather than inferred.
function creep_amp(Hf, nx, ny; band=:wall)
    js = band === :wall ? vcat(1:2, ny-1:ny) : (ny÷3+1):(2ny÷3)
    ux = [ mean(@view(Hf[2,i,js,:]) ./ @view(Hf[1,i,js,:])) for i in 1:nx ]
    c  = [ cos(2pi*(i-0.5)/nx) for i in 1:nx ]
    2*sum(ux .* c)/nx
end
ubar(Hf) = mean(@view(Hf[2,:,:,:]) ./ @view(Hf[1,:,:,:]))

function run(prof; label="")
    G  = build_haloed_cube(CuArray(M0))
    Mi = CUDA.zeros(Float64, 35, nx, ny, nz)
    march3d_order3_gpu!(G, dx, 1.0, nsteps; dts=fill(dt,nsteps), s3max=40.0,
        stage_bgk=true, Kn=kn_tau, Pr=2/3, omega=0.81, stage_bgk_exact=true,
        bc=:channel, wall_Tw=T0, wall_uw1=0.0, wall_uw2=0.0, wall_alpha=1.0,
        wall_Tw_prof=prof)
    CUDA.synchronize()
    interior_from_cube!(Mi, G); Hf = Array(Mi)
    m = sum(@view Hf[1,:,:,:]); m0 = nx*ny*nz*rho0
    (u = ubar(Hf), amp = creep_amp(Hf, nx, ny; band=:wall),
     core = creep_amp(Hf, nx, ny; band=:core), dmass = (m-m0)/m0)
end

@printf("thermal creep, GPU order 3: %dx%dx%d, Kn_H=%.2f, %d steps, dev=%s\n",
        nx, ny, nz, KNH, nsteps, CUDA.name(CUDA.device())); flush(stdout)
println("="^92)

# 1. NULL
r0 = run(nothing)
@printf("[1] uniform Tw      wall = %+11.4e   core = %+11.4e   CREEP(w-c) = %+11.4e   dmass/m = %+.2e\n",
        r0.amp, r0.core, r0.amp - r0.core, r0.dmass); flush(stdout)

# 2/3. PERIODIC sinusoidal Tw. Sign convention: Tw = T0(1 + A sin(2 pi x/L)) has
# dT/dx proportional to +cos, so creep (cold -> hot) gives a POSITIVE cos-projection.
for A in AMPS
    # PERIOD MUST BE nx, NOT nfx. The x wrap has period nx, so a consistent periodic wall
    # temperature must satisfy prof[a] == prof[a+nx]; building the sinusoid over the HALOED
    # extent nfx violates that and mis-phases it against the interior. Interior cell i sits at
    # cube index g+i, so this form gives prof[g+i] = T0(1 + A sin(2pi(i-0.5)/nx)) -- exactly
    # in phase with the cos projection below.
    prof = [T0*(1 + A*sin(2pi*(a - g - 0.5)/nx)) for a in 1:nfx]
    r = run(prof)
    cr = r.amp - r.core
    @printf("[2] A = %6.4f sin(x)  wall = %+11.4e  core = %+11.4e  CREEP(w-c) = %+11.4e  creep/A = %+.4e  dmass/m = %+.2e\n",
            A, r.amp, r.core, cr, cr/A, r.dmass); flush(stdout)
end
println("="^92)
println("READ: the creep signal is WALL MINUS CORE, not either band alone.")
println("  Creep is a SHEAR across the Knudsen layer, and both bands here sit on a large")
println("  common-mode circulation (~-2e-3) that swamps it. Every absolute statistic tried")
println("  during development read negative for that reason and was mistaken for a sign bug.")
println("  MEASURED 2026-07-28 (Kn_H=0.20, ny=24, order 3, A100). The previous reference block here
  did NOT reproduce: it claimed +1.10e-04 at A=0.05 (actual +1.589e-04) and "converged to
  0.4% between tendf 1.4 and 2.8" (actual 3.3% over that interval). Corrected:
     null (uniform Tw): CREEP is EXACTLY +0.000e+00, and dmass is exactly 0;
     A=0.05 -> +1.5894e-04, A=0.10 -> +2.9510e-04 at 12000 steps, POSITIVE = cold -> hot;
     convergence: 3000->6000 steps moves it 3.3%, 6000->12000 only 0.16-0.66%,
       so tendf ~ 2.8 (6000 steps) is converged and 3000 is NOT;
     linearity: the A=0.05 -> 0.10 ratio is 1.86x for a 2x gradient, i.e. SUB-linear at
       these amplitudes -- run the default 4-point ladder and read the creep/A column.
  600 steps is tendf ~ 0.28 and is NOT converged -- use >= 3000.")

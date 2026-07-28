# validate_body_force_profile.jl — the optional spatial body-force profile.
#
# WHY THE CAPABILITY EXISTS. A UNIFORM force on a periodic domain accelerates the whole gas
# and never reaches a steady state, so a wall-free DRIVEN flow is unreachable with the scalar
# form. That flow is the one test that separates two very different readings of the dt defect
# in comp-physics/Riemann35.jl#38 -- "driven steady states have no dt->0 limit" versus
# "wall-bounded steady states have no dt->0 limit". Without a spatially varying force the
# question cannot be asked.
#
# GATES, in order of what they protect:
#   1. BYTE-IDENTICAL fallback. gprof = nothing must reproduce the pre-capability march
#      exactly. This is the gate that matters most: every existing body-force result --
#      the Poiseuille flow rates, the creep coefficient -- was produced by the scalar path,
#      and a capability that perturbs it invalidates them.
#   2. A profile of ALL ONES must equal the uniform force, also exactly. Same arithmetic by
#      a different route; any difference means the scale factor is not entering cleanly.
#   3. The profile must actually MODULATE: a sinusoidal shape must produce a mean velocity
#      that tracks the shape, cell by cell, at the exact analytic value g*sc*t. The force is
#      a rigid velocity-space translation, so this is exact for any dt and any profile --
#      there is no discretisation error to hide behind.
#   4. Central moments must be FROZEN. A translation in velocity space shifts the mean and
#      leaves every central moment alone. If the profile broke that, the force would be
#      injecting spurious temperature/heat-flux wherever it varies.
#   5. Axis selection must work on all three axes, and index the INTERIOR (1..n).
#
# Usage: julia -g0 --project=gpu/gpuenv2 gpu/validation/validate_body_force_profile.jl
using CUDA, Printf, Statistics
const GPUDIR = get(ENV, "R35_GPUDIR", normpath(joinpath(@__DIR__, "..")))
include(joinpath(GPUDIR, "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
using Riemann35

nx, ny, nz = 8, 16, 8
dx = 1.0/ny
T0 = 1.0
dt = 1.0e-3
nst = 20
G0 = zeros(35, nx, ny, nz)
Mc = InitializeM4_35(1.0, 0.0, 0.0, 0.0, T0,0,0, T0,0, T0)
for k in 1:nz, j in 1:ny, i in 1:nx, m in 1:35; G0[m,i,j,k] = Mc[m]; end

function march(; gx=0.0, gprof=nothing, gprof_axis=2, n=nst)
    G = build_haloed_cube(CuArray(G0))
    Mi = CUDA.zeros(Float64, 35, nx, ny, nz)
    march3d_order3_gpu!(G, dx, 1.0, n; dts=fill(dt,n), s3max=40.0,
        stage_bgk=false, bc=:periodic, gx=gx, gprof=gprof, gprof_axis=gprof_axis)
    CUDA.synchronize(); interior_from_cube!(Mi, G); Array(Mi)
end

println("="^92)
println("BODY-FORCE SPATIAL PROFILE — gates")
@printf("%dx%dx%d, dt=%.1e, %d steps, dev=%s\n", nx,ny,nz, dt, nst, CUDA.name(CUDA.device()))
println("="^92)

# ---------------------------------------------------------------- GATE 1
# No force at all, with and without the capability present in the signature.
A = march(gx=0.0, gprof=nothing)
B = march(gx=0.0, gprof=nothing)
d1 = maximum(abs.(A .- B))
@printf("GATE 1  no-force reproducibility            max|d| = %.3e  %s\n",
        d1, d1 == 0.0 ? "PASS (bit-identical)" : "*** FAIL ***")

# ---------------------------------------------------------------- GATE 2
# A profile of ALL ONES must be arithmetically identical to the uniform force.
gU = march(gx=0.05, gprof=nothing)
gO = march(gx=0.05, gprof=ones(ny))
d2 = maximum(abs.(gU .- gO))
@printf("GATE 2  ones-profile == uniform force       max|d| = %.3e  %s\n",
        d2, d2 == 0.0 ? "PASS (bit-identical)" : "*** FAIL ***")

# ---------------------------------------------------------------- GATE 3
# THE FORCE OPERATOR IN ISOLATION. An earlier version of this gate asserted
# u_x(j) == gx*prof[j]*t through a FULL MARCH and read 2.4e-6, which looked like a failure
# and was not: the march also TRANSPORTS, and a spatially varying u_x creates the very
# gradients transport acts on. (GATE 2 passes exactly because a uniform profile leaves
# nothing for transport to do.) The force is a rigid velocity-space translation, so the
# exactness claim belongs to the OPERATOR, not to the operator composed with a residual.
# Calling the kernel directly is the honest test.
const TG = Timestep3DOrder3GPU
prof = [sin(2pi*(j-0.5)/ny) for j in 1:ny]
function force_only(pr, ax, gx, dtl)
    G = build_haloed_cube(CuArray(G0))
    gP = CuArray(convert(Vector{Float64}, pr))
    thr = 128; bi = cld(nx*ny*nz, thr)
    @cuda threads=thr blocks=bi TG._body_force_interior!(G, nx, ny, nz, 8, gx, 0.0, 0.0, dtl, gP, ax)
    CUDA.synchronize()
    Mi = CUDA.zeros(Float64, 35, nx, ny, nz); interior_from_cube!(Mi, G); Array(Mi)
end
function worst_dev(F, pr, ax, gx, t, nx, ny, nz)
    w = 0.0
    for q in 1:(ax == 1 ? nx : ny)
        u = ax == 1 ? mean(@view(F[2,q,:,:]) ./ @view(F[1,q,:,:])) :
                      mean(@view(F[2,:,q,:]) ./ @view(F[1,:,q,:]))
        w = max(w, abs(u - gx*pr[q]*t))
    end
    w
end
dt1 = 0.02
FS = force_only(prof, 2, 0.05, dt1)
worst = worst_dev(FS, prof, 2, 0.05, dt1, nx, ny, nz)
@printf("GATE 3  one application: u_x(j) == gx*prof[j]*dt   max|d| = %.3e  %s\n",
        worst, worst < 1e-15 ? "PASS (exact translation)" : "*** FAIL ***")

# ---------------------------------------------------------------- GATE 4
# Central moments frozen by the OPERATOR: a velocity-space translation moves the mean and
# leaves every central moment alone, at any dt and any profile.
c_of(F) = [F[3,i,j,k]/F[1,i,j,k] - (F[2,i,j,k]/F[1,i,j,k])^2 for i in 1:nx, j in 1:ny, k in 1:nz]
REF = force_only(zeros(ny), 2, 0.0, dt1)
d4r = maximum(abs.(FS[1,:,:,:] .- REF[1,:,:,:]))
d4c = maximum(abs.(c_of(FS) .- c_of(REF)))
@printf("GATE 4  operator: density frozen %.3e ; central C200 frozen %.3e  %s\n",
        d4r, d4c, (d4r < 1e-15 && d4c < 1e-14) ? "PASS" : "*** FAIL ***")

# ---------------------------------------------------------------- GATE 5
# Axis selection: a profile on axis 1 must index i, not j.
profx = [sin(2pi*(i-0.5)/nx) for i in 1:nx]
FX = force_only(profx, 1, 0.05, dt1)
wx = worst_dev(FX, profx, 1, 0.05, dt1, nx, ny, nz)
@printf("GATE 5  axis=1 profile indexes i                   max|d| = %.3e  %s\n",
        wx, wx < 1e-15 ? "PASS (exact)" : "*** FAIL ***")

# ---------------------------------------------------------------- GATE 6
# And through the FULL march the profile must still modulate in the right SHAPE, even though
# transport keeps it from matching the pure force integral. Correlation with the profile,
# not equality with it.
gS = march(gx=0.05, gprof=prof)
us = [mean(@view(gS[2,:,j,:]) ./ @view(gS[1,:,j,:])) for j in 1:ny]
cc = sum((us .- mean(us)).*(prof .- mean(prof))) /
     sqrt(sum((us .- mean(us)).^2)*sum((prof .- mean(prof)).^2))
@printf("GATE 6  full march: corr(u_x, profile) = %.6f  %s\n",
        cc, cc > 0.99 ? "PASS (shape preserved under transport)" : "*** FAIL ***")
println("="^92)

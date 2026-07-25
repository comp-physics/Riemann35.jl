#!/usr/bin/env julia
# validate_stagebgk_viscosity_gpu.jl -- CONTROL TEST for a suspected O(1) transport error.
#
# FINDING UNDER TEST. The shear-mode decay rate measured through the production order-3
# stepper is ~0.52x the Navier-Stokes prediction nu*k^2, flat across an 8x range of
# k*lambda (0.039..0.314) and grid-converged to 0.1%. Both shear and entropy modes
# converge to the SAME ~0.53, so Pr is unaffected (the factor cancels) but mu and k are
# each ~1.9x too small.
#
# SUSPECTED CAUSE. `stage_bgk` applies the collision with the FULL dt after every one of
# the three SSP-RK3 stages. With L=0 and C(sigma)=e*sigma:
#     u1 = e u0
#     u2 = e(3/4 + e/4) u0
#     u3 = e(1/3 + 2/3 e(3/4 + e/4)) u0
# expanding at e = 1-x, x = dt/tau:  u3/u0 = 1 - (11/6) x.
# So the deviatoric stress relaxes at (11/6)/tau instead of 1/tau -- over-relaxation by
# 1.83x, predicting a measured/theory ratio of 6/11 = 0.5455. That is an O(dt) local
# error in the solution, i.e. an O(1) error in the transport coefficients, NOT the
# first-order splitting consistency the code comments claim. `scheme = :recommended`
# enables stage_bgk by DEFAULT.
#
# THE CONTROL. Run the identical measurement two ways:
#   stage : collision every RK stage (the shipped default)  -> expect ratio ~ 6/11
#   once  : collisionless stages, ONE exact-exponential collision per step
#           -> expect ratio ~ 1 if the diagnosis is right
# The "once" path marches 1 step at a time with stage_bgk=false and applies the shipped
# bgk_relax_tup on the host (the slab interior is tiny, so the round-trip is cheap).
#
# If "once" comes out ~1 and "stage" ~0.55, the diagnosis is confirmed and this is a real
# defect in the default scheme. If BOTH are ~0.53, the cause is elsewhere and the
# stage-splitting is exonerated -- report that instead.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "gpuenv2"); io = devnull)

using CUDA, Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl"))
using .Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!
using Riemann35: InitializeM4_35
using Riemann35.ReconDev: bgk_relax_tup

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const KN    = parse(Float64, get(ENV, "SV_KN", "0.25"))
const NX    = parse(Int,     get(ENV, "SV_NX", "80"))
const LDOM  = parse(Float64, get(ENV, "SV_L", "10.0"))
const EPS   = parse(Float64, get(ENV, "SV_EPS", "1e-3"))
const PRN   = parse(Float64, get(ENV, "SV_PR", string(2/3)))
const OMG   = parse(Float64, get(ENV, "SV_OMEGA", "0.5"))
const NPER  = parse(Float64, get(ENV, "SV_NPER", "2.0"))
const BC_PX = ((2, 2, 0, 0, 0, 0), (false, false, false, false, false, false))

mode_amp(q, x, k) = 2 * mean(q .* sin.(k .* x))

function run_shear(collide::Symbol; Kn = KN, Nx = NX, L = LDOM, eps = EPS,
                   Pr = PRN, omega = OMG, nper = NPER, nchunk = 12)
    rho0, T0 = 1.0, 1.0
    dx = L/Nx; k = 2pi/L
    x = [(i-0.5)*dx for i in 1:Nx]
    M0 = zeros(Float64, 35, Nx, 1, 1)
    for i in 1:Nx
        M0[:, i, 1, 1] = InitializeM4_35(rho0, 0.0, eps*sin(k*x[i]), 0.0, T0,0,0, T0,0, T0)
    end
    tau_ref  = (Kn/2) * T0^(omega-1.0) / rho0
    decay_th = (T0*tau_ref) * k^2
    dt   = 0.2*dx/(5.0*sqrt(T0))
    nst  = max(24, ceil(Int, (nper/decay_th)/dt))
    per  = max(1, nst ÷ nchunk)

    G  = build_haloed_cube(CuArray(M0))
    Mi = CUDA.zeros(Float64, 35, Nx, 1, 1)
    amp() = begin
        interior_from_cube!(Mi, G); A = Array(Mi)
        abs(mode_amp([A[6,i,1,1]/A[1,i,1,1] for i in 1:Nx], x, k))
    end

    ts = Float64[]; amps = Float64[]; t = 0.0; done = 0
    while done < nst
        if collide === :stage || collide === :stage_exact
            nsub = min(per, nst - done)
            march3d_order3_gpu!(G, dx, 0.0, nsub; dts = fill(dt, nsub),
                                stage_bgk = true, Kn = Kn, Pr = Pr, omega = omega,
                                stage_bgk_exact = (collide === :stage_exact), bc = BC_PX)
            done += nsub; t += nsub*dt
        else
            # collisionless stages, then ONE collision per step on the host
            nsub = min(per, nst - done)
            for _ in 1:nsub
                march3d_order3_gpu!(G, dx, 0.0, 1; dts = [dt], stage_bgk = false, bc = BC_PX)
                interior_from_cube!(Mi, G); A = Array(Mi)
                for i in 1:Nx
                    mt = ntuple(q -> A[q, i, 1, 1], Val(35))
                    o  = bgk_relax_tup(mt, dt, Kn, Pr, omega)
                    for q in 1:35; A[q, i, 1, 1] = o[q]; end
                end
                CUDA.unsafe_free!(G)
                G = build_haloed_cube(CuArray(A))
            end
            done += nsub; t += nsub*dt
        end
        a = amp(); (isfinite(a) && a > 0) && (push!(ts, t); push!(amps, a))
    end
    length(amps) < 4 && return (NaN, decay_th)
    la = log.(amps); n = length(ts)
    slope = (n*sum(ts .* la) - sum(ts)*sum(la)) / (n*sum(ts.^2) - sum(ts)^2)
    (-slope, decay_th)
end

println("="^88)
println("CONTROL: does stage_bgk's per-stage collision cause the ~0.53 viscosity deficit?")
@printf("Kn=%.3f  Nx=%d  L=%.1f  Pr=%.4f  omega=%.2f   k*lambda=%.4f\n",
        KN, NX, LDOM, PRN, OMG, (2pi/LDOM)*(KN/2))
@printf("prediction: stage -> 6/11 = %.4f ;  stage_exact -> ~1.0 ;  once -> ~1.0\n", 6/11)
println("="^88)
@printf("%-8s %16s %16s %10s\n", "collide", "measured", "theory", "ratio")
res = Dict{Symbol,Float64}()
for c in (:stage, :stage_exact, :once)
    r, th = run_shear(c)
    res[c] = r/th
    @printf("%-8s %16.6e %16.6e %10.4f\n", c, r, th, r/th)
end
println("="^88)
if haskey(res,:stage_exact)
    @printf("\nstage_bgk_exact ratio = %.4f  (target 1.0; this is the fix under test)\n", res[:stage_exact])
    println(abs(res[:stage_exact]-1) < 0.05 ?
        "=> FIX CONFIRMED: corrected composite recovers NS transport with stage_bgk ON." :
        "=> FIX DID NOT WORK: report the number, do not claim the correction works.")
end
if haskey(res,:stage) && haskey(res,:once) && all(isfinite, values(res))
    @printf("stage/once = %.4f   (11/6 = %.4f expected if per-stage collision is the cause)\n",
            res[:once]/res[:stage], 11/6)
    if abs(res[:once] - 1) < 0.08 && abs(res[:stage] - 6/11) < 0.08
        println("=> CONFIRMED: per-stage collision over-relaxes; once-per-step recovers NS transport.")
    elseif abs(res[:once] - res[:stage]) < 0.05
        println("=> EXONERATED: both paths agree, so stage splitting is NOT the cause. Look elsewhere.")
    else
        println("=> INCONCLUSIVE: neither branch matches cleanly. Report the numbers, not a verdict.")
    end
end

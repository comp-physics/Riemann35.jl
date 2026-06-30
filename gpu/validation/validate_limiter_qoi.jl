#!/usr/bin/env julia
# validate_limiter_qoi.jl
#
# INTEGRAL-QoI gate for the LIMITER-active high-order march at Ma=100. This is the
# "good-enough" correctness test for numerics-CHANGING options (e.g. the Bunch-Kaufman
# realizability solver `REALIZ_SOLVER = PivotRealiz()`), where byte-identity is the wrong
# bar: the Ma=100 colliding-jets flow is chaotic, so two valid methods drift ~20% POINTWISE
# (turbulent rearrangement, not error). The robust metric is INTEGRAL quantities of interest
# over a SHORT, limiter-active march with a PRESCRIBED dt sequence (same physical time):
#
#   * conserved integrals (mass, momentum)  -> conservation intact   (tight tol 1e-7)
#   * solution integrals (energy, peak/min rho, total variation = a diffusion proxy)
#                                            -> physics preserved     (looser tol 1e-5)
#
# The march is kept SHORT (2 steps): at 2 steps the integral QoIs still resolve a method
# difference (a correct solver matches the reference to <1e-9 on the conserved QoIs and
# <1e-8 on total variation); by ~20 steps chaos amplifies any difference ~1e4x and the test
# loses discriminating power. The reference values below are the Jacobi (`JacobiRealiz()`)
# result; the CURRENT compiled solver is run against them. A genuinely wrong solver (e.g. an
# unpivoted-LDL^T realizability test) blows past these tolerances; the exact Bunch-Kaufman
# inertia passes with a large margin.
#
# ENV: read r3d_M.f64 / r3d.meta from $RIEMANN35_DATA (default <repo>/../data); write nothing.

import Pkg
Pkg.activate(joinpath(joinpath(@__DIR__, ".."), "gpuenv2"))

using CUDA, Printf
include(joinpath(joinpath(@__DIR__, ".."), "timestep3d_gpu.jl"))
using .Timestep3DGPU: march3d_gpu!

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const DATA = get(ENV, "RIEMANN35_DATA", joinpath(joinpath(@__DIR__, ".."), "..", "data"))
const HO_VACUUM_FLOOR = 0.001

# --- prescribed 2-step dt sequence (Jacobi CFL on the r3d state); marching with these makes
#     the comparison SAME-TIME, isolating the solver from the small dt divergence. ---
const DTS = (5.26805487824261219e-05, 5.33350438635948006e-05)

# --- reference integral QoIs: limiter march, 2 steps, prescribed DTS, JacobiRealiz(). ---
const REF = (mass   = 2.02170175416768529e+03,
             px     = -2.40546738481778448e+04,
             py     = -2.38316021186858488e+04,
             pz     = -2.38310315113372235e+04,
             energy = 1.46666472982312553e+07,
             peak   = 1.06162912736596282e+00,
             lo     = 7.06001522380309519e-04,
             tv     = 6.66078151545587275e+02)

const TOL_CONSERVED = 1.0e-7   # mass, momentum
const TOL_SOLUTION  = 1.0e-5   # energy, peak/min rho, total variation

rel(a, b) = abs(a - b) / max(abs(b), 1.0)

function main()
    meta = split(strip(read(joinpath(DATA, "r3d.meta"), String)), '\n')
    n  = parse(Int,     strip(meta[1]))
    dx = parse(Float64, strip(meta[2]))
    Ma = parse(Float64, strip(meta[3]))
    M0 = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_M.f64")))), 35, n, n, n)
    @printf("loaded n=%d dx=%.8g Ma=%.4g  (limiter march, %d steps, prescribed dt)\n", n, dx, Ma, length(DTS))

    Md = CuArray(M0)
    march3d_gpu!(Md, dx, Ma, length(DTS); dts=collect(DTS), vacuum_floor=HO_VACUUM_FLOOR, limiter=true)
    M = Array(Md); rho = @view M[1, :, :, :]
    @assert all(isfinite, M) "march produced non-finite moments"

    mass   = sum(rho)
    px     = sum(@view M[2, :, :, :]); py = sum(@view M[6, :, :, :]); pz = sum(@view M[16, :, :, :])
    energy = sum(@view M[3, :, :, :]) + sum(@view M[10, :, :, :]) + sum(@view M[20, :, :, :])
    peak   = maximum(rho); lo = minimum(rho)
    tv     = sum(abs.(diff(rho, dims=1))) + sum(abs.(diff(rho, dims=2))) + sum(abs.(diff(rho, dims=3)))

    conserved = (("mass", rel(mass, REF.mass)), ("px", rel(px, REF.px)),
                 ("py", rel(py, REF.py)), ("pz", rel(pz, REF.pz)))
    solution  = (("energy", rel(energy, REF.energy)), ("peak_rho", rel(peak, REF.peak)),
                 ("min_rho", rel(lo, REF.lo)), ("total_variation", rel(tv, REF.tv)))

    println("\n=== INTEGRAL QoIs vs Jacobi reference (same time) ===")
    ok = true
    for (nm, r) in conserved
        pass = r <= TOL_CONSERVED; ok &= pass
        @printf("  CONSERVED %-16s rel=%.3e  (tol %.0e)  %s\n", nm, r, TOL_CONSERVED, pass ? "PASS" : "FAIL")
    end
    for (nm, r) in solution
        pass = r <= TOL_SOLUTION; ok &= pass
        @printf("  SOLUTION  %-16s rel=%.3e  (tol %.0e)  %s\n", nm, r, TOL_SOLUTION, pass ? "PASS" : "FAIL")
    end
    @printf("\nGATE: %s\n", ok ? "PASS" : "FAIL")
    return ok
end

exit(main() ? 0 : 1)

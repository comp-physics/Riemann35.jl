#!/usr/bin/env julia
# validate_spatial_transport_gpu.jl -- GPU version of test/validate_spatial_transport.jl.
#
# WHY GPU: the CPU order-3 stepper runs ~0.6 s/step on a small slab, so the dx sweep AND
# the k*lambda sweep together are many hours. On GPU the march is orders faster, but
# ptxas costs ~13 min for the order-3 kernel stack -- so EVERY configuration is run in
# ONE process. Do not split this into per-config invocations; that pays the compile
# repeatedly for no reason.
#
# WHAT THE CPU PILOT FOUND (and why the sweep below is shaped this way): at k*lambda =
# 0.314 the measured decay rates came out BELOW nu*k^2 (shear ratio 0.512, entropy
# 0.803). That is the wrong sign for numerical-viscosity contamination, which ADDS
# dissipation -- so discretization was not the limiter. The likely cause is finite
# k*lambda: the kinetic dispersion relation damps sub-Navier-Stokes, making nu*k^2 the
# wrong target. Hence this sweeps k*lambda as well as dx, and reports both trends. The
# NS coefficient should only be recovered in the joint limit dx -> 0 AND k*lambda -> 0.
#
# Env: STG_KNS (comma list of Kn), STG_NXS (comma list), STG_L, STG_EPS, STG_PR,
#      STG_OMEGA, STG_NPER.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "gpuenv2"); io = devnull)

using CUDA, Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl"))
using .Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!
using Riemann35: InitializeM4_35, envparam, print_run_header

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const KNS   = parse.(Float64, split(envparam("STG_KNS", "1.0,0.5,0.25"), ","))
const NXS   = parse.(Int,     split(envparam("STG_NXS", "80,160,320"), ","))
const LDOM  = parse(Float64, envparam("STG_L", "10.0"))
const EPS   = parse(Float64, envparam("STG_EPS", "1e-3"))
const PRN   = parse(Float64, envparam("STG_PR", string(2/3)))
const OMG   = parse(Float64, envparam("STG_OMEGA", "0.5"))
const NPER  = parse(Float64, envparam("STG_NPER", "2.0"))

# periodic x, outflow y/z  (codes: 0=outflow, 1=inlet, 2=periodic)
const BC_PX = ((2, 2, 0, 0, 0, 0), (false, false, false, false, false, false))

mode_amp(q, x, k) = 2 * mean(q .* sin.(k .* x))

"""
    run_mode_gpu(kind, Nx, Kn) -> (rate, rate_theory, dx, klam)

March one linear mode on the GPU and least-squares fit its exponential decay rate.
Transverse extents are 1 cell (uniform in y,z; the outflow halo clamp preserves that),
so this is a thin slab rather than a wasteful cube.
"""
function run_mode_gpu(kind::Symbol, Nx::Int, Kn::Float64;
                      L = LDOM, eps = EPS, Pr = PRN, omega = OMG, nper = NPER,
                      nchunk = 12)
    rho0, T0 = 1.0, 1.0
    dx = L/Nx
    k  = 2pi/L
    x  = [(i - 0.5)*dx for i in 1:Nx]

    M0 = zeros(Float64, 35, Nx, 1, 1)
    for i in 1:Nx
        if kind === :shear
            M0[:, i, 1, 1] = InitializeM4_35(rho0, 0.0, eps*sin(k*x[i]), 0.0, T0,0,0, T0,0, T0)
        else
            r = rho0*(1 + eps*sin(k*x[i])); T = rho0*T0/r
            M0[:, i, 1, 1] = InitializeM4_35(r, 0.0, 0.0, 0.0, T,0,0, T,0, T)
        end
    end

    tau_ref  = (Kn/2) * T0^(omega-1.0) / rho0
    nu_th    = T0 * tau_ref
    decay_th = (kind === :shear ? nu_th : nu_th/Pr) * k^2
    Tend = nper/decay_th
    dt   = 0.2 * dx / (5.0*sqrt(T0))
    nst  = max(24, ceil(Int, Tend/dt))
    per  = max(1, nst ÷ nchunk)

    G  = build_haloed_cube(CuArray(M0))
    Mi = CUDA.zeros(Float64, 35, Nx, 1, 1)
    amp() = begin
        interior_from_cube!(Mi, G); A = Array(Mi)
        q = kind === :shear ? [A[6, i, 1, 1]/A[1, i, 1, 1] for i in 1:Nx] :
                              [A[1, i, 1, 1] for i in 1:Nx]
        abs(mode_amp(q, x, k))
    end

    ts = Float64[]; amps = Float64[]; t = 0.0
    done = 0
    while done < nst
        nsub = min(per, nst - done)
        march3d_order3_gpu!(G, dx, 0.0, nsub; dts = fill(dt, nsub),
                            stage_bgk = true, Kn = Kn, Pr = Pr, omega = omega, bc = BC_PX)
        done += nsub; t += nsub*dt
        a = amp()
        (isfinite(a) && a > 0) && (push!(ts, t); push!(amps, a))
    end
    length(amps) < 4 && return (NaN, decay_th, dx, k*(Kn/2))
    la = log.(amps); n = length(ts)
    slope = (n*sum(ts .* la) - sum(ts)*sum(la)) / (n*sum(ts.^2) - sum(ts)^2)
    (-slope, decay_th, dx, k*(Kn/2))
end

"linear fit y = a + b*x, return (a, b)"
function linfit(xs, ys)
    n = length(xs); sx = sum(xs); sy = sum(ys)
    b = (n*sum(xs .* ys) - sx*sy) / (n*sum(xs.^2) - sx^2)
    ((sy - b*sx)/n, b)
end

print_run_header("SPATIAL TRANSPORT (GPU): do mu and k emerge from the discretized PDE?";
                 extra = ("order" => "3", "device" => CUDA.name(CUDA.device())))
println("ratio = measured decay / (nu k^2 or alpha k^2). NS theory is recovered only as")
println("BOTH dx -> 0 and k*lambda -> 0; the CPU pilot showed finite k*lambda pushes the")
println("ratio BELOW 1 (sub-NS kinetic damping), which is not a discretization error.")
println("="^100)

summary = Dict{Tuple{Symbol,Float64},Any}()
for kind in (:shear, :entropy), Kn in KNS
    klam = (2pi/LDOM)*(Kn/2)
    @printf("\n--- %-7s  Kn=%.3f  k*lambda=%.4f ---\n", kind, Kn, klam)
    @printf("%-7s %-9s %14s %14s %9s\n", "Nx", "dx", "measured", "theory", "ratio")
    dxs = Float64[]; rs = Float64[]; th = NaN
    for Nx in NXS
        r, dth, dx, _ = run_mode_gpu(kind, Nx, Kn)
        th = dth
        @printf("%-7d %-9.4f %14.6e %14.6e %9.4f\n", Nx, dx, r, dth, r/dth)
        isfinite(r) && (push!(dxs, dx); push!(rs, r))
    end
    if length(dxs) >= 2
        r0, sl = linfit(dxs, rs)
        @printf("%-7s %-9s %14.6e %14.6e %9.4f  <- dx->0 (slope %.2e)\n", "dx->0","0", r0, th, r0/th, sl)
        summary[(kind, Kn)] = (r0 = r0, th = th, klam = klam,
                               fin = rs[end], contam = (rs[end]-r0)/max(rs[end],1e-30))
    end
end

println("\n", "="^100)
println("JOINT LIMIT: dx->0 ratio vs k*lambda  (extrapolate to k*lambda -> 0)")
@printf("%-9s %10s %12s %12s %12s\n", "mode", "k*lambda", "ratio(dx->0)", "numerical%", "Pr(dx->0)")
for Kn in KNS
    s = get(summary, (:shear, Kn), nothing); e = get(summary, (:entropy, Kn), nothing)
    (s === nothing || e === nothing) && continue
    @printf("%-9s %10.4f %12.4f %12.1f %12s\n", "shear", s.klam, s.r0/s.th, 100*s.contam, "")
    @printf("%-9s %10.4f %12.4f %12.1f %12.4f\n", "entropy", e.klam, e.r0/e.th, 100*e.contam, s.r0/e.r0)
end
# extrapolate the shear ratio to k*lambda -> 0
let ks = Float64[], rr = Float64[]
    for Kn in KNS
        s = get(summary, (:shear, Kn), nothing)
        s === nothing && continue
        push!(ks, s.klam); push!(rr, s.r0/s.th)
    end
    if length(ks) >= 2
        a, b = linfit(ks, rr)
        @printf("\nshear ratio extrapolated to k*lambda->0 : %.4f  (slope %.3f)\n", a, b)
        println(abs(a - 1) < 0.05 ?
            "=> NS viscosity RECOVERED in the joint limit (within 5%)." :
            "=> NS viscosity NOT recovered; report the trend, do not quote a single number.")
    end
end
println("="^100)

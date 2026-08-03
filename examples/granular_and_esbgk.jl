#!/usr/bin/env julia
# granular_and_esbgk.jl — THE TWO CAPABILITIES ADDED 2026-08-03, RUNNABLE ON A LAPTOP.
#
# Both are CPU-only here on purpose: the ES-BGK collision operator used to be GPU-only, which
# is part of how a real bug in it survived (CI could never exercise it). Everything below runs
# without a GPU, and every number printed is checked against a closed form rather than against
# another run of the same code.
#
#   1. `collide_es_cpu!`  — ES-BGK with a CORRELATED equilibrium (Riemann35.jl #73)
#   2. `granular_drain_tup` — inelastic energy drain for granular flow (#75)
#
# ---------------------------------------------------------------------------------------
# WHY THESE EXIST, because both were bugs before they were features
# ---------------------------------------------------------------------------------------
#
# ES-BGK. The equilibrium was built as a PRODUCT of three independent 1D Gaussians:
#
#     feq = A exp(bx vx + cx vx^2) exp(by vy + cy vy^2) exp(bz vz + cz vz^2)
#
# A product of 1D Gaussians has a strictly DIAGONAL covariance. But the ES-BGK equilibrium is
# Lambda = (1-k) Theta I + k C, whose off-diagonal entries are k*C_ij. Those cannot be
# represented in product form, so feq carried sigma_xy = 0 and the update degenerated to
# sigma_xy*exp(-Pr*y): SHEAR stress relaxed at Pr/tau while DIAGONAL stress relaxed at 1/tau.
#
# It survived because every test that could have caught it built states with all off-diagonals
# zero — the one subspace where the two models agree. Demo 1 below is the discriminating
# measurement, and it is deliberately a RATE, because every invariant (mass, momentum, energy,
# the Pr=1 limit) was satisfied by the broken version.
#
# GRANULAR. `dT/dt = -zeta*T` integrates exactly over a step as a rescaling of the peculiar
# velocity, s = exp(-zeta*dt/2), with every central moment of order n scaled by s^n. In recon
# variables that touches ONLY the three variances, because standardized moments
# S_ijk = C_ijk/(sx^i sy^j sz^k) are invariant under an isotropic rescaling. Hence it is three
# multiplications, allocation-free, and realizability-preserving for s <= 1 without a check —
# realizability constrains the standardized moments, which this does not touch.
#
# Run:  julia --project=. examples/granular_and_esbgk.jl
using Riemann35, Printf, LinearAlgebra

# =========================================================================================
# 1. ES-BGK: off-diagonal stress must relax at the same rate as diagonal stress
# =========================================================================================
# THE INVARIANT IS CONVENTION-FREE. Two conventions exist for tau — the collision time (stress
# relaxes at (1-nu)/tau = (1/Pr)/tau) and the viscous time tau_mu = mu/p (stress relaxes at
# 1/tau, Pr carried by the heat flux). Under EITHER, every deviatoric component relaxes at the
# SAME rate, because the deviatoric part of Lambda is k*C_dev with a single scalar k. So the
# check below cannot be argued away as a modelling choice.
println("="^88)
println("1. ES-BGK — does OFF-DIAGONAL stress relax like DIAGONAL stress?")
println("="^88)

nv, vmax = 24, 6.0
dv = 2vmax/nv; vh = [-vmax + dv*(i-0.5) for i in 1:nv]
C = [1.4 0.25 0.0; 0.25 0.8 0.0; 0.0 0.0 0.8]     # anisotropic AND correlated — both needed
Ci = inv(C); dC = det(C)
f0 = Array{Float64}(undef, nv, nv, nv, 1)
for c in 1:nv, b in 1:nv, a in 1:nv
    v = (vh[a], vh[b], vh[c]); q = 0.0
    for i in 1:3, j in 1:3; q += v[i]*Ci[i,j]*v[j]; end
    f0[a,b,c,1] = exp(-q/2)/sqrt((2pi)^3*dC)
end
f0 ./= (sum(f0)*dv^3)

function stresses(F)
    r = sum(F)*dv^3
    xy = sum(F[a,b,c,1]*vh[a]*vh[b] for a in 1:nv, b in 1:nv, c in 1:nv)*dv^3/r
    xx = sum(F[a,b,c,1]*vh[a]^2   for a in 1:nv, b in 1:nv, c in 1:nv)*dv^3/r
    yy = sum(F[a,b,c,1]*vh[b]^2   for a in 1:nv, b in 1:nv, c in 1:nv)*dv^3/r
    zz = sum(F[a,b,c,1]*vh[c]^2   for a in 1:nv, b in 1:nv, c in 1:nv)*dv^3/r
    (r, xy, xx - (xx+yy+zz)/3)
end
function rate(ts, ys)
    x = ts; y = log.(abs.(ys)); n = length(x); sx = sum(x); sy = sum(y)
    b = (n*sum(x.*y) - sx*sy)/(n*sum(x.^2) - sx^2)
    a = (sy - b*sx)/n
    ss = sum((y .- (a .+ b.*x)).^2); st = sum((y .- sy/n).^2)
    (-b, 1 - ss/max(st, 1e-300))
end

tau = 0.1; dt = 0.14*tau; nt = 30
@printf("\n  %-8s %-8s %-16s %-16s %-8s\n", "Pr", "", "rate(sigma_xy)*tau", "rate(sigma_xx)*tau", "ratio")
for Pr in (1.0, 2/3)
    F = copy(f0); ts = Float64[]; sxy = Float64[]; sxx = Float64[]
    for n in 0:nt
        _, xy, xx = stresses(F)
        push!(ts, n*dt); push!(sxy, abs(xy)); push!(sxx, abs(xx))
        n < nt && collide_es_cpu!(F, vh, dv, dt, 2tau, Pr, 1.0)
    end
    rxy, r2xy = rate(ts, sxy); rxx, r2xx = rate(ts, sxx)
    @printf("  %-8.4f %-8s %-16.6f %-16.6f %-8.5f  (R2 %.5f/%.5f)\n",
            Pr, "", rxy*tau, rxx*tau, rxy/rxx, r2xy, r2xx)
end
println("""
  Both columns must read 1.000000 at BOTH Pr, and the ratio must be 1.
  Before #73 the Pr=2/3 row read 0.666667 / 1.000000 — a split of exactly Pr — while mass,
  momentum, energy and the Pr=1 limit were all still exact. Only a RATE discriminates.""")

# =========================================================================================
# 2. Granular: Haff's law, T(t) = T0/(1 + t/t0)^2
# =========================================================================================
# The exponent is the gate. zeta ~ sqrt(T) for hard spheres, which makes the decay ALGEBRAIC
# rather than exponential; a fitted p near -1 means that T-dependence has been lost. Mass and
# momentum must not move at all — only energy may leave an inelastic collision.
println("\n" * "="^88)
println("2. GRANULAR — homogeneous cooling against Haff's law")
println("="^88)

function rhouT(M)
    rho = M[1]; u = M[2]/rho; v = M[6]/rho; w = M[16]/rho
    T = (M[3]/rho - u*u + M[10]/rho - v*v + M[20]/rho - w*w)/3
    (rho, T)
end

tau_ref = 0.1; dtg = 2e-3; tmax = 20.0; nstep = round(Int, tmax/dtg)
@printf("\n  %-6s %-13s %-13s %-11s %-10s %-11s %-11s\n",
        "e", "T_end", "T_Haff", "rel err", "fitted p", "d(mass)", "d(mom)")
for e in (1.0, 0.99, 0.95, 0.90, 0.80)
    M = collect(Float64, InitializeM4_35(1.0, 0.0,0.0,0.0, 1.0,0,0, 1.0,0, 1.0))
    m0 = M[1]; p0 = M[2]
    ts = Float64[]; Ts = Float64[]
    for n in 1:nstep
        _, T = rhouT(M)
        T <= 0 && break
        tau_c = tau_ref*sqrt(1.0/T)
        M = collect(Riemann35.ReconDev.bgk_relax_tup(NTuple{35,Float64}(M), dtg, 2*tau_c, 1.0, 0.5, false))
        if e < 1.0
            zeta = (1 - e^2)/(3*tau_c)
            M = collect(granular_drain_tup(NTuple{35,Float64}(M), exp(-zeta*dtg/2)))
        end
        if n % (nstep ÷ 40) == 0
            _, Tn = rhouT(M); push!(ts, n*dtg); push!(Ts, Tn)
        end
    end
    _, Tend = rhouT(M)
    dmass = abs(M[1] - m0); dmom = abs(M[2] - p0)
    if e == 1.0
        @printf("  %-6.2f %-13.6f %-13.6f %-11.3e %-10s %-11.2e %-11.2e\n",
                e, Tend, 1.0, abs(Tend-1.0), "n/a", dmass, dmom)
    else
        zeta0 = (1 - e^2)/(3*tau_ref); t0 = 2/zeta0
        Th = 1.0/(1 + ts[end]/t0)^2
        h = length(ts)÷2
        x = [log(1 + t/t0) for t in ts[h:end]]; y = [log(T) for T in Ts[h:end]]
        n = length(x); sx = sum(x); sy = sum(y)
        p = (n*sum(x.*y) - sx*sy)/(n*sum(x.^2) - sx^2)
        @printf("  %-6.2f %-13.6f %-13.6f %-11.3e %-10.4f %-11.2e %-11.2e\n",
                e, Tend, Th, abs(Tend-Th)/Th, p, dmass, dmom)
    end
end
println("""
  Fitted p must approach -2 (the Haff signature); e=1 must hold T EXACTLY constant; mass and
  momentum must not move for any e.

  On the GPU the same physics is a kwarg: march3d_order3_gpu!(...; restitution = 0.9). NOTE
  the drain is operator-split ONCE PER STEP, not per RK stage — placing it in the stage loop
  applies it three times per step and gives a fitted exponent of -2.88 with temperature 44-68%
  low, while mass, momentum and the elastic limit all stay exact. That bug is invisible to
  every invariant-style check.""")

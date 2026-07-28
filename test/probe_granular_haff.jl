#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# GRANULAR ENTRY POINT: inelastic collisions and Haff's cooling law.
#
# R.O. Fox's stated target list includes granular and gas-particle flows. The single
# structural change that opens that regime is that collisions stop conserving energy:
# a binary collision with normal restitution coefficient e < 1 returns a fraction of
# the relative normal kinetic energy to the environment.
#
# WHAT THIS IMPLEMENTS. The inelastic BGK/ES-BGK operator relaxes toward a Maxwellian
# whose temperature is DRAINED at the collisional rate. For a homogeneous granular gas
# of smooth spheres the standard cooling rate is
#
#     dT/dt = -zeta * T,        zeta = (1 - e^2)/(3 tau_c)      [smooth-sphere IHS]
#
# with tau_c the collision time. Relaxing toward a Maxwellian at the instantaneous T
# while draining T at that rate is the granular-BGK model of Brey-Moreno-Dufty; it is
# the minimal modification of what the solver already has, and it leaves the collision
# operator a convex map toward a realizable target, so realizability is preserved by
# exactly the argument already used for ES-BGK and the wall blend.
#
# THE TEST: HAFF'S LAW. In the homogeneous cooling state (HCS) -- no gradients, no
# walls, no forcing -- the temperature of a granular gas decays as
#
#     T(t) = T0 / (1 + t/t0)^2,      t0 = 2/(zeta_0)   with zeta_0 = zeta(T0),
#
# because zeta ~ sqrt(T) for hard spheres, making the decay algebraic rather than
# exponential. That t^-2 tail is the classic signature and is EXACT for the model,
# so it is a real pass/fail gate and not a plausibility check.
#
# TWO CONTROLS, because a cooling curve alone is easy to fake:
#   * e = 1 must give EXACTLY constant temperature (elastic limit, no drain).
#   * momentum and mass must be conserved to roundoff for every e; only energy leaves.
#
# CAVEAT ON SCOPE. This is the homogeneous problem: it exercises the collision operator
# only, with the transport terms idle. It says nothing yet about whether the 35-moment
# closure handles granular CLUSTERING, which is the interesting and hard part (the HCS
# is linearly unstable to shear and clustering modes at large enough system size).
# That needs the spatial solver and is the natural follow-on.
#
# Env: GR_E (restitution list), GR_TAU, GR_TMAX, GR_NT
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, Printf, LinearAlgebra, Statistics

const ES   = [parse(Float64, s) for s in split(envparam("GR_E", "1.0,0.99,0.95,0.9,0.8"), ",")]
const TAU0 = parse(Float64, envparam("GR_TAU", "0.05"))   # collision time at T=1
const TMAX = parse(Float64, envparam("GR_TMAX", "20.0"))
const NT   = parse(Int,     envparam("GR_NT", "20000"))

"temperature and bulk velocity of a 35-moment state"
function rhouT(M)
    rho = M[1]
    u = M[2]/rho; v = M[6]/rho; w = M[16]/rho
    T = (M[3]/rho - u*u + M[10]/rho - v*v + M[20]/rho - w*w)/3
    rho, (u,v,w), T
end

"""
    granular_step(M, dt, e, tau_ref) -> M

One homogeneous inelastic-BGK step. Two pieces, split:
  1. elastic BGK relaxation toward the Maxwellian at the current (rho,u,T);
  2. energy drain at rate zeta = (1-e^2)/(3 tau_c), applied as an exact rescaling of
     the CENTRAL velocity fluctuations, which keeps rho and momentum untouched by
     construction rather than by cancellation.

tau_c ~ tau_ref*sqrt(T0/T): hard-sphere collision frequency grows as sqrt(T), which is
what makes Haff's law algebraic instead of exponential.
"""
function granular_step(M::Vector{Float64}, dt::Float64, e::Float64, tau_ref::Float64)
    rho, u, T = rhouT(M)
    T <= 0 && return M
    tau_c = tau_ref*sqrt(1.0/T)               # collision time at this temperature
    # --- 1. elastic relaxation toward the Maxwellian at (rho,u,T) ---
    Mtup = NTuple{35,Float64}(M)
    Mrel = collect(Riemann35.ReconDev.bgk_relax_tup(Mtup, dt, 2*tau_c, 1.0, 0.5, false))
    # --- 2. inelastic energy drain, exact over the step ---
    # dT/dt = -zeta*T with zeta = (1-e^2)/(3 tau_c) held fixed across the substep:
    #   T -> T*exp(-zeta*dt), i.e. central velocities scale by s = exp(-zeta*dt/2).
    zeta = (1 - e^2)/(3*tau_c)
    s    = exp(-zeta*dt/2)
    s == 1.0 && return Mrel
    rho2, u2, _ = rhouT(Mrel)
    C, _ = M2CS4_35(Mrel)
    cg(i,j,k) = C[S_INDEX[(i,j,k)]]*s^(i+j+k)   # central moment of order n scales as s^n
    A = C4toM4_3D(rho2, u2[1], u2[2], u2[3],
        cg(2,0,0), cg(1,1,0), cg(1,0,1), cg(0,2,0), cg(0,1,1), cg(0,0,2),
        cg(3,0,0), cg(2,1,0), cg(2,0,1), cg(1,2,0), cg(1,1,1), cg(1,0,2),
        cg(0,3,0), cg(0,2,1), cg(0,1,2), cg(0,0,3),
        cg(4,0,0), cg(3,1,0), cg(3,0,1), cg(2,2,0), cg(2,1,1), cg(2,0,2),
        cg(1,3,0), cg(1,2,1), cg(1,1,2), cg(1,0,3),
        cg(0,4,0), cg(0,3,1), cg(0,2,2), cg(0,1,3), cg(0,0,4))
    idx = [1,2,3,4,5,6,7,8,9,11,12,13,16,17,21,26,27,28,29,51,52,53,
           76,77,101,31,32,33,36,37,41,56,57,81,61]
    [A[i] for i in idx]
end

print_run_header("GRANULAR HOMOGENEOUS COOLING STATE vs HAFF'S LAW";
                 extra = ("collision" => "inelastic BGK (Brey-Moreno-Dufty form)",))
println("Haff: T(t) = T0/(1 + t/t0)^2 with t0 = 2/zeta_0, zeta_0 = (1-e^2)/(3 tau_ref).")
println("Exponent p is fitted from the late-time slope of log T vs log(1 + t/t0); Haff says p = -2.")
println("="^100)
@printf("%6s %12s %12s %12s %12s %10s %10s\n",
        "e", "T_end", "T_Haff", "rel err", "fitted p", "d(mass)", "d(mom)")

function run()
    dt = TMAX/NT
    for e in ES
        M = collect(Float64, InitializeM4_35(1.0, 0.2, -0.1, 0.05, 1.0,0,0, 1.0,0, 1.0))
        rho0, mom0, T0 = rhouT(M)
        p0 = (M[2], M[6], M[16])
        zeta0 = (1 - e^2)/(3*TAU0)
        t0 = zeta0 > 0 ? 2/zeta0 : Inf
        ts = Float64[]; Ts = Float64[]
        t = 0.0
        for n in 1:NT
            M = granular_step(M, dt, e, TAU0)
            t += dt
            if n % max(1, NT÷200) == 0
                _,_,Tn = rhouT(M); push!(ts, t); push!(Ts, Tn)
            end
        end
        rho1, _, T1 = rhouT(M)
        Thaff = isfinite(t0) ? T0/(1 + TMAX/t0)^2 : T0
        relerr = abs(T1 - Thaff)/Thaff
        # late-time log-log slope against (1 + t/t0)
        p = NaN
        if isfinite(t0)
            m = length(ts); lo = max(1, m÷2)
            X = [log(1 + ts[i]/t0) for i in lo:m]
            Y = [log(Ts[i]/T0)     for i in lo:m]
            n = length(X); sx = sum(X); sy = sum(Y)
            p = (n*sum(X.*Y) - sx*sy)/(n*sum(X.^2) - sx^2)
        end
        dmass = abs(rho1 - rho0)
        dmom  = maximum(abs, (M[2]-p0[1], M[6]-p0[2], M[16]-p0[3]))
        @printf("%6.2f %12.6f %12.6f %12.3e %12.4f %10.2e %10.2e\n",
                e, T1, Thaff, relerr, p, dmass, dmom)
        flush(stdout)
    end
end
run()
println("="^100)
println("GATES: e=1.0 must hold T EXACTLY constant (rel err ~ 0, no drain); mass and momentum")
println("drift must be at roundoff for every e, since only energy is allowed to leave; and the")
println("fitted exponent must approach -2, the Haff signature. A p near -1 would mean the")
println("collision frequency is not tracking sqrt(T) and the cooling is exponential-like.")

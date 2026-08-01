#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# THE TEMPERATURE JUMP — the thermal analogue of velocity slip, measured the same way.
#
# WHY. sec:zeta-curve found the closure's slip coefficient nearly CONSTANT (~1.6 across a
# 30x range in Kn) where the true one FALLS steeply (2.090 -> 0.790 from Kn = 0.05 to 0.80):
# the closure reproduces the CONTINUUM slip law, not the kinetic one, and is wrong by a
# factor of two by Kn = 0.8.
#
# Is that a statement about WALLS, or about MOMENTUM? Fourier flow separates them. Two
# stationary plates at different temperatures drive a heat flux, and the gas temperature at
# the wall does not equal the wall temperature -- the temperature jump, with its own
# accommodation coefficient. It is the same wall but a DIFFERENT transport channel:
#
#     slip           tau_xy = <cx cy>        SECOND-order, shear stress
#     temp. jump     q_y    = <cy |c|^2>/2   THIRD-order, heat flux
#
# If the closure flattens here too, the defect is general to wall transport and lives in the
# half-space representation. If it does not, the defect is confined to the momentum channel,
# which points somewhere much narrower. Neither is currently known.
#
# THE CPU PATH IS USED DELIBERATELY. WALL_SPEC carries per-face (ylo, yhi) settings, so two
# different wall temperatures need no code change; the GPU entry point currently takes a
# single scalar wall_Tw (wall_Tw_prof varies along the TANGENT, not between faces).
#
# LINEAR RESPONSE. dT/T is kept small (default +/-5%). A large temperature ratio mixes in
# nonlinear Fourier behaviour and makes the coefficient amplitude-dependent -- the trap the
# creep ladder exposed, where a two-point check could not distinguish linear from monotone.
#
# THE ESTIMATOR MATCHES dvm_fourier_wall.jl EXACTLY: linear fit over the middle half,
# extrapolated to the wall, zeta_T = (T_wall - T_gas(wall))/(lambda |dT/dy|). A different
# fit window on the two sides would make the comparison meaningless even with the physics
# matched -- the lesson of the lambda_HS/lambda_mu trap, applied to the estimator.
#
# Env: FJ_KNH, FJ_CPL, FJ_NYMIN, FJ_DTR, FJ_ORDER, FJ_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC, envparam, print_run_header
# `using MPI` is EXPLICIT here. Every test file is included into the same Main, so a
# file that omits it still works as long as some earlier include did `using MPI` --
# an order-dependent coupling that breaks the moment the file is run on its own, or
# the include order changes. See issue #62.
using MPI
MPI.Initialized() || MPI.Init()

const KNH   = [parse(Float64,s) for s in split(envparam("FJ_KNH", "0.05,0.10,0.20,0.40,0.80"), ",")]
const CPL   = parse(Float64, envparam("FJ_CPL",   "6.0"))
const NYMIN = parse(Int,     envparam("FJ_NYMIN", "32"))
const DTR   = parse(Float64, envparam("FJ_DTR",   "0.05"))
const ORDER = parse(Int,     envparam("FJ_ORDER", "2"))
const TENDF = parse(Float64, envparam("FJ_TEND",  "3.0"))
const H = 1.0; const T0 = 1.0

"Cell temperature from the 35-moment vector: T = (C200+C020+C002)/3."
function cellT(M)
    r = M[1]; ux = M[2]/r; uy = M[6]/r; uz = M[16]/r
    ((M[3]/r - ux^2) + (M[10]/r - uy^2) + (M[20]/r - uz^2))/3
end

function fourier(KnH::Float64)
    halo = ORDER == 3 ? 8 : 2
    ny = max(NYMIN, round(Int, CPL/KnH))
    nx = halo + 2; nz = 1
    dy = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    lam = KnH*H; tau_ref = lam*sqrt(2.0); kn_tau = 2*tau_ref; nu = T0*tau_ref
    Tc = T0*(1-DTR); Th = T0*(1+DTR)

    WALL_SPEC[] = (ylo = (Tw=Tc, uw1=0.0, uw2=0.0, alpha=1.0),
                   yhi = (Tw=Th, uw1=0.0, uw2=0.0, alpha=1.0))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    dt = 0.2*dy/(5.0*sqrt(T0))
    nst = ceil(Int, (TENDF*H*H/nu)/dt)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j-halo)-0.5)*dy, 0.0, H)
        Tj = Tc + (Th-Tc)*yj/H
        M[i,j,k,:] = InitializeM4_35(1.0, 0.0,0.0,0.0, Tj,0,0, Tj,0, Tj)
    end
    m0 = sum(M[halo+1, halo+j, 1, 1] for j in 1:ny)
    for _ in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order=ORDER, stage_bgk_kn=kn_tau, stage_bgk_exact=true,
                           Pr=1.0, omega=1.0)
    end
    T = [cellT(@view M[halo+1, halo+j, 1, :]) for j in 1:ny]
    y = [(j-0.5)*dy for j in 1:ny]
    lo = max(1,ny÷4); hi = min(ny,3*ny÷4)
    ys, Ts = y[lo:hi], T[lo:hi]
    yb, Tb = mean(ys), mean(Ts)
    S = sum((ys.-yb).*(Ts.-Tb))/sum((ys.-yb).^2)
    b = Tb - S*yb
    r2 = 1 - sum((Ts .- (S.*ys .+ b)).^2)/max(sum((Ts.-Tb).^2), eps())
    jump = Th - (S*H + b)
    m1 = sum(M[halo+1, halo+j, 1, 1] for j in 1:ny)
    (ny=ny, nst=nst, S=S, r2=r2, jump=jump, zetaT=jump/(lam*abs(S)), dmass=(m1-m0)/m0)
end

print_run_header("TEMPERATURE JUMP (Fourier flow) -- the thermal analogue of slip";
                 extra = ("collision" => "BGK, Pr=1, omega=1 (matched to the DVM reference)",
                          "walls" => "stationary, fully diffuse, T_cold/T_hot"))
println("zeta_T = (T_wall - T_gas(wall)) / (lambda |dT/dy|), core fit over the middle half,")
println("identical to dvm_fourier_wall.jl. Compare the Kn-dependence against MOMENTUM slip:")
println("the closure's zeta is flat where truth falls steeply -- does zeta_T do the same?")
println("="^92)
@printf("%8s %6s %8s %13s %11s %12s %12s %11s\n",
        "Kn_H","ny","steps","dT/dy","R^2","jump","zeta_T","dmass/m")
flush(stdout)
for Kn in KNH
    r = fourier(Kn)
    @printf("%8.3f %6d %8d %13.6f %11.7f %12.6f %12.6f %11.2e\n",
            Kn, r.ny, r.nst, r.S, r.r2, r.jump, r.zetaT, r.dmass)
    flush(stdout)
end
println("="^92)

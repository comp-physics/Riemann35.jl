#!/usr/bin/env julia
# validate_spatial_transport.jl -- do mu and k actually EMERGE from the discretized PDE?
#
# test/validate_transport_coefficients.jl measures the relaxation times of the COLLISION
# OPERATOR (homogeneous, no space). That isolates the operator but does not prove the
# transport coefficients emerge from the spatial scheme. This does.
#
# METHOD. Periodic domain, small-amplitude linear modes, driven through the PRODUCTION
# stepper `step_highorder_3d!` (not a hand-rolled march):
#
#   shear mode    u_y = eps*sin(kx), uniform rho/T   ->  decays as exp(-nu k^2 t),
#                 nu = mu/rho.  Pure shear: no coupling to acoustics.
#   entropy mode  uniform pressure, rho' = eps*sin(kx) with T' offsetting it, u = 0
#                 ->  decays as exp(-alpha k^2 t), alpha = k_th/(rho c_p).
#
#   Pr = nu/alpha  needs no absolute normalization.
#
# THE CONTAMINATION PROBLEM (the reason for the grid sweep). The scheme's own numerical
# viscosity competes with the physical one: Rusanov-type dissipation ~ alpha_wave*dx/2
# with alpha_wave ~ 5, versus nu = Theta*tau_ref = Kn/2 at omega=0.5. At Kn=1, L=20,
# Nx=200 those are ~0.25 vs ~0.5 -- a 50% contamination. Quoting a single-grid number
# here would be meaningless. So we
#   (a) run at spatial_order=3 (WENO5's low dissipation is the point), and
#   (b) sweep Nx and extrapolate dx->0, where the numerical part vanishes.
# If the extrapolation does not cleanly separate the two, the honest answer is that the
# measurement cannot be made at reachable resolution -- report that, do not quote a
# contaminated number.
#
# Env: ST_KN, ST_NXS (comma list), ST_L, ST_EPS, ST_ORDER, ST_PR, ST_OMEGA.
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35.ReconDev: bgk_relax_tup
MPI.Initialized() || MPI.Init()

const KN    = parse(Float64, get(ENV, "ST_KN", "1.0"))
const NXS   = parse.(Int, split(get(ENV, "ST_NXS", "40,60,80"), ","))
const LDOM  = parse(Float64, get(ENV, "ST_L", "10.0"))
const EPS   = parse(Float64, get(ENV, "ST_EPS", "1e-3"))
const ORDER = parse(Int, get(ENV, "ST_ORDER", "3"))
const PRN   = parse(Float64, get(ENV, "ST_PR", string(2/3)))
const OMG   = parse(Float64, get(ENV, "ST_OMEGA", "0.5"))
const STAGE = parse(Int, get(ENV, "ST_STAGE", "1"))   # 1 = stage_bgk (shipped default), 0 = once/step
const SBEX  = parse(Int, get(ENV, "ST_EXACT", "0")) == 1   # correct the SSP-RK3 composite

const PERIODIC_X = (xlo=:periodic, xhi=:periodic, ylo=:periodic, yhi=:periodic,
                    zlo=:outflow, zhi=:outflow)

"project a 1D interior field onto sin(kx) — the mode amplitude"
mode_amp(q, x, k) = 2 * mean(q .* sin.(k .* x))

"""
    run_mode(kind, Nx; ...) -> (rate, k, nu_theory)

March one linear mode and least-squares fit its exponential decay rate.
`kind` is `:shear` (transverse velocity) or `:entropy` (uniform-pressure density).
"""
function run_mode(kind::Symbol, Nx::Int; Kn = KN, L = LDOM, eps = EPS, order = ORDER,
                  Pr = PRN, omega = OMG, nper = parse(Float64, get(ENV,"ST_NPER","2.0")), verbose = false)
    halo = order == 3 ? 8 : 2
    ny = order == 3 ? 8 : 4          # transverse: periodic, needs >= halo interior cells
    nz = 1
    dx = L/Nx; dy = dx; dz = dx
    decomp = setup_mpi_cartesian_3d(Nx, ny, nz, halo, MPI.COMM_WORLD)

    rho0, T0 = 1.0, 1.0
    k = 2pi/L
    xc(ih) = (ih - halo - 0.5) * dx                       # interior-cell centre

    M = zeros(Nx + 2halo, ny + 2halo, nz, 35)
    for kk in 1:nz, jh in 1:(ny + 2halo), ih in 1:(Nx + 2halo)
        xx = xc(ih)
        if kind === :shear
            uy = eps * sin(k*xx)
            M[ih, jh, kk, :] = InitializeM4_35(rho0, 0.0, uy, 0.0, T0,0,0, T0,0, T0)
        else
            # uniform pressure p = rho*T: perturb rho, offset T
            r = rho0 * (1 + eps*sin(k*xx))
            T = rho0*T0/r
            M[ih, jh, kk, :] = InitializeM4_35(r, 0.0, 0.0, 0.0, T,0,0, T,0, T)
        end
    end

    tau_ref = (Kn/2) * T0^(omega-1.0) / rho0
    nu_th   = T0 * tau_ref                                # mu/rho = Theta*tau_ref
    alpha_th = nu_th / Pr                                 # k_th/(rho c_p)
    decay_th = (kind === :shear ? nu_th : alpha_th) * k^2
    Tend = nper / decay_th                                # a few e-foldings
    dt = 0.2 * dx / (5.0 * sqrt(T0))
    nst = max(20, ceil(Int, Tend/dt))

    xs = [xc(ih) for ih in (halo+1):(halo+Nx)]
    ts = Float64[]; amps = Float64[]
    function amp_now()
        q = if kind === :shear
            [M[ih, halo+1, 1, 6] / M[ih, halo+1, 1, 1] for ih in (halo+1):(halo+Nx)]  # u_y
        else
            [M[ih, halo+1, 1, 1] for ih in (halo+1):(halo+Nx)]                        # rho
        end
        abs(mode_amp(q, xs, k))
    end
    a0 = amp_now()
    for s in 1:nst
        if STAGE == 1
            step_highorder_3d!(M, dt, decomp, PERIODIC_X, Nx, ny, nz, halo, dx, dy, dz, 0.0;
                               order = order, stage_bgk_kn = Kn, Pr = Pr, omega = omega,
                               stage_bgk_exact = SBEX)
        else
            # collisionless stages, then ONE exact-exponential collision per step
            step_highorder_3d!(M, dt, decomp, PERIODIC_X, Nx, ny, nz, halo, dx, dy, dz, 0.0;
                               order = order, stage_bgk_kn = nothing)
            @inbounds for kk in 1:nz, jh in (halo+1):(halo+ny), ih in (halo+1):(halo+Nx)
                mt = ntuple(q -> M[ih, jh, kk, q], Val(35))
                out = bgk_relax_tup(mt, dt, Kn, Pr, omega)
                for q in 1:35; M[ih, jh, kk, q] = out[q]; end
            end
        end
        if s % max(1, nst ÷ 12) == 0
            a = amp_now()
            if isfinite(a) && a > 1e-14*max(1.0, a0)
                push!(ts, s*dt); push!(amps, a)
            end
        end
    end
    length(amps) < 4 && return (NaN, k, decay_th, dx)
    # least-squares slope of log(amp) vs t
    la = log.(amps); n = length(ts)
    slope = (n*sum(ts .* la) - sum(ts)*sum(la)) / (n*sum(ts.^2) - sum(ts)^2)
    (-slope, k, decay_th, dx)
end

"Richardson-style fit rate(dx) = rate_0 + c*dx, returning rate_0"
function extrapolate(dxs, rates)
    n = length(dxs)
    sx = sum(dxs); sy = sum(rates); sxx = sum(dxs.^2); sxy = sum(dxs .* rates)
    slope = (n*sxy - sx*sy)/(n*sxx - sx^2)
    (sy - slope*sx)/n, slope
end

println("="^94)
println("SPATIAL TRANSPORT: do mu and k emerge from the discretized PDE?")
@printf("Kn=%.3f  Pr=%.4f  omega=%.2f  order=%d  L=%.1f  eps=%.1e\n", KN, PRN, OMG, ORDER, LDOM, EPS)
@printf("k*lambda = %.3f  (hydrodynamic limit needs << 1; larger biases nu by Burnett-order terms)\n", (2pi/LDOM)*(KN/2))
println("measured decay rate vs theory rate = nu*k^2 (shear) / alpha*k^2 (entropy);")
println("numerical viscosity is O(dx), so the dx->0 extrapolation is the physical value")
println("="^94)

results = Dict{Symbol,Any}()
for kind in (:shear, :entropy)
    @printf("\n--- %s mode ---\n", kind)
    @printf("%-8s %-10s %14s %14s %10s\n", "Nx", "dx", "rate_measured", "rate_theory", "ratio")
    dxs = Float64[]; rates = Float64[]; th = NaN
    for Nx in NXS
        r, k, dth, dx = run_mode(kind, Nx)
        th = dth
        @printf("%-8d %-10.4f %14.6e %14.6e %10.4f\n", Nx, dx, r, dth, r/dth)
        if isfinite(r); push!(dxs, dx); push!(rates, r); end
    end
    if length(dxs) >= 2
        r0, slope = extrapolate(dxs, rates)
        @printf("%-8s %-10s %14.6e %14.6e %10.4f   <- dx->0 extrapolated (slope %.3e)\n",
                "dx->0", "0", r0, th, r0/th, slope)
        results[kind] = (r0 = r0, th = th, slope = slope, rates = copy(rates), dxs = copy(dxs))
    else
        results[kind] = nothing
    end
end

println("\n", "="^94)
if haskey(results, :shear) && results[:shear] !== nothing &&
   haskey(results, :entropy) && results[:entropy] !== nothing
    s = results[:shear]; e = results[:entropy]
    pr_meas = e.r0 / s.r0 == 0 ? NaN : (s.r0 / e.r0)   # nu/alpha = Pr
    @printf("mu/rho  : measured(dx->0) %.6e  theory %.6e  ratio %.4f\n", s.r0/ (2pi/LDOM)^2, s.th/(2pi/LDOM)^2, s.r0/s.th)
    @printf("alpha   : measured(dx->0) %.6e  theory %.6e  ratio %.4f\n", e.r0/ (2pi/LDOM)^2, e.th/(2pi/LDOM)^2, e.r0/e.th)
    @printf("Pr      : measured %.6f   requested %.6f   rel err %.3e\n",
            pr_meas, PRN, abs(pr_meas - PRN)/PRN)
    # contamination diagnostic: how much did the extrapolation move the finest-grid value?
    @printf("\ncontamination check (finest grid vs dx->0):\n")
    @printf("  shear  : finest %.6e -> extrap %.6e  (%.1f%% of the finest value was numerical)\n",
            s.rates[end], s.r0, 100*(s.rates[end]-s.r0)/s.rates[end])
    @printf("  entropy: finest %.6e -> extrap %.6e  (%.1f%% of the finest value was numerical)\n",
            e.rates[end], e.r0, 100*(e.rates[end]-e.r0)/e.rates[end])
    println("\nIf the contamination fraction is large, the extrapolation is doing the work and")
    println("the result is only as good as its linearity -- judge it on the per-grid table above.")
else
    println("MEASUREMENT FAILED: not enough finite rates to extrapolate. Report as")
    println("not-measurable at this resolution rather than quoting a contaminated number.")
end
println("="^94)

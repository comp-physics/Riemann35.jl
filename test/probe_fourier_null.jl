#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# NULL CONTROL for the parity analysis behind the 26-moment reduction probes.
#
# THIS SCRIPT IS DESIGNED TO FAIL IF I AM WRONG. The claim it tests is mine, not
# Fox's, and it is the load-bearing step in the argument that wall-bounded shear
# discriminates where a shock does not:
#
#   The nine moments the reduction drops are exactly those ODD IN TWO AXES
#   (permutations of (3,1,0) and (2,1,1)). A flow therefore exposes them only if
#   it breaks the corresponding reflection symmetries.
#
# Planar FOURIER flow -- two walls at different temperatures, no wall motion, no
# shear -- is invariant under BOTH x -> -x AND z -> -z. Every moment odd in x or
# odd in z must vanish identically, and all nine dropped moments are odd in at
# least one of those two. So the prediction is sharp and unforgiving:
#
#   ALL NINE ARE ZERO TO ROUNDOFF, and the reduction is EXACTLY inert here,
#
# even though the flow is strongly non-equilibrium (a real heat flux, a real
# Knudsen layer, temperature jump at the wall). If any of the nine comes back
# nonzero at a level above roundoff, the parity argument is wrong and every
# conclusion drawn from it -- including "the Ma=2 shock could not have seen these
# moments" -- has to be withdrawn.
#
# The heat flux S300-along-y (i.e. S030 here) is printed as a POSITIVE control: it
# must be clearly nonzero, otherwise the run is just an equilibrium gas and the
# null result would be vacuous.
#
# Env: FN_KNH, FN_CPL, FN_DT_RATIO (Tw_hi/Tw_lo), FN_ORDER, FN_TEND
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC
using Riemann35: reduce26_S, reduce26_residual, S_INDEX, DROPPED_KEYS
MPI.Initialized() || MPI.Init()

const KNH   = [parse(Float64, s) for s in split(get(ENV, "FN_KNH", "1.0,0.5,0.2,0.1"), ",")]
const CPL   = parse(Float64, get(ENV, "FN_CPL", "6.0"))
const NYMIN = parse(Int,     get(ENV, "FN_NYMIN", "24"))
const TRAT  = parse(Float64, get(ENV, "FN_DT_RATIO", "2.0"))   # Tw_hi/Tw_lo
const ORDER = parse(Int,     get(ENV, "FN_ORDER", "2"))
const TENDF = parse(Float64, get(ENV, "FN_TEND", "0.3"))

stdz(Mv) = (M2CS4_35(collect(Float64, Mv)))[2]

"planar Fourier flow: stationary walls at T_lo and T_hi, no shear"
function fourier_field(KnH::Float64; cpl = CPL, order = ORDER, tendf = TENDF,
                       Tlo = 1.0, Thi = TRAT, Pr = 2/3, omega = 0.81, alpha = 1.0)
    halo = order == 3 ? 8 : 2
    H    = 1.0
    lam  = KnH*H
    ny   = max(NYMIN, round(Int, cpl/KnH))
    nx   = halo + 2; nz = 1
    dy   = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    rho0 = 1.0

    # NO wall motion: uw1 = uw2 = 0 on both walls. That is what makes this a null test.
    WALL_SPEC[] = (ylo = (Tw=Tlo, uw1=0.0, uw2=0.0, alpha=alpha),
                   yhi = (Tw=Thi, uw1=0.0, uw2=0.0, alpha=alpha))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)

    # lambda is set by the COLD wall state so the sweep variable stays well defined
    tau_ref = lam*sqrt(2.0)
    kn_tau  = 2*tau_ref
    nu      = Tlo*tau_ref

    # linear temperature guess at constant pressure
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = ((j - halo) - 0.5)*dy
        f  = clamp(yj/H, 0.0, 1.0)
        T  = Tlo + (Thi - Tlo)*f
        rho = rho0*Tlo/T                     # constant pressure
        M[i,j,k,:] = InitializeM4_35(rho, 0.0, 0.0, 0.0, T,0,0, T,0, T)
    end

    dt   = 0.2*dy/(5.0*sqrt(max(Tlo,Thi)))
    nst  = ceil(Int, (tendf*H*H/nu)/dt)
    for _ in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                           order = order, stage_bgk_kn = kn_tau, stage_bgk_exact = true,
                           Pr = Pr, omega = omega)
    end
    [M[halo+1, halo+j, 1, :] for j in 1:ny], ny, nst
end

println("="^104)
println("FOURIER NULL CONTROL: all nine dropped moments must vanish by x- and z-parity")
@printf("stationary walls at T=%.2f and %.2f, ES-BGK Pr=2/3 omega=0.81, ~%.0f cells/lambda\n",
        1.0, TRAT, CPL)
println("PREDICTION: max|S_dropped| ~ roundoff, closure residual ~ 0, reduction EXACTLY inert.")
println("POSITIVE CONTROL: S030 (wall-normal heat flux) must be clearly nonzero, else the")
println("flow is at equilibrium and the null result would be vacuous.")
println("="^104)
@printf("%6s %6s %14s %14s %14s %12s %10s\n",
        "Kn_H", "ny", "max|S_dropped|", "S030 (control)", "dT/T (control)", "residual", "verdict")
flush(stdout)

function run_sweep()
allpass = true
for KnH in KNH
    col, ny, nst = fourier_field(KnH)
    mx = 0.0; mxr = 0.0; q = 0.0
    Tlo_m = Inf; Thi_m = -Inf
    for j in 1:ny
        S = stdz(col[j])
        for k in DROPPED_KEYS
            mx = max(mx, abs(S[S_INDEX[k]]))
        end
        r, _ = reduce26_residual(S)
        mxr = max(mxr, r)
        q = max(q, abs(S[S_INDEX[(0,3,0)]]))
        C = M4toC4_3D(collect(Float64, col[j])...)
        Tc = (C[3,1,1] + C[1,3,1] + C[1,1,3])/3
        Tlo_m = min(Tlo_m, Tc); Thi_m = max(Thi_m, Tc)
    end
    dT = (Thi_m - Tlo_m)/Thi_m
    ok = mx < 1e-10 && q > 1e-3 && dT > 1e-2
    allpass &= ok
    @printf("%6.3f %6d %14.3e %14.3e %14.3e %12.3e %10s\n",
            KnH, ny, mx, q, dT, mxr, ok ? "PASS" : "FAIL")
    flush(stdout)
end
allpass
end
const allpass = run_sweep()
println("="^104)
if allpass
    println("PASS: the nine vanish to roundoff while the heat flux and temperature drop are large.")
    println("The parity argument survives its falsification test. A normal shock -- which has the")
    println("SAME x- and z-symmetries -- likewise cannot expose these moments, so the Ma=2 shock")
    println("comparison is silent on the reduction, as claimed.")
else
    println("FAIL: a dropped moment is nonzero in a flow whose symmetry forbids it, or the")
    println("positive control is dead. Either the parity analysis is wrong or the setup is not")
    println("the flow it claims to be. Do NOT build on the shear results until this is resolved.")
end

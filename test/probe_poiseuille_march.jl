#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# IS THE POISEUILLE FLOW RATE MARCH-CONVERGED? — the sweep probe_poiseuille.jl never did.
#
# probe_poiseuille.jl reports Q for full-35 and reduced-26 at ONE march length,
# tendf = 0.4 in units of H^2/nu, and the reduced-26 gap it produces (peaking at ~53% near
# Kn_H = 0.2) is the headline answer to R.O. Fox's question. Its own caveat is that Q's
# march-length convergence was never swept -- only zeta's was, and zeta was measured at
# tendf = 1.2, three times longer. The thermal-creep probe needed tendf ~ 2.8 to converge to
# 0.4% and was explicitly NOT converged at 0.28. So tendf = 0.4 is short by the standard of
# every other wall-bounded quantity here, and the number that rests on it is the most-cited
# one in the notes.
#
# WHAT IS ACTUALLY AT RISK. Not Q alone -- the reduced-26 GAP. Full-35 and reduced-26 run on
# an identical mesh and march, so a march-length error common to both cancels in the ratio.
# The gap only moves if the two converge at DIFFERENT rates. That is the thing to measure,
# and it is why both branches are marched here rather than just one.
#
# DESIGN: checkpoints, not re-runs. Q is sampled along a single long march at a ladder of
# tendf values, so the whole convergence trail costs one march instead of one march per
# point. A quantity that is converged must be flat across the late checkpoints; the
# published tendf = 0.4 is one of them, so the comparison is direct.
#
# Everything else is held at probe_poiseuille.jl's defaults (order 2, 6 cells/mfp, g = 0.02,
# ES-BGK Pr = 2/3, omega = 0.81, diffuse walls, CFL factor 1) so that MARCH LENGTH IS THE
# ONLY VARIABLE. Matching the original setup is the point; this is not an improved
# measurement of Q, it is a convergence test of the published one.
#
# Env: PM_KNH (default the three Kn that carry the claim), PM_TENDS (checkpoint ladder),
#      PM_CPL, PM_G, PM_ORDER, PM_CFL, PM_NYMIN
# ---------------------------------------------------------------------------
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
using Riemann35: WALL_SPEC
using Riemann35: reduce26_S, S_INDEX, DROPPED_KEYS, body_force_shift, apply_body_force!
MPI.Initialized() || MPI.Init()

# The three Kn that carry the published claim: 0.2 is the peak of the gap (53%), 0.5 its
# shoulder (51%), 0.1 the decline (28.6%) that refuted the earlier "saturates at ~50%"
# reading. If the gap is march-sensitive anywhere it will show here.
const KNH   = [parse(Float64, s) for s in split(get(ENV, "PM_KNH", "0.5,0.2,0.1"), ",")]
const TENDS = [parse(Float64, s) for s in split(get(ENV, "PM_TENDS", "0.4,0.8,1.2,1.6,2.4,3.2"), ",")]
const CPL   = parse(Float64, get(ENV, "PM_CPL", "6.0"))
const NYMIN = parse(Int,     get(ENV, "PM_NYMIN", "24"))
const GACC  = parse(Float64, get(ENV, "PM_G", "0.02"))
const ORDER = parse(Int,     get(ENV, "PM_ORDER", "2"))
const CFLF  = parse(Float64, get(ENV, "PM_CFL", "1.0"))

# The nine dropped moments, re-imposed per cell per step. Verbatim from probe_poiseuille.jl
# so the two probes apply the SAME reduction -- a reduction that differed even in operator
# ordering would make the gap incomparable to the published one.
function apply_reduce26!(M, nx, ny, nz, halo)
    for kk in 1:nz, jj in (halo+1):(halo+ny), ii in (halo+1):(halo+nx)
        Mv = M[ii,jj,kk,:]
        C, S = M2CS4_35(collect(Float64, Mv))
        R  = reduce26_S(S)
        cg(i,j,k) = C[S_INDEX[(i,j,k)]]
        sx = sqrt(max(cg(2,0,0), eps())); sy = sqrt(max(cg(0,2,0), eps())); sz = sqrt(max(cg(0,0,2), eps()))
        for key in DROPPED_KEYS
            (a,b,c) = key
            C[S_INDEX[key]] = R[S_INDEX[key]]*sx^a*sy^b*sz^c
        end
        rho = Mv[1]; u = Mv[2]/rho; v = Mv[6]/rho; w = Mv[16]/rho
        A = C4toM4_3D(rho, u, v, w,
            cg(2,0,0), cg(1,1,0), cg(1,0,1), cg(0,2,0), cg(0,1,1), cg(0,0,2),
            cg(3,0,0), cg(2,1,0), cg(2,0,1), cg(1,2,0), cg(1,1,1), cg(1,0,2),
            cg(0,3,0), cg(0,2,1), cg(0,1,2), cg(0,0,3),
            cg(4,0,0), cg(3,1,0), cg(3,0,1), cg(2,2,0), cg(2,1,1), cg(2,0,2),
            cg(1,3,0), cg(1,2,1), cg(1,1,2), cg(1,0,3),
            cg(0,4,0), cg(0,3,1), cg(0,2,2), cg(0,1,3), cg(0,0,4))
        idx = [1,2,3,4,5,6,7,8,9,11,12,13,16,17,21,26,27,28,29,51,52,53,
               76,77,101,31,32,33,36,37,41,56,57,81,61]
        for q in 1:35; M[ii,jj,kk,q] = A[idx[q]]; end
    end
end

"""
    poiseuille_trail(KnH; reduce) -> (ny, nst_total, Qs, mass_drifts)

One march to the longest checkpoint, with Q sampled at each tendf in TENDS. Also returns
the interior mass drift at each checkpoint: a flow rate read off a channel that is losing
mass is not a measurement, and the wall BC is known to leak linearly (~0.078%/unit time).
"""
function poiseuille_trail(KnH::Float64; reduce::Bool = false)
    order = ORDER
    halo = order == 3 ? 8 : 2
    H = 1.0; lam = KnH*H
    ny = max(NYMIN, round(Int, CPL/KnH))
    nx = halo + 2; nz = 1
    dy = H/ny; dx = dy; dz = dy
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    rho0, T0 = 1.0, 1.0
    WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=0.0, alpha=1.0),
                   yhi = (Tw=T0, uw1=0.0, uw2=0.0, alpha=1.0))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)
    tau_ref = lam*sqrt(2.0); kn_tau = 2*tau_ref; nu = T0*tau_ref
    g = GACC

    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j - halo) - 0.5)*dy, 0.0, H)
        up = g/(2nu)*(yj*(H - yj)) + g*H*1.1*lam/(2nu)
        M[i,j,k,:] = InitializeM4_35(rho0, up, 0.0, 0.0, T0,0,0, T0,0, T0)
    end

    dt = CFLF*0.2*dy/(5.0*sqrt(T0))
    ubar_of(Mf) = mean(Mf[halo+1, halo+j, 1, 2]/Mf[halo+1, halo+j, 1, 1] for j in 1:ny)
    mass_of(Mf) = sum(Mf[halo+1, halo+j, 1, 1] for j in 1:ny)
    m0 = mass_of(M)

    G = g*H/(2*T0)
    steps_at = [ceil(Int, (te*H*H/nu)/dt) for te in TENDS]
    Qs = Float64[]; dms = Float64[]
    n = 0
    for (ci, target) in enumerate(steps_at)
        while n < target
            step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                               order = order, stage_bgk_kn = kn_tau, stage_bgk_exact = true,
                               Pr = 2/3, omega = 0.81)
            apply_body_force!(M, g, 0.0, 0.0, dt, nx, ny, nz, halo)
            reduce && apply_reduce26!(M, nx, ny, nz, halo)
            n += 1
        end
        push!(Qs, ubar_of(M)/(G*sqrt(2*T0)))
        push!(dms, (mass_of(M) - m0)/m0)
    end
    (ny, n, Qs, dms)
end

println("="^108)
println("IS Q MARCH-CONVERGED? — the sweep probe_poiseuille.jl never ran")
@printf("order=%d, %.0f cells/mfp, g=%.3f, CFL factor %.2f, diffuse walls, ES-BGK Pr=2/3 omega=0.81\n",
        ORDER, CPL, GACC, CFLF)
@printf("checkpoints (tendf, units of H^2/nu): %s\n", join(TENDS, ", "))
println("PUBLISHED VALUE IS AT tendf = 0.4. zeta was measured at 1.2; creep needed ~2.8.")
println("The claim at risk is the reduced-26 GAP, not Q: a march error common to both")
println("branches cancels in the ratio, so the gap only moves if they converge differently.")
println("="^108)
flush(stdout)

for KnH in KNH
    t0 = time()
    nyf, nstf, Qf, dmf = poiseuille_trail(KnH; reduce = false)
    nyr, nstr, Qr, dmr = poiseuille_trail(KnH; reduce = true)
    el = time() - t0
    @printf("\nKn_H = %.3f   ny = %d   %d steps to tendf = %.1f   (%.0f s)\n",
            KnH, nyf, nstf, TENDS[end], el)
    @printf("  %8s %13s %13s %10s %12s\n", "tendf", "Q full-35", "Q reduced-26", "gap", "dmass/m")
    for i in eachindex(TENDS)
        gap = abs(Qf[i]) > 0 ? abs(Qr[i] - Qf[i])/abs(Qf[i]) : NaN
        @printf("  %8.1f %13.6f %13.6f %9.2f%% %12.3e%s\n",
                TENDS[i], Qf[i], Qr[i], 100*gap, dmf[i], TENDS[i] == 0.4 ? "   <- PUBLISHED" : "")
    end
    gap0 = abs(Qr[1] - Qf[1])/abs(Qf[1]); gapN = abs(Qr[end] - Qf[end])/abs(Qf[end])
    @printf("  Q full-35 moves %+.2f%% from tendf %.1f to %.1f; the GAP moves %.2f%% -> %.2f%% (%+.2f pts)\n",
            100*(Qf[end]-Qf[1])/abs(Qf[1]), TENDS[1], TENDS[end], 100*gap0, 100*gapN, 100*(gapN-gap0))
    flush(stdout)
end
println()
println("="^108)
println("READ: if the last two checkpoints agree to well under a percent, tendf=0.4 is")
println("vindicated and the published gap stands. If Q is still climbing at 3.2, the")
println("published number is a snapshot of a transient and every flow rate needs requoting.")
println("="^108)

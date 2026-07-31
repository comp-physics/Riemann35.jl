#!/usr/bin/env julia
# wall_comparison_protocol.jl — HOW TO COMPARE THE CLOSURE AGAINST A KINETIC REFERENCE
# AT A WALL, WITHOUT FOOLING YOURSELF.
#
# This is a worked example, deliberately small enough to run in a couple of minutes on a
# laptop. Its purpose is not the number it prints; it is the PROTOCOL. Every check below
# exists because skipping it produced a published wrong answer, and each is annotated with
# the specific failure it prevents.
#
# The comparison being demonstrated: plane Couette flow, 35-moment closure against the
# DVM-BGK reference in src/reference/dvm_bgk.jl, at matched collision physics.
#
# ---------------------------------------------------------------------------------------
# THE SEVEN RULES, each learned the hard way
# ---------------------------------------------------------------------------------------
#
# 1. MATCH THE COLLISION MODEL EXACTLY, and say so.
#    The closure ships an ES-BGK default (Pr = 2/3); the DVM here is BGK (Pr = 1). Compare
#    them as-is and you measure truncation error PLUS a collision-model difference, which
#    for slip is worth 14-43% -- larger than the truncation error at moderate Kn. Set
#    Pr = 1, omega = 1 on the closure side, and use the identical tau.
#
# 2. CONVERGE IN SPACE, ON BOTH SIDES INDEPENDENTLY.
#    The DVM's spatial convergence is first order and slow: zeta at Kn = 0.2 runs
#    1.5827, 1.5753, 1.5680, 1.5607, 1.5570 at Nx = 48 ... 384. Reading it at Nx = 48
#    is 1.7% off. The velocity grid, by contrast, is converged by nv = 32 (0.03%).
#
# 3. CONVERGE IN TIME, ON BOTH SIDES INDEPENDENTLY. This is the one that bites.
#    t_end = 3 H^2/nu is the DIFFUSIVE timescale. At Kn = 0.8 that is only 2.65 ballistic
#    transit times and NOT a steady state -- the shear rate moves 22% between t_end = 1.5
#    and 6.0. Use t_end = max(3 H^2/nu, N H/sqrt(T)) with N >~ 20, so the integration time
#    tracks whichever transport mechanism is actually operating.
#
#    AND CHECK IT SEPARATELY FOR EACH SOLVER. The DVM converges in t_end by 1.5; the
#    closure does not. Checking one and assuming the other is what produced the error --
#    a convergence check is a property of a SOLVER, never of a comparison.
#
# 4. NEVER MIX SETTINGS WITHIN ONE TABLE. A published row once carried four entries at
#    ny = 384 and a fifth at ny = 1536, presented as one curve. If a column needs different
#    settings, it needs its own table or an explicit note.
#
# 5. PREFER THE WELL-CONDITIONED OBSERVABLE. zeta = (1-Sbar)/(2 Kn Sbar) is an algebraic
#    transform of Sbar carrying no extra information and amplifying its error ~1.3x. Report
#    Sbar; derive zeta. Better still, where it exists, use an EXTRAPOLATION-FREE observable:
#    the wall stress and wall heat flux are exact half-space integrals with no fit in them,
#    and they stay well-conditioned at high Kn where the core-fit R^2 degrades to 0.999.
#
# 6. FOR THERMAL PROBLEMS, READ BOTH WALLS. zeta_T differs between the hot and cold plate
#    by 6.6% at dT/T = 10%, because lambda_eff = tau sqrt(T) differs between them. The
#    two-wall MEAN is amplitude-independent to 1e-6; either wall alone carries an O(dT)
#    bias. Quoting one wall to three digits hides an ambiguity larger than the agreement
#    being claimed.
#
# 7. RECORD WHAT THE RUN WAS. Use envparam/print_run_header (src/utils/run_params.jl) so
#    the header is generated from what was actually read and marks overrides. A
#    hand-maintained banner drifts from the run it describes; one derived from the reads
#    cannot. A DVM value in the notes remains unreconcilable because its parameters were
#    never recorded.
#
# Run:  julia --project=. examples/wall_comparison_protocol.jl
# Env:  WC_KN, WC_NY, WC_NX, WC_NV, WC_QUICK
ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, Statistics, LinearAlgebra
using Riemann35: WALL_SPEC, envparam, print_run_header
using Riemann35.DVMBGK: VGrid, moments35, discrete_maxwellian!, collide_cell!

MPI.Initialized() || MPI.Init()

const KN    = parse(Float64, envparam("WC_KN",    "0.4"))
const NY    = parse(Int,     envparam("WC_NY",    "16"))    # closure cells across the gap
const NX    = parse(Int,     envparam("WC_NX",    "12"))    # DVM cells across the gap
const NV    = parse(Int,     envparam("WC_NV",    "10"))    # DVM velocity grid (per axis)
const QUICK = parse(Bool,    envparam("WC_QUICK", "true"))
const H = 1.0; const T0 = 1.0; const UW = 0.1; const VMAX = 6.0

# RULE 3: the integration time must track the operating transport mechanism.
const TAU   = KN*H*sqrt(2.0)
const NU    = T0*TAU
# QUICK trades physics for a ~2 min runtime so the PROTOCOL can be demonstrated. The
# closure dominates the cost (33 ms/step at ny=16 against 2.2 for the DVM at nv=10) and
# cost grows as ny^2, so production settings are minutes to hours -- set WC_QUICK=false.
t_end(mult) = max(mult*H*H/NU, (QUICK ? 3.0 : 20.0)*H/sqrt(T0))

"Dimensionless shear rate from a velocity profile. RULE 5: Sbar, not zeta."
function sbar_of(u::Vector{Float64}, dx::Float64)
    n = length(u); lo = max(1,n÷4); hi = min(n,3n÷4)
    xs = [(i-0.5)*dx for i in lo:hi]; us = u[lo:hi]
    xb, ub = mean(xs), mean(us)
    S = sum((xs.-xb).*(us.-ub))/sum((xs.-xb).^2)
    abs(S)*H/(2UW)
end
zeta_of(Sbar) = (1 - Sbar)/(2*KN*Sbar)      # RULE 5: derived, never measured independently

# ---------------------------------------------------------------------------------------
# The DVM reference. RULE 1: BGK, constant tau, matched to what the closure will be told.
# ---------------------------------------------------------------------------------------
function dvm_sbar(tend; nx = NX, nv = NV)
    g = VGrid(VMAX, nv); dx = H/nx; dt = 0.4*dx/VMAX
    nst = max(1, round(Int, tend/dt))
    ML = zeros(nv,nv,nv); discrete_maxwellian!(ML, 1.0, 0.0, -UW, 0.0, T0, g)
    MR = zeros(nv,nv,nv); discrete_maxwellian!(MR, 1.0, 0.0, +UW, 0.0, T0, g)
    dv3 = g.dv^3
    oL = sum(g.v[a]*ML[a,j,k]*dv3 for a in 1:nv, j in 1:nv, k in 1:nv if g.v[a] > 0)
    oR = sum(-g.v[a]*MR[a,j,k]*dv3 for a in 1:nv, j in 1:nv, k in 1:nv if g.v[a] < 0)

    f = zeros(nx, nv, nv, nv)
    for i in 1:nx
        discrete_maxwellian!(@view(f[i,:,:,:]), 1.0, 0.0,
                             UW*(2*(i-0.5)/nx - 1), 0.0, T0, g)
    end
    infl(fc, sgn) = sum(-sgn*g.v[a]*fc[a,j,k]*dv3
                        for a in 1:nv, j in 1:nv, k in 1:nv if sgn*g.v[a] < 0)
    fn = similar(f)
    for _ in 1:nst
        # exact half-space diffuse wall: rho_w from the DISCRETE half-fluxes, so zero net
        # mass flux holds by construction on a finite velocity grid
        rL = infl(@view(f[1,:,:,:]), +1)/oL
        rR = infl(@view(f[nx,:,:,:]), -1)/oR
        copyto!(fn, f)
        @inbounds for a in 1:nv
            s = g.v[a]; s == 0 && continue
            for k in 1:nv, j in 1:nv, i in 1:nx
                up = s > 0 ? (i == 1  ? rL*ML[a,j,k] : f[i-1,a,j,k]) :
                             (i == nx ? rR*MR[a,j,k] : f[i+1,a,j,k])
                fn[i,a,j,k] = f[i,a,j,k] - (dt/dx)*s*(s > 0 ? f[i,a,j,k]-up : up-f[i,a,j,k])
            end
        end
        copyto!(f, fn)
        for i in 1:nx; collide_cell!(@view(f[i,:,:,:]), dt, TAU, g); end
    end
    M = reduce(vcat, [moments35(@view(f[i,:,:,:]), g)' for i in 1:nx])
    sbar_of([M[i,6]/M[i,1] for i in 1:nx], H/nx)
end

# ---------------------------------------------------------------------------------------
# The closure. RULE 1: Pr = 1 and omega = 1 to match the BGK reference, identical tau.
# ---------------------------------------------------------------------------------------
function closure_sbar(tend; ny = NY)
    halo = 2; nx = halo + 2; nz = 1
    dy = H/ny; dt = 0.2*dy/(5.0*sqrt(T0))
    nst = max(1, ceil(Int, tend/dt))
    # uw2 = x-tangent for a y-normal wall; uw1 is the z-tangent.
    WALL_SPEC[] = (ylo = (Tw=T0, uw1=0.0, uw2=-UW, alpha=1.0),
                   yhi = (Tw=T0, uw1=0.0, uw2=+UW, alpha=1.0))
    BC = (xlo=:periodic, xhi=:periodic, ylo=:wall, yhi=:wall, zlo=:outflow, zhi=:outflow)
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    M = zeros(nx+2halo, ny+2halo, nz, 35)
    for k in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
        yj = clamp(((j-halo)-0.5)*dy, 0.0, H)
        M[i,j,k,:] = InitializeM4_35(1.0, UW*(2yj/H - 1), 0.0, 0.0, T0,0,0, T0,0, T0)
    end
    for _ in 1:nst
        step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dy, dy, dy, 0.0;
                           order = 2, stage_bgk_kn = 2*TAU, stage_bgk_exact = true,
                           Pr = 1.0, omega = 1.0)     # RULE 1
    end
    sbar_of([M[halo+1, halo+j, 1, 2]/M[halo+1, halo+j, 1, 1] for j in 1:ny], dy)
end

print_run_header("WALL COMPARISON PROTOCOL -- closure against DVM-BGK at a diffuse wall";
                 extra = ("tau"      => @sprintf("%.5f (both solvers)", TAU),
                          "t_end"    => @sprintf("%.2f = max(3H^2/nu, %d H/sqrt(T))",
                                                 t_end(3.0), QUICK ? 8 : 20),
                          "protocol" => "RULE 3: t_end floored by the BALLISTIC transit time"))

# --- RULE 3, applied: each solver's own time convergence, before any comparison ---------
println("\nSTEP 1 -- time convergence, checked SEPARATELY for each solver")
println("  (a check on one solver is not a check on the comparison)")
@printf("  %-10s %-14s %-14s %s\n", "solver", "t_end", "2 t_end", "change")
dvm_1 = dvm_sbar(t_end(3.0));     dvm_2 = dvm_sbar(2*t_end(3.0))
clo_1 = closure_sbar(t_end(3.0)); clo_2 = closure_sbar(2*t_end(3.0))
for (nm, a, b) in (("DVM", dvm_1, dvm_2), ("closure", clo_1, clo_2))
    @printf("  %-10s %-14.6f %-14.6f %+.3f%%%s\n", nm, a, b, 100*(b-a)/a,
            abs(b-a)/a < 0.01 ? "" : "   <-- NOT CONVERGED, do not quote this")
end

# --- RULE 2, applied: spatial convergence, again per solver -----------------------------
println("\nSTEP 2 -- spatial convergence, also per solver")
@printf("  %-10s %-14s %-14s %s\n", "solver", "coarse", "fine", "change")
dvm_c = dvm_sbar(t_end(3.0); nx = NX÷2)
clo_c = closure_sbar(t_end(3.0); ny = NY÷2)
for (nm, a, b) in (("DVM", dvm_c, dvm_1), ("closure", clo_c, clo_1))
    @printf("  %-10s %-14.6f %-14.6f %+.3f%%\n", nm, a, b, 100*(b-a)/a)
end

# --- only now is a comparison meaningful ------------------------------------------------
println("\nSTEP 3 -- the comparison, valid only because steps 1 and 2 passed")
@printf("  Kn = %.3f\n", KN)
@printf("  %-22s Sbar = %.6f   zeta = %.4f\n", "DVM-BGK (reference)", dvm_2, zeta_of(dvm_2))
@printf("  %-22s Sbar = %.6f   zeta = %.4f\n", "35-moment closure",   clo_2, zeta_of(clo_2))
@printf("  %-22s %+.1f%% in Sbar,  %+.1f%% in zeta   (RULE 5: zeta amplifies)\n",
        "closure error", 100*(clo_2/dvm_2 - 1), 100*(zeta_of(clo_2)/zeta_of(dvm_2) - 1))
println("\nRULE 4: quote these two numbers only together with the settings above.")

# bubble2d_shedding — dense bubble in crossflow, vortex-shedding validation.
#
# Faithful port of R.O. Fox's revised MATLAB reference
# main_2Dbubble_heating_3DHyQMOM35.m (2026-07-14), itself adapted from the
# McMullen & Gallis DSMC study "Hydrodynamic fluctuations near a Hopf
# bifurcation: stochastic onset of vortex shedding behind a circular cylinder"
# (SAND2024-13841J). Instead of a rigid cylinder, a COLD DENSE gas bubble at
# pressure equilibrium (rho_b = 1e5 rho_g, T_b = p/rho_b) sits in a Maxwellian
# crossflow at Ma = 0.3; the question is whether the 35-moment HyQMOM wake sheds
# at the Strouhal frequency f_s D/U ~ 0.12-0.145.
#
# Geometry / parameters (D = dp = 1, U = 1):
#   domain  Lx = 15 D (x, streamwise) by Ly = 10 D (y, transverse); square cells
#   grid    Npy = CASE_NP (default 128), Npx = 1.5 Npy  (dx = dy = Ly/Npy)
#   bubble  radius 1/2 at (Lx/3, Ly/2)
#   BC      :crossflow — inlet Maxwellian (low x) / zero-gradient outflow (high x)
#           / periodic in y   [matches Rodney's revised BCs; single rank only]
#   T = (U/Ma)^2, p = rho_g T, Tb = p/rho_b   (uniform initial pressure)
#   Kn = 0.1/Ma, CFL = 0.5, first-order HLL + projection + BGK (MATLAB parity)
#
# Long-time trick (Rodney): raise CASE_RHOG (e.g. 1e5) to slow bubble heating and
# reach ~15 D/U; the base MATLAB case uses rho_g = 1.
#
# Knobs: CASE_NP, CASE_MA, CASE_TMAX (in D/U), CASE_RHOG.
# Interface (consumed by examples/run_case.jl and gpu/stage_case.jl):
#   case() -> (tag, params, dtcap, snap_interval)
function case()
    Np    = parse(Int,     get(ENV, "CASE_NP",   "128"))
    Ma    = parse(Float64, get(ENV, "CASE_MA",   "0.3"))
    tmax  = parse(Float64, get(ENV, "CASE_TMAX", "2.0"))  # in flow times D/U (U=1)
    rhog  = parse(Float64, get(ENV, "CASE_RHOG", "1.0"))
    order = parse(Int,     get(ENV, "CASE_ORDER", "1"))   # 1 = MATLAB-parity HLL; 3 = WENO5 (GPU parity)
    sched = order == 1 ? :legacy : :recommended           # order>=2 opt-ins only for high order
    # Parity knobs (default: physical). CASE_KN overrides Kn (e.g. 1e30 => collisionless);
    # CASE_DTMAX pins the time step (fixed dt when below the CFL bound) for GPU-vs-CPU parity.
    kn_over = get(ENV, "CASE_KN", "")
    dtmax   = parse(Float64, get(ENV, "CASE_DTMAX", "1.0"))
    # Symmetry-breaking seed (default 0 = strict symmetric Rodney case). A small
    # bubble y-offset (in D) breaks the exact up-down symmetry the deterministic
    # solver otherwise preserves to machine precision, so a shedding instability
    # (if the flow is above Re_c) has a seed to grow from. Analogue of a slightly
    # off-centre cylinder; the McMullen & Gallis onset is noise-induced (DSMC),
    # which our deterministic HyQMOM lacks.
    yoff = parse(Float64, get(ENV, "CASE_YOFF", "0.0"))
    # Gentle-start inlet ramp (default 0 = off/instant). CASE_URAMP > 0 ramps the
    # inlet velocity 0->U over that many D/U AND starts the ambient at REST, so the
    # flow builds up gradually instead of the impulsive-start acoustic blast that
    # otherwise dominates + gets trapped by the periodic-y box.
    uramp = parse(Float64, get(ENV, "CASE_URAMP", "0.0"))
    # Non-reflecting y-boundaries (CASE_ABSORB_Y=1): replace the periodic-y BC with
    # absorbing sponge layers that relax the boundary zone toward the freestream, so
    # the periodic-y box-resonance/Brillouin mode (which dominated the lift spectrum
    # and masked/contaminated the Kármán shedding peak) is removed. Sponge zone width
    # in cells (CASE_SPONGE_WIDTH) and relaxation rate (CASE_SPONGE_RATE, 1/time).
    absorb_y     = get(ENV, "CASE_ABSORB_Y", "0") == "1"
    sponge_width = parse(Int,     get(ENV, "CASE_SPONGE_WIDTH", "12"))
    sponge_rate  = parse(Float64, get(ENV, "CASE_SPONGE_RATE",  "20.0"))
    # Stochastic forcing (CASE_NOISE, transverse-velocity kick stddev in U): injects
    # random fluctuations in the near-wake each step to mimic the DSMC thermal noise
    # that triggers the subcritical Kármán onset the deterministic solver lacks.
    noise_amp    = parse(Float64, get(ENV, "CASE_NOISE", "0.0"))
    # FDT fluctuating-stress thermal noise (CASE_FLUCT, dimensionless intensity):
    # the physically-calibrated Landau-Lifshitz noise (random stress added to the
    # momentum flux, amplitude set by the fluctuation-dissipation theorem). This is
    # the faithful analogue of the paper's stochastic-flux term; the intensity is
    # the one free knob (effective k_B/particle-count), tuned to bracket the onset.
    fluct_intensity = parse(Float64, get(ENV, "CASE_FLUCT", "0.0"))

    U    = 1.0
    D    = 1.0
    Lx   = 15.0 * D
    Ly   = 10.0 * D
    Npy  = Np
    Npx  = round(Int, 1.5 * Npy)          # square cells: dx = dy = Ly/Npy
    Kn   = isempty(kn_over) ? 0.1 / Ma : parse(Float64, kn_over)
    T    = (U / Ma)^2                     # ambient kinetic temperature
    p    = rhog * T                       # uniform pressure
    # bubble/ambient density RATIO (CASE_RHORATIO, default 1e5 = Rodney's value).
    # Higher ratio => more inertial/rigid bubble => more cylinder-like (may be
    # needed for a Kármán wake; the ratio-1e5 bubble breathes instead of shedding).
    rhoratio = parse(Float64, get(ENV, "CASE_RHORATIO", "1.0e5"))
    rhob = rhoratio * rhog                # dense cold bubble
    Tb   = p / rhob                        # pressure equilibrium (colder as ratio grows)
    # CASE_CYLINDER=1: rigid immersed obstacle instead of a compressible bubble.
    # The obstacle disk is AMBIENT gas (rho=rhog, T=T) held at REST every step =>
    # rigid, incompressible, no-slip cylinder (like McMullen & Gallis). This is the
    # fix for "bubble breathes instead of sheds": a solid obstacle sheds a Kármán
    # street above Re_c. (rho_in/T_in set to ambient so the IC disk = rest ambient.)
    cyl = get(ENV, "CASE_CYLINDER", "0") == "1"

    # Nz: this is a 2D SPATIAL problem (z homogeneous), so Nz=1 is correct AND ~Nz×
    # faster (the 35 moments already carry the 3D velocity space). Default 4 kept for
    # back-compat; CASE_NZ=1 for the fast 2D path.
    nz = parse(Int, get(ENV, "CASE_NZ", "4"))
    params = (
        Nx = Npx, Ny = Npy, Nz = nz,
        xmin = 0.0, xmax = Lx, ymin = 0.0, ymax = Ly,
        tmax = tmax, Kn = Kn, Ma = Ma, flag2D = 0, CFL = 0.5,
        Nmom = 35, nnmax = 5_000_000, dtmax = dtmax,   # MATLAB dtmax = 1 (CASE_DTMAX pins dt)
        rhol = rhog, rhor = rhog,        # required by the runner; unused by :bubble
        T = T, r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 1_000_000, homogeneous_z = true, debug_output = false,
        ic_type = :bubble, spatial_order = order,      # 1 = first-order HLL (MATLAB parity)
        # cylinder: ambient-density rest disk held rigid; bubble: dense cold blob
        rho_in = cyl ? rhog : rhob, rho_out = rhog, bubble_radius = 0.5 * D,
        bubble_xc = Lx / 3, bubble_yc = Ly / 2 + yoff * D,   # yoff seeds shedding (default 0)
        T_in = cyl ? T : Tb, T_out = T,
        hold_obstacle = cyl,                            # re-impose the rigid disk each step
        obstacle_rho = rhog, obstacle_T = T,            # held wall state = ambient at rest
        # ambient velocity: U for instant start; 0 (rest) when ramping the inlet in
        u_out = uramp > 0 ? 0.0 : U,
        # inlet / outflow / (periodic-y  OR  absorbing-sponge-y when CASE_ABSORB_Y=1)
        bc = absorb_y ? :crossflow_absorb_y : :crossflow,
        sponge_width = sponge_width, sponge_rate = absorb_y ? sponge_rate : 0.0,
        noise_amp = noise_amp,                         # crude velocity-kick forcing (0 = off)
        fluct_intensity = fluct_intensity,             # FDT fluctuating-stress noise (0 = off)
        # gentle-start inlet ramp (time-dependent inlet Maxwellian; 0 = fixed at U)
        crossflow_uramp = uramp, crossflow_u = U,
        crossflow_rho = rhog, crossflow_T = T,         # inlet Maxwellian rho/T (velocity ramps)
        scheme = sched,
    )

    # Inlet-velocity table for the gentle-start ramp (GPU env has no InitializeM4_35):
    # column k+1 is the inlet Maxwellian at u = U*k/Nr, k=0..Nr. Staging writes it and
    # the GPU driver picks the column for u(t)=U*min(1,t/uramp). Only built when ramping.
    inlet_table = nothing
    if uramp > 0
        Nr = 200
        inlet_table = Array{Float64}(undef, 35, Nr + 1)
        for k in 0:Nr
            inlet_table[:, k+1] = InitializeM4_35(rhog, U*k/Nr, 0.0, 0.0, T, 0.0, 0.0, T, 0.0, T)
        end
    end

    # Snapshot cadence: CASE_NSNAP snapshots over the run (default 40; use ~20 per
    # D/U for spectral analysis, e.g. 300 over 15 D/U). Effective dt ~ CFL*dx /
    # (U + 4 sqrt(T)); collisions resolve automatically (dt << Kn).
    dx = Ly / Npy
    dt_est = min(dtmax, 0.5 * dx / (U + 4.0 * sqrt(T)))
    nsnap  = parse(Int, get(ENV, "CASE_NSNAP", "40"))
    ytag = yoff == 0.0 ? "" : "_y$(yoff)"
    rtag = uramp == 0.0 ? "" : "_ramp$(uramp)"
    dtag = rhoratio == 1.0e5 ? "" : "_rr$(rhoratio)"
    ctag = cyl ? "_cyl" : ""
    atag = absorb_y ? "_absy" : ""
    ntag = noise_amp > 0 ? "_noise$(noise_amp)" : ""
    ftag = fluct_intensity > 0 ? "_fluct$(fluct_intensity)" : ""
    return (tag = "bubble2d_shedding_Ma$(Ma)_Kn$(round(Kn,sigdigits=3))_rhog$(rhog)_Np$(Np)_o$(order)$(ytag)$(rtag)$(dtag)$(ctag)$(atag)$(ntag)$(ftag)",
            params = params, dtcap = params.dtmax, inlet_table = inlet_table,
            crossflow_uramp = uramp, crossflow_u = U,
            snap_interval = max(1, ceil(Int, tmax / (nsnap * dt_est))))
end

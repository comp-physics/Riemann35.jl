# bubble2d_wake — extended-domain variant of bubble2d, requested by R.O. Fox
# (2026-07-03): "extend the x domain to capture the wake behind the dense
# region"; at larger Ma he expects the wake to go unsteady. Domain x in
# [-0.5, 1.5] (bubble still at the origin, so 3 diameters of wake room),
# y in [-0.5, 0.5]; square cells (dx = dy = 1/Np, Nx = 2 Np).
#
# Best explored at Kn >= 0.01 where the collision dt cap is not binding and
# runs are CFL-limited (fast). Default tmax = 0.4 gives the ambient flow time
# to cross the wake region ~1.5x; copy BCs are crude in/outflow, so treat
# late-time y-boundary reflections with suspicion.
#
# Knobs: CASE_NP (y-resolution; default 128, production 512 -> 1024x512),
#        CASE_MA, CASE_KN, CASE_TMAX.
# Interface (consumed by examples/run_case.jl and gpu/stage_case.jl):
#   case() -> (tag, params, dtcap, snap_interval)
function case()
    Np   = parse(Int,     get(ENV, "CASE_NP",   "128"))
    Ma   = parse(Float64, get(ENV, "CASE_MA",   "1.0"))
    Kn   = parse(Float64, get(ENV, "CASE_KN",   "0.01"))
    tmax = parse(Float64, get(ENV, "CASE_TMAX", "0.4"))
    rho_in = 1000.0
    dtcap = Kn / sqrt(rho_in)            # resolve collisions in the dense bubble
    params = (
        Nx = 2Np, Ny = Np, Nz = 4,
        xmin = -0.5, xmax = 1.5,
        tmax = tmax, Kn = Kn, Ma = Ma, flag2D = 0, CFL = 1/3,
        Nmom = 35, nnmax = 1_000_000, dtmax = dtcap,
        rhol = 1.0, rhor = 1.0,          # required by the runner; unused by :bubble
        T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
        ic_type = :bubble, spatial_order = 2,
        rho_in = rho_in, rho_out = 1.0, bubble_radius = 0.125,
        T_in = 1/rho_in, T_out = 1.0, u_out = Ma,                # uniform p=1, ambient flow
        scheme = :recommended,
    )
    # ~20 snapshots: estimate the effective dt as min(collision cap, CFL dt at
    # the ambient state) — at Kn >= 0.01 the CFL estimate is the binding one.
    dt_est = min(dtcap, 0.3 / (Np * (abs(Ma) + 2.6)))
    return (tag = "bubble2d_wake_Ma$(Ma)_Kn$(Kn)_Np$(Np)",
            params = params, dtcap = dtcap,
            snap_interval = max(1, ceil(Int, tmax / (20 * dt_est))))
end

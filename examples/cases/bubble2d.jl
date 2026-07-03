# bubble2d — uniform-pressure dense bubble, quasi-2D. Proposed as a validation
# case by R.O. Fox (July 2026; MATLAB reference main_2Dbubble_heating_3DHyQMOM35.m,
# IC after the Rice et al. 2026 dense-bubble problem but cold and uniform-p):
# a cold dense disk (rho=1000, T=1e-3, r <= 1/8) at the origin in an ambient
# reference gas (rho=T=p=1) flowing past at u=(Ma,0,0). Quasi-2D flow past an
# effectively rigid cold cylinder with heat transfer. Copy BCs act as crude
# in/outflow — keep tmax small enough that disturbances stay mostly interior.
# dt is capped at Kn/sqrt(rho_bubble) to resolve collisions inside the bubble.
#
# Knobs: CASE_NP (default 128; reference production run 512), CASE_MA, CASE_KN,
#        CASE_TMAX.
# Interface (consumed by examples/run_case.jl and gpu/stage_case.jl):
#   case() -> (tag, params, dtcap, snap_interval)
function case()
    Np   = parse(Int,     get(ENV, "CASE_NP",   "128"))
    Ma   = parse(Float64, get(ENV, "CASE_MA",   "1.0"))
    Kn   = parse(Float64, get(ENV, "CASE_KN",   "0.001"))
    tmax = parse(Float64, get(ENV, "CASE_TMAX", "0.2"))
    rho_in = 1000.0
    dtcap = Kn / sqrt(rho_in)            # resolve collisions in the dense bubble
    params = (
        Nx = Np, Ny = Np, Nz = 4,
        tmax = tmax, Kn = Kn, Ma = Ma, flag2D = 0, CFL = 1/3,
        Nmom = 35, nnmax = 1_000_000, dtmax = dtcap,
        rhol = 1.0, rhor = 1.0,          # required by the runner; unused by :bubble
        T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
        ic_type = :bubble, spatial_order = 2,
        rho_in = rho_in, rho_out = 1.0, bubble_radius = 0.125,   # r <= 1/8 disk
        T_in = 1/rho_in, T_out = 1.0, u_out = Ma,                # uniform p=1, ambient flow
        scheme = :recommended,   # pressure recon + stage BGK (docs/design/scheme-graduation.md)
    )
    return (tag = "bubble2d_Ma$(Ma)_Kn$(Kn)_Np$(Np)",
            params = params, dtcap = dtcap,
            snap_interval = max(1, ceil(Int, tmax / (20 * dtcap))))   # ~20 snapshots
end

# riemann1d — 1D uniform-pressure stationary contact. Proposed as a validation
# case by R.O. Fox (July 2026): L: (rho,u,T) = (1,0,1) | R: (1000,0,1e-3), so
# p = 1 everywhere. At Kn=0 the exact solution is stationary — every deviation
# is solver error (regression gate: test/test_rodney_cases.jl; the recommended
# scheme preserves the contact to machine precision at second order). At Kn>0
# a non-equilibrium heat flux develops on the dilute side (validation vs a
# kinetic reference).
#
# Knobs: CASE_NP (default 256), CASE_KN (default 0.0), CASE_TMAX (default 0.1).
# Interface (consumed by examples/run_case.jl and gpu/stage_case.jl):
#   case() -> (tag, params, dtcap, snap_interval)
function case()
    Np   = parse(Int,     get(ENV, "CASE_NP",   "256"))
    Kn   = parse(Float64, get(ENV, "CASE_KN",   "0.0"))
    tmax = parse(Float64, get(ENV, "CASE_TMAX", "0.1"))
    params = (
        Nx = Np, Ny = 4, Nz = 4,
        tmax = tmax, Kn = Kn, Ma = 0.0, flag2D = 0, CFL = 1/3,
        Nmom = 35, nnmax = 1_000_000, dtmax = 1000.0,
        rhol = 1.0, rhor = 1000.0,       # :riemann1d L/R densities
        T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
        ic_type = :riemann1d, spatial_order = 2,
        # defaults: ul=ur=0, Tl=1, Tr=Tl*rhol/rhor=1e-3 (uniform p), interface at x=0
        scheme = :recommended,   # pressure recon + stage BGK (docs/design/scheme-graduation.md)
    )
    return (tag = "riemann1d_Kn$(Kn)_Np$(Np)",
            params = params, dtcap = Inf, snap_interval = 25)
end

# Single source for Rodney Fox's 2D uniform-pressure dense-bubble case
# (main_2Dbubble_heating_3DHyQMOM35.m, 2026-07-02): cold dense disk (rho=1000,
# T=1e-3, r <= 1/8) at the origin, ambient reference gas (rho=T=p=1) flowing
# past at u=(Ma,0,0); p is uniform 1; dt capped at Kn/sqrt(rho_bubble) to
# resolve collisions inside the bubble.
#
# Included by the CPU driver (rodney_validation_2d.jl) and the GPU pair
# (rodney_validation_2d_gpu_prep.jl / rodney_validation_2d_gpu.jl) so the case
# is defined exactly once; the IC field itself is only ever built by the
# runner's :bubble branch.

const RODNEY2D_RHO_IN = 1000.0

rodney2d_knobs() = (
    Np   = parse(Int,     get(ENV, "RODNEY_NP",   "128")),   # Rodney's production run: 512
    Ma   = parse(Float64, get(ENV, "RODNEY_MA",   "1.0")),
    Kn   = parse(Float64, get(ENV, "RODNEY_KN",   "0.001")),
    tmax = parse(Float64, get(ENV, "RODNEY_TMAX", "0.2")),
)

rodney2d_tag(; Np, Ma, Kn, tmax) = "rodney2d_Ma$(Ma)_Kn$(Kn)_Np$(Np)"

# The collision-resolving dt cap; binding for this case (the CFL dt is several
# times larger at every resolution of interest — the GPU driver asserts this).
rodney2d_dtmax(Kn) = Kn / sqrt(RODNEY2D_RHO_IN)

# ~20 snapshots over the run given the (binding) dt cap.
rodney2d_snapshot_interval(; Kn, tmax, kw...) =
    max(1, ceil(Int, tmax / (20 * rodney2d_dtmax(Kn))))

# Runner params NamedTuple; trailing `kw...` overrides (e.g. tmax = 0.0 for IC
# extraction, snapshot knobs on the CPU driver).
function rodney2d_params(; Np, Ma, Kn, tmax, kw...)
    rho_in = RODNEY2D_RHO_IN
    return (;
        Nx = Np, Ny = Np, Nz = 4,
        tmax = tmax, Kn = Kn, Ma = Ma, flag2D = 0, CFL = 1/3,
        Nmom = 35, nnmax = 1_000_000,
        dtmax = rodney2d_dtmax(Kn),          # Rodney: resolve collisions in the dense bubble
        rhol = 1.0, rhor = 1.0,              # required by the runner; unused by :bubble
        T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
        symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
        ic_type = :bubble, spatial_order = 2,
        rho_in = rho_in, rho_out = 1.0, bubble_radius = 0.125,   # r <= 1/8 disk, per his file
        T_in = 1/rho_in, T_out = 1.0, u_out = Ma,   # uniform p=1, ambient flow
        scheme = :recommended,   # pressure recon + stage BGK (docs/design/scheme-graduation.md)
        kw...,
    )
end

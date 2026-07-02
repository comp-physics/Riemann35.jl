# Rodney Fox's 2D uniform-pressure dense-bubble case (2026-07-02): cold dense
# disk (rho=1000, T=1e-3) at the origin, ambient reference gas (rho=T=p=1)
# flowing past at u=(Ma,0,0). Quasi-2D flow past an effectively rigid cold
# cylinder with heat transfer; t <= 0.2 (dense gas barely moves). Copy BCs act
# as crude in/outflow — keep tmax small enough that disturbances stay interior.
#
# Usage:
#   julia --project=. examples/rodney_validation_2d.jl
#   RODNEY_NP=512 RODNEY_MA=1.0 RODNEY_KN=0.01 mpiexec -n 4 julia --project=. examples/rodney_validation_2d.jl
# Output: output/runs/<tag>.jld2 + browseable bundle in output/viz/ (./serve.sh).
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35

Np    = parse(Int,     get(ENV, "RODNEY_NP",   "128"))
Ma    = parse(Float64, get(ENV, "RODNEY_MA",   "1.0"))
Kn    = parse(Float64, get(ENV, "RODNEY_KN",   "0.01"))
tmax  = parse(Float64, get(ENV, "RODNEY_TMAX", "0.2"))
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

rank0 && mkpath("output/runs")
tag = "rodney2d_Ma$(Ma)_Kn$(Kn)_Np$(Np)"

params = (
    Nx = Np, Ny = Np, Nz = 4,
    tmax = tmax, Kn = Kn, Ma = Ma, flag2D = 0, CFL = 1/3,
    Nmom = 35, nnmax = 1_000_000, dtmax = 1000.0,
    rhol = 1.0, rhor = 1.0,              # required by the runner; unused by :bubble
    T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
    ic_type = :bubble, spatial_order = 2,
    rho_in = 1000.0, rho_out = 1.0, bubble_radius = 0.15,
    T_in = 1e-3, T_out = 1.0, u_out = Ma,   # uniform p=1, ambient flow
    snapshot_interval = 25,
    snapshot_filename = "output/runs/$tag.jld2",
    web_dir = "output",
)
result = simulation_runner(params)
rank0 && println("done: $tag → output/runs/$tag.jld2 (browse: output/viz/serve.sh)")

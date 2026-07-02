# Rodney Fox's 1D uniform-pressure validation case (2026-07-02):
# stationary-contact Riemann problem. L: (rho,u,T)=(1,0,1); R: (1000,0,1e-3); p≡1.
# Kn=0 → exact solution is stationary (verification; first-order preserves it to
# machine precision, see test/test_rodney_cases.jl). Kn>0 → non-equilibrium heat
# flux develops on the dilute side (validation vs kinetic reference).
#
# Usage:
#   julia --project=. examples/rodney_validation_1d.jl
#   RODNEY_NP=512 RODNEY_KN=0.01 mpiexec -n 4 julia --project=. examples/rodney_validation_1d.jl
# Output: output/runs/<tag>.jld2 + browseable bundle in output/viz/ (./serve.sh).
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35

Np    = parse(Int,     get(ENV, "RODNEY_NP",   "256"))
Kn    = parse(Float64, get(ENV, "RODNEY_KN",   "0.0"))
tmax  = parse(Float64, get(ENV, "RODNEY_TMAX", "0.1"))
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

rank0 && mkpath("output/runs")
tag = "rodney1d_Kn$(Kn)_Np$(Np)"

params = (
    Nx = Np, Ny = 4, Nz = 4,
    tmax = tmax, Kn = Kn, Ma = 0.0, flag2D = 0, CFL = 1/3,
    Nmom = 35, nnmax = 1_000_000, dtmax = 1000.0,
    rhol = 1.0, rhor = 1000.0,           # :riemann1d L/R densities
    T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
    ic_type = :riemann1d, spatial_order = 2,
    # defaults: ul=ur=0, Tl=1, Tr=Tl*rhol/rhor=1e-3 (uniform p), interface at x=0
    snapshot_interval = 25,
    snapshot_filename = "output/runs/$tag.jld2",
    web_dir = "output",
)
result = simulation_runner(params)
rank0 && println("done: $tag → output/runs/$tag.jld2 (browse: output/viz/serve.sh)")

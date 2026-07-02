# Rodney Fox's uniform-pressure validation cases (email 2026-07-02; see
# docs/superpowers/specs/2026-07-02-rodney-validation-cases-design.md).
#
# 1D: L=(rho,u,T)=(1,0,1), R=(1000,0,1e-3), p≡1. Kn=0 Euler exact solution is a
# STATIONARY CONTACT — any velocity/pressure deviation is pure numerical error.
# 2D: dense cold bubble (same two states) in ambient flow u=(Ma,0,0), quasi-2D.
using MPI
MPI.Initialized() || MPI.Init()
using Test
using Riemann35

const RODNEY_RANK = MPI.Comm_rank(MPI.COMM_WORLD)

# Complete required-params set (pattern of test_highorder_3d.jl params_ho),
# overridable per test. tmax=0.0 → zero steps → rank 0 gets the gathered IC.
rodney_params(; kw...) = merge((
    Nx = 64, Ny = 4, Nz = 4,
    tmax    = 0.0,
    Kn      = 0.0,
    Ma      = 0.0,
    flag2D  = 0,
    CFL     = 1/3,
    Nmom    = 35,
    nnmax   = 100000,
    dtmax   = 1000.0,
    rhol    = 1.0,
    rhor    = 1000.0,
    T       = 1.0,
    r110    = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000,
    homogeneous_z = true,
    debug_output  = false,
    snapshot_interval = 0,
    ic_type = :riemann1d,
    spatial_order = 2,
), NamedTuple(kw))

# field helpers on the gathered (Nx,Ny,Nz,35) array
_rho(M) = M[:, :, :, 1]
_u(M)   = M[:, :, :, 2]  ./ _rho(M)
_v(M)   = M[:, :, :, 6]  ./ _rho(M)
_w(M)   = M[:, :, :, 16] ./ _rho(M)
function _pressure(M)
    rho = _rho(M); u = _u(M); v = _v(M); w = _w(M)
    T3 = (M[:,:,:,3] ./ rho .- u.^2) .+ (M[:,:,:,10] ./ rho .- v.^2) .+
         (M[:,:,:,20] ./ rho .- w.^2)
    return rho .* T3 ./ 3
end

@testset ":riemann1d IC — uniform-pressure default" begin
    M, t, steps, grid = simulation_runner(rodney_params())
    @test steps == 0
    if RODNEY_RANK == 0
        @test size(M) == (64, 4, 4, 35)
        @test all(isfinite, M)
        rho = _rho(M)
        @test rho[1, 1, 1]   ≈ 1.0
        @test rho[end, 1, 1] ≈ 1000.0
        # zero bulk velocity everywhere in the IC
        @test maximum(abs, M[:, :, :, 2]) == 0.0
        @test maximum(abs, M[:, :, :, 6]) == 0.0
        @test maximum(abs, M[:, :, :, 16]) == 0.0
        # default Tr = Tl*rhol/rhor ⇒ uniform pressure p = 1
        @test maximum(abs, _pressure(M) .- 1.0) < 1e-10
        # uniform in y and z
        @test M == repeat(M[:, 1:1, 1:1, :], 1, 4, 4, 1)
        # interface at the domain midpoint (default domain [-0.5,0.5], Nx=64)
        @test rho[32, 1, 1] ≈ 1.0
        @test rho[33, 1, 1] ≈ 1000.0
    end
end

@testset ":riemann1d IC — explicit states override defaults" begin
    p = rodney_params(rhol = 2.0, rhor = 3.0, ul = 0.5, ur = -0.25,
                      Tl = 2.0, Tr = 0.5, x_interface = -0.25)
    M, t, steps, grid = simulation_runner(p)
    if RODNEY_RANK == 0
        rho = _rho(M); u = _u(M)
        @test rho[1, 1, 1] ≈ 2.0
        @test u[1, 1, 1]   ≈ 0.5
        @test rho[end, 1, 1] ≈ 3.0
        @test u[end, 1, 1]   ≈ -0.25
        # x_interface=-0.25 on [-0.5,0.5] with Nx=64 → cells 1:16 left, 17:64 right
        @test rho[16, 1, 1] ≈ 2.0
        @test rho[17, 1, 1] ≈ 3.0
    end
end

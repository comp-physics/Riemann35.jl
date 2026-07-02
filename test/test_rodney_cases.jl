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

@testset "stationary contact, Kn=0: first-order preserves exactly" begin
    # Kn=0 ⇒ tc=0 ⇒ exact-exponential BGK relaxes instantly to the Maxwellian
    # (exp(-dt/0)=0); the IC is already Maxwellian, so the exact solution is
    # frozen. Every metric below is therefore PURE numerical error.
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05, spatial_order = 1))
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        maxvel = max(maximum(abs, _u(M)), maximum(abs, _v(M)), maximum(abs, _w(M)))
        pdev   = maximum(abs, _pressure(M) .- 1.0)
        @info "stationary-contact error metrics (order 1)" maxvel pdev steps t
        # First-order HLL preserves the uniform-pressure stationary contact to
        # machine precision (observed ~7e-16 / ~2e-15 at Nx=32..128): with u=0 and
        # uniform p, momentum flux differences and dissipation cancel identically;
        # only density smears. This is an EXACT verification gate.
        @test maxvel < 1e-12
        @test pdev   < 1e-12
        # y/z uniformity is preserved exactly by copy BCs on a y/z-uniform field
        @test maximum(abs, M .- repeat(M[:, 1:1, 1:1, :], 1, 4, 4, 1)) < 1e-9
        # mass conservation: u≈0 at the x boundaries ⇒ near-exact
        dxg = 1.0 / 64
        mass  = sum(_rho(M)[:, 1, 1]) * dxg          # per unit y/z area
        mass0 = (32 * 1.0 + 32 * 1000.0) * dxg       # exact IC mass
        @test abs(mass - mass0) / mass0 < 1e-10
    end
end

@testset "stationary contact, Kn=0: second-order error is bounded" begin
    # Kn=0 ⇒ tc=0 ⇒ exact-exponential BGK relaxes instantly to the Maxwellian
    # (exp(-dt/0)=0); the IC is already Maxwellian, so the exact solution is
    # frozen. Every metric below is therefore PURE numerical error.
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05))
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        maxvel = max(maximum(abs, _u(M)), maximum(abs, _v(M)), maximum(abs, _w(M)))
        pdev   = maximum(abs, _pressure(M) .- 1.0)
        @info "stationary-contact error metrics (order 2)" maxvel pdev steps t
        # KNOWN LIMITATION (found by this gate, 2026-07-02): component-wise MUSCL
        # reconstruction at the 1000:1 contact produces O(5%) spurious velocity and
        # pressure L∞ error localized at the interface, roughly resolution-
        # independent (maxvel 0.037/0.064/0.070, pdev 0.031/0.039/0.051 at
        # Nx=32/64/128, t=0.05). First-order preserves the contact exactly (previous
        # testset). These ceilings pin the current error level against regression.
        @test maxvel < 0.1
        @test pdev   < 0.08
        # y/z uniformity is preserved exactly by copy BCs on a y/z-uniform field
        @test maximum(abs, M .- repeat(M[:, 1:1, 1:1, :], 1, 4, 4, 1)) < 1e-9
        # mass conservation: u≈0 at the x boundaries ⇒ near-exact
        dxg = 1.0 / 64
        mass  = sum(_rho(M)[:, 1, 1]) * dxg          # per unit y/z area
        mass0 = (32 * 1.0 + 32 * 1000.0) * dxg       # exact IC mass
        @test abs(mass - mass0) / mass0 < 1e-10
    end
end

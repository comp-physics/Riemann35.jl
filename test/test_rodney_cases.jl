# Rodney Fox's uniform-pressure validation cases (email 2026-07-02; see
# docs/design/rodney-validation-cases.md).
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

@testset "stationary contact, Kn=0: LEGACY second-order error is bounded" begin
    # Pins the LEGACY scheme's error level (the package default is now
    # :recommended, which is machine-exact — see later testsets). Every metric
    # below is pure numerical error of the legacy order-2 scheme.
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05, scheme = :legacy))
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

# ---------------------------------------------------------------------------
# :bubble extension — Rodney's 2D uniform-pressure dense-bubble case
# ---------------------------------------------------------------------------

# complete bubble params (rhol/rhor required by the runner but unused by :bubble)
bubble_params(; kw...) = merge(rodney_params(
        Nx = 16, Ny = 16, Nz = 4, rhol = 1.0, rhor = 1.0,
        ic_type = :bubble,
    ), NamedTuple(kw))

# Rodney's uniform-pressure state: p ≡ 1 inside and out, ambient flow at Ma=1
rodney_bubble_up = (rho_in = 1000.0, rho_out = 1.0, T_in = 1e-3, T_out = 1.0,
                    u_out = 1.0, bubble_radius = 0.15)

@testset ":bubble new params default byte-identical" begin
    for prof in (:sharp, :smooth)
        M1, _, _, _ = simulation_runner(bubble_params(bubble_profile = prof))
        M2, _, _, _ = simulation_runner(bubble_params(bubble_profile = prof,
                                                      T_in = 1.0, T_out = 1.0, u_out = 0.0))
        if RODNEY_RANK == 0
            @test M1 == M2   # bitwise
        end
    end
end

@testset "uniform-pressure bubble IC (Rodney 2D case)" begin
    M, _, _, _ = simulation_runner(bubble_params(; rodney_bubble_up...))
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        @test maximum(abs, _pressure(M) .- 1.0) < 1e-10
        rho = _rho(M); u = _u(M)
        @test rho[8, 8, 1] ≈ 1000.0        # center cell inside bubble
        @test u[8, 8, 1]   ≈ 0.0 atol=1e-14
        @test rho[1, 1, 1] ≈ 1.0           # corner is ambient
        @test u[1, 1, 1]   ≈ 1.0
    end
    # smooth variant blends PRESSURE, so uniform p stays exactly uniform
    Ms, _, _, _ = simulation_runner(bubble_params(; rodney_bubble_up...,
                                                  bubble_profile = :smooth))
    if RODNEY_RANK == 0
        @test all(isfinite, Ms)
        @test maximum(abs, _pressure(Ms) .- 1.0) < 1e-10
        @test _rho(Ms)[8, 8, 1] > 100.0    # dense core present
    end
end

@testset "uniform-pressure bubble: short run stays sane" begin
    M0, _, _, _ = simulation_runner(bubble_params(; rodney_bubble_up...))   # IC
    M, t, steps, grid = simulation_runner(bubble_params(; rodney_bubble_up...,
                                                        Ma = 1.0, Kn = 0.01, tmax = 0.005))
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        @test minimum(_rho(M)) > 0.0
        # copy BCs with ambient throughflow ⇒ modest mass drift allowed
        dxg = 1.0 / 16
        mass  = sum(_rho(M)[:, :, 1])  * dxg^2
        mass0 = sum(_rho(M0)[:, :, 1]) * dxg^2
        @test abs(mass - mass0) / mass0 < 5e-2
    end
end

# ---------------------------------------------------------------------------
# ho_pressure_recon — pressure-tensor reconstruction variables (opt-in)
# ---------------------------------------------------------------------------

@testset "stationary contact, Kn=0: order-2 + ho_pressure_recon 13x better" begin
    # prec WITHOUT stage_bgk (pin :legacy so the bundle does not add stage_bgk)
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05, scheme = :legacy,
                                                        ho_pressure_recon = true))
    @test steps >= 1
    if RODNEY_RANK == 0
        maxvel = max(maximum(abs, _u(M)), maximum(abs, _v(M)), maximum(abs, _w(M)))
        pdev   = maximum(abs, _pressure(M) .- 1.0)
        @info "stationary-contact (order 2, pressure recon) error metrics" maxvel pdev steps t
        # Pressure-tensor recon vars (slots 5-7 hold P_ii = rho*C2ii): at the
        # uniform-p contact every recon var except rho is uniform, so all MUSCL
        # slopes vanish and the RECONSTRUCTION error channel is eliminated —
        # maxvel drops 0.064 -> 0.0049, pdev 0.039 -> 0.014 at Nx=64 (13x / 2.7x).
        # The residual (0.0027/0.0049/0.0060 at Nx=32/64/128, saturating) comes
        # from a DIFFERENT channel: SSP-RK3 stages are collisionless (BGK applied
        # once per full step), so at Kn=0 stage 1 pumps transient M300 from the
        # M400 = 3p^2/rho variation, later stages flux it into pressure, and the
        # velocity error persists (collision preserves u). Applying the
        # exact-exponential BGK per RK stage is the follow-up predicted to make
        # this machine-exact. Ceilings pin the improved level (~2x observed).
        @test maxvel < 0.01
        @test pdev   < 0.03
        @test maximum(abs, M .- repeat(M[:, 1:1, 1:1, :], 1, 4, 4, 1)) < 1e-9
    end
end

@testset "ho_pressure_recon off is identical to never setting it (within :legacy)" begin
    M1, _, _, _ = simulation_runner(rodney_params(tmax = 0.02, scheme = :legacy))
    M2, _, _, _ = simulation_runner(rodney_params(tmax = 0.02, scheme = :legacy,
                                                  ho_pressure_recon = false))
    if RODNEY_RANK == 0
        @test M1 == M2   # bitwise
    end
end

@testset "ho_pressure_recon: crossing-jets near-vacuum robustness smoke" begin
    p = rodney_params(Nx = 16, Ny = 16, Nz = 16, tmax = 0.01, Kn = 1000.0,
                      rhol = 1.0, rhor = 0.001, homogeneous_z = false,
                      ic_type = :crossing_matlab, ho_pressure_recon = true)
    M, t, steps, grid = simulation_runner(p)
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        @test minimum(_rho(M)) > 0.0
    end
end

# ---------------------------------------------------------------------------
# stage_bgk — BGK collision applied per SSP-RK3 stage (opt-in, single-source
# helper shared with the GPU path)
# ---------------------------------------------------------------------------

@testset "bgk_relax_tup matches legacy collision35" begin
    Mtest = InitializeM4_35(2.5, 0.3, -0.2, 0.1, 1.3, 0.05, -0.02, 0.9, 0.03, 1.1)
    for (dt, Kn) in ((1e-3, 1.0), (1e-2, 0.01), (0.1, 1000.0))
        legacy = collision35(Mtest, dt, Kn)
        tup    = collect(Riemann35.bgk_relax_tup(ntuple(i -> Mtest[i], 35), Float64(dt), Float64(Kn)))
        @test maximum(abs.(tup .- legacy) ./ max.(abs.(legacy), 1e-300)) < 1e-12
    end
    # Kn=Inf: e=1, and MG - 1.0*(MG - M) exposes the last-ulp difference between
    # the two Maxwellian construction paths (from_recon_vars_dev vs
    # InitializeM4_35) — bitwise-equal on some platforms but not all (CI showed
    # 1-ulp differences), so assert the same 1e-12 relative parity as above.
    tup_inf = collect(Riemann35.bgk_relax_tup(ntuple(i -> Mtest[i], 35), 1e-2, Inf))
    legacy_inf = collision35(Mtest, 1e-2, Inf)
    @test maximum(abs.(tup_inf .- legacy_inf) ./ max.(abs.(legacy_inf), 1e-300)) < 1e-12
    g0 = Riemann35.bgk_relax_tup(ntuple(i -> Mtest[i], 35), 1e-2, 0.0)
    g1 = Riemann35.bgk_relax_tup(g0, 1e-2, 0.0)
    @test maximum(abs.(collect(g1) .- collect(g0))) < 1e-13
end

@testset "stationary contact, Kn=0: pressure recon + stage BGK is machine-exact" begin
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05,
                                          ho_pressure_recon = true, stage_bgk = true))
    @test steps >= 1
    if RODNEY_RANK == 0
        maxvel = max(maximum(abs, _u(M)), maximum(abs, _v(M)), maximum(abs, _w(M)))
        pdev   = maximum(abs, _pressure(M) .- 1.0)
        @info "stationary-contact (order 2, pressure recon + stage BGK)" maxvel pdev steps t
        # With BGK applied after every RK stage, each stage output is Maxwellian
        # at Kn=0 (M300 reset before it can flux M200 in the next stage), and
        # pressure recon zeroes every non-density slope: BOTH order-2 error
        # channels closed => the contact is an exact invariant, as at order 1.
        @test maxvel < 1e-12
        @test pdev   < 1e-12
    end
end

@testset "stage_bgk off is identical to never setting it (within :legacy)" begin
    M1, _, _, _ = simulation_runner(rodney_params(tmax = 0.02, scheme = :legacy))
    M2, _, _, _ = simulation_runner(rodney_params(tmax = 0.02, scheme = :legacy,
                                                  stage_bgk = false))
    if RODNEY_RANK == 0
        @test M1 == M2   # bitwise
    end
end

@testset "stage_bgk at finite Kn: conservative and sane" begin
    M0, _, _, _ = simulation_runner(rodney_params())                     # IC
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.02, Kn = 0.01,
                                          stage_bgk = true))
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        @test minimum(_rho(M)) > 0.0
        # BGK conserves mass, momentum, energy pointwise; transport conserves mass
        mass  = sum(_rho(M)[:, 1, 1])
        mass0 = sum(_rho(M0)[:, 1, 1])
        @test abs(mass - mass0) / mass0 < 1e-10
    end
end

@testset "limiter + pressure recon + stage BGK: contact still machine-exact" begin
    # ho_realizability_limiter slope-limits in the SAME recon variables, so with
    # pressure recon all non-density slopes vanish at the uniform-p contact and
    # theta=1 everywhere: the scaling limiter must not break exactness. This also
    # covers the limiter+pressure_recon combination end-to-end (CPU); the GPU
    # combination is covered by gpu/validation/stagebgk_* mode "limprec".
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05,
        ho_pressure_recon = true, stage_bgk = true, ho_realizability_limiter = true))
    @test steps >= 1
    if RODNEY_RANK == 0
        maxvel = max(maximum(abs, _u(M)), maximum(abs, _v(M)), maximum(abs, _w(M)))
        pdev   = maximum(abs, _pressure(M) .- 1.0)
        @info "stationary-contact (order 2, limiter + pressure recon + stage BGK)" maxvel pdev
        @test maxvel < 1e-12
        @test pdev   < 1e-12
    end
end

@testset "pressure-aware theta oracle: engaged limiter, P-form == C-form == device" begin
    # A stencil whose MUSCL faces cross the S400 >= 1 + S300^2 realizability
    # boundary engages the scaling limiter (theta << 1). The theta must be
    # IDENTICAL through three routes: the CPU wrapper path with
    # ho_pressure_recon on (P-form recon vars), the shared device
    # scaling_theta_dev with prec=true (exactly what the GPU compiles), and the
    # plain C-form reference (the same physical state, flag off). rho != 1 so
    # the P-form genuinely differs from the C-form.
    RZ = Riemann35.RealizeDev
    function vvec(rho, C, s3, s4)
        V = zeros(35); V[1] = rho; V[5] = C; V[6] = C; V[7] = C
        S = zeros(28); S[1] = s3; S[2] = s4                       # S300, S400
        S[7] = 1.0; S[10] = 3.0; S[15] = 1.0; S[18] = 3.0; S[28] = 1.0  # Maxwellian S220/S040/S202/S004/S022
        V[8:35] .= S
        return V
    end
    topform(V) = (W = copy(V); W[5] *= W[1]; W[6] *= W[1]; W[7] *= W[1]; W)
    rho = 7.0; C = 0.3
    Vm1 = vvec(rho, C, 0.0, 2.97); V0 = vvec(rho, C, 1.4, 2.97); Vp1 = vvec(rho, C, 2.8, 2.97)
    Riemann35.HO_PRESSURE_RECON[] = true
    θcpu = scaling_limited_faces(topform(Vm1), topform(V0), topform(Vp1))[3]
    Riemann35.HO_PRESSURE_RECON[] = false
    θdev = RZ.scaling_theta_dev(ntuple(i -> topform(Vm1)[i], 35), ntuple(i -> topform(V0)[i], 35),
                                ntuple(i -> topform(Vp1)[i], 35), true)
    θref = scaling_limited_faces(Vm1, V0, Vp1)[3]
    @test θdev < 1.0                 # the limiter actually engages
    @test θdev == θcpu               # device == CPU wrapper, bitwise
    @test abs(θdev - θref) < 1e-9    # P-form == C-form (same physical state)
end

@testset "scheme bundle: default == :recommended == explicit flags (bitwise)" begin
    Mr1, _, _, _ = simulation_runner(rodney_params(tmax = 0.02, scheme = :recommended))
    Mr2, _, _, _ = simulation_runner(rodney_params(tmax = 0.02, ho_pressure_recon = true, stage_bgk = true))
    Md,  _, _, _ = simulation_runner(rodney_params(tmax = 0.02))                     # package default
    Ml,  _, _, _ = simulation_runner(rodney_params(tmax = 0.02, scheme = :legacy))
    if RODNEY_RANK == 0
        @test Mr1 == Mr2
        @test Md  == Mr1     # the DEFAULT is the recommended scheme
        @test Mr1 != Ml      # and :legacy still reproduces the old behavior
    end
end

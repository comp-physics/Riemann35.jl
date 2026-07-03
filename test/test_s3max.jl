# test_s3max.jl — the user-settable realizability |S3| clamp (R.O. Fox, 2026-07:
# the legacy MATLAB value was too small at larger Kn; runner default is now
# max(40, 4+|Ma|/2), function-level default stays the legacy 4+|Ma|/2 so the
# golden/MATLAB-parity batteries are unaffected).
using Test
using Riemann35
using Riemann35: realizable_3D_M4, InitializeM4_35, Flux_closure35_and_realizable_3D,
                 to_recon_vars, from_recon_vars, enforce_univariate

@testset "s3max clamp parameter" begin
    Ma = 1.0
    M = InitializeM4_35(1.0, 0.5, 0.0, 0.0, 1.0, 0.3, -0.1, 0.9, 0.2, 1.1)

    @testset "legacy pin is bitwise-identical to the no-arg default" begin
        @test realizable_3D_M4(M, Ma) == realizable_3D_M4(M, Ma, 4.0 + abs(Ma) / 2.0)
        a = Flux_closure35_and_realizable_3D(M, 0, Ma)
        b = Flux_closure35_and_realizable_3D(M, 0, Ma; s3max = 4.0 + abs(Ma) / 2.0)
        @test all(a[i] == b[i] for i in 1:4)
    end

    @testset "clamp engages: |S3|=8 state differs under s3max 4.5 vs 40" begin
        V = to_recon_vars(M)
        V[8] = 8.0                     # S300 well past the legacy clamp
        V[9] = 1 + 64 + 0.5            # S400 keeps H200 > 0
        Mx = from_recon_vars(V)
        c_legacy = realizable_3D_M4(Mx, Ma)          # univariate stage clips S300 to 4.5
        c_wide   = realizable_3D_M4(Mx, Ma, 40.0)    # univariate stage leaves S300 alone
        @test c_legacy != c_wide                      # (joint projection then acts on both)
        S_leg = to_recon_vars(collect(c_legacy))
        @test abs(S_leg[8]) <= 4.5 + 1e-12            # legacy clamp really clips
        # precise clamp semantics at the univariate stage itself:
        S3w, _, _ = enforce_univariate(8.0, 65.5, 1e-6, 40.0)
        S3l, _, _ = enforce_univariate(8.0, 65.5, 1e-6, 4.5)
        @test S3w == 8.0 && S3l == 4.5
    end

    @testset "runner threads params.s3max (order 2, Kn=1 engages the clamp)" begin
        base = (Nx = 8, Ny = 8, Nz = 4, tmax = 0.01, Kn = 1.0, Ma = 1.0, flag2D = 0,
                CFL = 1/3, Nmom = 35, nnmax = 100000, dtmax = 1000.0,
                rhol = 1.0, rhor = 1.0, T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
                symmetry_check_interval = 100000, homogeneous_z = true,
                debug_output = false, ic_type = :bubble, spatial_order = 2,
                rho_in = 1000.0, rho_out = 1.0, bubble_radius = 0.25,
                T_in = 1e-3, T_out = 1.0, u_out = 1.0, scheme = :recommended)
        if MPI.Comm_size(MPI.COMM_WORLD) == 1
            Mleg, _, _, _ = simulation_runner((; base..., s3max = 4.5))
            Mdef, _, _, _ = simulation_runner(base)          # default = max(40, 4.5) = 40
            Mpin, _, _, _ = simulation_runner((; base..., s3max = 40.0))
            @test Mdef == Mpin                                # default is 40 here
            @test Mleg != Mdef                                # and the clamp matters at Kn=1
        end
    end
end

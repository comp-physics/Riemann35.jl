# test_roeps3.jl — the opt-in parity-split Roe flux (riemann_solver = :roeps3).
# Single source src/numerics/roeps3_dev.jl, shared CPU/GPU. Design + validation:
# the July 2026 flux study (1D reference solver); headline property: uniform-
# pressure contacts are preserved exactly at any contact speed (constant
# even-sector dissipation coefficient — the parity theorem).
using Test
using LinearAlgebra
using StaticArrays
using Riemann35
using Riemann35: InitializeM4_35
using Riemann35.RoePS3Dev: roeps3_diss_dev, _marg_eigen5, _vandermonde_solve5!

@testset "RoePS3 flux" begin
    @testset "core: Vandermonde solve + marginal spectrum" begin
        x = MVector{5,Float64}(-2.1, -0.7, 0.3, 1.1, 2.4)
        y = MVector{5,Float64}(1.0, -0.5, 2.0, 0.7, -1.3)
        V = [x[j]^(k - 1) for k in 1:5, j in 1:5]
        a = copy(y)
        _vandermonde_solve5!(a, x)
        @test norm(V * Vector(a) .- Vector(y)) < 1e-12

        w = 1.0 .* [1, 0.2, 1.1, 0.5, 3.4]                 # skewed marginal
        l = collect(_marg_eigen5(w...)[1:5])
        a1 = w[2] / w[1]; s33 = w[3] - a1 * w[2]; s34 = w[4] - a1 * w[3]
        s35 = w[5] - a1 * w[4]
        a2 = s34 / s33 - w[2] / w[1]; b2 = s33 / w[1]
        b3 = max((s35 - a2 * s34 - b2 * w[3]) / s33, 1e-10) * 2.5
        a3 = (a1 + a2) / 2
        lref = sort(vcat(eigvals(Symmetric([a1 sqrt(b2); sqrt(b2) a2])),
                         eigvals(Symmetric([a1 sqrt(b2) 0; sqrt(b2) a2 sqrt(b3); 0 sqrt(b3) a3]))))
        @test norm(l .- lref) / norm(lref) < 1e-12
    end

    @testset "even-jump identity: D = q(u)·Δm at the uniform-p contact" begin
        ML = Tuple(InitializeM4_35(1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0))
        MR = Tuple(InitializeM4_35(1000.0, 0, 0, 0, 1e-3, 0, 0, 1e-3, 0, 1e-3))
        sl, sr = -2.5, 2.5
        qu = -2 * sl * sr / (sr - sl)
        for ax in 1:3
            D = roeps3_diss_dev(ML, MR, ax, sl, sr)
            Dref = qu .* (collect(MR) .- collect(ML))
            @test maximum(abs.(collect(D) .- Dref)) / maximum(abs.(Dref)) < 1e-13
        end
    end

    @testset "runner: stationary contact machine-exact at order 2" begin
        if MPI.Comm_size(MPI.COMM_WORLD) == 1
            base = (Nx = 32, Ny = 4, Nz = 4, tmax = 0.02, Kn = 0.0, Ma = 0.0,
                    flag2D = 0, CFL = 1/3, Nmom = 35, nnmax = 100000, dtmax = 1000.0,
                    rhol = 1.0, rhor = 1000.0, T = 1.0, r110 = 0.0, r101 = 0.0,
                    r011 = 0.0, symmetry_check_interval = 100000,
                    homogeneous_z = true, debug_output = false,
                    ic_type = :riemann1d, spatial_order = 2, scheme = :recommended,
                    riemann_solver = :roeps3)
            M, _, _, _ = simulation_runner(base)
            u = M[:, 1, 1, 2] ./ M[:, 1, 1, 1]
            p = M[:, 1, 1, 3] .- M[:, 1, 1, 2] .^ 2 ./ M[:, 1, 1, 1]
            @test maximum(abs, u) < 1e-12
            @test maximum(abs, p .- 1) < 1e-12
        end
    end

    @testset "poison face routed to HLL (shape gate + realizability backstop)" begin
        # The worst gate-fired face captured on the Ma=100 crossing-jets kill
        # (16^3 reproducer, 2026-07-03): rho/u/p continuous to 1e-6 but ~48%
        # jumps in the 3rd/4th-order x-marginal moments (different mixes of
        # two counter-streaming beams; kurtosis 1.78 vs 1.09). The wave-split
        # dissipation here put an anti-diffusive -0.31 in the MASS row against
        # drho = 2.6e-8 — 1e7x the Rusanov bound — and killed the run in 3
        # steps. The flux MUST route this face to HLL.
        ML = (0.036202114813959956, -5.573644655450923e-7, 142.7503137104818,
              -7898.911964371504, 999961.9684727857, -0.0028017859540235166,
              142.62382520846108, -7902.958539427523, 999687.1177882002,
              142.96790502915647, -7933.011847405899, 1.0027068374863835e6,
              -7937.117830204303, 1.0024362207271369e6, 1.0054611712939047e6,
              0.002800638209395703, 142.59120266437313, -7879.062226314581,
              998236.2097305928, 143.03234941002378, -7892.439621415258,
              1.0007159379035836e6, -7872.482462225589, 998984.7039345226,
              1.001472412376428e6, 142.75820500502283, -7899.358214130146,
              1.0000177776840269e6, -7913.152704724206, 1.0009778951137232e6,
              1.0027643095322745e6, -7896.468333401173, 1.0004374049952858e6,
              1.0007691641190496e6, 1.0034609489505584e6)
        MR = (0.03620214052497041, -5.573648613894697e-7, 142.75030035933113,
              2631.2097945730193, 611387.1876379917, -0.0028017879438726498,
              142.5912039898323, 2617.231627436366, 610502.1950244632,
              143.03236660449994, 2614.3382305630853, 612188.4402291763,
              2600.264190797183, 611302.1486446162, 612993.8989391038,
              0.002800640198429764, 142.62383733634283, 2639.9243238734803,
              611049.1515955298, 142.96790395900874, 2657.2882239139317,
              612725.9192152589, 2666.069238246919, 612390.0560045339,
              614070.7341811847, 142.75821286844172, 2631.356984651248,
              611421.1010201691, 2623.04409475001, 611847.551382906,
              612221.5208406678, 2643.3095729172874, 611838.253855129,
              612760.4079732308, 613525.7780347761)
        sL, sR = -96.28614047470464, 72.68375591847992
        FL = Tuple(Riemann35._phys_flux(collect(ML), 1))
        FR = Tuple(Riemann35._phys_flux(collect(MR), 1))
        Fr = Riemann35.riemann_flux_dev(2, 1, ML, MR, FL, FR, sL, sR)
        Fh = Riemann35.riemann_flux_dev(0, 1, ML, MR, FL, FR, sL, sR)
        @test all(Fr .=== Fh)

        # shape params: the gate must reject on |dK| and |dqhat|
        using Riemann35.RiemannFluxDev: _marg_shape, _state_realizable
        okL, qhL, KL = _marg_shape(ML[1], ML[2], ML[3], ML[4], ML[5])
        okR, qhR, KR = _marg_shape(MR[1], MR[2], MR[3], MR[4], MR[5])
        @test okL && okR
        @test abs(qhR - qhL) > 0.5 || abs(KR - KL) > 0.5

        # realizability primitive: Maxwellian passes, sub-Hamburger fails
        m = Tuple(InitializeM4_35(1.0, 0.1, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0))
        @test _state_realizable(m)
        bad = ntuple(j -> j == 5 ? 0.9 * (m[3]^2 / m[1]) : m[j], 35)  # K < 1
        @test !_state_realizable(bad)
    end

    @testset "selector rejects unknown solver" begin
        old = Riemann35.RIEMANN_SOLVER[]
        try
            Riemann35.RIEMANN_SOLVER[] = :bogus
            m = InitializeM4_35(1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0)
            @test_throws ArgumentError Riemann35.face_flux_1d(m, m, 1, 0.0)
        finally
            Riemann35.RIEMANN_SOLVER[] = old
        end
    end
end

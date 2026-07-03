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

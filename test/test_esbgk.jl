# ---------------------------------------------------------------------------
# ES-BGK + VHS transport (docs/design/esbgk-vhs-transport.md)
#
# The load-bearing test is the FIRST one: the kappa time-integration construction
# of spec section 4 claims the convex update is EXACT in the second moments. If
# that is wrong the whole design is wrong, so it is tested before anything else.
#
# The Isserlis target is validated against an INDEPENDENT route: a tensor
# Gauss-Hermite quadrature of the Gaussian with covariance Lambda. That is a
# genuinely different computation (numerical integration vs closed-form Wick
# pairing), so agreement is evidence rather than a restatement.
# ---------------------------------------------------------------------------
using Test, LinearAlgebra
using Riemann35
using Riemann35.ReconDev: bgk_relax_tup

# canonical (i,j,k) of each of the 35 raw-moment slots
const IJK35 = [
    (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
    (0,1,0),(1,1,0),(2,1,0),(3,1,0),
    (0,2,0),(1,2,0),(2,2,0),
    (0,3,0),(1,3,0),
    (0,4,0),
    (0,0,1),(1,0,1),(2,0,1),(3,0,1),
    (0,0,2),(1,0,2),(2,0,2),
    (0,0,3),(1,0,3),
    (0,0,4),
    (0,1,1),(1,1,1),(2,1,1),
    (0,2,1),(1,2,1),
    (0,3,1),
    (0,1,2),(1,1,2),
    (0,1,3),
    (0,2,2)]

"probabilists' Gauss-Hermite nodes/weights via Golub-Welsch (standard normal weight)"
function gh(n::Int)
    J = SymTridiagonal(zeros(n), [sqrt(k) for k in 1:n-1])
    F = eigen(J)
    F.values, [F.vectors[1, i]^2 for i in 1:n]
end

"35 raw moments of the Gaussian(rho; mean; covariance L) by tensor quadrature"
function gaussian_moments_quad(rho, u, v, w, L::Matrix{Float64}; n::Int = 7)
    x, ω = gh(n)
    A = cholesky(Symmetric(L)).L
    M = zeros(35)
    @inbounds for a in 1:n, b in 1:n, c in 1:n
        z  = (x[a], x[b], x[c])
        wt = ω[a]*ω[b]*ω[c]
        cx = A[1,1]*z[1]
        cy = A[2,1]*z[1] + A[2,2]*z[2]
        cz = A[3,1]*z[1] + A[3,2]*z[2] + A[3,3]*z[3]
        px, py, pz = u + cx, v + cy, w + cz
        for s in 1:35
            (i, j, k) = IJK35[s]
            M[s] += wt * px^i * py^j * pz^k
        end
    end
    rho .* M
end

"(Theta, C) from a raw 35-moment vector"
function theta_cov(M)
    rho = M[1]
    u = M[2]/rho; v = M[6]/rho; w = M[16]/rho
    C = [M[3]/rho-u*u   M[7]/rho-u*v   M[17]/rho-u*w
         M[7]/rho-u*v   M[10]/rho-v*v  M[26]/rho-v*w
         M[17]/rho-u*w  M[26]/rho-v*w  M[20]/rho-w*w]
    tr(C)/3, C
end

kappa_ref(Pr, y) = Pr == 1.0 ? 0.0 :
    -exp(-Pr*y) * expm1((Pr-1.0)*y) / expm1(-Pr*y)
tauref(Kn, Theta, rho, omega) = (Kn/2) * Theta^(omega-1.0) / rho

# an anisotropic, correlated, comfortably realizable state
const LTEST = [1.7 0.30 -0.20; 0.30 0.9 0.15; -0.20 0.15 1.1]
const MTEST = gaussian_moments_quad(1.3, 0.20, -0.10, 0.05, LTEST)

@testset "ES-BGK + VHS" begin

    @testset "kappa: second-moment exactness (falsifies spec section 4)" begin
        # After one step the deviatoric stress must decay as exp(-y), at the
        # Kn-defined rate, INDEPENDENT of Pr.
        worst = 0.0
        for Pr in (2/3, 0.7, 0.8), omega in (0.5, 0.74, 1.0),
            Kn in (1e-2, 1.0, 50.0), dt in (1e-6, 1e-3, 0.05, 0.5)
            M0 = Tuple(MTEST)
            Th, C0 = theta_cov(MTEST)
            y = dt / tauref(Kn, Th, MTEST[1], omega)
            Mout = bgk_relax_tup(M0, dt, Kn, Pr, omega)
            Th1, C1 = theta_cov(collect(Mout))
            Cex = Th*I + exp(-y)*(C0 - Th*I)
            worst = max(worst, maximum(abs, C1 - Cex)/maximum(abs, Cex))
        end
        @info "kappa second-moment exactness: worst rel error" worst
        @test worst < 1e-13
    end

    @testset "Isserlis target vs independent Gauss-Hermite quadrature" begin
        # Mout must equal (1-e)*G[Lambda] + e*M with G built by quadrature.
        worst = 0.0
        for Pr in (2/3, 0.8), omega in (0.5, 0.74), Kn in (0.1, 5.0), dt in (0.02, 0.3)
            Th, C0 = theta_cov(MTEST)
            y = dt / tauref(Kn, Th, MTEST[1], omega)
            e = exp(-Pr*y)
            k = kappa_ref(Pr, y)
            Lam = (1-k)*Th*Matrix(I,3,3) + k*C0
            G  = gaussian_moments_quad(MTEST[1], MTEST[2]/MTEST[1],
                                       MTEST[6]/MTEST[1], MTEST[16]/MTEST[1], Lam)
            expected = (1-e).*G .+ e.*MTEST
            got = collect(bgk_relax_tup(Tuple(MTEST), dt, Kn, Pr, omega))
            worst = max(worst, maximum(abs, got .- expected)/maximum(abs, expected))
        end
        @info "Isserlis vs quadrature: worst rel error" worst
        @test worst < 1e-11
    end

    @testset "kappa limits" begin
        @test kappa_ref(1.0, 1.0) === 0.0                      # bitwise zero
        @test isapprox(kappa_ref(2/3, 1e-12), -0.5; atol = 1e-9)   # -> nu
        @test abs(kappa_ref(2/3, 500.0)) < 1e-100                  # -> 0
        for y in 10.0 .^ (-12:2:6)                                 # stable over decades
            @test isfinite(kappa_ref(2/3, y)) && -0.5 <= kappa_ref(2/3, y) <= 0.0
        end
    end

    @testset "conservation (mass, momentum, trace) at arbitrary Pr/omega" begin
        for Pr in (2/3, 0.75, 1.0), omega in (0.5, 0.74, 1.0), Kn in (1e-3, 1.0, 100.0)
            got = collect(bgk_relax_tup(Tuple(MTEST), 0.05, Kn, Pr, omega))
            @test got[1]  ≈ MTEST[1]  rtol=1e-14           # mass
            @test got[2]  ≈ MTEST[2]  rtol=1e-13           # x-momentum
            @test got[6]  ≈ MTEST[6]  rtol=1e-13
            @test got[16] ≈ MTEST[16] rtol=1e-13
            @test got[3]+got[10]+got[20] ≈ MTEST[3]+MTEST[10]+MTEST[20] rtol=1e-13
        end
    end

    @testset "PD guard and degenerate inputs" begin
        # near-degenerate covariance (one direction nearly cold)
        Ldeg = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1e-10]
        Md = Tuple(gaussian_moments_quad(1.0, 0.0, 0.0, 0.0, Ldeg))
        out = bgk_relax_tup(Md, 0.05, 1.0, 2/3, 0.74)
        @test all(isfinite, out)
        # strongly correlated but still PD
        Lcorr = [1.0 0.95 0.0; 0.95 1.0 0.0; 0.0 0.0 1.0]
        Mc = Tuple(gaussian_moments_quad(1.0, 0.0, 0.0, 0.0, Lcorr))
        @test all(isfinite, bgk_relax_tup(Mc, 0.05, 1.0, 2/3, 0.74))
        # Kn = Inf (no collision) and dt = 0 are no-ops
        @test bgk_relax_tup(Tuple(MTEST), 0.05, Inf, 2/3, 0.74) == Tuple(MTEST)
        @test bgk_relax_tup(Tuple(MTEST), 0.0,  1.0, 2/3, 0.74) == Tuple(MTEST)
        # rho <= 0 short-circuits
        Mz = Tuple(vcat(0.0, MTEST[2:35]))
        @test bgk_relax_tup(Mz, 0.05, 1.0, 2/3, 0.74) == Mz
    end

    @testset "defaults are bitwise the historical operator" begin
        for Kn in (1e-3, 0.1, 1.0, 10.0), dt in (1e-4, 0.01, 0.2)
            a = bgk_relax_tup(Tuple(MTEST), dt, Kn)
            b = bgk_relax_tup(Tuple(MTEST), dt, Kn, 1.0, 0.5)
            @test a === b                                  # bitwise, not approx
        end
    end

    @testset "runner wiring: params reach the operator" begin
        # Guards the params -> simulation_runner -> step_highorder_3d!/collision35
        # threading. Without this a silent break would leave Pr/omega inert while
        # every operator-level test still passed.
        using MPI
        MPI.Initialized() || MPI.Init()
        P(; kw...) = merge((Nx=16, Ny=4, Nz=4, tmax=0.02, Kn=1.0, Ma=0.0, flag2D=0,
            CFL=1/3, Nmom=35, nnmax=100000, dtmax=1000.0, rhol=1.0, rhor=10.0, T=1.0,
            r110=0.0, r101=0.0, r011=0.0, symmetry_check_interval=100000,
            homogeneous_z=true, debug_output=false, snapshot_interval=0,
            ic_type=:riemann1d, spatial_order=2), NamedTuple(kw))
        a = simulation_runner(P())[1]
        b = simulation_runner(P(Pr = 1.0, omega = 0.5))[1]
        c = simulation_runner(P(Pr = 2/3, omega = 0.74))[1]
        @test a == b                       # explicit defaults change nothing
        @test a != c                       # ES mode actually reaches the solver
        @test all(isfinite, c)
        @test_throws ErrorException simulation_runner(P(Pr = 0.4))
        @test_throws ErrorException simulation_runner(P(omega = 1.5))
    end
end

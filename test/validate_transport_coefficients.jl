#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# ACCEPTANCE GATE for ES-BGK + VHS: measure the transport coefficients the
# operator actually produces, and check them against the requested Pr and omega.
#
# Chapman-Enskog for a single-relaxation moment closure gives
#     mu = p * tau_sigma,      k = (5/2) R p * tau_q,      Pr = tau_sigma / tau_q,
# so Pr is the RATIO of the two measured relaxation times and needs no absolute
# normalization. Under ES-BGK the deviatoric stress relaxes at 1/tau_ref while the
# heat flux relaxes at Pr/tau_ref — that decoupling IS how Pr moves off 1.
#
# Both are measured by fitting exponentials to the homogeneous decay of the
# CENTRAL moments (exact for fixed rho,u,Theta: a convex combination of two states
# with the same mean has the convex combination of their central moments). This
# isolates the collision operator from spatial discretization error.
#
# The VHS exponent is then read off mu = rho*Theta*tau_sigma across temperatures:
# omega = dlog(mu)/dlog(Theta).
#
# Run:  julia --project=<env> test/validate_transport_coefficients.jl
# ---------------------------------------------------------------------------
using Printf, LinearAlgebra, Statistics
using Riemann35
using Riemann35.ReconDev: bgk_relax_tup

const IJK35 = [
    (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),
    (0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),
    (3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),
    (2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

function gh(n::Int)
    J = SymTridiagonal(zeros(n), [sqrt(k) for k in 1:n-1]); F = eigen(J)
    F.values, [F.vectors[1,i]^2 for i in 1:n]
end

"35 raw moments of Gaussian(rho, mean, L) by tensor Gauss-Hermite quadrature"
function gauss_moments(rho, u, v, w, L::Matrix{Float64}; n::Int = 7)
    x, ω = gh(n); A = cholesky(Symmetric(L)).L; M = zeros(35)
    @inbounds for a in 1:n, b in 1:n, c in 1:n
        wt = ω[a]*ω[b]*ω[c]
        cx = A[1,1]*x[a]
        cy = A[2,1]*x[a] + A[2,2]*x[b]
        cz = A[3,1]*x[a] + A[3,2]*x[b] + A[3,3]*x[c]
        px, py, pz = u+cx, v+cy, w+cz
        for s in 1:35
            (i,j,k) = IJK35[s]; M[s] += wt * px^i * py^j * pz^k
        end
    end
    rho .* M
end

"central 2nd moments (covariance) and the xx-central-3rd, from raw M"
function centrals(M)
    rho = M[1]; u = M[2]/rho; v = M[6]/rho; w = M[16]/rho
    C200 = M[3]/rho-u*u; C020 = M[10]/rho-v*v; C002 = M[20]/rho-w*w
    C300 = M[4]/rho - 3u*M[3]/rho + 2u^3          # central 3rd, x-direction
    (C200, C020, C002, C300)
end

"least-squares decay rate of a positive, exponentially decaying series"
fitrate(t, f) = -( (length(t)*sum(t.*log.(f)) - sum(t)*sum(log.(f))) /
                   (length(t)*sum(t.^2) - sum(t)^2) )

"""measure (tau_sigma, tau_q) at temperature Theta for given Pr, omega, Kn"""
function measure_taus(Theta, Pr, omega, Kn; rho = 1.0, nsteps = 60)
    # (a) stress relaxation: anisotropic covariance, zero heat flux (a Gaussian)
    aniso = 0.15
    L = Matrix(Theta*I, 3, 3)
    L[1,1] = Theta*(1+aniso); L[2,2] = Theta*(1-aniso)    # traceless perturbation
    Ms = Tuple(gauss_moments(rho, 0.0, 0.0, 0.0, L))
    # (b) heat-flux relaxation: isotropic Gaussian + a small odd (3rd-moment) kick
    Mq = collect(gauss_moments(rho, 0.0, 0.0, 0.0, Matrix(Theta*I, 3, 3)))
    Mq[4] += 0.02 * rho * Theta^1.5        # M300 kick -> nonzero central 3rd
    Mq[11] += 0.01 * rho * Theta^1.5       # M120
    Mq = Tuple(Mq)

    tau_ref0 = (Kn/2) * Theta^(omega-1.0) / rho
    dt = 0.02 * tau_ref0                   # resolve the faster (heat-flux) rate
    ts = Float64[]; sig = Float64[]; qq = Float64[]
    Ss, Sq = Ms, Mq
    for n in 1:nsteps
        Ss = bgk_relax_tup(Ss, dt, Kn, Pr, omega)
        Sq = bgk_relax_tup(Sq, dt, Kn, Pr, omega)
        c = centrals(collect(Ss)); cq = centrals(collect(Sq))
        push!(ts, n*dt)
        push!(sig, abs(c[1] - c[2]))       # deviatoric (C200 - C020)
        push!(qq,  abs(cq[4]))             # central 3rd (heat flux proxy)
    end
    keep = (sig .> 1e-13) .& (qq .> 1e-13)
    tau_sigma = 1 / fitrate(ts[keep], sig[keep])
    tau_q     = 1 / fitrate(ts[keep], qq[keep])
    (tau_sigma, tau_q)
end

function main()
    Kn = 1.0; rho = 1.0
    println("="^78)
    println("ES-BGK + VHS measured transport coefficients")
    println("  Pr_measured = tau_sigma / tau_q     (Chapman-Enskog, monatomic)")
    println("  mu          = rho*Theta*tau_sigma")
    println("="^78)
    @printf("%-8s %-7s %-8s %14s %14s %12s %10s\n",
            "Pr_set", "omega", "Theta", "tau_sigma", "tau_q", "Pr_meas", "rel.err")
    worstPr = 0.0
    for (Pr, omega) in ((1.0,0.5), (2/3,0.5), (2/3,0.74), (0.75,0.74), (2/3,1.0))
        for Theta in (0.5, 1.0, 2.0)
            ts_, tq_ = measure_taus(Theta, Pr, omega, Kn; rho = rho)
            prm = ts_/tq_
            err = abs(prm-Pr)/Pr
            worstPr = max(worstPr, err)
            @printf("%-8.4f %-7.2f %-8.2f %14.6e %14.6e %12.6f %10.2e\n",
                    Pr, omega, Theta, ts_, tq_, prm, err)
        end
    end
    println("-"^78)
    # VHS exponent from mu(Theta) = rho*Theta*tau_sigma
    println("\nVHS exponent  omega_measured = dlog(mu)/dlog(Theta):")
    @printf("%-8s %-10s %14s %14s %12s %10s\n",
            "Pr_set", "omega_set", "mu(Theta=0.5)", "mu(Theta=2.0)", "omega_meas", "rel.err")
    worstOm = 0.0
    for (Pr, omega) in ((1.0,0.5), (2/3,0.5), (2/3,0.74), (0.75,0.9), (2/3,1.0))
        T1, T2 = 0.5, 2.0
        ts1, _ = measure_taus(T1, Pr, omega, Kn; rho = rho)
        ts2, _ = measure_taus(T2, Pr, omega, Kn; rho = rho)
        mu1 = rho*T1*ts1; mu2 = rho*T2*ts2
        om  = log(mu2/mu1)/log(T2/T1)
        err = abs(om-omega)/omega
        worstOm = max(worstOm, err)
        @printf("%-8.4f %-10.2f %14.6e %14.6e %12.6f %10.2e\n",
                Pr, omega, mu1, mu2, om, err)
    end
    println("-"^78)
    @printf("\nWORST relative error:  Pr %.3e    omega %.3e\n", worstPr, worstOm)
    ok = worstPr < 1e-6 && worstOm < 1e-6
    println(ok ? "\nGATE PASS" : "\nGATE FAIL")
    return ok
end

main()

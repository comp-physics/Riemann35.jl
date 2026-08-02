# test_esbgk_offdiagonal.jl -- ES-BGK must relax OFF-DIAGONAL stress like DIAGONAL stress.
#
# THE BUG THIS GUARDS (issue #71). `_collide_es_kernel!` builds its equilibrium as a PRODUCT
# of three 1D fits:
#
#     feq = A*exp(bx*vx + cx*vx^2) * exp(by*vy + cy*vy^2) * exp(bz*vz + cz*vz^2)
#
# A product of 1D Gaussians has a strictly DIAGONAL covariance. The ES-BGK equilibrium is
#
#     Lambda = (1-k) Theta I + k C
#
# whose off-diagonal entries are k*C_ij. Those cannot be represented in product form, so feq
# carried sigma_xy = 0 and the update degenerated to sigma_xy*exp(-Pr*y) -- relaxing the shear
# stress at Pr/tau while the diagonal relaxed at 1/tau. Measured split at Pr = 2/3: 0.666667
# versus 1.000000, both fits R2 = 1.00000.
#
# WHY THE EXISTING TESTS ALL PASSED. Every homogeneous control initialised a DIAGONAL
# anisotropy (TXX=1.5, TYY=TZZ=0.75), so no test ever excited an off-diagonal, and the two
# collision operators were reported as "agreeing to 1.5e-11". They agree on the subspace that
# was probed. This is the same failure mode as the wall flux (PR #72), where two independent
# "CPU and device agree" testsets both passed while the implementations used different
# covariance models -- because every test state had zero off-diagonals. Twice is a pattern:
# ANY test of a tensor quantity must excite the off-diagonal subspace explicitly.
#
# THE INVARIANT, and why it needs no Pr convention. Two conventions exist for tau: the
# collision time (stress relaxes at (1-nu)/tau = (1/Pr)/tau) and the viscous time
# tau_mu = mu/p (stress relaxes at 1/tau by construction, Pr carried by the heat flux). This
# codebase uses the second -- `stage_bgk` relaxes stress at 1/tau, and a Couette core fit
# recovers DSMC's own Kn_mu = lambda_mu/H to 0.8%. But the test below does NOT depend on that
# choice: under EITHER convention, every deviatoric component relaxes at the SAME rate,
# because the deviatoric part of Lambda is k*C_dev with a single scalar k. So the assertion is
# rate(sigma_xy) == rate(sigma_xx), which is convention-free and cannot be argued with.
using Test
using Riemann35
using CUDA
isdefined(Main, :DVMBGKGPU) || include(joinpath(@__DIR__, "..", "gpu", "dvm_bgk_gpu.jl"))
using .DVMBGKGPU
using LinearAlgebra: inv, det

const HAS_CUDA_ES = CUDA.functional()

"Exponential decay rate of a positive series, with R2 so a bad fit cannot pass as a result."
function _rate(ts, ys)
    keep = [i for i in eachindex(ys) if ys[i] > 1e-13]
    length(keep) < 5 && return (NaN, NaN)
    x = ts[keep]; y = log.(abs.(ys[keep]))
    n = length(x); sx = sum(x); sy = sum(y)
    b = (n*sum(x .* y) - sx*sy) / (n*sum(x .^ 2) - sx^2)
    a = (sy - b*sx)/n
    ss = sum((y .- (a .+ b .* x)) .^ 2); st = sum((y .- sy/n) .^ 2)
    (-b, 1 - ss/max(st, 1e-300))
end

if !HAS_CUDA_ES
    @info "CUDA not functional -- skipping ES-BGK off-diagonal test"
else
@testset "ES-BGK relaxes off-diagonal stress like diagonal stress (#71)" begin
    nv, vmax = 32, 6.0
    g   = DVMBGKGPU.VGridG(vmax, nv); vh = g.vh; dv3 = g.dv^3
    tau = 0.1; dt = 0.14*tau; nt = 40

    # A state carrying BOTH a diagonal anisotropy and an off-diagonal. Both are required: the
    # diagonal alone is the blind spot that hid the bug, and the off-diagonal alone leaves
    # nothing to compare against.
    C = [1.4 0.25 0.0; 0.25 0.8 0.0; 0.0 0.0 0.8]
    Ci = inv(C); dtC = det(C)
    h = zeros(nv, nv, nv)
    for c in 1:nv, b in 1:nv, a in 1:nv
        v = (vh[a], vh[b], vh[c]); q = 0.0
        for i in 1:3, j in 1:3; q += v[i]*Ci[i, j]*v[j]; end
        h[a, b, c] = exp(-q/2)/sqrt((2pi)^3*dtC)
    end
    h ./= (sum(h)*dv3)

    for Pr in (1.0, 2/3)
        f = CuArray(reshape(copy(h), nv, nv, nv, 1))
        ts = Float64[]; sxy = Float64[]; sxx = Float64[]
        for n in 0:nt
            H = reshape(Array(f), nv, nv, nv)
            rho = sum(H)*dv3
            xy = sum(H[a,b,c]*vh[a]*vh[b] for a in 1:nv, b in 1:nv, c in 1:nv)*dv3/rho
            xx = sum(H[a,b,c]*vh[a]*vh[a] for a in 1:nv, b in 1:nv, c in 1:nv)*dv3/rho
            yy = sum(H[a,b,c]*vh[b]*vh[b] for a in 1:nv, b in 1:nv, c in 1:nv)*dv3/rho
            zz = sum(H[a,b,c]*vh[c]*vh[c] for a in 1:nv, b in 1:nv, c in 1:nv)*dv3/rho
            push!(ts, n*dt); push!(sxy, abs(xy)); push!(sxx, abs(xx - (xx+yy+zz)/3))
            n < nt && DVMBGKGPU.collide_es!(f, g, dt, 2*tau, Pr, 1.0)
        end
        rxy, r2xy = _rate(ts, sxy)
        rxx, r2xx = _rate(ts, sxx)

        # a fit that is not exponential is not a measurement
        @test r2xy > 0.9999
        @test r2xx > 0.9999

        # THE INVARIANT -- convention-free
        @test isapprox(rxy, rxx; rtol = 1e-6)

        # and, in this codebase's viscous-time convention, both must be 1/tau: `stage_bgk`
        # relaxes stress at 1/tau and a Couette core fit recovers DSMC's Kn_mu to 0.8%, so a
        # Pr-dependent shear rate here would put the two solvers on different viscosities.
        @test isapprox(rxy*tau, 1.0; rtol = 1e-5)
        @test isapprox(rxx*tau, 1.0; rtol = 1e-5)
    end

    # Mass, momentum and energy must survive the correlated equilibrium unchanged -- the
    # discrete-moment fit exists to guarantee this, and a correlated fit must not lose it.
    f = CuArray(reshape(copy(h), nv, nv, nv, 1))
    m0 = sum(Array(f))*dv3
    e0 = sum(Array(f)[a,b,c]*(vh[a]^2 + vh[b]^2 + vh[c]^2)
             for a in 1:nv, b in 1:nv, c in 1:nv)*dv3
    for _ in 1:20; DVMBGKGPU.collide_es!(f, g, dt, 2*tau, 2/3, 1.0); end
    F = Array(f)
    m1 = sum(F)*dv3
    e1 = sum(F[a,b,c]*(vh[a]^2 + vh[b]^2 + vh[c]^2) for a in 1:nv, b in 1:nv, c in 1:nv)*dv3
    @test isapprox(m1, m0; rtol = 1e-12)
    @test isapprox(e1, e0; rtol = 1e-10)
    @test all(F .>= 0.0)                      # positivity: the fit must not manufacture f < 0
end
end

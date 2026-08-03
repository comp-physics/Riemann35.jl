# test_correlated_states.jl -- exercise the tensor-consuming paths on CORRELATED states.
#
# THE RULE THIS ENFORCES: any test of a tensor quantity must excite the OFF-DIAGONAL
# subspace explicitly. A test that only probes diagonal covariances is not a weak test of the
# operator, it is a test of a different operator.
#
# WHY, in three instances found in a single day:
#
#   1. `kfvs_wall_flux` (PR #72) factorised the interior half with a DIAGONAL covariance while
#      the device used a correlated one. Agreement was 5e-16 with diagonal states and up to
#      44% with off-diagonals present. TWO independent "CPU and device agree" testsets passed
#      throughout -- one of 90 assertions -- because every state was diagonal.
#   2. `collide_es!` (PR #73) built the ES equilibrium as a product of three 1D Gaussians,
#      which cannot carry Lambda's off-diagonal entries, so shear stress relaxed at Pr/tau
#      while diagonal stress relaxed at 1/tau. The homogeneous control that "proved" the two
#      collision operators agreed to 1.5e-11 used a diagonal anisotropy and never excited it.
#   3. A survey then found 37 test files building states via
#      InitializeM4_35(..., T,0,0, T,0, T) -- every off-diagonal exactly zero. That is the
#      blind spot both bugs lived in, and it is essentially the whole suite.
#
# This file does not attempt to convert those 37 files. It covers the paths that actually
# consume the full second-moment tensor, with states whose off-diagonals are large enough to
# matter, asserting invariants that must hold whatever the correlation.
using Test, Riemann35, LinearAlgebra

"Realizable 35-moment state; `f` scales the off-diagonals (f = 0 is what most tests build)."
function _corr_state(f::Float64; rho = 1.0, T = 1.0)
    c110 = f*0.30*T; c101 = f*0.22*T; c011 = f*-0.26*T
    C = [T c110 c101; c110 T c011; c101 c011 T]
    @assert minimum(eigvals(C)) > 0 "off-diagonal scale f=$f makes the covariance indefinite"
    collect(Float64, InitializeM4_35(rho, 0.15, -0.1, 0.08, T, c110, c101, T, c011, T))
end

@testset "correlated states: tensor-consuming paths" begin
    M0 = _corr_state(0.0)     # diagonal, i.e. what the rest of the suite uses
    Mc = _corr_state(1.0)     # correlated

    # the two states must genuinely differ, or everything below is vacuous
    @test count(i -> abs(M0[i] - Mc[i]) > 1e-14, 1:35) > 10

    # ---- KFVS wall passes zero net mass whatever the correlation ---------------------------
    for ax in 1:3, ow in (1.0, -1.0), M in (M0, Mc)
        _, _, mdot = kfvs_wall_flux(M, ax, ow, 1.0, 0.1, -0.05)
        @test abs(mdot) < 1e-12
    end

    # and the wall flux must actually RESPOND to the off-diagonals -- this is the assertion
    # whose absence let #72 live: the tangential momentum flux is carried by C_nt.
    Fd, _, _ = kfvs_wall_flux(M0, 2, +1.0, 1.0, 0.0, 0.0)
    Fc, _, _ = kfvs_wall_flux(Mc, 2, +1.0, 1.0, 0.0, 0.0)
    @test maximum(abs, collect(Fd) .- collect(Fc)) > 1e-6

    # ---- a uniform body force is a rigid velocity-space translation ------------------------
    # central moments, off-diagonals included, must come through untouched.
    for M in (M0, Mc)
        A = body_force_shift(M, 0.1, -0.05, 0.02, 1e-3)
        r0 = M[1]; rA = A[1]
        for (i, j, k) in ((7, 2, 6), (17, 2, 16), (26, 6, 16))     # C_xy, C_xz, C_yz
            c0 = M[i]/r0 - (M[j]/r0)*(M[k]/r0)
            cA = A[i]/rA - (A[j]/rA)*(A[k]/rA)
            @test abs(cA - c0) < 1e-12
        end
    end

    # ---- reconstruction round trip is lossless with correlation present --------------------
    for M in (M0, Mc)
        B = from_recon_vars(to_recon_vars(M))
        @test maximum(abs, (B .- M) ./ (abs.(M) .+ 1e-12)) < 1e-13
    end

    # ---- wave speeds are MARGINAL quantities, and that is correct --------------------------
    # Recorded because it looks like a blind spot and is not: the axis-n speed is built from
    # the axis-n marginal moments, and integrating out the transverse components removes
    # C_nt entirely. So the axis-1 speed is bit-identical between a diagonal and a correlated
    # state BY CONSTRUCTION. Anyone auditing for off-diagonal insensitivity will land here;
    # the assertion below states the reason so it is not "fixed" into a bug.
    @test all(abs(M0[i] - Mc[i]) < 1e-14 for i in Riemann35.MARG_VEC[1])
    for ax in 1:3
        s0 = Riemann35.realize_and_speed(M0, ax, 5.0)
        sc = Riemann35.realize_and_speed(Mc, ax, 5.0)
        v0 = s0 isa Tuple ? last(s0) : s0
        vc = sc isa Tuple ? last(sc) : sc
        @test isfinite(v0) && isfinite(vc)
        @test isapprox(Float64(v0), Float64(vc); rtol = 1e-12)
    end
end

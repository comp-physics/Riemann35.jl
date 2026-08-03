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

"""
Central moments of a zero-mean Gaussian with covariance `S`, straight from Isserlis
(`E[xi xj xk xl] = sij*skl + sik*sjl + sil*sjk`), packed in the 35-vector ordering.
Shares no code path with `S4toC4_3D_r`, which is what makes the comparison meaningful.
"""
function _isserlis_central(S)
    s(i, j) = S[i, j]
    C = zeros(35)
    C[1] = 1.0
    C[3] = s(1,1); C[7]  = s(1,2); C[17] = s(1,3)
    C[10] = s(2,2); C[26] = s(2,3); C[20] = s(3,3)
    # 3rd order is identically zero for a Gaussian; leave it.
    C[5]  = 3*s(1,1)^2                       # C400
    C[9]  = 3*s(1,1)*s(1,2)                  # C310
    C[19] = 3*s(1,1)*s(1,3)                  # C301
    C[12] = s(1,1)*s(2,2) + 2*s(1,2)^2       # C220
    C[28] = s(1,1)*s(2,3) + 2*s(1,2)*s(1,3)  # C211
    C[22] = s(1,1)*s(3,3) + 2*s(1,3)^2       # C202
    C[14] = 3*s(2,2)*s(1,2)                  # C130
    C[30] = s(2,2)*s(1,3) + 2*s(1,2)*s(2,3)  # C121
    C[33] = s(3,3)*s(1,2) + 2*s(1,3)*s(2,3)  # C112
    C[24] = 3*s(3,3)*s(1,3)                  # C103
    C[15] = 3*s(2,2)^2                       # C040
    C[31] = 3*s(2,2)*s(2,3)                  # C031
    C[35] = s(2,2)*s(3,3) + 2*s(2,3)^2       # C022
    C[34] = 3*s(3,3)*s(2,3)                  # C013
    C[25] = 3*s(3,3)^2                       # C004
    C
end

# ---------------------------------------------------------------------------------------
# InitializeM4_35 IS a correlated Gaussian -- the counterpart to the bugs above.
#
# The file header lists three places where correlation was silently dropped. This testset
# records the mirror-image error: documentation that accused CORRECT code of the same
# thing. `InitializeM4_35` passes the isotropic standardized moments (S400=3, S220=1, rest
# 0) to `S4toC4_3D_r`, which looks like an independent-Gaussian construction and was
# written up as one. It is not: `S4toC4_3D_r` forms `A = sqrtm(C2)` and applies the change
# of variables `v = A*xi`, so the isotropic set means `xi ~ N(0,I)` and the output is
# exactly `v ~ N(0,C2)`. (`S4toC4_3D`, no `_r`, is the pure rescale the old warning
# described -- but nothing calls it here.) Rodney Fox caught the error on 2026-08-03.
#
# Both failure directions cost the same thing, so both get the same fix: compare a
# correlated construction against an INDEPENDENT reference. Isserlis is that reference.
# ---------------------------------------------------------------------------------------
@testset "InitializeM4_35 is Gaussian for correlated covariance too" begin
    cases = (
        ("diagonal", [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
        ("weak",     [1.0 0.10 0.05; 0.10 1.0 -0.08; 0.05 -0.08 1.0]),
        ("strong",   [1.0 0.30 0.22; 0.30 1.0 -0.26; 0.22 -0.26 1.0]),
        ("aniso",    [2.0 0.55 -0.40; 0.55 0.8 0.19; -0.40 0.19 1.7]),
        ("stiff",    [1.0 0.85 0.10; 0.85 1.0 0.10; 0.10 0.10 0.5]),
    )
    for (name, S) in cases
        @test minimum(eigvals(S)) > 0            # a non-PD case would test nothing
        M = InitializeM4_35(1.0, 0.15, -0.1, 0.08,
                            S[1,1], S[1,2], S[1,3], S[2,2], S[2,3], S[3,3])
        Cnum, _ = M2CS4_35(collect(Float64, M))
        Cref = _isserlis_central(S)
        # absolute floor on the scale: most 3rd-order entries are exact zeros
        worst = maximum(abs(Cnum[i] - Cref[i]) / max(abs(Cref[i]), 1e-3) for i in 2:35)
        @test worst < 1e-12
    end

    # ---- the tolerance above must be able to SEE the defect it rules out ---------------
    # Otherwise this testset is the same kind of vacuous pass the file header is about.
    # `_plain_rescale_central` is the MATLAB `S4toC4_3D` semantics -- the routine the
    # retracted warning actually described: standardized moments scaled by the marginal
    # standard deviations, with no change of variables. Feeding it the isotropic set
    # leaves every 4th-order cross moment at its independent-Gaussian value, so it
    # disagrees with Isserlis by exactly the correlation terms (C220 short by 2*C110^2,
    # C310 short by 3*C200*C110, ...). Asserting that gap is large is what proves the
    # `< 1e-12` gate above is discriminating rather than merely satisfiable.
    function _plain_rescale_central(S)
        sx = sqrt(S[1,1]); sy = sqrt(S[2,2]); sz = sqrt(S[3,3])
        C = zeros(35)
        C[1] = 1.0
        C[3] = S[1,1]; C[7]  = S[1,2]; C[17] = S[1,3]
        C[10] = S[2,2]; C[26] = S[2,3]; C[20] = S[3,3]
        # isotropic standardized 4th order: S400=S040=S004=3, S220=S202=S022=1, rest 0
        C[5]  = 3*sx^4; C[15] = 3*sy^4; C[25] = 3*sz^4
        C[12] = sx^2*sy^2; C[22] = sx^2*sz^2; C[35] = sy^2*sz^2
        C   # every other entry stays 0 -- that is the whole defect
    end

    for (name, S) in cases
        gap = maximum(abs(_isserlis_central(S)[i] - _plain_rescale_central(S)[i]) for i in 2:35)
        if name == "diagonal"
            @test gap < 1e-14        # the two routines agree exactly when C is diagonal
        else
            @test gap > 1e-2         # ...and diverge grossly the moment it is not
        end
    end
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

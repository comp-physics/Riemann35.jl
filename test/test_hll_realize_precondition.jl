# test_hll_realize_precondition.jl -- the realizability pass in `_hll_states` is a
# PRECONDITION, not duplicated work (issue #39).
#
# `_hll_states` calls `realizable_3D_M4_dev` on each input and then
# `realize_and_speed_Mr_dev` on the result. Deleting the first is 1.32x on the HLL kernel,
# and on a smooth already-realizable field the answer moves by only 1.33e-15 -- which reads
# like the second pass subsumes the first. It does not.
#
# They are DIFFERENT operations. `realizable_3D_M4_dev` enforces the MOMENT CONE.
# `realize_and_speed_Mr_dev` enforces HYPERBOLICITY, and only on the complex-eigenvalue
# branch; on the real branch it returns the input verbatim, with no realizability
# enforcement at all. Outside the cone its tridiagonal Q4 solver takes sqrt of a negative
# recurrence coefficient.
#
# Measured over 2000 randomly perturbed states per row, `Ma = 1`, `s3f = 40`:
#
#     perturbation   sqrt(negative) without the pass   states differing >1e-10   worst
#     0.02              0                                   0                    7.4e-16
#     0.05              0                                   0                    6.9e-16
#     0.20              0                                 156                    9.5e-01
#     0.60            307  (15.3%)                       1387                    1.6e+00
#
# The near-equilibrium rows are what the original 1.33e-15 observation saw. The perturbed
# rows are the ones that matter, and they are not a corner case: WENO-reconstructed face
# states and halo cells -- which `_proj_interior!` never projects -- are exactly where
# large departures live.
#
# AND IT WOULD BE SILENT. On the CPU sqrt(negative) raises DomainError, which is how this
# was found. In a CUDA kernel there are no exceptions: it returns NaN, and the NaN enters
# the HLL diffusion coefficient.
using Test
using Riemann35
using Riemann35.RealizeDev: realizable_3D_M4_dev
using Riemann35.WavespeedDev: realize_and_speed_Mr_dev

# A state captured from that sweep: realizable-projected it is fine, raw it is not.
const M_OUTSIDE_CONE = [1, -0.41494667882537839, -0.11844438994107913, 0.19358869883526308, 3.545900116956374, 0.0039798678302329559, 0.014042849317266237, -0.036306801246032097, 0.013060674931098819, 0.71961016219987839, -0.63146063849890777, 0.93574199222145515, -0.051423255924699415, 0.029764018203181052, -0.32004171284760319, 0.04917004069398348, 0.12127387824107499, -0.31501618086006655, 0.21621912024855111, 0.6626994997683856, -0.2220143347761995, 0.43969041996124431, -0.52628683622488537, 0.10852731686809002, 6.1171074795831508, 0.0067907491653503458, -0.00075677418386342306, 0.0025312964648214488, -0.12973962707877484, 0.077943010961196352, 0.013788806057799929, -0.045125182307847581, 0.019781362756812481, 0.014408205820269412, 1.1163444227348505]

@testset "HLL realizability pass is a precondition (issue #39)" begin
    Ma, s3f, axis = 1.0, 40.0, 1

    @testset "with the pass: succeeds and is finite" begin
        Mr, lo, hi = realize_and_speed_Mr_dev(
            realizable_3D_M4_dev(M_OUTSIDE_CONE..., Ma, s3f)..., axis, Ma)
        @test all(isfinite, Mr)
        @test isfinite(lo) && isfinite(hi)
        @test lo <= hi
    end

    @testset "without the pass: fails -- so the pass cannot be deleted" begin
        # DomainError here; in a CUDA kernel the same sqrt returns NaN with no error.
        @test_throws DomainError realize_and_speed_Mr_dev(M_OUTSIDE_CONE..., axis, Ma)
    end

    @testset "the pass really is a no-op on an already-realizable state" begin
        # this is why the original 1.33e-15 measurement looked like redundancy
        T = 1.0
        M = collect(Float64, InitializeM4_35(1.0, 0.1, -0.05, 0.02, T,0,0, T,0, T))
        a = realize_and_speed_Mr_dev(realizable_3D_M4_dev(M..., Ma, s3f)..., axis, Ma)[1]
        b = realize_and_speed_Mr_dev(M..., axis, Ma)[1]
        @test maximum(abs.(collect(a) .- collect(b))) < 1e-12
    end
end

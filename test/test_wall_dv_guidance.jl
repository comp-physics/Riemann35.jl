# test_wall_dv_guidance.jl -- the wall velocity-resolution criterion (issue #59).
#
# The number this pins is measured, not chosen: at Kn = 0.73 against 8-seed DSMC the
# Knudsen-layer curvature runs 1.160 / 1.296 / 1.348 / 1.366 / 1.375 for
# dv = 0.387 / 0.255 / 0.190 / 0.126 / 0.094, against DSMC's 1.426 +/- 0.050. Only
# dv <~ 0.13 reaches the reference's own noise floor. If WALL_DV_MAX drifts away from
# that, the guidance stops matching the evidence behind it.
using Test
using Riemann35
# Include the DVM module only once: gpu/runtests_gpu.jl runs several files that all need
# it, and re-including replaces the module ("WARNING: replacing module DVMBGKGPU"),
# which would let a stale definition survive silently. Works standalone too.
isdefined(Main, :DVMBGKGPU) || include(joinpath(@__DIR__, "..", "gpu", "dvm_bgk_gpu.jl"))
using .DVMBGKGPU

@testset "wall velocity-grid guidance" begin
    @test DVMBGKGPU.WALL_DV_MAX ≈ 0.13

    @testset "the measured ladder is classified as measured" begin
        # (nv, vmax) -> dv, and whether it reached the DSMC noise floor
        for (nv, vmax, adequate) in ((32, 6.0, false),   # 5.3 SEM off -- 18% low on the layer
                                     (48, 6.0, false),   # 2.6 SEM
                                     (64, 6.0, false),   # 1.5 SEM
                                     (96, 6.0, true),    # 1.2 SEM
                                     (128, 6.0, true))   # 1.0 SEM
            dv = 2vmax/(nv-1)
            @test DVMBGKGPU.wall_dv_ok(dv) == adequate
        end
    end

    @testset "the criterion is dv, not vmax" begin
        # matched-dv pair measured to agree to 0.1% despite a 33% wider velocity domain
        @test DVMBGKGPU.wall_dv_ok(2*6.0/63) == DVMBGKGPU.wall_dv_ok(2*8.0/85)
    end

    @testset "it scales with thermal speed" begin
        # the criterion is dv/sqrt(T): a hotter gas tolerates a coarser grid
        @test  DVMBGKGPU.wall_dv_ok(0.20; T = 4.0)     # 0.20/2 = 0.10 <= 0.13
        @test !DVMBGKGPU.wall_dv_ok(0.20; T = 1.0)
    end

    @testset "a VGridG reports the dv it will be judged on" begin
        g = DVMBGKGPU.VGridG(6.0, 128)
        @test g.dv ≈ 2*6.0/127
        @test DVMBGKGPU.wall_dv_ok(g.dv)
        @test !DVMBGKGPU.wall_dv_ok(DVMBGKGPU.VGridG(6.0, 32).dv)
    end
end

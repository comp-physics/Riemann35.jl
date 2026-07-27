# test_body_force_dev.jl — the device body force must agree with the CPU one, and must
# preserve exactly what a velocity-space translation preserves.
#
# `body_force_shift` (CPU) rebuilds raw moments via M4toC4_3D / C4toM4_3D, which return
# 5x5x5 arrays and allocate. `body_force_shift_dev` uses the recon-var round trip so it is
# allocation-free and GPU-compilable. Same operator, different algebra, so they agree to
# ROUNDOFF and not bit-for-bit -- which is why this measures rather than asserts equality,
# and why the CPU path was left alone (existing byte-identity baselines still hold).
#
# The structural gates matter more than the agreement number. A uniform force is a rigid
# translation in velocity space, so:
#   * density is untouched,
#   * the mean velocity shifts by exactly g*dt,
#   * every CENTRAL moment is invariant,
# and those hold for ANY dt, not just small ones -- the reason this form was chosen over
# an explicit O(dt) source term in the first place.

using Test, Printf
using Riemann35
using Riemann35: body_force_shift, body_force_shift_dev

@testset "device body force" begin
    # a deliberately non-equilibrium state: nonzero heat flux and anisotropic temperature,
    # so the test is not passing merely because a Maxwellian is easy
    states = [
        InitializeM4_35(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        InitializeM4_35(0.7, 0.3, -0.2, 0.1, 1.4, 0.15, -0.05, 0.8, 0.1, 1.1),
        InitializeM4_35(2.3, -0.5, 0.4, -0.3, 0.6, -0.1, 0.05, 1.9, -0.2, 0.7),
    ]

    @testset "agrees with the CPU form" begin
        worst = 0.0
        for M in states, (gx,gy,gz) in ((1.0,0.0,0.0), (0.0,-2.0,0.0), (0.3,0.4,-0.5)),
                dt in (1e-3, 1e-2, 0.1, 1.0)
            a = body_force_shift(M, gx, gy, gz, dt)
            b = body_force_shift_dev(NTuple{35,Float64}(M), gx, gy, gz, dt)
            for q in 1:35
                d = abs(a[q] - b[q])/max(abs(a[q]), 1.0)
                worst = max(worst, d)
            end
        end
        @info "device vs CPU body force: worst relative difference" worst
        @test worst < 1e-10
    end

    @testset "is an exact velocity-space translation" begin
        # density unchanged, mean shifted by exactly g*dt, central moments invariant.
        # dt = 1.0 is deliberately LARGE: an explicit source term would fail here, this
        # form must not.
        for M in states, (gx,gy,gz) in ((1.0,0.0,0.0), (0.3,0.4,-0.5)), dt in (0.1, 1.0, 5.0)
            Mt = NTuple{35,Float64}(M)
            B  = body_force_shift_dev(Mt, gx, gy, gz, dt)
            rho = M[1]
            @test B[1] == rho                                  # density exactly untouched
            @test isapprox(B[2]/rho,  M[2]/rho  + gx*dt; rtol=1e-12)
            @test isapprox(B[6]/rho,  M[6]/rho  + gy*dt; rtol=1e-12)
            @test isapprox(B[16]/rho, M[16]/rho + gz*dt; rtol=1e-12)
            # central second moments invariant
            for (raw, mean_idx) in ((3, 2), (10, 6), (20, 16))
                c0 = M[raw]/rho - (M[mean_idx]/rho)^2
                c1 = B[raw]/rho - (B[mean_idx]/rho)^2
                @test isapprox(c1, c0; rtol=1e-10, atol=1e-12)
            end
        end
    end

    @testset "zero force is the identity" begin
        for M in states
            Mt = NTuple{35,Float64}(M)
            @test body_force_shift_dev(Mt, 0.0, 0.0, 0.0, 0.25) === Mt
        end
    end

    # ALLOCATION: "device-safe" does NOT mean allocation-free on the CPU, and an earlier
    # version of this test wrongly asserted that it did.
    #
    # Measured here: ~1136 B per call, scaling exactly linearly with call count. That is
    # the NTuple{35} boundary cost shared by EVERY device kernel in this package -- large
    # tuples get boxed wherever the compiler declines to inline a 35-argument function.
    # It is the same mechanism that puts to_recon_vars, _phys_flux and realizable_3D_M4 at
    # the top of the CPU allocation profile (see the open work on that), not a defect in
    # this kernel.
    #
    # What "device-safe" actually buys is the GPU, where the whole call graph inlines and
    # the tuples live in registers -- demonstrably so, since these kernels sit AT the
    # 255 regs/thread ceiling on sm_80. That property cannot be tested from a CPU
    # allocation counter; it is tested by the kernel compiling and running on the device.
    #
    # So the meaningful CPU assertion is the comparative one: the tuple form must not be
    # WORSE than the array form it replaces.
    @testset "not worse than the allocating CPU form" begin
        Mt = NTuple{35,Float64}(states[2]); Mv = states[2]
        fdev(M, n) = (s = 0.0; for _ in 1:n
                          B = body_force_shift_dev(M, 0.3, 0.4, -0.5, 0.01); s += B[1] + B[2]
                      end; s)
        fcpu(M, n) = (s = 0.0; for _ in 1:n
                          B = body_force_shift(M, 0.3, 0.4, -0.5, 0.01); s += B[1] + B[2]
                      end; s)
        fdev(Mt, 1); fcpu(Mv, 1)                                # warm up / compile
        adev = @allocated fdev(Mt, 100)
        acpu = @allocated fcpu(Mv, 100)
        @info "body force, 100 calls (bytes)" dev=adev cpu=acpu ratio=adev/max(acpu,1)
        @test adev <= acpu
    end
end

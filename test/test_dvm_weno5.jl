# test_dvm_weno5.jl -- the WENO5 DVM transport delivers its order, and the first-order path
# does not. Both are asserted: a high-order scheme that silently falls back to first order
# is worse than no high-order scheme, because it looks like a reference.
#
# The transport substep is linear scalar advection at constant speed per velocity node, so
# for a periodic domain the EXACT solution after time t is a rigid shift of the initial
# profile. That gives an error with no reference solver in it at all.
using Test, CUDA
include(joinpath(@__DIR__, "..", "gpu", "dvm_bgk_gpu.jl")); using .DVMBGKGPU

CUDA.functional() || (@info "no CUDA device; skipping DVM WENO5 tests"; exit(0))

const NV = 8                    # velocity resolution is irrelevant here: transport is
const VMAX = 3.0                # elementwise in the velocity index

"L1 error of pure advection of a smooth periodic profile over one full period."
function advect_err(nx::Int, scheme::Symbol)
    g = DVMBGKGPU.VGridG(VMAX, NV)
    dx = 1.0/nx
    # profile is smooth and well resolved so WENO's nonlinear weights stay near-optimal
    prof = [1.0 + 0.5*sinpi(2*((j-0.5)*dx)) for j in 1:nx]
    H = zeros(NV,NV,NV,nx)
    for j in 1:nx, c in 1:NV, b in 1:NV, a in 1:NV; H[a,b,c,j] = prof[j]; end
    f = CuArray(H)
    # advect for exactly one period at the speed of the node we will read back
    node = findfirst(x -> x > 0.5, g.vh); s = g.vh[node]
    tend = 1.0/s
    dt = 0.4*dx/VMAX; nst = max(8, ceil(Int, tend/dt)); dt = tend/nst
    if scheme === :weno5
        u1, u2, du = DVMBGKGPU.dvm_alloc_weno5(f)
        for _ in 1:nst; DVMBGKGPU.transport_weno5!(f, u1, u2, du, dt, dx, g; bc=:periodic); end
    else
        fn = similar(f)
        for _ in 1:nst; DVMBGKGPU.transport_upwind!(f, fn, dt, dx, g; bc=:periodic); end
    end
    A = Array(f)
    sum(abs, A[node,1,1,:] .- prof)/nx       # exact solution is the initial profile
end

@testset "DVM WENO5 transport" begin
    @testset "advection of a smooth periodic profile converges at high order" begin
        e128 = advect_err(128, :weno5)
        e256 = advect_err(256, :weno5)
        p = log2(e128/e256)
        @info "WENO5 observed order (one period, SSP-RK3, dt ~ dx)" e128 e256 order=p
        @test e256 < e128
        # SSP-RK3 with dt proportional to dx caps the achievable rate at 3; anything at or
        # above 2.5 confirms the spatial reconstruction is live and not degenerating.
        @test p > 2.5
    end

    @testset "it is decisively more accurate than the first-order path" begin
        ew = advect_err(128, :weno5)
        eu = advect_err(128, :upwind)
        @info "one period at nx=128" weno5=ew upwind=eu ratio=eu/ew
        @test ew < 0.05*eu
    end

    @testset "first-order upwind is confirmed first-order" begin
        # asserted so the MOTIVATION for this scheme cannot silently stop being true
        p = log2(advect_err(128, :upwind)/advect_err(256, :upwind))
        @info "upwind observed order" order=p
        @test 0.6 < p < 1.4
    end

    @testset "a constant profile is preserved exactly" begin
        g = DVMBGKGPU.VGridG(VMAX, NV); nx = 32; dx = 1.0/nx
        f = CUDA.fill(1.0, NV,NV,NV,nx)
        u1, u2, du = DVMBGKGPU.dvm_alloc_weno5(f)
        for _ in 1:20; DVMBGKGPU.transport_weno5!(f, u1, u2, du, 0.1*dx, dx, g; bc=:periodic); end
        @test maximum(abs, Array(f) .- 1.0) < 1e-12
    end

    @testset "mass is conserved to roundoff" begin
        g = DVMBGKGPU.VGridG(VMAX, NV); nx = 64; dx = 1.0/nx
        H = zeros(NV,NV,NV,nx)
        for j in 1:nx, c in 1:NV, b in 1:NV, a in 1:NV
            H[a,b,c,j] = 1.0 + 0.3*sinpi(2*((j-0.5)*dx))
        end
        f = CuArray(H); m0 = sum(Array(f))
        u1, u2, du = DVMBGKGPU.dvm_alloc_weno5(f)
        for _ in 1:50; DVMBGKGPU.transport_weno5!(f, u1, u2, du, 0.2*dx/VMAX, dx, g; bc=:periodic); end
        @test abs(sum(Array(f))/m0 - 1) < 1e-12
    end
end

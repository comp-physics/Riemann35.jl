# Halo refill must be a FUNCTION of its input for every interior extent, including n < g.
#
# The periodic wrap used to be `i + n`, which reaches an interior cell only when the interior
# extent along that axis is at least the halo width g=8. On a thinner axis it reached another
# GHOST cell that a concurrent thread in the same launch was still writing -- an inter-block
# read-write race on global memory. Two identical marches then disagreed, and the disagreement
# grew: on an expansion into vacuum it reached 3.3e-2 over 135 steps from a ~1e-9 seed.
#
# The failure is invisible to compute-sanitizer: memcheck and initcheck are clean (in-bounds,
# initialised) and racecheck only covers shared memory. So it is pinned HERE, behaviourally:
# march twice from a bit-identical start and demand bitwise-equal interiors. Bitwise, not
# approximate -- the property under test is reproducibility, and 1 ulp already breaks it.
#
# n = 4 and 6 are below g and reproduce the original bug; 8 and 12 are at and above it and
# always passed. Outflow is included as the negative control: it clamps to an interior index,
# so it was never affected and must stay unaffected.
using Test, CUDA

if !CUDA.functional()
    @info "test_halo_wrap_determinism: no GPU, skipping"
else
    include(joinpath(@__DIR__, "..", "gpu", "timestep3d_order3_gpu.jl"))
    using .Timestep3DOrder3GPU
    using Riemann35: InitializeM4_35

    const _G = 8

    "Two identical marches from one initial cube; return the number of differing interior words."
    function _interior_bit_diff(nt::Int, bc::Symbol; nx::Int = 32, nstep::Int = 4)
        dx = 1.0 / nx
        # A sharp jump makes the halo content actually matter: on a uniform field every ghost
        # equals every interior cell and the race is benign, so a smooth test would pass even
        # against the broken wrap.
        M0 = zeros(Float64, 35, nx, nt, nt)
        for k in 1:nt, j in 1:nt, i in 1:nx
            rho = (i - 0.5) * dx < 0.5 ? 1.0 : 1e-6
            M0[:, i, j, k] .= collect(Float64,
                InitializeM4_35(rho, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
        end
        G = build_haloed_cube(CuArray(M0))
        dt = 0.2 * dx / 6.0
        A = copy(G); B = copy(G)
        for H in (A, B)
            march3d_order3_gpu!(H, dx, 0.0, nstep; dts = fill(dt, nstep), s3max = 40.0,
                                stage_bgk = false, Kn = Inf, bc = bc, idp = true)
        end
        CUDA.synchronize()
        sl(X) = reinterpret(UInt64, vec(Array(X)[:, _G+1:_G+nx, _G+1:_G+nt, _G+1:_G+nt]))
        count(sl(A) .!= sl(B))
    end

    @testset "halo refill is deterministic for any interior extent" begin
        @testset "periodic, transverse n=$nt (n<g: regression)" for nt in (4, 6)
            @test _interior_bit_diff(nt, :periodic) == 0
        end
        @testset "periodic, transverse n=$nt (n>=g: always held)" for nt in (8, 12)
            @test _interior_bit_diff(nt, :periodic) == 0
        end
        @testset "outflow control, transverse n=$nt" for nt in (4, 8)
            @test _interior_bit_diff(nt, :copy) == 0
        end
    end

    @testset "_wrap always lands on an interior cell" begin
        # Exhaustive over the ghost range, for extents both below and above the halo.
        for n in 1:20, i in 1:(2*_G + n)
            s = Timestep3DOrder3GPU._wrap(i, n, _G)
            @test _G + 1 <= s <= _G + n
        end
        # And it must reduce to the old expressions wherever those were valid (n >= g), so no
        # existing result can move.
        for n in _G:20
            for i in 1:_G;                 @test Timestep3DOrder3GPU._wrap(i, n, _G) == i + n; end
            for i in (_G+n+1):(2*_G+n);    @test Timestep3DOrder3GPU._wrap(i, n, _G) == i - n; end
            for i in (_G+1):(_G+n);        @test Timestep3DOrder3GPU._wrap(i, n, _G) == i;     end
        end
    end

    @testset "wall BC rejects an interior thinner than the halo" begin
        dx = 1.0 / 32
        M0 = zeros(Float64, 35, 32, 4, 4)
        for k in 1:4, j in 1:4, i in 1:32
            M0[:, i, j, k] .= collect(Float64,
                InitializeM4_35(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
        end
        G = build_haloed_cube(CuArray(M0))
        # :channel puts walls on y, whose extent (4) is below the halo (8).
        @test_throws ErrorException march3d_order3_gpu!(G, dx, 0.0, 1; dts = [1e-4],
            s3max = 40.0, stage_bgk = false, Kn = Inf, bc = :channel, idp = true)
    end
end

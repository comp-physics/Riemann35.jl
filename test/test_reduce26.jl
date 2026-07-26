# test_reduce26.jl — the opt-in 26-moment reduction (Rodney Fox).
# CPU-only checks that run anywhere (no GPU). GPU byte-parity is checked separately
# by gpu/test_reduce26_gpu.jl on a CUDA node.
using Test, Riemann35, Random, LinearAlgebra

const _R26_TRIP = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
                   (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
                   (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
                   (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]
# the nine ODD fourth-order moments the reduction closes (canonical index)
const _R26_ODD9 = [9,14,19,24,28,30,31,33,34]
# a random realizable state: moments of a positive-weight node cloud
function _r26_state(rng; npts=6)
    x=randn(rng,npts); y=randn(rng,npts); z=randn(rng,npts); w=rand(rng,npts); w./=sum(w)
    [sum(w .* x.^i .* y.^j .* z.^k) for (i,j,k) in _R26_TRIP]
end

@testset "26-moment reduction (reduce26)" begin
    @test Riemann35.REDUCE26[] == false        # default off => byte-identical evolution

    rng = MersenneTwister(1234)
    for _ in 1:200
        M = _r26_state(rng); M[1] > 1e-8 || continue
        R = Riemann35.reduce26_moments(M)
        # (1) only the nine odd fourth-order moments change
        changed = findall(k -> abs(R[k]-M[k]) > 1e-11*max(1,abs(M[k])), 1:35)
        @test issubset(Set(changed), Set(_R26_ODD9))
        # (2) all moments up to and including the even set are untouched (mass/momentum/
        #     energy and every kept moment): everything outside ODD9 is unchanged
        for k in 1:35
            k in _R26_ODD9 && continue
            @test isapprox(R[k], M[k]; atol=1e-11, rtol=1e-11)
        end
        # (3) idempotent: the reduced state is on the reduced manifold
        @test norm(Riemann35.reduce26_moments(R) .- R) < 1e-10
    end
end

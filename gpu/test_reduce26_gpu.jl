# test_reduce26_gpu.jl — GPU byte-parity check for the 26-moment reduction.
# The device function reduce26_relax_tup (host-callable pure scalar) must reproduce the
# CPU Riemann35.reduce26_moments to ~machine precision on random realizable states.
# Run on a CUDA node:  julia --project=gpu/gpuenv2 gpu/test_reduce26_gpu.jl
using Riemann35, Random, LinearAlgebra, Printf
include(joinpath(@__DIR__, "timestep3d_order3_gpu.jl"))
using .Timestep3DOrder3GPU
const rlt = Timestep3DOrder3GPU.reduce26_relax_tup

const TRIP = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
              (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
              (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
              (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]
function randM(rng; npts=6)
    x=randn(rng,npts); y=randn(rng,npts); z=randn(rng,npts); w=rand(rng,npts); w./=sum(w)
    [sum(w .* x.^i .* y.^j .* z.^k) for (i,j,k) in TRIP]
end
function main()
    rng = MersenneTwister(7); worst = 0.0; nn = 0
    for _ in 1:5000
        M = randM(rng); M[1] <= 1e-8 && continue; nn += 1
        a = Riemann35.reduce26_moments(M)                 # CPU reference
        b = collect(rlt(ntuple(m -> M[m], Val(35))))      # GPU device fn (host-called)
        worst = max(worst, maximum(abs.(a .- b)) / max(1e-30, maximum(abs.(a))))
    end
    @printf("states=%d  worst rel |CPU-GPU| = %.3e  %s\n", nn, worst,
            worst < 1e-10 ? "GPU_PASS" : "GPU_FAIL")
end
main()

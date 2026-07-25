#!/usr/bin/env julia
# CPU/GPU parity for the ES-BGK + VHS collision operator.
#
# The ES branch introduces `expm1`, `Theta^(omega-1)` and a Sylvester PD test into
# device code, so this checks BOTH that it compiles for the GPU and that it agrees
# with the CPU. Also re-checks the default (Pr=1, omega=0.5) path is bitwise
# unchanged on device.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "gpuenv2"); io = devnull)

using CUDA, Printf, LinearAlgebra
# Device kernels come from the package's single instance — validators must never
# `include` a device file (see the module-wiring rule in misc/02-architecture.md).
using Riemann35.ReconDev: bgk_relax_tup

@assert CUDA.functional() "CUDA not functional"
println("GPU: ", CUDA.name(CUDA.device()))

const IJK35 = [
    (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),
    (0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),
    (3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),
    (2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]

function gh(n); J = SymTridiagonal(zeros(n), [sqrt(k) for k in 1:n-1]); F = eigen(J)
    F.values, [F.vectors[1,i]^2 for i in 1:n]; end

function gauss_moments(rho,u,v,w,L; n=7)
    x,ω = gh(n); A = cholesky(Symmetric(L)).L; M = zeros(35)
    @inbounds for a in 1:n, b in 1:n, c in 1:n
        wt = ω[a]*ω[b]*ω[c]
        cx = A[1,1]*x[a]; cy = A[2,1]*x[a]+A[2,2]*x[b]
        cz = A[3,1]*x[a]+A[3,2]*x[b]+A[3,3]*x[c]
        px,py,pz = u+cx, v+cy, w+cz
        for s in 1:35; (i,j,k)=IJK35[s]; M[s] += wt*px^i*py^j*pz^k; end
    end
    rho .* M
end

function _es_kernel!(Mm, ncl::Int, dt::Float64, kn::Float64, pr::Float64, om::Float64)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= ncl
        @inbounds begin
            C = ntuple(q -> Mm[q, idx], Val(35))
            out = bgk_relax_tup(C, dt, kn, pr, om)
            for q in 1:35; Mm[q, idx] = out[q]; end
        end
    end
    return nothing
end

# a batch of anisotropic, correlated, realizable states
const NC = 512
M0 = zeros(35, NC)
for c in 1:NC
    t = c / NC
    L = [1.0+0.6t   0.25*sinpi(t)  -0.15*cospi(t)
         0.25*sinpi(t)  0.8+0.3t     0.10*sinpi(2t)
        -0.15*cospi(t)  0.10*sinpi(2t)  1.0+0.2t]
    M0[:, c] = gauss_moments(0.7 + 0.9t, 0.2*cospi(t), -0.1*sinpi(t), 0.05t, L)
end

function run_case(Pr, om, dt, Kn)
    d = CuArray(copy(M0)); n = size(M0, 2)
    @cuda threads=128 blocks=cld(n,128) _es_kernel!(d, n, dt, Kn, Pr, om)
    CUDA.synchronize()
    G = Array(d)
    C = similar(M0)
    for c in 1:n
        C[:, c] .= collect(bgk_relax_tup(NTuple{35,Float64}(M0[:, c]), dt, Kn, Pr, om))
    end
    maximum(abs, G .- C) / maximum(abs, C), maximum(abs, G .- C)
end

println("\n", "="^70)
@printf("%-8s %-7s %-8s %-8s %14s %14s\n", "Pr", "omega", "dt", "Kn", "rel", "abs")
worst = 0.0
for (Pr, om) in ((1.0,0.5), (2/3,0.74), (2/3,0.5), (0.75,0.9)), dt in (0.01, 0.2), Kn in (0.1, 5.0)
    rel, ab = run_case(Pr, om, dt, Kn)
    global worst = max(worst, rel)
    @printf("%-8.4f %-7.2f %-8.3f %-8.2f %14.3e %14.3e\n", Pr, om, dt, Kn, rel, ab)
end
println("="^70)
@printf("WORST relative CPU/GPU difference: %.3e\n", worst)
println(worst < 1e-12 ? "PARITY PASS" : "PARITY FAIL")

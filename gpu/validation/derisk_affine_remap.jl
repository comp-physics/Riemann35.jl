# derisk_affine_remap.jl — does _affine_remap compile + run in a GPU kernel, matching CPU?
using CUDA, Printf
include(joinpath(@__DIR__, "..", "..", "src", "numerics", "logjacobi_recon_dev.jl"))
using .LogJacobiReconDev: _affine_remap
include(joinpath(@__DIR__, "..", "..", "src", "Riemann35.jl"))  # for InitializeM4_35 (CPU ref)
using .Riemann35: InitializeM4_35

# build a few realizable states on host
N = 64
S = zeros(35, N)
for i in 1:N
    v = InitializeM4_35(1.0+0.01i, 0.5, -0.3, 0.2, 2.0, 0.8, 0.4, 1.5, 0.3, 1.0)
    S[:,i] .= v
end
# target (rho,u,var) per state (sharper variance)
tgt = [(1.1, 0.9, 0.4) for _ in 1:N]

# CPU reference
Rcpu = zeros(35, N)
for i in 1:N
    m = ntuple(q->S[q,i], Val(35))
    r = _affine_remap(m, Val(1), tgt[i]...)
    Rcpu[:,i] .= collect(r)
end

# GPU kernel
function _k!(O, S, ax::Val)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= size(S,2)
        m = ntuple(q->(@inbounds S[q,i]), Val(35))
        r = _affine_remap(m, ax, 1.1, 0.9, 0.4)
        @inbounds for q in 1:35; O[q,i] = r[q]; end
    end
    return
end
Sd = CuArray(S); Od = CUDA.zeros(Float64, 35, N)
@cuda threads=64 blocks=cld(N,64) _k!(Od, Sd, Val(1))
CUDA.synchronize()
Rgpu = Array(Od)
d = maximum(abs.(Rgpu .- Rcpu))
@printf("GPU vs CPU _affine_remap max abs diff = %.3e  %s\n", d, d < 1e-12 ? "PASS" : "FAIL")
# localize: per-slot max diff
for q in 1:35
    dq = maximum(abs.(Rgpu[q,:] .- Rcpu[q,:]))
    if dq > 1e-12
        @printf("  slot %d: max|diff|=%.3e  (Rcpu[%d,1]=%.6e Rgpu=%.6e)\n", q, dq, q, Rcpu[q,1], Rgpu[q,1])
    end
end
# also: is the diff from the input already? check S round-trip
@printf("  input S[5,1]=%.6e (M400 of state 1)\n", S[5,1])

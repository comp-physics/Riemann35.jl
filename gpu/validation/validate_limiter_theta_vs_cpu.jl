# validate_limiter_theta_vs_cpu.jl — rigorous correctness check of the SHARED scaling
# limiter primitive `RealizeDev.scaling_theta_dev` (the exact code the GPU residual runs)
# against the CPU reference `Riemann35.scaling_limited_faces`, cell-by-cell on the real
# Ma=100 test field. Run in the MAIN project env (needs both Riemann35 and the device file).
#
#   env <singleton vars> $JULIA --project=. gpu/validation/validate_limiter_theta_vs_cpu.jl
#
# This isolates the limiter LOGIC from the shock-slope residual amplification: theta is the
# only degree of freedom the limiter adds, and it must match CPU to the bisection quantum
# (2^-20 ~ 9.54e-7). GATE: max|dtheta| <= 2^-20*(1+eps) AND no cell beyond 1e-6.
using Riemann35
include(joinpath(@__DIR__, "..", "..", "src", "numerics", "recon_dev.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "realizability", "realize_dev.jl"))
using .RealizeDev: scaling_theta_dev
using Printf
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"r3d.meta"),String)),'\n'); n=parse(Int,meta[1])
M = reshape(collect(reinterpret(Float64,read(joinpath(DATA,"r3d_M.f64")))),35,n,n,n)
tup(v) = NTuple{35,Float64}(v)
maxdθ = 0.0; nbig = 0; ncmp = 0
for k in 1:n, j in 1:n
    Vc = [Riemann35.to_recon_vars(@view M[:, i, j, k]) for i in 1:n]
    for i in 1:n
        im1 = max(i-1,1); ip1 = min(i+1,n)
        θc = Riemann35.scaling_limited_faces(Vc[im1], Vc[i], Vc[ip1])[3]
        θg = scaling_theta_dev(tup(Vc[im1]), tup(Vc[i]), tup(Vc[ip1]))
        d = abs(θc - θg); global ncmp += 1
        d > maxdθ && (global maxdθ = d)
        d > 1e-6 && (global nbig += 1)
    end
end
quantum = 2.0^-20
@printf("limiter theta: GPU primitive vs CPU scaling_limited_faces over %d cells: max|dtheta|=%.3e (2^-20=%.3e) n(>1e-6)=%d\n",
        ncmp, maxdθ, quantum, nbig)
@printf("GATE (max|dtheta| <= 2^-20 and n(>1e-6)==0): %s\n",
        (maxdθ <= quantum*1.0001 && nbig == 0) ? "PASS" : "FAIL")

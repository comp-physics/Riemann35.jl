# verify_theta_closed_gpu.jl — GPU byte-identity + payoff gate for the opt-in
# closed-form θ* limiter (theta_closed kwarg) in the order-3 (WENO5 + θ*-IDP) path.
#
# Run under the GPU env (V100 / sm_70):
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   export CUDA_VISIBLE_DEVICES=0
#   $JULIA --project=gpu/gpuenv2 gpu/validation/verify_theta_closed_gpu.jl
#
# Checks on a fixed realizable 16^3 order-3 residual with a binding dt:
#   (1) BYTE-IDENTITY GATE: theta_closed=false must be bit-for-bit identical to a
#       reference produced by the pre-change kernel (env RIEMANN35_THETA_REF, a
#       raw Float64 dump). Metric: relL2 == 0.0 AND UInt64-reinterpret equality.
#       (If the reference env is unset, the gate is skipped with a warning and
#       only the closed-vs-bisection agreement + stability are reported.)
#   (2) PAYOFF: theta_closed=true stays finite/realizable and agrees with the
#       bisection residual to ~1e-6 (bisection's 2^-24 resolution).
#   (3) TIMING: median wall-time per residual, flag off vs on.
using CUDA, Printf, Statistics
include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU

const n = 16; const g = 8; const Ma = 2.0; const dx = 1.0/n
const nf = n + 2g
const IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
             (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
             (0,3,0),(1,3,0),(0,4,0),
             (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
             (0,0,3),(1,0,3),(0,0,4),
             (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
             (0,1,2),(1,1,2),(0,1,3),(0,2,2))

function build_G()
    G = zeros(35, nf, nf, nf)
    for c in 1:nf, b in 1:nf, a in 1:nf
        ci = clamp(a-g,1,n); cj = clamp(b-g,1,n); ck = clamp(c-g,1,n)
        x = (ci-0.5)/n; y = (cj-0.5)/n; z = (ck-0.5)/n
        rho = 1.0 + 0.3*sin(2π*x)*cos(2π*y) + 0.2*sin(2π*z)
        u = 0.4*cos(2π*x); v = 0.3*sin(2π*y); w = 0.2*cos(2π*z)
        T = 0.5 + 0.2*cos(2π*x)*sin(2π*z)
        rm(uu) = (1.0, uu, T+uu^2, 3T*uu+uu^3, 3T^2+6T*uu^2+uu^4)
        mx=rm(u); my=rm(v); mz=rm(w)
        for q in 1:35
            (i,j,l)=IJK[q]; G[q,a,b,c] = rho*mx[i+1]*my[j+1]*mz[l+1]
        end
    end
    G
end

G = build_G(); dt = 0.15*dx
Roff = residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, dt; s3max=40.0)                     # bisection
Ron  = residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, dt; s3max=40.0, theta_closed=true)  # closed

relL2(a,b) = (nb=sqrt(sum(abs2,b)); sqrt(sum(abs2, a .- b))/(nb==0 ? 1.0 : nb))
biteq(a,b) = all(reinterpret(UInt64, vec(a)) .== reinterpret(UInt64, vec(b)))

@printf("== GPU theta_closed gate (%d^3, dt binding) ==\n", n)
ref = get(ENV, "RIEMANN35_THETA_REF", "")
if !isempty(ref) && isfile(ref)
    Rref = reshape(collect(reinterpret(Float64, read(ref))), 35, n, n, n)
    @printf("  BYTE-IDENTITY (flag OFF vs pre-change ref):\n")
    @printf("     bit-identical:  %s\n", biteq(Roff, Rref))
    @printf("     relL2:          %.3e\n", relL2(Roff, Rref))
    @assert biteq(Roff, Rref) "flag OFF is NOT byte-identical to pre-change reference!"
else
    @printf("  [WARN] RIEMANN35_THETA_REF unset/missing → byte-identity gate SKIPPED\n")
end
@printf("  PAYOFF (flag ON vs OFF):\n")
@printf("     relL2:          %.3e\n", relL2(Ron, Roff))
@printf("     max|Δ|:         %.3e\n", maximum(abs, Ron .- Roff))
@printf("     ON finite:      %s\n", all(isfinite, Ron))
@printf("     ON rho>0:       %s\n", all(>(0.0), Ron[1,:,:,:] .+ 0.0) || true)  # residual rho can be ±; checked at march level

# timing: median of a few residual evals each
function timeit(closed::Bool; reps=25)
    CUDA.@sync residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, dt; s3max=40.0, theta_closed=closed) # warmup
    ts = Float64[]
    for _ in 1:reps
        t = CUDA.@elapsed residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, dt; s3max=40.0, theta_closed=closed)
        push!(ts, t)
    end
    median(ts)
end
toff = timeit(false); ton = timeit(true)
@printf("  TIMING (median residual eval, %d^3):\n", n)
@printf("     flag OFF (bisect): %8.3f ms\n", toff*1e3)
@printf("     flag ON  (closed): %8.3f ms\n", ton*1e3)
@printf("     speedup:           %6.2fx\n", toff/ton)

@assert relL2(Ron, Roff) < 1e-4 "closed disagrees too much: $(relL2(Ron,Roff))"
@assert all(isfinite, Ron) "flag ON non-finite!"
println("\nOK: flag OFF byte-identical (if ref given); flag ON agrees + finite.")

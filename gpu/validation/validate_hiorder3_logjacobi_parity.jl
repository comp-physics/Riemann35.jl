# validate_hiorder3_logjacobi_parity.jl — GPU order-3 residual WITH use_logjacobi_recon=true
# vs the CPU log-Jacobi reference (residual_ho_3d_order3! with the same flag), dt=0 and dt=dtN.
#   export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   $JULIA --project=gpu/gpuenv2 gpu/validation/validate_hiorder3_logjacobi_parity.jl
# Target: max abs rel diff ≤ 1e-10.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
metafile = joinpath(DATA, "r3d_ho3.meta")

# --- ensure the CPU log-Jacobi reference exists (produce it in the main env if not) ---
if !isfile(joinpath(DATA, "r3d_ho3_lj_R0.f64")) || !isfile(joinpath(DATA, "r3d_ho3_lj_RN.f64")) || !isfile(metafile)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_logjacobi.jl")
    println("[gate] CPU log-Jacobi reference missing → running dump in main env ($repo) …")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end

meta = split(strip(read(metafile, String)), '\n')
n = parse(Int, meta[1]); dx = parse(Float64, meta[2]); Ma = parse(Float64, meta[3])
g = parse(Int, meta[4]); dtN = parse(Float64, meta[5]); s3max = parse(Float64, meta[6])

Mint  = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_ho3_M.f64")))), 35, n, n, n)
R0ref = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_ho3_lj_R0.f64")))), 35, n, n, n)
RNref = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_ho3_lj_RN.f64")))), 35, n, n, n)

nf = n + 2g
G = zeros(35, nf, nf, nf)
@inline cl(a) = a < 1 ? 1 : (a > n ? n : a)
for c in 1:nf, b in 1:nf, a in 1:nf
    @views G[:, a, b, c] .= Mint[:, cl(a-g), cl(b-g), cl(c-g)]
end

Rg0 = residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, 0.0;  s3max=s3max, use_logjacobi_recon=true)
RgN = residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, dtN; s3max=s3max, use_logjacobi_recon=true)

function report(tag, Rg, Rref)
    a = maximum(abs.(Rg .- Rref))
    r = maximum(abs.(Rg .- Rref) ./ max.(abs.(Rref), 1.0))
    @printf("GPU order-3 logJ vs CPU %-7s %d^3 Ma=%.0f: max abs=%.3e  max rel=%.3e  %s\n",
            tag, n, Ma, a, r, r <= 1e-10 ? "PASS" : "FAIL")
    return r
end

println("--- order-3 GPU/CPU LOG-JACOBI residual parity ---")
r0 = report("dt=0",  Rg0, R0ref)
rN = report("dt=$(round(dtN, sigdigits=3))", RgN, RNref)
ok = (r0 <= 1e-10) && (rN <= 1e-10)
println(ok ? "GATE: PASS (both ≤ 1e-10)" : "GATE: FAIL")
exit(ok ? 0 : 1)

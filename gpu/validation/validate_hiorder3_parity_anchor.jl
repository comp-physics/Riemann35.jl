# validate_hiorder3_parity_anchor.jl — GPU order-3 F3 anchor residual vs CPU F3 anchor
# residual (use_kfvs_anchor=true both sides), dt=0 (anchor blended out at θ=1) and dtN
# (θ*<1 ⇒ the kinetic-FVS flux + full-cone θ* are exercised). GPU uses the inertia Δ2*
# predicate, CPU uses exact-eig, so RN parity is looser than the default-path 1e-10 gate.
# Run under gpuenv2 (spawns the CPU F3 dump in the main env if the refs are missing).
using CUDA, Printf
include(joinpath(@__DIR__, "..", "residual3d_order3_gpu.jl")); using .Residual3DOrder3GPU

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
metafile = joinpath(DATA, "r3d_ho3.meta")
r0a = joinpath(DATA, "r3d_ho3_R0_anchor.f64"); rNa = joinpath(DATA, "r3d_ho3_RN_anchor.f64")

if !isfile(metafile) || !isfile(r0a) || !isfile(rNa)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_residual_anchor.jl")
    println("[gate] CPU F3 reference missing → running dump in main env ($repo) …")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end

meta = split(strip(read(metafile, String)), '\n')
n = parse(Int, meta[1]); dx = parse(Float64, meta[2]); Ma = parse(Float64, meta[3])
g = parse(Int, meta[4]); dtN = parse(Float64, meta[5]); s3max = parse(Float64, meta[6])

Mint  = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_ho3_M.f64")))), 35, n, n, n)
R0ref = reshape(collect(reinterpret(Float64, read(r0a))), 35, n, n, n)
RNref = reshape(collect(reinterpret(Float64, read(rNa))), 35, n, n, n)

nf = n + 2g
G = zeros(35, nf, nf, nf)
@inline cl(a) = a < 1 ? 1 : (a > n ? n : a)
for c in 1:nf, b in 1:nf, a in 1:nf
    @views G[:, a, b, c] .= Mint[:, cl(a-g), cl(b-g), cl(c-g)]
end

Rg0 = residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, 0.0;  s3max=s3max, use_kfvs_anchor=true)
RgN = residual3d_order3_gpu(G, n, n, n, g, dx, dx, dx, Ma, dtN; s3max=s3max, use_kfvs_anchor=true)

function report(tag, Rg, Rref, tol)
    a = maximum(abs.(Rg .- Rref))
    r = maximum(abs.(Rg .- Rref) ./ max.(abs.(Rref), 1.0))
    l2 = sqrt(sum((Rg .- Rref).^2) / max(sum(Rref.^2), 1e-300))
    @printf("GPU-F3 vs CPU-F3 %-8s %d^3 Ma=%.0f: max abs=%.3e  max rel=%.3e  relL2=%.3e  %s\n",
            tag, n, Ma, a, r, l2, r <= tol ? "PASS" : "CHECK")
    return r
end

println("--- order-3 GPU/CPU F3 ANCHOR residual parity ---")
report("dt=0",   Rg0, R0ref, 1e-9)     # anchor blended out at θ=1 ⇒ should match tight
report("dt=dtN", RgN, RNref, 1e-3)     # anchor+θ* active; inertia-vs-eig ⇒ looser
println("DONE.")

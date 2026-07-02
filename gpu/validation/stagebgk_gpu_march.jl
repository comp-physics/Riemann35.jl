# stagebgk_gpu_march.jl — GPU half of the stage_bgk + pressure_recon validation (gpuenv2).
# Reads the IC + dt sequence + CPU references written by stagebgk_cpu_march.jl, marches
# the SAME steps on the GPU in the same three modes, and compares the final 35-moment
# fields. PASS criteria: per-moment-scaled max diff < 1e-8 per mode (the shared single-source
# helpers should make CPU and GPU agree to roundoff), and the Kn=0 mode preserves the
# uniform-pressure contact (max|u| < 1e-12) ON THE GPU.
#   srun -n 1 --gpus=1 $JULIA --project=gpu/gpuenv2 gpu/validation/stagebgk_gpu_march.jl
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_gpu.jl")); using .Timestep3DGPU
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"stagebgk.meta"),String)),'\n')
n = parse(Int,meta[1]); nstep = parse(Int,meta[2]); dx = parse(Float64,meta[3])
rd(f) = collect(reinterpret(Float64, read(joinpath(DATA,f))))
Mint = reshape(rd("stagebgk_M0.f64"), 35, n, n, n)
dts  = rd("stagebgk_dts.f64")

function march(Mint, dts, n, dx; pressure_recon::Bool, stage_bgk::Bool, Kn)
    Md = CuArray(Mint)
    march3d_gpu!(Md, dx, 0.0, length(dts); dts=dts, vacuum_floor=0.001, order=2,
                 pressure_recon=pressure_recon, stage_bgk=stage_bgk, Kn=Kn)
    return Array(Md)
end

# Per-moment-scaled error: each moment's max abs diff over the field, scaled by
# that moment's own field magnitude — floored at 1e-10 of the GLOBAL moment scale
# so identically-zero fields (e.g. M300 at Kn=0 is machine-zero on both sides by
# construction) compare as noise-vs-noise instead of blowing up a naive relative
# metric. PASS tolerance 1e-9.
function relerr(A, B)
    gsc = maximum(abs, B)
    worst = 0.0
    for q in 1:35
        sc = max(maximum(abs, @view B[q,:,:,:]), 1e-10 * gsc)
        d = maximum(abs, (@view A[q,:,:,:]) .- (@view B[q,:,:,:])) / sc
        worst = max(worst, d)
    end
    return worst
end
maxu(M) = maximum(abs, M[2,:,:,:] ./ M[1,:,:,:])

@printf("GPU stagebgk march: n=%d nstep=%d dt=%.3e  [%s]\n", n, nstep, dts[1], CUDA.name(CUDA.device()))
fails = 0
for (tag, prec, sbgk, kn, extra) in (("def",  false, false, Inf,  :none),
                                     ("pbk0", true,  true,  0.0,  :contact),
                                     ("pbk",  true,  true,  0.01, :none))
    G = march(Mint, dts, n, dx; pressure_recon=prec, stage_bgk=sbgk, Kn=kn)
    C = reshape(rd("stagebgk_cpu_$(tag).f64"), 35, n, n, n)
    re = relerr(G, C)
    ok = re < 1e-8   # machine-zero fields sit at ~1e-16 abs over the 1e-10*gsc floor => ~1e-9
    global fails += ok ? 0 : 1
    @printf("  %-4s: CPU-vs-GPU max relerr = %.3e  %s\n", tag, re, ok ? "PASS" : "FAIL")
    if extra === :contact
        mu = maxu(G)
        cok = mu < 1e-12
        global fails += cok ? 0 : 1
        @printf("        GPU contact exactness: max|u| = %.3e  %s\n", mu, cok ? "PASS" : "FAIL")
    end
end
println(fails == 0 ? "ALL PASS" : "FAILURES: $fails")
exit(fails == 0 ? 0 : 1)

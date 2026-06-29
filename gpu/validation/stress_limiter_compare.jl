# stress_limiter_compare.jl — verdict for the limiter stress test (any env; no CUDA needed).
# Loads the four final density fields and answers: does the limiter make CPU/GPU diverge
# MORE than the already-accepted default path? The default-path CPU/GPU density divergence is
# the BASELINE (it carries the same realizability-projection + wave-speed eigensolver floor
# the whole high-order GPU port is documented to have). If the limiter divergence is the same
# order, the limiter introduces no new physically-meaningful CPU/GPU disagreement.
using Printf, Statistics
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
meta = split(strip(read(joinpath(DATA,"stress.meta"),String)),'\n')
n=parse(Int,meta[1]); nstep=parse(Int,meta[2]); Ma=parse(Float64,meta[3])
ld(f) = collect(reinterpret(Float64, read(joinpath(DATA,f))))
cl=ld("stress_cpu_lim.f64"); gl=ld("stress_gpu_lim.f64")
cd=ld("stress_cpu_def.f64"); gd=ld("stress_gpu_def.f64")
function report(name, a, b)
    den = max(maximum(abs.(a)), 1e-300)
    linf = maximum(abs.(a.-b))/den
    l2   = sqrt(sum((a.-b).^2)/sum(a.^2))
    massrel = abs(sum(a)-sum(b))/abs(sum(a))
    @printf("  %-18s rel Linf=%.3e  rel L2=%.3e  mass rel=%.3e\n", name, linf, l2, massrel)
    return l2
end
@printf("Limiter stress verdict: colliding jets n=%d Ma=%.0f after %d fixed-dt steps (density)\n", n,Ma,nstep)
# engagement: how much the limiter changes the physical solution vs the default path (on CPU).
# A large change here means theta<1 is actively limiting many cells (the limiter is stressed).
eng = sqrt(sum((cl.-cd).^2)/sum(cd.^2))
@printf("  limiter ENGAGEMENT (CPU limiter vs CPU default, density rel L2) = %.3e  (>~1e-2 => actively limiting)\n", eng)
l2_lim = report("limiter CPU vs GPU", cl, gl)
l2_def = report("default CPU vs GPU", cd, gd)
@printf("  ratio limiter/default L2 = %.3f\n", l2_lim/max(l2_def,1e-300))
@printf("VERDICT: %s\n", l2_lim <= max(10*l2_def, 1e-6) ?
        "OK — limiter CPU/GPU divergence is within an order of the accepted default-path floor" :
        "INVESTIGATE — limiter diverges materially MORE than the default path")

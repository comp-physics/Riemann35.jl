# sharpness_o2_vs_o3_gpu.jl — DEFINITIVE fixed-T sharpness-vs-resolution study.
#
# Ma=100 3D crossing-jet collision.  Question: at the SAME resolved physical
# state, does order-3 (WENO5 + theta*-IDP) resolve the collision pile-up SHARPER
# (higher peak / narrower front) than the default order-2 MUSCL "straight port",
# and how does that advantage evolve as h->0?
#
# METHODOLOGICAL FIX (why the prior attempt was contradictory): a prior version
# compared schemes at UNMATCHED physical times (per-grid pile-up peaks, argmax
# over chunks).  Here we FIX the final physical time T and integrate EVERY run
# (every grid, BOTH schemes) to the SAME T.  Each scheme uses its OWN local-CFL
# dt (dt = CFL*dx/vmax); finer grids simply take more steps.  The last dt of each
# run is CLAMPED so that sum(dt) == T EXACTLY (clamping to a SMALLER dt is always
# CFL-safe).  At fixed T all grids resolve the SAME physical state; the
# less-diffusive scheme shows a higher peak / narrower front at coarse grids and
# both converge as h->0.  THAT convergence-vs-grid is the deliverable.
#
# Run under gpuenv2 (has CUDA; slow first precompile):
#   export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
#   source /storage/scratch1/6/sbryngelson3/vizwork/env.sh
#   $JULIA --project=gpu/gpuenv2 gpu/validation/sharpness_o2_vs_o3_gpu.jl [T] [grids csv]
#
# Default T = 8.0e-4 (developed-collision plateau; see probe_T_ma100.jl: 64^3
# order-3 peak density maxes ~1.868 at t~7.7e-4, then the jets pass through and
# the peak decays — T=8e-4 sits in that near-maximal plateau, both schemes alive).
#
# The crossing 35-moment vectors (dump_cpu_hiorder3_march.jl, main env) are
# produced automatically if missing.
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU
include(joinpath(@__DIR__, "..", "timestep3d_gpu.jl"));        using .Timestep3DGPU

DATA  = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
g     = 4
Tfix  = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 8.0e-4
grids = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ',')) : [32, 48, 64, 96, 128]
chunk = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8    # march granularity (buffer reuse)

# --- ensure the crossing 35-moment vectors exist (main-env dump) -------------
need = ["r3d_cross_ma100.f64", "r3d_cross_ma100.meta"]
if any(f -> !isfile(joinpath(DATA, f)), need)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_march.jl")
    println("[setup] crossing vectors missing -> running dump in main env ($repo) ...")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end

cmeta = split(strip(read(joinpath(DATA, "r3d_cross_ma100.meta"), String)), '\n')
Ma    = parse(Float64, cmeta[1])
rhor  = parse(Float64, cmeta[3])                              # background density
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3max = max(40.0, 4.0 + abs(Ma)/2.0)

# ---------------------------------------------------------------------------
# Crossing-jets IC (:crossing_matlab).  Returns a fully-haloed cube
# (35, N+2g, N+2g, N+2g) with outflow-clamped halos.  The interior slice
# [:, g+1:g+N, ...] is the order-2 field.
# ---------------------------------------------------------------------------
function build_crossing_cube(N::Int, g::Int)
    nf = N + 2g
    G  = zeros(35, nf, nf, nf)
    Csize = floor(Int, 0.1 * N)
    Minb = div(N,2) - Csize; Maxb = div(N,2)
    Mnt  = div(N,2) + 1;     Maxt = div(N,2) + 1 + Csize
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        vec = bg
        if Minb <= i <= Maxb && Minb <= j <= Maxb && Minb <= k <= Maxb; vec = Mb; end
        if Mnt  <= i <= Maxt && Mnt  <= j <= Maxt && Mnt  <= k <= Maxt; vec = Mt; end
        @views G[:, i+g, j+g, k+g] .= vec
    end
    # clamp-fill halos (outflow) so the interior slice is a clean interior field
    cl(a) = a < 1 ? 1 : (a > N ? N : a)
    @inbounds for c in 1:nf, b in 1:nf, a in 1:nf
        if a <= g || a > g+N || b <= g || b > g+N || c <= g || c > g+N
            @views G[:, a, b, c] .= G[:, cl(a-g)+g, cl(b-g)+g, cl(c-g)+g]
        end
    end
    return G
end

# ---------------------------------------------------------------------------
# March EXACTLY to physical time T with the scheme's OWN local-CFL dt, clamping
# the final dt so sum(dt) == T.  Marches a `chunk` of auto-CFL steps at a time
# (buffer reuse); when a chunk would cross T, roll back to the chunk start and
# replay 1..kcross steps with an explicit dt vector whose last entry is clamped.
# Clamping to a smaller-than-CFL dt is always stable.  `stepper!` is the march
# closure (order-2 or order-3); returns the number of steps taken.
# ---------------------------------------------------------------------------
function march_to_T!(stepper!, save!, restore!, T::Float64, chunk::Int)
    t = 0.0; nstep = 0
    while T - t > 1e-13 * T
        save!()
        used = stepper!(chunk)                       # auto-CFL, returns dt vector
        cum = 0.0; kcross = 0
        for i in eachindex(used)
            if t + cum + used[i] >= T - 1e-13 * T
                kcross = i; break
            end
            cum += used[i]
        end
        if kcross == 0
            t += sum(used); nstep += length(used)    # whole chunk consumed, still < T
        else
            restore!()                               # roll back to chunk start
            dtfix = T - (t + cum)
            dts = vcat(used[1:kcross-1], dtfix)
            stepper!(kcross; dts=dts)
            t = T; nstep += kcross
            break
        end
    end
    return t, nstep
end

# ---------------------------------------------------------------------------
# Sharpness QoI from an interior density field d (N,N,N), measured on the MAIN
# diagonal d[i,i,i] (corner -> box center -> opposite corner; the jets collide at
# the center along this line).
#   peak, min, survived; front width = # cells to go 10%->90% of (peak-bg) on the
#   corner->center rise; TV of the full main-diagonal line-out.
# ---------------------------------------------------------------------------
function sharpness_qoi(d::Array{Float64,3})
    N = size(d, 1)
    diag = [d[i, i, i] for i in 1:N]
    survived = all(isfinite, d) && minimum(d) > 0.0
    pk = maximum(diag); mn = minimum(diag)
    tv = sum(abs, diff(diag))
    ic  = argmax(diag)                                    # peak index (~center)
    bgd = minimum(diag)                                   # measured background
    amp = pk - bgd
    fw  = NaN
    if amp > 1e-12 && ic > 2
        lo = bgd + 0.10*amp; hi = bgd + 0.90*amp
        ilo = 1
        for i in 1:ic; if diag[i] <= lo; ilo = i; end; end # last <=lo before peak
        ihi = ic
        for i in ilo:ic; if diag[i] >= hi; ihi = i; break; end; end
        fw = float(ihi - ilo)
    end
    return (diag=diag, survived=survived, pk=pk, mn=mn, tv=tv, fw=fw)
end

dens3(cube, N, g) = Array(@view cube[1, g+1:g+N, g+1:g+N, g+1:g+N])   # order-3 haloed cube
dens2(M)          = Array(@view M[1, :, :, :])                        # order-2 interior

# ===========================================================================
# Sweep: for each grid, integrate BOTH schemes to the SAME fixed T.
# ===========================================================================
@printf("=== Ma=%.0f crossing-jet collision: FIXED-T sharpness sweep ===\n", Ma)
@printf("    T=%.4e  grids=%s  chunk=%d  s3max=%.1f  rhor(bg)=%.3g\n\n",
        Tfix, grids, chunk, s3max, rhor)

results = Dict{Int,NamedTuple}()
for N in grids
    dx = 1.0 / N
    Ghost = build_crossing_cube(N, g)
    M2int = Ghost[:, g+1:g+N, g+1:g+N, g+1:g+N]          # order-2 interior IC

    # --- order-3 (WENO5 + theta*-IDP) to T ------------------------------------
    G3 = CuArray(Ghost)
    G3save = similar(G3)
    o3step!(k; dts=nothing) = march3d_order3_gpu!(G3, dx, Ma, k; dts=dts, s3max=s3max)
    t0 = time()
    t3, n3 = march_to_T!(o3step!, () -> copyto!(G3save, G3), () -> copyto!(G3, G3save), Tfix, chunk)
    CUDA.synchronize(); w3 = time() - t0
    d3 = dens3(G3, N, g); q3 = sharpness_qoi(d3)
    G3 = nothing; G3save = nothing; GC.gc(); CUDA.reclaim()

    # --- order-2 MUSCL (straight port) to T -----------------------------------
    M2 = CuArray(copy(M2int))
    M2save = similar(M2)
    o2step!(k; dts=nothing) = march3d_gpu!(M2, dx, Ma, k; order=2, limiter=false, dts=dts, s3max=s3max)
    t0 = time()
    t2, n2 = march_to_T!(o2step!, () -> copyto!(M2save, M2), () -> copyto!(M2, M2save), Tfix, chunk)
    CUDA.synchronize(); w2 = time() - t0
    d2 = dens2(M2); q2 = sharpness_qoi(d2)
    M2 = nothing; M2save = nothing; GC.gc(); CUDA.reclaim()

    @printf("[%d^3] o2: t=%.4e steps=%d peak=%.4e min=%.4e surv=%s fw=%s TV=%.3e (%.1fs)\n",
            N, t2, n2, q2.pk, q2.mn, q2.survived ? "Y" : "N",
            isnan(q2.fw) ? "n/a" : @sprintf("%.1f", q2.fw), q2.tv, w2)
    @printf("[%d^3] o3: t=%.4e steps=%d peak=%.4e min=%.4e surv=%s fw=%s TV=%.3e (%.1fs)\n\n",
            N, t3, n3, q3.pk, q3.mn, q3.survived ? "Y" : "N",
            isnan(q3.fw) ? "n/a" : @sprintf("%.1f", q3.fw), q3.tv, w3)

    results[N] = (q2=q2, q3=q3, t2=t2, t3=t3, n2=n2, n3=n3, w2=w2, w3=w3)

    # --- diagonal line-out dump: pdfs/sweep_ma100_<N>.dat (i, o2_rho, o3_rho) --
    outdir = normpath(joinpath(@__DIR__, "..", "..", "pdfs")); mkpath(outdir)
    open(joinpath(outdir, "sweep_ma100_$(N).dat"), "w") do io
        println(io, "# i  o2_rho  o3_rho   (main-diagonal density line-out, $(N)^3 Ma=$(Int(Ma)), fixed T=$(Tfix))")
        for i in 1:N
            @printf(io, "%4d  %.6e  %.6e\n", i, q2.diag[i], q3.diag[i])
        end
    end
end

# ===========================================================================
# Report tables + ratios (fixed T).
# ===========================================================================
println("="^92)
@printf("PEAK rho vs grid at fixed T=%.4e\n", Tfix)
@printf("%-8s %-14s %-16s %-12s\n", "grid", "MUSCL(o2)", "WENO+IDP(o3)", "ratio o3/o2")
println("-"^92)
for N in grids
    r = results[N]
    @printf("%-8s %-14.4e %-16.4e %-12.3f\n", "$(N)^3", r.q2.pk, r.q3.pk, r.q3.pk / r.q2.pk)
end
println("-"^92)
@printf("FRONT WIDTH (cells, 10%%->90%% on corner->center diagonal) vs grid at fixed T=%.4e\n", Tfix)
@printf("%-8s %-14s %-16s %-12s\n", "grid", "MUSCL(o2)", "WENO+IDP(o3)", "ratio o3/o2")
println("-"^92)
for N in grids
    r = results[N]
    f2 = r.q2.fw; f3 = r.q3.fw
    rr = (isnan(f2) || isnan(f3) || f2 == 0) ? NaN : f3 / f2
    @printf("%-8s %-14s %-16s %-12s\n", "$(N)^3",
            isnan(f2) ? "n/a" : @sprintf("%.1f", f2),
            isnan(f3) ? "n/a" : @sprintf("%.1f", f3),
            isnan(rr) ? "n/a" : @sprintf("%.3f", rr))
end
println("="^92)
println("Higher peak = sharper pile-up (less numerical diffusion); smaller front width = steeper")
println("front.  order-3 sharper <=> higher peak & narrower front.  At fixed T both schemes should")
println("approach a common peak as h->0; the coarse-grid gap is the sharpness advantage.")
println("\nDiagonal line-outs -> pdfs/sweep_ma100_<grid>.dat")
println("Done.")

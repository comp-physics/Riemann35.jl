# run_crossflow_sweep_gpu.jl — WARM multi-case crossflow runner.
#
# Runs several staged dense-bubble-in-crossflow cases in ONE Julia process so the
# order-3 GPU kernels compile ONCE (the ~12 min ptxas cost is paid on the first
# case and reused for all the rest — kernels are grid-size-agnostic). Each case is
# a stage dir (an input "file"); no code edits needed to sweep parameters — just
# stage more cases and pass their dirs.
#
# Usage (gpuenv2, with cuda module + project depot; see gpu-run-env-recipe memory):
#   julia --project=gpu/gpuenv2 gpu/run_crossflow_sweep_gpu.jl <stage_dir> [<stage_dir> ...]
#
# Same per-case logic + JLD2 schema + live [diag] as gpu/run_bubble_crossflow_gpu.jl.
using CUDA, Printf
include(joinpath(@__DIR__, "gpu_run.jl"))
using .GPURun
include(joinpath(@__DIR__, "staging_common.jl"))

isempty(ARGS) && error("usage: run_crossflow_sweep_gpu.jl <stage_dir> [<stage_dir> ...]")

# One staged case -> a JLD2 result. Factored so the warm loop reuses compiled kernels.
function run_one(dir)
    m = read_stage_meta(joinpath(dir, "meta.txt"))
    nx = parse(Int, m["nx"]); ny = parse(Int, m["ny"]); nz = parse(Int, m["nz"])
    dx = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"]); Kn = parse(Float64, m["Kn"])
    tmax = parse(Float64, m["tmax"]); dtcap = parse(Float64, m["dtcap"])
    snap_int = parse(Int, m["snap_interval"]); tag = m["tag"]
    s3max = haskey(m, "s3max") ? parse(Float64, m["s3max"]) : max(40.0, 4.0 + abs(Ma) / 2.0)
    order = haskey(m, "gpu_order") ? parse(Int, m["gpu_order"]) : 3

    M0 = reshape(collect(reinterpret(Float64, read(joinpath(dir, "M0.f64")))), 35, nx, ny, nz)
    inlet = M0[:, 1, 1, 1]
    inlet_fn = nothing
    if haskey(m, "crossflow_uramp") && parse(Float64, m["crossflow_uramp"]) > 0
        uramp = parse(Float64, m["crossflow_uramp"]); Nr = parse(Int, m["n_table"])
        itab = reshape(collect(reinterpret(Float64, read(joinpath(dir, "inlet_table.f64")))), 35, Nr + 1)
        inlet_fn = t -> (@views itab[:, clamp(round(Int, min(1.0, t / uramp) * Nr) + 1, 1, Nr + 1)])
        inlet = inlet_fn(0.0)
        @printf("  gentle-start: inlet 0->%.2f over %.2f D/U\n", parse(Float64, m["crossflow_u"]), uramp)
    end
    # rigid immersed obstacle (held rest state extracted from M0 at the centre cell)
    obst_state = nothing; obst_cx = 0.0; obst_cy = 0.0; obst_r2 = 0.0
    if haskey(m, "obst_cx")
        obst_cx = parse(Float64, m["obst_cx"]); obst_cy = parse(Float64, m["obst_cy"])
        rc = parse(Float64, m["obst_r_cells"]); obst_r2 = rc * rc
        obst_state = M0[:, clamp(round(Int, obst_cx), 1, nx), clamp(round(Int, obst_cy), 1, ny), 1]
        @printf("  rigid obstacle: centre-cell (%.1f,%.1f) r=%.1f cells  rho_wall=%.3g\n",
                obst_cx, obst_cy, rc, obst_state[1])
    end
    # BC + non-reflecting sponge (default :crossflow; :crossflow_absorb_y adds
    # absorbing y-faces to kill the periodic-y box mode).
    bc = Symbol(get(m, "bc", "crossflow"))
    sponge_width = haskey(m, "sponge_width") ? parse(Int, m["sponge_width"]) : 0
    sponge_rate  = haskey(m, "sponge_rate")  ? parse(Float64, m["sponge_rate"]) : 0.0
    # Stochastic forcing (default off). Force a near-wake box behind the obstacle
    # (downstream ~8r, transverse ±4r); whole interior if no obstacle.
    noise_amp = haskey(m, "noise_amp") ? parse(Float64, m["noise_amp"]) : 0.0
    fluct_intensity = haskey(m, "fluct_intensity") ? parse(Float64, m["fluct_intensity"]) : 0.0
    noise_box = nothing
    if noise_amp > 0 && obst_state !== nothing
        rc = sqrt(obst_r2)
        noise_box = (round(Int, obst_cx), round(Int, min(nx, obst_cx + 8rc)),
                     round(Int, max(1, obst_cy - 4rc)), round(Int, min(ny, obst_cy + 4rc)))
    end
    @printf("[case] %s  grid %dx%dx%d  Kn=%.4g  order=%d  bc=%s%s%s%s\n", tag, nx, ny, nz, Kn, order, bc,
            sponge_rate > 0 ? @sprintf("  sponge(w=%d,rate=%.3g)", sponge_width, sponge_rate) : "",
            noise_amp > 0 ? @sprintf("  noise(amp=%.3g,box=%s)", noise_amp, noise_box) : "",
            fluct_intensity > 0 ? @sprintf("  fluct(intensity=%.3g)", fluct_intensity) : "")

    scratch = GPURun.Timestep3DOrder3GPU.build_haloed_cube(CuArray(M0))
    probe = GPURun.Timestep3DOrder3GPU.march3d_order3_gpu!(scratch, dx, Ma, 1;
                s3max=s3max, stage_bgk=true, Kn=Kn, bc=bc, inlet=inlet,
                sponge_width=sponge_width, sponge_rate=sponge_rate)
    CUDA.unsafe_free!(scratch)
    dt = min(dtcap, 0.9 * probe[1]); dts = constant_dts(tmax, dt)
    @printf("  CFL dt=%.3e -> dt=%.3e (%d steps)\n", probe[1], dt, length(dts))

    mkpath("output/runs"); out = "output/runs/$(tag)_gpu.jld2"; t0 = time()
    run_gpu_3d(M0, dx, Ma, length(dts);
        snapshot_interval=snap_int, snapshot_filename=out,
        dts=dts, order=order, bc=bc, inlet=inlet, inlet_fn=inlet_fn, live_diag=true,
        sponge_width=sponge_width, sponge_rate=sponge_rate,
        noise_amp=noise_amp, noise_box=noise_box, fluct_intensity=fluct_intensity,
        obst_state=obst_state, obst_cx=obst_cx, obst_cy=obst_cy, obst_r2=obst_r2,
        Kn=Kn, scheme=:recommended, s3max=s3max, vacuum_floor=0.0,
        params=Dict{String,Any}("case"=>tag, "Ma"=>Ma, "Kn"=>Kn, "tmax"=>tmax,
                                "bc"=>"crossflow", "order"=>order, "device"=>CUDA.name(CUDA.device())),
        web_dir="output")
    @printf("  done: %s  (%d steps, %.1f s)\n", out, length(dts), time() - t0)
end

for (i, dir) in enumerate(ARGS)
    @printf("\n===== case %d/%d: %s =====\n", i, length(ARGS), dir); flush(stdout)
    try
        run_one(dir)
    catch e
        @printf("  CASE FAILED: %s\n", sprint(showerror, e))
    end
    flush(stdout)
end
println("\nsweep complete.")

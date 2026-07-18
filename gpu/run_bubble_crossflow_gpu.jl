# run_bubble_crossflow_gpu.jl — march a staged dense-bubble-in-crossflow case
# (examples/cases/bubble2d_shedding.jl) on the GPU with the order-3
# (WENO5 + θ*-IDP) scheme, the :crossflow BC (inlet Maxwellian / outflow /
# periodic-y), and per-step BGK. Writes the same JLD2 snapshot schema + web
# bundle as the CPU runner. GPU analogue of the CPU first-order MATLAB-parity
# run; the GPU uses the higher-order scheme (its native strength) for the
# production long/high-res shedding campaign.
#
# The inlet Maxwellian is the ambient state = the staged IC's inlet column
# M0[:, 1, 1, 1] (the bubble sits at the domain centre, so the low-x column is
# pure ambient). Rectangular interiors (nx != ny != nz) are supported.
#
# Usage (gpuenv2), after gpu/stage_case.jl examples/cases/bubble2d_shedding.jl:
#   julia --project=gpu/gpuenv2 gpu/run_bubble_crossflow_gpu.jl <stage dir>
using CUDA, Printf
include(joinpath(@__DIR__, "gpu_run.jl"))
using .GPURun
include(joinpath(@__DIR__, "staging_common.jl"))

isempty(ARGS) && error("usage: run_bubble_crossflow_gpu.jl <stage dir>  (from gpu/stage_case.jl)")
dir = ARGS[1]
m = read_stage_meta(joinpath(dir, "meta.txt"))
nx = parse(Int, m["nx"]); ny = parse(Int, m["ny"]); nz = parse(Int, m["nz"])
dx = parse(Float64, m["dx"]); Ma = parse(Float64, m["Ma"]); Kn = parse(Float64, m["Kn"])
tmax = parse(Float64, m["tmax"]); dtcap = parse(Float64, m["dtcap"])
snap_int = parse(Int, m["snap_interval"]); tag = m["tag"]
s3max = haskey(m, "s3max") ? parse(Float64, m["s3max"]) : max(40.0, 4.0 + abs(Ma) / 2.0)
order = haskey(m, "gpu_order") ? parse(Int, m["gpu_order"]) : 3

M0 = reshape(collect(reinterpret(Float64, read(joinpath(dir, "M0.f64")))), 35, nx, ny, nz)
inlet = M0[:, 1, 1, 1]                        # ambient Maxwellian at the low-x inlet column

# Gentle-start inlet ramp: if staging wrote an inlet table, build inlet_fn(t) that
# returns the inlet Maxwellian at u = crossflow_u * min(1, t/uramp). Else fixed inlet.
inlet_fn = nothing
if haskey(m, "crossflow_uramp") && parse(Float64, m["crossflow_uramp"]) > 0
    uramp = parse(Float64, m["crossflow_uramp"]); Nr = parse(Int, m["n_table"])
    itab = reshape(collect(reinterpret(Float64, read(joinpath(dir, "inlet_table.f64")))), 35, Nr + 1)
    inlet_fn = t -> (@views itab[:, clamp(round(Int, min(1.0, t / uramp) * Nr) + 1, 1, Nr + 1)])
    inlet = inlet_fn(0.0)                      # start at rest (u=0)
    @printf("gentle-start: inlet velocity ramps 0->%.2f over %.2f D/U (%d-col table)\n",
            parse(Float64, m["crossflow_u"]), uramp, Nr + 1)
end
@printf("crossflow GPU: grid %dx%dx%d  Ma=%.3f Kn=%.3f  order=%d  inlet(rho,rho*u)=(%.3f,%.3f)\n",
        nx, ny, nz, Ma, Kn, order, inlet[1], inlet[2])

# dt probe: one adaptive order-3 crossflow step on a scratch cube (matches the
# run_staged policy: dt = min(dtcap, 0.9 x probed CFL dt), then a constant dts).
scratch = GPURun.Timestep3DOrder3GPU.build_haloed_cube(CuArray(M0))
probe = GPURun.Timestep3DOrder3GPU.march3d_order3_gpu!(scratch, dx, Ma, 1;
            s3max=s3max, stage_bgk=true, Kn=Kn, bc=:crossflow, inlet=inlet)
CUDA.unsafe_free!(scratch)
dt = min(dtcap, 0.9 * probe[1])
dts = constant_dts(tmax, dt)
@printf("CFL dt = %.3e, dtcap = %.3e -> dt = %.3e (%d steps)  [%s]\n",
        probe[1], dtcap, dt, length(dts), CUDA.name(CUDA.device()))

mkpath("output/runs")
out = "output/runs/$(tag)_gpu.jld2"
t0 = time()
run_gpu_3d(M0, dx, Ma, length(dts);
    snapshot_interval = snap_int, snapshot_filename = out,
    dts = dts, order = order, bc = :crossflow, inlet = inlet, inlet_fn = inlet_fn, live_diag = true,
    Kn = Kn, scheme = :recommended, s3max = s3max, vacuum_floor = 0.0,
    params = Dict{String,Any}("case" => tag, "Ma" => Ma, "Kn" => Kn, "tmax" => tmax,
                              "scheme" => "recommended", "bc" => "crossflow", "order" => order,
                              "device" => CUDA.name(CUDA.device())),
    web_dir = "output")
el = time() - t0
@printf("done: %s  (%d steps, %.1f s wall, %.4f s/step)\n", out, length(dts), el, el / length(dts))

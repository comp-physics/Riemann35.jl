# staging_common.jl — shared pure-Base helpers for the CPU->GPU case handoff.
# Included by BOTH gpu/stage_case.jl (main package env) and gpu/run_staged.jl
# (gpuenv2), so the meta schema and dt-sequence construction exist once.

# Constant-dt sequence covering [0, tmax] with a trimmed final step.
function constant_dts(tmax, dt)
    nstep = ceil(Int, tmax / dt)
    dts = fill(Float64(dt), nstep)
    dts[end] = tmax - (nstep - 1) * dt
    return dts
end

# meta.txt schema (key=value lines, order-independent).
function write_stage_meta(path; kv...)
    open(path, "w") do io
        for (k, v) in kv
            println(io, k, "=", v)
        end
    end
end

function read_stage_meta(path)
    d = Dict{String,String}()
    for ln in eachline(path)
        k, v = split(ln, "="; limit = 2)
        d[strip(k)] = strip(v)
    end
    return d
end

# Stage a case for the GPU march: build the IC with the CPU runner (tmax = 0
# returns the gathered IC — case setup is never duplicated between paths) and
# write M0 + meta. `runner` is `simulation_runner`, passed as an argument so
# this file stays pure Base and loadable from the CUDA env too.
function stage_case(runner, c; dir = "output/stage_$(c.tag)")
    p = (; c.params..., tmax = 0.0)
    M = first(runner(p))                      # (Nx,Ny,Nz,35)
    M0 = permutedims(M, (4, 1, 2, 3))         # (35,nx,ny,nz) device layout
    mkpath(dir)
    xmin = get(c.params, :xmin, -0.5); xmax = get(c.params, :xmax, 0.5)
    extra = haskey(c.params, :s3max) ? (s3max = c.params.s3max,) : (;)
    # Gentle-start inlet ramp: if the case provides an inlet-velocity table, write
    # it + ramp metadata so the GPU driver can update the inlet Maxwellian over time.
    # (35, Nr+1) table; column k+1 = inlet Maxwellian at u = crossflow_u * k/Nr.
    itab = get(c, :inlet_table, nothing)
    ramp = get(c, :crossflow_uramp, 0.0)
    if itab !== nothing && ramp > 0
        write(joinpath(dir, "inlet_table.f64"), Float64.(itab))
        extra = (; extra..., crossflow_uramp = ramp,
                 crossflow_u = get(c, :crossflow_u, 1.0), n_table = size(itab, 2) - 1)
    end
    # Rigid immersed obstacle: emit its geometry in CELL-INDEX units so the GPU
    # driver can re-impose the held rest state (extracted from M0 at the centre).
    if get(c.params, :hold_obstacle, false)
        ymin = get(c.params, :ymin, -0.5); ymax = get(c.params, :ymax, 0.5)
        dxg = (xmax - xmin) / c.params.Nx; dyg = (ymax - ymin) / c.params.Ny
        oxc = get(c.params, :obstacle_xc, get(c.params, :bubble_xc, (xmin+xmax)/2))
        oyc = get(c.params, :obstacle_yc, get(c.params, :bubble_yc, (ymin+ymax)/2))
        orad = get(c.params, :obstacle_radius, get(c.params, :bubble_radius, 0.5))
        extra = (; extra...,
                 obst_cx = (oxc - xmin) / dxg + 0.5, obst_cy = (oyc - ymin) / dyg + 0.5,
                 obst_r_cells = orad / dxg)
    end
    # BC + non-reflecting sponge config for the GPU driver (default :crossflow for
    # back-compat with existing staged cases that omit `bc`).
    extra = (; extra..., bc = String(get(c.params, :bc, :crossflow)))
    if Float64(get(c.params, :sponge_rate, 0.0)) > 0 || haskey(c.params, :sponge_width)
        extra = (; extra..., sponge_width = Int(get(c.params, :sponge_width, 0)),
                 sponge_rate = Float64(get(c.params, :sponge_rate, 0.0)))
    end
    if Float64(get(c.params, :noise_amp, 0.0)) > 0
        extra = (; extra..., noise_amp = Float64(get(c.params, :noise_amp, 0.0)))
    end
    if Float64(get(c.params, :fluct_intensity, 0.0)) > 0
        extra = (; extra..., fluct_intensity = Float64(get(c.params, :fluct_intensity, 0.0)))
    end
    write_stage_meta(joinpath(dir, "meta.txt");
        nx = size(M0, 2), ny = size(M0, 3), nz = size(M0, 4),
        dx = (xmax - xmin) / c.params.Nx,
        Ma = c.params.Ma, Kn = c.params.Kn, tmax = c.params.tmax,
        dtcap = c.dtcap, snap_interval = c.snap_interval, tag = c.tag, extra...)
    write(joinpath(dir, "M0.f64"), M0)
    return dir
end

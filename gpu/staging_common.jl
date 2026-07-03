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
    write_stage_meta(joinpath(dir, "meta.txt");
        nx = size(M0, 2), ny = size(M0, 3), nz = size(M0, 4),
        dx = (xmax - xmin) / c.params.Nx,
        Ma = c.params.Ma, Kn = c.params.Kn, tmax = c.params.tmax,
        dtcap = c.dtcap, snap_interval = c.snap_interval, tag = c.tag)
    write(joinpath(dir, "M0.f64"), M0)
    return dir
end

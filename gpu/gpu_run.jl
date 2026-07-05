"""
    gpu_run.jl — GPU run + snapshot-dump driver.

Wraps the PURE on-device march (`march3d_gpu!` / `march3d_slab_gpu!`) and streams
snapshots to a JLD2 file in the SAME schema the CPU `run_simulation` writes, so the
existing readers / visualization (`src/visualization/interactive_3d_timeseries_streaming.jl`,
`examples/run_3d_jets_timeseries.jl`) open GPU output unchanged. The march stays
compute-only; this driver owns I/O.

Schema (matches simulation_runner):
  meta/params, meta/snapshot_interval, meta/n_snapshots
  snapshots/NNNNNN/{M, t, step}     # M is (Nx,Ny,Nz,35) — moment LAST (host layout)

Snapshots are STREAMED one at a time (never all in host memory). The resident GPU
field is host-staged only at snapshot times. `S`/`C` (standardized/central fields) are
NOT written here — they are derivable post-hoc from `M` via `compute_standardized_field`
/ `compute_central_field` in the main package env.

Single-GPU: `run_gpu_3d(M0, dx, Ma, nstep; snapshot_interval, snapshot_filename, …)`.
Multi-GPU (z-slab): pass `comm` (an MPI communicator) and this rank's slab interior as
`M0` (35,n,n,nz_loc); slabs are gathered to rank 0, which writes.
"""
module GPURun

using CUDA, JLD2, MPI
include(joinpath(@__DIR__, "timestep3d_gpu.jl"))
using .Timestep3DGPU: march3d_gpu!, march3d_slab_gpu!, HO_VACUUM_FLOOR_DEFAULT
include(joinpath(@__DIR__, "timestep3d_order3_gpu.jl"))   # order-3 WENO5+θ*-IDP single-GPU march (opt-in via order=3)
using .Timestep3DOrder3GPU: march3d_order3_gpu!, build_haloed_cube, interior_from_cube!
include(joinpath(@__DIR__, "..", "src", "web_export.jl"))   # browser-viewable export (single source; shared with the CPU path)
using .WebExport: maybe_export_web

export run_gpu_3d

# (35,nx,ny,nz) device field -> (nx,ny,nz,35) host array (snapshot layout)
_to_snapshot_layout(Md::CuArray{Float64,4}) = permutedims(Array(Md), (2, 3, 4, 1))

# write one snapshot record into an open JLD2 file
function _write_snap!(jf, idx::Int, Mhost::Array{Float64,4}, t::Float64, step::Int)
    key = lpad(idx, 6, '0')
    jf["snapshots/$key/M"] = Mhost
    jf["snapshots/$key/t"] = t
    jf["snapshots/$key/step"] = step
    return nothing
end

"""
    run_gpu_3d(M0, dx, Ma, nstep; snapshot_interval, snapshot_filename,
               comm=nothing, halo=2, dts=nothing, vacuum_floor=…, threads=128,
               params=Dict(), include_initial=true) -> snapshot_filename (or nothing off-rank-0)

Advance `M0` (host `(35,nx,ny,nz)`) for `nstep` SSP-RK3 steps on the GPU, dumping a
snapshot every `snapshot_interval` steps to `snapshot_filename` (JLD2). With `comm`
given, `M0` is this rank's z-slab interior and snapshots are the gathered global field.
"""
function run_gpu_3d(M0::Array{Float64,4}, dx::Real, Ma::Real, nstep::Integer;
                    snapshot_interval::Integer, snapshot_filename::AbstractString,
                    comm=nothing, halo::Int=2, dts=nothing, order::Int=2, proj_first_order::Bool=false, riemann_solver::Symbol=:hll, limiter::Bool=false,
                    scheme::Symbol=:recommended, pressure_recon=nothing, stage_bgk=nothing, Kn::Real=Inf,
                    s3max::Real=max(40.0, 4.0 + abs(Ma) / 2.0),
                    vacuum_floor::Real=HO_VACUUM_FLOOR_DEFAULT, threads::Int=128,
                    params=Dict{String,Any}(), include_initial::Bool=true, web_dir=nothing)
    @assert size(M0, 1) == 35 "M0 must be (35,nx,ny,nz)"
    @assert snapshot_interval >= 1 "snapshot_interval must be >= 1"
    # scheme bundle (matches the CPU runner): :recommended (the default) turns on
    # pressure_recon, and stage_bgk when a finite Kn is supplied (at the Kn=Inf
    # collisionless default a stage-BGK pass is a physical no-op that would only
    # add kernel launches and ulp churn). Explicit kwargs override. Evidence:
    # docs/design/scheme-graduation.md.
    scheme in (:legacy, :recommended) ||
        throw(ArgumentError("unknown scheme=$scheme; available :recommended (default), :legacy"))
    pressure_recon = pressure_recon === nothing ? (scheme === :recommended) : Bool(pressure_recon)
    stage_bgk      = stage_bgk      === nothing ? (scheme === :recommended && isfinite(Kn)) : Bool(stage_bgk)
    multigpu = comm !== nothing
    if multigpu
        rank = MPI.Comm_rank(comm); nranks = MPI.Comm_size(comm)
    else
        rank = 0; nranks = 1
    end
    writer = (rank == 0)

    n = size(M0, 2); nzloc = size(M0, 4)
    Md = CuArray(M0)
    dts_host = dts === nothing ? nothing : Float64.(collect(dts))

    # --- order-3 (WENO5 + θ*-IDP) single-GPU setup ------------------------------
    # The order-3 march operates on a g=4 outflow-haloed CUBE, not the interior
    # field. Build the cube ONCE from the interior; each segment marches it in place
    # (halos are refilled internally per stage) and syncs the interior back into `Md`
    # so snapshots/return keep the standard (35,nx,ny,nz) layout. Order-1/2 paths are
    # untouched (this whole block is skipped unless order==3).
    # order-3 single-GPU builds the cube here; order-3 MULTI-GPU is handled by the
    # z-slab march (march3d_slab_gpu!(order=3)) in the segment loop below.
    order3 = (order == 3)
    order3_single = order3 && !multigpu
    G3 = nothing
    if order3_single
        (size(M0, 3) == n && nzloc == n) ||
            error("order-3 GPU requires a cubic interior (nx==ny==nz); got interior $(size(M0)[2:4])")
        (limiter || proj_first_order || riemann_solver !== :hll) &&
            error("order-3 GPU path does not support the order-1/2 flux options " *
                  "(limiter/proj_first_order/riemann_solver); got limiter=$limiter " *
                  "proj_first_order=$proj_first_order riemann_solver=:$riemann_solver. " *
                  "Order-3 uses WENO5 reconstruction + a θ*-IDP Riemann layer.")
        G3 = build_haloed_cube(Md; threads=threads)   # shared interior→haloed-cube bridge
    end

    # --- gather this rank's slab to a global (nx,ny,Nz,35) host array on rank 0 ---
    function gather_global()
        if !multigpu
            return _to_snapshot_layout(Md)
        end
        sb = vec(Array(Md))                         # (35*n*n*nzloc) this rank
        counts = fill(35 * n * n * nzloc, nranks)
        rbuf = writer ? Vector{Float64}(undef, 35 * n * n * nzloc * nranks) : Float64[]
        MPI.Gatherv!(sb, writer ? MPI.VBuffer(rbuf, counts) : nothing, comm; root=0)
        writer || return nothing
        Nz = nzloc * nranks                          # rank order == z order
        return permutedims(reshape(rbuf, 35, n, n, Nz), (2, 3, 4, 1))
    end

    jf = writer ? jldopen(snapshot_filename, "w") : nothing
    if writer
        jf["meta/params"] = params
        jf["meta/snapshot_interval"] = snapshot_interval
    end

    snap = 0; t = 0.0; step = 0
    function dump!()
        G = gather_global()
        if writer
            snap += 1
            _write_snap!(jf, snap, G, t, step)
        end
        multigpu && MPI.Barrier(comm)
    end

    include_initial && dump!()                        # snapshot the IC
    while step < nstep
        k = min(snapshot_interval, nstep - step)
        seg = dts_host === nothing ? nothing : dts_host[step+1:step+k]
        used = if order3_single
            u = march3d_order3_gpu!(G3, dx, Ma, k; dts=seg, s3max=s3max,
                                    stage_bgk=stage_bgk, Kn=Kn, threads=threads)
            interior_from_cube!(Md, G3; threads=threads)   # sync interior for the snapshot
            u
        elseif multigpu
            march3d_slab_gpu!(Md, dx, Ma, k, comm; halo=halo, dts=seg,
                              vacuum_floor=vacuum_floor, order=order, proj_first_order=proj_first_order, riemann_solver=riemann_solver, limiter=limiter,
                              pressure_recon=pressure_recon, stage_bgk=stage_bgk, Kn=Kn, s3max=s3max, threads=threads)
        else
            march3d_gpu!(Md, dx, Ma, k; dts=seg, vacuum_floor=vacuum_floor, order=order, proj_first_order=proj_first_order, riemann_solver=riemann_solver, limiter=limiter,
                         pressure_recon=pressure_recon, stage_bgk=stage_bgk, Kn=Kn, s3max=s3max, threads=threads)
        end
        t += sum(used); step += k
        dump!()
    end

    if writer
        jf["meta/n_snapshots"] = snap
        close(jf)
        maybe_export_web(snapshot_filename, web_dir)
        return snapshot_filename
    end
    return nothing
end

end # module

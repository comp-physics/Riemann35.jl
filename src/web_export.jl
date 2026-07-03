"""
    web_export.jl — export simulation snapshots to a self-contained, browser-viewable
    bundle (the `casedata` web viewer: `viewer.html` + per-case JSON + manifest).

Open the bundle with any browser via a static server (a `serve.sh` is dropped in the
output dir). No JLD2/Julia needed on the viewing machine.

Public:
  * `export_web(outdir, name, snaps; Ma, Kn, rescap)` — from in-memory `(M,t)` snapshots
    (`M` sized `(Nx,Ny,Nz,35)`).
  * `export_jld2_web(jld2path, outdir; name, rescap)` — convert a saved snapshot JLD2.

Multiple cases can share one `outdir`: each `export_*` appends to `manifest.json`, so the
viewer's Case dropdown lists them all by name. The reduced field volumes (18 physical
fields per snapshot) are downsampled to `rescap` per axis (default 30) for responsiveness;
quasi-2D cases (Nz <= 4) export one z-plane at the finer `rescap2d` in-plane cap (default 512).
Cases are written as metadata JSON + one Float32 .bin per snapshot, fetched lazily by the viewer.
"""
module WebExport

using JLD2

export export_web, export_jld2_web, maybe_export_web

# Field schema — the single source for the viewer (serialized to fields.json).
# `s` (signed) selects the symmetric diverging colormap; `r` pins a fixed
# color range. Order defines the .bin field-major layout.
const WEB_FIELD_META = [
    (n = "density",), (n = "|velocity|",), (n = "u", s = true), (n = "v", s = true),
    (n = "w", s = true), (n = "temperature",), (n = "s110", s = true),
    (n = "s101", s = true), (n = "s011", s = true), (n = "‖s‖", r = (0.0, 1.7321)),
    (n = "skewness", s = true), (n = "kurtosis",), (n = "S2 margin",),
    (n = "pressure",), (n = "Mach",), (n = "Tx",), (n = "Ty",), (n = "Tz",),
]
const WEB_FIELDS = [f.n for f in WEB_FIELD_META]
const _ASSETS = joinpath(@__DIR__, "web")   # viewer.html, serve.sh

_nf(x) = (r = round(Float64(x); digits=4); isfinite(r) ? r : 0.0)
_nn(x) = (x === nothing || !(x isa Number) || !isfinite(x)) ? "null" : string(round(Float64(x); digits=4))

# 35 raw moments (M4 canonical order) -> 18 physical fields (0 where vacuum/unrealizable).
@inline function _fvec(m)
    rho = m[1]; rho <= 1e-9 && return ntuple(_ -> 0.0, 18)
    u = m[2]/rho; v = m[6]/rho; w = m[16]/rho; um = sqrt(u^2 + v^2 + w^2)
    c200 = m[3]/rho - u^2; c020 = m[10]/rho - v^2; c002 = m[20]/rho - w^2
    T = (c200 + c020 + c002)/3; p = rho*T; Mach = um/sqrt(max(T, 1e-30))
    (c200 > 1e-12 && c020 > 1e-12 && c002 > 1e-12) ||
        return (rho, um, u, v, w, T, 0., 0., 0., 0., 0., 0., 0., p, Mach, max(c200,0), max(c020,0), max(c002,0))
    c110 = m[7]/rho - u*v; c101 = m[17]/rho - u*w; c011 = m[26]/rho - v*w
    s110 = c110/sqrt(c200*c020); s101 = c101/sqrt(c200*c002); s011 = c011/sqrt(c020*c002)
    c300 = m[4]/rho - 3u*(m[3]/rho) + 2u^3
    c400 = m[5]/rho - 4u*(m[4]/rho) + 6u^2*(m[3]/rho) - 3u^4
    (rho, um, u, v, w, T, s110, s101, s011, sqrt(s110^2 + s101^2 + s011^2),
     c300/c200^1.5, c400/c200^2, 1 + 2*s110*s101*s011 - (s110^2 + s101^2 + s011^2),
     p, Mach, c200, c020, c002)
end

function _ensure_assets(outdir)
    mkpath(outdir)
    cp(joinpath(_ASSETS, "viewer.html"), joinpath(outdir, "viewer.html"); force=true)
    cp(joinpath(_ASSETS, "serve.sh"),    joinpath(outdir, "serve.sh");    force=true)
    cp(joinpath(_ASSETS, "README.md"),   joinpath(outdir, "README.md");   force=true)
    try; chmod(joinpath(outdir, "serve.sh"), 0o755); catch; end
    open(joinpath(outdir, "fields.json"), "w") do io
        ent = map(WEB_FIELD_META) do f
            e = "{\"name\":\"$(f.n)\""
            get(f, :s, false) && (e *= ",\"signed\":true")
            r = get(f, :r, nothing)
            r === nothing || (e *= ",\"range\":[$(_nf(r[1])),$(_nf(r[2]))]")
            e * "}"
        end
        print(io, "[", join(ent, ","), "]")
    end
end

# manifest = concatenation of the per-case `.meta` sidecars (append-friendly across runs)
function _rebuild_manifest(outdir)
    metas = sort(filter(f -> endswith(f, ".json.meta"), readdir(outdir)))
    entries = [strip(read(joinpath(outdir, m), String)) for m in metas]
    open(joinpath(outdir, "manifest.json"), "w") do io
        print(io, "[", join(entries, ","), "]")
    end
end

# snaps :: Vector of (M, t), M sized (Nx,Ny,Nz,35)
#
# Format 2 (the default for every case): the case JSON carries only metadata —
# snapshot times, view dims, per-field global ranges — plus one raw Float32
# (little-endian, field-major) .bin per snapshot that the viewer fetches
# LAZILY as the user scrubs. Full in-plane resolution up to `rescap2d`
# (default 512) for quasi-2D cases (Nz <= 4; single z-plane, the rest are
# duplicates) and up to `rescap` per axis (default 30) for 3D point-cloud
# views. A monolithic float-text JSON at 512^2 x 18 fields x ~20 snapshots
# would be ~700MB — unloadable; f32 bins are 4x smaller and parse-free.
function _write_case(outdir, name, Ma, Kn, snaps; rescap=30, rescap2d=512)
    NF = length(WEB_FIELDS)
    nx = ny = nz = 0; nsnap = length(snaps)
    smeta = String[]
    gl = fill(Inf, NF); gh = fill(-Inf, NF)
    for (si, (M, t)) in enumerate(snaps)
        nx, ny, nz = size(M, 1), size(M, 2), size(M, 3)
        cap = nz <= 4 ? rescap2d : rescap
        sx = max(1, cld(nx, cap)); sy = max(1, cld(ny, cap)); sz = max(1, cld(nz, cap))
        ix = collect(1:sx:nx); iy = collect(1:sy:ny); iz = nz <= 4 ? [1] : collect(1:sz:nz)
        vx, vy, vz = length(ix), length(iy), length(iz)
        V = [Vector{Float32}(undef, vx*vy*vz) for _ in 1:NF]; p = 1
        for k in iz, jj in iy, ii in ix        # column-major: ii fastest
            r = _fvec(@view M[ii, jj, k, :])
            for q in 1:NF
                V[q][p] = Float32(r[q])
                r[q] < gl[q] && (gl[q] = r[q]); r[q] > gh[q] && (gh[q] = r[q])
            end
            p += 1
        end
        bin = "case_$(name)_s$(lpad(si - 1, 3, '0')).bin"
        open(joinpath(outdir, bin), "w") do bio
            for q in 1:NF; write(bio, V[q]); end
        end
        push!(smeta, string("{\"t\":", round(Float64(t); digits=6),
                            ",\"vn\":[$vx,$vy,$vz],\"bin\":\"$bin\"}"))
    end
    open(joinpath(outdir, "case_$(name).json"), "w") do io
        print(io, "{\"name\":\"$name\",\"format\":2,\"Ma\":", _nn(Ma), ",\"Kn\":", _nn(Kn),
              ",\"grange\":[", join(("[$(_nf(gl[q])),$(_nf(gh[q]))]" for q in 1:NF), ","), "]",
              ",\"snaps\":[", join(smeta, ","), "]}")
    end
    open(joinpath(outdir, "case_$(name).json.meta"), "w") do mio
        print(mio, "{\"file\":\"$name\",\"Ma\":$(_nn(Ma)),\"Kn\":$(_nn(Kn)),\"nx\":$nx,\"ny\":$ny,\"nz\":$nz,\"nsnap\":$nsnap}")
    end
    nsnap
end

"""
    export_web(outdir, name, snaps; Ma=nothing, Kn=nothing, rescap=30, rescap2d=512)

Write/append a browser-viewable case from in-memory snapshots. The bundle is placed in
`outdir/viz/` (the viewer files live in their own dir, separate from raw run data);
returns that path. `snaps` is a vector of `(M, t)` with `M` sized `(Nx,Ny,Nz,35)`.
"""
# The browseable bundle always lives in a `viz/` subdir (kept separate from raw run
# data like `runs/`). Passing a dir already named `viz` is used as-is (no `viz/viz`).
_vizdir(dir) = basename(rstrip(String(dir), '/')) == "viz" ? String(dir) : joinpath(String(dir), "viz")

function export_web(outdir, name, snaps; Ma=nothing, Kn=nothing, rescap=30, rescap2d=512)
    vd = _vizdir(outdir)
    _ensure_assets(vd)
    ns = _write_case(vd, string(name), Ma, Kn, snaps; rescap=rescap, rescap2d=rescap2d)
    _rebuild_manifest(vd)
    @info "web viewer bundle written" case=name viz=vd snapshots=ns serve="run ./serve.sh in $vd"
    vd
end

"""
    export_jld2_web(jld2path, outdir; name=nothing, rescap=30, rescap2d=512)

Convert a saved snapshot JLD2 into a browser-viewable case in `outdir` (case name
defaults to the file's basename). Reads Ma/Kn from `meta/params` or the filename.
"""
function export_jld2_web(jld2path, outdir; name=nothing, rescap=30, rescap2d=512)
    nm = name === nothing ? replace(basename(jld2path), ".jld2" => "") : string(name)
    JLD2.jldopen(jld2path) do f
        haskey(f, "snapshots") || error("no /snapshots group in $jld2path")
        par = try; (haskey(f, "meta") && haskey(f["meta"], "params")) ? f["meta"]["params"] : nothing; catch; nothing; end
        gp(k) = try; par isa AbstractDict ? get(par, k, get(par, Symbol(k), nothing)) : getproperty(par, Symbol(k)); catch; nothing; end
        Ma = gp("Ma"); Kn = gp("Kn")
        if Ma === nothing
            mm = match(r"Ma([0-9.]+)", basename(jld2path)); Ma = mm === nothing ? nothing : parse(Float64, mm.captures[1])
        end
        sk = sort(collect(keys(f["snapshots"])))
        snaps = [(f["snapshots"][k]["M"], f["snapshots"][k]["t"]) for k in sk]
        export_web(outdir, nm, snaps; Ma=Ma, Kn=Kn, rescap=rescap, rescap2d=rescap2d)
    end
end

"""
    maybe_export_web(snapshot_filename, web_dir)

Shared opt-in run hook for both the CPU (`simulation_runner`) and GPU (`run_gpu_3d`)
paths: if `web_dir !== nothing`, export the just-saved snapshot JLD2 to a
browser-viewable bundle there. Never throws — a failed export must not break an
otherwise-completed simulation (the JLD2 is already on disk). No-op if `web_dir` is nothing.
"""
function maybe_export_web(snapshot_filename, web_dir)
    web_dir === nothing && return nothing
    try
        export_jld2_web(snapshot_filename, web_dir)
        @info "web viewer bundle written" web_dir=web_dir serve="run ./serve.sh in $web_dir"
    catch e
        @warn "web_dir export failed (snapshot JLD2 still saved)" exception=e
    end
    nothing
end

end # module

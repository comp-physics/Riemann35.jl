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
fields per snapshot) are downsampled to `rescap` per axis (default 30) for responsiveness.
"""
module WebExport

using JLD2

export export_web, export_jld2_web

const WEB_FIELDS = ["density","|velocity|","u","v","w","temperature","s110","s101","s011",
                    "‖s‖","skewness","kurtosis","S2 margin","pressure","Mach","Tx","Ty","Tz"]
const _ASSETS = joinpath(@__DIR__, "web")   # viewer.html, serve.sh

_nf(x) = (r = round(Float64(x); digits=4); isfinite(r) ? r : 0.0)
_j(v)  = string("[", join((_nf(x) for x in v), ","), "]")
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
    try; chmod(joinpath(outdir, "serve.sh"), 0o755); catch; end
    open(joinpath(outdir, "fields.json"), "w") do io
        print(io, "[\"", join(WEB_FIELDS, "\",\""), "\"]")
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
function _write_case(outdir, name, Ma, Kn, snaps; rescap=30)
    NF = length(WEB_FIELDS)
    io = open(joinpath(outdir, "case_$(name).json"), "w")
    print(io, "{\"name\":\"$name\",\"Ma\":", _nn(Ma), ",\"Kn\":", _nn(Kn), ",\"snaps\":[")
    nx = ny = nz = vx = vy = vz = 0; nsnap = 0
    for (si, (M, t)) in enumerate(snaps)
        nx, ny, nz = size(M, 1), size(M, 2), size(M, 3)
        sx = max(1, cld(nx, rescap)); sy = max(1, cld(ny, rescap)); sz = max(1, cld(nz, rescap))
        ix = collect(1:sx:nx); iy = collect(1:sy:ny); iz = collect(1:sz:nz)
        vx, vy, vz = length(ix), length(iy), length(iz)
        V = [Vector{Float64}(undef, vx*vy*vz) for _ in 1:NF]; p = 1
        for k in iz, jj in iy, ii in ix        # column-major: ii fastest
            r = _fvec(@view M[ii, jj, k, :]); for q in 1:NF; V[q][p] = r[q]; end; p += 1
        end
        print(io, si > 1 ? "," : "", "{\"t\":", round(Float64(t); digits=6), ",\"vn\":[$vx,$vy,$vz],\"F\":[",
              join((_j(V[q]) for q in 1:NF), ","), "]}")
        nsnap += 1
    end
    print(io, "]}"); close(io)
    open(joinpath(outdir, "case_$(name).json.meta"), "w") do mio
        print(mio, "{\"file\":\"$name\",\"Ma\":$(_nn(Ma)),\"Kn\":$(_nn(Kn)),\"nx\":$nx,\"ny\":$ny,\"nz\":$nz,\"nsnap\":$nsnap}")
    end
    nsnap
end

"""
    export_web(outdir, name, snaps; Ma=nothing, Kn=nothing, rescap=30)

Write/append a browser-viewable case to `outdir` from in-memory snapshots.
`snaps` is a vector of `(M, t)` with `M` sized `(Nx,Ny,Nz,35)`.
"""
function export_web(outdir, name, snaps; Ma=nothing, Kn=nothing, rescap=30)
    _ensure_assets(outdir)
    ns = _write_case(outdir, string(name), Ma, Kn, snaps; rescap=rescap)
    _rebuild_manifest(outdir)
    @info "web viewer bundle written" case=name outdir=outdir snapshots=ns serve="run ./serve.sh in $outdir"
    outdir
end

"""
    export_jld2_web(jld2path, outdir; name=nothing, rescap=30)

Convert a saved snapshot JLD2 into a browser-viewable case in `outdir` (case name
defaults to the file's basename). Reads Ma/Kn from `meta/params` or the filename.
"""
function export_jld2_web(jld2path, outdir; name=nothing, rescap=30)
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
        export_web(outdir, nm, snaps; Ma=Ma, Kn=Kn, rescap=rescap)
    end
end

end # module

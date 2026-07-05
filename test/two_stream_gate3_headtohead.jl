# two_stream_gate3_headtohead.jl — Gate 3 (3D head-to-head).
#
# One dimensional-split first-order finite-volume driver, run two ways on the SAME
# crossing-jets config (48x48x16), so the comparison is apples-to-apples:
#
#   PRODUCTION (single-stream): one 35-moment field; Rusanov faces from the
#     production closure `flux_closure35_dev`; realizability enforced by the
#     production projection `realizable_3D_M4` (firings counted).
#
#   TWO-STREAM: two 35-moment fields (+/-) split at gauge c=0; donor-cell x-flux
#     (`xflux_plus35`/`xflux_minus35`, no projection), Rusanov y/z per stream from
#     the production closure, and the BGK stream coupling. Chain clips counted.
#
# Reports, per Mach number: min realizability margins, projection firings vs chain
# clips, interpenetration structure, and wall time.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, Printf, LinearAlgebra

const N35 = NTuple{35,Float64}
const VACR = 1e-10
const VACFLOOR = 1e-8          # stream density floor (positivity safeguard; NOT projection)
const flux_closure35_dev = Riemann35.flux_closure35_dev
const _VACSTATE = (mv = InitializeM4_35(VACFLOOR, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 1e-6, 0.0, 1e-6);
                   ntuple(q -> mv[q], 35))::N35
const _Z35 = ntuple(_ -> 0.0, 35)::N35

# vacuum/positivity safeguard for a stream: floor near-vacuum or non-finite cells to a
# tiny isotropic Maxwellian so the transverse closure / BGK never see a degenerate state.
@inline sanitize(M::N35) = (all(isfinite, M) && M[1] > VACFLOOR) ? M : _VACSTATE
# transverse (y,z) production fluxes, guarded at the vacuum floor.
@inline fyfz(M::N35) = (all(isfinite, M) && M[1] > VACFLOOR) ?
    (F = flux_closure35_dev(M...); (ntuple(q -> F[35+q], 35), ntuple(q -> F[70+q], 35))) :
    (_Z35, _Z35)

@inline function wavespeed(M::N35)
    ρ = M[1]; ρ <= VACR && return 0.0
    u = M[2]/ρ; v = M[6]/ρ; w = M[16]/ρ
    T = max((M[3]/ρ-u^2 + M[10]/ρ-v^2 + M[20]/ρ-w^2)/3, 1e-12)
    return max(abs(u), abs(v), abs(w)) + 3*sqrt(T)
end

# ------------------------------------------------------------------ IC
function crossing_ic(Ma; Nx=48, Ny=48, Nz=16)
    xs = range(-0.5, 0.5; length=Nx); ys = range(-0.5, 0.5; length=Ny); zs = range(-0.25, 0.25; length=Nz)
    dx = step(xs); dy = step(ys); dz = step(zs)
    Uc = Ma/sqrt(2.0); jw = 0.12; off = 0.15
    rhol = 1.0; rhor = 1e-3; T = 1.0
    cell(ρ, u, v, w) = (mv = InitializeM4_35(ρ, u, v, w, T, 0.0, 0.0, T, 0.0, T); ntuple(q -> mv[q], 35))
    M = Array{N35}(undef, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = xs[i]; y = ys[j]
        injet1 = abs(x+off) <= jw/2 && abs(y+off) <= jw/2     # SW-origin jet moving NE (+,+)
        injet2 = abs(x-off) <= jw/2 && abs(y-off) <= jw/2     # NE-origin jet moving SW (-,-)
        M[i,j,k] = injet1 ? cell(rhol, Uc, Uc, 0.0) :
                   injet2 ? cell(rhol, -Uc, -Uc, 0.0) :
                            cell(rhor, 0.0, 0.0, 0.0)
    end
    return M, dx, dy, dz
end

# ------------------------------------------------- production single-stream run
function run_single(Ma; nsteps=60, CFL=0.2, Kn=1000.0)
    M, dx, dy, dz = crossing_ic(Ma)
    Nx, Ny, Nz = size(M)
    Fx = similar(M); Fy = similar(M); Fz = similar(M)
    h = min(dx, dy, dz)
    fired_total = 0; min_margin = Inf
    t0 = time()
    for step in 1:nsteps
        α = 0.0
        for c in M; α = max(α, wavespeed(c)); end
        dt = CFL*h/max(α, 1e-12)
        for idx in eachindex(M)
            F = flux_closure35_dev(M[idx]...)
            Fx[idx] = ntuple(q -> F[q], 35); Fy[idx] = ntuple(q -> F[35+q], 35); Fz[idx] = ntuple(q -> F[70+q], 35)
        end
        Mnew = copy(M)
        for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
            fxR = 0.5 .* (Fx[i,j,k] .+ Fx[i+1,j,k]) .- (0.5α) .* (M[i+1,j,k] .- M[i,j,k])
            fxL = 0.5 .* (Fx[i-1,j,k] .+ Fx[i,j,k]) .- (0.5α) .* (M[i,j,k] .- M[i-1,j,k])
            fyR = 0.5 .* (Fy[i,j,k] .+ Fy[i,j+1,k]) .- (0.5α) .* (M[i,j+1,k] .- M[i,j,k])
            fyL = 0.5 .* (Fy[i,j-1,k] .+ Fy[i,j,k]) .- (0.5α) .* (M[i,j,k] .- M[i,j-1,k])
            fzR = 0.5 .* (Fz[i,j,k] .+ Fz[i,j,k+1]) .- (0.5α) .* (M[i,j,k+1] .- M[i,j,k])
            fzL = 0.5 .* (Fz[i,j,k-1] .+ Fz[i,j,k]) .- (0.5α) .* (M[i,j,k] .- M[i,j,k-1])
            Mnew[i,j,k] = M[i,j,k] .- dt .* ((fxR .- fxL)./dx .+ (fyR .- fyL)./dy .+ (fzR .- fzL)./dz)
        end
        # BGK + production realizability projection (count firings)
        for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
            c = collect(Mnew[i,j,k])
            c = collision35(c, dt, Kn)
            mg = realizability_margin(c)
            if mg < 0
                fired_total += 1
                c = realizable_3D_M4(c, Ma)
                mg = realizability_margin(c)
            end
            c[1] > 0.05 && (min_margin = min(min_margin, mg))
            Mnew[i,j,k] = ntuple(q -> c[q], 35)
        end
        M = Mnew
    end
    wall = time() - t0
    return (fired=fired_total, min_margin=min_margin, wall=wall, M=M)
end

# ------------------------------------------------------------ two-stream run
function run_twostream(Ma; nsteps=60, CFL=0.2, Kn=1000.0, c=0.0)
    M0, dx, dy, dz = crossing_ic(Ma)
    Nx, Ny, Nz = size(M0)
    Mp = Array{N35}(undef, Nx, Ny, Nz); Mm = similar(Mp)
    for idx in eachindex(M0)
        ρ = M0[idx][1]; u = M0[idx][2]/ρ; v = M0[idx][6]/ρ; w = M0[idx][16]/ρ
        T = max((M0[idx][3]/ρ-u^2 + M0[idx][10]/ρ-v^2 + M0[idx][20]/ρ-w^2)/3, 1e-12)
        Mp[idx], Mm[idx] = split_maxwellian35(ρ, u, v, w, T, c)
    end
    Xp = similar(Mp); Xm = similar(Mp)
    FyP = similar(Mp); FzP = similar(Mp); FyM = similar(Mp); FzM = similar(Mp)
    h = min(dx, dy, dz)
    reset_chain_clips!()
    min_margin = Inf; min_tot_margin = Inf; interpen = 0; str_proj = 0
    t0 = time()
    for step in 1:nsteps
        α = 0.0
        for idx in eachindex(Mp); α = max(α, wavespeed(Mp[idx]), wavespeed(Mm[idx])); end
        dt = CFL*h/max(α, 1e-12)
        for idx in eachindex(Mp)
            Xp[idx] = xflux_plus35(Mp[idx]...); Xm[idx] = xflux_minus35(Mm[idx])
            FyP[idx], FzP[idx] = fyfz(Mp[idx]); FyM[idx], FzM[idx] = fyfz(Mm[idx])
        end
        Np = copy(Mp); Nm = copy(Mm)
        for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
            # + stream: donor-cell x (inflow from left) + Rusanov y/z
            xdivP = (Xp[i,j,k] .- Xp[i-1,j,k]) ./ dx
            fyR = 0.5 .* (FyP[i,j,k] .+ FyP[i,j+1,k]) .- (0.5α) .* (Mp[i,j+1,k] .- Mp[i,j,k])
            fyL = 0.5 .* (FyP[i,j-1,k] .+ FyP[i,j,k]) .- (0.5α) .* (Mp[i,j,k] .- Mp[i,j-1,k])
            fzR = 0.5 .* (FzP[i,j,k] .+ FzP[i,j,k+1]) .- (0.5α) .* (Mp[i,j,k+1] .- Mp[i,j,k])
            fzL = 0.5 .* (FzP[i,j,k-1] .+ FzP[i,j,k]) .- (0.5α) .* (Mp[i,j,k] .- Mp[i,j,k-1])
            Np[i,j,k] = Mp[i,j,k] .- dt .* (xdivP .+ (fyR .- fyL)./dy .+ (fzR .- fzL)./dz)
            # - stream: donor-cell x (inflow from right) + Rusanov y/z
            xdivM = (Xm[i+1,j,k] .- Xm[i,j,k]) ./ dx
            gyR = 0.5 .* (FyM[i,j,k] .+ FyM[i,j+1,k]) .- (0.5α) .* (Mm[i,j+1,k] .- Mm[i,j,k])
            gyL = 0.5 .* (FyM[i,j-1,k] .+ FyM[i,j,k]) .- (0.5α) .* (Mm[i,j,k] .- Mm[i,j-1,k])
            gzR = 0.5 .* (FzM[i,j,k] .+ FzM[i,j,k+1]) .- (0.5α) .* (Mm[i,j,k+1] .- Mm[i,j,k])
            gzL = 0.5 .* (FzM[i,j,k-1] .+ FzM[i,j,k]) .- (0.5α) .* (Mm[i,j,k] .- Mm[i,j,k-1])
            Nm[i,j,k] = Mm[i,j,k] .- dt .* (xdivM .+ (gyR .- gyL)./dy .+ (gzR .- gzL)./dz)
        end
        # vacuum safeguard + production transverse (y,z) realizability correction per
        # stream (the spec keeps the production hyperbolicity path for those directions;
        # the x-direction is realizability-safe by donor-cell construction) + BGK.
        for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
            p = sanitize(Np[i,j,k]); m = sanitize(Nm[i,j,k])
            if p[1] > 0.05
                cp = collect(p)
                if realizability_margin(cp) < 0
                    str_proj += 1; cp = realizable_3D_M4(cp, Ma); p = ntuple(q -> cp[q], 35)
                end
            end
            if m[1] > 0.05
                cm = collect(m)
                if realizability_margin(cm) < 0
                    str_proj += 1; cm = realizable_3D_M4(cm, Ma); m = ntuple(q -> cm[q], 35)
                end
            end
            Np[i,j,k], Nm[i,j,k] = bgk_stream_relax(p, m, dt, Kn, c)
        end
        Mp = Np; Mm = Nm
    end
    wall = time() - t0
    # metrics on the final state
    for k in 2:size(Mp,3)-1, j in 2:size(Mp,2)-1, i in 2:size(Mp,1)-1
        p = Mp[i,j,k]; m = Mm[i,j,k]
        p[1] > 0.05 && (min_margin = min(min_margin, realizability_margin(collect(p))))
        m[1] > 0.05 && (min_margin = min(min_margin, realizability_margin(collect(m))))
        tot = p .+ m
        tot[1] > 0.05 && (min_tot_margin = min(min_tot_margin, realizability_margin(collect(tot))))
        (p[1] > 0.05 && m[1] > 0.05) && (interpen += 1)
    end
    return (clips=chain_clips(), str_proj=str_proj, min_margin=min_margin, min_tot_margin=min_tot_margin,
            interpen=interpen, wall=wall, Mp=Mp, Mm=Mm)
end

# ----------------------------------------------------------------------- main
function main()
    # per-Ma settings: Ma=5 is comfortably resolved at CFL=0.2; Ma=100 is a known-hard
    # regime, so this standalone driver's first-order transverse Rusanov needs CFL=0.1.
    cases = length(ARGS) >= 1 ? [(parse(Float64, ARGS[1]),
                                  length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.2,
                                  length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 60)] :
                                 [(5.0, 0.2, 60), (100.0, 0.1, 30)]
    println("=== Gate 3: two-stream vs production, crossing jets 48x48x16 ===")
    @printf("%-6s | %-40s | %-56s | %s\n", "Ma", "PRODUCTION (single-stream)", "TWO-STREAM", "wall x")
    for (Ma, CFL, nsteps) in cases
        s = run_single(Ma; nsteps=nsteps, CFL=CFL)
        d = run_twostream(Ma; nsteps=nsteps, CFL=CFL)
        @printf("Ma=%-3g | proj_fires=%-8d min_margin=%+.2e | x_clips=%-6d yz_proj=%-6d min_str_margin=%+.2e tot_margin=%+.2e interpen=%-5d | %.2fx\n",
                Ma, s.fired, s.min_margin, d.clips, d.str_proj, d.min_margin, d.min_tot_margin, d.interpen, d.wall/s.wall)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

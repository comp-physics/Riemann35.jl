# two_stream_gate3_headtohead.jl — Gate 3 (3D head-to-head).
#
# One dimensional-split finite-volume driver, run two ways on the SAME crossing-jets
# config (48x48x16), sharing the production transverse (HLL) machinery so the only
# difference is the split axis:
#
#   PRODUCTION (single-stream): one 35-moment field; every direction uses the
#     production path — `Flux_closure35_and_realizable_3D` fluxes, the
#     `eigenvalues6{,z}_hyperbolic_3D` wave speeds, and `pas_HLL` — plus the
#     production realizability projection `realizable_3D_M4` (firings counted).
#
#   TWO-STREAM: two 35-moment fields (+/-) split at gauge c=0. The split axis (x)
#     uses the donor-cell half-space stream flux (`xflux_plus35`/`xflux_minus35`, no
#     projection — realizability by construction). The transverse axes (y,z) use the
#     UNCHANGED production path per stream (Component 3 of the spec: a half-space-in-x
#     measure is an ordinary full-line measure in y,z). BGK couples the streams.
#     x chain-clips and per-stream transverse corrections are counted.
#
# Reports, per Mach number: projection firings vs chain clips, min realizability
# margins, interpenetration structure, and wall time.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, Printf, LinearAlgebra

const N35 = NTuple{35,Float64}
const VACR = 1e-10
const VACFLOOR = 1e-8
const HLL_FULL = 1e-2   # cells below this density use the cheap closure (physics negligible)
const flux_closure35_dev = Riemann35.flux_closure35_dev
const _VACSTATE = collect(InitializeM4_35(VACFLOOR, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 1e-6, 0.0, 1e-6))

@inline function wavespeed(m)
    ρ = m[1]; ρ <= VACR && return 0.0
    u = m[2]/ρ; v = m[6]/ρ; w = m[16]/ρ
    T = max((m[3]/ρ-u^2 + m[10]/ρ-v^2 + m[20]/ρ-w^2)/3, 1e-12)
    return max(abs(u), abs(v), abs(w)) + 3*sqrt(T)
end

# sanitize a near-vacuum / non-finite state to a tiny isotropic Maxwellian (positivity
# safeguard; NOT a realizability projection).
@inline function sanitize!(v::AbstractVector)
    (all(isfinite, v) && v[1] > VACFLOOR) || (v .= _VACSTATE)
    return v
end

# ------------------------------------------------------------------ IC
function crossing_ic(Ma; Nx=48, Ny=48, Nz=16)
    xs = range(-0.5, 0.5; length=Nx); ys = range(-0.5, 0.5; length=Ny); zs = range(-0.25, 0.25; length=Nz)
    dx = step(xs); dy = step(ys); dz = step(zs)
    Uc = Ma/sqrt(2.0); jw = 0.12; off = 0.15
    rhol = 1.0; rhor = 1e-3; T = 1.0
    cell(ρ, u, v, w) = InitializeM4_35(ρ, u, v, w, T, 0.0, 0.0, T, 0.0, T)
    M = Array{Float64}(undef, Nx, Ny, Nz, 35)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = xs[i]; y = ys[j]
        injet1 = abs(x+off) <= jw/2 && abs(y+off) <= jw/2
        injet2 = abs(x-off) <= jw/2 && abs(y-off) <= jw/2
        c = injet1 ? cell(rhol, Uc, Uc, 0.0) :
            injet2 ? cell(rhol, -Uc, -Uc, 0.0) : cell(rhor, 0.0, 0.0, 0.0)
        @views M[i,j,k,:] .= c
    end
    return M, dx, dy, dz
end

@inline tup35(M, i, j, k) = ntuple(q -> @inbounds(M[i,j,k,q]), 35)

# global CFL time step from a cheap wave-speed bound (identical for both schemes)
function global_dt(fields, dx, dy, dz, CFL)
    α = 0.0
    for M in fields
        Nx, Ny, Nz, _ = size(M)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            α = max(α, wavespeed(tup35(M, i, j, k)))
        end
    end
    return CFL*min(dx, dy, dz)/max(α, 1e-12)
end

# two-stream dt: the donor-cell x-CFL is set by the TRUE stream support node
# (stream_xspeed), the transverse by the wave-speed bound; take the tighter.
function global_dt_2s(Mp, Mm, dx, dy, dz, CFL)
    sx = 1e-12; syz = 1e-12
    Nx, Ny, Nz, _ = size(Mp)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        tp = tup35(Mp, i, j, k); tm = tup35(Mm, i, j, k)
        sx = max(sx, stream_xspeed(tp), stream_xspeed(tm))
        syz = max(syz, wavespeed(tp), wavespeed(tm))
    end
    return CFL*min(dx/sx, dy/syz, dz/syz)
end

# ---------------------------------------------------------------------------
# One production HLL sweep along `axis` on a 35-moment field M (in place).
# Fluxes + wave speeds from the production path; realizability-preserving.
# ---------------------------------------------------------------------------
function hll_sweep!(M::Array{Float64,4}, dt, ds, axis::Int, Ma)
    Nx, Ny, Nz, _ = size(M)
    axis == 3 && Nz == 1 && return
    F = Array{Float64}(undef, Nx, Ny, Nz, 35)
    vmin = Array{Float64}(undef, Nx, Ny, Nz); vmax = similar(vmin)
    off = axis == 1 ? 0 : axis == 2 ? 35 : 70
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mt = tup35(M, i, j, k)
        if mt[1] < HLL_FULL
            # near-vacuum background: the production realizability loop is wasted here
            # (flux is negligible); use the cheap alloc-free closure + wave-speed bound.
            Fdev = flux_closure35_dev(mt...)
            @inbounds for q in 1:35; F[i,j,k,q] = Fdev[off+q]; end
            ws = wavespeed(mt); vmin[i,j,k] = -ws; vmax[i,j,k] = ws
        else
            m = collect(mt)
            Fx, Fy, Fz, _ = Flux_closure35_and_realizable_3D(m, 0, Ma)
            Fsel = axis == 1 ? Fx : axis == 2 ? Fy : Fz
            @views F[i,j,k,:] .= Fsel
            lo, hi, _ = axis == 3 ? eigenvalues6z_hyperbolic_3D(m, 0, Ma) :
                                    eigenvalues6_hyperbolic_3D(m, axis, 0, Ma)
            vmin[i,j,k] = isfinite(lo) ? lo : -wavespeed(m)
            vmax[i,j,k] = isfinite(hi) ? hi :  wavespeed(m)
        end
    end
    Mnew = copy(M)
    if axis == 1
        for k in 1:Nz, j in 1:Ny
            @views Mnew[:,j,k,:] .= pas_HLL(M[:,j,k,:], F[:,j,k,:], dt, ds, vmin[:,j,k], vmax[:,j,k])
        end
    elseif axis == 2
        for k in 1:Nz, i in 1:Nx
            @views Mnew[i,:,k,:] .= pas_HLL(M[i,:,k,:], F[i,:,k,:], dt, ds, vmin[i,:,k], vmax[i,:,k])
        end
    else
        for j in 1:Ny, i in 1:Nx
            @views Mnew[i,j,:,:] .= pas_HLL(M[i,j,:,:], F[i,j,:,:], dt, ds, vmin[i,j,:], vmax[i,j,:])
        end
    end
    M .= Mnew
    return
end

# ------------------------------------------------- production single-stream run
function run_single(Ma; nsteps=40, CFL=0.2, Kn=1000.0)
    M, dx, dy, dz = crossing_ic(Ma)
    Nx, Ny, Nz, _ = size(M)
    fired = 0; min_margin = Inf
    t0 = time()
    for step in 1:nsteps
        dt = global_dt((M,), dx, dy, dz, CFL)
        hll_sweep!(M, dt, dx, 1, Ma)
        hll_sweep!(M, dt, dy, 2, Ma)
        hll_sweep!(M, dt, dz, 3, Ma)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            m = collect(tup35(M, i, j, k))
            m = collision35(m, dt, Kn)
            mg = realizability_margin(m)
            if mg < 0
                fired += 1; m = realizable_3D_M4(m, Ma); mg = realizability_margin(m)
            end
            m[1] > 0.05 && (min_margin = min(min_margin, mg))
            @views M[i,j,k,:] .= m
        end
    end
    return (fired=fired, min_margin=min_margin, wall=time()-t0)
end

# is a stream cell's x-marginal chain-realizable? (mirror the − stream to [0,∞))
@inline function stream_chain_ok(M, i, j, k, isplus)
    m0 = M[i,j,k,1]
    marg = isplus ? (m0, M[i,j,k,2], M[i,j,k,3], M[i,j,k,4], M[i,j,k,5]) :
                    (m0, -M[i,j,k,2], M[i,j,k,3], -M[i,j,k,4], M[i,j,k,5])
    return chain_realizable(chain(marg))
end

# ------------------------------------------------------------ two-stream run
function run_twostream(Ma; nsteps=40, CFL=0.2, Kn=1000.0, c=0.0)
    M0, dx, dy, dz = crossing_ic(Ma)
    Nx, Ny, Nz, _ = size(M0)
    Mp = Array{Float64}(undef, Nx, Ny, Nz, 35); Mm = similar(Mp)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        m = tup35(M0, i, j, k); ρ = m[1]
        u = m[2]/ρ; v = m[6]/ρ; w = m[16]/ρ
        T = max((m[3]/ρ-u^2 + m[10]/ρ-v^2 + m[20]/ρ-w^2)/3, 1e-12)
        p, mm = split_maxwellian35(ρ, u, v, w, T, c)
        @views Mp[i,j,k,:] .= p; @views Mm[i,j,k,:] .= mm
    end
    reset_chain_clips!()
    str_proj = 0; bulk_clips = 0; min_margin = Inf; interpen = 0
    Xp = Array{Float64}(undef, Nx, Ny, Nz, 35); Xm = similar(Xp)
    t0 = time()
    for step in 1:nsteps
        dt = global_dt_2s(Mp, Mm, dx, dy, dz, CFL)
        # x-sweep: donor-cell half-space stream flux (no projection)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            @views Xp[i,j,k,:] .= xflux_plus35(tup35(Mp, i, j, k)...)
            @views Xm[i,j,k,:] .= xflux_minus35(tup35(Mm, i, j, k))
        end
        for k in 1:Nz, j in 1:Ny, i in 2:Nx-1, q in 1:35
            Mp[i,j,k,q] -= (dt/dx)*(Xp[i,j,k,q] - Xp[i-1,j,k,q])   # + inflow from left
            Mm[i,j,k,q] -= (dt/dx)*(Xm[i+1,j,k,q] - Xm[i,j,k,q])   # − inflow from right
        end
        # y/z-sweeps: UNCHANGED production HLL path, per stream
        hll_sweep!(Mp, dt, dy, 2, Ma); hll_sweep!(Mm, dt, dy, 2, Ma)
        hll_sweep!(Mp, dt, dz, 3, Ma); hll_sweep!(Mm, dt, dz, 3, Ma)
        # transverse realizability correction (production path) + BGK coupling
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            p = sanitize!(collect(tup35(Mp, i, j, k)))
            m = sanitize!(collect(tup35(Mm, i, j, k)))
            if p[1] > 0.05 && realizability_margin(p) < 0
                str_proj += 1; p = realizable_3D_M4(p, Ma)
            end
            if m[1] > 0.05 && realizability_margin(m) < 0
                str_proj += 1; m = realizable_3D_M4(m, Ma)
            end
            pp, mm = bgk_stream_relax(ntuple(q -> p[q], 35), ntuple(q -> m[q], 35), dt, Kn, c)
            (all(isfinite, pp) && pp[1] > VACFLOOR) || (pp = ntuple(q -> _VACSTATE[q], 35))
            (all(isfinite, mm) && mm[1] > VACFLOOR) || (mm = ntuple(q -> _VACSTATE[q], 35))
            @views Mp[i,j,k,:] .= pp; @views Mm[i,j,k,:] .= mm
        end
        # per-step diagnostics: bulk chain-clips (x-realizability of mass-bearing streams),
        # peak interpenetration (jets pass through then separate, so this peaks mid-crossing),
        # and min stream realizability margin.
        ip = 0
        for k in 1:Nz, j in 1:Ny, i in 2:Nx-1
            pm = Mp[i,j,k,1]; mm = Mm[i,j,k,1]
            pm > 0.05 && !stream_chain_ok(Mp, i, j, k, true) && (bulk_clips += 1)
            mm > 0.05 && !stream_chain_ok(Mm, i, j, k, false) && (bulk_clips += 1)
            (pm > 0.05 && mm > 0.05) && (ip += 1)
            pm > 0.05 && (min_margin = min(min_margin, realizability_margin(collect(tup35(Mp, i, j, k)))))
            mm > 0.05 && (min_margin = min(min_margin, realizability_margin(collect(tup35(Mm, i, j, k)))))
        end
        interpen = max(interpen, ip)
    end
    wall = time() - t0
    return (clips=chain_clips(), bulk_clips=bulk_clips, str_proj=str_proj,
            min_margin=min_margin, interpen=interpen, wall=wall)
end

# ----------------------------------------------------------------------- main
function main()
    # (Ma, CFL, nsteps, Kn). Production CFL = 1/3. Ma=100 crossing is (nearly-)collisionless
    # (Kn=1e3 gives e~1e-5/step); the counter-streaming shared-Maxwellian BGK target is only
    # stable at the crossing in the collisionless limit, so Ma=100 uses Kn=1e9 (see docs).
    cases = length(ARGS) >= 1 ? [(parse(Float64, ARGS[1]),
                                  length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1/3,
                                  length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40,
                                  length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e3)] :
                                 [(5.0, 1/3, 40, 1e3), (100.0, 1/3, 40, 1e9)]
    println("=== Gate 3: two-stream vs production, crossing jets 48x48x16 ===")
    for (Ma, CFL, nsteps, Kn) in cases
        s = run_single(Ma; nsteps=nsteps, CFL=CFL, Kn=Kn)
        d = run_twostream(Ma; nsteps=nsteps, CFL=CFL, Kn=Kn)
        @printf("Ma=%-4g (CFL=%.3f, %d steps, Kn=%g)\n", Ma, CFL, nsteps, Kn)
        @printf("  PRODUCTION : projection35_fires=%-7d min_margin=%+.2e wall=%.1fs\n",
                s.fired, s.min_margin, s.wall)
        @printf("  TWO-STREAM : bulk_x_clips=%-5d flux_clips=%-7d transverse_hyp_corr=%-6d min_str_margin=%+.2e peak_interpen=%-5d wall=%.1fs\n",
                d.bulk_clips, d.clips, d.str_proj, d.min_margin, d.interpen, d.wall)
        @printf("  wall ratio (two-stream / production) = %.2fx\n", d.wall/s.wall)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

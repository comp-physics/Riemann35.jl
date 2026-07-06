# two_stream_transverse_isolation.jl — the decisive dimensional-isolation test.
#
# Question: at Ma=100 the full 3D two-stream run clips 333k times and loses
# interpenetration, but the 1D pipeline (Gate 2) crosses cleanly (44 flux clips,
# H=+2.000). Two candidate causes for the 3D failure:
#   (a) the transverse (y,z) HLL wiring: production's full-line closure/CFL is
#       applied to a half-space stream measure with NO realizability guarantee.
#   (b) a fundamental x-closure limit that only bites in the 3D config.
#
# This isolates the transverse by toggling the y,z HLL sweeps ON/OFF while holding
# everything else fixed, crossed with CFL (a second confound spotted in the logs:
# CFL=0.2 crossed cleanly, CFL=0.333 was catastrophic). A 2x2 matrix at Ma=100
# collisionless (Kn=1e9) fully separates the two effects.
#
#   flux_clips track transverse (ON high, OFF low)  -> hypothesis (a), fixable wiring
#   flux_clips track CFL only                       -> a timestep/x-CFL issue, not (a)/(b)
#   OFF also clips heavily & loses interpenetration  -> hypothesis (b), fundamental
#
# Reuses every helper from the Gate-3 driver (its main() is guarded, so include is safe).

include(joinpath(@__DIR__, "two_stream_gate3_headtohead.jl"))

# ---------------------------------------------------------------------------
# run_twostream with (Nx,Ny,Nz) params and a `transverse` toggle. Verbatim copy
# of the driver's run_twostream body except: grid is parameterized, and the y,z
# HLL sweeps (the "transverse wiring") are gated behind `transverse`.
# ---------------------------------------------------------------------------
function run_twostream_iso(Ma; nsteps=40, CFL=0.2, Kn=1e9, c=0.0,
                           transverse::Bool=true, Nx=48, Ny=48, Nz=16)
    M0, dx, dy, dz = crossing_ic(Ma; Nx=Nx, Ny=Ny, Nz=Nz)
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
            Mp[i,j,k,q] -= (dt/dx)*(Xp[i,j,k,q] - Xp[i-1,j,k,q])
            Mm[i,j,k,q] -= (dt/dx)*(Xm[i+1,j,k,q] - Xm[i,j,k,q])
        end
        # y/z-sweeps: production HLL path, per stream — THE ISOLATED VARIABLE.
        if transverse
            hll_sweep!(Mp, dt, dy, 2, Ma); hll_sweep!(Mm, dt, dy, 2, Ma)
            hll_sweep!(Mp, dt, dz, 3, Ma); hll_sweep!(Mm, dt, dz, 3, Ma)
        end
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

function iso_main()
    Ma = 100.0; Kn = 1e9; nsteps = 40
    println("=== Ma=100 collisionless transverse×CFL isolation (48x48x16, $nsteps steps, Kn=$Kn) ===")
    println("    transverse=OFF  => only x donor-cell + BGK act (y,z HLL wiring removed)")
    @printf("%-6s %-5s | %-9s %-9s %-9s %-11s %-9s %-7s\n",
            "transv", "CFL", "flux_clip", "bulk_clip", "trans_prj", "min_margin", "interpen", "wall")
    for transverse in (false, true), CFL in (0.2, 1/3)
        d = run_twostream_iso(Ma; nsteps=nsteps, CFL=CFL, Kn=Kn, transverse=transverse)
        @printf("%-6s %-5.3f | %-9d %-9d %-9d %+.3e %-9d %-6.1fs\n",
                transverse ? "ON" : "OFF", CFL, d.clips, d.bulk_clips, d.str_proj,
                d.min_margin, d.interpen, d.wall)
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    iso_main()
end

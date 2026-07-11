# kfvs_invariant_restore_test.jl — TEST-BEFORE-BUILD for cheap Route B.
#
# The debt-by-order decomposition (kfvs_debt_by_order.jl) showed the collision invariants
# are reproduced by the inversion to machine precision except on a few outlier cells, so
# Route B looks cheap: on the out-of-cone cells, use the measure update M̃ (in-cone, kills
# drift) but RESTORE the conserved low-order moments from M (to keep conservation).
#
# UNTESTED design inference: restoring M's low-order moments onto M̃ could push M̃ back
# OUT of the cone (the max per-cell energy defect was 0.37). This tests it directly on the
# actual drifted cells before any build.
#
# At N=20 steps (F3, Ma=100, 16^3), for every DRIFTED cell (realizability_margin<0):
#   base   :  M̃ = M[Q(M)]  — is it realizable? (measure update, should be ~100%)
#             + how many M̃ already conserve energy vs M within 1e-10 (no restore needed)
#   variant A (restore degree<=2): M̃ with ALL moments of total degree <=2 overwritten by M
#             (mass, momentum, full 2nd-moment tensor exact; 3rd/4th from M̃) — realizable?
#   variant B (restore invariants + energy trace): M̃ with {m000,m100,m010,m001} = M and the
#             diagonal 2nd moments scaled so m200+m020+m002 matches M's energy (cross 2nd
#             kept from M̃) — realizable?
# Realizable = is_realizable (Δ2* cross cone) AND _marginal_regularized (marginal caps).
# Reports the fraction of drifted cells each variant keeps in-cone, and the residual energy
# error. If A or B stays realizable on ~all drifted cells, cheap Route B is viable.
#
# CPU-only.  HYQMOM_RESTORE_STEPS default 20.

ENV["HYQMOM_SKIP_PLOTTING"] = "true"; ENV["CI"] = "true"
using Riemann35, MPI, Printf, LinearAlgebra, Statistics
MPI.Initialized() || MPI.Init()
const RM = Riemann35

const TRIP = collect(RM._ANCHOR_IJK)
const ORD  = [t[1]+t[2]+t[3] for t in TRIP]
const DLE2 = findall(ORD .<= 2)                            # degree <=2 channels
const INV  = [findfirst(==(t), TRIP) for t in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]  # mass+mom
const IX_E = [findfirst(==(t), TRIP) for t in ((2,0,0),(0,2,0),(0,0,2))]          # energy (trace)

function build_ic(Np, halo, Ma)
    C = 1.0; vj = 0.15 * Ma
    bg = InitializeM4_35(0.05, 0.0, 0.0, 0.0, C, 0.0, 0.0, C, 0.0, C)
    l  = InitializeM4_35(1.0,  vj,  vj / 2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    r  = InitializeM4_35(1.0, -vj, -vj / 2, 0.0, C, 0.0, 0.0, C, 0.0, C)
    nx = ny = nz = Np; M = zeros(Float64, nx + 2halo, ny + 2halo, nz, 35)
    Cs = max(1, floor(Int, 0.2 * Np))
    lo = div(Np, 2) - Cs; hi = div(Np, 2); lo2 = div(Np, 2) + 1; hi2 = div(Np, 2) + 1 + Cs
    for k in 1:nz, i in 1:nx, j in 1:ny
        Mr = bg
        (lo <= i <= hi && lo <= j <= hi)     && (Mr = l)
        (lo2 <= i <= hi2 && lo2 <= j <= hi2) && (Mr = r)
        M[i + halo, j + halo, k, :] = Mr
    end
    M, nx, ny, nz
end

function requad(m::AbstractVector)
    q = RM._anchor_quad(collect(m)); q === nothing && return nothing
    (nn, nux, nuy, nuz, Nn) = q; Mt = zeros(35)
    @inbounds for c in 1:Nn; RM._anchor_accum!(Mt, nn[c], nux[c], nuy[c], nuz[c]); end
    return Mt
end

isreal_full(m, Ma, s3max) = RM.is_realizable(collect(m)) && RM._marginal_regularized(NTuple{35,Float64}(m), Ma, s3max)
energy(m) = m[IX_E[1]] + m[IX_E[2]] + m[IX_E[3]]

function main()
    Ma = 100.0; Np = 16; halo = 4
    dt = 0.12 * (1.0/Np) / (Ma / 2 + 5)
    s3max = max(40.0, 4.0 + abs(Ma) / 2.0)
    set_kfvs_xfloor!(0.0)
    ns = parse(Int, get(ENV, "HYQMOM_RESTORE_STEPS", "20"))
    println("KFVS invariant-restore realizability test  (Ma=$Ma, Np=$Np, N=$ns steps)")
    println("  degree<=2 channels=$(length(DLE2))  invariants(mass+mom)=$INV  energy=$IX_E")
    println("="^88)

    M, nx, ny, nz = build_ic(Np, halo, Ma)
    comm = MPI.COMM_WORLD; bc = :copy
    decomp = setup_mpi_cartesian_3d(Np, Np, Np, halo, comm)
    halo_exchange_3d!(M, decomp, bc)
    for _ in 1:ns
        step_highorder_3d!(M, dt, decomp, bc, nx, ny, nz, halo, 1.0/Np, 1.0/Np, 1.0/Np, Ma;
                           order = 3, s3max = s3max, use_kfvs_anchor = true)
    end

    ndrift = 0; nMt_real = 0; n_econserve = 0
    nA_real = 0; nB_real = 0; nA_deg = 0; nB_deg = 0
    eA = Float64[]; eB = Float64[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = Vector{Float64}(M[i+halo, j+halo, k, :])
        realizability_margin(c) < 0 || continue
        ndrift += 1
        Mt = requad(c)
        if Mt === nothing; continue; end
        isreal_full(Mt, Ma, s3max) && (nMt_real += 1)
        abs(energy(Mt) - energy(c)) / (abs(energy(c)) + 1e-300) < 1e-10 && (n_econserve += 1)

        # variant A: restore all degree<=2 moments from M
        A = copy(Mt); A[DLE2] .= c[DLE2]
        A[1] > 0 ? (isreal_full(A, Ma, s3max) ? (nA_real += 1) : nothing) : (nA_deg += 1)
        push!(eA, abs(energy(A) - energy(c)) / (abs(energy(c)) + 1e-300))

        # variant B: restore mass+mom, scale diagonal 2nd to match M's energy trace
        B = copy(Mt); B[INV] .= c[INV]
        Ecur = energy(B); Etar = energy(c)
        if Ecur > 0
            s = Etar / Ecur
            B[IX_E] .*= s
        end
        B[1] > 0 ? (isreal_full(B, Ma, s3max) ? (nB_real += 1) : nothing) : (nB_deg += 1)
        push!(eB, abs(energy(B) - energy(c)) / (abs(energy(c)) + 1e-300))
    end
    pct(x) = ndrift > 0 ? 100 * x / ndrift : 0.0
    @printf("\n drifted cells (margin<0): %d\n", ndrift)
    @printf("  base  M̃ realizable            : %5d (%.1f%%)\n", nMt_real, pct(nMt_real))
    @printf("  base  M̃ already conserves E   : %5d (%.1f%%)   (|ΔE|/E < 1e-10, no restore needed)\n", n_econserve, pct(n_econserve))
    @printf("  A  restore deg<=2 realizable  : %5d (%.1f%%)   degenerate=%d  resid E err median=%.2e\n",
            nA_real, pct(nA_real), nA_deg, isempty(eA) ? NaN : median(eA))
    @printf("  B  restore inv+Etrace real.   : %5d (%.1f%%)   degenerate=%d  resid E err median=%.2e\n",
            nB_real, pct(nB_real), nB_deg, isempty(eB) ? NaN : median(eB))
    println("\nREAD: A or B near 100% ⇒ cheap Route B viable (measure update on out-of-cone cells +")
    println("      restore that invariant set stays in-cone). Low % ⇒ restoration breaks the cone,")
    println("      a more careful conservative correction is needed.")
    println("DONE.")
end
main()

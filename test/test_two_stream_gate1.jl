# test_two_stream_gate1.jl — Gate 1 for the two-stream half-space mode.
#
# Ports the `jac15` checker pattern from roe-writeup-scripts/verify_halfspace_2d15.jl
# to the full 35-moment half-space x-flux (flux_halfspace35). Catches transcription
# errors before any solver run exists. Checks, on random off-manifold half-space
# states (x-support > 0):
#   (a) full 35x35 x-flux Jacobian spectrum real, min eig >= -1e-6 relative;
#   (b) the marginal (grade-0) n=5 half-line block embeds exactly;
#   (c) grade-graded (j+k) block lower-triangularity: upper blocks vanish;
#   (d) per-channel diagonal block spectra real and nonnegative;
#   (e) separable-state transverse channel closures exact to ~1e-14.
#
# The Jacobian is finite-differenced in CHAIN coordinates (m0, zeta1..4) for the
# x-marginal (NOT raw moments — raw FD is ill-conditioned at high Ma) plus the 30
# transverse raw moments, exactly as jac15.

using LinearAlgebra, Printf, Random, Test

const SRC = get(ENV, "RIEMANN35_SRC",
                joinpath(@__DIR__, "..", "src"))
if !isdefined(Main, :Chain)
    include(joinpath(SRC, "moments", "chain.jl"));             using .Chain
    include(joinpath(SRC, "numerics", "flux_closure_dev.jl")); using .FluxClosureDev
    include(joinpath(SRC, "numerics", "halfline_closure.jl")); using .HalflineClosure
    include(joinpath(SRC, "numerics", "flux_halfspace35.jl")); using .FluxHalfspace35
end

# canonical M4 (i,j,k) exponents
const IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
             (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
             (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
             (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))

# grade = j+k; slot groups ordered by grade (block lower-triangular structure)
const GRADE = ntuple(q -> IJK[q][2] + IJK[q][3], 35)
const GRP = [findall(==(s), collect(GRADE)) for s in 0:4]  # slot indices per grade

gaussmom(v, T) = (1.0, v, v^2 + T, v^3 + 3v*T, v^4 + 6v^2*T + 3T^2)

# random half-space (x-support > 0) state: sum of positive-x atoms x Gaussian(y,z)
function random_state35(rng)
    M = zeros(35)
    for _ in 1:rand(rng, 4:7)
        w = rand(rng) + 0.05
        x = exp(randn(rng) * 0.8)             # positive x-node
        gy = gaussmom(randn(rng)*1.2, exp(randn(rng)-0.5))
        gz = gaussmom(randn(rng)*1.2, exp(randn(rng)-0.5))
        @inbounds for (q, (i, j, k)) in enumerate(IJK)
            M[q] += w * x^i * gy[j+1] * gz[k+1]
        end
    end
    M
end

# reconstruct M from theta = (m0, zeta1..4, transverse raw[30]); marginal via hseq
function M_from_theta(t)
    m0 = t[1]; z = (t[2], t[3], t[4], t[5])
    h = hseq(m0, z...)
    vcat(collect(h[1:5]), t[6:35])
end

# full 35x35 x-flux Jacobian, FD in theta
function jac35(t)
    Mt = zeros(35, 35); Ft = zeros(35, 35)
    for j in 1:35
        h = 1e-7 * max(abs(t[j]), 1e-9)
        tp = copy(t); tp[j] += h; tm = copy(t); tm[j] -= h
        Mp = M_from_theta(tp); Mm = M_from_theta(tm)
        Fp = collect(xflux_plus35(Mp...)); Fm = collect(xflux_plus35(Mm...))
        Mt[:, j] = (Mp .- Mm) ./ (2h)
        Ft[:, j] = (Fp .- Fm) ./ (2h)
    end
    cond(Mt) > 1e10 && return nothing
    Ft * inv(Mt)
end

# marginal n=5 half-line closure Jacobian (grade-0 block), FD in (m0, zeta)
function marg_eigs(marg)
    z = chain((marg[1], marg[2], marg[3], marg[4], marg[5]))
    t = vcat(marg[1], collect(z))
    Mt = zeros(5, 5); Ft = zeros(5, 5)
    for j in 1:5
        h = 1e-7 * max(abs(t[j]), 1e-9)
        tp = copy(t); tp[j] += h; tm = copy(t); tm[j] -= h
        hp = hseq(tp[1], tp[2], tp[3], tp[4], tp[5])
        hm = hseq(tm[1], tm[2], tm[3], tm[4], tm[5])
        Mt[:, j] = (collect(hp[1:5]) .- collect(hm[1:5])) ./ (2h)
        Ft[:, j] = (collect(hp[2:6]) .- collect(hm[2:6])) ./ (2h)  # m1..m5 (m5=h5 closure)
    end
    eigvals(Ft * inv(Mt))
end

@testset "Gate 1: two-stream half-space unit/spectrum/consistency" begin
    rng = MersenneTwister(113)
    N = 1500
    n = 0; nreal = 0; nneg = 0; nblock = 0; ntri = 0
    worst_im = 0.0; worst_min = Inf; worst_blk = 0.0; worst_tri = 0.0
    chan_min = fill(Inf, 5); chan_im = zeros(5)
    for _ in 1:N
        M = random_state35(rng)
        (M[1] > 1e-6 && all(isfinite, M)) || continue
        z = chain((M[1], M[2], M[3], M[4], M[5]))
        all(x -> isfinite(x) && x > 0, z) || continue
        t = vcat(M[1], collect(z), M[6:35])
        A = jac35(t); A === nothing && continue
        lam = eigvals(A)
        sc = maximum(abs.(lam)) + 1e-300
        n += 1
        imr = maximum(abs.(imag.(lam))) / sc
        rl = sort(real.(lam))
        imr < 1e-6 && (nreal += 1); worst_im = max(worst_im, imr)
        rl[1] / sc > -1e-6 && (nneg += 1); worst_min = min(worst_min, rl[1] / sc)
        # marginal block embedding
        e0 = sort(real.(marg_eigs(M[1:5])))
        bm = maximum(minimum(abs.(rl .- e)) for e in e0) / sc
        bm < 1e-5 && (nblock += 1); worst_blk = max(worst_blk, bm)
        # grade triangularity: upper blocks (grade r rows, grade r' cols, r' > r)
        tri = 0.0
        for r in 1:5, rp in r+1:5
            tri = max(tri, maximum(abs.(A[GRP[r], GRP[rp]])))
        end
        tri / sc < 1e-6 && (ntri += 1); worst_tri = max(worst_tri, tri / sc)
        # per-grade diagonal block spectra
        for r in 1:5
            B = A[GRP[r], GRP[r]]
            le = eigvals(B); s2 = maximum(abs.(le)) + 1e-300
            chan_im[r] = max(chan_im[r], maximum(abs.(imag.(le))) / s2)
            chan_min[r] = min(chan_min[r], minimum(real.(le)) / s2)
        end
    end
    @printf("--- 35-moment half-space x-flux, %d valid states ---\n", n)
    @printf("  full spectrum real     : %d/%d  (worst Im/|l| = %.1e)\n", nreal, n, worst_im)
    @printf("  full spectrum >= 0     : %d/%d  (worst min l/|l| = %+.1e)\n", nneg, n, worst_min)
    @printf("  marginal block embeds  : %d/%d  (worst mismatch = %.1e)\n", nblock, n, worst_blk)
    @printf("  grade triangularity    : %d/%d  (worst upper-block/|l| = %.1e)\n", ntri, n, worst_tri)
    println("  per-grade diagonal blocks (worst Im ratio, worst min eig ratio):")
    for r in 1:5
        @printf("    grade=%d (%d slots): Im = %.1e   min = %+.1e\n",
                r-1, length(GRP[r]), chan_im[r], chan_min[r])
    end
    @test n > 1200
    @test nreal / n > 0.99
    @test worst_min > -1e-6
    @test nblock / n > 0.98
    @test worst_tri < 1e-6
    for r in 1:5
        @test chan_min[r] > -1e-6
    end
end

@testset "Gate 1: separable-state channel closures exact" begin
    rng = MersenneTwister(7)
    # closure slots -> (nr, j, k)
    closures = [(9,4,1,0),(12,3,2,0),(14,2,3,0),(15,1,4,0),
                (19,4,0,1),(22,3,0,2),(24,2,0,3),(25,1,0,4),
                (28,3,1,1),(30,2,2,1),(31,1,3,1),(33,2,1,2),(34,1,1,3),(35,1,2,2)]
    worst = 0.0
    for _ in 1:400
        # x-measure A (positive atoms) -> A0..A4
        A = zeros(5)
        for _ in 1:rand(rng, 3:5)
            w = rand(rng)+0.05; x = exp(randn(rng)*0.7)
            for i in 0:4; A[i+1] += w*x^i; end
        end
        By = gaussmom(randn(rng)*1.0, exp(randn(rng)-0.3))
        Cz = gaussmom(randn(rng)*1.0, exp(randn(rng)-0.3))
        M = zeros(35)
        for (q,(i,j,k)) in enumerate(IJK); M[q] = A[i+1]*By[j+1]*Cz[k+1]; end
        F = xflux_plus35(M...)
        for (slot,nr,j,k) in closures
            ref = A[nr+1]*By[j+1]*Cz[k+1]
            # scale by channel carried-moment magnitude (avoid near-zero inflation)
            scale = maximum(abs(A[i+1]*By[j+1]*Cz[k+1]) for i in 0:nr-1) + 1e-300
            worst = max(worst, abs(F[slot]-ref)/scale)
        end
    end
    @printf("  worst separable channel-closure rel error = %.2e\n", worst)
    @test worst < 1e-12
end

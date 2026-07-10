"""
    chyqmom_nodes_3d_dev.jl — alloc-free, device-compatible per-cell CHyQMOM inversion.

Faithful scalar/register port of the CPU reference
`chyqmom_nodes_3d(M)` (`src/moments/chyqmom_nodes_3d.jl`): length-35 raw moments →
nonnegative 3D velocity quadrature `(n, U)` with ≤27 nodes. Written in the
codebase's device idiom (NTuple-carried fixed-size state, `@inline`/`@noinline`
scalar helpers, no heap, no closures, no StaticArrays in the kernel body, fp64
throughout), so the body ports verbatim to a one-thread-per-cell CUDA kernel.

This is the load-bearing inversion for the proposed kinetic-FVS realizable anchor.
It is a **PURE ADDITION** under `gpu/` — nothing in the existing solver path calls
it (no `projection35` / residual changes). Downstream anchor pieces (per-cell
storage, `measure_update`, theta-star blend) are built on top of this in later increments.

# What it returns
`chyqmom_nodes_3d_dev(M::NTuple{35,Float64})` → `(n, Ux, Uy, Uz, Nn)`, each of
`n/Ux/Uy/Uz` an `NTuple{27,Float64}` with slots `> Nn` zero; `Nn::Int` is the node
count. Node weights are scaled back to physical density (`Σn ≈ M000`), all `≥ 0`.

# Structure (fixed layout x≤3, (x,y)≤9, final≤27; adaptive-N via running counts)
- **1D primitive** `hyqmom_1d_dev` (≤13×/cell): closed-form N=3 Lagrange / N=2
  Gauss Vandermonde weight solves, adaptive-N (N∈{1,2,3}), realizability gates as
  scalar branches. Removes the CPU's `SingularException` throw (guarded closed-form
  denominator replaces the LU `\`).
- **Central moments** `central(M,i,j,k,bu,bv,bw)`: binomial expansion over the
  35-tuple via `mget` (literal-indexed), no 5×5×5 array.
- **y|x fit** `gram_fit_y`: 1D-conditioning, monomials {1,x,x²}, incremental
  condition-gated selection, N≤3 Cholesky solve.
- **z|x,y fit** `gram_fit_z`: 2D-conditioning, monomials (i,j) low-total-degree
  first, incremental condition-gated selection, **N=9** Cholesky solve. (The N=6
  cap in the original design was FALSIFIED by real cells: 8 columns is the dominant
  case, up to 9 occur — see gpu/validation/README_kfvs.md. The solver is sized N=9.)

# FAITHFUL CONDITION GATE (the fidelity fix vs the prototype's pivot proxy)
The CPU gate `_design_cond(B) = svdvals(B)[1]/svdvals(B)[end]` rejects a candidate
column when `κ₂(B) ≥ 1e4` on the TALL weighted design `B = sqrt(pw).*Phi`. An
earlier prototype used a Cholesky-pivot-RATIO proxy on the Gram `G = BᵀB`, which
under-estimates κ(G) by up to ~10 orders on near-collinear geometries → ~0.18% of
real cells admitted one spurious column → a blown-up high-order z cross-moment.

Here the gate is EXACT: `G` is SPD and symmetric, so `σᵢ(B) = sqrt(λᵢ(G))` and
therefore `κ₂(B) = sqrt(λmax(G)/λmin(G))`. `condkappaN` computes the extreme
eigenvalues of the small (≤9×9) SPD `G` without LAPACK — `λmax` by power
iteration, `λmin` by inverse power iteration through the SAME Cholesky factor the
SPD solve already needs (free), returning `κ(G)=λmax/λmin`. The gate is then the
CPU's own `sqrt(κ(G)) < 1e4` (i.e. `κ(G) < 1e8`), no tuned threshold. VERIFIED on
>310k real column decisions: 0 accept/reject mismatches vs the CPU svd gate (the
proxy mismatched ~0.25%). See gpu/validation/fix1_gate_probe.jl,
gpu/validation/fix1_itereig_probe.jl.

`@fastmath` deliberately OFF (matching the other device modules): the gate
decision is sign/tolerance-sensitive at the κ boundary. Pure addition under
`gpu/`; not wired into production; fp64.
"""
module KFVSInversionDev

export chyqmom_nodes_3d_dev, chyqmom_nodes_3d_store_dev!, NODEMAX

const NODEMAX = 27

# CPU gate cutoff: reject a candidate column when κ₂(B) ≥ CONDMAX_B (== the CPU
# `_design_cond`'s condmax). Equivalently κ(G) = κ₂(B)² ≥ CONDMAX_B².
const CONDMAX_B = 1.0e4
const CONDMAX_G = CONDMAX_B * CONDMAX_B     # 1e8

# ===========================================================================
# 35-moment tuple accessor. The 35-set is downward-closed; every sub-moment the
# central-moment expansions touch is present. `mget(M,a,b,d)` returns the raw
# moment M_{a b d} from the 35-tuple, or 0.0 for the 90 absent (5x5x5) entries.
# All indices are compile-time literals at every call site below.
# ===========================================================================
@inline function mget(M::NTuple{35,Float64}, a::Int, b::Int, d::Int)
    # ordering per _CHYQ_TRIPLES (chyqmom_nodes_3d.jl)
    if     a==0 && b==0 && d==0; return M[1]
    elseif a==1 && b==0 && d==0; return M[2]
    elseif a==2 && b==0 && d==0; return M[3]
    elseif a==3 && b==0 && d==0; return M[4]
    elseif a==4 && b==0 && d==0; return M[5]
    elseif a==0 && b==1 && d==0; return M[6]
    elseif a==1 && b==1 && d==0; return M[7]
    elseif a==2 && b==1 && d==0; return M[8]
    elseif a==3 && b==1 && d==0; return M[9]
    elseif a==0 && b==2 && d==0; return M[10]
    elseif a==1 && b==2 && d==0; return M[11]
    elseif a==2 && b==2 && d==0; return M[12]
    elseif a==0 && b==3 && d==0; return M[13]
    elseif a==1 && b==3 && d==0; return M[14]
    elseif a==0 && b==4 && d==0; return M[15]
    elseif a==0 && b==0 && d==1; return M[16]
    elseif a==1 && b==0 && d==1; return M[17]
    elseif a==2 && b==0 && d==1; return M[18]
    elseif a==3 && b==0 && d==1; return M[19]
    elseif a==0 && b==0 && d==2; return M[20]
    elseif a==1 && b==0 && d==2; return M[21]
    elseif a==2 && b==0 && d==2; return M[22]
    elseif a==0 && b==0 && d==3; return M[23]
    elseif a==1 && b==0 && d==3; return M[24]
    elseif a==0 && b==0 && d==4; return M[25]
    elseif a==0 && b==1 && d==1; return M[26]
    elseif a==1 && b==1 && d==1; return M[27]
    elseif a==2 && b==1 && d==1; return M[28]
    elseif a==0 && b==2 && d==1; return M[29]
    elseif a==1 && b==2 && d==1; return M[30]
    elseif a==0 && b==3 && d==1; return M[31]
    elseif a==0 && b==1 && d==2; return M[32]
    elseif a==1 && b==1 && d==2; return M[33]
    elseif a==0 && b==1 && d==3; return M[34]
    elseif a==0 && b==2 && d==2; return M[35]
    else
        return 0.0
    end
end

# ===========================================================================
# 1D HyQMOM primitive (device). Given a length-5 raw moment sequence, returns a
# fixed 3-slot result (w1,w2,w3, u1,u2,u3, N) with N in {1,2,3} and unused slots
# = 0. Faithful port of hyqmom_quadrature_1d.jl with closed-form Vandermonde.
# @noinline to cap register pressure (invoked <=13x/cell).
# ===========================================================================
@noinline function hyqmom_1d_dev(M0::Float64, M1::Float64, M2::Float64,
                                 M3::Float64, M4::Float64)
    wtol   = -1.0e-12
    gaptol = 1.0e-9
    vartol = 1.0e-12

    # M0>0 guaranteed by callers (weights>0). Guard anyway.
    if !(M0 > 0.0)
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    end
    ubar = M1 / M0
    c2 = M2/M0 - ubar*ubar
    c3 = M3/M0 - 3.0*ubar*(M2/M0) + 2.0*ubar*ubar*ubar
    c4 = M4/M0 - 4.0*ubar*(M3/M0) + 6.0*ubar*ubar*(M2/M0) - 3.0*ubar*ubar*ubar*ubar

    # N=1 : vacuum/cold
    if !(c2 > vartol)
        return (M0, 0.0, 0.0, ubar, 0.0, 0.0, 1)
    end

    scale = sqrt(c2)
    # ---- N=3 attempt ----
    q   = c3 / (scale * c2)
    eta = c4 / (c2 * c2)
    disc = 4.0*eta - 3.0*q*q
    if disc > 0.0
        D = sqrt(disc)
        if D > gaptol
            up1 = (q - D)/2.0
            up2 = 0.0
            up3 = (q + D)/2.0
            u1 = ubar + scale*up1
            u2 = ubar + scale*up2
            u3 = ubar + scale*up3
            # closed-form 3x3 Vandermonde solve V w = [M0,M1,M2], V=[1;u;u^2].
            # Lagrange form: denominators are the pairwise differences.
            d12 = u1 - u2; d13 = u1 - u3; d23 = u2 - u3
            # guard the denominator (already gated by D>gaptol on standardized sep)
            if abs(d12) > 0.0 && abs(d13) > 0.0 && abs(d23) > 0.0
                # w_k = (M0*u_i*u_j - M1*(u_i+u_j) + M2) / ((u_k-u_i)(u_k-u_j))
                w1 = (M0*u2*u3 - M1*(u2+u3) + M2) / (d12*d13)
                w2 = (M0*u1*u3 - M1*(u1+u3) + M2) / ((-d12)*d23)
                w3 = (M0*u1*u2 - M1*(u1+u2) + M2) / ((-d13)*(-d23))
                if w1 >= wtol && w2 >= wtol && w3 >= wtol
                    w1 = w1 < 0.0 ? 0.0 : w1
                    w2 = w2 < 0.0 ? 0.0 : w2
                    w3 = w3 < 0.0 ? 0.0 : w3
                    return (w1, w2, w3, u1, u2, u3, 3)
                end
            end
        end
    end

    # ---- N=2 attempt ----
    # c2>0 here. standardized skewness q (recompute for clarity; same value).
    s = sqrt(1.0 + q*q/4.0)
    up1 = q/2.0 - s
    up2 = q/2.0 + s
    u1 = ubar + scale*up1
    u2 = ubar + scale*up2
    d = u1 - u2
    if abs(d) > 0.0
        # V w = [M0, M0*ubar], V=[1 1; u1 u2]; d = u1-u2.
        # w1 = (M1 - u2*M0)/(u1-u2),  w2 = (u1*M0 - M1)/(u1-u2), M1=M0*ubar.
        w1 = (M0*ubar - M0*u2) / d
        w2 = (M0*u1 - M0*ubar) / d
        if w1 >= wtol && w2 >= wtol
            w1 = w1 < 0.0 ? 0.0 : w1
            w2 = w2 < 0.0 ? 0.0 : w2
            return (w1, w2, 0.0, u1, u2, 0.0, 2)
        end
    end

    # ---- N=1 monokinetic fallback ----
    return (M0, 0.0, 0.0, ubar, 0.0, 0.0, 1)
end

# ===========================================================================
# _shape_moments (device): raw moment sequence [m0..m4] of a 1D distribution with
# weight w, mean mu, variance s2>=0, standardized skewness q, kurtosis eta.
# ===========================================================================
@inline function shape_moments_dev(w::Float64, mu::Float64, s2::Float64,
                                   q::Float64, eta::Float64)
    s2 = s2 < 0.0 ? 0.0 : s2
    sig = sqrt(s2)
    s3 = sig*s2          # sig^3
    s4 = s2*s2           # s2^2
    m0 = w
    m1 = w * mu
    m2 = w * (mu*mu + s2)
    m3 = w * (mu*mu*mu + 3.0*mu*s2 + q*s3)
    m4 = w * (mu*mu*mu*mu + 6.0*mu*mu*s2 + 4.0*mu*q*s3 + eta*s4)
    return (m0, m1, m2, m3, m4)
end

# ===========================================================================
# Cholesky-pivot condition gate + SPD solve, fixed size N<=6.
# Reused from proto_kfvs_gpu_la.jl. G given as flat lower-tri NTuple{21}
# (lower triangle, i>=j, position = i*(i-1)/2 + j).
# ===========================================================================
@inline _lidx(i::Int, j::Int) = (i * (i - 1)) >> 1 + j

# ===========================================================================
# FAITHFUL condition gate: true kappa(G) via the shared Cholesky factor.
#
# G is SPD and symmetric => sigma_i(B) = sqrt(lambda_i(G)), so the CPU svd gate
# kappa2(B) < 1e4 is EXACTLY kappa(G) = lambda_max/lambda_min < 1e8. We compute the
# extreme eigenvalues of the small (<=9x9) SPD G without LAPACK:
#   * lambda_max : power iteration on G.
#   * lambda_min : inverse power iteration, solving G y = x via the SAME Cholesky
#     factor L (G = L Lᵀ) the SPD solve already forms — two triangular solves per
#     iter, no extra factorization.
# Cholesky breakdown (a nonpositive pivot) => the candidate column is (numerically)
# linearly dependent => reject (Inf), matching the CPU rank-deficient -> Inf branch.
#
# The iteration counts are fixed (unrolled-friendly, no data-dependent loop trip
# beyond the compile-time cap). Near the kappa boundary the accept/reject decision
# is insensitive to sub-percent eigenvalue error (verified: 0 gate mismatches vs
# CPU on >310k real column decisions).
# ===========================================================================
const _EIG_ITERS = 40   # power / inverse-power iterations (fixed)

# Multiply y = G x for the leading nbxnb block of the flat lower-tri Gram gl (len L).
# Uses symmetry: G[i,j] = G[j,i] = gl[_lidx(max,min)]. n<=9.
@inline function _gram_matvec9(gl, x1,x2,x3,x4,x5,x6,x7,x8,x9, nb::Int)
    @inline gij(i,j) = (i >= j) ? gl[_lidx(i,j)] : gl[_lidx(j,i)]
    y1=0.0;y2=0.0;y3=0.0;y4=0.0;y5=0.0;y6=0.0;y7=0.0;y8=0.0;y9=0.0
    @inbounds for i in 1:9
        (i > nb) && break
        s = 0.0
        for j in 1:9
            (j > nb) && break
            xj = j==1 ? x1 : (j==2 ? x2 : (j==3 ? x3 : (j==4 ? x4 : (j==5 ? x5 :
                 (j==6 ? x6 : (j==7 ? x7 : (j==8 ? x8 : x9)))))))
            s += gij(i,j) * xj
        end
        if i==1; y1=s elseif i==2; y2=s elseif i==3; y3=s elseif i==4; y4=s
        elseif i==5; y5=s elseif i==6; y6=s elseif i==7; y7=s elseif i==8; y8=s else; y9=s end
    end
    return (y1,y2,y3,y4,y5,y6,y7,y8,y9)
end

# True kappa(G) = lambda_max/lambda_min of the leading nb block of flat lower-tri
# Gram gl (length-45 form). Inf if not SPD (Cholesky breakdown) => reject.
@inline function condkappa9(gl::NTuple{45,Float64}, nb::Int)
    gl[1] <= 0.0 && return Inf
    # --- Cholesky factor L (flat lower-tri, reuse pivots to detect PD failure) ---
    L_1_1 = sqrt(gl[1])
    L_2_1 = nb >= 2 ? gl[2]/L_1_1 : 0.0
    d2 = gl[3] - L_2_1*L_2_1; (nb>=2 && d2<=0.0) && return Inf
    L_2_2 = nb >= 2 ? sqrt(d2) : 1.0
    L_3_1 = nb >= 3 ? gl[4]/L_1_1 : 0.0
    L_3_2 = nb >= 3 ? (gl[5]-L_3_1*L_2_1)/L_2_2 : 0.0
    d3 = gl[6]-L_3_1*L_3_1-L_3_2*L_3_2; (nb>=3 && d3<=0.0) && return Inf
    L_3_3 = nb >= 3 ? sqrt(d3) : 1.0
    L_4_1 = nb >= 4 ? gl[7]/L_1_1 : 0.0
    L_4_2 = nb >= 4 ? (gl[8]-L_4_1*L_2_1)/L_2_2 : 0.0
    L_4_3 = nb >= 4 ? (gl[9]-L_4_1*L_3_1-L_4_2*L_3_2)/L_3_3 : 0.0
    d4 = gl[10]-L_4_1*L_4_1-L_4_2*L_4_2-L_4_3*L_4_3; (nb>=4 && d4<=0.0) && return Inf
    L_4_4 = nb >= 4 ? sqrt(d4) : 1.0
    L_5_1 = nb >= 5 ? gl[11]/L_1_1 : 0.0
    L_5_2 = nb >= 5 ? (gl[12]-L_5_1*L_2_1)/L_2_2 : 0.0
    L_5_3 = nb >= 5 ? (gl[13]-L_5_1*L_3_1-L_5_2*L_3_2)/L_3_3 : 0.0
    L_5_4 = nb >= 5 ? (gl[14]-L_5_1*L_4_1-L_5_2*L_4_2-L_5_3*L_4_3)/L_4_4 : 0.0
    d5 = gl[15]-L_5_1*L_5_1-L_5_2*L_5_2-L_5_3*L_5_3-L_5_4*L_5_4; (nb>=5 && d5<=0.0) && return Inf
    L_5_5 = nb >= 5 ? sqrt(d5) : 1.0
    L_6_1 = nb >= 6 ? gl[16]/L_1_1 : 0.0
    L_6_2 = nb >= 6 ? (gl[17]-L_6_1*L_2_1)/L_2_2 : 0.0
    L_6_3 = nb >= 6 ? (gl[18]-L_6_1*L_3_1-L_6_2*L_3_2)/L_3_3 : 0.0
    L_6_4 = nb >= 6 ? (gl[19]-L_6_1*L_4_1-L_6_2*L_4_2-L_6_3*L_4_3)/L_4_4 : 0.0
    L_6_5 = nb >= 6 ? (gl[20]-L_6_1*L_5_1-L_6_2*L_5_2-L_6_3*L_5_3-L_6_4*L_5_4)/L_5_5 : 0.0
    d6 = gl[21]-L_6_1*L_6_1-L_6_2*L_6_2-L_6_3*L_6_3-L_6_4*L_6_4-L_6_5*L_6_5; (nb>=6 && d6<=0.0) && return Inf
    L_6_6 = nb >= 6 ? sqrt(d6) : 1.0
    L_7_1 = nb >= 7 ? gl[22]/L_1_1 : 0.0
    L_7_2 = nb >= 7 ? (gl[23]-L_7_1*L_2_1)/L_2_2 : 0.0
    L_7_3 = nb >= 7 ? (gl[24]-L_7_1*L_3_1-L_7_2*L_3_2)/L_3_3 : 0.0
    L_7_4 = nb >= 7 ? (gl[25]-L_7_1*L_4_1-L_7_2*L_4_2-L_7_3*L_4_3)/L_4_4 : 0.0
    L_7_5 = nb >= 7 ? (gl[26]-L_7_1*L_5_1-L_7_2*L_5_2-L_7_3*L_5_3-L_7_4*L_5_4)/L_5_5 : 0.0
    L_7_6 = nb >= 7 ? (gl[27]-L_7_1*L_6_1-L_7_2*L_6_2-L_7_3*L_6_3-L_7_4*L_6_4-L_7_5*L_6_5)/L_6_6 : 0.0
    d7 = gl[28]-L_7_1*L_7_1-L_7_2*L_7_2-L_7_3*L_7_3-L_7_4*L_7_4-L_7_5*L_7_5-L_7_6*L_7_6; (nb>=7 && d7<=0.0) && return Inf
    L_7_7 = nb >= 7 ? sqrt(d7) : 1.0
    L_8_1 = nb >= 8 ? gl[29]/L_1_1 : 0.0
    L_8_2 = nb >= 8 ? (gl[30]-L_8_1*L_2_1)/L_2_2 : 0.0
    L_8_3 = nb >= 8 ? (gl[31]-L_8_1*L_3_1-L_8_2*L_3_2)/L_3_3 : 0.0
    L_8_4 = nb >= 8 ? (gl[32]-L_8_1*L_4_1-L_8_2*L_4_2-L_8_3*L_4_3)/L_4_4 : 0.0
    L_8_5 = nb >= 8 ? (gl[33]-L_8_1*L_5_1-L_8_2*L_5_2-L_8_3*L_5_3-L_8_4*L_5_4)/L_5_5 : 0.0
    L_8_6 = nb >= 8 ? (gl[34]-L_8_1*L_6_1-L_8_2*L_6_2-L_8_3*L_6_3-L_8_4*L_6_4-L_8_5*L_6_5)/L_6_6 : 0.0
    L_8_7 = nb >= 8 ? (gl[35]-L_8_1*L_7_1-L_8_2*L_7_2-L_8_3*L_7_3-L_8_4*L_7_4-L_8_5*L_7_5-L_8_6*L_7_6)/L_7_7 : 0.0
    d8 = gl[36]-L_8_1*L_8_1-L_8_2*L_8_2-L_8_3*L_8_3-L_8_4*L_8_4-L_8_5*L_8_5-L_8_6*L_8_6-L_8_7*L_8_7; (nb>=8 && d8<=0.0) && return Inf
    L_8_8 = nb >= 8 ? sqrt(d8) : 1.0
    L_9_1 = nb >= 9 ? gl[37]/L_1_1 : 0.0
    L_9_2 = nb >= 9 ? (gl[38]-L_9_1*L_2_1)/L_2_2 : 0.0
    L_9_3 = nb >= 9 ? (gl[39]-L_9_1*L_3_1-L_9_2*L_3_2)/L_3_3 : 0.0
    L_9_4 = nb >= 9 ? (gl[40]-L_9_1*L_4_1-L_9_2*L_4_2-L_9_3*L_4_3)/L_4_4 : 0.0
    L_9_5 = nb >= 9 ? (gl[41]-L_9_1*L_5_1-L_9_2*L_5_2-L_9_3*L_5_3-L_9_4*L_5_4)/L_5_5 : 0.0
    L_9_6 = nb >= 9 ? (gl[42]-L_9_1*L_6_1-L_9_2*L_6_2-L_9_3*L_6_3-L_9_4*L_6_4-L_9_5*L_6_5)/L_6_6 : 0.0
    L_9_7 = nb >= 9 ? (gl[43]-L_9_1*L_7_1-L_9_2*L_7_2-L_9_3*L_7_3-L_9_4*L_7_4-L_9_5*L_7_5-L_9_6*L_7_6)/L_7_7 : 0.0
    L_9_8 = nb >= 9 ? (gl[44]-L_9_1*L_8_1-L_9_2*L_8_2-L_9_3*L_8_3-L_9_4*L_8_4-L_9_5*L_8_5-L_9_6*L_8_6-L_9_7*L_8_7)/L_8_8 : 0.0
    d9 = gl[45]-L_9_1*L_9_1-L_9_2*L_9_2-L_9_3*L_9_3-L_9_4*L_9_4-L_9_5*L_9_5-L_9_6*L_9_6-L_9_7*L_9_7-L_9_8*L_9_8; (nb>=9 && d9<=0.0) && return Inf
    L_9_9 = nb >= 9 ? sqrt(d9) : 1.0

    # --- lambda_max via power iteration on G ---
    x1=1.0;x2=1.0;x3=1.0;x4=1.0;x5=1.0;x6=1.0;x7=1.0;x8=1.0;x9=1.0
    lam_max = gl[1]
    @inbounds for _ in 1:_EIG_ITERS
        (y1,y2,y3,y4,y5,y6,y7,y8,y9) = _gram_matvec9(gl, x1,x2,x3,x4,x5,x6,x7,x8,x9, nb)
        nrm2 = y1*y1+y2*y2+y3*y3+y4*y4+y5*y5+y6*y6+y7*y7+y8*y8+y9*y9
        nrm2 <= 0.0 && break
        nrm = sqrt(nrm2)
        x1=y1/nrm; x2=y2/nrm; x3=y3/nrm; x4=y4/nrm; x5=y5/nrm
        x6=y6/nrm; x7=y7/nrm; x8=y8/nrm; x9=y9/nrm
        (z1,z2,z3,z4,z5,z6,z7,z8,z9) = _gram_matvec9(gl, x1,x2,x3,x4,x5,x6,x7,x8,x9, nb)
        lam_max = x1*z1+x2*z2+x3*z3+x4*z4+x5*z5+x6*z6+x7*z7+x8*z8+x9*z9
    end

    # --- lambda_min via inverse power iteration through L (G = L Lᵀ) ---
    v1=1.0;v2=1.0;v3=1.0;v4=1.0;v5=1.0;v6=1.0;v7=1.0;v8=1.0;v9=1.0
    lam_min = lam_max
    @inbounds for _ in 1:_EIG_ITERS
        # forward solve L w = v
        w1 = v1/L_1_1
        w2 = nb>=2 ? (v2 - L_2_1*w1)/L_2_2 : 0.0
        w3 = nb>=3 ? (v3 - L_3_1*w1 - L_3_2*w2)/L_3_3 : 0.0
        w4 = nb>=4 ? (v4 - L_4_1*w1 - L_4_2*w2 - L_4_3*w3)/L_4_4 : 0.0
        w5 = nb>=5 ? (v5 - L_5_1*w1 - L_5_2*w2 - L_5_3*w3 - L_5_4*w4)/L_5_5 : 0.0
        w6 = nb>=6 ? (v6 - L_6_1*w1 - L_6_2*w2 - L_6_3*w3 - L_6_4*w4 - L_6_5*w5)/L_6_6 : 0.0
        w7 = nb>=7 ? (v7 - L_7_1*w1 - L_7_2*w2 - L_7_3*w3 - L_7_4*w4 - L_7_5*w5 - L_7_6*w6)/L_7_7 : 0.0
        w8 = nb>=8 ? (v8 - L_8_1*w1 - L_8_2*w2 - L_8_3*w3 - L_8_4*w4 - L_8_5*w5 - L_8_6*w6 - L_8_7*w7)/L_8_8 : 0.0
        w9 = nb>=9 ? (v9 - L_9_1*w1 - L_9_2*w2 - L_9_3*w3 - L_9_4*w4 - L_9_5*w5 - L_9_6*w6 - L_9_7*w7 - L_9_8*w8)/L_9_9 : 0.0
        # back solve Lᵀ y = w
        y9 = nb>=9 ? w9/L_9_9 : 0.0
        y8 = nb>=8 ? (w8 - L_9_8*y9)/L_8_8 : 0.0
        y7 = nb>=7 ? (w7 - L_8_7*y8 - L_9_7*y9)/L_7_7 : 0.0
        y6 = nb>=6 ? (w6 - L_7_6*y7 - L_8_6*y8 - L_9_6*y9)/L_6_6 : 0.0
        y5 = nb>=5 ? (w5 - L_6_5*y6 - L_7_5*y7 - L_8_5*y8 - L_9_5*y9)/L_5_5 : 0.0
        y4 = nb>=4 ? (w4 - L_5_4*y5 - L_6_4*y6 - L_7_4*y7 - L_8_4*y8 - L_9_4*y9)/L_4_4 : 0.0
        y3 = nb>=3 ? (w3 - L_4_3*y4 - L_5_3*y5 - L_6_3*y6 - L_7_3*y7 - L_8_3*y8 - L_9_3*y9)/L_3_3 : 0.0
        y2 = nb>=2 ? (w2 - L_3_2*y3 - L_4_2*y4 - L_5_2*y5 - L_6_2*y6 - L_7_2*y7 - L_8_2*y8 - L_9_2*y9)/L_2_2 : 0.0
        y1 = (w1 - L_2_1*y2 - L_3_1*y3 - L_4_1*y4 - L_5_1*y5 - L_6_1*y6 - L_7_1*y7 - L_8_1*y8 - L_9_1*y9)/L_1_1
        nrm2 = y1*y1+y2*y2+y3*y3+y4*y4+y5*y5+y6*y6+y7*y7+y8*y8+y9*y9
        nrm2 <= 0.0 && break
        nrm = sqrt(nrm2)
        v1=y1/nrm; v2=y2/nrm; v3=y3/nrm; v4=y4/nrm; v5=y5/nrm
        v6=y6/nrm; v7=y7/nrm; v8=y8/nrm; v9=y9/nrm
        (z1,z2,z3,z4,z5,z6,z7,z8,z9) = _gram_matvec9(gl, v1,v2,v3,v4,v5,v6,v7,v8,v9, nb)
        lam_min = v1*z1+v2*z2+v3*z3+v4*z4+v5*z5+v6*z6+v7*z7+v8*z8+v9*z9
    end
    lam_min <= 0.0 && return Inf
    return lam_max/lam_min
end

# 6-cap (N<=6) faithful gate: zero-pad the flat lower-tri length-21 form up to the
# length-45 form and call condkappa9. Used by the y-level (nb<=3).
@inline function condkappa6(gl::NTuple{21,Float64}, nb::Int)
    gl45 = (gl[1],gl[2],gl[3],gl[4],gl[5],gl[6],gl[7],gl[8],gl[9],gl[10],
            gl[11],gl[12],gl[13],gl[14],gl[15],gl[16],gl[17],gl[18],gl[19],gl[20],gl[21],
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    return condkappa9(gl45, nb)
end

# Condition gate on the SPD Gram G (flat lower-tri). Returns an order-of-magnitude
# proxy for kappa(G); accept iff < 1e8 (= (kappa(B)=1e4)^2).
# RETAINED for reference/measurement only (fix1_gate_probe.jl compares it); the
# production gate is condkappa6/condkappa9 (true kappa). NOT called on the hot path.
#
# IMPORTANT (measured on real cells, see KFVS_INVERSION_KERNEL_REPORT.md): the
# bare Cholesky PIVOT RATIO max_i d_i / min_i d_i UNDER-estimates kappa(G) by many
# orders of magnitude when the near-linear-dependence is spread across the basis
# (a near-collinear parent cloud) — it admitted a cubic column that the CPU svd
# gate rejected, blowing up the fit. The robust, still-free signal is the SMALLEST
# pivot relative to the matrix SCALE (the largest Gram diagonal, = the natural
# column normalization): a near-dependent column drives some pivot d_i toward 0
# while max_k G_kk stays O(1), so
#     condproxy = max(max_diag(G), max_i d_i) / min_i d_i
# tracks kappa(G) faithfully (it is an UPPER bound: min d_i <= sigma_min(G), and
# max G_kk >= sigma_max(G)/n). This correctly rejects the near-dependent cubic
# (min-pivot/max-diag ~ 6e-15 => proxy ~ 1e17 >> 1e8) and matches the CPU gate.
@inline function cond_proxy6(gl::NTuple{21,Float64}, nb::Int)
    g1 = gl[1]
    g1 <= 0.0 && return Inf
    l11 = sqrt(g1)
    dmin = g1; dmax = g1
    gdmax = gl[1]                    # running max Gram diagonal (= scale)
    nb < 2 && return dmax/dmin
    gdmax = gl[3] > gdmax ? gl[3] : gdmax
    l21 = gl[2]/l11
    d2 = gl[3] - l21*l21; d2 <= 0.0 && return Inf
    dmin = d2 < dmin ? d2 : dmin; dmax = d2 > dmax ? d2 : dmax
    nb < 3 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[6] > gdmax ? gl[6] : gdmax
    l31 = gl[4]/l11; l32 = (gl[5] - l31*l21)/sqrt(d2)
    d3 = gl[6] - l31*l31 - l32*l32; d3 <= 0.0 && return Inf
    dmin = d3 < dmin ? d3 : dmin; dmax = d3 > dmax ? d3 : dmax
    nb < 4 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[10] > gdmax ? gl[10] : gdmax
    l22 = sqrt(d2); l33 = sqrt(d3)
    l41 = gl[7]/l11; l42 = (gl[8] - l41*l21)/l22
    l43 = (gl[9] - l41*l31 - l42*l32)/l33
    d4 = gl[10] - l41*l41 - l42*l42 - l43*l43; d4 <= 0.0 && return Inf
    dmin = d4 < dmin ? d4 : dmin; dmax = d4 > dmax ? d4 : dmax
    nb < 5 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[15] > gdmax ? gl[15] : gdmax
    l44 = sqrt(d4)
    l51 = gl[11]/l11; l52 = (gl[12] - l51*l21)/l22
    l53 = (gl[13] - l51*l31 - l52*l32)/l33
    l54 = (gl[14] - l51*l41 - l52*l42 - l53*l43)/l44
    d5 = gl[15] - l51*l51 - l52*l52 - l53*l53 - l54*l54; d5 <= 0.0 && return Inf
    dmin = d5 < dmin ? d5 : dmin; dmax = d5 > dmax ? d5 : dmax
    nb < 6 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[21] > gdmax ? gl[21] : gdmax
    l55 = sqrt(d5)
    l61 = gl[16]/l11; l62 = (gl[17] - l61*l21)/l22
    l63 = (gl[18] - l61*l31 - l62*l32)/l33
    l64 = (gl[19] - l61*l41 - l62*l42 - l63*l43)/l44
    l65 = (gl[20] - l61*l51 - l62*l52 - l63*l53 - l64*l54)/l55
    d6 = gl[21] - l61*l61 - l62*l62 - l63*l63 - l64*l64 - l65*l65; d6 <= 0.0 && return Inf
    dmin = d6 < dmin ? d6 : dmin; dmax = d6 > dmax ? d6 : dmax
    return _cp_ret(dmax, gdmax, dmin)
end
@inline _cp_ret(dmax, gdmax, dmin) = (gdmax > dmax ? gdmax : dmax) / dmin

# SPD solve G x = b, leading nb block. Returns NTuple{6} (garbage beyond nb).
@inline function spd_solve6(gl::NTuple{21,Float64}, b::NTuple{6,Float64}, nb::Int)
    L11 = sqrt(gl[1])
    L21 = nb >= 2 ? gl[2]/L11 : 0.0
    L31 = nb >= 3 ? gl[4]/L11 : 0.0
    L41 = nb >= 4 ? gl[7]/L11 : 0.0
    L51 = nb >= 5 ? gl[11]/L11 : 0.0
    L61 = nb >= 6 ? gl[16]/L11 : 0.0
    d2  = gl[3] - L21*L21
    L22 = nb >= 2 ? sqrt(d2) : 1.0
    L32 = nb >= 3 ? (gl[5] - L31*L21)/L22 : 0.0
    L42 = nb >= 4 ? (gl[8] - L41*L21)/L22 : 0.0
    L52 = nb >= 5 ? (gl[12] - L51*L21)/L22 : 0.0
    L62 = nb >= 6 ? (gl[17] - L61*L21)/L22 : 0.0
    d3  = gl[6] - L31*L31 - L32*L32
    L33 = nb >= 3 ? sqrt(d3) : 1.0
    L43 = nb >= 4 ? (gl[9] - L41*L31 - L42*L32)/L33 : 0.0
    L53 = nb >= 5 ? (gl[13] - L51*L31 - L52*L32)/L33 : 0.0
    L63 = nb >= 6 ? (gl[18] - L61*L31 - L62*L32)/L33 : 0.0
    d4  = gl[10] - L41*L41 - L42*L42 - L43*L43
    L44 = nb >= 4 ? sqrt(d4) : 1.0
    L54 = nb >= 5 ? (gl[14] - L51*L41 - L52*L42 - L53*L43)/L44 : 0.0
    L64 = nb >= 6 ? (gl[19] - L61*L41 - L62*L42 - L63*L43)/L44 : 0.0
    d5  = gl[15] - L51*L51 - L52*L52 - L53*L53 - L54*L54
    L55 = nb >= 5 ? sqrt(d5) : 1.0
    L65 = nb >= 6 ? (gl[20] - L61*L51 - L62*L52 - L63*L53 - L64*L54)/L55 : 0.0
    d6  = gl[21] - L61*L61 - L62*L62 - L63*L63 - L64*L64 - L65*L65
    L66 = nb >= 6 ? sqrt(d6) : 1.0

    y1 = b[1]/L11
    y2 = nb >= 2 ? (b[2] - L21*y1)/L22 : 0.0
    y3 = nb >= 3 ? (b[3] - L31*y1 - L32*y2)/L33 : 0.0
    y4 = nb >= 4 ? (b[4] - L41*y1 - L42*y2 - L43*y3)/L44 : 0.0
    y5 = nb >= 5 ? (b[5] - L51*y1 - L52*y2 - L53*y3 - L54*y4)/L55 : 0.0
    y6 = nb >= 6 ? (b[6] - L61*y1 - L62*y2 - L63*y3 - L64*y4 - L65*y5)/L66 : 0.0

    x6 = nb >= 6 ? y6/L66 : 0.0
    x5 = nb >= 5 ? (y5 - L65*x6)/L55 : 0.0
    x4 = nb >= 4 ? (y4 - L54*x5 - L64*x6)/L44 : 0.0
    x3 = nb >= 3 ? (y3 - L43*x4 - L53*x5 - L63*x6)/L33 : 0.0
    x2 = nb >= 2 ? (y2 - L32*x3 - L42*x4 - L52*x5 - L62*x6)/L22 : 0.0
    x1 = (y1 - L21*x2 - L31*x3 - L41*x4 - L51*x5 - L61*x6)/L11
    return (x1, x2, x3, x4, x5, x6)
end

# ===========================================================================
# N=9 fixed-size Cholesky gate + solve + z-Gram builder (auto-generated, unrolled).
# Used by the z-level fit (real cells admit up to 9 z-mean columns).
# ===========================================================================
@inline function cond_proxy9(gl::NTuple{45,Float64}, nb::Int)
    gl[1] <= 0.0 && return Inf
    L_1_1 = sqrt(gl[1])
    dmin = gl[1]; dmax = gl[1]; gdmax = gl[1]
    nb < 2 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[3] > gdmax ? gl[3] : gdmax
    L_2_1 = (gl[2])/L_1_1
    d2 = gl[3] - L_2_1*L_2_1; d2 <= 0.0 && return Inf
    dmin = d2 < dmin ? d2 : dmin; dmax = d2 > dmax ? d2 : dmax
    L_2_2 = sqrt(d2)
    nb < 3 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[6] > gdmax ? gl[6] : gdmax
    L_3_1 = (gl[4])/L_1_1
    L_3_2 = (gl[5] - L_3_1*L_2_1)/L_2_2
    d3 = gl[6] - L_3_1*L_3_1 - L_3_2*L_3_2; d3 <= 0.0 && return Inf
    dmin = d3 < dmin ? d3 : dmin; dmax = d3 > dmax ? d3 : dmax
    L_3_3 = sqrt(d3)
    nb < 4 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[10] > gdmax ? gl[10] : gdmax
    L_4_1 = (gl[7])/L_1_1
    L_4_2 = (gl[8] - L_4_1*L_2_1)/L_2_2
    L_4_3 = (gl[9] - L_4_1*L_3_1 - L_4_2*L_3_2)/L_3_3
    d4 = gl[10] - L_4_1*L_4_1 - L_4_2*L_4_2 - L_4_3*L_4_3; d4 <= 0.0 && return Inf
    dmin = d4 < dmin ? d4 : dmin; dmax = d4 > dmax ? d4 : dmax
    L_4_4 = sqrt(d4)
    nb < 5 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[15] > gdmax ? gl[15] : gdmax
    L_5_1 = (gl[11])/L_1_1
    L_5_2 = (gl[12] - L_5_1*L_2_1)/L_2_2
    L_5_3 = (gl[13] - L_5_1*L_3_1 - L_5_2*L_3_2)/L_3_3
    L_5_4 = (gl[14] - L_5_1*L_4_1 - L_5_2*L_4_2 - L_5_3*L_4_3)/L_4_4
    d5 = gl[15] - L_5_1*L_5_1 - L_5_2*L_5_2 - L_5_3*L_5_3 - L_5_4*L_5_4; d5 <= 0.0 && return Inf
    dmin = d5 < dmin ? d5 : dmin; dmax = d5 > dmax ? d5 : dmax
    L_5_5 = sqrt(d5)
    nb < 6 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[21] > gdmax ? gl[21] : gdmax
    L_6_1 = (gl[16])/L_1_1
    L_6_2 = (gl[17] - L_6_1*L_2_1)/L_2_2
    L_6_3 = (gl[18] - L_6_1*L_3_1 - L_6_2*L_3_2)/L_3_3
    L_6_4 = (gl[19] - L_6_1*L_4_1 - L_6_2*L_4_2 - L_6_3*L_4_3)/L_4_4
    L_6_5 = (gl[20] - L_6_1*L_5_1 - L_6_2*L_5_2 - L_6_3*L_5_3 - L_6_4*L_5_4)/L_5_5
    d6 = gl[21] - L_6_1*L_6_1 - L_6_2*L_6_2 - L_6_3*L_6_3 - L_6_4*L_6_4 - L_6_5*L_6_5; d6 <= 0.0 && return Inf
    dmin = d6 < dmin ? d6 : dmin; dmax = d6 > dmax ? d6 : dmax
    L_6_6 = sqrt(d6)
    nb < 7 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[28] > gdmax ? gl[28] : gdmax
    L_7_1 = (gl[22])/L_1_1
    L_7_2 = (gl[23] - L_7_1*L_2_1)/L_2_2
    L_7_3 = (gl[24] - L_7_1*L_3_1 - L_7_2*L_3_2)/L_3_3
    L_7_4 = (gl[25] - L_7_1*L_4_1 - L_7_2*L_4_2 - L_7_3*L_4_3)/L_4_4
    L_7_5 = (gl[26] - L_7_1*L_5_1 - L_7_2*L_5_2 - L_7_3*L_5_3 - L_7_4*L_5_4)/L_5_5
    L_7_6 = (gl[27] - L_7_1*L_6_1 - L_7_2*L_6_2 - L_7_3*L_6_3 - L_7_4*L_6_4 - L_7_5*L_6_5)/L_6_6
    d7 = gl[28] - L_7_1*L_7_1 - L_7_2*L_7_2 - L_7_3*L_7_3 - L_7_4*L_7_4 - L_7_5*L_7_5 - L_7_6*L_7_6; d7 <= 0.0 && return Inf
    dmin = d7 < dmin ? d7 : dmin; dmax = d7 > dmax ? d7 : dmax
    L_7_7 = sqrt(d7)
    nb < 8 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[36] > gdmax ? gl[36] : gdmax
    L_8_1 = (gl[29])/L_1_1
    L_8_2 = (gl[30] - L_8_1*L_2_1)/L_2_2
    L_8_3 = (gl[31] - L_8_1*L_3_1 - L_8_2*L_3_2)/L_3_3
    L_8_4 = (gl[32] - L_8_1*L_4_1 - L_8_2*L_4_2 - L_8_3*L_4_3)/L_4_4
    L_8_5 = (gl[33] - L_8_1*L_5_1 - L_8_2*L_5_2 - L_8_3*L_5_3 - L_8_4*L_5_4)/L_5_5
    L_8_6 = (gl[34] - L_8_1*L_6_1 - L_8_2*L_6_2 - L_8_3*L_6_3 - L_8_4*L_6_4 - L_8_5*L_6_5)/L_6_6
    L_8_7 = (gl[35] - L_8_1*L_7_1 - L_8_2*L_7_2 - L_8_3*L_7_3 - L_8_4*L_7_4 - L_8_5*L_7_5 - L_8_6*L_7_6)/L_7_7
    d8 = gl[36] - L_8_1*L_8_1 - L_8_2*L_8_2 - L_8_3*L_8_3 - L_8_4*L_8_4 - L_8_5*L_8_5 - L_8_6*L_8_6 - L_8_7*L_8_7; d8 <= 0.0 && return Inf
    dmin = d8 < dmin ? d8 : dmin; dmax = d8 > dmax ? d8 : dmax
    L_8_8 = sqrt(d8)
    nb < 9 && return _cp_ret(dmax, gdmax, dmin)
    gdmax = gl[45] > gdmax ? gl[45] : gdmax
    L_9_1 = (gl[37])/L_1_1
    L_9_2 = (gl[38] - L_9_1*L_2_1)/L_2_2
    L_9_3 = (gl[39] - L_9_1*L_3_1 - L_9_2*L_3_2)/L_3_3
    L_9_4 = (gl[40] - L_9_1*L_4_1 - L_9_2*L_4_2 - L_9_3*L_4_3)/L_4_4
    L_9_5 = (gl[41] - L_9_1*L_5_1 - L_9_2*L_5_2 - L_9_3*L_5_3 - L_9_4*L_5_4)/L_5_5
    L_9_6 = (gl[42] - L_9_1*L_6_1 - L_9_2*L_6_2 - L_9_3*L_6_3 - L_9_4*L_6_4 - L_9_5*L_6_5)/L_6_6
    L_9_7 = (gl[43] - L_9_1*L_7_1 - L_9_2*L_7_2 - L_9_3*L_7_3 - L_9_4*L_7_4 - L_9_5*L_7_5 - L_9_6*L_7_6)/L_7_7
    L_9_8 = (gl[44] - L_9_1*L_8_1 - L_9_2*L_8_2 - L_9_3*L_8_3 - L_9_4*L_8_4 - L_9_5*L_8_5 - L_9_6*L_8_6 - L_9_7*L_8_7)/L_8_8
    d9 = gl[45] - L_9_1*L_9_1 - L_9_2*L_9_2 - L_9_3*L_9_3 - L_9_4*L_9_4 - L_9_5*L_9_5 - L_9_6*L_9_6 - L_9_7*L_9_7 - L_9_8*L_9_8; d9 <= 0.0 && return Inf
    dmin = d9 < dmin ? d9 : dmin; dmax = d9 > dmax ? d9 : dmax
    L_9_9 = sqrt(d9)
    return _cp_ret(dmax, gdmax, dmin)
end

@inline function spd_solve9(gl::NTuple{45,Float64}, b::NTuple{9,Float64}, nb::Int)
    L_1_1 = sqrt(gl[1])
    L_2_1 = nb >= 2 ? (gl[2])/L_1_1 : 0.0
    d2 = gl[3] - L_2_1*L_2_1
    L_2_2 = nb >= 2 ? sqrt(d2) : 1.0
    L_3_1 = nb >= 3 ? (gl[4])/L_1_1 : 0.0
    L_3_2 = nb >= 3 ? (gl[5] - L_3_1*L_2_1)/L_2_2 : 0.0
    d3 = gl[6] - L_3_1*L_3_1 - L_3_2*L_3_2
    L_3_3 = nb >= 3 ? sqrt(d3) : 1.0
    L_4_1 = nb >= 4 ? (gl[7])/L_1_1 : 0.0
    L_4_2 = nb >= 4 ? (gl[8] - L_4_1*L_2_1)/L_2_2 : 0.0
    L_4_3 = nb >= 4 ? (gl[9] - L_4_1*L_3_1 - L_4_2*L_3_2)/L_3_3 : 0.0
    d4 = gl[10] - L_4_1*L_4_1 - L_4_2*L_4_2 - L_4_3*L_4_3
    L_4_4 = nb >= 4 ? sqrt(d4) : 1.0
    L_5_1 = nb >= 5 ? (gl[11])/L_1_1 : 0.0
    L_5_2 = nb >= 5 ? (gl[12] - L_5_1*L_2_1)/L_2_2 : 0.0
    L_5_3 = nb >= 5 ? (gl[13] - L_5_1*L_3_1 - L_5_2*L_3_2)/L_3_3 : 0.0
    L_5_4 = nb >= 5 ? (gl[14] - L_5_1*L_4_1 - L_5_2*L_4_2 - L_5_3*L_4_3)/L_4_4 : 0.0
    d5 = gl[15] - L_5_1*L_5_1 - L_5_2*L_5_2 - L_5_3*L_5_3 - L_5_4*L_5_4
    L_5_5 = nb >= 5 ? sqrt(d5) : 1.0
    L_6_1 = nb >= 6 ? (gl[16])/L_1_1 : 0.0
    L_6_2 = nb >= 6 ? (gl[17] - L_6_1*L_2_1)/L_2_2 : 0.0
    L_6_3 = nb >= 6 ? (gl[18] - L_6_1*L_3_1 - L_6_2*L_3_2)/L_3_3 : 0.0
    L_6_4 = nb >= 6 ? (gl[19] - L_6_1*L_4_1 - L_6_2*L_4_2 - L_6_3*L_4_3)/L_4_4 : 0.0
    L_6_5 = nb >= 6 ? (gl[20] - L_6_1*L_5_1 - L_6_2*L_5_2 - L_6_3*L_5_3 - L_6_4*L_5_4)/L_5_5 : 0.0
    d6 = gl[21] - L_6_1*L_6_1 - L_6_2*L_6_2 - L_6_3*L_6_3 - L_6_4*L_6_4 - L_6_5*L_6_5
    L_6_6 = nb >= 6 ? sqrt(d6) : 1.0
    L_7_1 = nb >= 7 ? (gl[22])/L_1_1 : 0.0
    L_7_2 = nb >= 7 ? (gl[23] - L_7_1*L_2_1)/L_2_2 : 0.0
    L_7_3 = nb >= 7 ? (gl[24] - L_7_1*L_3_1 - L_7_2*L_3_2)/L_3_3 : 0.0
    L_7_4 = nb >= 7 ? (gl[25] - L_7_1*L_4_1 - L_7_2*L_4_2 - L_7_3*L_4_3)/L_4_4 : 0.0
    L_7_5 = nb >= 7 ? (gl[26] - L_7_1*L_5_1 - L_7_2*L_5_2 - L_7_3*L_5_3 - L_7_4*L_5_4)/L_5_5 : 0.0
    L_7_6 = nb >= 7 ? (gl[27] - L_7_1*L_6_1 - L_7_2*L_6_2 - L_7_3*L_6_3 - L_7_4*L_6_4 - L_7_5*L_6_5)/L_6_6 : 0.0
    d7 = gl[28] - L_7_1*L_7_1 - L_7_2*L_7_2 - L_7_3*L_7_3 - L_7_4*L_7_4 - L_7_5*L_7_5 - L_7_6*L_7_6
    L_7_7 = nb >= 7 ? sqrt(d7) : 1.0
    L_8_1 = nb >= 8 ? (gl[29])/L_1_1 : 0.0
    L_8_2 = nb >= 8 ? (gl[30] - L_8_1*L_2_1)/L_2_2 : 0.0
    L_8_3 = nb >= 8 ? (gl[31] - L_8_1*L_3_1 - L_8_2*L_3_2)/L_3_3 : 0.0
    L_8_4 = nb >= 8 ? (gl[32] - L_8_1*L_4_1 - L_8_2*L_4_2 - L_8_3*L_4_3)/L_4_4 : 0.0
    L_8_5 = nb >= 8 ? (gl[33] - L_8_1*L_5_1 - L_8_2*L_5_2 - L_8_3*L_5_3 - L_8_4*L_5_4)/L_5_5 : 0.0
    L_8_6 = nb >= 8 ? (gl[34] - L_8_1*L_6_1 - L_8_2*L_6_2 - L_8_3*L_6_3 - L_8_4*L_6_4 - L_8_5*L_6_5)/L_6_6 : 0.0
    L_8_7 = nb >= 8 ? (gl[35] - L_8_1*L_7_1 - L_8_2*L_7_2 - L_8_3*L_7_3 - L_8_4*L_7_4 - L_8_5*L_7_5 - L_8_6*L_7_6)/L_7_7 : 0.0
    d8 = gl[36] - L_8_1*L_8_1 - L_8_2*L_8_2 - L_8_3*L_8_3 - L_8_4*L_8_4 - L_8_5*L_8_5 - L_8_6*L_8_6 - L_8_7*L_8_7
    L_8_8 = nb >= 8 ? sqrt(d8) : 1.0
    L_9_1 = nb >= 9 ? (gl[37])/L_1_1 : 0.0
    L_9_2 = nb >= 9 ? (gl[38] - L_9_1*L_2_1)/L_2_2 : 0.0
    L_9_3 = nb >= 9 ? (gl[39] - L_9_1*L_3_1 - L_9_2*L_3_2)/L_3_3 : 0.0
    L_9_4 = nb >= 9 ? (gl[40] - L_9_1*L_4_1 - L_9_2*L_4_2 - L_9_3*L_4_3)/L_4_4 : 0.0
    L_9_5 = nb >= 9 ? (gl[41] - L_9_1*L_5_1 - L_9_2*L_5_2 - L_9_3*L_5_3 - L_9_4*L_5_4)/L_5_5 : 0.0
    L_9_6 = nb >= 9 ? (gl[42] - L_9_1*L_6_1 - L_9_2*L_6_2 - L_9_3*L_6_3 - L_9_4*L_6_4 - L_9_5*L_6_5)/L_6_6 : 0.0
    L_9_7 = nb >= 9 ? (gl[43] - L_9_1*L_7_1 - L_9_2*L_7_2 - L_9_3*L_7_3 - L_9_4*L_7_4 - L_9_5*L_7_5 - L_9_6*L_7_6)/L_7_7 : 0.0
    L_9_8 = nb >= 9 ? (gl[44] - L_9_1*L_8_1 - L_9_2*L_8_2 - L_9_3*L_8_3 - L_9_4*L_8_4 - L_9_5*L_8_5 - L_9_6*L_8_6 - L_9_7*L_8_7)/L_8_8 : 0.0
    d9 = gl[45] - L_9_1*L_9_1 - L_9_2*L_9_2 - L_9_3*L_9_3 - L_9_4*L_9_4 - L_9_5*L_9_5 - L_9_6*L_9_6 - L_9_7*L_9_7 - L_9_8*L_9_8
    L_9_9 = nb >= 9 ? sqrt(d9) : 1.0
    y1 = b[1]/L_1_1
    y2 = nb >= 2 ? (b[2] - L_2_1*y1)/L_2_2 : 0.0
    y3 = nb >= 3 ? (b[3] - L_3_1*y1 - L_3_2*y2)/L_3_3 : 0.0
    y4 = nb >= 4 ? (b[4] - L_4_1*y1 - L_4_2*y2 - L_4_3*y3)/L_4_4 : 0.0
    y5 = nb >= 5 ? (b[5] - L_5_1*y1 - L_5_2*y2 - L_5_3*y3 - L_5_4*y4)/L_5_5 : 0.0
    y6 = nb >= 6 ? (b[6] - L_6_1*y1 - L_6_2*y2 - L_6_3*y3 - L_6_4*y4 - L_6_5*y5)/L_6_6 : 0.0
    y7 = nb >= 7 ? (b[7] - L_7_1*y1 - L_7_2*y2 - L_7_3*y3 - L_7_4*y4 - L_7_5*y5 - L_7_6*y6)/L_7_7 : 0.0
    y8 = nb >= 8 ? (b[8] - L_8_1*y1 - L_8_2*y2 - L_8_3*y3 - L_8_4*y4 - L_8_5*y5 - L_8_6*y6 - L_8_7*y7)/L_8_8 : 0.0
    y9 = nb >= 9 ? (b[9] - L_9_1*y1 - L_9_2*y2 - L_9_3*y3 - L_9_4*y4 - L_9_5*y5 - L_9_6*y6 - L_9_7*y7 - L_9_8*y8)/L_9_9 : 0.0
    x9 = nb >= 9 ? y9/L_9_9 : 0.0
    x8 = nb >= 8 ? (y8 - L_9_8*x9)/L_8_8 : 0.0
    x7 = nb >= 7 ? (y7 - L_8_7*x8 - L_9_7*x9)/L_7_7 : 0.0
    x6 = nb >= 6 ? (y6 - L_7_6*x7 - L_8_6*x8 - L_9_6*x9)/L_6_6 : 0.0
    x5 = nb >= 5 ? (y5 - L_6_5*x6 - L_7_5*x7 - L_8_5*x8 - L_9_5*x9)/L_5_5 : 0.0
    x4 = nb >= 4 ? (y4 - L_5_4*x5 - L_6_4*x6 - L_7_4*x7 - L_8_4*x8 - L_9_4*x9)/L_4_4 : 0.0
    x3 = nb >= 3 ? (y3 - L_4_3*x4 - L_5_3*x5 - L_6_3*x6 - L_7_3*x7 - L_8_3*x8 - L_9_3*x9)/L_3_3 : 0.0
    x2 = nb >= 2 ? (y2 - L_3_2*x3 - L_4_2*x4 - L_5_2*x5 - L_6_2*x6 - L_7_2*x7 - L_8_2*x8 - L_9_2*x9)/L_2_2 : 0.0
    x1 = (y1 - L_2_1*x2 - L_3_1*x3 - L_4_1*x4 - L_5_1*x5 - L_6_1*x6 - L_7_1*x7 - L_8_1*x8 - L_9_1*x9)/L_1_1
    return (x1, x2, x3, x4, x5, x6, x7, x8, x9)
end

@inline function _build_gram_flat_z9(pw::NTuple{9,Float64}, cx::NTuple{9,Float64},
                                     cy::NTuple{9,Float64}, Np::Int,
                                     si::NTuple{9,Int}, sj::NTuple{9,Int}, nsel::Int,
                                     xi::Int, xj::Int)
    ncol = nsel + (xi >= 0 ? 1 : 0)
    ei1 = ncol>=1 ? (1<=nsel ? si[1] : xi) : 0
    ej1 = ncol>=1 ? (1<=nsel ? sj[1] : xj) : 0
    ei2 = ncol>=2 ? (2<=nsel ? si[2] : xi) : 0
    ej2 = ncol>=2 ? (2<=nsel ? sj[2] : xj) : 0
    ei3 = ncol>=3 ? (3<=nsel ? si[3] : xi) : 0
    ej3 = ncol>=3 ? (3<=nsel ? sj[3] : xj) : 0
    ei4 = ncol>=4 ? (4<=nsel ? si[4] : xi) : 0
    ej4 = ncol>=4 ? (4<=nsel ? sj[4] : xj) : 0
    ei5 = ncol>=5 ? (5<=nsel ? si[5] : xi) : 0
    ej5 = ncol>=5 ? (5<=nsel ? sj[5] : xj) : 0
    ei6 = ncol>=6 ? (6<=nsel ? si[6] : xi) : 0
    ej6 = ncol>=6 ? (6<=nsel ? sj[6] : xj) : 0
    ei7 = ncol>=7 ? (7<=nsel ? si[7] : xi) : 0
    ej7 = ncol>=7 ? (7<=nsel ? sj[7] : xj) : 0
    ei8 = ncol>=8 ? (8<=nsel ? si[8] : xi) : 0
    ej8 = ncol>=8 ? (8<=nsel ? sj[8] : xj) : 0
    ei9 = ncol>=9 ? (9<=nsel ? si[9] : xi) : 0
    ej9 = ncol>=9 ? (9<=nsel ? sj[9] : xj) : 0
    G11=0.0
    G21=0.0
    G22=0.0
    G31=0.0
    G32=0.0
    G33=0.0
    G41=0.0
    G42=0.0
    G43=0.0
    G44=0.0
    G51=0.0
    G52=0.0
    G53=0.0
    G54=0.0
    G55=0.0
    G61=0.0
    G62=0.0
    G63=0.0
    G64=0.0
    G65=0.0
    G66=0.0
    G71=0.0
    G72=0.0
    G73=0.0
    G74=0.0
    G75=0.0
    G76=0.0
    G77=0.0
    G81=0.0
    G82=0.0
    G83=0.0
    G84=0.0
    G85=0.0
    G86=0.0
    G87=0.0
    G88=0.0
    G91=0.0
    G92=0.0
    G93=0.0
    G94=0.0
    G95=0.0
    G96=0.0
    G97=0.0
    G98=0.0
    G99=0.0
    for p in 1:9
        (p > Np) && break
        w = pw[p]; x = cx[p]; y = cy[p]
        m1 = ncol>=1 ? _mono2(ei1,ej1,x,y) : 0.0
        m2 = ncol>=2 ? _mono2(ei2,ej2,x,y) : 0.0
        m3 = ncol>=3 ? _mono2(ei3,ej3,x,y) : 0.0
        m4 = ncol>=4 ? _mono2(ei4,ej4,x,y) : 0.0
        m5 = ncol>=5 ? _mono2(ei5,ej5,x,y) : 0.0
        m6 = ncol>=6 ? _mono2(ei6,ej6,x,y) : 0.0
        m7 = ncol>=7 ? _mono2(ei7,ej7,x,y) : 0.0
        m8 = ncol>=8 ? _mono2(ei8,ej8,x,y) : 0.0
        m9 = ncol>=9 ? _mono2(ei9,ej9,x,y) : 0.0
        G11 += w*m1*m1
        G21 += w*m2*m1
        G22 += w*m2*m2
        G31 += w*m3*m1
        G32 += w*m3*m2
        G33 += w*m3*m3
        G41 += w*m4*m1
        G42 += w*m4*m2
        G43 += w*m4*m3
        G44 += w*m4*m4
        G51 += w*m5*m1
        G52 += w*m5*m2
        G53 += w*m5*m3
        G54 += w*m5*m4
        G55 += w*m5*m5
        G61 += w*m6*m1
        G62 += w*m6*m2
        G63 += w*m6*m3
        G64 += w*m6*m4
        G65 += w*m6*m5
        G66 += w*m6*m6
        G71 += w*m7*m1
        G72 += w*m7*m2
        G73 += w*m7*m3
        G74 += w*m7*m4
        G75 += w*m7*m5
        G76 += w*m7*m6
        G77 += w*m7*m7
        G81 += w*m8*m1
        G82 += w*m8*m2
        G83 += w*m8*m3
        G84 += w*m8*m4
        G85 += w*m8*m5
        G86 += w*m8*m6
        G87 += w*m8*m7
        G88 += w*m8*m8
        G91 += w*m9*m1
        G92 += w*m9*m2
        G93 += w*m9*m3
        G94 += w*m9*m4
        G95 += w*m9*m5
        G96 += w*m9*m6
        G97 += w*m9*m7
        G98 += w*m9*m8
        G99 += w*m9*m9
    end
    return (G11, G21, G22, G31, G32, G33, G41, G42, G43, G44, G51, G52, G53, G54, G55, G61, G62, G63, G64, G65, G66, G71, G72, G73, G74, G75, G76, G77, G81, G82, G83, G84, G85, G86, G87, G88, G91, G92, G93, G94, G95, G96, G97, G98, G99)
end


# ===========================================================================
# Conditional-direction fit at the Y level (1D conditioning: coords = centered
# x-abscissas Upx[1..Nx], D=1). Monomial basis {1, x, x^2}, low-degree first,
# condition-gated. Returns Vf[1..3] (per x-node conditional mean deviation),
# var[1..3], shared q, eta. Mirrors _condition_direction/_gram_fit with D=1.
#
# targets for the mean: e=0,1,2 with values (0, c110, c210).
# targets for the var : e=0,1,2 with values (c020, c120, c220).
# ===========================================================================

# Build the Gram (flat lower-tri) and rhs for a 1D-conditioning fit over Np<=3
# parent nodes with weights pw[1..3] and centered coords cx[1..3], selected
# columns given by exponent list. We do incremental low-degree-first selection.
# Returns fitted per-node values fit[1..3] (fit_p = sum_m g_m * cx_p^e_m).
@inline function gram_fit_y(pw1, pw2, pw3, cx1, cx2, cx3, Np::Int,
                            t0, t1, t2)
    # candidate exponents in low-degree order: 0,1,2 with targets t0,t1,t2.
    # Incremental selection: for each candidate, tentatively add column, form
    # leading Gram, test PD + condproxy<1e8 + count<=Np. Build selected exponent
    # set as up-to-3 slots.
    # We represent selection with explicit selected-exponent scalars and count.
    # (Gate uses the faithful condkappa6 at CONDMAX_G — see _trial_gram3.)
    # monomial value phi(e,p) = cx_p^e ; weighted col sqrt(pw)*phi
    # Gram entry G[m,k] = sum_p pw_p * cx_p^(e_m) * cx_p^(e_k)
    # rhs t[m] = target_m
    # We attempt to admit exponents 0,1,2 in order.
    # accumulate selected exponents
    e_sel1 = 0; e_sel2 = 0; e_sel3 = 0
    tsel1 = 0.0; tsel2 = 0.0; tsel3 = 0.0
    nsel = 0
    # helper: for a trial set of exponents (nsel+1 of them), build flat lower-tri
    # Gram (size up to 3 here) and return condproxy via cond_proxy6 (nb<=3).
    # We inline three explicit candidate steps.
    # --- candidate exponent 0 (target t0) ---
    # column all-zero check: phi=cx^0=1, never all-zero for Np>=1
    # tentative selected set = {0}
    # Gram 1x1 = sum pw
    @inline mono(e::Int, x) = e==0 ? 1.0 : (e==1 ? x : x*x)
    # try adding candidate with exponent `ec`, target `tc`. Uses current selection.
    # We can't loop with function-valued closures on device; unroll 3 candidates.
    # ---- candidate 0 ----
    ec = 0; tc = t0
    # all-zero column? no (constant). build trial Gram of size nsel+1.
    # trial exponents: (e_sel1..e_sel_nsel, ec)
    # form G flat lower tri (size k=nsel+1)
    # Because nsel starts 0, this is straightforward — but to keep it uniform we
    # write a fixed builder for k up to 3 using the selected slots + candidate.
    accept, g1,g2,g3,g4,g5,g6 = _trial_gram3(pw1,pw2,pw3,cx1,cx2,cx3,Np,
                                             e_sel1,e_sel2,e_sel3,nsel,ec)
    if accept
        nsel += 1
        if nsel==1; e_sel1=ec; tsel1=tc
        elseif nsel==2; e_sel2=ec; tsel2=tc
        else; e_sel3=ec; tsel3=tc; end
    end
    # ---- candidate 1 ----
    ec = 1; tc = t1
    if nsel < Np
        accept, g1,g2,g3,g4,g5,g6 = _trial_gram3(pw1,pw2,pw3,cx1,cx2,cx3,Np,
                                                 e_sel1,e_sel2,e_sel3,nsel,ec)
        if accept
            nsel += 1
            if nsel==1; e_sel1=ec; tsel1=tc
            elseif nsel==2; e_sel2=ec; tsel2=tc
            else; e_sel3=ec; tsel3=tc; end
        end
    end
    # ---- candidate 2 ----
    ec = 2; tc = t2
    if nsel < Np
        accept, g1,g2,g3,g4,g5,g6 = _trial_gram3(pw1,pw2,pw3,cx1,cx2,cx3,Np,
                                                 e_sel1,e_sel2,e_sel3,nsel,ec)
        if accept
            nsel += 1
            if nsel==1; e_sel1=ec; tsel1=tc
            elseif nsel==2; e_sel2=ec; tsel2=tc
            else; e_sel3=ec; tsel3=tc; end
        end
    end

    if nsel == 0
        return (0.0, 0.0, 0.0)
    end

    # Build final Gram (flat lower-tri 21) + rhs for the selected exponents.
    gl = _build_gram_flat(pw1,pw2,pw3,cx1,cx2,cx3,Np, e_sel1,e_sel2,e_sel3,nsel)
    bb = (tsel1, tsel2, tsel3, 0.0, 0.0, 0.0)
    gx = spd_solve6(gl, bb, nsel)
    # fitted value per node p: sum_m g_m * cx_p^{e_m}
    f1 = _apply_fit(gx, e_sel1,e_sel2,e_sel3,nsel, cx1)
    f2 = _apply_fit(gx, e_sel1,e_sel2,e_sel3,nsel, cx2)
    f3 = _apply_fit(gx, e_sel1,e_sel2,e_sel3,nsel, cx3)
    return (f1, f2, f3)
end

@inline function _apply_fit(gx::NTuple{6,Float64}, e1::Int,e2::Int,e3::Int, nsel::Int, x)
    v = 0.0
    if nsel >= 1; v += gx[1] * (e1==0 ? 1.0 : (e1==1 ? x : x*x)); end
    if nsel >= 2; v += gx[2] * (e2==0 ? 1.0 : (e2==1 ? x : x*x)); end
    if nsel >= 3; v += gx[3] * (e3==0 ? 1.0 : (e3==1 ? x : x*x)); end
    return v
end

# Build flat lower-tri Gram (length-21 NTuple) for the selected 1D-conditioning
# exponents (up to 3) over Np parent nodes.
@inline function _build_gram_flat(pw1,pw2,pw3,cx1,cx2,cx3, Np::Int,
                                  e1::Int,e2::Int,e3::Int, nsel::Int)
    @inline ph(e::Int, x) = e==0 ? 1.0 : (e==1 ? x : x*x)
    # phi values per column per node
    p11 = ph(e1,cx1); p12 = ph(e1,cx2); p13 = ph(e1,cx3)
    p21 = nsel>=2 ? ph(e2,cx1) : 0.0; p22 = nsel>=2 ? ph(e2,cx2) : 0.0; p23 = nsel>=2 ? ph(e2,cx3) : 0.0
    p31 = nsel>=3 ? ph(e3,cx1) : 0.0; p32 = nsel>=3 ? ph(e3,cx2) : 0.0; p33 = nsel>=3 ? ph(e3,cx3) : 0.0
    w1 = Np>=1 ? pw1 : 0.0; w2 = Np>=2 ? pw2 : 0.0; w3 = Np>=3 ? pw3 : 0.0
    G11 = w1*p11*p11 + w2*p12*p12 + w3*p13*p13
    G21 = w1*p21*p11 + w2*p22*p12 + w3*p23*p13
    G22 = w1*p21*p21 + w2*p22*p22 + w3*p23*p23
    G31 = w1*p31*p11 + w2*p32*p12 + w3*p33*p13
    G32 = w1*p31*p21 + w2*p32*p22 + w3*p33*p23
    G33 = w1*p31*p31 + w2*p32*p32 + w3*p33*p33
    # flat lower-tri length 21: positions 1,2,3,4,5,6 = (1,1)(2,1)(2,2)(3,1)(3,2)(3,3)
    return (G11, G21, G22, G31, G32, G33,
            0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0)
end

# Try adding candidate exponent `ec` to current selection; return
# (accept::Bool, plus 6 dummy floats for signature uniformity). Tests PD +
# condproxy<1e8 over the leading (nsel+1) Gram, and rejects all-zero column.
@inline function _trial_gram3(pw1,pw2,pw3,cx1,cx2,cx3, Np::Int,
                              e1::Int,e2::Int,e3::Int, nsel::Int, ec::Int)
    @inline ph(e::Int, x) = e==0 ? 1.0 : (e==1 ? x : x*x)
    # all-zero candidate column? (col_p = ph(ec, cx_p)); reject if all zero.
    c1 = Np>=1 ? ph(ec,cx1) : 0.0
    c2 = Np>=2 ? ph(ec,cx2) : 0.0
    c3 = Np>=3 ? ph(ec,cx3) : 0.0
    if c1==0.0 && c2==0.0 && c3==0.0
        return (false, 0.0,0.0,0.0,0.0,0.0,0.0)
    end
    # build trial exponent set with candidate appended
    te1 = e1; te2 = e2; te3 = e3
    k = nsel + 1
    if k==1; te1=ec
    elseif k==2; te2=ec
    else; te3=ec; end
    gl = _build_gram_flat(pw1,pw2,pw3,cx1,cx2,cx3,Np, te1,te2,te3, k)
    # FAITHFUL gate: true kappa(G) = kappa2(B)^2; accept iff kappa2(B) < CONDMAX_B.
    cp = condkappa6(gl, k)
    accept = (cp < CONDMAX_G) && (k <= Np)
    return (accept, 0.0,0.0,0.0,0.0,0.0,0.0)
end

# ===========================================================================
# Z-level conditional-direction fit (2D conditioning: coords = centered (x,y),
# D=2). Monomial candidates ordered low total-degree first. For the z-MEAN
# staircase: exponents (i,j), i+j<=3 -> 10 candidates. For the z-VARIANCE: i+j<=2
# -> 6 candidates. Up to Np<=9 parents; Gram capped at N=6 (design decision:
# treat >6 admitted as drop-to-6-lowest-degree). Uses a 9-parent SPD solve
# capped to nb<=6.
#
# Because the z-level needs up to 9 parents and up to 6 columns, we write a
# dedicated fixed-size 2D fit. Coords carried as cx[1..9], cy[1..9], weights
# pw[1..9]. Candidate monomial list is compile-time fixed.
# ===========================================================================

# monomial (i,j) over (x,y): x^i * y^j
@inline function _mono2(i::Int, j::Int, x, y)
    v = 1.0
    if i==1; v *= x elseif i==2; v *= x*x elseif i==3; v *= x*x*x end
    if j==1; v *= y elseif j==2; v *= y*y elseif j==3; v *= y*y*y end
    return v
end

# The z fit is the most complex. We carry parent state in NTuples of length 9.
# Candidate exponent lists are provided as NTuples of (i,j). We do incremental
# selection up to 6 columns using cond_proxy6 on a 6-cap Gram, then spd_solve6.
#
# To keep the device code unrolled and literal-indexed, we implement the
# incremental selection over a fixed candidate ORDER passed as two NTuples ei,ej
# (length NC) and targets tv (length NC). NC is a compile-time literal (10 for
# mean, 6 for var).
# Sized to N=9: the z-mean staircase over the actual real-cell parent geometry
# admits up to 9 columns (MEASURED: 8 columns is the dominant case on real cells,
# and up to 9 occur — see KFVS_INVERSION_KERNEL_REPORT.md). The design's tentative
# N=6 cap ("drop-to-6") was FALSIFIED by the data (it truncated the dominant case
# and blew up the fit), so we size the fully-unrolled solver to N=9 as the design's
# own fallback (§1.3) instructed.
@inline function gram_fit_z(pw::NTuple{9,Float64}, cx::NTuple{9,Float64}, cy::NTuple{9,Float64},
                            Np::Int, ei::NTuple{10,Int}, ej::NTuple{10,Int},
                            tv::NTuple{10,Float64}, NC::Int)
    se_i = (0,0,0,0,0,0,0,0,0); se_j = (0,0,0,0,0,0,0,0,0)
    se_t = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    nsel = 0
    for k in 1:10
        (k > NC) && break
        nsel >= 9 && break
        nsel >= Np && break
        ci = ei[k]; cj = ej[k]; ct = tv[k]
        allz = true
        for p in 1:9
            (p > Np) && break
            if _mono2(ci, cj, cx[p], cy[p]) != 0.0
                allz = false; break
            end
        end
        allz && continue
        knew = nsel + 1
        gl = _build_gram_flat_z9(pw, cx, cy, Np, se_i, se_j, nsel, ci, cj)
        # FAITHFUL gate: true kappa(G) via Cholesky + iterative extreme eigenvalues.
        cp = condkappa9(gl, knew)
        if cp < CONDMAX_G && knew <= Np
            se_i = _set9i(se_i, knew, ci)
            se_j = _set9i(se_j, knew, cj)
            se_t = _set9f2(se_t, knew, ct)
            nsel = knew
        end
    end

    if nsel == 0
        return (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    end

    gl = _build_gram_flat_z9(pw, cx, cy, Np, se_i, se_j, nsel, -1, -1)
    bb = (se_t[1], se_t[2], se_t[3], se_t[4], se_t[5],
          se_t[6], se_t[7], se_t[8], se_t[9])
    gx = spd_solve9(gl, bb, nsel)
    f1 = _apply_fit_z(gx, se_i, se_j, nsel, cx[1], cy[1])
    f2 = _apply_fit_z(gx, se_i, se_j, nsel, cx[2], cy[2])
    f3 = _apply_fit_z(gx, se_i, se_j, nsel, cx[3], cy[3])
    f4 = _apply_fit_z(gx, se_i, se_j, nsel, cx[4], cy[4])
    f5 = _apply_fit_z(gx, se_i, se_j, nsel, cx[5], cy[5])
    f6 = _apply_fit_z(gx, se_i, se_j, nsel, cx[6], cy[6])
    f7 = _apply_fit_z(gx, se_i, se_j, nsel, cx[7], cy[7])
    f8 = _apply_fit_z(gx, se_i, se_j, nsel, cx[8], cy[8])
    f9 = _apply_fit_z(gx, se_i, se_j, nsel, cx[9], cy[9])
    return (f1,f2,f3,f4,f5,f6,f7,f8,f9)
end

# Branchless explicit tuple setters (NO capturing ntuple closures — those force
# GPUCompiler dynamic dispatch; the whole kernel failed to compile with them).
@inline _set9i(t::NTuple{9,Int}, k::Int, v::Int) = (
    k==1 ? v : t[1], k==2 ? v : t[2], k==3 ? v : t[3],
    k==4 ? v : t[4], k==5 ? v : t[5], k==6 ? v : t[6],
    k==7 ? v : t[7], k==8 ? v : t[8], k==9 ? v : t[9])
@inline _set9f2(t::NTuple{9,Float64}, k::Int, v::Float64) = (
    k==1 ? v : t[1], k==2 ? v : t[2], k==3 ? v : t[3],
    k==4 ? v : t[4], k==5 ? v : t[5], k==6 ? v : t[6],
    k==7 ? v : t[7], k==8 ? v : t[8], k==9 ? v : t[9])

@inline function _apply_fit_z(gx::NTuple{9,Float64}, si::NTuple{9,Int}, sj::NTuple{9,Int},
                              nsel::Int, x, y)
    v = 0.0
    for m in 1:9
        (m > nsel) && break
        v += gx[m] * _mono2(si[m], sj[m], x, y)
    end
    return v
end

# ===========================================================================
# Central-moment expansions. We need, per-density, the normalized central moments
# cm(i,j,k) = _chyq_central(Mraw,i,j,k,bu,bv,bw)/rho. We compute them directly
# from the 35-tuple via mget with binomial coefficients baked in as literals.
# We only need the specific set the algorithm touches (enumerated below).
# ===========================================================================

# General central-moment via binomial expansion, but with i,j,k small compile-time
# literals so the triple product unrolls. Returns UN-normalized central moment.
@inline function central(M::NTuple{35,Float64}, i::Int, j::Int, k::Int,
                         bu::Float64, bv::Float64, bw::Float64)
    s = 0.0
    for a in 0:i, b in 0:j, d in 0:k
        cij = _binom(i,a) * _binom(j,b) * _binom(k,d)
        s += cij * _pw(-bu, i-a) * _pw(-bv, j-b) * _pw(-bw, k-d) * mget(M, a, b, d)
    end
    return s
end

@inline function _binom(n::Int, r::Int)
    # small binomials for n<=4
    if r==0 || r==n; return 1.0 end
    if n==2; return 2.0 end
    if n==3; return r==1 ? 3.0 : 3.0 end
    if n==4; return r==1 ? 4.0 : (r==2 ? 6.0 : 4.0) end
    return 1.0
end
@inline function _pw(x::Float64, e::Int)
    if e==0; return 1.0
    elseif e==1; return x
    elseif e==2; return x*x
    elseif e==3; return x*x*x
    else; return x*x*x*x end
end

# ===========================================================================
# MAIN device entry: chyqmom_nodes_3d_dev(M::NTuple{35,Float64})
# Returns (n::NTuple{27}, Ux::NTuple{27}, Uy::NTuple{27}, Uz::NTuple{27}, Nn::Int).
# Node weights are scaled back to physical density; slots > Nn are 0.
# ===========================================================================
@noinline function chyqmom_nodes_3d_dev(M::NTuple{35,Float64})
    Z27 = ntuple(_->0.0, Val(27))
    rho = M[1]
    if !(rho > 0.0)
        return (Z27, Z27, Z27, Z27, 0)
    end
    bu = M[2] / rho
    bv = M[6] / rho
    bw = M[16] / rho

    @inline cm(i,j,k) = central(M, i, j, k, bu, bv, bw) / rho

    # ---------------- x level ----------------
    mx0 = 1.0
    mx1 = bu
    mx2 = M[3]/rho
    mx3 = M[4]/rho
    mx4 = M[5]/rho
    (wx1,wx2,wx3, Ux1,Ux2,Ux3, Nx) = hyqmom_1d_dev(mx0, mx1, mx2, mx3, mx4)
    # centered x-abscissas
    upx1 = Ux1 - bu; upx2 = Ux2 - bu; upx3 = Ux3 - bu

    # ---------------- y | x ----------------
    # mean targets (e=0,1,2): (0, c110, c210); var targets: (c020, c120, c220)
    c110 = cm(1,1,0); c210 = cm(2,1,0)
    c020 = cm(0,2,0); c120 = cm(1,2,0); c220 = cm(2,2,0)
    c030 = cm(0,3,0); c040 = cm(0,4,0)
    (Vf1,Vf2,Vf3) = gram_fit_y(wx1,wx2,wx3, upx1,upx2,upx3, Nx, 0.0, c110, c210)
    (sy1,sy2,sy3) = gram_fit_y(wx1,wx2,wx3, upx1,upx2,upx3, Nx, c020, c120, c220)
    vary1 = sy1 - Vf1*Vf1; vary1 = vary1 < 0.0 ? 0.0 : vary1
    vary2 = sy2 - Vf2*Vf2; vary2 = vary2 < 0.0 ? 0.0 : vary2
    vary3 = sy3 - Vf3*Vf3; vary3 = vary3 < 0.0 ? 0.0 : vary3
    # shared skewness/kurtosis (Fox closure)
    (qY, etaY) = _shared_shape(wx1,wx2,wx3, Vf1,Vf2,Vf3, vary1,vary2,vary3, Nx, c030, c040)

    # build (x,y) parents (up to 9). Carry as length-9 NTuples.
    # We accumulate into locals then pack.
    # Parent arrays:
    pwxy = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    pxx  = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)   # Uxy_x (physical, uncentered)
    pxy  = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)   # Uxy_y
    Npxy = 0
    # unroll over the (up to) 3 x-nodes
    for a in 1:3
        (a > Nx) && break
        wxa = a==1 ? wx1 : (a==2 ? wx2 : wx3)
        (wxa > 0.0) || continue
        Uxa = a==1 ? Ux1 : (a==2 ? Ux2 : Ux3)
        Vfa = a==1 ? Vf1 : (a==2 ? Vf2 : Vf3)
        vya = a==1 ? vary1 : (a==2 ? vary2 : vary3)
        muy = bv + Vfa
        (my0,my1,my2,my3,my4) = shape_moments_dev(wxa, muy, vya, qY, etaY)
        (wsy1,wsy2,wsy3, uy1,uy2,uy3, Ny) = hyqmom_1d_dev(my0,my1,my2,my3,my4)
        for b in 1:3
            (b > Ny) && break
            wsb = b==1 ? wsy1 : (b==2 ? wsy2 : wsy3)
            uyb = b==1 ? uy1 : (b==2 ? uy2 : uy3)
            Npxy += 1
            pwxy = _set9f(pwxy, Npxy, wsb)
            pxx  = _set9f(pxx,  Npxy, Uxa)
            pxy  = _set9f(pxy,  Npxy, uyb)
        end
    end

    if Npxy == 0
        return (Z27, Z27, Z27, Z27, 0)
    end

    # ---------------- z | x,y ----------------
    # centered (x,y) coords for parents (explicit tuples — no capturing ntuple).
    cxp = (pxx[1]-bu, pxx[2]-bu, pxx[3]-bu, pxx[4]-bu, pxx[5]-bu,
           pxx[6]-bu, pxx[7]-bu, pxx[8]-bu, pxx[9]-bu)
    cyp = (pxy[1]-bv, pxy[2]-bv, pxy[3]-bv, pxy[4]-bv, pxy[5]-bv,
           pxy[6]-bv, pxy[7]-bv, pxy[8]-bv, pxy[9]-bv)
    # z-mean staircase targets c_{ij1}, i+j<=3 (10 candidates). The CPU
    # (_gram_fit) builds targets as `for i in 0:3, j in 0:(3-i)` then STABLE-SORTS
    # by total degree, giving the exact order:
    #   (0,0) | (0,1),(1,0) | (0,2),(1,1),(2,0) | (0,3),(1,2),(2,1),(3,0)
    # We MUST match this order — the incremental condition-gated selection is
    # order-dependent (within a degree, the first-tried near-collinear column wins
    # or loses the slot), so a permuted order drifts the admitted moment set.
    mei = (0, 0,1, 0,1,2, 0,1,2,3)
    mej = (0, 1,0, 2,1,0, 3,2,1,0)
    mtv = (cm(0,0,1),
           cm(0,1,1), cm(1,0,1),
           cm(0,2,1), cm(1,1,1), cm(2,0,1),
           cm(0,3,1), cm(1,2,1), cm(2,1,1), cm(3,0,1))
    Wf = gram_fit_z(pwxy, cxp, cyp, Npxy, mei, mej, mtv, 10)
    # z-var staircase targets c_{ij2}, i+j<=2 (6 candidates), same CPU stable-sort
    # order: (0,0) | (0,1),(1,0) | (0,2),(1,1),(2,0)
    vei = (0, 0,1, 0,1,2, 0,0,0,0)
    vej = (0, 1,0, 2,1,0, 0,0,0,0)
    vtv = (cm(0,0,2),
           cm(0,1,2), cm(1,0,2),
           cm(0,2,2), cm(1,1,2), cm(2,0,2),
           0.0,0.0,0.0,0.0)
    Sz = gram_fit_z(pwxy, cxp, cyp, Npxy, vei, vej, vtv, 6)
    # per-parent variance and shared shape
    c003 = cm(0,0,3); c004 = cm(0,0,4)
    # variances (explicit — no capturing ntuple)
    vz1 = Sz[1]-Wf[1]*Wf[1]; vz1 = vz1<0.0 ? 0.0 : vz1
    vz2 = Sz[2]-Wf[2]*Wf[2]; vz2 = vz2<0.0 ? 0.0 : vz2
    vz3 = Sz[3]-Wf[3]*Wf[3]; vz3 = vz3<0.0 ? 0.0 : vz3
    vz4 = Sz[4]-Wf[4]*Wf[4]; vz4 = vz4<0.0 ? 0.0 : vz4
    vz5 = Sz[5]-Wf[5]*Wf[5]; vz5 = vz5<0.0 ? 0.0 : vz5
    vz6 = Sz[6]-Wf[6]*Wf[6]; vz6 = vz6<0.0 ? 0.0 : vz6
    vz7 = Sz[7]-Wf[7]*Wf[7]; vz7 = vz7<0.0 ? 0.0 : vz7
    vz8 = Sz[8]-Wf[8]*Wf[8]; vz8 = vz8<0.0 ? 0.0 : vz8
    vz9 = Sz[9]-Wf[9]*Wf[9]; vz9 = vz9<0.0 ? 0.0 : vz9
    varz = (vz1,vz2,vz3,vz4,vz5,vz6,vz7,vz8,vz9)
    (qZ, etaZ) = _shared_shape9(pwxy, Wf, varz, Npxy, c003, c004)

    # ---------------- build final nodes ----------------
    nn  = ntuple(_->0.0, Val(27))
    nux = ntuple(_->0.0, Val(27))
    nuy = ntuple(_->0.0, Val(27))
    nuz = ntuple(_->0.0, Val(27))
    Nn = 0
    for p in 1:9
        (p > Npxy) && break
        wp = pwxy[p]
        (wp > 0.0) || continue
        muz = bw + Wf[p]
        t2 = varz[p]
        (mz0,mz1,mz2,mz3,mz4) = shape_moments_dev(wp, muz, t2, qZ, etaZ)
        (wsz1,wsz2,wsz3, uz1,uz2,uz3, Nz) = hyqmom_1d_dev(mz0,mz1,mz2,mz3,mz4)
        pxxp = pxx[p]; pxyp = pxy[p]
        for c in 1:3
            (c > Nz) && break
            wsc = c==1 ? wsz1 : (c==2 ? wsz2 : wsz3)
            uzc = c==1 ? uz1 : (c==2 ? uz2 : uz3)
            Nn += 1
            nn  = _set27f(nn,  Nn, wsc * rho)   # scale back to physical density
            nux = _set27f(nux, Nn, pxxp)
            nuy = _set27f(nuy, Nn, pxyp)
            nuz = _set27f(nuz, Nn, uzc)
        end
    end

    return (nn, nux, nuy, nuz, Nn)
end

# shared skewness/kurtosis over the (up to 3) parent nodes (Fox closure), 1D.
@inline function _shared_shape(pw1,pw2,pw3, Wf1,Wf2,Wf3, v1,v2,v3, Np::Int,
                               cd3, cd4)
    s1 = sqrt(v1); s2 = sqrt(v2); s3 = sqrt(v3)
    w1 = Np>=1 ? pw1 : 0.0; w2 = Np>=2 ? pw2 : 0.0; w3 = Np>=3 ? pw3 : 0.0
    den3 = w1*s1*s1*s1 + w2*s2*s2*s2 + w3*s3*s3*s3
    num3 = cd3 - (w1*(Wf1*Wf1*Wf1 + 3.0*Wf1*v1) +
                  w2*(Wf2*Wf2*Wf2 + 3.0*Wf2*v2) +
                  w3*(Wf3*Wf3*Wf3 + 3.0*Wf3*v3))
    q = abs(den3) > 1.0e-14 ? num3/den3 : 0.0
    den4 = w1*v1*v1 + w2*v2*v2 + w3*v3*v3
    num4 = cd4 - (w1*(Wf1*Wf1*Wf1*Wf1 + 6.0*Wf1*Wf1*v1 + 4.0*Wf1*s1*s1*s1*q) +
                  w2*(Wf2*Wf2*Wf2*Wf2 + 6.0*Wf2*Wf2*v2 + 4.0*Wf2*s2*s2*s2*q) +
                  w3*(Wf3*Wf3*Wf3*Wf3 + 6.0*Wf3*Wf3*v3 + 4.0*Wf3*s3*s3*s3*q))
    eta = abs(den4) > 1.0e-14 ? num4/den4 : (q*q + 1.0)
    return (q, eta)
end

# shared skewness/kurtosis over up to 9 parents.
@inline function _shared_shape9(pw::NTuple{9,Float64}, Wf::NTuple{9,Float64},
                                varz::NTuple{9,Float64}, Np::Int, cd3, cd4)
    den3 = 0.0; num3acc = 0.0
    for p in 1:9
        (p > Np) && break
        w = pw[p]; v = varz[p]; s = sqrt(v); wf = Wf[p]
        den3 += w*s*s*s
        num3acc += w*(wf*wf*wf + 3.0*wf*v)
    end
    num3 = cd3 - num3acc
    q = abs(den3) > 1.0e-14 ? num3/den3 : 0.0
    den4 = 0.0; num4acc = 0.0
    for p in 1:9
        (p > Np) && break
        w = pw[p]; v = varz[p]; s = sqrt(v); wf = Wf[p]
        den4 += w*v*v
        num4acc += w*(wf*wf*wf*wf + 6.0*wf*wf*v + 4.0*wf*s*s*s*q)
    end
    num4 = cd4 - num4acc
    eta = abs(den4) > 1.0e-14 ? num4/den4 : (q*q + 1.0)
    return (q, eta)
end

@inline _set9f(t::NTuple{9,Float64}, k::Int, v::Float64) = (
    k==1 ? v : t[1], k==2 ? v : t[2], k==3 ? v : t[3],
    k==4 ? v : t[4], k==5 ? v : t[5], k==6 ? v : t[6],
    k==7 ? v : t[7], k==8 ? v : t[8], k==9 ? v : t[9])

# ===========================================================================
# SPLIT-KERNEL PHASE-1 variant: invert-and-STORE. Identical math to
# chyqmom_nodes_3d_dev up to the z-shape parameters, but the final ≤27 nodes are
# written DIRECTLY to per-cell global storage as they are produced — no 27-slot
# NTuple accumulators (nn/nux/nuy/nuz) held live in registers. This is the design
# §1.5 storage split: it removes the 27×4 output-buffer liveness from the
# inversion's hot region (they become straight stores), which is one of the two
# suspected register-pressure sources. Returns Nn; writes nodes via `store4!`.
#
# `store4!(NW,UX,UY,UZ, ci, q, w, ux, uy, uz)` is a caller-supplied @inline that
# writes node q of cell ci (host callers pass a closure that indexes the global
# arrays). Everything else matches the fused entry EXACTLY (same helpers, same
# faithful gate) so the stored quadrature is byte-identical to the fused output.
@noinline function chyqmom_nodes_3d_store_dev!(store4!::F, NW, UX, UY, UZ, ci::Int,
                                               M::NTuple{35,Float64}) where {F}
    rho = M[1]
    if !(rho > 0.0); return 0; end
    bu = M[2]/rho; bv = M[6]/rho; bw = M[16]/rho
    @inline cm(i,j,k) = central(M, i, j, k, bu, bv, bw) / rho

    mx0=1.0; mx1=bu; mx2=M[3]/rho; mx3=M[4]/rho; mx4=M[5]/rho
    (wx1,wx2,wx3, Ux1,Ux2,Ux3, Nx) = hyqmom_1d_dev(mx0,mx1,mx2,mx3,mx4)
    upx1=Ux1-bu; upx2=Ux2-bu; upx3=Ux3-bu

    c110=cm(1,1,0); c210=cm(2,1,0); c020=cm(0,2,0); c120=cm(1,2,0); c220=cm(2,2,0)
    c030=cm(0,3,0); c040=cm(0,4,0)
    (Vf1,Vf2,Vf3)=gram_fit_y(wx1,wx2,wx3, upx1,upx2,upx3, Nx, 0.0, c110, c210)
    (sy1,sy2,sy3)=gram_fit_y(wx1,wx2,wx3, upx1,upx2,upx3, Nx, c020, c120, c220)
    vary1=sy1-Vf1*Vf1; vary1=vary1<0.0 ? 0.0 : vary1
    vary2=sy2-Vf2*Vf2; vary2=vary2<0.0 ? 0.0 : vary2
    vary3=sy3-Vf3*Vf3; vary3=vary3<0.0 ? 0.0 : vary3
    (qY,etaY)=_shared_shape(wx1,wx2,wx3, Vf1,Vf2,Vf3, vary1,vary2,vary3, Nx, c030, c040)

    pwxy=(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    pxx =(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    pxy =(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    Npxy=0
    for a in 1:3
        (a > Nx) && break
        wxa = a==1 ? wx1 : (a==2 ? wx2 : wx3)
        (wxa > 0.0) || continue
        Uxa = a==1 ? Ux1 : (a==2 ? Ux2 : Ux3)
        Vfa = a==1 ? Vf1 : (a==2 ? Vf2 : Vf3)
        vya = a==1 ? vary1 : (a==2 ? vary2 : vary3)
        muy = bv + Vfa
        (my0,my1,my2,my3,my4)=shape_moments_dev(wxa, muy, vya, qY, etaY)
        (wsy1,wsy2,wsy3, uy1,uy2,uy3, Ny)=hyqmom_1d_dev(my0,my1,my2,my3,my4)
        for b in 1:3
            (b > Ny) && break
            wsb = b==1 ? wsy1 : (b==2 ? wsy2 : wsy3)
            uyb = b==1 ? uy1 : (b==2 ? uy2 : uy3)
            Npxy += 1
            pwxy=_set9f(pwxy,Npxy,wsb); pxx=_set9f(pxx,Npxy,Uxa); pxy=_set9f(pxy,Npxy,uyb)
        end
    end
    if Npxy == 0; return 0; end

    cxp=(pxx[1]-bu,pxx[2]-bu,pxx[3]-bu,pxx[4]-bu,pxx[5]-bu,pxx[6]-bu,pxx[7]-bu,pxx[8]-bu,pxx[9]-bu)
    cyp=(pxy[1]-bv,pxy[2]-bv,pxy[3]-bv,pxy[4]-bv,pxy[5]-bv,pxy[6]-bv,pxy[7]-bv,pxy[8]-bv,pxy[9]-bv)
    mei=(0, 0,1, 0,1,2, 0,1,2,3); mej=(0, 1,0, 2,1,0, 3,2,1,0)
    mtv=(cm(0,0,1), cm(0,1,1),cm(1,0,1), cm(0,2,1),cm(1,1,1),cm(2,0,1),
         cm(0,3,1),cm(1,2,1),cm(2,1,1),cm(3,0,1))
    Wf=gram_fit_z(pwxy, cxp, cyp, Npxy, mei, mej, mtv, 10)
    vei=(0, 0,1, 0,1,2, 0,0,0,0); vej=(0, 1,0, 2,1,0, 0,0,0,0)
    vtv=(cm(0,0,2), cm(0,1,2),cm(1,0,2), cm(0,2,2),cm(1,1,2),cm(2,0,2), 0.0,0.0,0.0,0.0)
    Sz=gram_fit_z(pwxy, cxp, cyp, Npxy, vei, vej, vtv, 6)
    c003=cm(0,0,3); c004=cm(0,0,4)
    vz1=Sz[1]-Wf[1]*Wf[1]; vz1=vz1<0.0 ? 0.0 : vz1
    vz2=Sz[2]-Wf[2]*Wf[2]; vz2=vz2<0.0 ? 0.0 : vz2
    vz3=Sz[3]-Wf[3]*Wf[3]; vz3=vz3<0.0 ? 0.0 : vz3
    vz4=Sz[4]-Wf[4]*Wf[4]; vz4=vz4<0.0 ? 0.0 : vz4
    vz5=Sz[5]-Wf[5]*Wf[5]; vz5=vz5<0.0 ? 0.0 : vz5
    vz6=Sz[6]-Wf[6]*Wf[6]; vz6=vz6<0.0 ? 0.0 : vz6
    vz7=Sz[7]-Wf[7]*Wf[7]; vz7=vz7<0.0 ? 0.0 : vz7
    vz8=Sz[8]-Wf[8]*Wf[8]; vz8=vz8<0.0 ? 0.0 : vz8
    vz9=Sz[9]-Wf[9]*Wf[9]; vz9=vz9<0.0 ? 0.0 : vz9
    varz=(vz1,vz2,vz3,vz4,vz5,vz6,vz7,vz8,vz9)
    (qZ,etaZ)=_shared_shape9(pwxy, Wf, varz, Npxy, c003, c004)

    Nn = 0
    for p in 1:9
        (p > Npxy) && break
        wp = pwxy[p]
        (wp > 0.0) || continue
        muz = bw + Wf[p]
        t2 = varz[p]
        (mz0,mz1,mz2,mz3,mz4)=shape_moments_dev(wp, muz, t2, qZ, etaZ)
        (wsz1,wsz2,wsz3, uz1,uz2,uz3, Nz)=hyqmom_1d_dev(mz0,mz1,mz2,mz3,mz4)
        pxxp = pxx[p]; pxyp = pxy[p]
        for c in 1:3
            (c > Nz) && break
            wsc = c==1 ? wsz1 : (c==2 ? wsz2 : wsz3)
            uzc = c==1 ? uz1 : (c==2 ? uz2 : uz3)
            Nn += 1
            store4!(NW, UX, UY, UZ, ci, Nn, wsc*rho, pxxp, pxyp, uzc)  # direct global store
        end
    end
    return Nn
end
@inline _set27f(t::NTuple{27,Float64}, k::Int, v::Float64) = (
    k==1 ? v : t[1], k==2 ? v : t[2], k==3 ? v : t[3], k==4 ? v : t[4],
    k==5 ? v : t[5], k==6 ? v : t[6], k==7 ? v : t[7], k==8 ? v : t[8],
    k==9 ? v : t[9], k==10 ? v : t[10], k==11 ? v : t[11], k==12 ? v : t[12],
    k==13 ? v : t[13], k==14 ? v : t[14], k==15 ? v : t[15], k==16 ? v : t[16],
    k==17 ? v : t[17], k==18 ? v : t[18], k==19 ? v : t[19], k==20 ? v : t[20],
    k==21 ? v : t[21], k==22 ? v : t[22], k==23 ? v : t[23], k==24 ? v : t[24],
    k==25 ? v : t[25], k==26 ? v : t[26], k==27 ? v : t[27])

end # module

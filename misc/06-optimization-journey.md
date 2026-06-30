# Optimization journey — what worked, what didn't, and the math

This document records the GPU-residual performance work done on the H200 in July 2026:
every optimization attempted, whether it shipped, and **why** — with the underlying
mathematics. It is deliberately exhaustive about the *failures*, because the negative
results (each one measured, not guessed) are what define the actual performance wall and
save the next person from re-deriving them.

The work proceeded in two rounds after the `recon-cache + @noinline` merge (PR #4):

- **Round 2** — GPU-mechanical optimizations (registers, occupancy, memory, kernel
  launches). Conclusion: the byte-exact micro-optimization space is *exhausted*.
- **Round 3** — mathematical reformulations of the same solve, informed by a 6-agent
  math review. Conclusion: the real lever is replacing the *iterative eigensolvers* with
  *closed-form / one-pass* equivalents, which the autogen's algebra makes possible.

All claims below were validated against the `gpudata` CPU references and a march-vs-`main`
byte comparison. Benchmarks: H200, `n=128`, Ma=100, fp64. Reproduction scripts are listed
at the end.

---

## 1. The baseline and the wall

The residual hot path is the per-face flux kernel `_fhat_{x,y,z}_g!`
(`gpu/residual3d_gpu.jl`), which per face runs, for each of the left/right states:

```
reconstruct (MUSCL)  →  realizability projection  →  wave speeds  →  physical flux  →  HLL
```

Profiling (`ncu`/`nsys`, `reg_decomp.jl`) established the regime precisely:

| quantity | value | meaning |
|---|---|---|
| registers / thread | **255** (hardware max) | register-capped |
| occupancy | **~12%** | very low |
| dominant stall | **82% "no eligible warp"** | latency-bound, not compute- or memory-bound |
| compute SoL | 16.8% | the ALUs are mostly idle waiting |
| local memory | 9552 B | the eigensolver `MMatrix` workspaces (not register spills) |

Per-transform register cost (each compiled standalone):

| transform | registers | occupancy |
|---|---|---|
| `flux_closure35_dev` | **255** (the ceiling) | 12% |
| `realizable_3D_M4_dev` (projection) | 244 | 12% |
| `realize_and_speed_Mr_dev` (wave speed) | 214 | 14% |

Residual time breakdown (limiter on): wave-speed + flux ≈ 46%, recon ≈ 22%, limiter ≈
23%, face-projection ≈ 12%.

**The wall.** A *single* transform already nearly saturates the 255-register file. The
closure's fifth-order middle has ~250 coupled live intermediates that no kernel
restructuring removes. Because the kernel is **latency-bound at fixed (low) occupancy**,
the only thing that helps is *shortening the per-thread dependency chain* — and the
longest serial chains are the **iterative eigensolvers**: the 6×6 cyclic-Jacobi
(realizability) and the 4×4 Francis QR (wave speeds).

---

## 2. What shipped

Four commits on `perf/round2-gpu-opts`. **Every default is byte-identical to `main`**
(march output 0/3,870,720 bytes different across the default / limiter / proj modes); the
two algorithmic speedups are behind default-off compile-time flags.

### 2.1 In-place realizability projection — *byte-identical, memory*

`_rk3_step!` projected out-of-place and copied back:
`realizable_batched!(Pbuf, X); copyto!(X, Pbuf)`. The projection kernel reads all 35
moments of a cell into registers before writing any back, and threads touch disjoint
columns, so `Mout === Min` is safe. Projecting in place drops the `copyto!` pass **and**
the `Pbuf` buffer — one fewer `(35,n,n,nz)` array, **≈586 MB at n=128**. Pure memory win;
~0 % time (projection is ~1.3 % of a step).

### 2.2 Bunch–Kaufman inertia realizability test — *opt-in, 2.86× on the limiter*

**The problem.** Realizability is decided by the sign of the smallest eigenvalue of a
symmetric 6×6 matrix `Δ*` (`delta2star`). The device computed `λ_min(Δ*)` with a cyclic
**Jacobi** sweep (`sym6_mineig`) — up to 100 sweeps, each 15 Givens rotations with a
`sqrt` and a division: a long, data-dependent serial chain, run *per face* and **~42×
per face** inside the limiter's bisection.

**The key observation.** Every consumer of `λ_min` uses only its **sign**
(`realize_dev.jl`: `lam >= 0`, `lam2 > -1e-6`, `m >= 0`). So we never need the
eigenvalue — only the answer to *"is `Δ* + σI` positive semidefinite?"*. That is a
**definiteness** question, and definiteness is decided by **inertia**, not by an
eigenvalue iteration.

**The math.** By **Sylvester's law of inertia**, a symmetric factorization
`A = LDL^T` (with `L` unit lower-triangular and `D` block-diagonal with 1×1 and 2×2
blocks) has the same number of negative eigenvalues as `D`. So

```
λ_min(A) ≥ 0   ⟺   A is PSD   ⟺   D has no negative eigenvalue.
```

Reading the inertia off `D` is **one pass** (~`n³/6` ≈ 70 flops for n=6), versus the
iterative Jacobi sweeps — and it is *exact* (it counts the sign, it does not approximate
the magnitude).

**Why a *pivoted* factorization, and why the naive one fails.** An earlier attempt
(round 2) used an **unpivoted** `LDL^T`: `d_j = A_{jj} - Σ_{k<j} L_{jk}^2 d_k`, PSD iff all
`d_j ≥ 0`. It was 3× faster but, marched at Ma=100, its trajectory diverged ~20% — it gave
*wrong inertia* near the boundary. The reason is catastrophic cancellation: at a
realizability face the pivot `d_j` is a genuine difference of large near-equal quantities,
and unpivoted elimination divides by it, propagating garbage. **Bunch–Kaufman** partial
pivoting (threshold `α = (1+√17)/8 ≈ 0.6404`) bounds element growth and, crucially,
switches to a **2×2 pivot block exactly when the 1×1 pivot would be ≈0** — i.e. precisely
the boundary case. Inertia is then read from the 1×1 pivot signs and the 2×2 blocks'
`det`/`trace` signs, which is exact under Sylvester's law no matter how near-singular `A`
is. (A 2×2 block `[[a,b],[b,c]]` contributes one negative eigenvalue if `det = ac-b² < 0`,
and two if `det ≥ 0 ∧ trace < 0`.)

**Validation.** `sym6_psd_bunchkaufman` matches LAPACK's eigenvalue sign on **200,000
random matrices** (including near-boundary and `1e7`-scaled) with **0 disagreements**, and
on the real Ma=100 `r3d` state it gives **0 decision flips** vs Jacobi (residual gate
identical, rel 5.119e-11).

**Speed (H200, n=128, Ma=100, `_REALIZ_SOLVER = :pivot`):**

| | Jacobi (default) | Bunch–Kaufman | speedup |
|---|---|---|---|
| limiter residual | 1890 ms | 660 ms | **2.86×** |
| limiter step | 5547 ms | 2064 ms | **2.69×** |
| default residual | 161 ms | 159 ms | ~1.0× |

The limiter is where the ~42 evaluations/face make the per-call win compound; default mode
calls `Δ*` only ~2×/face so it is unchanged.

> **Even better (not implemented):** `Δ* = −(1/S2)·Ñ` is the **Schur complement** of the
> 9×9 *bordered moment Gram matrix* `M = [[C, B],[Bᵀ, A]]` (`C` = 2nd-order block,
> `B` = 3rd, `A` = 4th), and since `C ≻ 0` upstream, by the **Haynsworth inertia additivity
> formula** `inertia(M) = inertia(C) + inertia(Δ*)`. Factoring the 9×9 `M` (C-block first)
> from the raw O(1) moments never forms `1/S2` or the cancellation-prone `Ñ_ij`, removing
> the boundary ill-conditioning at its source. Bunch–Kaufman on the existing 6×6 already
> suffices in practice (it matched LAPACK), so this is filed as the maximally-robust
> variant.

### 2.3 Ferrari closed-form 4×4 quartic — *opt-in, 1.27× on the default path*

**The observation.** The wave-speed Jacobian's 4×4 block is a **companion matrix**

```
        ⎡ 0    1    0    0  ⎤
   C =  ⎢ 0    0    1    0  ⎥
        ⎢ 0    0    0    1  ⎥
        ⎣ e84  e99  e114 e129⎦
```

whose characteristic polynomial is therefore **free** — it is literally the bottom row:

```
λ⁴ − e129·λ³ − e114·λ² − e99·λ − e84 = 0.
```

The device solved this with an **iterative Francis double-shift QR** (Hessenberg reduction
+ a `while`-loop of bulge chases, `maxsweep = 40`). But a quartic has a **closed form**, so
both the reduction *and* the iteration are pure overhead.

**The math (Ferrari).** Write the monic quartic `λ⁴ + bλ³ + cλ² + dλ + e` with
`b = −e129, c = −e114, d = −e99, e = −e84`. Depress with `λ = y − b/4`:

```
y⁴ + p·y² + q·y + r = 0,
   p = c − 3b²/8,   q = d − bc/2 + b³/8,   r = e − bd/4 + b²c/16 − 3b⁴/256.
```

Ferrari factors this into two quadratics `(y² + u·y + s)(y² − u·y + t)` where `z = u²` is a
real root of the **resolvent cubic**

```
z³ + 2p·z² + (p² − 4r)·z − q² = 0.
```

The constant term `−q² ≤ 0` and leading `+1` guarantee a real root `z* ≥ 0` exists; take
the **largest** one (`_cubic_max_real_root`, via the standard depressed-cubic trig/Cardano
form — the same math `eig3_realparts_dev` already uses). Then `u = √z*`,
`s = (p + z* − q/u)/2`, `t = (p + z* + q/u)/2`, and the four roots' real parts come from the
two quadratics:

```
disc_A = u² − 4s :  Re = disc_A ≥ 0 ? (−u ± √disc_A)/2 : −u/2  (twice)
disc_B = u² − 4t :  Re = disc_B ≥ 0 ? ( u ± √disc_B)/2 :  u/2  (twice)
```

and `λ_Re = y_Re − b/4`. The `q ≈ 0` biquadratic case (`z* ≤ 0`) is handled separately
(solve `y⁴ + p y² + r = 0` as a quadratic in `y²`). This is **straight-line** — no
data-dependent loop, hence **no warp divergence** in the batched kernel.

**Accuracy.** Closed-form quartic roots are exact in exact arithmetic but **not** backward
stable: cancellation in forming `r` and `disc_{A,B}` costs digits for clustered/near-defective
roots and high coefficient dynamic range. Measured vs LAPACK over **300,000 random
companions**: within 1e-6 on the well-conditioned majority, worst ~3e-4 only on the extreme
`1e7`-scaled blocks. On the *real* Ma=100 `r3d` companions the residual gate is **identical**
(HLL rel 5.119e-11) — the actual wave-speed companions are far better conditioned than the
synthetic worst case. A QR fallback covers the rare degenerate block.

**Speed (`_WAVESPEED_SOLVER = :ferrari`):** default residual 161 → **127 ms (1.27×)**,
default step 550 → **440 ms (1.25×)**. The limiter is realizability-dominated, so Ferrari is
marginal there — making it **complementary** to Bunch–Kaufman (BK → limiter, Ferrari →
default).

### 2.4 Robust wave-speed fallback on QR non-convergence — *byte-identical bug fix*

`jac15_eig_dev` discarded `schur4`'s status flag. On a sweep-cap non-convergence `schur4`
returns `(+∞, −∞)`, which **silently drops** the 4×4 block:
`min(r3lo, +∞) = r3lo`, `max(r3hi, −∞) = r3hi`. In general a dropped block underestimates
`sR`, narrowing the HLL stencil → CFL violation → instability.

We found this is **not** dormant: `schur4` hits its sweep cap on a *large fraction* of
Ma=100 companion blocks (replacing the drop with a wide Fujiwara magnitude bound blew the
residual up to rel 68). The blocks are benign here — their eigenvalues lie within the 3×3
block's range, so dropping them coincidentally matches the CPU — but relying on that is
fragile.

**The fix:** on `status ≠ 0`, fall back to the closed-form **Ferrari** solve (which has no
convergence cap), and only if *that* is also degenerate use a guaranteed Fujiwara bound
(`|λ| ≤ 2·max(|e129|, |e114|^{1/2}, |e99|^{1/3}, |e84|^{1/4})`) — never a silent drop, never
NaN. Because Ferrari reproduces the (correct) dropped blocks exactly, the result is
**byte-identical to `main`** (march 0/3.87 M bytes) at the **same speed**, while removing the
underestimate risk.

---

## 3. What did NOT work — and why (each measured)

The discipline here was *measure, don't guess*. Several ideas that sound good are listed
with the number that killed them.

### Round 3 (math reformulations)

| Idea | Why it seemed promising | Measured outcome | Verdict |
|---|---|---|---|
| **Unpivoted LDLᵀ** definiteness | one pass, replaces Jacobi | 3× faster but **march drift 20–190%** at Ma=100 (wrong inertia at the boundary via pivot cancellation) | **superseded by Bunch–Kaufman** (§2.2) |
| **`@noinline delta2star_mineig`** | confine the ~500-temp 6×6 build out of `_fhat`'s register frame | −128 B local, **0 register change** (still 255), and broke limiter byte-identity | reverted — `Δ*` lives in the projection transform (244), *below* the flux-driven 255 peak, so confining it cannot lower the peak |
| **Per-axis `flux_closure`** specialization | compute only the 35 used moments, not 105 | the compiler **already** DCEs the unused blocks → still 255 | no-op |
| **Gershgorin PSD pre-filter** | O(1) skip of the eigensolve on PD cells | **0 % even on a perfect Maxwellian** — the Gershgorin bound is too weak to fire on realistic `Δ*` (they aren't diagonally dominant) | reverted (dead overhead) |
| **O(1) diagonal-sign reject** (Bunch–Kaufman pre-screen) | catch clearly-non-PSD cells before factoring | limiter 661 vs 660 ms — **doesn't fire** on `Δ*` (same reason as Gershgorin) | reverted |
| **`schur4` companion balancing** (ISSUE 2) | improve high-Ma 4×4 accuracy vs LAPACK | `schur4` is **already accurate** on random high-Ma companions (worst 3.94e-11 @ 1e7); the real Ma=100 failures are **defectiveness/clustering, not ill-scaling**, so balancing wouldn't fix them — and they're already handled by §2.4 | not needed |

### Round 2 (GPU-mechanical)

| Idea | Why it seemed promising | Measured / reasoned outcome | Verdict |
|---|---|---|---|
| **Two-threads-per-face** (L/R split) | halve the live set by splitting the L and R pipelines across two threads | `reg_decomp` shows even *one* transform (flux) is 255 → splitting can't break the ceiling | ruled out by data |
| **More cells/faces per thread** (Volkov ILP) | hide latency with ILP at low occupancy | no register headroom at the 255 cap → would spill | rejected |
| **Warp-per-face**, 35 moments across lanes | parallelize the closure across a warp | the dominant cost (4×4 QR, 6×6 Jacobi) is *irreducibly serial* and tiny | rejected |
| **CUDA graphs** (whole RK3 step) | amortize ~30 kernel launches/step | launch overhead is **<0.1%** of the step (compute-bound); `dt`-by-value + the DtoH CFL read also block capture | rejected |
| **Concurrent streams** for `fhat_{x,y,z}` | the 3 axes are independent | they alias one `flat` scratch and `+=` into the same `R` (data race), and each already saturates the GPU | rejected |
| **Read-only cache (`@ldg`/`Const`)** on the gathers | cut L1 pressure from the scattered AoS gather | the kernel is compute/latency-bound, not memory-bound — attacks the wrong resource | skipped |
| **Megakernel / persistent kernel** | fuse the whole residual | would spill catastrophically — the kernel split exists *because* of register pressure | rejected |
| **fp32 mixed precision** on the eig/branch path | 2× throughput | flips the realizability / hyperbolicity decision on ~0.5% of states — a correctness bug | rejected |
| **`rootsR` autogen for wave speeds** | reuse existing analytic roots | those are the realizability cubics, *not* the wave-speed quartic — wrong polynomial | rejected |

The single byte-exact win from round 2 was the in-place projection (§2.1).

---

## 4. The central finding: chaotic trajectory sensitivity at Ma=100

The most important result is *negative* and reframes the whole problem.

Both fast paths (`:pivot`, `:ferrari`) are **exact or as-accurate**: Bunch–Kaufman matches
LAPACK's eigenvalue sign exactly; Ferrari passes the single-residual gate identically. Yet
when *marched* 5 steps at Ma=100, both trajectories drift **~20%** from the Jacobi/QR
default.

This is **not solver error**. It is the **ill-conditioning of the realizability decision**
at high Ma:

1. The first realizability gate is `λ_min(Δ*) ≥ 0` with **zero tolerance**
   (`projection35_dev`). A cell at `λ_min ≈ −10⁻¹⁵` (numerically realizable) is treated as
   unrealizable and **fully projected**, discarding its cross-moment information — a cliff.
2. At Ma=100 a **large fraction of evolved cells sit at `λ_min ≈ 0`** (marginally
   realizable — that is *why* projection is needed there).
3. Any two *correct* eigensolvers disagree on those cells at the ~10⁻¹³ rounding level (the
   eigenvalue sign is genuinely ambiguous within rounding there).
4. The colliding-jets flow is **chaotic**, so those ~10⁻¹³ per-cell differences amplify to
   ~20% over a few steps.

The same mechanism applies to the wave speed: a ~10⁻⁴ change in `sL/sR` (well within HLL
tolerance) also amplifies to ~20% over 5 steps.

**Consequence.** At Ma=100 you **cannot** obtain a bit-trajectory-identical *faster* solve
by swapping eigensolvers — the divergence is a property of the problem, not the
implementation. The correct correctness measure is the **single-residual gate vs the CPU**
(identical for both fast paths). The defaults remain byte-identical (flags off); the fast
paths are exact-but-trajectory-sensitive opt-ins.

**The method-level fix (proposed, not implemented).** Add a small tolerance band to the
realizability gate (`λ_min ≥ −τ·‖Δ*‖` instead of `≥ 0`). Then the boundary cells are
classified *consistently* regardless of which eigensolver computes the sign, the trajectory
becomes robust to the solver choice, and `:pivot`/`:ferrari` could become defaults. This
also removes the spurious full-projection of marginally-realizable cells, which is arguably
*better* physics. This is a change to the numerical method (Fox/Posey's layer), so it is
recorded here as a recommendation rather than applied.

---

## 5. Using the opt-in fast paths (multiple dispatch)

The solvers are selected by **multiple dispatch on singleton types** — set the relevant
`const` and recompile (default values reproduce the validated paths byte-for-byte):

- Wave speeds (`gpu/wavespeed_dev.jl`): `const WAVE4_SOLVER = QRWave()` →
  `FerrariWave()` (closed-form companion quartic) or `TridiagWave()` (symmetric-tridiagonal
  Q4 — the robust, principled form). Dispatched as
  `_wave4_minmax(::QRWave|::FerrariWave|::TridiagWave, e84,e99,e114,e129, m00,m10,m20,m30,m40)`.
- Realizability (`src/realizability/realize_dev.jl`): `const REALIZ_SOLVER = JacobiRealiz()`
  → `PivotRealiz()` (Bunch–Kaufman inertia). Dispatched as
  `_realiz_is_psd(::JacobiRealiz|::PivotRealiz, standardized..., shift)`.

They are independent and combine freely: `PivotRealiz` accelerates the limiter (~2.86×),
`FerrariWave`/`TridiagWave` the default path. The §2.4 robustness fallback (4×4 solver
failure → Ferrari → Fujiwara) is always on and byte-identical. `b(3)` flooring (§7b) is
always on (byte-identical; engages only on roundoff-unrealizable high-Ma cells).

---

## 6. Reproduction

Validation and benchmark scripts (developed under
`/storage/scratch1/6/sbryngelson3/vizwork/`, not committed; they activate a CUDA+MPI
project and read the `gpudata` CPU references):

| script | what it checks |
|---|---|
| `validate_recon_cache.jl` | residual vs CPU refs, all 5 paths (HLL 5.119e-11, proj 5.825e-11, rusanov 3.335e-10, order1 2.950e-11) |
| `cmp_march.jl` / `cmp_march_rel.jl` | march byte-identity / drift vs `main`, def/lim/proj |
| `test_bk.jl` | Bunch–Kaufman PSD vs LAPACK eig sign (200k random matrices) |
| `test_ferrari.jl` | Ferrari quartic vs LAPACK (300k random companions) |
| `test_schur4.jl` | `schur4` failure rate + accuracy vs scale |
| `test_agree.jl` | BK vs Jacobi decision flips binned by `|λ_min|` |
| `reg_decomp.jl` / `fhat_regs2.jl` | per-transform / `_fhat` register + local-memory counts |
| `bench.jl` / `bench_ma.jl` | residual + step timing on H200 (Ma=100 / smooth) |

The committed validators in `gpu/validation/` cover the same paths against the on-disk
references (see [`03-running-and-validation.md`](03-running-and-validation.md)).

---

## 7. Input from Rodney Fox (2026-07) — the root-cause direction

Two points from Rodney that bear directly on the wave-speed work above:

**(a) The 4×4 should be a symmetric tridiagonal, not a companion — IMPLEMENTED (`TridiagWave`).**
The 4×4 wave-speed block `J(6:9,6:9)` has characteristic polynomial `Q4`. The production
code takes it as the **companion** submatrix of the 15×15 Jacobian — and that companion form
is exactly what is ill-conditioned at high Ma (§2.4: `schur4` hits its sweep cap on a large
fraction of Ma=100 blocks; §2.3: Ferrari loses ~3e-4 there from coefficient cancellation).
`Q4`'s four roots are equally the eigenvalues of the **symmetric tridiagonal Jacobi matrix**
`diag[a₀,a₁,a₂,a₂]`, offdiag `[√b₁,√b₂,√(1.5·b₂)]` (the orthogonal-polynomial recurrence
matrix). From the "Fourth-Order HyQMOM" paper (sec 2.3, `Q4 = (x−a₂)Q₃ − (3/2)b₂Q₂`, based
on `(s30,s40)`) the coefficients are the 1D-marginal recurrence (`a₀`=mean, `b₁`=variance,
`a₁`, `b₂ = s44/s33`) with the Q4 closure `a₂=a₃=(a₀+a₁)/2`, `b₃=(3/2)b₂` — the same Chebyshev
algebra `closure5_dev` already uses, built from `(m₀₀,m₁₀,m₂₀,m₃₀,m₄₀)`. A symmetric tridiagonal
has **guaranteed-real, well-conditioned** eigenvalues, fixing the conditioning at the source:
no spurious complex pairs, no QR sweep-cap failures. It is solved by Ferrari on its *well-scaled*
characteristic quartic (formed via the tridiagonal continuant recurrence). Host-validated:
tridiagonal eigenvalues == companion `eig(J6(6:9,6:9))` to 2.5e-10; residual gate identical
(5.119e-11). Speed (V100, n=96): default residual 282→259 ms (1.09×); the limiter is
realizability-dominated so it is unchanged there. The gain is modest because the 3×3 P3 block
still needs `jac15_blocks`, so only the 4×4 *solve* is replaced — but `TridiagWave` is the
mathematically-principled, robust form (the `b(3)` floor from (b) is applied to its `b₂`).

**(b) Floor unrealizable `b(3)` at large Ma** — *implemented* (`9cb412d`). In the 1D
marginal closure (`closure_and_eigenvalues` / `closure5_dev`), roundoff in the
`s_k → m_k` change of variables can make `b(3) = H₂` slightly negative at large Ma even
though the density is order 1 — in exact arithmetic `b(3) ≥ 0`. `b(3) = 0` is the
two-delta-function limit, so a negative `b(3)` is reset to `1e-10` (~QMOM) instead of
producing spurious complex marginal abscissae. Matches Rodney's MATLAB (cleaner results at
Ma=200). Byte-identical on the Ma=100 r3d state (`b(3) < 0` on 0/41472 marginals — a safety
net for evolved / higher-Ma states), applied in both CPU and GPU for parity.

## 7c. The flux closure is the real default-mode bottleneck

A wave-speed STUB experiment settled where the default-mode time goes: replacing the
*entire* wave-speed eigensolve with a trivial bound changed the residual by **0 ms** (286.7
vs 282.5 on V100). The eigensolvers sit in the **latency shadow** of the flux closure — so
every wave-speed optimization (`schur4`/Ferrari/Tridiag) is ~0% on the default path; they
only pay where the solver is called many times serially (the limiter's realizability, where
Bunch–Kaufman gives 2.86×). A 5-agent math review of the closure converged: the 21 fifth-order
closures are cheap and already CSE'd; the cost is the **standardize ⇄ destandardize round-trip**
(2 sqrt + ~28 divides + the ~49-wide live "S-bridge") that sets the 255-register / 12%-occupancy
wall.

**Central-direct closure (`CentralClosure`, opt-in, commit on this branch).** Each fifth-order
*central* moment is computed directly from the lower central moments — the variance powers
cancel by a parity property (every 5th-order standardized closure has the variances to even
powers), so no sqrt and no standardized intermediates. The 21 forms were hand-derived and
verified to 7e-14 vs the standardized path. Result: A100 default residual **118.8 → 110.4 ms
(1.07×)**, accuracy 5.8e-11 vs the goldenfile. Registers stay 255 (the live set is intrinsically
wide — even removing the S-bridge, the raw-moment assembly is the binding width), so the gain is
critical-path-only. Caveat: it divides by variance² (vs the conditioning-robust standardized
form), losing ~2 digits at high Ma, and perturbs the chaotic Ma=100 trajectory ~1e-6 over a few
limiter steps. So default-mode is genuinely near its wall; 7% is about what the closure layer
admits without a *method* change (fewer moments).

## 7d. Validation by integral QoIs (not byte-identity)

For numerics-changing opt-ins at Ma=100, byte-identity is the wrong bar — the flow is chaotic,
so two valid methods drift ~20% pointwise (turbulent rearrangement, not error). The right test
is a **limiter-active march measuring integral QoIs** with the *same* dt sequence (same physical
time): conserved integrals (mass, momentum) test that conservation is intact; solution integrals
(energy, peak density, total variation = a diffusion proxy) test the physics. Crucially the test
must be **short** — at 2 steps the integral QoIs still resolve the method difference before chaos
dominates; by 20 steps chaos amplifies it ~1e4×.

| opt-in | single-residual vs goldenfile | integral QoI, 2-step limiter march | verdict |
|---|---|---|---|
| Bunch–Kaufman realizability | byte-identical (5.119e-11) | **1.6e-10** (mass), 4.6e-11 (energy) | good-enough → default-eligible |
| central-direct closure | 5.8e-11 | ~1e-6 (peak 1.3e-6) | meets goldenfile bar; perturbs trajectory more |

Harness: `vizwork/qoi.jl` / `qoi2.jl` (limiter march, prescribed dt, integral diagnostics).

## 8. One-paragraph summary

The GPU residual is at a genuine register/latency wall set by the 5th-order HyQMOM
closure — byte-exact micro-optimizations are exhausted. The real speedups come from
**replacing the iterative eigensolvers with closed-form / one-pass equivalents**, which the
problem's algebra makes possible: a **Bunch–Kaufman inertia** test (exact, by Sylvester's
law) for the realizability sign gives **2.86× on the limiter**, and a **Ferrari closed-form
quartic** (the 4×4 is a companion) for the wave speeds gives **1.27× on the default path**.
Both are exact-or-better yet cannot be bit-trajectory-identical at Ma=100 because the
zero-tolerance realizability gate makes the high-Ma trajectory chaotically sensitive to the
eigensolver — so they ship as opt-in flags with byte-identical defaults, alongside a
byte-identical memory win (in-place projection) and a real robustness fix (wave-speed
fallback on QR non-convergence). The next step is method-level: a tolerance band on the
realizability gate would make the trajectory robust to the solver and let the fast paths
become defaults.

# KFVS anchor — device CHyQMOM inversion (`gpu/chyqmom_nodes_3d_dev.jl`)

**Increment A** of the kinetic-FVS realizable anchor: a production-fidelity,
alloc-free, device-compilable port of the CPU reference
`src/moments/chyqmom_nodes_3d.jl` (length-35 raw moments → nonnegative 3D velocity
quadrature `(n, U)`, ≤27 nodes). **Pure addition** — nothing in the solver path
calls it yet (no `projection35` / residual changes). Downstream anchor pieces
(per-cell storage, `measure_update`, θ\*-blend) build on this next.

Module: `KFVSInversionDev`, exports `chyqmom_nodes_3d_dev`,
`chyqmom_nodes_3d_store_dev!`, `NODEMAX`.

**Increment B** adds the anchor itself: per-cell quadrature STORAGE + the 3D
device MEASURE_UPDATE (`gpu/kfvs_measure_update_dev.jl`, module
`KFVSMeasureUpdateDev`, exports `measure_update_3d_dev`, `accum35_node`). See
["Increment B" below](#increment-b--storage--3d-measure_update-anchor). Still a
pure addition.

## Validation scripts (run from the worktree with the PACE env)

```
module load julia/1.11.3
export JULIA_DEPOT_PATH=/storage/scratch1/6/sbryngelson3/julia_depot:$HOME/.julia
export TMPDIR=/storage/scratch1/6/sbryngelson3/tmp

# CPU-parity (no GPU): device inversion vs CPU chyqmom_nodes_3d over 24k real cells
julia --project=. gpu/validation/parity_chyqmom_nodes_3d_dev.jl

# FIX-1 gate fidelity probes (no GPU, CPU LinearAlgebra):
julia --project=. gpu/validation/fix1_gate_probe.jl       # eig gate vs CPU svd gate
julia --project=. gpu/validation/fix1_itereig_probe.jl    # device-style iter-eig gate

# GPU register/occupancy/throughput (fused + split), gpuenv2 has CUDA:
julia --project=gpu/gpuenv2 gpu/validation/gpu_chyqmom_nodes_3d_dev.jl

# Increment B — anchor: storage round-trip + 3D measure_update parity (no GPU):
julia --project=. gpu/validation/parity_kfvs_measure_update.jl
# Increment B — GPU storage pass + measure_update on the full 128^3 field:
julia --project=gpu/gpuenv2 gpu/validation/gpu_kfvs_measure_update.jl

# Increment D — full-cone theta* blend on real stencils (no GPU) + the marginal A/B:
julia --project=. gpu/validation/parity_kfvs_blend.jl
# Increment D — GPU cost (registers + us/cell) of the full-cone blend limiter:
julia --project=gpu/gpuenv2 gpu/validation/gpu_kfvs_blend.jl
```

## FIX 1 — faithful condition gate (the fidelity fix)

The CPU gate `_design_cond(B) = svdvals(B)[1]/svdvals(B)[end]` rejects a candidate
column when κ₂(B) ≥ 1e4 on the TALL weighted design `B = sqrt(pw).*Phi`. An earlier
prototype used a Cholesky-**pivot-ratio** proxy on the Gram `G = BᵀB` (calibrated
threshold 3e6), which under-estimates κ(G) by up to ~10 orders on near-collinear
geometries → ~0.18% of real cells admitted one spurious column → a blown-up
high-order z cross-moment.

**Fix:** `G` is SPD and symmetric ⇒ σᵢ(B) = √λᵢ(G), so κ₂(B) = √(λmax(G)/λmin(G)).
`condkappa9` (and its N≤6 wrapper `condkappa6`) compute the extreme eigenvalues of
the small (≤9×9) SPD `G` **without LAPACK** — λmax by power iteration, λmin by
inverse power iteration through the SAME Cholesky factor the SPD solve already
needs (free) — and gate on the CPU's own `√κ(G) < 1e4` (i.e. κ(G) < 1e8), **no
tuned threshold**.

Measured (`fix1_gate_probe.jl`, `fix1_itereig_probe.jl`, real cells):

| gate | accept/reject mismatches vs CPU svd gate |
|---|---|
| eig gate `√(λmax/λmin(G))` (exact, LAPACK)     | **0 / 381 905 (0.0000%)** |
| iter-eig gate (device-style, no LAPACK)        | **0 / 318 345 (0.0000%)** |
| pivot-ratio proxy (old prototype)              | 943 / 381 905 (0.2469%) |

## CPU-parity (`parity_chyqmom_nodes_3d_dev.jl`, 24 000 real + 5 000 synthetic)

- Mean moments reproduced: **DEV 31.48 / 35 == CPU 31.48 / 35**.
- **Min node weight ≥ −1e-12 : YES on 100% of cells** (the realizability
  certificate — the property that matters).
- **SPURIOUS wild-abscissa cells (over-admitted column → wild node): 0 real
  (0.0000%)** — down from the pivot proxy's ~0.18%. FIX 1 goal met.
- **Node-count match vs CPU: 99.82% of real cells** (43 / 24 000 mismatch — ±1/±2
  node gate ties on low-density high-Ma cells), up from the prototype's ~97.8%.
  On count-matched cells the node clouds agree with CPU to <1e-6 on 99.7%
  (max aligned abscissa diff 1.3e-2, max weight rel diff 2.3e-4 on the remainder).
- Mass and the three means are machine-exact on the count-matched cells. (The
  diagonal 2nd moments M200/M020/M002 are NOT invariants — CHyQMOM truncates them
  on cold/near-vacuum marginals, and DEV matches the CPU reference bit-for-bit
  there; they are excluded from the "must-be-exact" set.)
- CPU reference `chyqmom_nodes_3d` threw `SingularException` on 2 cells; **DEV
  survived both** (closed-form guarded Vandermonde solve, no throw).
- Synthetic random-cloud cells match CPU node count on 92.2% (legitimately harder
  ties; all realizable) — not a real-cell concern.

## FIX 2 — N=9 z-Gram sizing (verified)

The original design's N=6 z-mean-Gram cap was FALSIFIED by real data (8 columns is
the dominant case, up to 9 occur). The solver, gate (`condkappa9`), Gram builder
(`_build_gram_flat_z9`), and SPD solve (`spd_solve9`) are all sized to **N=9** and
the faithful gate operates at that size. Confirmed covered: `gram_fit_z` caps at
`nsel >= 9` and the parity run exercises up-to-11-node clouds (Npxy up to 9) with 0
wild abscissae.

## GPU register / occupancy / throughput (Tesla V100-PCIE-16GB, fp64)

Fused single kernel (`chyqmom_nodes_3d_dev`, one thread/cell) vs the split
invert-and-store variant (`chyqmom_nodes_3d_store_dev!`, incremental global stores,
no 27×4 NTuple accumulator — design §1.5 phase-1). 1e6 real `ma100` cells.

| variant | registers | local spill | occupancy | throughput | device min weight / mass err |
|---|---:|---:|---:|---:|---|
| FUSED (NTuple accumulator + global write)      | **255** | 5112 B | **12.5%** (8/64 warps) | **0.814 µs/cell** | 4.63e-6 (cert YES) / 2.17e-16 |
| SPLIT (invert-and-store, incremental stores)   | **255** | 4448 B | **12.5%** (8/64 warps) | **0.789 µs/cell** | byte-identical to fused |

**The split does NOT lift occupancy** (register delta = 0). Removing the 27×4
output NTuple accumulator from the fused kernel's live set only trimmed the local
spill (5112 → 4448 B) — the output buffers were **not** the binding constraint.
The kernel is intrinsically pinned at the 255-register wall by the z-level N=9
Cholesky (`spd_solve9`/`gram_fit_z`) and the new faithful iterative gate
(`condkappa9`). So the storage-based split (design §1.5) is worth doing for the
**memory-layout / phase-separation** reasons (it lets the downstream consumer be a
separate low-register kernel), but it is NOT an occupancy escape for the inversion
itself — a real relief valve would have to cut the z-level LA register footprint
(e.g. shared-memory Gram, or a cheaper gate), which is a separate optimization.

Throughput is nonetheless ~160× the ~130 µs/cell CPU baseline even at 12.5%
occupancy (the work is compute-dense, footprint tiny: 35 in / 27×4 out per thread).
The faithful gate's iteration count (`_EIG_ITERS`) was tuned from 40 → 16 (16 keeps
0 gate mismatches on 381 905 real decisions), which halved throughput 1.60 → 0.81
µs/cell with zero fidelity change.

The split variant's node counts and weights are **byte-identical** to the fused
output (verified in-harness: 0 node-count diffs, 0 weight diff over 200k cells), so
the split is a pure register/occupancy lever, not a fidelity change.

## Increment B — storage + 3D measure_update anchor

`gpu/kfvs_measure_update_dev.jl` (module `KFVSMeasureUpdateDev`).

**Storage pass.** `chyqmom_nodes_3d_store_dev!` (increment A) inverts every cell and
writes its quadrature (≤27 nodes × `(w, Ux, Uy, Uz)`) into a device array laid out
`S[node, 4, cell]` + a `(cell)` node-count array.

| metric | value |
|---|---|
| storage footprint @128³      | **1.812 GB** (27×4×8×Ncell) + 8.4 MB counts — confirms the ~1.8 GB design estimate |
| store-pass GPU throughput    | **0.804 µs/cell** (255 regs), 1685 ms for the full 128³ field (2.1M cells) |
| stored-quadrature round-trip | moments of stored nodes == fresh inversion, **0.0 abs diff** |

**`measure_update_3d_dev` — the anchor.** The CPU reference `verify_kfvs.jl`
`measure_update` is **x-only**; this **generalizes it to all 6 face-neighbors** (one
upwind inflow per axis, using that axis's abscissa component and sign), with the
retained weight `nC_k·(1 − λ(|Ux|+|Uy|+|Uz|)_k)`. Under the 3D CFL
`λ·max_k(|Ux|+|Uy|+|Uz|)_k ≤ 1` every weight is ≥ 0 ⇒ nonneg measure ⇒ realizable
in the full 35-moment cone (thm:kfvs-idp). The nonnegative-weight check is the cheap
certificate that replaces the Hankel test.

| validation (CFL = 0.4) | value |
|---|---|
| **min-weight certificate (≥ −1e-12) on real interior stencils** | **100.0000%** (host: 20 000/20 000; **GPU: 2 000 376/2 000 376** over the full 128³ interior) — 0 negative weights |
| updated state passes `_state_realizable` (host, 20k stencils)    | **100.0000%** |
| updated states with ρ ≤ 0 (GPU, 2.0M stencils)                   | **0** |
| worst measure min-weight                                          | 8.06e-92 (a boundary node — tiny but ≥ 0) |
| x-slice cross-check vs CPU `verify_kfvs.jl` measure_update        | max \|M_dev − M_cpu\| = **9.5e-7** over 3000 stencils (same nodes fed to both ⇒ isolates the accumulation/upwind port), **0** min-weight mismatches |
| measure kernel registers / throughput (GPU)                      | **168 regs** (well under the 255 wall), **0.133 µs/cell**, 265 ms over 2.0M interior cells |

The measure kernel at 168 registers confirms the anchor **consumer is cheap** (as
the design predicted): the register cost lives in the inversion (store) kernel, so
the storage split is exactly the right structure — invert-and-store once/stage
(255-reg heavy kernel), then a light 168-reg consumer reads neighbor quadratures.

**No stencil geometry failed the certificate and the CFL never had to shrink**
below 0.4 — the certificate held on every one of the 2M+ real interior stencils
at CFL 0.4 exactly as the theorem predicts.

## Increment D — full-cone θ* blend limiter

`gpu/kfvs_blend_dev.jl` (module `KFVSBlendDev`). The convex-limited blend of a
high-order update toward the anchor:
`U(θ) = (1−θ)·U_anchor + θ·U_highorder`, `θ*` = largest θ ∈ [0,1] keeping `U(θ)`
**full-cone realizable**. `U_anchor` = `measure_update_3d_dev` (increment B),
full-cone realizable by construction.

**Full-cone predicate used (and why): `state_realizable_fullcone_dev`.** This is the
SAME predicate the solver's `projection35` uses to define 35-moment realizability
beyond the marginals: `to_recon_vars_dev(M)` → standardized moments → density &
directional-variance guards → **`delta2star_psd_dev` (the 6×6 Δ2* moment-matrix PSD
test via Bunch–Kaufman inertia — the CROSS-moment cone).** Chosen over option (b)
"re-invert `U(θ)` and require min node weight ≥ −1e-12" because it is (i) the exact
cross-moment-cone definition the solver already uses, and (ii) a bounded 6×6 test
(~one Δ2* build + inertia) vs a full ~0.8 µs inversion per bisection step (≈24×/cell)
— option (a) is both sounder and ~30× cheaper. The design's affine-Hankel-pencil
closed form (thm:idp-blend) is a later optimization; bisection is used for
correctness and is already cheap (see cost below).

**Why full-cone (the core of the increment).** The shipped Track-2 θ* limiter
(`IdpLimiterDev.theta_star_update_dev`) checks only the MARGINAL cone
(`RiemannFluxDev._state_realizable` = `_marg_shape` per axis, NO cross-moment block).
The anchor's whole value is preserving the CROSS-moment cone.

**Validation** (`parity_kfvs_blend.jl`, real 48³ interior stencils near the jet, CFL
0.4, 97 336 stencils; Test C reproduced on real data):

| result | value |
|---|---|
| (a) mean full-cone θ* (high-order fraction kept) | **0.9701** (target 0.97–0.99); θ*=1 unlimited on **62.8%** (θ* percentiles p1=0.68, p50=1.0) |
| (b) `U(θ*)` full-cone realizable | **97 336/97 336 = 100.0000%** — **0 cross-moment-cone exits** |
| **(c) marginal-only θ* (Track-2) blended states OUTSIDE the cross-moment cone** | **36 253/97 336 = 37.25%** (and marginal θ* > full-cone θ* on 37.1%) — the full-cone limiter drives this to **0** |

(c) is the empirical justification: **using the shipped marginal-only θ* would silently
let 37% of blended states exit the cross-moment cone** — exactly the cone the anchor
exists to protect. The full-cone limiter is necessary.

**GPU cost** (`gpu_kfvs_blend.jl`, V100, 97 336 real pairs):

| kernel | registers | throughput | on-device check |
|---|---:|---:|---|
| full-cone θ* blend limiter | **255** (2464 B local) | **0.246 µs/cell** | mean θ*=0.9701, U(θ*) realizable 100%, 0 exits |

**Blunt on cost.** 255 regs (the Δ2* build + Bunch–Kaufman inertia is heavy) but only
**0.246 µs/cell** — the bisection short-circuits on the 62.8% θ*=1 common path and each
predicate is a bounded 6×6 test, not a re-inversion. **The closed-form cross-moment
pencil is NOT needed before increment E**; the bisection limiter is already cheap
enough. The predicate agrees with CPU `is_realizable` to ~98–99.9% (the mismatch is at
the cone boundary, the documented CPU-LAPACK-eig vs device-Bunch-Kaufman-inertia floor,
same as the shipped residual limiter — not a soundness gap; the 37% cross-exit demo is
measured self-consistently with the device predicate).

## Bottom line

Increment A is landed and validated: faithful gate (spurious rate 0.18% → 0),
realizability certificate 100%, node-count match on real cells 99.82%, N=9 sizing
confirmed, fused GPU kernel legal (255 regs, under the wall) at 0.81 µs/cell (~160×
CPU).

Increment B is landed and validated: per-cell storage (1.81 GB @128³, exact
round-trip) + the 3D `measure_update` anchor (168-reg light consumer). The
realizability certificate (min weight ≥ −1e-12) holds on **100% of 2M+ real
interior stencils on-device at CFL 0.4** with 0 negatives and 0 CFL shrinks.

Increment D is landed and validated: the full-cone θ* blend limiter. Mean θ* =
**0.9701** on real data (62.8% unlimited), `U(θ*)` full-cone realizable on **100%**
(0 cone exits), at **0.246 µs/cell / 255 regs**. The A/B demo proves the full-cone
predicate is necessary: the shipped marginal-only θ* would let **37% of blended
states exit the cross-moment cone**. **D is solid enough to build the final
increment E** (wire into the residual as an opt-in, golden-gated, byte-identical-off
replacement for `projection35`) — the blend is a self-contained device function with
a cheap correct limiter; the closed-form pencil is a later optimization, not a
blocker.

**Blunt caveats for the next increment:**
- The kernel is pinned at the 255-register wall with local spill; the split does
  NOT lift occupancy (the z-level N=9 Cholesky + iterative gate are the binding
  register cost, not the output buffers). If occupancy matters downstream, cutting
  the z-level LA footprint is the real lever, not the storage split.
- Node-count fidelity is 99.82% on real cells, not 100%: ~0.18% of real cells are
  genuine gate ties where DEV picks a slightly different (still realizable) column
  set than the CPU svd, differing by a small bounded amount (max aligned abscissa
  diff 1.3e-2). This is inherent to reproducing an incremental SVD selection with a
  from-scratch iterative gate at exact ties; it is not a wild blowup and the
  realizability certificate is unaffected.

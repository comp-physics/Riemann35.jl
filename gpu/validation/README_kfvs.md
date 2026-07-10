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

# Increment E — CPU order-3 byte-identity gate (flag off) + flag-on validation:
julia --project=. gpu/validation/kfvs_cpu_order3_baseline.jl              # flag OFF
KFVS_ANCHOR=1 HYQMOM_ANCHOR_STATS=1 julia --project=. gpu/validation/kfvs_cpu_order3_baseline.jl  # flag ON
# Increment E — GPU order-3 golden (flag-off byte-identity + flag-on smoke):
julia --project=gpu/gpuenv2 gpu/validation/kfvs_gpu_order3_golden.jl

# Increment F2/F3 — Ma=100 anchor conservation+realizability validation (CPU):
julia --project=. gpu/validation/kfvs_ma100_anchor_validate.jl   # OFF vs ON (F3 flux-level), 8/20/40 steps
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

## Increment E — wire the anchor→blend into the order-3 march (opt-in, byte-identical off)

The FIRST increment that touches the solver path. A runtime `use_kfvs_anchor::Bool`
(default **false**) is threaded, mirroring the `stage_bgk` opt-in, through the
order-3 entry points; when false the code path is **untouched**.

**Files modified (the solver path — existing files):**
- `src/numerics/highorder_3d.jl` — `step_highorder_3d!` gains `use_kfvs_anchor`;
  the per-stage realizability step becomes `use_kfvs_anchor ? _anchor_interior! :
  _project_interior!`. Adds `_anchor_interior!` (CPU flag-on path) + anchor stats.
  Includes the CUDA-free `gpu/chyqmom_nodes_3d_dev.jl` (hardened inversion) for the
  anchor only.
- `src/simulation_runner.jl` — reads `get(params, :use_kfvs_anchor, false)`, passes it.
- `src/Riemann35.jl` — exports `reset_anchor_stats!`, `anchor_stats`.
- `gpu/timestep3d_order3_gpu.jl` — `march3d_order3_gpu!` gains `use_kfvs_anchor`;
  guards the projection kernel + a stage-input cube snapshot (`Gin` allocated only
  when opt-in). Adds `_copy_cube!` + an interim `_anchor_interior!` GPU kernel (==
  `realizable_3D_M4_dev` projection for now — the real device anchor kernel is a
  follow-up).
- `gpu/gpu_run.jl` — `run_gpu_3d` gains `use_kfvs_anchor`, passes it.

**CPU BYTE-IDENTITY (flag off — the hard requirement):** order-3 crossing-jets 16³
Ma=10, 4 steps, flag OFF vs pristine `main`:
```
pristine (main) : L2 = 2.92509950810618875e+02  sum = 1.70493403083572848e+04
flag-off (branch): L2 = 2.92509950810618875e+02  sum = 1.70493403083572848e+04
```
**relL2 = 0.0 EXACTLY (bit-identical).** `test_hiorder3_cpu.jl` 8/8,
`test_highorder_3d.jl` pass. The device-inversion include is inert on the off path.

**CPU FLAG-ON validation** (well-posed crossing-jets, order=3):
- Ma=10, 12 steps: finite, **0 unrealizable**, conservation **identical** to the
  projection path (mass drift 9.19e-8, energy 8.21e-8); **mean θ* = 1.0,
  projection-would-fire = 0, fallback = 0 ⇒ projection RETIRED** (blend realizable
  by construction).
- The anchor uses the **hardened device inversion** `chyqmom_nodes_3d_dev` (FIX-1
  gated): the CPU reference `chyqmom_nodes_3d` **threw on 5 and produced wild
  abscissas (max|U|=7.6e4) on 23** of a 4096-cell real block — feeding those into
  measure_update produces NaN, so the hardened inversion is load-bearing here.

**GPU golden** (order-3 march, 16³ crossing-jets, V100, exact FNV-1a bitsum of the
final field):
```
main (pristine) flag-off : L2=2.40090644131310427e+02  bitsum=0x7c27cd25463cbf83
branch          flag-off : L2=2.40090644131310427e+02  bitsum=0x7c27cd25463cbf83
```
**Identical bitsum ⇒ the GPU order-3 march flag-off is BYTE-IDENTICAL to pristine
`main` on device** (the plumbing did not perturb the GPU path). Both flag-off runs
are finite (nonfinite=0).

**GPU flag-on is a FOLLOW-UP.** The full device measure_update→blend kernel is not
built yet; the interim GPU `_anchor_interior!` kernel applies the SAME
`realizable_3D_M4_dev` projection as the default path (code-identical to
`_proj_interior!`), so GPU flag-on == flag-off by construction (safe, well-defined).
The stage-input cube snapshot (`Gin`/`_copy_cube!`) scaffolding is in place for the
real kernel. The validated anchor flag-on path is the **CPU** one above.

**BLUNT limitation (a real gap — not hidden):** the anchor-blend enforces the
**Δ2* cross-moment cone** (its designed target — the cone split-HLL loses) but NOT
the **marginal s3max skewness cap / variance floors** that `realizable_3D_M4` also
applies. On a raw-snapshot stress block (itself unstable on the OFF path:
unreal=2106, mass drift 5.6e-2) the anchor path went non-finite where the marginal
clamp would have stabilized. So the flag-on path is validated as the **cross-moment
projection replacement on well-posed / moderate-Ma cases**; it is **NOT yet a
validated high-Ma drop-in** — the marginal skewness guard remains a separate concern
(the task scoped the replacement to the per-cell cross-moment projection; the
reconstruction/marginal guards were left as-is). A production high-Ma path should
either (a) apply the marginal s3max clamp to the anchor output as a floor, or (b)
fold the s3max bound into the blend predicate. Flagged for increment E follow-up.

## Increment F2 — close the high-Ma marginal gap (CPU anchor path)

The gap flagged in E: production `realizable_3D_M4` does BOTH (a) the Δ2* cross-moment
projection (anchor+blend replaces this) AND (b) a MARGINAL regularization — the
s3max standardized-skewness cap (`s3max = max(40, 4+|Ma|/2) = 54` at Ma=100, Fox's
high-Ma stabilizer), the Hankel/variance floors (`H2ii ≥ 1e-6`,
`S4ii ≥ S3ii²+1+1e-6`), and the S2 floor. The anchor path only did (a).

**What F2 does** (CPU `_anchor_interior!` only; GPU still placeholder; opt-in):
`_marginal_regularized(M,Ma,s3max)` is the marginal predicate (via `M2CS4_35`,
byte-consistent with `is_realizable`). The anchor is regularized with
`realizable_3D_M4` (marginal clamp; its Δ2* projection is a near-no-op on the
nonneg-measure anchor) so θ=0 is realizable in BOTH the cross-moment cone AND the
marginal set ⇒ θ* always feasible ⇒ realizable-by-construction. The **θ* predicate =
`is_realizable` (Δ2*) AND `_marginal_regularized` (marginals)**, so θ* never leaves
the marginal set (chosen over a post-blend clamp, which s3max's nonlinearity could
re-violate).

**Ma=100 validation** (well-posed crossing-jets, OFF path stable; CPU, 8/20/40 steps):

| metric | OFF (projection) | ON (anchor+F2) |
|---|---|---|
| non-finite / ρ≤0 / unrealizable | 0 / 0 / 48–272 | **0 / 0 / 0** (all horizons) |
| mass drift (8/20/40)  | 1.1e-8 / 2.4e-8 / 7.7e-7 | 1.8e-2 / 5.4e-3 / 1.4e-2 |
| energy drift (8/20/40)| 4.0e-11 / 2.6e-8 / 4.8e-7 | 2.4e-2 / 5.1e-2 / **9.7e-1** |
| θ* engaged (θ*<1)     | — | **0 cells (θ*=1 everywhere)** |
| would-regularize / fallback | — | 0 / 0 |

**F2 SUCCESS on the primary goal: the anchor path is FINITE, ρ>0, 0-unrealizable at
Ma=100 across all horizons** — the non-finite blowup gap from E is closed at the
stability level, with the marginal guard in place.

**But F2 exposes a deeper, real conservation issue — and I am blunt about it.** On
this well-posed Ma=100 case the high-order update is **already fully realizable**
(θ*=1 everywhere, would-regularize=0), so the anchor/blend **never engages** — yet
energy drift grows to **~97% by 40 steps** vs the projection's 4.8e-7. Diagnosed
root cause: **`realizable_3D_M4` is a LOSSY round-trip that perturbs EVERY cell** —
measured mean 1.6e-4 / max 1.5e-2 even on *strictly-realizable* cells (margin ≥ 1e-8,
where `projection35` should be a no-op). That per-stage cross-moment perturbation
incidentally damps the high moments and stabilizes energy conservation. The anchor,
by design faithful to the high-order cross-moments at θ*=1, **omits this incidental
damping**, so conservation regresses even though nothing was unrealizable.

**Verdict: F2 closes the stability (non-finite) gap but the anchor is NOT yet a
conservation-neutral Ma=100 drop-in for the projection.** The surviving failure mode
is specific: `projection35`'s lossy per-cell round-trip does non-realizability work
(implicit high-moment damping) that stabilizes energy at high Ma; the anchor
correctly replaces the *realizability* role but not this *incidental damping* role,
so on well-posed Ma=100 (where the blend is a no-op) energy conservation drifts to
~O(1). Retiring the projection at Ma=100 therefore requires either (i) accepting a
different (less lossy, more faithful) high-Ma conservation behavior and validating it
physically, or (ii) adding an explicit, conservative high-moment damping to the
anchor path to replace the projection's incidental one. This is a genuine finding,
not a plumbing bug — byte-identity-off remains exact and CPU tests pass.

## Increment F3 — flux-level face-shared-θ anchor (conservative by construction)

**Root cause of F2's non-conservation** (theory in notes: thm:idp-conservative /
cor:proj-noncons): E/F2 blended **states** per-cell with a *per-cell* θ*, which is
non-conservative — at a shared face the implied flux is double-valued when
`θ_c ≠ θ_c'`. **F3 does the blend at the FLUX/FACE level with a face-shared θ**,
reusing the residual's EXISTING conservative θ*-IDP machinery (which already blends
fluxes `F = F_LO + θ(F_HO−F_LO)` with `θ_interface = min` over the two adjacent cells
⇒ single-valued face flux ⇒ telescoping ⇒ conservative). Two swaps under
`use_kfvs_anchor`, both in `src/numerics/highorder_3d.jl`:

1. **F_LO anchor flux: HLL → kinetic-FVS.** `_kfvs_face_flux_tup(cL,cR,axis)` = the
   upwind quadrature flux (L-cell nodes with `U·n>0` + R-cell nodes with `U·n<0`,
   times the moment monomials) — exactly the interface flux implied by the
   `measure_update` anchor, via the hardened device inversion. Single-valued in the
   shared `(cL,cR)` ⇒ the two adjacent cells see the same flux ⇒ conserves.
2. **θ* predicate: marginal-only → full-cone + marginal.** `_theta_star_fullcone` =
   `is_realizable` (Δ2* cross-moment cone) AND `_marginal_regularized` (s3max/floors),
   with the face-min reconciliation kept. Projection **and** the E/F2 per-cell state
   blend are DROPPED on the anchor path (`realize!` is a no-op; the flux blend is
   realizable-by-construction).

**Ma=100 validation** (well-posed crossing-jets, CPU, 8/20/40 steps):

| metric | OFF (projection) | ON (F3 flux-level) | ON (F2 per-cell — for contrast) |
|---|---|---|---|
| mass drift (8/20/40)   | 1.1e-8 / 2.4e-8 / 7.7e-7 | **5.4e-9 / 2.6e-8 / 1.3e-7** | 1.8e-2 / 5.4e-3 / 1.4e-2 |
| **energy drift** (8/20/40) | 4.0e-11 / 2.6e-8 / 4.8e-7 | **2.8e-10 / 7.1e-10 / 2.8e-7** | 2.4e-2 / 5.1e-2 / **9.7e-1** |
| finite / ρ>0 | yes | **yes** | yes |
| mean flux θ* (retention) | — | **0.919** (θ<1 on 11.6% of faces) | θ*=1 (blend never engaged) |

**F3 CONSERVES TO MACHINE PRECISION at Ma=100** (energy drift 2.8e-10, comparable to
the projection's 4.0e-11 and ~9 orders better than F2's 9.7e-1) — **thm:idp-conservative
confirmed**. And the flux limiter genuinely engages (mean θ*=0.92, 92% high-order
retention), unlike the F2 per-cell blend which was a no-op.

**SURVIVING GAP — cross-moment-cone drift (blunt):** `margin < −1e-8` (the Δ2* cone)
holds on **224 / 704 / 960 cells** at 8/20/40 steps, worst margin **−0.99**, vs the
projection path's cells all at machine-zero (worst −9.5e-13). **A SINGLE F3 forward-
Euler step is fully realizable (0 cells with margin<−1e-8)** — so the θ*-IDP full-cone
bound holds per stage (thm:idp-cons-real for one step); the exit is a **multi-step
accumulation** without the per-stage re-projection: boundary cells slowly drift out of
the cross-moment cone over many steps (count grows ~linearly, field stays finite —
bounded, not a blowup). The projection path avoids this only because it re-clamps every
stage. Closing it without breaking conservation needs either a *conservative* full-cone
re-projection (Guermond-Popov limiter on the assembled state) or accepting the bounded
drift — flagged for follow-up.

**Verdict:** F3 makes the anchor CONSERVATIVE at Ma=100 (the F2 pass/fail) — machine-
precision mass+energy, 92% high-order retention, finite/ρ>0, projection retired. It is
**a genuine conservative drop-in for the cross-moment projection on conservation**, but
NOT yet fully realizability-neutral: cross-moment-cone violations accumulate over many
steps (bounded, worst −0.99 by 40 steps) because there is no per-stage re-projection.
Byte-identity-off exact; CPU tests 8/8.

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
states exit the cross-moment cone**.

Increment E is landed with the hard requirement met: the opt-in `use_kfvs_anchor`
(default false) is threaded into the order-3 CPU + GPU march, and flag-off is
**byte-identical to pristine main** on BOTH CPU (relL2=0.0 exactly) and GPU
(identical bitsum). The CPU flag-on path (anchor → full-cone θ*-blend, projection
retired) is validated on well-posed moderate-Ma cases.

Increment F2 closed the high-Ma **stability** gap (marginal s3max/variance guard folded
in ⇒ finite/ρ>0 at Ma=100) but its per-cell STATE blend was non-conservative (energy
drift ~0.97 by 40 steps) — because a per-cell θ* makes the implied face flux
double-valued.

Increment F3 fixes that by moving the blend to the **FLUX/FACE level with a
face-shared θ** (kinetic-FVS F_LO + full-cone θ* predicate + face-min θ, projection
and per-cell blend dropped). At Ma=100 this **conserves mass AND energy to machine
precision** (energy drift 2.8e-10 / 7.1e-10 / 2.8e-7 at 8/20/40 steps, comparable to
the projection's 4.0e-11 / 2.6e-8 / 4.8e-7 and ~9 orders better than F2) with 92%
high-order retention (mean θ*=0.92), finite/ρ>0, projection retired — **thm:idp-
conservative confirmed; the conservation pass/fail is a PASS.** One surviving gap: the
Δ2* cross-moment cone accumulates violations over many steps (margin<−1e-8 on
224/704/960 cells, worst −0.99 by 40 steps) — a **single** F3 step is fully realizable,
so it's a multi-step accumulation from the absence of per-stage re-projection, bounded
(field stays finite), not a blowup. Remaining before production: the GPU device flux
path (E/F3 follow-up — GPU still the projection placeholder), a **conservative** full-
cone re-projection or Guermond-Popov limiter to close the multi-step cone drift without
breaking conservation, and a physical high-Ma accuracy comparison vs the projection.

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

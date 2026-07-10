# KFVS anchor — device CHyQMOM inversion (`gpu/chyqmom_nodes_3d_dev.jl`)

**Increment A** of the kinetic-FVS realizable anchor: a production-fidelity,
alloc-free, device-compilable port of the CPU reference
`src/moments/chyqmom_nodes_3d.jl` (length-35 raw moments → nonnegative 3D velocity
quadrature `(n, U)`, ≤27 nodes). **Pure addition** — nothing in the solver path
calls it yet (no `projection35` / residual changes). Downstream anchor pieces
(per-cell storage, `measure_update`, θ\*-blend) build on this next.

Module: `KFVSInversionDev`, exports `chyqmom_nodes_3d_dev`,
`chyqmom_nodes_3d_store_dev!`, `NODEMAX`.

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

<!-- GPU_NUMBERS -->
_(filled by `gpu_chyqmom_nodes_3d_dev.jl`; see the SUMMARY block it prints.)_

The split variant's node-count and weights are byte-identical to the fused output
(verified in-harness), so the storage-based split is a pure register/occupancy
lever, not a fidelity change.

## Bottom line

Increment A is landed and validated: faithful gate (spurious rate 0.18% → 0),
realizability certificate 100%, node-count match on real cells 99.82%, N=9 sizing
confirmed. Solid enough to build the anchor's per-cell storage + `measure_update`
+ θ\*-blend on top of next.

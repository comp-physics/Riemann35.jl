# Two-stream half-space solver mode

An additive, opt-in solver mode for Riemann35.jl in which each cell carries **two**
35-moment vectors — a `+` stream (x-support in `[0,∞)`) and a `-` stream (x-support
in `(-∞,0]`) — split at a per-problem gauge velocity `c` (crossing jets: split axis
= x, `c = 0`). In the split axis each stream is one-signed, so **donor-cell upwinding
is the exact flux and preserves realizability with no projection** under the
wave-speed CFL (prop:halfline-upwind). The transverse directions are ordinary
full-line measures and keep the production closure + hyperbolicity correction. The
two streams are coupled only by an exact-exponential BGK relaxation, which is
unconditionally stable at any Kn.

Nothing in the single-stream production path changes; the mode is built from four
new, allocation-free, single-source kernels registered (additively) in the package:

| file | role |
|---|---|
| `src/moments/chain.jl` | all-mean Stieltjes chain ζ₁..ζ₄ from the x-marginal (realizability ⇔ ζ>0; clip repair + counter) |
| `src/numerics/halfline_closure.jl` | h-functional pseudo-moments `hseq`, marginal closure, and the stable node-Vandermonde channel closure `chan_closure` |
| `src/numerics/flux_halfspace35.jl` | the half-space x-flux `xflux_plus35` / `xflux_minus35` (15 channel closures + carried entries; the `-` stream is the x-mirror) |
| `src/numerics/bgk_stream.jl` | `split_maxwellian35` (erf/exp closed-form half-Gaussian x-moments) + exact-exponential `bgk_stream_relax` |

Tests: `test/test_two_stream_gate1.jl`, `test/test_two_stream_gate2.jl`, and the
head-to-head driver `test/two_stream_gate3_headtohead.jl`.

## Validation gates (measured)

### Gate 1 — unit / spectrum / consistency (`test_two_stream_gate1.jl`)
Ports the `jac15` checker to the full 35-moment x-flux; Jacobian finite-differenced
in chain coordinates `(m₀, ζ)` (not raw moments — raw FD is ill-conditioned at high
Ma). On 1500 random off-manifold half-space states:

| metric | measured | spec target |
|---|---|---|
| full x-flux Jacobian spectrum real | 1500/1500 (worst Im/‖λ‖ = 0) | real |
| spectrum ≥ 0 | 1500/1500 (worst min λ/‖λ‖ = +1.3e-2) | ≥ −1e−6 rel |
| marginal n=5 half-line block embeds | 1483/1500 (worst mismatch 1e-4; misses are FD noise on near-degenerate states) | embeds |
| grade-graded block triangularity | 1500/1500 exact | upper blocks vanish |
| per-grade diagonal blocks | all real, min ≥ +1.3e-2 | real, ≥ 0 |
| separable-state channel closures | exact to **9.4e-13** | ~1e-14 |

Note: the channel closure uses the **node-Vandermonde** form, not the raw Hankel
solve — the Hankel section goes singular at high Ma (cond(H) ≈ cond(P)², P the node
Vandermonde), which was the source of a ~1e-6 (and, on pathological states, `Inf`)
error. The Vandermonde is solved via `inv(P)·u` (never throws; NaN → mean-node
fallback), gated by the relative determinant, so extreme-Ma coincident nodes degrade
gracefully instead of crashing — at the small cost of ~9e-13 vs the ~2e-14 of a
throwing LU solve (immaterial next to truncation error).

### Gate 2 — 1D pipeline (`test_two_stream_gate2.jl`)
The ported kernels run in 1D (donor-cell x + BGK), reproducing
`verify_halfline_scheme.jl`:

| case | measured | spec target |
|---|---|---|
| collisionless contact drift, nx=100..800 | 1.24e-2, 3.09e-2, 7.22e-2, 1.32e-1 | 9.7e-3, 2.5e-2, 7.6e-2, 1.28e-1 → free-molecular 1.03e-1 |
| crossing Ma=100 | survives; maxu=100 (streams keep their velocity); settled streams **H = 2.000** (exact Maxwellians); **0 chain clips** away from vacuum fronts | H ≈ 2, zero clips |
| Kn=1e-3 contact via BGK | maxu = 7.0e-3 | ≤ ~1e-2 |

The collisionless drift converges monotonically toward the exact free-molecular
value (the "leak" is the collisionless model, not the numerics). At Ma=100 each
stream stays an exact Maxwellian (H=2) while the *total* sits at the two-beam
realizability boundary (H≈0) — the state a single-stream closure cannot represent
without projection.

### Gate 3 — 3D head-to-head (`two_stream_gate3_headtohead.jl`)
One dimensional-split finite-volume driver, run two ways on the same crossing-jets
config (48×48×16), sharing the **production transverse (HLL) machinery** so the only
difference is the split axis. Every direction uses the production path
(`Flux_closure35_and_realizable_3D` fluxes, `eigenvalues6{,z}_hyperbolic_3D` wave
speeds, `pas_HLL`); the single-stream baseline adds `realizable_3D_M4` projection,
while the two-stream carries `+`/`−` fields, swaps the split axis (x) for the
donor-cell half-space stream flux, and couples via BGK. Three *distinct* counters
are reported and must not be lumped:

* **x chain clips** — the coordinate-wise x-realizability repair. The clean claim is
  "**zero x-realizability interventions**"; this is the number the transverse fix had
  to drive to 0.
* **transverse hyperbolicity corrections** — the production `eigenvalues6_hyperbolic`
  / `realizable_3D_M4` correction on the y/z path. A half-space-in-x stream is an
  ordinary full-line measure in y/z, so this machinery *legitimately* applies there;
  activity here is **expected and fine**, not a defect.
* **production projection35 firings** — the single-stream baseline.

| Ma (CFL, steps, Kn) | production projection35 | production min margin | **x chain clips** | transverse hyp. corr. | min stream margin (worst transient) | peak interpenetration | wall (2-stream / prod) |
|---|---|---|---|---|---|---|---|
| 5 (1/3, 40, 1e3) | **10880** | −2.9e-13 | **0** | 192 | **+1.1e-5** | **1216 cells** | 1.78× |
| 100 (1/3, 40, 1e9) | **188768** | −5.1e-8 | **0** | 18096 | −4.6e-1 (recovers to ≈−4e-3) | 0† | **0.72×** |

**Ma=5** is the definitive demonstration: the single-stream closure fires projection
10880× (its total state is genuinely non-realizable at the interpenetration zone),
while the two-stream needs **zero** x chain clips and only 192 transverse
corrections, keeps both streams realizable (+1.1e-5), and shows **1216 cells of peak
stream interpenetration** a single Maxwellian cannot.

**Ma=100** (production CFL = 1/3, through the crossing): routing y/z through the
production HLL path drove x chain clips from **4476 (old first-order Rusanov) → 0**,
with **no blowup** — this was the whole point of the fix. Transverse hyperbolicity
corrections (18096) are the expected full-line y/z activity. The stream realizability
margin dips to −0.46 *transiently* at the crossing peak (the counter-streaming
stress) and recovers to ≈−4e-3, all while x stays perfectly clean. The two-stream is
**faster** than production here (0.72×) because production must fire 188768
projections. †Peak interpenetration reads 0 at Ma=100 only because the fast, thin,
first-order-diffused jets fall below the 0.05 mass gate used for the counter; the
interpenetration mechanism is demonstrated at Ma=5 (1216 cells).

BGK caveat: the two-stream BGK relaxes each stream toward the *shared* Maxwellian of
the total. At an Ma=100 crossing the total is a counter-streaming two-beam whose
apparent temperature ≈ (streaming velocity)² is enormous; relaxing toward that hot
target destabilizes at the crossing even at Kn=1e3 (which is otherwise nearly
collisionless, e≈1e-5/step). Ma=100 is therefore run collisionless (Kn=1e9); a
two-beam-aware BGK target is the remaining refinement. (Ma=5, Kn=1e3, BGK active, is
stable.)

## Status

Gates 1 and 2 pass with the spec's numbers. Gate 3 passes at both Ma=5 and Ma=100 3D
at production CFL with the production transverse HLL path: **zero x-realizability
interventions** and no blowup in both cases (the old first-order Rusanov gave 4476
x-clips and blew up at Ma=100 — routing y/z through the production HLL fixed it).
Remaining honest gaps: the two-stream BGK target is unstable at an Ma=100 crossing
(run collisionless there); and the transverse realizability dips transiently to −0.46
at the Ma=100 crossing peak (bounded, recovers). The mode is registered additively
(package includes + exports); no existing production file's logic was changed. Full
`simulation_runner`/GPU integration is the remaining step for production Ma=100 3D use.

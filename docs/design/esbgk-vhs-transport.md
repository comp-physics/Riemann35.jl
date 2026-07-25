# ES-BGK collision + VHS viscosity law — design

**Date:** 2026-07-24
**Status:** approved design, pending implementation
**Scope:** replace the single-relaxation BGK collision operator with an opt-in
ellipsoidal-statistical (ES-BGK) operator carrying a correct Prandtl number, and
generalize the relaxation time to a variable-hard-sphere (VHS) viscosity law.

---

## 1. Motivation

The objective this serves is making 35-moment HyQMOM an accurate and fast
alternative to DSMC for non-equilibrium flows. Two transport mismatches block
quantitative comparison against DSMC, and both live in one function.

**Prandtl number.** `collision35` / `bgk_relax_tup` relax all 35 moments at a
single rate toward an *isotropic* Maxwellian. That pins `Pr = 1`; a monatomic
dilute gas is `Pr = 2/3`. Viscosity and thermal conductivity cannot both match
DSMC, which is first-order in any case where heat transfer matters (shock
structure, micro-channel heat transfer).

**Viscosity temperature exponent.** The current `tc = Kn/(2*rho*sqrt(Theta))`
gives `mu = p*tc = (Kn/2)*sqrt(Theta)`, i.e. `mu ~ T^0.5` — hard spheres, VHS
with `omega = 0.5`. DSMC for argon typically runs `omega ~ 0.74`. This is an
independent mismatch from Pr.

Both are fixed here. The method remains entirely moment-based: the ES target is
an anisotropic Gaussian whose 35 moments are closed-form in its covariance
(Isserlis), so there is no velocity grid, no distribution reconstruction, and no
particles anywhere in the change.

### Why ES-BGK and not Shakhov

Both fix Pr. Shakhov adds a heat-flux correction to the Maxwellian, producing a
target that is **not a positive distribution** in general. That breaks two
properties this solver is built on: the target is no longer realizable by
construction, and the update is no longer a convex combination of realizable
states. ES-BGK's target is a genuine Gaussian and preserves both (§5). ES-BGK
also has a proven H-theorem (Andries, Le Tallec, Perlat & Perthame, 2000).

---

## 2. Scope

**In scope.** ES target + VHS exponent in the single-source device collision
(`bgk_relax_tup`) and the legacy `collision35`; parameter threading through
`simulation_runner` and `gpu_run`; transport-coefficient extraction tests; 1D
shock-structure validation.

**Non-goals.** Kinetic wall boundary conditions (needed for Couette/Fourier
validation; separate work). Shakhov. Multi-species / mixtures. Internal degrees
of freedom (polyatomic). Changing any default: `Pr = 1.0`, `omega = 0.5`
reproduce current behavior bitwise, and the `:recommended` bundle is unchanged.

---

## 3. Parameterization

| param | default | range | meaning |
|---|---|---|---|
| `Pr` | `1.0` | `[2/3, 1]` | Prandtl number. ES parameter `nu = 1 - 1/Pr`; `Pr=2/3` gives `nu=-1/2`. |
| `omega` | `0.5` | `[0.5, 1.0]` | VHS viscosity exponent, `mu ~ T^omega`. `0.5` hard sphere, `~0.74` argon, `1.0` Maxwell molecules. |

Both are argument-checked with a clear error; out-of-range values raise
`ArgumentError` (matching the `riemann_solver` precedent).

`Pr` is exposed rather than `nu` because it is the physically meaningful
quantity and the one a DSMC comparison is specified in. The supported upper
bound is `Pr = 1` (i.e. `nu <= 0`); `Pr > 1` is admissible in ES-BGK theory
(`nu` up to 1) but introduces an `expm1` overflow path in §4 and has no use case
here. Extending the range requires revisiting that guard.

### Reference relaxation time

```
tau_ref = (Kn/2) * Theta^(omega - 1) / rho
```

which collapses to the current `Kn/(2*rho*sqrt(Theta))` at `omega = 0.5`.

Defining `tau_ref` this way makes `mu = p * tau_ref` **independent of Pr**. This
is load-bearing: in ES-BGK the deviatoric stress relaxes at `(1-nu)/tau` while
heat flux relaxes at `1/tau`, so naively swapping the target at fixed `tau`
would change viscosity by a factor of `Pr` as a side effect of changing the
Prandtl number. Anchoring on `tau_ref` decouples them — `omega` sets `mu(T)`,
`Pr` sets the ratio, neither disturbs the other.

---

## 4. Time integration

### The problem

The current operator is exact-exponential, and that is true for BGK *only*
because the Maxwellian target is built from `(rho, u, Theta)`, all conserved by
collision. The target is constant over the step, so `M^{n+1} = (1-e)*MG + e*M^n`
is the exact solution.

The ES target is **not** constant: it depends on the full covariance, and the
deviatoric stress is precisely what collision relaxes. A frozen-target
exponential fails in the collisional limit — as `dt -> inf` it converges to
`G[Lambda(C^n)]`, leaving a spurious residual stress `nu*sigma^n` instead of
relaxing to the Maxwellian.

### The scheme

With `y = dt / tau_ref`:

```
a     = exp(-y)                 # deviatoric stress decay, Kn-defined, Pr-independent
e     = exp(-Pr*y)              # non-conserved moment decay
kappa = (a - e) / (1 - e)
Lambda = (1 - kappa)*Theta*delta + kappa*C
M^{n+1} = (1 - e)*G[Lambda] + e*M^n
```

where `C` is the full 3x3 covariance and `G[Lambda]` is the 35-moment vector of
the Gaussian with covariance `Lambda` (§5).

Choosing `kappa` this way makes the update **exact in the second moments**: the
deviatoric stress decays as `exp(-y)`, at the Kn-defined rate, independent of
`Pr`. Both asymptotic limits are correct:

- `dt -> 0`: `kappa -> nu`, recovering the standard ES target
  `Lambda = (1-nu)*Theta*delta + nu*C`.
- `dt -> inf`: `kappa -> 0`, `Lambda -> Theta*delta`, so `M^{n+1} -> Maxwellian`
  with no residual stress.

Derivation: requiring the convex combination to land on the exact second-moment
solution `C(dt) = Theta*delta + (C^0 - Theta*delta)*a` gives
`(1-e)*Lambda + e*C^0 = C(dt)`; solving for `Lambda` with `D^0 = C^0 -
Theta*delta` (traceless, since `tr C^0 = 3*Theta`) yields
`Lambda = Theta*delta + [(a-e)/(1-e)] * D^0`.

Cost over the current operator: one extra `exp`, one division, three extra
moment reads.

### Numerical evaluation of kappa

`(a-e)/(1-e)` is `0/0` as `dt -> 0`. Evaluate instead as

```
kappa = -exp(-Pr*y) * expm1((Pr-1)*y) / expm1(-Pr*y)
```

which is stable across the whole range of `y`. `expm1` is available on device
(CUDA.jl intrinsic).

At `Pr = 1`: `expm1(0) = 0`, so `kappa = 0` **exactly**, `Lambda = Theta*delta`,
and the update reduces to the current formula bitwise.

---

## 5. Target construction

`G[Lambda]` needs the standardized moments of a *correlated* Gaussian. Write
`Lambda_ii` for the diagonal and

```
r1 = Lambda_xy / sqrt(Lambda_xx * Lambda_yy)     # -> S110
r2 = Lambda_xz / sqrt(Lambda_xx * Lambda_zz)     # -> S101
r3 = Lambda_yz / sqrt(Lambda_yy * Lambda_zz)     # -> S011
```

Note `Lambda_xy = kappa * C110` exactly, since the isotropic part contributes no
off-diagonal.

Third-order standardized moments are all zero (Gaussian). Fourth-order, by
Isserlis:

| moment | value | | moment | value |
|---|---|---|---|---|
| `S400` | `3` | | `S130` | `3*r1` |
| `S310` | `3*r1` | | `S121` | `r2 + 2*r1*r3` |
| `S301` | `3*r2` | | `S112` | `r1 + 2*r2*r3` |
| `S220` | `1 + 2*r1^2` | | `S103` | `3*r2` |
| `S211` | `r3 + 2*r1*r2` | | `S040` | `3` |
| `S202` | `1 + 2*r2^2` | | `S031` | `3*r3` |
| `S022` | `1 + 2*r3^2` | | `S013` | `3*r3` |
| `S004` | `3` | | | |

No eigendecomposition; all closed-form. At `r1=r2=r3=0` these collapse
**exactly** to the literals currently hardcoded in `bgk_relax_tup`
(`S400=S040=S004=3`, `S220=S202=S022=1`, rest `0`), which is what makes the
`Pr=1` fast path bitwise identical rather than merely close.

These feed `from_recon_vars_dev` directly — it already takes diagonal variances
plus standardized correlations in exactly this layout. The three off-diagonal
covariances require three extra reads beyond what `bgk_relax_tup` computes
today:

```
C110 = M[7]/rho  - u*v
C101 = M[17]/rho - u*w
C011 = M[26]/rho - v*w
```

### Trap: do not use `InitializeM4_35`

`InitializeM4_35` hardcodes the *independent*-Gaussian standardized moments
(`S220=1`, `S310=0`, ...) while accepting a full covariance including
off-diagonals. It is therefore a true Gaussian **only when the covariance is
diagonal**. This is harmless today — every IC passes `r110=r101=r011=0` (see
`src/initial_conditions.jl:59,105,154`) — but routing an anisotropic ES target
through it would silently produce a non-Gaussian target with the wrong
fourth-order moments. Build the target via `from_recon_vars_dev` with the
Isserlis values above.

Add a comment to `InitializeM4_35` recording this limitation.

---

## 6. Realizability

Preserved structurally; this is the central reason for choosing ES-BGK.

`Lambda = (1-kappa)*Theta*delta + kappa*C` with `kappa` in `[-1/2, 0]` has
eigenvalues `(1-kappa)*Theta + kappa*c_i`, where `c_i` are the eigenvalues of
`C`. For PSD `C` these are strictly positive: the tightest case `kappa = -1/2`
gives `(3/2)*Theta - (1/2)*c_i`, which is positive unless `c_i = tr(C) = 3*Theta`
— impossible for positive-definite `C`, and merely degenerate (zero) for
singular `C`.

So `G[Lambda]` is the moment set of a genuine Gaussian, hence realizable, and

```
M^{n+1} = (1-e)*G[Lambda] + e*M^n
```

is a **convex combination of two realizable states** — the same argument the
current `bgk_relax_tup` comment relies on, unchanged. `tr(Lambda) = 3*Theta` is
preserved, so energy conservation is untouched and mass/momentum are unaffected
by construction.

### Guard

The guard is numerical only, against covariances that are marginally
non-PSD from cancellation (the existing code already floors `Theta` at `1e-14`
for the same reason).

Test `Lambda` for positive-definiteness by Sylvester's criterion (three leading
minors — device-safe, no eigensolve). On failure, set `kappa = 0` outright,
falling back to the isotropic Maxwellian target for that cell and step.

`Lambda(kappa)` is affine in `kappa` and positive-definite at `kappa = 0`
(`Lambda = Theta*delta`, `Theta > 0` by the existing floor), so this always
succeeds. A partial retreat (bisection for the largest admissible `kappa`, the
theta-limiter idiom of `idp_limiter_dev.jl`) is deliberately **not** used: by
§6 the guard is unreachable for any genuinely PSD `C`, so it fires only on
states that are already numerically corrupt, where recovering a fraction of the
ES correction has no value and the branch-free fallback is cheaper on device.

Count retreat activations behind the existing projection-counter pattern
(`reset_proj_counter!` / `proj_correction_count`) — a nonzero count away from
vacuum is a diagnostic that something upstream is wrong.

---

## 7. Integration points

`bgk_relax_tup` remains the single source and gains two **positional** arguments:

```julia
bgk_relax_tup(M, dt, Kn)                    # 3-arg forwarder -> (…, 1.0, 0.5)
bgk_relax_tup(M, dt, Kn, Pr, omega)         # 5-arg
```

> **Correction (2026-07-24, found in implementation).** This section originally
> specified `Pr`/`omega` as *keyword* arguments. That does not compile for the GPU:
> the keyword-argument sorter appears in device code as dynamic `getindex` /
> `convert` / `jl_f_tuple` calls and the kernel fails with `InvalidIRError`.
> They must be positional. The 3-argument forwarder preserves every existing call
> site, so nothing else changed.

> **Correction (2026-07-24, found in implementation).** The ES branch must live in
> its own `@noinline` method (`_esbgk_relax_tup`), not inline inside
> `bgk_relax_tup`. With both branches in one body the method exceeded what Julia
> could infer for the GPU and the kernel failed to compile — **including when the ES
> branch was unreachable**, because dead code is still inferred. Bisection confirmed
> it: deleting the ES body fixed compilation, while merely early-returning before it
> did not. Splitting the method fixes it and leaves the default path byte-identical.
> This is the same `@noinline`-as-a-tool practice already documented at the top of
> `recon_dev.jl`, used there for FP parity rather than compilability.

Call sites:

| site | path |
|---|---|
| `src/numerics/highorder_3d.jl:583` | CPU stage-BGK (`stage_bgk`) |
| `gpu/timestep3d_gpu.jl:97` | GPU order-2 `_bgk_kernel!` |
| `gpu/timestep3d_order3_gpu.jl:356` | GPU order-3 |
| `src/numerics/collision35.jl` | legacy post-step operator |

`highorder_3d.jl` threads `Pr`/`omega` alongside the existing `stage_bgk_kn`.
`simulation_runner` reads them from params; `gpu_run.jl` takes them as kwargs
next to `Kn`.

### Byte-identity contract

The branch `Pr == 1.0 && omega == 0.5` takes the **verbatim current code path**.
This is deliberate belt-and-braces: the algebra provably collapses (§4, §5), but
an explicit branch also avoids two floating-point traps that the collapse alone
does not.

1. `Lambda = (1-kappa)*Theta + kappa*C200` with `kappa = 0.0` evaluates
   `0.0 * C200`, which is `NaN` when `C200` is `Inf`. Deep-vacuum states in this
   solver have produced non-finite intermediates before (see
   `docs/ma100-highorder-crash-analysis.md`).
2. `Theta^(omega-1)` at `omega = 0.5` is `Theta^(-0.5)`, which is **not**
   bitwise equal to the current `1/sqrt(Theta)`.

The same `@noinline` / `@fastmath` discipline documented at the top of
`recon_dev.jl` applies to any new helper: do not inline the fastmath blocks, for
the reasons recorded there.

---

## 8. Tests

### Must not move (the default-path acceptance criterion)

- `test/test_golden_files.jl:300` — `collision35` golden, bitwise.
- `test/test_rodney_cases.jl` — including the machine-exact stationary contact.
- GPU byte-identity goldens and the multi-GPU bit-identical timestep.

No existing golden changes. If one does, the byte-identity fast path is wrong.

### New

**Unit.**
- Isserlis target vs an independently computed correlated-Gaussian 35-moment set,
  at several correlation strengths.
- `kappa` limits: exactly `0.0` at `Pr=1`; `-> nu` as `dt -> 0`; `-> 0` as
  `dt -> inf`; stability of the `expm1` form across `y` spanning many decades.
- Second-moment exactness: deviatoric stress after one step matches
  `sigma^0 * exp(-y)` to machine precision.
- PD guard on adversarial near-degenerate and marginally non-PSD covariances;
  retreat always terminates with a realizable target.
- Conservation: mass, momentum, and `tr(P)` unchanged by the operator to machine
  precision at arbitrary `Pr`, `omega`.

**Parity.** CPU vs GPU in ES mode (`Pr=2/3`, `omega=0.74`), to the existing
residual tolerance; multi-GPU vs single-GPU bit-identity retained.

**Transport coefficients (the CI gate).** Extract `mu` and `k` numerically from
decay rates of small shear and thermal perturbations on a periodic domain, and
assert `Pr = 2/3` and the `omega` exponent to a calibrated tolerance. This is
the test that proves the operator does what it claims, and it is cheap,
deterministic, and wall-free.

**Physics validation.** 1D shock structure, Ma 1.2-8, against Alsmeyer's density
profiles and published DSMC. Wall-free, one-dimensional, and the canonical
rarefied benchmark — this is the result that supports a DSMC-alternative claim.
Expect the `Pr=1` baseline to show the known BGK density-thickness error and the
ES run to improve it.

---

## 9. Risks

- **Second-moment exactness rests on the `kappa` derivation.** If it is wrong the
  error is subtle (correct limits, wrong intermediate rate). The
  stress-decay unit test above is the direct check and should be written first.
- **`expm1` on device.** Verify CUDA.jl lowers it to the intrinsic rather than a
  slow path; if not, a guarded series expansion for small `y` is the fallback.
- **Shock-structure comparison needs a reference dataset.** Alsmeyer's profiles
  are digitized in several papers; source them before committing to the
  benchmark, or fall back to published DSMC profiles.
- **`omega != 0.5` changes the CFL-relevant timescale** at fixed `Kn`. Check the
  timestep controller does not assume the hard-sphere form anywhere.

---

## 10. References

- Holway, "New statistical models for kinetic theory: methods of construction,"
  Phys. Fluids 9, 1658 (1966) — the ES-BGK model.
- Andries, Le Tallec, Perlat & Perthame, "The Gaussian-BGK model of Boltzmann
  equation with small Prandtl number," Eur. J. Mech. B 19, 813 (2000) —
  H-theorem and transport coefficients.
- Bird, *Molecular Gas Dynamics and the Direct Simulation of Gas Flows* — VHS
  model and `omega` values.
- Alsmeyer, "Density profiles in argon and nitrogen shock waves measured by the
  absorption of an electron beam," J. Fluid Mech. 74, 497 (1976).

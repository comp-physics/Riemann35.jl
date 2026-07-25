# `stage_bgk` over-relaxation: an O(1) error in the transport coefficients

**Date:** 2026-07-25
**Status:** measured; fix implemented as `stage_bgk_exact` (OPT-IN, default `false`)
**Affects:** `scheme = :recommended` (which enables `stage_bgk`) at **finite Kn**

---

## Summary

`stage_bgk` applies the BGK/ES-BGK collision with the **full `dt` after each of the three
SSP-RK3 stages**. The convex combinations make the composite relax the deviatoric stress
at `(11/6)/tau` instead of `1/tau`.

**Consequence: with `stage_bgk` on, the effective viscosity and thermal conductivity are
each ~1.85x smaller than the collision operator specifies.**

The Prandtl number is *unaffected* — `mu` and `k` are scaled by the same factor, so it
cancels. That is precisely why this survived undetected: the operator-level transport
gate (`test/validate_transport_coefficients.jl`) measures relaxation times homogeneously
and verifies `Pr` and `omega` to ~1e-14, and every ratio-based check is blind to a common
factor. It is visible only in a **spatial** measurement of an absolute rate.

## The mechanism

With transport off (`L = 0`) and a per-stage relaxation factor `s`, SSP-RK3's convex
combinations give

```
u1 = s*u0
u2 = s*(3/4 + s/4)*u0
u3 = s*(1/3 + (2/3)*s*(3/4 + s/4))*u0
```

so the one-step composite factor is

```
F(s) = s/3 + s^2/2 + s^3/6 = s(s+1)(s+2)/6
```

`F(0) = 0`, `F(1) = 1`, and — the key number — **`F'(1) = 11/6 = 1.8333`**. Expanding at
`s = 1-x`, `x = dt/tau`, gives `F = 1 - (11/6)x`: three full-`dt` applications relax the
stress 11/6 times too fast.

The local error is `O(dt)` **in the solution**, hence `O(1)` in the transport
coefficients. This is *not* the first-order splitting consistency the original
`stage_bgk` comments claimed.

## Measurements

Shear-wave decay on a periodic domain through the production order-3 stepper, fitting
`u_y ~ exp(-nu k^2 t)` and comparing against `nu = Theta*tau_ref`.

**Ratio measured/theory, `stage_bgk` on**, versus `k*lambda` (Nx=80, order 3):

| k*lambda | 0.3142 | 0.1571 | 0.0785 | 0.0393 |
|---|---|---|---|---|
| shear | 0.5120 | 0.5295 | 0.5305 | 0.5210 |
| entropy | 0.8043 | 0.6045 | 0.5358 | 0.5301 |

Flat in `k*lambda` for shear; entropy converges to the same value. **Both modes converge
to ~0.53, not 1** — which is why `Pr` came out correct while the magnitudes did not.

**Controls** (Kn=0.25, Nx=80, `k*lambda = 0.0785`):

| collision | ratio |
|---|---|
| per RK stage (shipped default) | **0.5305** |
| once per step | **0.9836** |

`stage/once = 1.8542` against the predicted `11/6 = 1.8333` (1.1%), and once-per-step
recovers Navier-Stokes to 1.6%. Predicted ratio `6/11 = 0.5455` vs measured 0.53.

**Ruled out as causes:**
- *Discretization.* Grid-converged: Nx=40 -> 80 moves the rate by 0.1%, slope ~5e-4.
  Numerical viscosity would also push the ratio **above** 1, not below.
- *Finite Knudsen.* The ratio is flat across an 8x range of `k*lambda`.
- *One implementation.* CPU `step_highorder_3d!` and GPU `march3d_order3_gpu!` agree on
  the shear ratio to 0.3% (0.5124 vs 0.5107).

## The fix: `stage_bgk_exact`

Solve for the per-stage factor `s` whose composite is the exact one-step relaxation:

```
s(s+1)(s+2) = 6*E ,      E = exp(-dt/tau)
```

`F` is monotone on `[0,1]`, so Newton from `s0 = min(1, 3E)` converges in a few
iterations; `_rk3_stage_factor` in `src/numerics/recon_dev.jl` inverts it to 3.2e-16 over
`E` from 1e-12 to 1.

`tau` is **per-cell** (it depends on `rho` and `Theta`), so the correction must live
inside the per-cell operator; a caller-side `dt` rescale cannot express it.

For the ES-BGK path **both** rates are corrected (`a` for the deviatoric stress, `e` for
the rest) and `kappa` is then built from the *stage* values, so each stage stays
second-moment-exact and the composite lands on the target deviatoric factor.

Verified in isolation — three corrected stages compose to exactly one plain step:

| | 3 stages, off | 3 stages, on | target (1 step) |
|---|---|---|---|
| BGK | 0.39553354 | **0.43904772** | **0.43904772** |
| ES | 0.39553354 | **0.43904772** | **0.43904772** |

### Scope of the fix

Exact for the **relaxation**. Interleaved transport still carries the usual `O(dt)`
Lie-splitting error. This removes an `O(1)` error in the transport coefficients; it does
not make the operator split exact.

## Usage and the default

```julia
simulation_runner(merge(params, (stage_bgk_exact = true,)))     # CPU
march3d_order3_gpu!(G, dx, Ma, n; stage_bgk = true, stage_bgk_exact = true, …)   # GPU
```

**Default is `false`, so nothing shipped changes and no golden moves.** Enabling it is
*not* byte-identical — it is a physics fix, and results at finite Kn will differ.

**Read this before trusting a transport number.** With defaults (`scheme = :recommended`,
`stage_bgk` on), `mu` and `k` are ~1.85x smaller than the operator specifies. `Pr` and the
viscosity exponent `omega` are correct either way. So:

- **Turn it on** for anything where an absolute transport coefficient matters: viscous or
  thermal-conduction problems, shock structure, Couette/Fourier, comparison against DSMC
  or experiment.
- **Leaving it off** is fine for `Kn = 0` (everything Maxwellianizes regardless, which is
  why the 2026-07-02 graduation study's stationary-contact result is unaffected) and for
  reproducing pre-2026-07-25 results bit-for-bit.

### Why `stage_bgk` exists at all

It was adopted because `ho_pressure_recon + stage_bgk` makes the uniform-pressure
stationary contact machine-exact (3.2e-16 vs 6.4e-2 legacy; see
`docs/design/scheme-graduation.md`). That benefit is real and is **at Kn = 0**, where
over-relaxation is harmless. The transport damage appears at **finite Kn**. The two
effects live in different regimes, which is why the trade went unnoticed — and why
`stage_bgk_exact` is the right resolution: it keeps the contact benefit and removes the
transport error.

# Minimal-norm hyperbolicity correction (opt-in)

**Date:** 2026-07-22. **Status:** opt-in prototype (default off, byte-identical);
PR #22. **Origin:** a suggestion by Rodney Fox (email 2026-07-22).

An alternative to the shipped "blunt reset" hyperbolicity correction. It restores
hyperbolicity with a much smaller, more faithful change to the moment state, at
higher coverage — but as a prototype it is slower and allocation-heavy, so it is
opt-in and the default path is untouched.

## The problem it solves

When the 35-moment 3D flux Jacobian's wave-speed check
(`eigenvalues6_hyperbolic_3D`) finds a **complex** eigenvalue in one of the six
axis-plane `3x3` blocks (rows `13:15` = `m03,m13,m04`), the system is locally
non-hyperbolic and the state must be nudged back onto the hyperbolic set.

The shipped correction (`correct_moments_dev`, "blunt reset") **zeros all six
third-order cross standardized moments** `S210,S120,S201,S102,S021,S012` and floors
`S220,S202,S022` at `1/3`. That is robust but discards the state's entire
third-order correlation structure whenever any one flux direction goes complex.

## The idea (eq (41)/(43)/(45))

In the preprint's 2D notation (Appendix B), the coupling term that spoils the
block factorization is
```
(41):  s31 - [ s11*s40 + (3/2)*s30*(s21 - s11*s30) ]
```
and eq (43)'s closure for `s31` is exactly the bracket. So `s310` (a 4th-order
moment) is a **second lever** on hyperbolicity, alongside the 3rd-order `s210` the
blunt reset uses. This correction minimizes over **both**: the six `S210`-perms and
the six `S310`-perms (the eq-(43) axis).

Two facts drove the design, both verified numerically:

- **`s310` alone cannot restore hyperbolicity for a hard core of states.** Where
  `s310 <- (43)` (keeping it as a variable) fails, *no* value of `s310` anywhere
  restores reality (0/500), while adjusting `s210` fixes 92%. So `s310` (4th order)
  modulates the block but cannot override a bad 3rd-order configuration. It is an
  *added axis*, not a replacement.
- The always-real guarantee of eq (45) (`discriminant 4*s11^2 + s12^2 >= 0`)
  belongs to the 13-moment **closure** (drop `s31,s13` as variables), not to
  overwriting the evolved `s31`.

## What the correction does

`correct_moments_minnorm` (`src/numerics/moment_correction_minnorm.jl`):

1. Standardize the 35-moment vector (`M4toC4_3D` -> divide by `sigma` powers).
2. Floor `S220,S202,S022` at `1/3`.
3. **Joint KKT projection** of the 12 adjustable moments (six `S210`-perms + six
   `S310`-perms) onto the block real-rootedness boundary — minimize the weighted
   norm `w3*|dS210...| + w4*|dS310...|` subject to every axis-plane block's cubic
   discriminant being `>= margin`. The step is
   `dx = Winv A^T (A Winv A^T)^{-1} (margin - d)` over the violated constraints,
   Newton-iterated with a damped line search.
4. **Targeted fallback**: any plane still firing in the lab frame gets its two
   third-order cross moments zeroed (production-style, but only for that plane).
5. Rebuild raw moments (`C4toM4_3D`). Conserved (`<= 2nd-order`) moments are exact.

Two 3D subtleties that the 2D prototype did not have, both handled:

- **Coupled constraints.** The six plane blocks *share* moments (`S210` is plane
  UV's `s21` and plane VU's `s12`), so the six discriminant constraints are solved
  jointly, not per-plane.
- **Frame dependence + boundary roundoff.** The production check is the sub-block
  in the lab (mean-shifted) frame, which is frame-dependent, and driving the
  discriminant to *exactly* zero fails on floating-point roundoff. Both are handled
  by targeting `discriminant >= margin` (default `1e-3`) rather than `>= 0`.

## Using it (opt-in)

```julia
using Riemann35
Riemann35.HYP_CORRECTION[] = :minnorm   # default is :blunt
# ... run the CPU low-order solver as usual; the correction is used in the
#     wave-speed / flux stage of simulation_runner.
Riemann35.HYP_FIRE_COUNT[]              # diagnostic: # of correction firings
```

From the crossing-jets example runner:

```bash
HYP_CORRECTION=minnorm julia --project=. examples/run_3d_custom_jets.jl \
    --config crossing --Ma 100.0 --order 1 --nnmax 30 --no-viz true
```

The default (`HYP_CORRECTION[] === :blunt`) leaves
`correct_moments_hyperbolic_3D` **byte-identical** to `correct_moments_dev`
(verified `maxdiff = 0.0`; full `Pkg.test` passes 1252 / 0 fail).

## Measured (random-realizable firing states, `3x3`-block criterion)

| correction | coverage | 3rd+-order moment change |
|---|---|---|
| minimal-norm | **97.5%** | **~0.5 (8.8x gentler)** |
| blunt reset (shipped) | 83.5% | ~4.6 |

Conserved moments exact (drift `~1e-16`).

Real **Ma=100 order-1 crossing-jets** run (via the exact production correction
chain + `Flux_closure35_3D`):

| metric | blunt | minimal-norm |
|---|---|---|
| correction magnitude `\|\|Mr-M\|\|` | 1.55 | **0.49** (3.2x gentler) |
| flux perturbation `\|\|F(Mr)-F(M)\|\|` | 6.58 | **2.28** (2.9x more faithful) |

Since the correction only feeds the flux, a smaller `||Mr-M||` means a flux closer
to the uncorrected ideal — more faithful dynamics. Over a 30-step run the choice of
correction changes the solution ~15% relative. Trade-off found: minimal-norm
**fires more often** (gentler corrections keep states near the boundary, so they
re-fire; the blunt reset flings them far away).

## Limitations / future work

- **Prototype-grade**: `Dict`-based, allocates; ~2.7x slower per step. An
  allocation-free rewrite is required before this could be a default rather than an
  opt-in.
- **Coverage is ~97.5%, not 100%.** The residual few percent that the per-plane
  projection cannot reach is exactly where the **13-moment closure** (drop the odd
  4th-order moments as variables) gives a complete guarantee — verified separately
  to have a fully real spectrum on all realizable states. That closure is a larger
  structural change (a moment-set reduction), tracked as a separate follow-up.
- "More faithful flux" is not the same as proven "more accurate" without a
  converged / DVM-BGK reference; a `26`-vs-`35`-moment-vs-DVM accuracy study is the
  natural next check.

## References

- Preprint eqs (41), (42), (43), (45); Appendix B (`~/main_unmarked.pdf`,
  Fox/Laurent).
- `src/numerics/moment_correction_minnorm.jl` — the implementation.
- `src/numerics/moment_correction_dev.jl` — the shipped blunt reset it alternates
  with.
- Analysis scripts (not in-repo): `scratch/scripts/idea2_*.jl`, `stdmargin.jl`,
  `idea3_j13.jl` (the 13-moment-closure global-hyperbolicity check).

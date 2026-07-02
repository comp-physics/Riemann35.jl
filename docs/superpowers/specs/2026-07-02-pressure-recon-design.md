# Pressure-tensor reconstruction variables (`ho_pressure_recon`) — design + plan

**Date:** 2026-07-02 · **Status:** approved (conversation) · **Branch:** `feat/primitive-reconstruction`

## Problem

The order-2 stationary-contact gate (PR #8, `test/test_rodney_cases.jl`) shows
~4–7% spurious velocity / pressure L∞ error at the 1000:1 uniform-pressure
contact, roughly resolution-independent (maxvel 0.037/0.064/0.070 at
Nx=32/64/128, t=0.05). First-order preserves the contact to machine precision.

## Root cause

The high-order path already reconstructs in bounded variables
`V = [ρ, u, v, w, C200, C020, C002, 28 standardized moments]`
(`to_recon_vars`/`from_recon_vars`, Posey/Fox recipe). Across the contact, ρ
and the temperature-like variances C2ii **both jump**; independently limited
slopes make the face pressure ρ_f·T_f oscillate even though u and p are
uniform. Classic pressure-oscillation mechanism, expressed in these variables.

## Fix (opt-in, default byte-identical)

Variant bijection where slots 5–7 hold the **pressure-tensor diagonal
P_ii = ρ·C2ii** instead of C2ii. At the uniform-p contact every recon var
except ρ is then uniform (u=0, P_ii=p, standardized moments Maxwellian on both
sides) ⇒ all slopes vanish ⇒ face states differ only in ρ ⇒ the same flux
cancellation that makes first-order exact applies at second order.

**Measured outcome (2026-07-02):** maxvel 0.064 → 0.0049 (13×), pdev
0.039 → 0.014 (2.7×) at Nx=64. NOT machine-exact: a second, independent error
channel remains — SSP-RK3 stages are collisionless (BGK is applied once per
full step), so at Kn=0 stage 1 pumps transient M300 from the M400 = 3p²/ρ
variation across the contact and later stages flux it into pressure; the
resulting velocity error persists (collision preserves u) and saturates with
resolution (0.0027/0.0049/0.0060 at Nx=32/64/128; dt/dx fixed by CFL ⇒ no
decay). First-order is exact because it is single-stage: collision resets M300
between every flux evaluation. **Follow-up (separate opt-in): apply the
exact-exponential BGK per RK stage — predicted machine-exact, testable with
the same gate.** The reconstruction channel itself is confirmed eliminated.

## Implementation (CPU path only; GPU `_dev` kernels untouched — follow-up)

1. `const HO_PRESSURE_RECON = Ref(false)` next to `HO_VACUUM_FLOOR`
   (src/Riemann35.jl:27).
2. Branch in the two CPU wrappers (src/numerics/reconstruction.jl:116-127),
   the single choke point for all CPU call sites (`residual_line`,
   `residual_1d`, `recon_face_pair`, `recon_faces_limited`,
   `scaling_limited_faces`):
   - `to_recon_vars`: after the existing kernel, if flag: `V[5:7] .*= V[1]`.
   - `from_recon_vars`: if flag: copy V, `W[5:7] ./= W[1]`, then existing kernel.
   Flag off ⇒ identical code path (no FP change ⇒ byte-identical; goldens gate).
   `recon_vars_ok` needs no change: P_ii>0 ⟺ C2ii>0 given ρ>0 (checked).
3. `HO_PRESSURE_RECON[] = get(params, :ho_pressure_recon, false)` in
   simulation_runner's opt-in block, with doc comment following the
   `ho_realizability_limiter` pattern.
4. Docstring update on the bijection + note in examples/README.md (1D case).

## Tests (TDD; test/test_rodney_cases.jl)

- **Acceptance:** order-2 contact with `ho_pressure_recon=true` ⇒ maxvel and
  pdev < 1e-12 (same ceilings as the order-1 exact gate). Flag reset after.
- **Default-unchanged:** flag-off run bitwise-equal to a run that never set
  the param (and existing goldens/CI cover the default path globally).
- **Robustness smoke:** crossing-jets (`:crossing_matlab`, params of
  test_highorder_3d) short run with flag on: finite, positive ρ, mass drift
  small — guards the ρ-division near the rhor=0.001 background (vacuum-floor
  fallback path unchanged).

## Non-goals

GPU port (`to/from_recon_vars_dev` single-source kernels stay byte-identical —
the @noinline/@fastmath parity rules make that a deliberate separate change);
characteristic reconstruction; changing the default.

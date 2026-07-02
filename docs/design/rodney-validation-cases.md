# Rodney's uniform-pressure validation cases (1D contact + 2D dense bubble)

**Date:** 2026-07-02
**Status:** approved design, pending implementation
**Motivation:** Rodney Fox's proposed validation ladder for the 35-moment model
(email 2026-07-02): uniform-pressure ICs whose Kn=0 Euler solution is trivial
(stationary contact / passive dense bubble), so that (a) the Riemann solver can be
verified against an exact solution and (b) Kn>0 runs isolate non-equilibrium heat
flux for later comparison against kinetic references (DVM-BGK / DSMC).

## Cases (Rodney's definitions, reference state m0=1, T=1, p=1)

- **1D Riemann:** left = reference (rho=1, u=0, T=1); right = dense cold gas
  (rho=1000, u=0, T=1/1000). Uniform p=1. Kn=0 exact solution: stationary
  contact — *nothing moves*; any velocity or pressure deviation is pure numerical
  error.
- **2D bubble:** disk of rho=1000, u=0, T=1/1000 centered at origin, ambient
  reference state moving at u=(Ma,0,0). Uniform p=1. Quasi-2D flow past an
  effectively rigid cold cylinder with heat transfer (t <= 0.2).
- **3D bubble** = same IC (the bubble block is z-uniform today; a spherical variant
  is future work, out of scope here).

## Scope (per user decision)

ICs + runnable example cases + automated stationary-contact test. The 1D
deterministic DVM-BGK kinetic reference solver is explicitly **out of scope** for
this pass (next step, separate PR).

## Design

### 1. New IC `ic_type = :riemann1d` (src/simulation_runner.jl)

New `elseif` branch in the IC section, following the `:bubble` block's per-rank
global-index pattern (so MPI decomposition works unchanged).

Left/right densities reuse the runner's already-required `rhol` / `rhor` params
(the names literally mean left/right; no new density params). The rest via
`get(params, key, default)`:

| param | default | meaning |
|---|---|---|
| `ul, Tl` | `0.0, 1.0` | left state (u is x-velocity; v=w=0) |
| `ur, Tr` | `0.0, Tl*rhol/rhor` | right state — **default is uniform pressure** (Rodney's case = `rhol=1, rhor=1000` with these defaults) |
| `x_interface` | `(xmin+xmax)/2` | jump location (cell-center test `x < x_interface`) |

Each side's covariance uses that side's T with the existing `r110/r101/r011`
correlations: `C200=C020=C002=T_side`, `C110 = r110*sqrt(C200*C020)`, etc.
Cells are filled with `InitializeM4_35(rho, u, 0, 0, C200, C110, C101, C020, C011, C002)`,
uniform in y and z. Two precomputed moment vectors (`Mr_l`, `Mr_r`), same as the
sharp bubble.

### 2. Extend `ic_type = :bubble` (same file, existing block)

New opt-in params:

| param | default | meaning |
|---|---|---|
| `T_in`  | `params.T` | temperature inside the bubble |
| `T_out` | `params.T` | ambient temperature |
| `u_out` | `0.0` | ambient x-velocity (inside stays u=0) |

- **Sharp profile:** `Mr_in = InitializeM4_35(rho_in, 0,0,0, C_in...)`,
  `Mr_out = InitializeM4_35(rho_out, u_out,0,0, C_out...)` with per-state
  covariances from `T_in`/`T_out`. With defaults, the constructor arguments are
  numerically identical to today's, so the default is byte-identical with no
  branch needed.
- **Smooth profile:** blend **pressure**, not temperature:
  `w = exp(-r^2/2 width^2)`, `rho_loc = rho_out + (rho_in-rho_out) w` (unchanged),
  `p_loc = p_out + (p_in-p_out) w`, `T_loc = p_loc/rho_loc`,
  `u_loc = u_out (1-w)`. This makes Rodney's uniform-p case *exactly* uniform-p in
  the smooth variant too (`p_in == p_out` ⇒ `p_loc` constant), which is what the
  grid-convergence study needs.
  **Byte-identity guard:** because `p_loc/rho_loc` is not bitwise `T` even when
  `T_in == T_out`, the smooth path keeps the existing expressions when
  `T_in == T_out && u_out == 0` and only takes the new blend otherwise.

Rodney's 2D case = `:bubble` + `rho_in=1000, T_in=1e-3, T_out=1.0, u_out=Ma,
bubble_profile=:sharp`, run quasi-2D (small Nz, `homogeneous_z=true`, copy BCs).

### 3. Example scripts (examples/)

- `examples/rodney_validation_1d.jl` — quasi-1D box (Nx=`RODNEY_NP` env or 256,
  Ny=Nz=4), `:riemann1d` defaults, Kn from `RODNEY_KN` (default 0.0),
  `spatial_order=2`. Writes `output/runs/rodney1d_Kn<...>_Np<...>.jld2` and sets
  `web_dir="output"` (auto web-viewer export).
- `examples/rodney_validation_2d.jl` — quasi-2D (Np² × 4, `homogeneous_z=true`),
  `:bubble` with the uniform-p parameters above, `RODNEY_MA` (default 1.0) and
  `RODNEY_KN` (default 0.01), t_final ≈ 0.2. Same output/web wiring.
- Both runnable via the documented MPI recipe (`mpiexec -n N julia --project ...`)
  and single-rank.

### 4. Automated test (test/test_rodney_cases.jl, added to runtests.jl)

Header `using MPI; MPI.Initialized() || MPI.Init()` (pattern of
`test_highorder_3d.jl`).

- **Stationary-contact test (the real correctness gate):** `:riemann1d` defaults,
  Kn=0, small grid (Nx=64, Ny=Nz=4), short run (~t=0.05, spatial_order=2). Assert:
  1. `max |u|, |v|, |w|` over the domain stays below a calibrated tolerance
     (expected: truncation-error small, ≪ 1; exact ceiling set with margin from
     the first converged run — the point is it must be *orders below* any physical
     velocity scale, and stable under repetition).
  2. Pressure `rho*T` deviation from 1 below a calibrated tolerance.
  3. y/z uniformity preserved to near machine precision (strong invariant: the IC
     is y,z-uniform and copy BCs preserve that exactly).
  4. Total mass conserved to machine precision.
- **Bubble smoke test:** 16×16×4 uniform-p bubble with `u_out=1.0`, Kn=0.01, a few
  steps: all moments finite, mass conserved, run completes (no realizability
  abort).
- **Default-unchanged test:** an existing-style `:bubble` IC built with and
  without the new params at their defaults produces a byte-identical `M` field
  (guards the byte-identity promise directly).

No new test-only deps (MPI, Test, Riemann35 already declared — per the
Project.toml test-deps rule).

### 5. Documentation

- `examples/README.md`: new "Rodney validation cases" section — the physics
  (why uniform-p / stationary contact is an exact test), how to run 1D/2D, the
  parameters, and the web-viewer tie-in.
- Docstrings/comments on the new IC branch and new bubble params, citing the
  email's case definitions.

## Error handling

- `:riemann1d` asserts `rhol, rhor, Tl, Tr > 0` with a clear error.
- Everything else inherits the runner's existing positivity/realizability
  safeguards; no new failure modes introduced (new params are pure IC-time).

## Non-goals

- No DVM-BGK reference solver (next PR).
- No new boundary conditions (`:copy` extrapolation is sufficient while
  disturbances stay inside the domain; examples size the domain/t_final
  accordingly and say so).
- No spherical 3D bubble variant.
- No GPU-path IC builder (GPU driver takes a prebuilt `M0`; a host-side IC can
  reuse these formulas later if needed).

## Compatibility / house rules

- All new behavior opt-in; defaults byte-identical (explicitly tested, see §4).
- Branch `feat/rodney-validation-cases`, PR to `comp-physics/Riemann35.jl`,
  merge on green CI.

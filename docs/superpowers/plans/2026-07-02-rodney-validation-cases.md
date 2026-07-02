# Rodney Validation Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Rodney Fox's uniform-pressure validation cases (1D stationary-contact Riemann problem, 2D dense-cold bubble in ambient flow) as opt-in ICs with runnable examples and an automated exact-solution test.

**Architecture:** Two IC additions inside `run_simulation`'s existing IC dispatch in `src/simulation_runner.jl` (a new `:riemann1d` branch; three opt-in params on the existing `:bubble` branch), following the established per-rank global-index fill pattern so MPI decomposition works unchanged. One new test file exercises them through the public `simulation_runner(params)` entry point (the `tmax=0.0` run returns the gathered IC on rank 0, making IC-level assertions cheap). Two example scripts wire Rodney's parameters to the snapshot + web-viewer machinery.

**Tech Stack:** Julia, MPI.jl (single-rank in tests), JLD2 snapshots, existing Riemann35 runner.

**Spec:** `docs/superpowers/specs/2026-07-02-rodney-validation-cases-design.md`

## Global Constraints

- All new params opt-in via `get(params, key, default)`; defaults byte-identical to current behavior (explicitly tested with `==` on the full moment field).
- Branch: `feat/rodney-validation-cases` (already created, spec committed). PR to `comp-physics/Riemann35.jl`, merge on green CI.
- No new dependencies (test file uses only MPI, Test, Riemann35 — all already declared).
- Work from repo root `/storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl`; run tests as `julia --project=. test/test_rodney_cases.jl` (single-rank; the file MPI-inits itself). If `julia` is not on PATH, `module load julia` first.
- The runner's return contract (from `test/test_highorder_3d.jl`): `simulation_runner(params)` → `(M_final, t_final, steps, grid)` on rank 0, where `M_final` is the gathered **interior** global field `(Nx,Ny,Nz,35)`; other ranks get `nothing` for `M_final`/`grid`.
- Moment indexing for assertions: `rho=M[..,1]`, `rho*u=M[..,2]`, `M200=M[..,3]`, `rho*v=M[..,6]`, `M020=M[..,10]`, `rho*w=M[..,16]`, `M002=M[..,20]`.

---

### Task 1: `:riemann1d` IC + test scaffolding

**Files:**
- Modify: `src/simulation_runner.jl` (insert new `elseif` immediately before the `elseif haskey(params, :ic_type) && params.ic_type == :crossing_matlab` branch, currently line 324)
- Create: `test/test_rodney_cases.jl`
- Modify: `test/runtests.jl` (add include after `include("test_highorder_3d.jl")`)

**Interfaces:**
- Consumes: existing in-scope locals in the IC section: `nx, ny, nz, halo, i0i1, xmin, xmax, dx_global, rhol, rhor, r110, r101, r011, M`; `InitializeM4_35(rho,u,v,w,C200,C110,C101,C020,C011,C002)`.
- Produces: `ic_type = :riemann1d` honoring params `ul` (default `0.0`), `Tl` (`1.0`), `ur` (`0.0`), `Tr` (`Tl*rhol/rhor` — uniform pressure), `x_interface` (`(xmin+xmax)/2`). Left/right densities come from the already-required `rhol`/`rhor`.

- [ ] **Step 1: Write the failing test file**

Create `test/test_rodney_cases.jl`:

```julia
"""
Rodney Fox's uniform-pressure validation cases (email 2026-07-02; see
docs/superpowers/specs/2026-07-02-rodney-validation-cases-design.md).

1D: L=(rho,u,T)=(1,0,1), R=(1000,0,1e-3), p≡1. Kn=0 Euler exact solution is a
STATIONARY CONTACT — any velocity/pressure deviation is pure numerical error.
2D: dense cold bubble (same two states) in ambient flow u=(Ma,0,0), quasi-2D.
"""
using MPI
MPI.Initialized() || MPI.Init()
using Test
using Riemann35

const RODNEY_RANK = MPI.Comm_rank(MPI.COMM_WORLD)

# Complete required-params set (pattern of test_highorder_3d.jl params_ho),
# overridable per test. tmax=0.0 → zero steps → rank 0 gets the gathered IC.
rodney_params(; kw...) = merge((
    Nx = 64, Ny = 4, Nz = 4,
    tmax    = 0.0,
    Kn      = 0.0,
    Ma      = 0.0,
    flag2D  = 0,
    CFL     = 1/3,
    Nmom    = 35,
    nnmax   = 100000,
    dtmax   = 1000.0,
    rhol    = 1.0,
    rhor    = 1000.0,
    T       = 1.0,
    r110    = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000,
    homogeneous_z = true,
    debug_output  = false,
    snapshot_interval = 0,
    ic_type = :riemann1d,
    spatial_order = 2,
), NamedTuple(kw))

# field helpers on the gathered (Nx,Ny,Nz,35) array
_rho(M) = M[:, :, :, 1]
_u(M)   = M[:, :, :, 2]  ./ _rho(M)
_v(M)   = M[:, :, :, 6]  ./ _rho(M)
_w(M)   = M[:, :, :, 16] ./ _rho(M)
function _pressure(M)
    rho = _rho(M); u = _u(M); v = _v(M); w = _w(M)
    T3 = (M[:,:,:,3] ./ rho .- u.^2) .+ (M[:,:,:,10] ./ rho .- v.^2) .+
         (M[:,:,:,20] ./ rho .- w.^2)
    return rho .* T3 ./ 3
end

@testset ":riemann1d IC — uniform-pressure default" begin
    M, t, steps, grid = simulation_runner(rodney_params())
    @test steps == 0
    if RODNEY_RANK == 0
        @test size(M) == (64, 4, 4, 35)
        @test all(isfinite, M)
        rho = _rho(M)
        @test rho[1, 1, 1]   ≈ 1.0
        @test rho[end, 1, 1] ≈ 1000.0
        # zero bulk velocity everywhere in the IC
        @test maximum(abs, M[:, :, :, 2]) == 0.0
        @test maximum(abs, M[:, :, :, 6]) == 0.0
        @test maximum(abs, M[:, :, :, 16]) == 0.0
        # default Tr = Tl*rhol/rhor ⇒ uniform pressure p = 1
        @test maximum(abs, _pressure(M) .- 1.0) < 1e-10
        # uniform in y and z
        @test M == repeat(M[:, 1:1, 1:1, :], 1, 4, 4, 1)
        # interface at the domain midpoint (default domain [-0.5,0.5], Nx=64)
        @test rho[32, 1, 1] ≈ 1.0
        @test rho[33, 1, 1] ≈ 1000.0
    end
end

@testset ":riemann1d IC — explicit states override defaults" begin
    p = rodney_params(rhol = 2.0, rhor = 3.0, ul = 0.5, ur = -0.25,
                      Tl = 2.0, Tr = 0.5, x_interface = -0.25)
    M, t, steps, grid = simulation_runner(p)
    if RODNEY_RANK == 0
        rho = _rho(M); u = _u(M)
        @test rho[1, 1, 1] ≈ 2.0
        @test u[1, 1, 1]   ≈ 0.5
        @test rho[end, 1, 1] ≈ 3.0
        @test u[end, 1, 1]   ≈ -0.25
        # x_interface=-0.25 on [-0.5,0.5] with Nx=64 → cells 1:16 left, 17:64 right
        @test rho[16, 1, 1] ≈ 2.0
        @test rho[17, 1, 1] ≈ 3.0
    end
end
```

- [ ] **Step 2: Register the test file and run it to verify it fails**

In `test/runtests.jl`, after the line `include("test_highorder_3d.jl")` add:

```julia
        include("test_rodney_cases.jl")
```

Run: `cd /storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl && julia --project=. test/test_rodney_cases.jl`
Expected: FAIL — `:riemann1d` is unknown, so the runner falls through to the default crossing-jets IC and the density/pressure assertions fail (or an error is raised). Confirm the failure is IC-related, not a harness problem.

- [ ] **Step 3: Implement the `:riemann1d` branch**

In `src/simulation_runner.jl`, immediately BEFORE the line
`    elseif haskey(params, :ic_type) && params.ic_type == :crossing_matlab`
insert:

```julia
    elseif haskey(params, :ic_type) && params.ic_type == :riemann1d
        # 1D Riemann problem along x, uniform in y/z: two Maxwellian states
        # (rhol, ul, Tl) | (rhor, ur, Tr) split at x_interface. The default
        # Tr = Tl*rhol/rhor gives UNIFORM PRESSURE — Rodney Fox's validation
        # case (2026-07-02): at Kn=0 the exact Euler solution is a stationary
        # contact, so any velocity/pressure deviation is pure numerical error.
        # See docs/superpowers/specs/2026-07-02-rodney-validation-cases-design.md.
        ul = get(params, :ul, 0.0)
        Tl = get(params, :Tl, 1.0)
        ur = get(params, :ur, 0.0)
        Tr = get(params, :Tr, Tl * rhol / rhor)
        x_interface = get(params, :x_interface, (xmin + xmax) / 2)
        (rhol > 0 && rhor > 0 && Tl > 0 && Tr > 0) ||
            error(":riemann1d requires rhol, rhor, Tl, Tr > 0 (got rhol=$rhol rhor=$rhor Tl=$Tl Tr=$Tr)")

        Cl110 = r110 * sqrt(Tl * Tl); Cl101 = r101 * sqrt(Tl * Tl); Cl011 = r011 * sqrt(Tl * Tl)
        Cr110 = r110 * sqrt(Tr * Tr); Cr101 = r101 * sqrt(Tr * Tr); Cr011 = r011 * sqrt(Tr * Tr)
        M_left  = InitializeM4_35(rhol, ul, 0.0, 0.0, Tl, Cl110, Cl101, Tl, Cl011, Tl)
        M_right = InitializeM4_35(rhor, ur, 0.0, 0.0, Tr, Cr110, Cr101, Tr, Cr011, Tr)

        for kk in 1:nz
            for ii in 1:nx
                gi = i0i1[1] + ii - 1                     # global i index
                xcoord = xmin + (gi - 0.5) * dx_global
                Mside = (xcoord < x_interface) ? M_left : M_right
                for jj in 1:ny
                    M[ii + halo, jj + halo, kk, :] = Mside
                end
            end
        end
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/test_rodney_cases.jl`
Expected: PASS (both testsets). If `steps == 0` fails because `tmax=0.0` still takes a step, inspect the time loop (`while t < tmax ...`) — the correct fix is asserting `steps == 0` against actual runner semantics, NOT changing the runner. If the runner errors on `tmax=0.0` (e.g. division in progress reporting), use `tmax=0.0` plus `nnmax=0` in `rodney_params` and drop the `steps == 0` assertion.

- [ ] **Step 5: Commit**

```bash
cd /storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl
git add src/simulation_runner.jl test/test_rodney_cases.jl test/runtests.jl
git commit -m "feat: :riemann1d uniform-pressure Riemann IC (Rodney 1D validation case)"
```

---

### Task 2: Stationary-contact evolution test (the exact-solution gate)

**Files:**
- Modify: `test/test_rodney_cases.jl` (append testset)

**Interfaces:**
- Consumes: `rodney_params`, `_rho/_u/_v/_w/_pressure` helpers and `:riemann1d` from Task 1.
- Produces: the calibrated tolerances documented in the testset (later tasks don't consume them).

- [ ] **Step 1: Append the evolution testset**

Append to `test/test_rodney_cases.jl`:

```julia
@testset "stationary contact, Kn=0: nothing moves" begin
    # Kn=0 ⇒ tc=0 ⇒ exact-exponential BGK relaxes instantly to the Maxwellian
    # (exp(-dt/0)=0); the IC is already Maxwellian, so the exact solution is
    # frozen. Every metric below is therefore PURE numerical error.
    M, t, steps, grid = simulation_runner(rodney_params(tmax = 0.05))
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        maxvel = max(maximum(abs, _u(M)), maximum(abs, _v(M)), maximum(abs, _w(M)))
        pdev   = maximum(abs, _pressure(M) .- 1.0)
        @info "stationary-contact error metrics" maxvel pdev steps t
        # Calibration ceilings (tightened to ~5x observed after first green run;
        # see Step 3). These are correctness gates: growth here = solver regression.
        @test maxvel < 1e-2
        @test pdev   < 2e-2
        # y/z uniformity is preserved exactly by copy BCs on a y/z-uniform field
        @test maximum(abs, M .- repeat(M[:, 1:1, 1:1, :], 1, 4, 4, 1)) < 1e-9
        # mass conservation: u≈0 at the x boundaries ⇒ near-exact
        dxg = 1.0 / 64
        mass  = sum(_rho(M)[:, 1, 1]) * dxg          # per unit y/z area
        mass0 = (32 * 1.0 + 32 * 1000.0) * dxg       # exact IC mass
        @test abs(mass - mass0) / mass0 < 1e-10
    end
end
```

- [ ] **Step 2: Run and record observed metrics**

Run: `julia --project=. test/test_rodney_cases.jl`
Expected: PASS, with an `@info` line reporting `maxvel` and `pdev`. Record both numbers in the commit message. If it FAILS on `maxvel`/`pdev`: that is a genuine finding about the scheme at a 1000:1 contact — do NOT loosen silently; report the observed values to the user before proceeding.

- [ ] **Step 3: Tighten tolerances to ~5x observed**

Edit the two `@test` ceilings to `5x` the observed `maxvel` and `pdev` (rounded up to one significant digit), keeping the comment. Re-run: `julia --project=. test/test_rodney_cases.jl` → PASS.

- [ ] **Step 4: Commit**

```bash
git add test/test_rodney_cases.jl
git commit -m "test: stationary-contact exact-solution gate (Kn=0, observed maxvel=<X>, pdev=<Y>)"
```

---

### Task 3: `:bubble` extension (T_in / T_out / u_out) with byte-identity guard

**Files:**
- Modify: `src/simulation_runner.jl:278-326` (the `:bubble` branch)
- Modify: `test/test_rodney_cases.jl` (append testsets)

**Interfaces:**
- Consumes: existing `:bubble` block locals (`rho_in, rho_out, radius, xc, yc, profile, width, T, r110, r101, r011`).
- Produces: `:bubble` params `T_in` (default `params.T`), `T_out` (default `params.T`), `u_out` (default `0.0`). Smooth profile blends pressure: `p_loc = p_out + (p_in-p_out)·w`, `T_loc = p_loc/rho_loc`, `u_loc = u_out·(1-w)` — but ONLY when `(T_in != T_out || u_out != 0)`; the default path keeps today's exact expressions.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_rodney_cases.jl`:

```julia
# complete bubble params (rhol/rhor required by the runner but unused by :bubble)
bubble_params(; kw...) = merge(rodney_params(
        Nx = 16, Ny = 16, Nz = 4, rhol = 1.0, rhor = 1.0,
        ic_type = :bubble,
    ), NamedTuple(kw))

@testset ":bubble new params default byte-identical" begin
    for prof in (:sharp, :smooth)
        M1, _, _, _ = simulation_runner(bubble_params(bubble_profile = prof))
        M2, _, _, _ = simulation_runner(bubble_params(bubble_profile = prof,
                                                      T_in = 1.0, T_out = 1.0, u_out = 0.0))
        if RODNEY_RANK == 0
            @test M1 == M2   # bitwise
        end
    end
end

@testset "uniform-pressure bubble IC (Rodney 2D case)" begin
    # rho_in=1000, T_in=1e-3, T_out=1, u_out=1 ⇒ p ≡ 1, ambient flow at Ma=1
    up = (rho_in = 1000.0, rho_out = 1.0, T_in = 1e-3, T_out = 1.0,
          u_out = 1.0, bubble_radius = 0.15)
    M, _, _, _ = simulation_runner(bubble_params(; up...))
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        @test maximum(abs, _pressure(M) .- 1.0) < 1e-10
        rho = _rho(M); u = _u(M)
        @test rho[8, 8, 1] ≈ 1000.0        # center cell inside bubble
        @test u[8, 8, 1]   ≈ 0.0 atol=1e-14
        @test rho[1, 1, 1] ≈ 1.0           # corner is ambient
        @test u[1, 1, 1]   ≈ 1.0
    end
    # smooth variant blends PRESSURE, so uniform p stays exactly uniform
    Ms, _, _, _ = simulation_runner(bubble_params(; up..., bubble_profile = :smooth))
    if RODNEY_RANK == 0
        @test all(isfinite, Ms)
        @test maximum(abs, _pressure(Ms) .- 1.0) < 1e-10
        @test _rho(Ms)[8, 8, 1] > 100.0    # dense core present
    end
end
```

- [ ] **Step 2: Run to verify failure**

Run: `julia --project=. test/test_rodney_cases.jl`
Expected: byte-identity testset may already pass trivially (new params ignored today) — that's fine; the uniform-pressure testset must FAIL (T_in/T_out/u_out have no effect yet, so pressure is 1000·1=1000 inside, u=0 outside). Confirm exactly that failure mode.

- [ ] **Step 3: Implement the bubble extension**

In `src/simulation_runner.jl`, `:bubble` branch. Replace the block

```julia
        C200 = T
        C020 = T
        C002 = T
        C110 = r110 * sqrt(C200 * C020)
        C101 = r101 * sqrt(C200 * C002)
        C011 = r011 * sqrt(C020 * C002)

        Mr_in  = InitializeM4_35(rho_in,  0.0, 0.0, 0.0, C200, C110, C101, C020, C011, C002)
        Mr_out = InitializeM4_35(rho_out, 0.0, 0.0, 0.0, C200, C110, C101, C020, C011, C002)
```

with

```julia
        # Opt-in non-isothermal / moving-ambient extension (Rodney Fox's
        # uniform-pressure bubble, 2026-07-02): T_in=1/rho_in with T_out=1 gives
        # p≡1; u_out is the ambient x-velocity (bubble interior stays at rest).
        # Defaults (T_in=T_out=T, u_out=0) reproduce the isothermal Rice et al.
        # case byte-identically.
        T_in  = get(params, :T_in, T)
        T_out = get(params, :T_out, T)
        u_out = get(params, :u_out, 0.0)

        Ci110 = r110 * sqrt(T_in * T_in);  Ci101 = r101 * sqrt(T_in * T_in);  Ci011 = r011 * sqrt(T_in * T_in)
        Co110 = r110 * sqrt(T_out * T_out); Co101 = r101 * sqrt(T_out * T_out); Co011 = r011 * sqrt(T_out * T_out)

        Mr_in  = InitializeM4_35(rho_in,  0.0,   0.0, 0.0, T_in,  Ci110, Ci101, T_in,  Ci011, T_in)
        Mr_out = InitializeM4_35(rho_out, u_out, 0.0, 0.0, T_out, Co110, Co101, T_out, Co011, T_out)
```

and replace the smooth-profile body

```julia
                    if profile == :smooth
                        rho_loc = rho_out + (rho_in - rho_out) * exp(-rr2 / (2 * width^2))
                        M[ii + halo, jj + halo, kk, :] =
                            InitializeM4_35(rho_loc, 0.0, 0.0, 0.0, C200, C110, C101, C020, C011, C002)
                    else
```

with

```julia
                    if profile == :smooth
                        if T_in == T_out && u_out == 0.0
                            # existing isothermal path — expressions kept
                            # verbatim so the default stays byte-identical
                            rho_loc = rho_out + (rho_in - rho_out) * exp(-rr2 / (2 * width^2))
                            M[ii + halo, jj + halo, kk, :] =
                                InitializeM4_35(rho_loc, 0.0, 0.0, 0.0, T_out, Co110, Co101, T_out, Co011, T_out)
                        else
                            # blend PRESSURE (not T) so a uniform-p case stays
                            # exactly uniform-p at every resolution: T = p/rho.
                            wgt = exp(-rr2 / (2 * width^2))
                            rho_loc = rho_out + (rho_in - rho_out) * wgt
                            p_loc = rho_out * T_out + (rho_in * T_in - rho_out * T_out) * wgt
                            T_loc = p_loc / rho_loc
                            u_loc = u_out * (1 - wgt)
                            Cs110 = r110 * sqrt(T_loc * T_loc); Cs101 = r101 * sqrt(T_loc * T_loc); Cs011 = r011 * sqrt(T_loc * T_loc)
                            M[ii + halo, jj + halo, kk, :] =
                                InitializeM4_35(rho_loc, u_loc, 0.0, 0.0, T_loc, Cs110, Cs101, T_loc, Cs011, T_loc)
                        end
                    else
```

NOTE: the old locals `C200, C020, C002, C110, C101, C011` are removed from this branch entirely — grep the `:bubble` block afterward to confirm no dangling references (`sed -n '278,340p' src/simulation_runner.jl`). The default smooth path passes `T_out`/`Co110`… which equal the old `C200`/`C110`… values bitwise when defaults are in play (`T_out = T`, and `r110*sqrt(T_out*T_out)` is the same expression as `r110*sqrt(C200*C020)` with `C200=C020=T`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `julia --project=. test/test_rodney_cases.jl`
Expected: PASS — including both byte-identity cases (the true guard is that they still pass AFTER this edit).

- [ ] **Step 5: Run the pre-existing bubble/IC test files to catch regressions**

Run: `julia --project=. test/test_initial_conditions.jl && julia --project=. test/test_simulation_runner.jl`
Expected: PASS (these exercise the runner's IC paths; the `:bubble` default must be unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/simulation_runner.jl test/test_rodney_cases.jl
git commit -m "feat: :bubble opt-in T_in/T_out/u_out (uniform-pressure bubble, Rodney 2D case)"
```

---

### Task 4: Bubble short-run smoke test

**Files:**
- Modify: `test/test_rodney_cases.jl` (append testset)

**Interfaces:**
- Consumes: `bubble_params` and the uniform-pressure params from Task 3.

- [ ] **Step 1: Append the smoke testset**

```julia
@testset "uniform-pressure bubble: short run stays sane" begin
    p = bubble_params(rho_in = 1000.0, rho_out = 1.0, T_in = 1e-3, T_out = 1.0,
                      u_out = 1.0, bubble_radius = 0.15, Ma = 1.0,
                      Kn = 0.01, tmax = 0.005)
    M, t, steps, grid = simulation_runner(p)
    @test steps >= 1
    if RODNEY_RANK == 0
        @test all(isfinite, M)
        @test minimum(_rho(M)) > 0.0
        # copy BCs with ambient throughflow ⇒ modest mass drift allowed
        dxg = 1.0 / 16
        mass  = sum(_rho(M)[:, :, 1]) * dxg^2
        mass0 = sum(_rho(simulation_runner(bubble_params(rho_in = 1000.0, rho_out = 1.0,
                        T_in = 1e-3, T_out = 1.0, u_out = 1.0,
                        bubble_radius = 0.15))[1])[:, :, 1]) * dxg^2
        @test abs(mass - mass0) / mass0 < 5e-2
    end
end
```

(Note: `mass0` re-runs the `tmax=0.0` IC — cheap at 16².)

- [ ] **Step 2: Run to verify it passes**

Run: `julia --project=. test/test_rodney_cases.jl`
Expected: PASS. If the run aborts on realizability at the bubble edge (rho 1000:1, T 1000:1 in adjacent cells), report the failure details to the user — that is a finding about the scheme, not a test bug.

- [ ] **Step 3: Commit**

```bash
git add test/test_rodney_cases.jl
git commit -m "test: uniform-pressure bubble short-run smoke test (Kn=0.01, Ma=1)"
```

---

### Task 5: Example scripts

**Files:**
- Create: `examples/rodney_validation_1d.jl`
- Create: `examples/rodney_validation_2d.jl`

**Interfaces:**
- Consumes: `:riemann1d` (Task 1), `:bubble` extension (Task 3), `simulation_runner` snapshot mode (`snapshot_interval > 0` + `snapshot_filename`), `web_dir` auto-export (already on main).

- [ ] **Step 1: Write `examples/rodney_validation_1d.jl`**

```julia
# Rodney Fox's 1D uniform-pressure validation case (2026-07-02):
# stationary-contact Riemann problem. L: (rho,u,T)=(1,0,1); R: (1000,0,1e-3); p≡1.
# Kn=0 → exact solution is stationary (verification). Kn>0 → non-equilibrium
# heat flux develops on the dilute side (validation vs kinetic reference).
#
# Usage:
#   julia --project=. examples/rodney_validation_1d.jl
#   RODNEY_NP=512 RODNEY_KN=0.01 mpiexec -n 4 julia --project=. examples/rodney_validation_1d.jl
# Output: output/runs/<tag>.jld2 + browseable bundle in output/viz/ (./serve.sh).
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35

Np    = parse(Int,     get(ENV, "RODNEY_NP",   "256"))
Kn    = parse(Float64, get(ENV, "RODNEY_KN",   "0.0"))
tmax  = parse(Float64, get(ENV, "RODNEY_TMAX", "0.1"))
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

rank0 && mkpath("output/runs")
tag = "rodney1d_Kn$(Kn)_Np$(Np)"

params = (
    Nx = Np, Ny = 4, Nz = 4,
    tmax = tmax, Kn = Kn, Ma = 0.0, flag2D = 0, CFL = 1/3,
    Nmom = 35, nnmax = 1_000_000, dtmax = 1000.0,
    rhol = 1.0, rhor = 1000.0,           # :riemann1d L/R densities
    T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
    ic_type = :riemann1d, spatial_order = 2,
    # defaults: ul=ur=0, Tl=1, Tr=Tl*rhol/rhor=1e-3 (uniform p), interface at x=0
    snapshot_interval = 25,
    snapshot_filename = "output/runs/$tag.jld2",
    web_dir = "output",
)
result = simulation_runner(params)
rank0 && println("done: $tag → output/runs/$tag.jld2 (browse: output/viz/serve.sh)")
```

- [ ] **Step 2: Write `examples/rodney_validation_2d.jl`**

```julia
# Rodney Fox's 2D uniform-pressure dense-bubble case (2026-07-02): cold dense
# disk (rho=1000, T=1e-3) at the origin, ambient reference gas (rho=T=p=1)
# flowing past at u=(Ma,0,0). Quasi-2D flow past an effectively rigid cold
# cylinder with heat transfer; t <= 0.2 (dense gas barely moves). Copy BCs act
# as crude in/outflow — keep t small enough that disturbances stay interior.
#
# Usage:
#   julia --project=. examples/rodney_validation_2d.jl
#   RODNEY_NP=512 RODNEY_MA=1.0 RODNEY_KN=0.01 mpiexec -n 4 julia --project=. examples/rodney_validation_2d.jl
# Output: output/runs/<tag>.jld2 + browseable bundle in output/viz/ (./serve.sh).
using MPI
MPI.Initialized() || MPI.Init()
using Riemann35

Np    = parse(Int,     get(ENV, "RODNEY_NP",   "128"))
Ma    = parse(Float64, get(ENV, "RODNEY_MA",   "1.0"))
Kn    = parse(Float64, get(ENV, "RODNEY_KN",   "0.01"))
tmax  = parse(Float64, get(ENV, "RODNEY_TMAX", "0.2"))
rank0 = MPI.Comm_rank(MPI.COMM_WORLD) == 0

rank0 && mkpath("output/runs")
tag = "rodney2d_Ma$(Ma)_Kn$(Kn)_Np$(Np)"

params = (
    Nx = Np, Ny = Np, Nz = 4,
    tmax = tmax, Kn = Kn, Ma = Ma, flag2D = 0, CFL = 1/3,
    Nmom = 35, nnmax = 1_000_000, dtmax = 1000.0,
    rhol = 1.0, rhor = 1.0,              # required by the runner; unused by :bubble
    T = 1.0, r110 = 0.0, r101 = 0.0, r011 = 0.0,
    symmetry_check_interval = 100000, homogeneous_z = true, debug_output = false,
    ic_type = :bubble, spatial_order = 2,
    rho_in = 1000.0, rho_out = 1.0, bubble_radius = 0.15,
    T_in = 1e-3, T_out = 1.0, u_out = Ma,   # uniform p=1, ambient flow
    snapshot_interval = 25,
    snapshot_filename = "output/runs/$tag.jld2",
    web_dir = "output",
)
result = simulation_runner(params)
rank0 && println("done: $tag → output/runs/$tag.jld2 (browse: output/viz/serve.sh)")
```

- [ ] **Step 3: Smoke-run both scripts small**

Run:
```bash
RODNEY_NP=32 RODNEY_TMAX=0.01 julia --project=. examples/rodney_validation_1d.jl
RODNEY_NP=32 RODNEY_TMAX=0.005 julia --project=. examples/rodney_validation_2d.jl
```
Expected: both finish, print `done: ...`, create `output/runs/rodney1d_*.jld2` / `rodney2d_*.jld2`, and the web export logs a bundle into `output/viz/` (cases appear in `output/viz/manifest.json`).

- [ ] **Step 4: Commit**

```bash
git add examples/rodney_validation_1d.jl examples/rodney_validation_2d.jl
git commit -m "feat: runnable example scripts for Rodney's 1D/2D validation cases"
```

---

### Task 6: Documentation, full-suite verification, PR

**Files:**
- Modify: `examples/README.md` (new section after the existing example sections)

- [ ] **Step 1: Add README section**

Append to `examples/README.md`:

```markdown
### `rodney_validation_1d.jl` / `rodney_validation_2d.jl` - Uniform-Pressure Validation Cases

Rodney Fox's validation ladder for the 35-moment model (2026-07-02). Both cases
start from **uniform pressure** (p ≡ 1), so the Kn=0 Euler solution is trivial:

- **1D** (`ic_type = :riemann1d`): stationary contact — L: (ρ,u,T)=(1,0,1),
  R: (1000,0,10⁻³). Nothing moves at Kn=0, so any velocity or pressure deviation
  is pure numerical error (exact verification of the Riemann solver). At Kn>0,
  non-equilibrium heat flux develops on the dilute side (validation vs kinetic
  references). Defaults give uniform pressure: `Tr = Tl·rhol/rhor`.
- **2D** (`ic_type = :bubble` + `T_in/T_out/u_out`): the same dense cold state as
  a disk, ambient gas flowing past at `u_out = Ma` — quasi-2D flow past an
  effectively rigid cold cylinder with heat transfer (t ≤ 0.2).

```bash
# 1D, default Rodney parameters (Np=256, Kn=0)
julia --project=. examples/rodney_validation_1d.jl

# 2D at Ma=1, Kn=0.01 on 512² with 4 ranks
RODNEY_NP=512 RODNEY_MA=1.0 RODNEY_KN=0.01 mpiexec -n 4 julia --project=. examples/rodney_validation_2d.jl
```

Env knobs: `RODNEY_NP`, `RODNEY_KN`, `RODNEY_TMAX`, and (2D) `RODNEY_MA`.
Snapshots land in `output/runs/`, with a browser-viewable bundle auto-exported to
`output/viz/` (see `output/viz/README.md`). Outer boundaries are copy/extrapolation;
keep `tmax` small enough that disturbances stay interior. The automated
exact-solution gate lives in `test/test_rodney_cases.jl`.
```

- [ ] **Step 2: Run the full test suite**

Run: `cd /storage/project/r-sbryngelson3-0/sbryngelson3/Riemann35.jl && julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: all tests pass, including the new `test_rodney_cases.jl` (verifies the runtests.jl inclusion and no test-dep gaps — per the project's Pkg.test-not-direct-run rule).

- [ ] **Step 3: Commit docs, push, open PR**

```bash
git add examples/README.md
git commit -m "docs: Rodney validation cases in examples README"
git push -u origin feat/rodney-validation-cases
gh pr create --repo comp-physics/Riemann35.jl \
  --title "Rodney's uniform-pressure validation cases: :riemann1d IC + :bubble T_in/T_out/u_out" \
  --body "$(cat <<'EOF'
Implements Rodney Fox's proposed validation ladder (2026-07-02) — see
docs/superpowers/specs/2026-07-02-rodney-validation-cases-design.md.

- New opt-in `ic_type = :riemann1d`: uniform-pressure 1D Riemann problem
  (default Tr = Tl·rhol/rhor ⇒ p≡1; Rodney's case is rhol=1, rhor=1000).
  At Kn=0 the exact solution is a stationary contact — the new test gate
  asserts max|u| and pressure deviation stay below calibrated ceilings.
- `:bubble` gains opt-in `T_in`/`T_out`/`u_out` (defaults byte-identical —
  tested bitwise). Smooth profile blends pressure, so uniform-p stays exact.
- Example scripts `examples/rodney_validation_{1d,2d}.jl` with snapshot +
  web-viewer auto-export; README section.
- Out of scope (next PR): 1D DVM-BGK kinetic reference solver.
EOF
)"
```

Expected: PR opens; CI (Julia 1.9/1.10/1.11 + MPI tests) must go green before merge (merge only on explicit user go-ahead, per standing practice).

---

## Self-Review Notes

- Spec coverage: §1 riemann1d → Task 1; §2 bubble → Task 3; §3 examples → Task 5; §4 tests → Tasks 1–4 (contact gate = Task 2, byte-identity = Task 3 Step 1, smoke = Task 4); §5 docs → Task 6. Error-handling bullet → Task 1 Step 3 (`error(...)` guard).
- Type consistency: `rodney_params`/`bubble_params` return NamedTuples merged with overrides; `simulation_runner(params)` 4-tuple contract used uniformly; helper names `_rho/_u/_v/_w/_pressure` consistent across Tasks 1–4.
- Known judgment points flagged inline: `tmax=0.0` semantics (Task 1 Step 4 fallback), tolerance calibration (Task 2 Steps 2–3), realizability at the bubble edge (Task 4 Step 2) — all instruct "report, don't silently loosen."

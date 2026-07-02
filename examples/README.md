# Examples - 3D Crossing Jets Simulations

All examples support **full parameter control** via both code defaults and command-line arguments.

## Available Examples

### `run_3d_jets_timeseries.jl` - Interactive Time-Series (Recommended)
**Main example with interactive 3D visualization over time. Works in both serial and MPI parallel modes.**

```bash
# Serial run
julia --project=. examples/run_3d_jets_timeseries.jl

# Serial with custom parameters
julia --project=. examples/run_3d_jets_timeseries.jl --Nx 60 --Ny 60 --tmax 0.1 --snapshot-interval 5

# MPI parallel (4 ranks)
mpiexec -n 4 julia --project=. examples/run_3d_jets_timeseries.jl --Nx 100 --Ny 100

# MPI parallel with custom parameters
mpiexec -n 8 julia --project=. examples/run_3d_jets_timeseries.jl --Nx 120 --Ny 120 --Nz 60 --snapshot-interval 5
```

**Features:**
- Automatic serial/MPI support (no separate MPI example needed!)
- Interactive GLMakie 3D viewer with physical + moment space
- Time-series animation with play/pause
- Multiple quantities (density, U/V/W velocities)
- Isosurface visualization with positive/negative values
- Full parameter control via command-line

### `run_3d_custom_jets.jl` - Custom Configurations
**Flexible jet configurations with interactive 3D visualization. Works in both serial and MPI parallel modes.**

```bash
# Use predefined configurations
julia --project=. examples/run_3d_custom_jets.jl --config crossing
julia --project=. examples/run_3d_custom_jets.jl --config triple-jet
julia --project=. examples/run_3d_custom_jets.jl --config quad-jet

# MPI parallel
mpiexec -n 4 julia --project=. examples/run_3d_custom_jets.jl --config crossing --Nx 100 --Ny 100
```

**Features:**
- Multiple jet configurations (crossing, triple-jet, quad-jet, spiral, etc.)
- Automatic serial/MPI support
- Interactive GLMakie 3D viewer with physical + moment space
- Standardized moments enabled by default
- Full parameter control via command-line

## Quick Start

### First Time Setup
```bash
cd HyQMOM.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Run with Defaults
```bash
julia --project=. examples/run_3d_jets_timeseries.jl
```

### Get Help
```bash
julia --project=. examples/run_3d_jets_timeseries.jl --help
```

## Parameter Control

### Command-Line Override System
**Priority (highest to lowest):**
1. Command-line arguments (`--Nx 60 --Ny 60`)
2. Code defaults in the example file
3. Global defaults from `parse_params.jl`

### Common Parameters

#### Grid & Domain
```bash
--Nx N          # Grid resolution in x direction (default: 40)
--Ny N          # Grid resolution in y direction (default: 40)
--Nz N          # Grid resolution in z direction (default: 20)
--xmin X        # Domain x minimum (default: -0.5)
--xmax X        # Domain x maximum (default: 0.5)
# Similar for --ymin, --ymax, --zmin, --zmax
```

**Note:** Grid spacing `dx`, `dy`, `dz` are computed automatically!

#### Time & Physics
```bash
--tmax T        # Maximum time (default: 0.05)
--CFL C         # CFL number (default: 0.7)
--Ma M          # Mach number (default: 1.0)
--Kn K          # Knudsen number (default: 1.0)
```

#### Snapshots & Diagnostics
```bash
--snapshot-interval N    # Save every N steps (0=disabled, default varies)
--homogeneous-z BOOL     # Z-homogeneous mode (default: false)
```

### Usage Examples

#### Quick Low-Resolution Test
```bash
julia run_3d_jets_timeseries.jl --Nx 20 --Ny 20 --Nz 10 --tmax 0.01 --snapshot-interval 5
```

#### High-Resolution Run
```bash
mpiexec -n 4 julia run_3d_jets_timeseries.jl --Nx 80 --Ny 80 --Nz 40 --tmax 0.2
```

#### Custom Domain
```bash
julia run_3d_jets_timeseries.jl \
  --xmin 0 --xmax 2 --ymin 0 --ymax 2 --zmin 0 --zmax 2 --Nx 80 --Ny 80
```

#### Parameter Sweep
```bash
for Ma in 0.5 0.7 1.0 1.5; do
  julia run_3d_jets_timeseries.jl --Ma $Ma --tmax 0.1 --snapshot-interval 5
done
```

#### Production Run (No Visualization)
```bash
julia run_3d_jets_timeseries.jl --Nx 100 --Ny 100 --tmax 1.0 --snapshot-interval 0
```

## Implementation Details

### Parameter Parsing System
Located in `parse_params.jl`:

```julia
include("parse_params.jl")

# Parse with code defaults + command-line overrides
params = parse_simulation_params(
    Nx = 40,    # Code default, override with --Nx N
    Ny = 40,    # Code default, override with --Ny N
    Nz = 20,    # Code default, override with --Nz N
    tmax = 0.05 # Code default, override with --tmax T
)

# Print parameter summary
print_params_summary(params, rank=rank)
```

### Key Functions
- `parse_simulation_params()`: Merges defaults + code + CLI
- `print_params_summary()`: Pretty-prints all parameters
- `get_default_params()`: Returns global defaults

## Tips & Best Practices

### For Quick Tests
- Use low resolution: `--Nx 20 --Ny 20 --Nz 10`
- Short time: `--tmax 0.01`
- Fewer snapshots: `--snapshot-interval 10`

### For Production
- Use MPI: `mpiexec -n 8 julia run_3d_jets_timeseries.jl`
- Higher resolution: `--Nx 100 --Ny 100 --Nz 50`
- Disable snapshots if not needed: `--snapshot-interval 0`

### For Visualization
- Balance snapshot interval with memory
- Too many snapshots (interval=1) -> uses lots of RAM
- Too few snapshots (interval=20) -> choppy animation
- Sweet spot: `--snapshot-interval 2` to `5`

### For Parameter Sweeps
- Use command-line args in loops
- Save results to different directories
- Consider using `--snapshot-interval 0` for sweep

## Troubleshooting

### Out of Memory
```bash
# Reduce resolution
julia ... --Nx 30 --Ny 30 --Nz 15

# Increase snapshot interval
julia ... --snapshot-interval 10

# Disable snapshots
julia ... --snapshot-interval 0
```

### Simulation Crashes (NaN)
```bash
# Reduce Mach number
julia ... --Ma 0.7

# Reduce CFL
julia ... --CFL 0.5

# Increase resolution
julia ... --Nx 60 --Ny 60 --Nz 30
```

### GLMakie Viewer Doesn't Open
- Check GLMakie is installed: `using Pkg; Pkg.add("GLMakie")`
- On remote systems, enable X11 forwarding: `ssh -Y user@host`
- Use headless mode and visualize locally: `--no-viz true`
- Run on rank 0 only (already handled in MPI examples)

### Performance Issues
- Use MPI for large grids: `mpiexec -n 4 julia run_3d_jets_timeseries.jl --Nx 100 --Ny 100`
- Reduce snapshot frequency: `--snapshot-interval 10`
- Disable diagnostics: `--debug-output false`

## Advanced Usage

### Creating Your Own Example

```julia
using HyQMOM
using MPI

include("parse_params.jl")

MPI.Init()

# Parse parameters with your defaults
params = parse_simulation_params(
    Nx = 40,
    Ny = 40,
    # ... your custom defaults
)

print_params_summary(params, rank=MPI.Comm_rank(MPI.COMM_WORLD))

# Run simulation
if params.snapshot_interval > 0
    # Snapshots are automatically streamed to disk
    snapshot_filename, grid = simulation_runner(params)
else
    M_final, final_time, time_steps, grid = simulation_runner(params)
end

MPI.Finalize()
```

### Modifying Defaults

Edit `parse_params.jl` function `get_default_params()` to change global defaults.

### Adding New Parameters

1. Add to `get_default_params()` in `parse_params.jl`
2. Add to help text in `print_help()`
3. Use in your example with `params.your_new_param`

## References

- Main documentation: `../README.md`
- Parameter details: `parse_params.jl` source
- Grid spacing removal: See commit history (dx/dy/dz now automatic)

## Questions?

Run any example with `--help` to see all available options:
```bash
julia --project=. examples/run_3d_jets_timeseries.jl --help
```

## GPU example

`run_gpu_crossing_jets.jl` — end-to-end GPU run: builds a crossing-jet IC, advances it
on the GPU, and streams snapshots to JLD2 in the canonical schema (so the existing
analysis/visualization consume it unchanged). Runs under the GPU project:

```bash
srun --mpi=pmix -n 1 --gpus=1 julia --project=gpu/gpuenv2 examples/run_gpu_crossing_jets.jl
# overrides: NX, NZ (=1 for 2D), MA, NSTEP, SNAP, OUT (env vars)
```

It prints the exact commands to analyze (`compute_standardized_field`) and visualize
(`interactive_3d_timeseries_streaming`) the output in the main package env. See
`misc/03-running-and-validation.md` and the JLD2-version note in `misc/04-gotchas.md`.

### `rodney_validation_1d.jl` / `rodney_validation_2d.jl` - Uniform-Pressure Validation Cases

Rodney Fox's validation ladder for the 35-moment model (2026-07-02). Both cases
start from **uniform pressure** (p ≡ 1), so the Kn=0 Euler solution is trivial:

- **1D** (`ic_type = :riemann1d`): stationary contact — L: (ρ,u,T)=(1,0,1),
  R: (1000,0,10⁻³). Nothing moves at Kn=0, so any velocity or pressure deviation
  is pure numerical error (exact verification of the Riemann solver; the
  first-order scheme preserves it to machine precision). At Kn>0,
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

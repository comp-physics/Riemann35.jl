# Riemann35.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-order, **realizability-preserving**, GPU- and multi-GPU-capable solver for the
**35-moment 3D HyQMOM** kinetic closure. Riemann35.jl targets low numerical diffusion at
high Mach number (crossing jets, large density ratios) while keeping the moment state
inside the realizable moment cone at every step.

It shares the 35-moment HyQMOM closure with [HyQMOM.jl](https://github.com/comp-physics/HyQMOM.jl)
but is a **distinct method**: a high-order MUSCL spatial scheme in standardized-moment
variables with a convex moment-cone projection, plus GPU-native numerics (custom
in-kernel eigensolvers replacing LAPACK, single-source device kernels, z-slab multi-GPU
with host-staged halo exchange).

## What's distinctive

- **High-order realizability-preserving reconstruction** — order-2 (MUSCL) reconstruction
  in standardized moments with a convex-`R` (delta2star 6×6) projection + scaling limiter,
  so faces and updated states stay realizable. This is the diffusion-reduction core.
- **GPU-native per-cell physics** — flux closure, realizability projection, and wave
  speeds run on the GPU. The wave-speed eig uses a custom batched **real-Schur 4×4** kernel
  and an in-kernel symmetric solver (no LAPACK in the kernel; fp64 throughout).
- **Single-source kernels** — the alloc-free per-cell device functions in `src/*_dev.jl`
  are shared verbatim by the CPU solver (which delegates to them) and the GPU kernels.
- **Multi-GPU** — z-slab domain decomposition with resident fields and host-staged halo
  exchange; the multi-GPU timestep is bit-identical to single-GPU.
- **2D and 3D** — a 2D run is a `nz=1` spatial grid (the velocity space is always 3D);
  single-GPU 2D is validated against MATLAB to machine precision.

## Install

```julia
import Pkg
Pkg.add(url="https://github.com/comp-physics/Riemann35.jl")
```

For the GPU path, use the `gpu/gpuenv2` project (CUDA + MPI). On a fresh machine the
simplest setup needs no system MPI (MPI.jl's bundled MPICH works) and no system CUDA
toolkit (CUDA.jl downloads one):

```bash
julia --project=gpu/gpuenv2 -e 'using Pkg; Pkg.add(["CUDA","MPI","MPIPreferences","StaticArrays"])'
```

See [`misc/01-environment.md`](misc/01-environment.md) for the full (portable + cluster)
setup. Requirements: Julia 1.10+, an NVIDIA GPU + driver (for the GPU path), MPI (only
for multi-GPU).

## Quickstart

```julia
using Riemann35
# run a high-order 3D simulation (CPU); see examples/ and docs/src/quickstart.md
```

GPU single-GPU and multi-GPU timesteps:

```bash
# single GPU
julia --project=gpu/gpuenv2 gpu/validate_residual3d_gpu.jl
# multi-GPU (bundled MPI)
julia --project=gpu/gpuenv2 -e 'using MPI; run(`$(mpiexec()) -n 2 julia --project=gpu/gpuenv2 gpu/validate_timestep3d_mpi.jl`)'
```

## Documentation

- [`misc/`](misc/) — **GPU + MPI developer reference**: environment, architecture,
  every validator + expected result, gotchas, performance.
- [`gpu/README.md`](gpu/README.md) — the GPU/multi-GPU milestone log (how each piece was
  built and validated).
- [`HIGHORDER.md`](HIGHORDER.md) — the high-order spatial scheme: status, usage, limits.
- [`docs/`](docs/) — method notes (diffusion reduction, realizability literature,
  Riemann-solver scope) and `docs/src/` user/dev guides.

## Validation (highlights)

- GPU 3D residual vs the CPU reference: rel ~5e-11.
- Multi-GPU (z-slab) residual + timestep vs single-GPU: **bit-identical** (1/2/4 ranks).
- GPU 2D realizability + flux vs the MATLAB golden: rel **4.4e-16** (machine precision).
- Multi-GPU scaling (2× GPU, resident field + halo-only transfer): ~1.9× compute.

Full tables in [`misc/05-results.md`](misc/05-results.md); reproduce via the `gpu/validate_*.jl`
harnesses (see [`misc/03-running-and-validation.md`](misc/03-running-and-validation.md)).

## License

MIT — see [`license.md`](license.md). Copyright (c) 2025 Georgia Institute of Technology.

This project implements the HyQMOM (hyperbolic quadrature method of moments) closure of
R. O. Fox and collaborators; see [HyQMOM.jl](https://github.com/comp-physics/HyQMOM.jl)
for the base method.

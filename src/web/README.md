# HyQMOM case viewer

Interactive browser viewer for simulation snapshots — density, velocity, second-order
moments / realizability, and more. Just static files; no Julia or JLD2 needed to view.

## Run it

From your laptop, in **one** SSH session that both tunnels the port and serves:

```bash
ssh -L 8000:localhost:8000 phoenix
cd <repo>/output/viz && ./serve.sh
# then open http://localhost:8000/viewer.html in your browser
```

`./serve.sh [port]` defaults to port 8000; pass another (e.g. `./serve.sh 8137`) if it's
busy, and change both `8000`s in the `ssh` command to match. `Ctrl-C` stops the server.
(It serves static files only — fine to run on the login node.)

## Using the viewer

- **Case** — dropdown of available runs (name + resolution + Ma + snapshot count).
- **View**:
  - *Isosurface* — 3D isosurface of the selected field, with a **level** slider,
    **shells** (nested surfaces), and **opacity**.
  - *Realizability* — scatter of the standardized second-order moments (s₁₁₀, s₁₀₁,
    s₀₁₁) inside the elliptope realizability surface.
  - *Slice* — 2D heatmap on an x/y/z plane (or a colored plane embedded in 3D).
- **Field** — density, |velocity|, u/v/w, temperature, pressure, Mach, Tx/Ty/Tz,
  s₁₁₀/s₁₀₁/s₀₁₁, ‖s‖, skewness, kurtosis, S2 margin.
- **Time** slider + play/pause (arrow keys also step time).
- **log** color scale, **lock range** (freeze the color range across time),
  **compare** (two cases side by side with linked cameras).
- View / case / field / time are encoded in the URL — bookmark or share a view.

## Adding cases

The bundle here is generated (git-ignored). Produce or append cases with Riemann35:

```julia
using Riemann35
export_jld2_web("output/runs/myrun.jld2", "output")   # → output/viz/, shows up in the dropdown
```

or auto-export during a run by adding to the sim params:

```julia
snapshot_filename = "output/runs/myrun.jld2",
web_dir           = "output"      # bundle → output/viz/ (works for CPU and GPU runs)
```

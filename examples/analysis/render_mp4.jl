# render_mp4.jl — animate a bubble2d_shedding run to mp4 (density + transverse
# velocity). Streams one snapshot at a time (never holds the whole series).
# Headless: uses CairoMakie (pure software, no GL/xvfb needed).
#
# Usage (viz env with CairoMakie + JLD2):
#   julia --project=<vizenv> examples/analysis/render_mp4.jl <run.jld2> [out.mp4] [stride] [fps]
using JLD2, Printf, CairoMakie
CairoMakie.activate!()

f      = ARGS[1]
outmp4 = length(ARGS) >= 2 ? ARGS[2] : replace(f, r"\.jld2$" => ".mp4")
stride = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
fps    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20

jf = jldopen(f, "r")
p  = jf["meta/params"]
ns = 0; while haskey(jf, "snapshots/" * lpad(ns + 1, 6, "0") * "/M"); global ns += 1; end
frames = collect(1:stride:ns)
M1 = jf["snapshots/000001/M"]
Nx, Ny = size(M1, 1), size(M1, 2)
xmin = get(p, :xmin, 0.0); xmax = get(p, :xmax, 15.0)
ymin = get(p, :ymin, 0.0); ymax = get(p, :ymax, 10.0)
xs = range(xmin, xmax; length = Nx); ys = range(ymin, ymax; length = Ny)

readfield(s) = (M = jf["snapshots/" * lpad(s, 6, "0") * "/M"];
                rho = M[:, :, 1, 1]; (log10.(max.(rho, 1e-30)), M[:, :, 1, 6] ./ rho,
                                      jf["snapshots/" * lpad(s, 6, "0") * "/t"]))

lr0, v0, t0 = readfield(1)
LR = Observable(lr0); V = Observable(v0); TT = Observable(t0)

fig = Figure(size = (1100, 900))
Label(fig[0, 1:2], @sprintf("dense bubble in crossflow — Ma=%.2f, Kn=%.3g, %dx%d",
        get(p,:Ma,0.3), get(p,:Kn,0.33), Nx, Ny), fontsize = 20)
ax1 = Axis(fig[1, 1], title = @lift(@sprintf("log10 density   t = %.2f D/U", $TT)),
           xlabel = "x/D", ylabel = "y/D", aspect = DataAspect())
hm1 = heatmap!(ax1, xs, ys, LR; colormap = :turbo, colorrange = (0.0, 5.0))
Colorbar(fig[1, 2], hm1)
ax2 = Axis(fig[2, 1], title = "transverse velocity v", xlabel = "x/D", ylabel = "y/D",
           aspect = DataAspect())
vmax = 0.15
hm2 = heatmap!(ax2, xs, ys, V; colormap = :balance, colorrange = (-vmax, vmax))
Colorbar(fig[2, 2], hm2)

@printf("rendering %d frames (of %d snaps, stride %d) -> %s\n", length(frames), ns, stride, outmp4)
record(fig, outmp4, frames; framerate = fps) do s
    lr, v, t = readfield(s)
    LR[] = lr; V[] = v; TT[] = t
end
close(jf)
println("done: ", outmp4)

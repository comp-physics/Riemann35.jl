# lift_force.jl — fluctuating transverse-force (lift) diagnostic C_L(t) for the
# cylinder, the central observable of McMullen & Gallis (SAND2024-13841J).
#
# In the 35-moment representation the RAW second moments ARE the momentum-flux
# tensor (M110 = rho*u*v + P_xy, M020 = rho*v^2 + P_yy), so the transverse force
# on the cylinder is the control-volume momentum balance over a box enclosing it:
#
#   F_y(t) = - d/dt ∫_box M010 dA  -  ∮_S (M110 n_x + M020 n_y) dl
#
# (convective + pressure + viscous stress all included; the held-cylinder cells
# carry ~0 momentum so they may sit inside the box). C_L = F_y / (1/2 rho_inf U^2 D).
#
# Usage:
#   julia --project=. examples/analysis/lift_force.jl <run.jld2> \
#         [obst_cx_cell obst_cy_cell box_half_cells dx D rho_inf U]
# Defaults infer the box from the held-cylinder cells if geometry not given.
#
# Reports: mean C_L, RMS of the fluctuating lift C_L', Strouhal (lift-PSD peak),
# and the skewness/kurtosis of C_L' (the paper's NON-GAUSSIAN fluctuation signature).
# Dependency-free (own DFT); writes <run>_lift.txt (t, C_L) for plotting.
using JLD2, Printf

f = ARGS[1]
argn(i, d) = length(ARGS) >= i ? parse(Float64, ARGS[i]) : d

jf = jldopen(f, "r")
ns = 0
while haskey(jf, "snapshots/" * lpad(ns + 1, 6, "0") * "/M"); global ns += 1; end
ns >= 4 || error("need >= 4 snapshots for a lift time series (have $ns)")
M1 = jf["snapshots/000001/M"]
nx, ny = size(M1, 1), size(M1, 2)

# geometry: cell size dx, cylinder D, reference rho_inf, U (nondim defaults 1)
dx    = argn(5, 1.0)
D     = argn(6, 1.0)
rhinf = argn(7, 1.0)
Uinf  = argn(8, 1.0)

# control box around the cylinder. If center/half not given, infer the held-cylinder
# cells (near-constant rho, ~0 velocity) as the low-|v|, off-freestream-rho region.
if length(ARGS) >= 4
    ocx = round(Int, argn(2, nx/2)); ocy = round(Int, argn(3, ny/2)); half = round(Int, argn(4, 30))
else
    # crude auto-detect: densest cluster of near-rest cells in the first snapshot
    rho = M1[:, :, 1, 1]; v = M1[:, :, 1, 6] ./ rho; u = M1[:, :, 1, 2] ./ rho
    speed = sqrt.(u.^2 .+ v.^2)
    mask = speed .< 0.05 * Uinf
    xs = [i for i in 1:nx for j in 1:ny if mask[i, j]]
    ys = [j for i in 1:nx for j in 1:ny if mask[i, j]]
    isempty(xs) && error("could not auto-detect the cylinder; pass obst_cx obst_cy box_half")
    ocx = round(Int, sum(xs)/length(xs)); ocy = round(Int, sum(ys)/length(ys))
    half = round(Int, 2.0 * D / dx)
    @printf("auto-detected cylinder center ~(%d,%d); box half=%d cells (%.1fD)\n", ocx, ocy, half, half*dx/D)
end
i0 = clamp(ocx - half, 2, nx-1); i1 = clamp(ocx + half, 2, nx-1)
j0 = clamp(ocy - half, 2, ny-1); j1 = clamp(ocy + half, 2, ny-1)

# read a snapshot's fields we need (raw 2nd moments + M010)
function fields(s)
    M = jf["snapshots/" * lpad(s, 6, "0") * "/M"]
    (M110 = M[:, :, 1, 7], M020 = M[:, :, 1, 10], M010 = M[:, :, 1, 6],
     t = jf["snapshots/" * lpad(s, 6, "0") * "/t"])
end

# surface flux ∮_S (M110 n_x + M020 n_y) dl over the box boundary
function surf_flux(F)
    sx = 0.0
    @inbounds for j in j0:j1
        sx += (F.M110[i1, j] - F.M110[i0, j]) * dx      # right(+x) - left(-x), dl=dx (=dy)
    end
    sy = 0.0
    @inbounds for i in i0:i1
        sy += (F.M020[i, j1] - F.M020[i, j0]) * dx      # top(+y) - bottom(-y)
    end
    sx + sy
end
box_mom(F) = sum(@view F.M010[i0:i1, j0:j1]) * dx * dx  # ∫_box M010 dA

# assemble the lift time series (central-difference the unsteady term)
snaps = [fields(s) for s in 1:ns]
ts    = [S.t for S in snaps]
flux  = [surf_flux(S) for S in snaps]
mom   = [box_mom(S) for S in snaps]
CL = Float64[]; tt = Float64[]
qref = 0.5 * rhinf * Uinf^2 * D
for k in 2:ns-1
    dmomdt = (mom[k+1] - mom[k-1]) / (ts[k+1] - ts[k-1])
    Fy = -dmomdt - flux[k]
    push!(CL, Fy / qref); push!(tt, ts[k])
end

# statistics
n = length(CL)
mean_CL = sum(CL)/n
Cp = CL .- mean_CL                                       # fluctuating lift
rms = sqrt(sum(Cp.^2)/n)
sk  = (sum(Cp.^3)/n) / (rms^3 + eps())
ku  = (sum(Cp.^4)/n) / (rms^4 + eps())                  # 3.0 = Gaussian

# lift PSD (Hann-windowed DFT) -> Strouhal peak
w = [0.5 - 0.5*cos(2pi*(k-1)/(n-1)) for k in 1:n]
xw = Cp .* w
dt = (tt[end] - tt[1]) / (n - 1)
nf = n ÷ 2
freqs = [(k-1)/(n*dt) for k in 1:nf]                    # f in 1/(D/U) = Strouhal
psd = zeros(nf)
for kf in 1:nf
    re = 0.0; im = 0.0; ω = 2pi*(kf-1)/n
    @inbounds for m in 1:n
        re += xw[m]*cos(ω*(m-1)); im -= xw[m]*sin(ω*(m-1))
    end
    psd[kf] = re*re + im*im
end
order = sortperm(psd[2:end], rev=true) .+ 1
peaks = [(round(freqs[p], digits=3), round(psd[p]/maximum(psd[2:end]), digits=3)) for p in order[1:min(5,length(order))]]

@printf("\n=== LIFT DIAGNOSTIC  %s ===\n", basename(f))
@printf("box: x[%d:%d] y[%d:%d] cells (center %d,%d, half %.1fD)  |  %d snapshots, dt_snap=%.4g D/U\n",
        i0, i1, j0, j1, ocx, ocy, half*dx/D, ns, dt)
@printf("mean C_L      = %+.4e\n", mean_CL)
@printf("RMS C_L'      = %.4e   (fluctuating lift amplitude)\n", rms)
@printf("skewness      = %+.3f\n", sk)
@printf("kurtosis      = %.3f   (3.0 = Gaussian; >3 heavy-tailed / intermittent)\n", ku)
@printf("lift-PSD peaks (Strouhal f*D/U, rel power): %s\n", join(["($(p[1]), $(p[2]))" for p in peaks], "  "))
@printf("   McMullen-Gallis shedding band: f*D/U ~ 0.12-0.145\n")
open(replace(f, r"\.jld2$" => "_lift.txt"), "w") do io
    println(io, "# t  C_L")
    for k in 1:n; @printf(io, "%.6f %.8e\n", tt[k], CL[k]); end
end
println("wrote ", replace(f, r"\.jld2$" => "_lift.txt"))
close(jf)

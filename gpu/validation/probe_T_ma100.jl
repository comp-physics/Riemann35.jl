# probe_T_ma100.jl — Step 1: pick a fixed final time T for the sharpness sweep.
# March ONE 64^3 order-3 (WENO5+theta*-IDP) crossing-jet case and print peak
# density vs cumulative physical time so we can locate the developed-collision
# window (max pile-up).
using CUDA, Printf
include(joinpath(@__DIR__, "..", "timestep3d_order3_gpu.jl")); using .Timestep3DOrder3GPU

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
g = 4
N = 64

need = ["r3d_cross_ma100.f64", "r3d_cross_ma100.meta"]
if any(f -> !isfile(joinpath(DATA, f)), need)
    repo = normpath(joinpath(@__DIR__, "..", ".."))
    dump = joinpath(@__DIR__, "dump_cpu_hiorder3_march.jl")
    println("[setup] crossing vectors missing -> running dump in main env ($repo) ...")
    run(setenv(`$(Base.julia_cmd()) --project=$repo $dump`, ENV))
end
cmeta = split(strip(read(joinpath(DATA, "r3d_cross_ma100.meta"), String)), '\n')
Ma   = parse(Float64, cmeta[1]); rhor = parse(Float64, cmeta[3])
cross = reshape(collect(reinterpret(Float64, read(joinpath(DATA, "r3d_cross_ma100.f64")))), 35, 3)
bg = cross[:, 1]; Mt = cross[:, 2]; Mb = cross[:, 3]
s3max = max(40.0, 4.0 + abs(Ma)/2.0)

function build_crossing_cube(N::Int, g::Int)
    nf = N + 2g; G = zeros(35, nf, nf, nf)
    Csize = floor(Int, 0.1 * N)
    Minb = div(N,2) - Csize; Maxb = div(N,2)
    Mnt  = div(N,2) + 1;     Maxt = div(N,2) + 1 + Csize
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        vec = bg
        if Minb <= i <= Maxb && Minb <= j <= Maxb && Minb <= k <= Maxb; vec = Mb; end
        if Mnt  <= i <= Maxt && Mnt  <= j <= Maxt && Mnt  <= k <= Maxt; vec = Mt; end
        @views G[:, i+g, j+g, k+g] .= vec
    end
    cl(a) = a < 1 ? 1 : (a > N ? N : a)
    @inbounds for c in 1:nf, b in 1:nf, a in 1:nf
        if a <= g || a > g+N || b <= g || b > g+N || c <= g || c > g+N
            @views G[:, a, b, c] .= G[:, cl(a-g)+g, cl(b-g)+g, cl(c-g)+g]
        end
    end
    return G
end

dens3(cube, N, g) = Array(@view cube[1, g+1:g+N, g+1:g+N, g+1:g+N])

dx = 1.0 / N
G = CuArray(build_crossing_cube(N, g))
@printf("=== Probe: Ma=%.0f  N=%d^3  order-3  dx=%.4e  s3max=%.1f ===\n", Ma, N, dx, s3max)
@printf("%-6s %-13s %-13s %-13s\n", "step", "t_cum", "peak_rho", "min_rho")
t = 0.0
d = dens3(G, N, g)
@printf("%-6d %-13.5e %-13.5e %-13.5e\n", 0, t, maximum(d), minimum(d))
chunk = 4
for c in 1:45
    dts = march3d_order3_gpu!(G, dx, Ma, chunk; s3max=s3max)
    global t += sum(dts)
    d = dens3(G, N, g)
    surv = all(isfinite, d) && minimum(d) > 0
    @printf("%-6d %-13.5e %-13.5e %-13.5e %s\n", c*chunk, t, maximum(d), minimum(d), surv ? "" : "<-DEAD")
    if !surv; break; end
end
println("Done probe.")

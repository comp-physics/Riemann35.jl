# dump_cpu_hiorder3_residual.jl — CPU order-3 (WENO5 + θ*-IDP) residual reference for the
# GPU order-3 parity gate. Run in the MAIN package env. Builds a small REALIZABLE box
# (isotropic-Maxwellian IC + mild sinusoid), writes the interior moments and the CPU
# residual_ho_3d_order3! result for BOTH dt=0 (pure WENO5, θ=1) and a nonzero dt (θ*-IDP
# active). Halos are OUTFLOW edge copies — identical to the GPU gate's cube construction.
using Riemann35
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
mkpath(DATA)

n = 24; g = 8; Ma = 2.0; dx = 1.0 / n; s3max = 40.0   # g=8 matches the GPU order-3 halo (HALO3)
dtN = 2.0 * dx                       # large λ=2 → factor-6 bound (6λ=12) drives θ*<1 in steep cells

# --- Realizable interior field U (35,n,n,n): Maxwellian with a mild 3D sinusoid ---
U = zeros(35, n, n, n)
for k in 1:n, j in 1:n, i in 1:n
    x = (i - 0.5) / n; y = (j - 0.5) / n; z = (k - 0.5) / n
    rho  = 1.0 + 0.25 * sin(2π*x) * cos(2π*y) * sin(2π*z)
    u    = 0.30 * sin(2π*x)
    v    = 0.30 * sin(2π*y)
    w    = 0.30 * sin(2π*z)
    C200 = 1.0 + 0.15 * cos(2π*x)
    C020 = 1.0 + 0.15 * cos(2π*y)
    C002 = 1.0 + 0.15 * cos(2π*z)
    U[:, i, j, k] .= InitializeM4_35(rho, u, v, w, C200, 0.0, 0.0, C020, 0.0, C002)
end

# --- CPU haloed array Mcpu (n+2g, n+2g, n, 35), OUTFLOW edge-copy x/y halos ---
Mcpu = zeros(n+2g, n+2g, n, 35)
for k in 1:n, j in 1:n, i in 1:n; @views Mcpu[i+g, j+g, k, :] .= U[:, i, j, k]; end
for k in 1:n
    for j in 1:n+2g, hh in 1:g
        @views Mcpu[hh, j, k, :]     .= Mcpu[g+1, j, k, :]
        @views Mcpu[n+g+hh, j, k, :] .= Mcpu[n+g, j, k, :]
    end
    for i in 1:n+2g, hh in 1:g
        @views Mcpu[i, hh, k, :]     .= Mcpu[i, g+1, k, :]
        @views Mcpu[i, n+g+hh, k, :] .= Mcpu[i, n+g, k, :]
    end
end

# --- CPU order-3 residual for dt=0 and dt=dtN, interior written in GPU (35,n,n,n) layout ---
R = zeros(n+2g, n+2g, n, 35)
residual_ho_3d_order3!(R, Mcpu, n, n, n, g, dx, dx, dx, Ma, 0.0; s3max=s3max)
write(joinpath(DATA, "r3d_ho3_R0.f64"),
      reinterpret(UInt8, vec(Float64.(permutedims(R[g+1:g+n, g+1:g+n, :, :], (4,1,2,3))))))

R = zeros(n+2g, n+2g, n, 35)
residual_ho_3d_order3!(R, Mcpu, n, n, n, g, dx, dx, dx, Ma, dtN; s3max=s3max)
write(joinpath(DATA, "r3d_ho3_RN.f64"),
      reinterpret(UInt8, vec(Float64.(permutedims(R[g+1:g+n, g+1:g+n, :, :], (4,1,2,3))))))

# --- interior moments + meta ---
write(joinpath(DATA, "r3d_ho3_M.f64"), reinterpret(UInt8, vec(Float64.(U))))
write(joinpath(DATA, "r3d_ho3.meta"), "$n\n$dx\n$Ma\n$g\n$dtN\n$s3max\n")
println("wrote r3d_ho3_{M,R0,RN}.f64 + meta (CPU order-3, n=$n g=$g Ma=$Ma dtN=$dtN)")

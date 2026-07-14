# dump_cpu_hiorder3_logjacobi.jl — CPU order-3 reference WITH theta_closed=false,
# for the GPU bisection parity gate. Identical IC/halo to dump_cpu_hiorder3_residual.jl;
# only the flag differs. Writes r3d_ho3_bis_{R0,RN}.f64 (M/meta shared with the raw dump).
using Riemann35
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
mkpath(DATA)

n = 24; g = 8; Ma = 2.0; dx = 1.0 / n; s3max = 40.0
dtN = 2.0 * dx

U = zeros(35, n, n, n)
for k in 1:n, j in 1:n, i in 1:n
    x = (i - 0.5) / n; y = (j - 0.5) / n; z = (k - 0.5) / n
    rho  = 1.0 + 0.25 * sin(2π*x) * cos(2π*y) * sin(2π*z)
    u    = 0.30 * sin(2π*x); v = 0.30 * sin(2π*y); w = 0.30 * sin(2π*z)
    C200 = 1.0 + 0.15 * cos(2π*x); C020 = 1.0 + 0.15 * cos(2π*y); C002 = 1.0 + 0.15 * cos(2π*z)
    U[:, i, j, k] .= InitializeM4_35(rho, u, v, w, C200, 0.0, 0.0, C020, 0.0, C002)
end
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
R = zeros(n+2g, n+2g, n, 35)
residual_ho_3d_order3!(R, Mcpu, n, n, n, g, dx, dx, dx, Ma, 0.0; s3max=s3max, theta_closed=false)
write(joinpath(DATA, "r3d_ho3_bis_R0.f64"),
      reinterpret(UInt8, vec(Float64.(permutedims(R[g+1:g+n, g+1:g+n, :, :], (4,1,2,3))))))
R = zeros(n+2g, n+2g, n, 35)
residual_ho_3d_order3!(R, Mcpu, n, n, n, g, dx, dx, dx, Ma, dtN; s3max=s3max, theta_closed=false)
write(joinpath(DATA, "r3d_ho3_bis_RN.f64"),
      reinterpret(UInt8, vec(Float64.(permutedims(R[g+1:g+n, g+1:g+n, :, :], (4,1,2,3))))))
# ensure shared M/meta exist (same as raw dump)
isfile(joinpath(DATA,"r3d_ho3_M.f64")) || write(joinpath(DATA, "r3d_ho3_M.f64"), reinterpret(UInt8, vec(Float64.(U))))
isfile(joinpath(DATA,"r3d_ho3.meta"))  || write(joinpath(DATA, "r3d_ho3.meta"), "$n\n$dx\n$Ma\n$g\n$dtN\n$s3max\n")
println("wrote r3d_ho3_bis_{R0,RN}.f64 (CPU order-3 bisection, n=$n g=$g Ma=$Ma)")

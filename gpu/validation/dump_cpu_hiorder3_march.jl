# dump_cpu_hiorder3_march.jl — CPU reference for the GPU order-3 SSP-RK3 march gate.
# Run in the MAIN package env.  Produces:
#   (Gate A) a small REALIZABLE Ma=2 box marched K steps with `step_highorder_3d!`
#            (order=3, :copy BC, no stage-BGK) — the EXACT operator the GPU
#            `march3d_order3_gpu!` reproduces — dumping the interior IC and the
#            interior after K steps, plus the constant dt used.
#   (Gate B) the three 35-moment vectors of the Ma=100 :crossing_matlab IC
#            (background, top jet, bottom jet) built with the real InitializeM4_35,
#            so the pure-GPU Ma=100 driver can assemble the cube faithfully.
using Riemann35
using MPI
MPI.Initialized() || MPI.Init()

DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
mkpath(DATA)

# ===========================================================================
# Gate A reference — step_highorder_3d! (order=3) marched K steps.
# ===========================================================================
n = 16; g = 4; Ma = 2.0; dx = 1.0 / n; s3max = 40.0; K = 4

# realizable interior IC: Maxwellian + mild 3D sinusoid (same recipe as the
# residual gate) — realizable, exercises WENO5 + θ*-IDP.
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

# constant, stable dt from the IC (same speed formula as the GPU CFL helper).
vmax = 0.0
for k in 1:n, j in 1:n, i in 1:n
    r = U[1,i,j,k]
    ui = U[2,i,j,k]/r; vi = U[6,i,j,k]/r; wi = U[16,i,j,k]/r
    cx = max(U[3,i,j,k]/r - ui*ui, 0.0)
    cy = max(U[10,i,j,k]/r - vi*vi, 0.0)
    cz = max(U[20,i,j,k]/r - wi*wi, 0.0)
    sp = max(abs(ui),abs(vi),abs(wi)) + 4.0*2.334*sqrt(max(cx,cy,cz)+1e-12)
    global vmax = max(vmax, sp)
end
dt = (1.0/3.0) * dx / max(vmax, 1e-12)

# CPU haloed array M (n+2g, n+2g, n, 35): x/y halos are :copy edge copies filled
# by halo_exchange inside step_highorder_3d!; z is padded per line by the residual.
halo = g
M = zeros(n+2halo, n+2halo, n, 35)
for k in 1:n, j in 1:n, i in 1:n; @views M[i+halo, j+halo, k, :] .= U[:, i, j, k]; end
decomp = setup_mpi_cartesian_3d(n, n, n, halo, MPI.COMM_WORLD)

for _ in 1:K
    step_highorder_3d!(M, dt, decomp, :copy, n, n, n, halo, dx, dx, dx, Ma;
                       order=3, s3max=s3max)
end

MK = zeros(35, n, n, n)
for k in 1:n, j in 1:n, i in 1:n; @views MK[:, i, j, k] .= M[i+halo, j+halo, k, :]; end

write(joinpath(DATA, "r3d_march_M0.f64"), reinterpret(UInt8, vec(Float64.(U))))
write(joinpath(DATA, "r3d_march_MK.f64"), reinterpret(UInt8, vec(Float64.(MK))))
write(joinpath(DATA, "r3d_march.meta"), "$n\n$dx\n$Ma\n$g\n$dt\n$K\n$s3max\n")
println("wrote r3d_march_{M0,MK}.f64 + meta (CPU step_highorder_3d! order=3, n=$n Ma=$Ma dt=$dt K=$K)")

# ===========================================================================
# Gate B — Ma=100 :crossing_matlab moment vectors (background / top / bottom).
# ===========================================================================
MaB = 100.0; rhol = 1.0; rhor = 0.05; T = 1.0
r110 = 0.0; r101 = 0.0; r011 = 0.0
C200 = T; C020 = T; C002 = T
C110 = r110 * sqrt(C200*C020); C101 = r101 * sqrt(C200*C002); C011 = r011 * sqrt(C020*C002)
Uc = MaB / sqrt(3.0)
bg = InitializeM4_35(rhor, 0.0, 0.0, 0.0, C200, C110, C101, C020, C011, C002)
Mt = InitializeM4_35(rhol, -Uc, -Uc, -Uc, C200, C110, C101, C020, C011, C002)
Mb = InitializeM4_35(rhol,  Uc,  Uc,  Uc, C200, C110, C101, C020, C011, C002)
cross = hcat(Float64.(bg), Float64.(Mt), Float64.(Mb))   # (35, 3)
write(joinpath(DATA, "r3d_cross_ma100.f64"), reinterpret(UInt8, vec(cross)))
write(joinpath(DATA, "r3d_cross_ma100.meta"), "$MaB\n$rhol\n$rhor\n$T\n$Uc\n")
println("wrote r3d_cross_ma100.f64 (bg/top/bottom 35-moment vectors, Ma=$MaB rhol=$rhol rhor=$rhor)")

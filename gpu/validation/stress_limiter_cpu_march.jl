# stress_limiter_cpu_march.jl — CPU half of the limiter stress test (MAIN env).
#
# Question (from review): on a REAL case where the scaling limiter engages heavily and the
# realizability eigenvalues are sensitive (Ma=100 colliding jets, 1000:1 density), do CPU and
# GPU evolve to MEANINGFULLY different physical states — or is the CPU/GPU divergence just the
# already-accepted eigensolver-floor noise that the default (non-limiter) path also has?
#
# This script marches the colliding-jets IC for NSTEP fixed-dt SSP-RK3 steps on the CPU, both
# with the limiter (use_limiter=true) and with the default path, and writes the final DENSITY
# fields (+ the IC and the dt sequence) for the GPU half to reproduce EXACTLY. Density (M000)
# is the well-conditioned physical observable; the conserved mass is also reported.
#
#   env <singleton vars> $JULIA --project=. gpu/validation/stress_limiter_cpu_march.jl
# ENV: NSTRESS (grid n, default 20), NSTEP (default 20), MA (default 100), VACF (default 0.001)
using Riemann35, Printf
DATA = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
n    = parse(Int,   get(ENV, "NSTRESS", "20"))
nstep= parse(Int,   get(ENV, "NSTEP",   "20"))
Ma   = parse(Float64, get(ENV, "MA",    "100"))
vacf = parse(Float64, get(ENV, "VACF",  "0.001"))
g = 2; dx = 1.0/n
Riemann35.HO_VACUUM_FLOOR[] = vacf

# ---- IC: Rodney's colliding jets (diagonal-Gaussian per cell, 1000:1, Uc=Ma/sqrt3) ----
const TRIPLES = [
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),
 (1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),
 (1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),
 (0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2)]
_gm(m,mu,s2) = m==0 ? 1.0 : m==1 ? mu : m==2 ? mu^2+s2 : m==3 ? mu^3+3mu*s2 : mu^4+6mu^2*s2+3s2^2
function setcell!(Mint,i,j,k,rho,u,v,w,T)
    @inbounds for q in 1:35
        a,b,c = TRIPLES[q]; Mint[q,i,j,k] = rho*_gm(a,u,T)*_gm(b,v,T)*_gm(c,w,T)
    end
end
Mint = Array{Float64}(undef,35,n,n,n)
Uc = Ma/sqrt(3.0); Cs = floor(Int,0.1*n); h = div(n,2)
for k in 1:n, j in 1:n, i in 1:n; setcell!(Mint,i,j,k, 1.0e-3, 0.0,0.0,0.0, 1.0); end
for k in (h-Cs):h, j in (h-Cs):h, i in (h-Cs):h;          setcell!(Mint,i,j,k, 1.0,  Uc, Uc, Uc, 1.0); end
for k in (h+1):(h+1+Cs), j in (h+1):(h+1+Cs), i in (h+1):(h+1+Cs); setcell!(Mint,i,j,k, 1.0, -Uc,-Uc,-Uc, 1.0); end

# fixed, stable dt (constant): CFL*dx/(1.5*Ma)
dt = (1.0/3.0)*dx/(1.5*Ma); dts = fill(dt, nstep)

# ---- self-contained CPU SSP-RK3 march (mirrors step_highorder_3d!) with outflow halos ----
function refill_xy!(Mc,n,g)
    for k in 1:n
        for j in 1:n+2g, hh in 1:g; @views Mc[hh,j,k,:].=Mc[g+1,j,k,:]; @views Mc[n+g+hh,j,k,:].=Mc[n+g,j,k,:]; end
        for i in 1:n+2g, hh in 1:g; @views Mc[i,hh,k,:].=Mc[i,g+1,k,:]; @views Mc[i,n+g+hh,k,:].=Mc[i,n+g,k,:]; end
    end
end
function project!(Mc,n,g,Ma)
    for k in 1:n, j in 1:n, i in 1:n; Mc[i+g,j+g,k,:] = realizable_3D_M4(Mc[i+g,j+g,k,:], Ma); end
end
function march(Mint, nstep, dts, n, g, dx, Ma; use_limiter)
    Mc = zeros(n+2g,n+2g,n,35)
    for k in 1:n, j in 1:n, i in 1:n; @views Mc[i+g,j+g,k,:].=Mint[:,i,j,k]; end
    R = zeros(n+2g,n+2g,n,35); int = (g+1:g+n, g+1:g+n, 1:n, :)
    L!(M) = (refill_xy!(M,n,g); residual_ho_3d!(R,M,n,n,n,g,dx,dx,dx,Ma; order=2, use_limiter=use_limiter); R)
    for s in 1:nstep
        dt = dts[s]; M0 = copy(Mc)
        L!(Mc); @views Mc[int...] .= M0[int...] .+ dt.*R[int...];                       project!(Mc,n,g,Ma)
        L!(Mc); @views Mc[int...] .= (3/4).*M0[int...] .+ (1/4).*(Mc[int...].+dt.*R[int...]); project!(Mc,n,g,Ma)
        L!(Mc); @views Mc[int...] .= (1/3).*M0[int...] .+ (2/3).*(Mc[int...].+dt.*R[int...]); project!(Mc,n,g,Ma)
    end
    rho = Array{Float64}(undef,n,n,n)
    for k in 1:n, j in 1:n, i in 1:n; rho[i,j,k] = Mc[i+g,j+g,k,1]; end
    return rho
end

@printf("CPU stress march: n=%d nstep=%d Ma=%.0f dt=%.3e vacf=%.3g\n", n,nstep,Ma,dt,vacf)
rho_lim = march(Mint, nstep, dts, n, g, dx, Ma; use_limiter=true)
rho_def = march(Mint, nstep, dts, n, g, dx, Ma; use_limiter=false)
@printf("  CPU limiter: rho range [%.4e, %.4e] mass=%.6e\n", minimum(rho_lim), maximum(rho_lim), sum(rho_lim))
@printf("  CPU default: rho range [%.4e, %.4e] mass=%.6e\n", minimum(rho_def), maximum(rho_def), sum(rho_def))

open(joinpath(DATA,"stress.meta"),"w") do io; println(io,n); println(io,nstep); println(io,Ma); println(io,vacf); println(io,dx); end
write(joinpath(DATA,"stress_M0.f64"),  reinterpret(UInt8, vec(Mint)))
write(joinpath(DATA,"stress_dts.f64"), reinterpret(UInt8, vec(dts)))
write(joinpath(DATA,"stress_cpu_lim.f64"), reinterpret(UInt8, vec(rho_lim)))
write(joinpath(DATA,"stress_cpu_def.f64"), reinterpret(UInt8, vec(rho_def)))
println("wrote stress_{meta,M0,dts,cpu_lim,cpu_def}")

# stagebgk_cpu_march.jl — CPU half of the stage_bgk + pressure_recon GPU validation (MAIN env).
#
# Marches the uniform-pressure stationary contact (rho 1|1000, T 1|1e-3, u=0, p≡1 —
# Rodney's 1D validation case) with a self-contained SSP-RK3 march that mirrors
# step_highorder_3d! (same structure as stress_limiter_cpu_march.jl), in three modes:
#   def    — flags off (regression: GPU default must still match CPU bitwise-ish)
#   pbk0   — pressure_recon + stage_bgk at Kn=0   (contact must stay machine-exact)
#   pbk    — pressure_recon + stage_bgk at Kn=0.01 (finite-Kn e=exp(-dt/tc) exercised)
# Writes the IC, dt sequence, and final 35-moment interior fields for the GPU half.
#
#   $JULIA --project=. gpu/validation/stagebgk_cpu_march.jl
# ENV: NVAL (grid n, default 16), NSTEP (default 15)
using Riemann35, Printf
DATA  = get(ENV, "RIEMANN35_DATA", joinpath(@__DIR__, "..", "..", "data"))
n     = parse(Int, get(ENV, "NVAL",  "16"))
nstep = parse(Int, get(ENV, "NSTEP", "15"))
g = 2; dx = 1.0/n
Riemann35.HO_VACUUM_FLOOR[] = 0.001

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
# uniform-pressure contact along x: left = (1,0,1), right = (1000,0,1e-3), p = 1
Mint = Array{Float64}(undef,35,n,n,n)
for k in 1:n, j in 1:n, i in 1:n
    if i <= div(n,2); setcell!(Mint,i,j,k, 1.0,    0.0,0.0,0.0, 1.0)
    else;             setcell!(Mint,i,j,k, 1000.0, 0.0,0.0,0.0, 1.0e-3)
    end
end
# fixed stable dt: hot-side speed ~ 4*2.334*sqrt(1) ≈ 9.34
dt = (1.0/3.0)*dx/10.0; dts = fill(dt, nstep)

function refill_xy!(Mc,n,g)
    for k in 1:n
        for j in 1:n+2g, hh in 1:g; @views Mc[hh,j,k,:].=Mc[g+1,j,k,:]; @views Mc[n+g+hh,j,k,:].=Mc[n+g,j,k,:]; end
        for i in 1:n+2g, hh in 1:g; @views Mc[i,hh,k,:].=Mc[i,g+1,k,:]; @views Mc[i,n+g+hh,k,:].=Mc[i,n+g,k,:]; end
    end
end
function project!(Mc,n,g,Ma)
    for k in 1:n, j in 1:n, i in 1:n; Mc[i+g,j+g,k,:] = realizable_3D_M4(Mc[i+g,j+g,k,:], Ma); end
end
function bgkstage!(Mc,n,g,dt,kn)
    for k in 1:n, j in 1:n, i in 1:n
        Mt = ntuple(q -> Mc[i+g,j+g,k,q], Val(35))
        out = Riemann35.bgk_relax_tup(Mt, dt, kn)
        for q in 1:35; Mc[i+g,j+g,k,q] = out[q]; end
    end
end
function march(Mint, nstep, dts, n, g, dx; prec::Bool, kn)   # kn=nothing → no stage BGK
    Ma = 0.0
    Riemann35.HO_PRESSURE_RECON[] = prec
    Mc = zeros(n+2g,n+2g,n,35)
    for k in 1:n, j in 1:n, i in 1:n; @views Mc[i+g,j+g,k,:].=Mint[:,i,j,k]; end
    R = zeros(n+2g,n+2g,n,35); int = (g+1:g+n, g+1:g+n, 1:n, :)
    L!(M) = (refill_xy!(M,n,g); residual_ho_3d!(R,M,n,n,n,g,dx,dx,dx,Ma; order=2); R)
    bgk!(M,dt) = kn === nothing ? nothing : bgkstage!(M,n,g,dt,kn)
    for s in 1:nstep
        dt = dts[s]; M0 = copy(Mc)
        L!(Mc); @views Mc[int...] .= M0[int...] .+ dt.*R[int...];                            project!(Mc,n,g,Ma); bgk!(Mc,dt)
        L!(Mc); @views Mc[int...] .= (3/4).*M0[int...] .+ (1/4).*(Mc[int...].+dt.*R[int...]); project!(Mc,n,g,Ma); bgk!(Mc,dt)
        L!(Mc); @views Mc[int...] .= (1/3).*M0[int...] .+ (2/3).*(Mc[int...].+dt.*R[int...]); project!(Mc,n,g,Ma); bgk!(Mc,dt)
    end
    Riemann35.HO_PRESSURE_RECON[] = false
    out = Array{Float64}(undef,35,n,n,n)
    for k in 1:n, j in 1:n, i in 1:n, q in 1:35; out[q,i,j,k] = Mc[i+g,j+g,k,q]; end
    return out
end

@printf("CPU stagebgk march: n=%d nstep=%d dt=%.3e\n", n, nstep, dt)
M_def  = march(Mint, nstep, dts, n, g, dx; prec=false, kn=nothing)
M_pbk0 = march(Mint, nstep, dts, n, g, dx; prec=true,  kn=0.0)
M_pbk  = march(Mint, nstep, dts, n, g, dx; prec=true,  kn=0.01)
maxu(M) = maximum(abs, M[2,:,:,:] ./ M[1,:,:,:])
@printf("  CPU def : mass=%.10e max|u|=%.3e\n", sum(M_def[1,:,:,:]),  maxu(M_def))
@printf("  CPU pbk0: mass=%.10e max|u|=%.3e (contact-exactness: expect ~1e-16)\n", sum(M_pbk0[1,:,:,:]), maxu(M_pbk0))
@printf("  CPU pbk : mass=%.10e max|u|=%.3e\n", sum(M_pbk[1,:,:,:]),  maxu(M_pbk))

mkpath(DATA)
open(joinpath(DATA,"stagebgk.meta"),"w") do io; println(io,n); println(io,nstep); println(io,dx); end
write(joinpath(DATA,"stagebgk_M0.f64"),   reinterpret(UInt8, vec(Mint)))
write(joinpath(DATA,"stagebgk_dts.f64"),  reinterpret(UInt8, vec(dts)))
write(joinpath(DATA,"stagebgk_cpu_def.f64"),  reinterpret(UInt8, vec(M_def)))
write(joinpath(DATA,"stagebgk_cpu_pbk0.f64"), reinterpret(UInt8, vec(M_pbk0)))
write(joinpath(DATA,"stagebgk_cpu_pbk.f64"),  reinterpret(UInt8, vec(M_pbk)))
println("wrote stagebgk_{meta,M0,dts,cpu_def,cpu_pbk0,cpu_pbk}")

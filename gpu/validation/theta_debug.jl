# theta_debug.jl — dissect a few closed-form failures: for each axis eval standardized
# minors D1,D2,D3 as functions of θ at θc (closed) and θb (bisection), and show which
# minor is negative (state non-realizable) or which root the cubic missed.
using Riemann35, Printf, Random
include(joinpath(@__DIR__, "..", "..", "src", "numerics", "idp_limiter_dev.jl"))
using .IdpLimiterDev
const ID = IdpLimiterDev
const _real = ID.RiemannFluxDev._state_realizable
const MI = ID.RiemannFluxDev.RoePS3Dev.MomentIndices.MARG_IDX

# standardized minors for one axis marginal (m0..m4) — same transform as _axis_theta_bound
function minors_axis(m)
    a0,a1,a2,a3,a4 = m
    a0<=0 && return (a0, NaN, NaN)
    u=a1/a0
    ca2=a2-2u*a1+u*u*a0; ca3=a3-3u*a2+3u*u*a1-u^3*a0; ca4=a4-4u*a3+6u*u*a2-4u^3*a1+u^4*a0
    σ2=ca2/a0; s=σ2>0 ? 1/sqrt(σ2) : 1.0
    sa2=ca2*s^2; sa3=ca3*s^3; sa4=ca4*s^4
    D1=a0; D2=a0*sa2; D3=a0*(sa2*sa4-sa3^2)-0.0+sa2*(0.0-sa2*sa2)  # det std Hankel, m1=0
    (D1,D2,D3)
end
axis_m(M,ax,θ,dM)=ntuple(j->M[MI[ax][j]]+θ*dM[MI[ax][j]], 5)

randchain(rng)=(n=rand(rng,2:3); w=rand(rng,n).+0.05; x=(rand(rng,n).-0.5).*8; ntuple(k->sum(w[i]*x[i]^(k-1) for i in 1:n),5))
function assemble(rng)
    M=zeros(35); ρ=0.3+rand(rng)
    for ax in 1:3; c=randchain(rng); s=ρ/c[1]; for j in 1:5; M[MI[ax][j]]=c[j]*s; end; end
    ntuple(i->M[i],Val(35))
end

rng=MersenneTwister(11); shown=0
for t in 1:300000
    Mlo=assemble(rng); _real(Mlo) || continue
    dv=zeros(35); for ax in 1:3,j in 1:5; dv[MI[ax][j]]=(rand(rng)-0.5)*abs(Mlo[MI[ax][j]]+0.5)*8; end
    dM=ntuple(i->dv[i],Val(35))
    θc=ID.theta_star_update_closed(Mlo,dM); θb=ID.theta_star_update_dev(Mlo,dM)
    isunder = θb - θc > 1e-3   # closed OVER-LIMITS: returns θc far below bisection θb
    if isunder && shown<5
        global shown+=1
        mid=0.5*(θc+θb)
        @printf("\n=== OVER-LIMIT #%d  θc=%.6f θb=%.6f  _real@θb=%s _real@mid=%s ===\n",
                shown, θc, θb, _real(ntuple(i->Mlo[i]+θb*dM[i],Val(35))), _real(ntuple(i->Mlo[i]+mid*dM[i],Val(35))))
        for ax in 1:3
            Dc=minors_axis(axis_m(Mlo,ax,θc,dM)); Db=minors_axis(axis_m(Mlo,ax,θb,dM)); D0=minors_axis(axis_m(Mlo,ax,0.0,dM))
            @printf(" ax%d  θ=0:(%.3e,%.3e,%.3e)  θc:(%.3e,%.3e,%.3e)  θb:(%.3e,%.3e,%.3e)\n",
                    ax, D0..., Dc..., Db...)
        end
    end
    shown>=5 && break
end

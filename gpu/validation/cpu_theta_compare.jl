# cpu_theta_compare.jl — closed-form θ* vs bisection on GENERAL (non-Gaussian) marginal
# chains (built from random node measures, so realizable + arbitrary shape). θ* reads
# only the 15 marginal slots (MARG_IDX), so cross slots are irrelevant. Find the bug.
using Riemann35, Printf, Random
include(joinpath(@__DIR__, "..", "..", "src", "numerics", "idp_limiter_dev.jl"))
using .IdpLimiterDev: theta_star_update_dev, theta_star_update_closed
const _real = IdpLimiterDev.RiemannFluxDev._state_realizable
const MI = IdpLimiterDev.RiemannFluxDev.RoePS3Dev.MomentIndices.MARG_IDX

# realizable raw marginal chain (m0..m4) from n weighted nodes
randchain(rng) = (n=rand(rng,2:3); w=rand(rng,n).+0.05; x=(rand(rng,n).-0.5).*8;
                  ntuple(k->sum(w[i]*x[i]^(k-1) for i in 1:n), 5))
# assemble a 35-vec with the 3 axes' marginal chains in MARG_IDX slots (share m0=ρ), rest 0
function assemble(rng)
    M=zeros(35); ρ=0.3+rand(rng)
    for ax in 1:3
        c=randchain(rng); s=ρ/c[1]                     # scale so m0=ρ
        for j in 1:5; M[MI[ax][j]] = c[j]*s; end
    end
    ntuple(i->M[i], Val(35))
end
rng=MersenneTwister(11)
worst=0.0; nbad=0; nover=0; nover_nonreal=0; ex=0
for t in 1:300000
    Mlo=assemble(rng)
    _real(Mlo) || continue
    # random strong dM on the marginal slots (general direction)
    dv=zeros(35); for ax in 1:3, j in 1:5; dv[MI[ax][j]] = (rand(rng)-0.5)*abs(Mlo[MI[ax][j]]+0.5)*8; end
    dM=ntuple(i->dv[i], Val(35))
    θc=theta_star_update_closed(Mlo,dM); θb=theta_star_update_dev(Mlo,dM)
    d=abs(θc-θb); d>worst && (global worst=d)
    if d>1e-3
        global nbad+=1
        θc>θb+1e-9 && (global nover+=1)
        stc=ntuple(i->Mlo[i]+θc*dM[i], Val(35))
        (θc>θb+1e-9 && !_real(stc)) && (global nover_nonreal+=1)
        if ex<8
            global ex+=1
            @printf("θc=%.6f θb=%.6f Δ=%.2e dir=%s closed-state real? %s\n",
                    θc,θb,d, θc>θb ? ">θb OVERSHOOT" : "<θb over-limit", _real(stc))
        end
    end
end
@printf("\nworst |θc-θb| = %.3e over 3e5 general-marginal samples\n", worst)
@printf("disagreements(>1e-3): %d ; θc>θb (non-conservative): %d ; NON-realizable at θc: %d\n", nbad, nover, nover_nonreal)

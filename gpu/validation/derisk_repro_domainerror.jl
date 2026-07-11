# derisk_repro_domainerror.jl (SCRATCH) — CPU reproduction of the anchor-ON crash.
# chyqmom_nodes_3d_dev is pure scalar Julia (device-safe ⇒ CPU-runnable). Feed it
# non-realizable 35-moment states (perturbed Maxwellians) and catch the DomainError
# to PINPOINT the unguarded sqrt — no GPU, no ptxas. Confirms root cause.
include(joinpath(@__DIR__, "..", "chyqmom_nodes_3d_dev.jl"))
using .KFVSInversionDev: chyqmom_nodes_3d_dev
using Printf, Random
Random.seed!(12345)

# a realizable Maxwellian 35-moment vector (unit variance, zero mean), ρ set by caller.
# raw moments of N(0,1) per axis, independent: central = raw here (mean 0).
# Build via the standard 35 ordering used in _KFVSA_TRIPLES.
const TRIP = (
 (0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
 (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
 (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))
# 1D standard-normal raw moments m0..m4 = 1,0,1,0,3
sm(n) = (1.0,0.0,1.0,0.0,3.0)[n+1]
function maxwellian(rho, ux, uy, uz, T)
    # independent Gaussians each axis, variance T, mean u.  raw moment of axis =
    # sum_k C(e,k) mean^(e-k) * central_k ; use central moments of N(0,T): 1,0,T,0,3T^2
    cm = (1.0, 0.0, T, 0.0, 3.0*T*T)
    axisraw(e, mu) = sum(binomial(e,k)*mu^(e-k)*cm[k+1] for k in 0:e)
    M = ntuple(35) do n
        (i,j,k) = TRIP[n]
        rho * axisraw(i,ux) * axisraw(j,uy) * axisraw(k,uz)
    end
    return M
end

# sanity: a clean Maxwellian must invert without error
M0 = maxwellian(1.0, 0.3, -0.2, 0.1, 1.0)
try
    r = chyqmom_nodes_3d_dev(M0); @printf("clean Maxwellian OK: Nn=%d\n", r[5])
catch e
    println("!! clean Maxwellian THREW: ", e); rethrow()
end

# fuzz: perturb the high/cross moments (indices >5) to push conditional variances
# negative, and catch the first DomainError with its stacktrace.
ndomain = 0; nother = 0; first_bt = nothing
for t in 1:200000
    scale = 0.2 + 2.0*rand()   # perturbation magnitude
    Mp = ntuple(35) do n
        n <= 5 ? M0[n] : M0[n] * (1.0 + scale*(2.0*rand()-1.0))
    end
    try
        chyqmom_nodes_3d_dev(Mp)
    catch e
        if e isa DomainError
            global ndomain += 1
            if first_bt === nothing
                global first_bt = catch_backtrace()
                println("\n=== FIRST DomainError at trial $t (scale=$(round(scale,digits=3))) ===")
                println(e)
                for (i,fr) in enumerate(stacktrace(first_bt))
                    println("  [$i] ", fr)
                    i >= 8 && break
                end
            end
        else
            global nother += 1
        end
    end
end
@printf("\nfuzz done: DomainError=%d  other-exceptions=%d / 200000 trials\n", ndomain, nother)

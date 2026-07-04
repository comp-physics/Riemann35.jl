"""
    riemann_flux_dev.jl — the interface Riemann flux, single-source CPU/GPU.

One implementation of the numerical flux F(MLr, MRr, FL, FR, sL, sR) for every
`riemann_solver` selector value, consumed by BOTH the CPU `face_flux_1d`
(`src/numerics/highorder_flux.jl`) and the GPU `_face_flux_core`
(`gpu/residual3d_gpu.jl`). Also the single source of the selector-symbol →
code mapping and its error message (`rs_code`). Operation order matches the
previous per-platform implementations exactly (byte-identical results).

Inputs are the already realizability/hyperbolicity-corrected face states and
their physical fluxes for the given axis, plus the HLL wave-speed bounds.
"""
module RiemannFluxDev

include(joinpath(@__DIR__, "roeps3_dev.jl"))
using .RoePS3Dev: roeps3_diss_dev
using .RoePS3Dev.MomentIndices: MARG_IDX

# standardized shape parameters (q̂ = central-3rd/σ³, K = central-4th/σ⁴) of a
# 5-moment marginal chain — the dimensionless "is this side Maxwellian-like,
# and does the shape match the other side" numbers used by the contact gate.
@inline function _marg_shape(w1, w2, w3, w4, w5)
    u = w2 / w1
    c2 = w3 / w1 - u * u
    c2 > 0.0 || return false, 0.0, 0.0
    m3c = w4 / w1 - 3u * (w3 / w1) + 2u^3
    m4c = w5 / w1 - 4u * (w4 / w1) + 6u^2 * (w3 / w1) - 3u^4
    s3 = c2 * sqrt(c2)
    return true, m3c / s3, m4c / (c2 * c2)
end

# necessary realizability of a 35-moment state via its three axis marginals:
# ρ > 0, σ² > 0 and the Hamburger bound K ≥ 1 + q̂² on each 5-moment chain.
# Cheap, device-safe; catches every poison mode diagnosed on the Ma=100
# crossing jets (negative mass, high-moment kicks below the moment cone).
@inline function _state_realizable(m::NTuple{35,Float64})
    (isfinite(m[1]) && m[1] > 0.0) || return false
    for ax in 1:3
        idx = MARG_IDX[ax]
        ok, qh, K = _marg_shape(m[idx[1]], m[idx[2]], m[idx[3]], m[idx[4]], m[idx[5]])
        (ok && isfinite(K) && isfinite(qh) && K - 1.0 - qh * qh > 0.0) || return false
    end
    return true
end


export riemann_flux_dev, rs_code

"selector symbol -> device code (single source of the accepted values)"
rs_code(s::Symbol) =
    s === :hll         ? 0 :
    s === :rusanov     ? 1 :
    s === :roeps3      ? 2 :
    s === :roeps3theta ? 3 :
    throw(ArgumentError("unknown riemann_solver=$(s); available: :hll (default), :rusanov, :roeps3, :roeps3theta"))

# HLL flux (device-safe; the θ* limiter's realizability-preserving baseline)
@inline function _hll_flux(MLr::NTuple{35,Float64}, MRr::NTuple{35,Float64},
                           FL::NTuple{35,Float64}, FR::NTuple{35,Float64},
                           sL::Float64, sR::Float64)
    if sL >= 0.0
        return FL
    elseif sR <= 0.0
        return FR
    else
        den = sR - sL; ss = sL * sR
        return ntuple(Val(35)) do j
            (sR * FL[j] - sL * FR[j] + ss * (MRr[j] - MLr[j])) / den
        end
    end
end

# Are BOTH HLL-form intermediate states of the blended flux
# F(θ) = F_HLL + θ(F̂ − F_HLL) realizable? Device-safe helper (no captured
# closure — must compile into CUDA kernels). See θ* limiter below.
@inline _blend_flux(θ::Float64, FH::NTuple{35,Float64}, Fhat::NTuple{35,Float64}) =
    ntuple(Val(35)) do j
        FH[j] + θ * (Fhat[j] - FH[j])
    end

@inline function _blend_realizable(θ::Float64,
                                   MLr::NTuple{35,Float64}, MRr::NTuple{35,Float64},
                                   FL::NTuple{35,Float64}, FR::NTuple{35,Float64},
                                   FH::NTuple{35,Float64}, Fhat::NTuple{35,Float64},
                                   sL::Float64, sR::Float64)
    msL = ntuple(Val(35)) do j
        MLr[j] + ((FH[j] + θ * (Fhat[j] - FH[j])) - FL[j]) / sL
    end
    _state_realizable(msL) || return false
    msR = ntuple(Val(35)) do j
        MRr[j] + ((FH[j] + θ * (Fhat[j] - FH[j])) - FR[j]) / sR
    end
    _state_realizable(msR)
end

@inline function riemann_flux_dev(rs::Int, axis::Int,
                                  MLr::NTuple{35,Float64}, MRr::NTuple{35,Float64},
                                  FL::NTuple{35,Float64}, FR::NTuple{35,Float64},
                                  sL::Float64, sR::Float64)
    if rs == 1
        # Rusanov / local Lax-Friedrichs: 0.5(FL+FR) - 0.5*max(|sL|,|sR|)*(MR-ML)
        a = max(abs(sL), abs(sR))
        return ntuple(Val(35)) do j
            0.5 * (FL[j] + FR[j]) - 0.5 * a * (MRr[j] - MLr[j])
        end
    end
    if rs == 2 || rs == 3
        # rs==2 (:roeps3): RoePS3 with the BINARY realizability backstop —
        # accept F̂ if both HLL-form intermediate states are realizable, else
        # fall back to HLL.
        # rs==3 (:roeps3theta): the SAME gate and dissipation, but the binary
        # backstop is replaced by the EXACT convex limiter θ*
        # (roe1d/theta-star-derivation.md, notes §10.7): return
        # F_HLL + θ*(F̂ − F_HLL) with θ* the largest θ∈[0,1] keeping BOTH bar
        # states realizable. θ*=1 recovers rs==2's accept; θ*=0 its HLL
        # fallback; θ*∈(0,1) keeps the largest admissible fraction of the sharp
        # flux the binary backstop discarded. θ* is found here by bisection
        # (the bar states are affine in θ so the admissible set is the interval
        # [0,θ*] — bisection converges to its edge); the closed-form Hankel-
        # pencil solve of the derivation is the O(1) production alternative.
        # θ* sits BEHIND the shape gate, not instead of it: the gate catches
        # poison mode 1 (anti-diffusive mass-row term, invisible to the
        # marginal realizability test), θ* sharpens mode 2 (near-vacuum cone
        # excursions) — the two defenses are orthogonal (§9(m), §10.7).
        # RoePS3 parity-split treatment, applied ONLY at contact-like faces
        # (small velocity and pressure jumps relative to the wave scale) —
        # exactly where its validated benefits live (contact exactness +
        # sharpness, the parity theorem). Everywhere else: HLL. Central-form
        # scalar dissipation is NOT robust at violently asymmetric faces —
        # the Ma=100 crossing-jets stress test kills it (and the pre-existing
        # :rusanov) in 2 steps, vacuum floor or not: HLL survives because its
        # dissipation carries the dF-proportional upwinding term.
        if sL < 0.0 && sR > 0.0
            # CONTACT gate: the parity-split treatment is applied only where a
            # true contact structure exists across the face —
            #   (i)  the FULL velocity vector continuous (normal-only admitted
            #        the crossing-jets shear faces);
            #   (ii) the pressure continuous;
            #   (iii) the STANDARDIZED SHAPE (q̂ = heat flux/σ³, K = kurtosis)
            #        of the face-normal marginal continuous. A contact joins
            #        two near-equilibrium states: q̂≈0, K≈3 on BOTH sides, so
            #        this costs nothing there. The Ma=100 crossing-jets
            #        interpenetration faces pass (i)+(ii) exactly (same ρ,u,p)
            #        but carry O(1) jumps in q̂ and K (different beam mixes →
            #        bimodal marginals, K≈1.4); through the wave-resolved
            #        matrix dissipation such a jump feeds an ANTI-diffusive
            #        mass-row term ~|λ|Δw₄/σ³ that dwarfs a·Δρ by ~1e7 and
            #        drives near-vacuum densities negative in 3 steps
            #        (diagnosed face-by-face, 16³ reproducer, 2026-07-03).
            wjump = 0.05 * (sR - sL)
            uL1 = MLr[2] / MLr[1];  uR1 = MRr[2] / MRr[1]
            uL2 = MLr[6] / MLr[1];  uR2 = MRr[6] / MRr[1]
            uL3 = MLr[16] / MLr[1]; uR3 = MRr[16] / MRr[1]
            pL = (MLr[3] + MLr[10] + MLr[20]) / 3 -
                 (MLr[2]^2 + MLr[6]^2 + MLr[16]^2) / (3 * MLr[1]^2) * MLr[1]
            pR = (MRr[3] + MRr[10] + MRr[20]) / 3 -
                 (MRr[2]^2 + MRr[6]^2 + MRr[16]^2) / (3 * MRr[1]^2) * MRr[1]
            idx = MARG_IDX[axis]
            okL, qhL, KL = _marg_shape(MLr[idx[1]], MLr[idx[2]], MLr[idx[3]],
                                       MLr[idx[4]], MLr[idx[5]])
            okR, qhR, KR = _marg_shape(MRr[idx[1]], MRr[idx[2]], MRr[idx[3]],
                                       MRr[idx[4]], MRr[idx[5]])
            if abs(uR1 - uL1) <= wjump && abs(uR2 - uL2) <= wjump &&
               abs(uR3 - uL3) <= wjump &&
               abs(pR - pL) <= 0.2 * min(pL, pR) && pL > 0 && pR > 0 &&
               okL && okR && abs(qhR - qhL) <= 0.5 && abs(KR - KL) <= 0.5
                D = roeps3_diss_dev(MLr, MRr, axis, sL, sR)
                ok = true
                for j in 1:35
                    isfinite(D[j]) || (ok = false; break)
                end
                if ok
                    Fhat = ntuple(Val(35)) do j
                        0.5 * (FL[j] + FR[j]) - 0.5 * D[j]
                    end
                    # REALIZABILITY BACKSTOP (the literature recipe: limit
                    # toward a realizability-preserving baseline flux): accept
                    # the parity-split flux only if both HLL-form intermediate
                    # states  m*_L = m_L + (F̂−F_L)/s_L,  m*_R = m_R + (F̂−F_R)/s_R
                    # stay in the moment cone (necessary marginal conditions).
                    # HLL passes by construction (both equal the Riemann-fan
                    # average); at an exact contact F̂ is the exact flux whose
                    # intermediate states are physical — so exactness is kept.
                    # This is what actually protects the near-vacuum cells: on
                    # the Ma=100 crossing jets, gate-admitted faces still fed
                    # >10x high-moment kicks to ρ~1e-3 cells (non-realizable →
                    # closure wave speeds explode → NaN by step 3).
                    mstarL = ntuple(Val(35)) do j
                        MLr[j] + (Fhat[j] - FL[j]) / sL
                    end
                    mstarR = ntuple(Val(35)) do j
                        MRr[j] + (Fhat[j] - FR[j]) / sR
                    end
                    bothok = _state_realizable(mstarL) && _state_realizable(mstarR)
                    if rs == 2
                        bothok && return Fhat
                    else
                        # rs==3: θ* limiter. θ=1 admissible ⇒ no limiting.
                        if bothok
                            return Fhat
                        end
                        FH = _hll_flux(MLr, MRr, FL, FR, sL, sR)
                        # bisect for θ* on [0,1]; θ=0 (HLL) is realizable by
                        # convexity, θ=1 is not (just failed) ⇒ single edge.
                        lo = 0.0; hi = 1.0
                        for _ in 1:20
                            mid = 0.5 * (lo + hi)
                            if _blend_realizable(mid, MLr, MRr, FL, FR, FH, Fhat, sL, sR)
                                lo = mid
                            else
                                hi = mid
                            end
                        end
                        # blend via a helper (θ a plain arg) — do NOT capture the
                        # loop-mutated `lo` in a closure, or Julia boxes it and
                        # the kernel fails to compile.
                        return _blend_flux(lo, FH, Fhat)
                    end
                end
            end
        end
        # fall through to HLL
    end
    # HLL (default)
    if sL >= 0.0
        return FL
    elseif sR <= 0.0
        return FR
    else
        den = sR - sL
        ss  = sL * sR
        return ntuple(Val(35)) do j
            (sR * FL[j] - sL * FR[j] + ss * (MRr[j] - MLr[j])) / den
        end
    end
end

end # module

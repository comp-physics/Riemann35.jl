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

export riemann_flux_dev, rs_code

"selector symbol -> device code (single source of the accepted values)"
rs_code(s::Symbol) =
    s === :hll     ? 0 :
    s === :rusanov ? 1 :
    s === :roeps3  ? 2 :
    throw(ArgumentError("unknown riemann_solver=$(s); available: :hll (default), :rusanov, :roeps3"))

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
    if rs == 2
        # RoePS3 parity-split dissipation (roeps3_dev.jl)
        D = roeps3_diss_dev(MLr, MRr, axis, sL, sR)
        return ntuple(Val(35)) do j
            0.5 * (FL[j] + FR[j]) - 0.5 * D[j]
        end
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

"""
    hiorder3_recon_dev.jl — SINGLE-SOURCE device-safe order-3 reconstruction primitives.

The per-cell / per-interface WENO5 + realizability-scaling math shared VERBATIM by
the CPU `residual_line3` (src/numerics/highorder_3d.jl) and the GPU order-3 kernels
(gpu/residual3d_gpu.jl) — one source, two consumers (the riemann_flux_dev.jl pattern).
Pure NTuple arithmetic, no allocation, no throw / NaN-on-bad-input: GPU-compilable.

Pipeline, per axis line (all four steps device-safe):
  1. recon_point_dev  : 5 raw cell-avg NTuples → recon-var POINT value (smooth5-gated
                        deconv5 per component, then to_recon_vars_dev).
  2. recon_avg_dev    : 5 recon-var point NTuples → recon-var cell AVERAGE (conv5).
  3. weno_faces_dev   : the 6 recon-var averages straddling an interface + the two raw
                        cell means → the two realizability-scaled raw face states.
  4. weno_scaled_face_dev : continuous Zhang–Shu / Fan–Huang–Wu scaling of one WENO face.
"""
module HiOrder3ReconDev

include(joinpath(@__DIR__, "weno5_dev.jl"));       using .Weno5Dev: weno5z, deconv5, conv5, smooth5
include(joinpath(@__DIR__, "recon_dev.jl"));       using .ReconDev: to_recon_vars_dev, from_recon_vars_dev,
    to_recon_vars_tup, from_recon_vars_tup
include(joinpath(@__DIR__, "riemann_flux_dev.jl")); using .RiemannFluxDev: _state_realizable

export recon_point_dev, recon_avg_dev, weno_faces_dev, weno_scaled_face_dev,
       recon_vars_realizable

# Validity of a reconstructed recon-var vector (density slot 1 = M000, variances
# slots 5,6,7 = C200,C020,C002). from_recon_vars_dev takes sqrt of the three
# variances, so any ≤ 0 would sqrt a negative — test BEFORE converting (the GPU has
# no throw to catch, only a silent NaN).
@inline recon_vars_realizable(v::NTuple{35,Float64}) =
    v[1] > 0.0 && v[5] > 0.0 && v[6] > 0.0 && v[7] > 0.0

# Step 1: smooth5-gated deconvolution of a 5-cell RAW-moment stencil → recon-var
# point value. Per component: deconv5 where smooth, else keep the cell average.
@inline recon_point_dev(cm2::NTuple{35,Float64}, cm1::NTuple{35,Float64},
                        c0::NTuple{35,Float64},  cp1::NTuple{35,Float64},
                        cp2::NTuple{35,Float64}) = to_recon_vars_tup(
    ntuple(q -> smooth5(cm2[q], cm1[q], c0[q], cp1[q], cp2[q]) ?
                deconv5(cm2[q], cm1[q], c0[q], cp1[q], cp2[q]) : c0[q], Val(35)))

# Step 2: forward-convolve a 5-cell recon-var POINT stencil → recon-var cell average.
@inline recon_avg_dev(pm2::NTuple{35,Float64}, pm1::NTuple{35,Float64},
                      p0::NTuple{35,Float64},  pp1::NTuple{35,Float64},
                      pp2::NTuple{35,Float64}) =
    ntuple(q -> conv5(pm2[q], pm1[q], p0[q], pp1[q], pp2[q]), Val(35))

# Step 4: continuous Zhang–Shu / Fan–Huang–Wu realizability scaling of ONE WENO face.
# Blend the recon-var face `vW` toward the cell's recon-var vector `Vc` (raw image =
# realizable cell mean) by the LARGEST θ ∈ [0,1] keeping the converted raw face
# realizable: v(θ) = Vc + θ(vW − Vc). θ = 1 (smooth) recovers the unlimited face
# byte-for-byte; θ → 0 only at the realizability front. Admissible set taken [0,θ*]
# (θ = 0 = cell mean is realizable). Device-safe: NTuple, bisection, NO throw / NaN.
@inline function weno_scaled_face_dev(vW::NTuple{35,Float64}, Vc::NTuple{35,Float64},
                                      cmean::NTuple{35,Float64}; nb::Int = 20)
    if recon_vars_realizable(vW)
        mW = from_recon_vars_tup(vW)
        _state_realizable(mW) && return mW
    end
    lo = 0.0; hi = 1.0
    for _ in 1:nb
        mid = 0.5 * (lo + hi)
        vθ = ntuple(q -> Vc[q] + mid * (vW[q] - Vc[q]), Val(35))
        ok = recon_vars_realizable(vθ) && _state_realizable(from_recon_vars_tup(vθ))
        ok ? (lo = mid) : (hi = mid)
    end
    # `lo` is loop-mutated; copy into a fresh, never-reassigned binding before the
    # final ntuple so the closure captures a plain Float64 (a captured loop-mutated
    # variable is boxed → dynamic dispatch → InvalidIR on the GPU). CPU-identical.
    θf = lo
    vθf = ntuple(q -> Vc[q] + θf * (vW[q] - Vc[q]), Val(35))
    recon_vars_realizable(vθf) ? from_recon_vars_tup(vθf) : cmean
end

# Step 3: the two realizability-scaled raw face states at an interface. W1..W6 are
# the recon-var cell averages Vavg[il-2 .. il+3] straddling the interface between
# cells il (left) and ir = il+1 (right); cL, cR are the two cells' raw means.
# Left face  (right-going WENO at il): weno5z(W1,W2,W3,W4,W5).
# Right face (left-going  WENO at ir): weno5z(W6,W5,W4,W3,W2).
@inline function weno_faces_dev(W1::NTuple{35,Float64}, W2::NTuple{35,Float64},
                                W3::NTuple{35,Float64}, W4::NTuple{35,Float64},
                                W5::NTuple{35,Float64}, W6::NTuple{35,Float64},
                                cL::NTuple{35,Float64}, cR::NTuple{35,Float64})
    vL = ntuple(q -> weno5z(W1[q], W2[q], W3[q], W4[q], W5[q]), Val(35))
    vR = ntuple(q -> weno5z(W6[q], W5[q], W4[q], W3[q], W2[q]), Val(35))
    mL = weno_scaled_face_dev(vL, to_recon_vars_tup(cL), cL)
    mR = weno_scaled_face_dev(vR, to_recon_vars_tup(cR), cR)
    (mL, mR)
end

end # module

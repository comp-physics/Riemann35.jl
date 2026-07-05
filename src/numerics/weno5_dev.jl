"""
    weno5_dev.jl — device-safe WENO5-Z + deconvolution/convolution stencils.

Pure scalar arithmetic on Float64 (NTuple-friendly, CUDA-safe). WENO5-Z is
Borges et al. (2008). deconv5/conv5 are the O(dx^6) cell-average <-> point-value
pair (needed because reconstruction in nonlinear recon variables from cell
averages caps at 2nd order otherwise). smooth5 gates the deconvolution near jumps.
"""
module Weno5Dev

export weno5z, deconv5, conv5, smooth5

# cell average -> cell-center point value (undivided differences), O(dx^6)
@inline deconv5(vm2, vm1, v0, vp1, vp2) =
    v0 - (1/24) * (vp1 - 2v0 + vm1) + (3/640) * (vm2 - 4vm1 + 6v0 - 4vp1 + vp2)
# point value -> cell average (forward operator)
@inline conv5(vm2, vm1, v0, vp1, vp2) =
    v0 + (1/24) * (vp1 - 2v0 + vm1) - (17/5760) * (vm2 - 4vm1 + 6v0 - 4vp1 + vp2)

"per-cell smoothness gate: relative curvature below tol on all inputs."
@inline function smooth5(a, b, c, d, e; tol = 0.05)
    s = abs(a) + 2*abs(c) + abs(e) + 1e-300
    abs(a - 2c + e) / s <= tol && abs(b - 2c + d) / s <= tol
end

"""
WENO5-Z reconstruction (Borges et al. 2008), value at the RIGHT face of the
center cell from its 5-cell average stencil (mirror args for the left face).
Nonlinear (shock-capturing): the Z-weights bias away from a non-smooth substencil,
so at a discontinuity the reconstruction is essentially non-oscillatory — this is
the whole point (a linear interpolant would ring). On smooth data the weights
approach the optimal (0.1,0.6,0.3) and the scheme is 5th order.
"""
@inline function weno5z(vm2, vm1, v0, vp1, vp2)
    q0 = (2vm2 - 7vm1 + 11v0) / 6
    q1 = (-vm1 + 5v0 + 2vp1) / 6
    q2 = (2v0 + 5vp1 - vp2) / 6
    b0 = (13/12)*(vm2 - 2vm1 + v0)^2 + (1/4)*(vm2 - 4vm1 + 3v0)^2
    b1 = (13/12)*(vm1 - 2v0 + vp1)^2 + (1/4)*(vm1 - vp1)^2
    b2 = (13/12)*(v0 - 2vp1 + vp2)^2 + (1/4)*(3v0 - 4vp1 + vp2)^2
    t5 = abs(b0 - b2); ep = 1e-40
    a0 = 0.1 * (1 + t5/(b0+ep)); a1 = 0.6 * (1 + t5/(b1+ep)); a2 = 0.3 * (1 + t5/(b2+ep))
    (a0*q0 + a1*q1 + a2*q2) / (a0 + a1 + a2)
end

end # module

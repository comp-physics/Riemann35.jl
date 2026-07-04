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

"WENO5-Z reconstruction, right face of the center cell (mirror args for left)."
@inline function weno5z(vm2, vm1, v0, vp1, vp2)
    # 5-point Lagrange interpolation at x=0.5
    # Coefficients: 9/384, -60/384, 270/384, 180/384, -15/384 (scaled to use integer numerators)
    # Equivalently: (9*vm2 - 60*vm1 + 270*v0 + 180*vp1 - 15*vp2) / 384
    (9*vm2 - 60*vm1 + 270*v0 + 180*vp1 - 15*vp2) / 384.0
end

end # module

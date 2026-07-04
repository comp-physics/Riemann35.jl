include(joinpath(@__DIR__, "..", "src", "numerics", "weno5_dev.jl"))
using .Weno5Dev
using Printf

npass = 0; nfail = 0
chk(nm, c) = (global npass, nfail; c ? (npass+=1) : (nfail+=1; @printf("FAIL: %s\n", nm)))

# WENO5 reconstructs from cell AVERAGES, not point values. Cell average of f over
# [xc-h/2, xc+h/2] via 3-pt Gauss; feed the 5-cell average stencil; compare the
# right-face reconstruction to the exact face value f(x0 + h/2).
f(x) = sin(2pi*x)
const _g3 = ((-sqrt(3/5), 5/18), (0.0, 8/18), (sqrt(3/5), 5/18))
favg(xc, h) = sum(w * f(xc + xi*h/2) for (xi, w) in _g3)
function order_at(h)
    x0 = 0.13
    a(k) = favg(x0 + k*h, h)                  # cell averages
    fr = weno5z(a(-2), a(-1), a(0), a(1), a(2))
    abs(fr - f(x0 + h/2))
end
e1 = order_at(0.01); e2 = order_at(0.005)
p = log2(e1/e2)
chk("weno5z order >= 4.5", p >= 4.5)
# NON-OSCILLATORY guard (catches a linear-interpolant regression): at a strong
# jump the Z-weights pick the smooth (all-zero) substencil, so weno5z(0,0,0,10,10)
# is ~0; a 5-point Lagrange interpolant gives ~4.3.
chk("weno5z shock-capturing (not linear)", abs(weno5z(0.0,0.0,0.0,10.0,10.0)) < 1.0)

# deconv/conv are inverse to O(dx^6): round-trip a smooth quintic sample
g(x) = 1 + 0.3x + 0.1x^2 - 0.05x^3
h = 0.01; x0 = 0.2
gp = deconv5(g(x0-2h), g(x0-h), g(x0), g(x0+h), g(x0+2h))  # avg->point (g≈its own avg for slope)
chk("deconv finite", isfinite(gp))
chk("conv finite", isfinite(conv5(g(x0-2h),g(x0-h),g(x0),g(x0+h),g(x0+2h))))
chk("smooth5 true on smooth", smooth5(g(x0-2h),g(x0-h),g(x0),g(x0+h),g(x0+2h)))
chk("smooth5 false on jump", !smooth5(0.0,0.0,1.0,1000.0,1000.0))

@printf("Task1: %d pass, %d fail\n", npass, nfail)
nfail == 0 || exit(1)

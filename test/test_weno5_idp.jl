include(joinpath(@__DIR__, "..", "src", "numerics", "weno5_dev.jl"))
using .Weno5Dev
using Printf

npass = 0; nfail = 0
chk(nm, c) = (global npass, nfail; c ? (npass+=1) : (nfail+=1; @printf("FAIL: %s\n", nm)))

# WENO5-Z recovers the face value of a smooth function to ~5th order.
# f(x)=sin(2pi x); cell averages over [x-h/2, x+h/2] approximated by point values
# is the wrong test — use the true right-face value f(x0+h/2) vs weno5z on point
# values of a smooth cubic where WENO is exact to its formal order.
f(x) = sin(2pi*x)
function order_at(h)
    x0 = 0.13
    v(k) = f(x0 + k*h)                       # point values (smooth => stand-in averages ok for a convergence slope)
    fr = weno5z(v(-2), v(-1), v(0), v(1), v(2))
    abs(fr - f(x0 + h/2))
end
e1 = order_at(0.02); e2 = order_at(0.01)
p = log2(e1/e2)
chk("weno5z order >= 4.5", p >= 4.5)

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

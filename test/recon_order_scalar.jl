"""
recon_order_scalar.jl — isolate the reconstruction order of the order-3 pipeline
(deconv5-gated → conv5 → weno5z), on a SCALAR smooth function, free of the
35-moment machinery and the residual-difference metric.

Reports L1 AND L∞ convergence order of the right-face reconstruction vs the exact
face value, for:
  (a) weno5z alone (WENO5-Z on cell averages)
  (b) the full deconv5→conv5→weno5z pipeline (what residual_line3 uses per component)
on f(x)=sin(2πx) (HAS critical points at x=¼,¾) and on a critical-point-free
window measurement. L1≈5 & L∞≈3-4 ⟹ benign WENO-Z critical-point reduction;
both ≈4 ⟹ a genuine order cap in the pipeline.

Run:  \$JULIA --project=. test/recon_order_scalar.jl
"""

include(joinpath(@__DIR__, "..", "src", "numerics", "weno5_dev.jl"))
using .Weno5Dev: weno5z, deconv5, conv5, smooth5
using Printf
using Statistics: mean

f(x) = sin(2pi*x)
# exact cell average of f over [xc-h/2, xc+h/2]  (∫ analytic)
favg(xc, h) = (cos(2pi*(xc-h/2)) - cos(2pi*(xc+h/2))) / (2pi*h)

# right-face reconstruction at interface between cell i and i+1, two variants.
weno_only(a, i) = weno5z(a(i-2), a(i-1), a(i), a(i+1), a(i+2))
function pipeline(a, i)
    # deconv (smooth-gated, always smooth here) → point values; conv back → averages; weno
    ppt(k) = smooth5(a(k-2),a(k-1),a(k),a(k+1),a(k+2)) ?
                 deconv5(a(k-2),a(k-1),a(k),a(k+1),a(k+2)) : a(k)
    vavg(k) = conv5(ppt(k-2), ppt(k-1), ppt(k), ppt(k+1), ppt(k+2))
    weno5z(vavg(i-2), vavg(i-1), vavg(i), vavg(i+1), vavg(i+2))
end

# errors at resolution n (periodic), returns (L1, Linf) over all interfaces
function errs(n, recon)
    h = 1.0/n
    a(i) = favg((mod(i-1, n) + 0.5)*h, h)     # periodic cell average, cell i center (i-0.5)h
    e = Float64[]
    for i in 1:n
        xf = i*h                               # right face of cell i
        push!(e, abs(recon(a, i) - f(xf)))
    end
    (mean(e), maximum(e))
end

# errors EXCLUDING cells near the critical points x=1/4,3/4 (|f'|≈0)
function errs_noncrit(n, recon)
    h = 1.0/n
    a(i) = favg((mod(i-1, n) + 0.5)*h, h)
    e = Float64[]
    for i in 1:n
        xf = i*h
        # skip a window of width 0.06 around each critical point
        (abs(xf-0.25) < 0.06 || abs(xf-0.75) < 0.06) && continue
        push!(e, abs(recon(a, i) - f(xf)))
    end
    (mean(e), maximum(e))
end

ordr(e1, e2, r) = log(e1/e2)/log(r)
const NS = [32, 64, 128, 256]

function table(name, recon, errfn)
    println("\n  $name")
    @printf("  %-6s %-12s %-8s %-12s %-8s\n", "n", "L1", "L1_ord", "Linf", "Linf_ord")
    prev = nothing
    for n in NS
        l1, li = errfn(n, recon)
        if prev === nothing
            @printf("  %-6d %-12.3e %-8s %-12.3e %-8s\n", n, l1, "—", li, "—")
        else
            @printf("  %-6d %-12.3e %-8.2f %-12.3e %-8.2f\n", n, l1,
                    ordr(prev[1], l1, 2), li, ordr(prev[2], li, 2))
        end
        prev = (l1, li)
    end
end

println("="^64)
println("RECONSTRUCTION ORDER (scalar f=sin2πx; L1 vs L∞ isolates crit-points)")
println("="^64)
table("weno5z alone — full domain (incl. critical points):", weno_only, errs)
table("full deconv→conv→weno pipeline — full domain:",        pipeline,  errs)
table("full pipeline — EXCLUDING critical-point windows:",    pipeline,  errs_noncrit)
println("\n  Read: L1≈5 & Linf≈3-4 on full domain, but ≈5 excluding crit pts")
println("  ⟹ benign WENO-Z critical-point reduction (design order intact).")
println("  Both L1 and non-crit ≈4 ⟹ genuine cap — investigate deconv/conv.")

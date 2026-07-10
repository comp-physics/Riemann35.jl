# logjacobi_recon_validation.jl — CPU validation of the OPT-IN log-Jacobi marginal
# reconstruction (use_logjacobi_recon) in the order-3 WENO5 path.
#
# Three checks (the port's make-or-break decision metrics):
#   1. BYTE-IDENTITY OFF: residual_ho_3d_order3! with the flag OFF must be
#      byte-for-byte identical to the flag not existing (relL2 == 0.0 exactly).
#   2. ORDER RETAINED: on smooth non-Gaussian marginal data, flag-ON marginal
#      reconstruction must still be ~5th order (deconvolution pipeline ported).
#   3. CONTACT FIDELITY: on a 3D moving uniform-p x-contact, measure the induced
#      spurious velocity/pressure flag-ON vs flag-OFF (the payoff number).
#
# Run: julia --project=<wt-logjacobi> gpu/validation/logjacobi_recon_validation.jl
using Riemann35
using Riemann35: residual_ho_3d_order3!
using Riemann35.LogJacobiReconDev: marg_m_to_J, marg_J_to_m, logjacobi_marginal_faces
using Printf, LinearAlgebra

# ---- build a haloed 3D array (nx+2g, ny+2g, nz, 35) with outflow x/y halos ----
function haloed(n, g, ic)
    M = zeros(n+2g, n+2g, n, 35)
    for k in 1:n, j in 1:n, i in 1:n; @views M[i+g, j+g, k, :] .= ic(i,j,k); end
    refill_outflow!(M, n, g)
    M
end
function refill_outflow!(M, n, g)
    for k in 1:n
        for j in 1:n+2g, hh in 1:g
            @views M[hh, j, k, :]     .= M[g+1, j, k, :]
            @views M[n+g+hh, j, k, :] .= M[n+g, j, k, :]
        end
        for i in 1:n+2g, hh in 1:g
            @views M[i, hh, k, :]     .= M[i, g+1, k, :]
            @views M[i, n+g+hh, k, :] .= M[i, n+g, k, :]
        end
    end
end
interior(R, n, g) = R[g+1:g+n, g+1:g+n, :, :]

# =========================== 1. BYTE-IDENTITY OFF ===========================
function test_byte_identity()
    println("=== 1. BYTE-IDENTITY (flag OFF vs baseline) ===")
    n = 20; g = 8; Ma = 10.0; dx = 1.0/n; s3max = 40.0
    ic(i,j,k) = begin
        x=(i-0.5)/n; y=(j-0.5)/n; z=(k-0.5)/n
        InitializeM4_35(1.0+0.25sin(2π*x)*cos(2π*y), 0.3sin(2π*x), 0.3cos(2π*y), 0.2sin(2π*z),
                        1.0+0.15cos(2π*x), 0.0, 0.0, 1.0+0.15cos(2π*y), 0.0, 1.0+0.15cos(2π*z))
    end
    M = haloed(n, g, ic)
    allexact = true
    for dt in (0.0, 2.0*dx)
        Rref = zeros(size(M)); Roff = zeros(size(M))
        residual_ho_3d_order3!(Rref, M, n,n,n, g, dx,dx,dx, Ma, dt; s3max=s3max)
        residual_ho_3d_order3!(Roff, M, n,n,n, g, dx,dx,dx, Ma, dt; s3max=s3max, use_logjacobi_recon=false)
        d = interior(Roff,n,g) .- interior(Rref,n,g)
        rel = norm(vec(d)) / max(norm(vec(interior(Rref,n,g))), 1e-300)
        allexact &= (rel == 0.0)
        @printf("   dt=%.4g  relL2(off vs baseline) = %.3e   %s\n",
                dt, rel, rel == 0.0 ? "EXACT" : "*** NONZERO ***")
    end
    println(allexact ? "   -> byte-identical OFF: PASS\n" : "   -> byte-identical OFF: FAIL\n")
    allexact
end

# =============================== 2. ORDER =================================
# Manufacture a smooth, genuinely non-Gaussian x-marginal chain (varying skew C300
# and excess kurtosis) so a2,b3 in J actually move. Feed the SAME marginal pipeline
# the residual uses (logjacobi_marginal_faces) and measure L∞ of the reconstructed
# left-face m0 vs the exact face value. Refine; report order.
function test_order()
    println("=== 2. ORDER (flag ON, marginal m0 left-face value) ===")
    m0f(x)=1.0+0.3sinpi(2x); uf(x)=0.4cospi(2x); Tf(x)=1.0+0.2sinpi(2x)
    q3f(x)=0.15sinpi(2x); k4extra(x)=0.10cospi(2x)      # skew C300, excess kurtosis over 3T^2
    # exact raw marginal moments at point x
    function mpt(x)
        r=m0f(x); u=uf(x); T=Tf(x); c3=q3f(x); c4=3T^2+k4extra(x)
        (r, r*u, r*(T+u^2), r*(c3+3u*T+u^3), r*(c4+4u*c3+6u^2*T+u^4))
    end
    # exact cell average via 8-pt Gauss
    gx=(-0.9602898564975363,-0.7966664774136267,-0.525532409916329,-0.1834346424956498,
         0.1834346424956498,0.525532409916329,0.7966664774136267,0.9602898564975363)
    gw=(0.1012285362903763,0.2223810344533745,0.313706645877887,0.3626837833783620,
        0.3626837833783620,0.313706645877887,0.2223810344533745,0.1012285362903763)
    function cellavg(xc,h)
        acc=zeros(5)
        for (xi,wi) in zip(gx,gw); a=mpt(xc+0.5h*xi); acc .+= 0.5wi .* collect(a); end
        (acc[1],acc[2],acc[3],acc[4],acc[5])
    end
    function err(n)
        g=8; h=1.0/n; n2g=n+2g
        Mmarg=[cellavg((mod(k-g-1,n)+0.5)*h, h) for k in 1:n2g]   # periodic
        ok,L,R = logjacobi_marginal_faces(Mmarg, g)
        ok || return NaN
        e=0.0
        for f in 1:n
            xf = (f-1)*h                  # interface f is at x=(f-1)*h; L[f] = left-of-interface m0
            e = max(e, abs(L[f][1] - m0f(xf)))
        end
        e
    end
    prev=NaN; println("   n      L∞(m0 face)     order")
    for n in (32,64,128,256)
        e=err(n); ord=isnan(prev) ? NaN : log2(prev/e)
        @printf("   %-5d  %.4e      %5.2f\n", n, e, ord); prev=e
    end
    println("   (expect ~5; a cap at 2 means the deconvolution pipeline is NOT ported)\n")
end

# =========================== 3. CONTACT FIDELITY ===========================
# 3D moving uniform-p contact along x (rho 1->1000 at x=0.5, u=(0.5,0,0), uniform p).
# Manual SSP-RK3 march using residual_ho_3d_order3! directly (outflow halos refilled
# + realizability projection each stage), flag OFF vs ON. Metric: max spurious
# |u_x - 0.5| and |p - p0| over interior cells (contact-pollution metric).
function test_contact()
    println("=== 3. CONTACT FIDELITY (3D moving uniform-p x-contact, rho 1:1000) ===")
    n = 24; g = 8; Ma = 1.0; u0 = 0.5; p0 = 1.0; ratio = 1000.0; s3max = 40.0
    ic(i,j,k) = begin
        rho = i <= n÷2 ? 1.0 : ratio; T = p0/rho
        InitializeM4_35(rho, u0, 0.0, 0.0, T, 0.0, 0.0, T, 0.0, T)
    end
    dx = 1.0/n
    vmax = u0 + 4.0*sqrt(p0)
    dt = 0.2 * dx / vmax
    nsteps = 15
    results = Tuple{String,Float64,Float64}[]
    for (nm, flag) in (("flag OFF (raw WENO5)", false), ("flag ON (log-Jacobi)", true))
        M = haloed(n, g, ic)
        for _ in 1:nsteps
            march_step!(M, n, g, dx, Ma, dt, s3max, flag)
        end
        umax, pmax = contact_metrics(M, n, g, u0, p0)
        push!(results, (nm, umax, pmax))
        @printf("   %-24s  max|u_x-0.5| = %.3e   max|p-p0| = %.3e\n", nm, umax, pmax)
    end
    uoff, poff = results[1][2], results[1][3]
    uon,  pon  = results[2][2], results[2][3]
    @printf("   -> contact velocity pollution reduction: %.1fx    pressure: %.1fx\n\n",
            uoff/max(uon,1e-300), poff/max(pon,1e-300))
    (uoff, uon, poff, pon)
end

# one SSP-RK3 step via the order-3 residual, outflow ghosts, per-stage projection.
function march_step!(M, n, g, dx, Ma, dt, s3max, flag)
    R = zeros(size(M)); M0 = copy(M)
    int = (g+1:g+n, g+1:g+n, 1:n, :)
    stage!(Mc) = begin
        refill_outflow!(Mc, n, g)
        residual_ho_3d_order3!(R, Mc, n,n,n, g, dx,dx,dx, Ma, dt; s3max=s3max, use_logjacobi_recon=flag)
    end
    stage!(M); @views M[int...] .= M0[int...] .+ dt .* R[int...]; project!(M,n,g,Ma,s3max)
    stage!(M); @views M[int...] .= 0.75.*M0[int...] .+ 0.25.*(M[int...] .+ dt.*R[int...]); project!(M,n,g,Ma,s3max)
    stage!(M); @views M[int...] .= (1/3).*M0[int...] .+ (2/3).*(M[int...] .+ dt.*R[int...]); project!(M,n,g,Ma,s3max)
    refill_outflow!(M, n, g)
end
function project!(M,n,g,Ma,s3max)
    for k in 1:n, j in 1:n, i in 1:n
        M[i+g,j+g,k,:] = realizable_3D_M4(M[i+g,j+g,k,:], Ma, s3max)
    end
end
function contact_metrics(M, n, g, u0, p0)
    umax=0.0; pmax=0.0
    for k in 1:n, j in 1:n, i in 1:n
        m = @view M[i+g,j+g,k,:]; rho=m[1]; rho>0 || continue
        umax=max(umax, abs(m[2]/rho - u0)); pmax=max(pmax, abs((m[3]-m[2]^2/rho) - p0))
    end
    (umax, pmax)
end

# ===================== 4. CROSS-MOMENT NO-REGRESSION (Ma=100) ==================
# Counter-streaming beams at Ma=100 (cross-moment stress). Confirm flag ON does
# NOT introduce blowups or extra realizability-projection load vs flag OFF (J only
# fixes marginals; the cross-moment cone stays the anchor/projection's job).
function test_ma100_noregress()
    println("=== 4. CROSS-MOMENT NO-REGRESSION (Ma=100 counter-streaming) ===")
    ENV["HYQMOM_PROJ_COUNT"] = "1"
    n = 16; g = 8; Ma = 100.0; s3max = max(40.0, 4.0 + Ma/2); dx = 1.0/n; U = 100.0
    ic(i,j,k) = (x=(i-0.5)/n; ux = x<0.5 ? U : -U;
                 InitializeM4_35(1.0, ux, 0.0, 0.0, 1.0,0.0,0.0,1.0,0.0,1.0))
    dt = 0.15*dx/(U+4.0); nsteps = 6
    for (nm, flag) in (("flag OFF", false), ("flag ON ", true))
        reset_proj_counter!()
        M = haloed(n, g, ic); blew = false
        for s in 1:nsteps
            march_step!(M, n, g, dx, Ma, dt, s3max, flag)
            any(!isfinite, M) && (blew = true; @printf("   %s BLEW UP at step %d\n", nm, s); break)
        end
        blew || @printf("   %s survived %d steps; proj-corrections=%d; min rho=%.3e\n",
                        nm, nsteps, proj_correction_count(), minimum(M[g+1:g+n,g+1:g+n,:,1]))
    end
    println()
end

function main()
    println("###### log-Jacobi marginal reconstruction — CPU validation ######\n")
    test_byte_identity()
    test_order()
    test_contact()
    test_ma100_noregress()
    println("###### done ######")
end

main()

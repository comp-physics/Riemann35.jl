module DVMBGKGPU
# dvm_bgk_gpu.jl — THE DVM GROUND TRUTH ON THE GPU.
#
# WHY. The DVM is the reference every wall result is measured against, and it had become the
# rate limiter: the Couette and Fourier references run single-threaded, hours per point, which
# is why three of the four points in the published zeta curve were never convergence-checked
# and why the velocity grid sat at nv = 24. A reference you cannot afford to refine is a
# reference you cannot fully trust.
#
# The structure is close to ideal for a GPU and it was simply never exploited:
#   * transport is first-order upwind in x, elementwise in the 3D velocity index;
#   * the collision is per spatial cell, a REDUCTION over the velocity grid followed by a
#     5-parameter Newton solve that costs nothing next to the reduction.
#
# LAYOUT: f[a,b,c,i], velocity-major. Both hot kernels then read coalesced -- transport has
# consecutive threads touching consecutive `a` at fixed neighbour `i-1`, and the collision
# reduction walks the velocity block contiguously. The CPU code uses f[i,a,b,c], which is
# wrong for both on a device.
#
# THE NEWTON IS WARM-STARTED from the previous step's coefficients, held per cell in `A`.
# This is the single most important optimisation: from a cold start the Mieussens solve takes
# 5-10 iterations, from the previous timestep it takes 1-2, and the collision is ~95% of the
# cost. The CPU version restarts cold every cell every step because it never keeps state.
#
# WHAT IS NOT APPROXIMATED. The discrete Maxwellian still matches rho, rho*u and rho*E
# EXACTLY on the grid (that is the whole point of Mieussens -- it makes the collision
# conservative on a finite velocity grid), and the wall is still the exact half-space
# condition with rho_w set from the DISCRETE half-fluxes. This is a port, not a model change,
# and validate_dvm_gpu.jl checks it against the CPU field to roundoff rather than trusting
# that claim.
using CUDA, LinearAlgebra

export VGridG, dvm_alloc, transport_upwind!, transport_walls!, collide!, moments35_field,
       discrete_maxwellian_host!, wall_pair_host, cellT_from_M

const IJK35 = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
               (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),
               (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),
               (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,1,2),(1,1,2),(0,3,1),(0,1,3),(0,2,2))

struct VGridG
    v::CuVector{Float64}
    vh::Vector{Float64}
    dv::Float64
    n::Int
end
function VGridG(vmax::Real, n::Int)
    vh = collect(range(-Float64(vmax), Float64(vmax), length = n))
    VGridG(CuArray(vh), vh, vh[2] - vh[1], n)
end

"f, work buffer, and the per-cell Newton coefficient state."
function dvm_alloc(Nx::Int, g::VGridG)
    (CUDA.zeros(Float64, g.n, g.n, g.n, Nx),
     CUDA.zeros(Float64, g.n, g.n, g.n, Nx),
     CUDA.zeros(Float64, 5, Nx))
end

# =======================================================================================
# TRANSPORT — first-order upwind in x. Identical stencil to the CPU `transport!`.
# =======================================================================================
function _upwind_kernel!(fn, f, v, n, Nx, lam, bc::Int32)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = n * n * n * Nx
    if t <= N
        @inbounds begin
            a = (t - 1) % n + 1;   r = (t - 1) ÷ n
            b = r % n + 1;         r = r ÷ n
            c = r % n + 1;         i = r ÷ n + 1
            s = v[a]
            if s > 0
                up = i == 1 ? (bc == Int32(1) ? f[a,b,c,Nx] : f[a,b,c,1]) : f[a,b,c,i-1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(f[a,b,c,i] - up)
            elseif s < 0
                up = i == Nx ? (bc == Int32(1) ? f[a,b,c,1] : f[a,b,c,Nx]) : f[a,b,c,i+1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(up - f[a,b,c,i])
            else
                fn[a,b,c,i] = f[a,b,c,i]
            end
        end
    end
    return nothing
end

"Periodic (bc=:periodic) or zero-gradient (bc=:copy) upwind transport."
function transport_upwind!(f, fn, dt, dx, g::VGridG; bc::Symbol = :copy)
    Nx = size(f, 4); N = g.n^3 * Nx
    thr = 256; blk = cld(N, thr)
    @cuda threads=thr blocks=blk _upwind_kernel!(fn, f, g.v, g.n, Nx,
                                                 dt/dx, bc === :periodic ? Int32(1) : Int32(0))
    copyto!(f, fn); f
end

# ---------------------------------------------------------------------------------------
# WALLS — the exact half-space diffuse condition. f for the INCOMING velocities IS the wall
# Maxwellian; no ghost cell and no Gaussian assumption about the interior. rho_w comes from
# the DISCRETE half-fluxes, so zero net mass flux holds by construction on the finite grid.
# The continuum rho*sqrt(T/Tw) is the formula the moment solver had to abandon (rem:wall-fix)
# and using it here would quietly reintroduce a mass leak into the reference itself.
# ---------------------------------------------------------------------------------------
function _influx_kernel!(out, f, v, n, Nx, dv3)
    # out[1] = influx at the LO wall (v>0 leaves, so v<0 arrives), out[2] = at the HI wall
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if t <= n*n*n
        @inbounds begin
            a = (t - 1) % n + 1; r = (t - 1) ÷ n
            b = r % n + 1;       c = r ÷ n + 1
            vx = v[a]
            if vx < 0
                CUDA.@atomic out[1] += -vx*f[a,b,c,1]*dv3
            elseif vx > 0
                CUDA.@atomic out[2] +=  vx*f[a,b,c,Nx]*dv3
            end
        end
    end
    return nothing
end

function _wall_transport_kernel!(fn, f, v, n, Nx, lam, ML, MR, rw)
    t = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = n*n*n*Nx
    if t <= N
        @inbounds begin
            a = (t - 1) % n + 1;   r = (t - 1) ÷ n
            b = r % n + 1;         r = r ÷ n
            c = r % n + 1;         i = r ÷ n + 1
            s = v[a]
            if s > 0
                up = i == 1 ? rw[1]*ML[a,b,c] : f[a,b,c,i-1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(f[a,b,c,i] - up)
            elseif s < 0
                up = i == Nx ? rw[2]*MR[a,b,c] : f[a,b,c,i+1]
                fn[a,b,c,i] = f[a,b,c,i] - lam*s*(up - f[a,b,c,i])
            else
                fn[a,b,c,i] = f[a,b,c,i]
            end
        end
    end
    return nothing
end

"""
    transport_walls!(f, fn, dt, dx, g, ML, MR, outL, outR, inflx, rw)

Upwind transport with exact diffuse walls at both ends. `ML`/`MR` are unit-density wall
Maxwellians on the device, `outL`/`outR` their outgoing half-fluxes (host scalars from
`wall_pair_host`). `inflx` and `rw` are length-2 device scratch.
"""
function transport_walls!(f, fn, dt, dx, g::VGridG, ML, MR, outL::Float64, outR::Float64,
                          inflx, rw)
    Nx = size(f, 4); n = g.n
    fill!(inflx, 0.0)
    @cuda threads=256 blocks=cld(n^3, 256) _influx_kernel!(inflx, f, g.v, n, Nx, g.dv^3)
    h = Array(inflx)
    # NB both `out*` and the influx are POSITIVE magnitudes -- dividing by a negated outflux
    # gives a negative ghost density and a singular Newton downstream. That sign error cost
    # an afternoon in the CPU version.
    copyto!(rw, [h[1]/outL, h[2]/outR])
    @cuda threads=256 blocks=cld(n^3*Nx, 256) _wall_transport_kernel!(
        fn, f, g.v, n, Nx, dt/dx, ML, MR, rw)
    copyto!(f, fn)
    (h[1]/outL, h[2]/outR)
end

# =======================================================================================
# COLLISION — one block per spatial cell.
#
# Each block runs the Mieussens Newton on its own cell: threads accumulate the 5 moments and
# the 15 unique entries of the symmetric 5x5 Jacobian over the velocity grid, tree-reduce in
# shared memory, and thread 1 does the Cholesky solve. The Jacobian is a Gram matrix
# sum(b b' f) with f > 0, hence symmetric positive definite, so Cholesky is both valid and
# cheaper than the generic solve the CPU uses.
#
# Warm start: A[:,i] carries the previous step's coefficients. `tol` is checked on the
# residual so a converged cell exits after ONE reduction.
# =======================================================================================
const NRED = 20   # 5 moments + 15 unique Jacobian entries

function _collide_kernel!(f, v, n, Nx, dv3, A, dt, tau, iters::Int32, tol)
    i = blockIdx().x
    tid = threadIdx().x
    nt = blockDim().x
    sh = CUDA.CuDynamicSharedArray(Float64, NRED * nt)
    sa = CUDA.CuDynamicSharedArray(Float64, 8, NRED * nt * sizeof(Float64))
    npts = n*n*n

    # ---- target collision invariants from the CURRENT f: <1>, <v>, <|v|^2> ----
    m0 = 0.0; m1 = 0.0; m2 = 0.0; m3 = 0.0; m4 = 0.0
    @inbounds for t in tid:nt:npts
        a = (t - 1) % n + 1; r = (t - 1) ÷ n
        b = r % n + 1;       c = r ÷ n + 1
        vx = v[a]; vy = v[b]; vz = v[c]
        w = f[a,b,c,i]*dv3
        m0 += w; m1 += w*vx; m2 += w*vy; m3 += w*vz
        m4 += w*(vx*vx + vy*vy + vz*vz)
    end
    @inbounds begin
        sh[tid] = m0; sh[nt+tid] = m1; sh[2nt+tid] = m2; sh[3nt+tid] = m3; sh[4nt+tid] = m4
    end
    sync_threads()
    s = nt >> 1
    while s > 0
        if tid <= s
            @inbounds for q in 0:4; sh[q*nt+tid] += sh[q*nt+tid+s]; end
        end
        sync_threads(); s >>= 1
    end
    @inbounds if tid == 1
        for q in 0:4; sa[q+1] = sh[q*nt+1]; end     # sa[1..5] = target moments
    end
    sync_threads()
    @inbounds tgt0 = sa[1]; @inbounds tgt1 = sa[2]; @inbounds tgt2 = sa[3]
    @inbounds tgt3 = sa[4]; @inbounds tgt4 = sa[5]

    # ---- warm start; fall back to the Maxwellian coefficients on the first visit ----
    @inbounds a1 = A[1,i]; a2 = A[2,i]; a3 = A[3,i]; a4 = A[4,i]; a5 = A[5,i]
    if a5 == 0.0                                   # never initialised
        rho = tgt0
        ux = tgt1/rho; uy = tgt2/rho; uz = tgt3/rho
        T = (tgt4/rho - (ux*ux + uy*uy + uz*uz))/3
        a1 = log(rho*(2*pi*T)^(-1.5)) - (ux*ux + uy*uy + uz*uz)/(2T)
        a2 = ux/T; a3 = uy/T; a4 = uz/T; a5 = -1/(2T)
    end

    it = Int32(0)
    while it < iters
        it += Int32(1)
        # accumulate moments (5) and the unique upper triangle of J (15)
        r1 = 0.0; r2 = 0.0; r3 = 0.0; r4 = 0.0; r5 = 0.0
        j11=0.0; j12=0.0; j13=0.0; j14=0.0; j15=0.0
        j22=0.0; j23=0.0; j24=0.0; j25=0.0
        j33=0.0; j34=0.0; j35=0.0
        j44=0.0; j45=0.0; j55=0.0
        @inbounds for t in tid:nt:npts
            aa = (t - 1) % n + 1; rr = (t - 1) ÷ n
            bb = rr % n + 1;      cc = rr ÷ n + 1
            vx = v[aa]; vy = v[bb]; vz = v[cc]
            s2 = vx*vx + vy*vy + vz*vz
            fv = exp(a1 + a2*vx + a3*vy + a4*vz + a5*s2)*dv3
            r1 += fv;      r2 += vx*fv;   r3 += vy*fv;  r4 += vz*fv;  r5 += s2*fv
            j11 += fv;     j12 += vx*fv;  j13 += vy*fv; j14 += vz*fv; j15 += s2*fv
            j22 += vx*vx*fv; j23 += vx*vy*fv; j24 += vx*vz*fv; j25 += vx*s2*fv
            j33 += vy*vy*fv; j34 += vy*vz*fv; j35 += vy*s2*fv
            j44 += vz*vz*fv; j45 += vz*s2*fv; j55 += s2*s2*fv
        end
        @inbounds begin
            sh[      tid] = r1;  sh[   nt+tid] = r2;  sh[ 2nt+tid] = r3
            sh[ 3nt+tid] = r4;  sh[ 4nt+tid] = r5
            sh[ 5nt+tid] = j11; sh[ 6nt+tid] = j12; sh[ 7nt+tid] = j13
            sh[ 8nt+tid] = j14; sh[ 9nt+tid] = j15; sh[10nt+tid] = j22
            sh[11nt+tid] = j23; sh[12nt+tid] = j24; sh[13nt+tid] = j25
            sh[14nt+tid] = j33; sh[15nt+tid] = j34; sh[16nt+tid] = j35
            sh[17nt+tid] = j44; sh[18nt+tid] = j45; sh[19nt+tid] = j55
        end
        sync_threads()
        s = nt >> 1
        while s > 0
            if tid <= s
                @inbounds for q in 0:(NRED-1); sh[q*nt+tid] += sh[q*nt+tid+s]; end
            end
            sync_threads(); s >>= 1
        end

        @inbounds if tid == 1
            b1 = sh[0nt+1] - tgt0; b2 = sh[1nt+1] - tgt1; b3 = sh[2nt+1] - tgt2
            b4 = sh[3nt+1] - tgt3; b5 = sh[4nt+1] - tgt4
            nrm = sqrt(b1*b1 + b2*b2 + b3*b3 + b4*b4 + b5*b5)
            if nrm < tol*max(1.0, tgt0)
                sa[6] = 1.0                       # converged: signal the block to stop
            else
                sa[6] = 0.0
                # Cholesky on the SPD Gram matrix J = sum(b b' f), then solve J d = b.
                L11 = sqrt(sh[5nt+1])
                L21 = sh[6nt+1]/L11; L31 = sh[7nt+1]/L11
                L41 = sh[8nt+1]/L11; L51 = sh[9nt+1]/L11
                L22 = sqrt(sh[10nt+1] - L21*L21)
                L32 = (sh[11nt+1] - L31*L21)/L22
                L42 = (sh[12nt+1] - L41*L21)/L22
                L52 = (sh[13nt+1] - L51*L21)/L22
                L33 = sqrt(sh[14nt+1] - L31*L31 - L32*L32)
                L43 = (sh[15nt+1] - L41*L31 - L42*L32)/L33
                L53 = (sh[16nt+1] - L51*L31 - L52*L32)/L33
                L44 = sqrt(sh[17nt+1] - L41*L41 - L42*L42 - L43*L43)
                L54 = (sh[18nt+1] - L51*L41 - L52*L42 - L53*L43)/L44
                L55 = sqrt(sh[19nt+1] - L51*L51 - L52*L52 - L53*L53 - L54*L54)
                y1 = b1/L11
                y2 = (b2 - L21*y1)/L22
                y3 = (b3 - L31*y1 - L32*y2)/L33
                y4 = (b4 - L41*y1 - L42*y2 - L43*y3)/L44
                y5 = (b5 - L51*y1 - L52*y2 - L53*y3 - L54*y4)/L55
                d5 = y5/L55
                d4 = (y4 - L54*d5)/L44
                d3 = (y3 - L43*d4 - L53*d5)/L33
                d2 = (y2 - L32*d3 - L42*d4 - L52*d5)/L22
                d1 = (y1 - L21*d2 - L31*d3 - L41*d4 - L51*d5)/L11
                sa[1] = d1; sa[2] = d2; sa[3] = d3; sa[4] = d4; sa[5] = d5
            end
        end
        sync_threads()
        @inbounds if sa[6] == 1.0
            break
        end
        @inbounds begin
            a1 -= sa[1]; a2 -= sa[2]; a3 -= sa[3]; a4 -= sa[4]; a5 -= sa[5]
        end
        sync_threads()
    end

    @inbounds if tid == 1
        A[1,i] = a1; A[2,i] = a2; A[3,i] = a3; A[4,i] = a4; A[5,i] = a5
    end

    # ---- relax: f = feq + (f - feq)*exp(-dt/tau) ----
    e = exp(-dt/tau)
    @inbounds for t in tid:nt:npts
        aa = (t - 1) % n + 1; rr = (t - 1) ÷ n
        bb = rr % n + 1;      cc = rr ÷ n + 1
        vx = v[aa]; vy = v[bb]; vz = v[cc]
        feq = exp(a1 + a2*vx + a3*vy + a4*vz + a5*(vx*vx + vy*vy + vz*vz))
        f[aa,bb,cc,i] = feq + (f[aa,bb,cc,i] - feq)*e
    end
    return nothing
end

"BGK collision on every cell. `A` is the persistent per-cell Newton state from `dvm_alloc`."
function collide!(f, g::VGridG, A, dt, tau; iters::Int = 30, tol = 1e-13, threads::Int = 128)
    Nx = size(f, 4)
    shmem = (NRED*threads + 8)*sizeof(Float64)
    @cuda threads=threads blocks=Nx shmem=shmem _collide_kernel!(
        f, g.v, g.n, Nx, g.dv^3, A, Float64(dt), Float64(tau), Int32(iters), Float64(tol))
    f
end

# =======================================================================================
# DIAGNOSTICS — the 35 raw moments per cell, on the host (called once at the end).
# =======================================================================================
function moments35_field(f, g::VGridG)
    Nx = size(f, 4); n = g.n; H = Array(f); dv3 = g.dv^3; v = g.vh
    M = zeros(Nx, 35)
    P = [v[a]^p for a in 1:n, p in 0:4]
    @inbounds for i in 1:Nx, c in 1:n, b in 1:n, a in 1:n
        w = H[a,b,c,i]*dv3
        w == 0.0 && continue
        for (m,(p,q,r)) in enumerate(IJK35)
            M[i,m] += w*P[a,p+1]*P[b,q+1]*P[c,r+1]
        end
    end
    M
end

"T = (C200+C020+C002)/3 from a row of 35 raw moments."
function cellT_from_M(Mrow)
    r = Mrow[1]; ux = Mrow[2]/r; uy = Mrow[6]/r; uz = Mrow[16]/r
    ((Mrow[3]/r - ux^2) + (Mrow[10]/r - uy^2) + (Mrow[20]/r - uz^2))/3
end

# ---- host-side discrete Maxwellian, for building the wall states once at setup ----
function discrete_maxwellian_host!(feq, rho, ux, uy, uz, T, vh::Vector{Float64}, dv::Float64;
                                   iters = 80, tol = 1e-14)
    n = length(vh); dv3 = dv^3
    E = 1.5*T + 0.5*(ux^2 + uy^2 + uz^2)
    tgt = [rho, rho*ux, rho*uy, rho*uz, rho*2E]
    a = [log(rho*(2pi*T)^(-1.5)) - (ux^2+uy^2+uz^2)/(2T), ux/T, uy/T, uz/T, -1/(2T)]
    for _ in 1:iters
        mom = zeros(5); J = zeros(5,5)
        @inbounds for k in 1:n, j in 1:n, i in 1:n
            vx = vh[i]; vy = vh[j]; vz = vh[k]; s2 = vx^2 + vy^2 + vz^2
            fval = exp(a[1] + a[2]*vx + a[3]*vy + a[4]*vz + a[5]*s2)*dv3
            b = (1.0, vx, vy, vz, s2)
            for p in 1:5
                mom[p] += b[p]*fval
                for q in 1:5; J[p,q] += b[p]*b[q]*fval; end
            end
        end
        r = mom .- tgt
        norm(r) < tol*max(1.0, rho) && break
        a .-= J\r
    end
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        feq[i,j,k] = exp(a[1] + a[2]*vh[i] + a[3]*vh[j] + a[4]*vh[k] +
                         a[5]*(vh[i]^2 + vh[j]^2 + vh[k]^2))
    end
    feq
end

"""
    wall_pair_host(g, Tw, sgn) -> (CuArray unit-density wall Maxwellian, outgoing half-flux)

`sgn = +1` for the LO wall (outgoing means vx > 0), `-1` for the HI wall.
"""
function wall_pair_host(g::VGridG, Tw::Float64, sgn::Int)
    n = g.n
    M1 = zeros(n, n, n)
    discrete_maxwellian_host!(M1, 1.0, 0.0, 0.0, 0.0, Tw, g.vh, g.dv)
    dv3 = g.dv^3; out = 0.0
    @inbounds for k in 1:n, j in 1:n, a in 1:n
        vx = g.vh[a]
        (sgn*vx > 0) && (out += sgn*vx*M1[a,j,k]*dv3)
    end
    (CuArray(M1), out)
end

end # module

# dvm_bgk.jl -- DISCRETE-VELOCITY BGK reference solver: 1D physical x 3D velocity.
# Ground truth for the 35-moment closure. Solves  d_t f + vx d_x f = (f_eq - f)/tau
# with the EXACT-EXPONENTIAL collision (matches our BGK), extracts the 35 raw moments
# (IJK order) for head-to-head closure comparison.  Built for CORRECTNESS first:
# validated on homogeneous relaxation (analytic exp), free-streaming (moment translation),
# and equilibrium (stationarity) before it is trusted as a reference.
module DVMBGK
using LinearAlgebra, Printf
export VGrid, moments35, maxwellian!, collide_cell!, transport!, rho_u_T, IJK35, discrete_maxwellian!

const IJK35 = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),
               (0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),
               (0,3,0),(1,3,0),(0,4,0),
               (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),
               (0,0,3),(1,0,3),(0,0,4),
               (0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),
               (0,1,2),(1,1,2),(0,1,3),(0,2,2))

struct VGrid
    v::Vector{Float64}; dv::Float64; n::Int
end
function VGrid(vmax::Real, n::Int)
    v=collect(range(-vmax,vmax,length=n)); VGrid(v, v[2]-v[1], n)
end

# ---- 35 raw moments of a single-cell distribution f3[a,b,c] (midpoint quadrature) ----
function moments35(f3::AbstractArray{Float64,3}, g::VGrid)
    M=zeros(35); dv3=g.dv^3; v=g.v; n=g.n
    # precompute powers vx^p for p=0..4
    P=[v[a]^p for a in 1:n, p in 0:4]
    @inbounds for c in 1:n, b in 1:n, a in 1:n
        w=f3[a,b,c]*dv3; w==0.0 && continue
        for (m,(p,q,r)) in enumerate(IJK35)
            M[m]+= w*P[a,p+1]*P[b,q+1]*P[c,r+1]
        end
    end
    M
end
function rho_u_T(f3,g::VGrid)
    M=moments35(f3,g); rho=M[1]; ux=M[2]/rho; uy=M[6]/rho; uz=M[16]/rho
    T=((M[3]/rho-ux^2)+(M[10]/rho-uy^2)+(M[20]/rho-uz^2))/3
    (rho,ux,uy,uz,T,M)
end

# ---- continuous Maxwellian sampled on the grid ----
function maxwellian!(feq,rho,ux,uy,uz,T,g::VGrid)
    c=rho*(2pi*T)^(-1.5); v=g.v; n=g.n
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        feq[i,j,k]=c*exp(-((v[i]-ux)^2+(v[j]-uy)^2+(v[k]-uz)^2)/(2T))
    end
    feq
end

# ---- discrete Maxwellian (Mieussens): match rho,rho*u,rho*E EXACTLY on the grid via
# 5-parameter Newton on f=exp(a0 + a.v + a4|v|^2). Guarantees collision conservation. ----
function discrete_maxwellian!(feq, rho,ux,uy,uz,T, g::VGrid; iters=60, tol=1e-13)
    v=g.v; n=g.n; dv3=g.dv^3
    # target collision-invariant moments: [rho, rho ux, rho uy, rho uz, rho*(3T+|u|^2)]
    E=1.5*T+0.5*(ux^2+uy^2+uz^2)                 # energy per mass = (1/2)<|v|^2>/rho
    tgt=[rho, rho*ux, rho*uy, rho*uz, rho*2E]    # <1>,<vx>,<vy>,<vz>,<|v|^2>
    a=[log(rho*(2pi*T)^(-1.5)) - (ux^2+uy^2+uz^2)/(2T), ux/T, uy/T, uz/T, -1/(2T)]
    local mom
    for _ in 1:iters
        mom=zeros(5); J=zeros(5,5)
        @inbounds for k in 1:n, j in 1:n, i in 1:n
            vx=v[i];vy=v[j];vz=v[k]; s2=vx^2+vy^2+vz^2
            fval=exp(a[1]+a[2]*vx+a[3]*vy+a[4]*vz+a[5]*s2)*dv3
            b=(1.0,vx,vy,vz,s2)
            for p in 1:5; mom[p]+=b[p]*fval; for q in 1:5; J[p,q]+=b[p]*b[q]*fval; end; end
        end
        r=mom.-tgt; (norm(r)<tol*max(1.0,rho)) && break
        a .-= J\r
    end
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        feq[i,j,k]=exp(a[1]+a[2]*v[i]+a[3]*v[j]+a[5]*(v[i]^2+v[j]^2+v[k]^2)+a[4]*v[k])
    end
    feq
end

# ---- exact-exponential collision on one cell (in place) ----
function collide_cell!(f3, dt, tau, g::VGrid; discrete=true)
    rho,ux,uy,uz,T,_=rho_u_T(f3,g)
    feq=similar(f3)
    discrete ? discrete_maxwellian!(feq,rho,ux,uy,uz,T,g) : maxwellian!(feq,rho,ux,uy,uz,T,g)
    e=exp(-dt/tau)
    @inbounds @. f3 = feq + (f3-feq)*e
    nothing
end

# ---- 1st-order upwind transport in x for the full field f[Nx, n,n,n], speed vx=v[a] ----
# bc=:copy (zero-gradient). MUSCL upgrade later; 1st order + fine grid for now.
function transport!(f::Array{Float64,4}, dt, dx, g::VGrid; bc::Symbol=:copy)
    # `bc` USED TO BE ACCEPTED AND IGNORED: the wrap-around branch did not exist and every
    # call, including bc=:periodic, got zero-gradient. No published result was affected --
    # every caller used the default, and reduce26_shear.jl needed periodicity, noticed, and
    # wrote its own wrapping transport (see the comment at its line 66). But a kwarg that
    # silently does something else is a trap, and validate_dvm_gpu.jl walked straight into
    # it: GATE 1 compared true-periodic on the device against zero-gradient here and read
    # 7.5e-2, which looks exactly like a broken port.
    Nx=size(f,1); n=g.n; lam=dt/dx; fn=copy(f)
    per = bc === :periodic
    @inbounds for a in 1:n
        s=g.v[a]
        if s>0
            for k in 1:n, j in 1:n, i in 1:Nx
                im1 = i==1 ? (per ? Nx : 1) : i-1
                fn[i,a,j,k]=f[i,a,j,k]-lam*s*(f[i,a,j,k]-f[im1,a,j,k])
            end
        elseif s<0
            for k in 1:n, j in 1:n, i in 1:Nx
                ip1 = i==Nx ? (per ? 1 : Nx) : i+1
                fn[i,a,j,k]=f[i,a,j,k]-lam*s*(f[ip1,a,j,k]-f[i,a,j,k])
            end
        end
    end
    copyto!(f,fn); nothing
end
# NB: f layout is f[ix, a(vx), b(vy), c(vz)] -- vx index second so transport strides ok.

end # module

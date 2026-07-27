# reduce26_gpu.jl — OPT-IN 26-moment reduction on the GPU (Rodney Fox), the device
# port of `src/numerics/moment_reduce26.jl`. Included INSIDE module Timestep3DOrder3GPU
# so the bare `_recon_centrals`, `_c4tom4_35`, `_EPSF` resolve via that module's
# `using ...ReconDev: …` imports.
#
# `reduce26_relax_tup` replaces the nine ODD fourth-order standardized moments (six
# s310-type via eq (43), three s211-type via S011+S300·S111, +perms) by their closure
# values, keeping only the six EVEN fourth-order moments independent. It is built from
# the SAME device subsets the flux/realizability kernels use — `_recon_centrals`
# (raw→central, the verbatim @fastmath M4toC4 subset) and `_c4tom4_35` (central→raw) —
# and reuses the CPU σ = sqrt(max(C, eps())) standardization, so it reproduces the CPU
# `reduce26_moments` to ~1 ULP (test/test_reduce26_gpu.jl). Allocation-free: every
# helper returns a flat NTuple, so the whole thing runs in-kernel.
#
# `_reduce26_interior!` applies it per interior cell (mirrors `_bgk_interior!`). Wired
# into `march3d_order3_gpu!` as an opt-in per-step operator-split projection (kwarg
# `reduce26=false` default => the march is byte-identical to the pre-change kernel).

# 35-tuple (M4 canonical raw order) -> 35-tuple with the nine odd 4th-order moments
# projected onto their closure. Pure scalar; host- and device-callable.
@inline function reduce26_relax_tup(M::NTuple{35,Float64})
    (cC200,cC020,cC002,cC300,cC400,cC110,cC210,cC310,cC120,cC220,cC030,cC130,cC040,
     cC101,cC201,cC301,cC011,cC111,cC211,cC021,cC121,cC031,cC102,cC202,cC012,cC112,
     cC022,cC003,cC103,cC013,cC004) = _recon_centrals(
        M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9],M[10],M[11],M[12],M[13],M[14],M[15],
        M[16],M[17],M[18],M[19],M[20],M[21],M[22],M[23],M[24],M[25],M[26],M[27],M[28],M[29],M[30],
        M[31],M[32],M[33],M[34],M[35])

    # σ from eps-floored variances (matches CPU reduce26_moments)
    sx = sqrt(max(cC200, _EPSF)); sy = sqrt(max(cC020, _EPSF)); sz = sqrt(max(cC002, _EPSF))

    # standardized moments feeding the closures
    s110 = cC110/(sx*sy);     s101 = cC101/(sx*sz);     s011 = cC011/(sy*sz)
    s111 = cC111/(sx*sy*sz)
    s300 = cC300/sx^3;        s030 = cC030/sy^3;        s003 = cC003/sz^3
    s400 = cC400/sx^4;        s040 = cC040/sy^4;        s004 = cC004/sz^4
    s210 = cC210/(sx^2*sy);   s120 = cC120/(sx*sy^2)
    s201 = cC201/(sx^2*sz);   s102 = cC102/(sx*sz^2)
    s021 = cC021/(sy^2*sz);   s012 = cC012/(sy*sz^2)

    e43(a,b,c,d) = a*b + 1.5*c*(d - a*c)
    v310 = e43(s110,s400,s300,s210); v130 = e43(s110,s040,s030,s120)
    v301 = e43(s101,s400,s300,s201); v103 = e43(s101,s004,s003,s102)
    v031 = e43(s011,s040,s030,s021); v013 = e43(s011,s004,s003,s012)
    v211 = s011 + s300*s111; v121 = s101 + s030*s111; v112 = s110 + s003*s111

    # destandardize the nine closed values back to central moments
    nC310 = v310*sx^3*sy;   nC130 = v130*sx*sy^3
    nC301 = v301*sx^3*sz;   nC103 = v103*sx*sz^3
    nC031 = v031*sy^3*sz;   nC013 = v013*sy*sz^3
    nC211 = v211*sx^2*sy*sz; nC121 = v121*sx*sy^2*sz; nC112 = v112*sx*sy*sz^2

    u = M[2]/M[1]; v = M[6]/M[1]; w = M[16]/M[1]
    return _c4tom4_35(M[1], u, v, w,
        cC200,cC110,cC101,cC020,cC011,cC002,
        cC300,cC210,cC201,cC120,cC111,cC102,cC030,cC021,cC012,cC003,
        cC400,nC310,nC301,cC220,nC211,cC202,nC130,nC121,nC112,nC103,
        cC040,nC031,cC022,nC013,cC004)
end

# per-interior-cell 26-moment reduction on the haloed cube (mirrors _bgk_interior!).
function _reduce26_interior!(G, nx::Int, ny::Int, nz::Int, g::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= nx * ny * nz
        @inbounds begin
            i = (idx - 1) % nx + 1; r = (idx - 1) ÷ nx
            j = r % ny + 1;         k = r ÷ ny + 1
            ga = g + i; gb = g + j; gc = g + k
            C = ntuple(m -> G[m, ga, gb, gc], Val(35))
            out = reduce26_relax_tup(C)
            for m in 1:35; G[m, ga, gb, gc] = out[m]; end
        end
    end
    return nothing
end

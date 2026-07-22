"""
    moment_correction_minnorm.jl — OPT-IN minimal-norm hyperbolicity correction.

An alternative to the blunt reset in `correct_moments_dev` (which zeros all six
third-order cross standardized moments). Instead of the blunt reset, this finds
the *smallest weighted change* to the six S_{210}-permutations AND the six
S_{310}-permutations (the eq-(43)-closure axis, Rodney 2026-07-22) that makes all
six axis-plane blocks real, via a joint constrained (KKT) projection onto the
block real-rootedness boundary, with a targeted fallback for the few near-boundary
planes. Keeps conserved (<=2nd-order) moments exact; on random-realizable firing
states it is ~8x gentler than the blunt reset at higher coverage.

OPT-IN: selected only when `HYP_CORRECTION[] === :minnorm`; the default `:blunt`
leaves `correct_moments_hyperbolic_3D` byte-identical to `correct_moments_dev`.

Prototype-grade (Dict-based, allocates); an allocation-free rewrite is future work.
Uses the package's `jacobian15`, `M4toC4_3D`, `C4toM4_3D`, `eig3_realparts`.

Full rationale, the eq-(41)/(43)/(45) mechanism, the 3D coupling/frame/margin
subtleties, and the measured coverage/fidelity numbers are documented in
`docs/design/minimal-norm-hyperbolicity-correction.md`.
"""

# 12 adjustable standardized moments: 6 third-order + 6 fourth-order (eq-43 axis)
const _MN_K12 = (:S210,:S120,:S201,:S102,:S021,:S012,:S310,:S130,:S301,:S103,:S031,:S013)
const _MN_KPOS = Dict(k=>i for (i,k) in enumerate(_MN_K12))
# each plane -> (s11,s30,s40,s03,s04, s21,s12,s31,s13, s22) as 3D S-moment keys
const _MN_PLANES = (
 (:S110,:S300,:S400,:S030,:S040,:S210,:S120,:S310,:S130,:S220),  # UV
 (:S101,:S300,:S400,:S003,:S004,:S201,:S102,:S301,:S103,:S202),  # UW
 (:S110,:S030,:S040,:S300,:S400,:S120,:S210,:S130,:S310,:S220),  # VU
 (:S011,:S030,:S040,:S003,:S004,:S021,:S012,:S031,:S013,:S022),  # VW
 (:S101,:S003,:S004,:S300,:S400,:S102,:S201,:S103,:S301,:S202),  # WU
 (:S011,:S003,:S004,:S030,:S040,:S012,:S021,:S013,:S031,:S022))  # WV
const _MN_IJK35 = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))

# discriminant of the monic characteristic cubic of the 3x3 block ( >=0 <=> real )
@inline function _mn_discr(B)
    any(!isfinite, B) && return -1e300
    b = -(B[1,1]+B[2,2]+B[3,3])
    c = B[1,1]*B[2,2]-B[1,2]*B[2,1] + B[1,1]*B[3,3]-B[1,3]*B[3,1] + B[2,2]*B[3,3]-B[2,3]*B[3,2]
    d = -(B[1,1]*(B[2,2]*B[3,3]-B[2,3]*B[3,2]) - B[1,2]*(B[2,1]*B[3,3]-B[2,3]*B[3,1]) + B[1,3]*(B[2,1]*B[3,2]-B[2,2]*B[3,1]))
    18*b*c*d - 4*b^3*d + b^2*c^2 - 4*c^3 - 27*d^2
end
# standardized-frame plane block discriminant from a plane's 10 standardized moments
@inline function _mn_planeDisc(s11,s21,s12,s22,s30,s40,s03,s04,s31,s13)
    J = jacobian15(1.0,0.0,1.0,s03,s04, 0.0,s11,s12,s13, 1.0,s21,s22, s30,s31,s40)
    _mn_discr(@view J[13:15,13:15])
end
@inline _mn_Dp(D, pm, x) = _mn_planeDisc(D[pm[1]], x[_MN_KPOS[pm[6]]], x[_MN_KPOS[pm[7]]],
    D[pm[10]], D[pm[2]], D[pm[3]], D[pm[4]], D[pm[5]], x[_MN_KPOS[pm[8]]], x[_MN_KPOS[pm[9]]])

function _mn_gradDp!(g, D, pm, x)
    fill!(g, 0.0); h = 1e-6
    for key in (pm[6],pm[7],pm[8],pm[9])
        p = _MN_KPOS[key]; xp = copy(x); xm = copy(x); xp[p]+=h; xm[p]-=h
        g[p] = (_mn_Dp(D,pm,xp) - _mn_Dp(D,pm,xm))/(2h)
    end
    g
end

# raw 35-vector -> Dict of standardized moments + (M000,u,v,w,sx,sy,sz)
function _mn_raw_to_S(M)
    C = M4toC4_3D(M...)
    sx = sqrt(max(C[3,1,1],eps())); sy = sqrt(max(C[1,3,1],eps())); sz = sqrt(max(C[1,1,3],eps()))
    D = Dict{Symbol,Float64}()
    for (nm,(i,j,k)) in ((:S110,(1,1,0)),(:S101,(1,0,1)),(:S011,(0,1,1)),(:S300,(3,0,0)),(:S030,(0,3,0)),(:S003,(0,0,3)),
        (:S400,(4,0,0)),(:S040,(0,4,0)),(:S004,(0,0,4)),(:S220,(2,2,0)),(:S202,(2,0,2)),(:S022,(0,2,2)),
        (:S210,(2,1,0)),(:S120,(1,2,0)),(:S201,(2,0,1)),(:S102,(1,0,2)),(:S021,(0,2,1)),(:S012,(0,1,2)),
        (:S310,(3,1,0)),(:S130,(1,3,0)),(:S301,(3,0,1)),(:S103,(1,0,3)),(:S031,(0,3,1)),(:S013,(0,1,3)),
        (:S111,(1,1,1)),(:S211,(2,1,1)),(:S121,(1,2,1)),(:S112,(1,1,2)))
        D[nm] = C[i+1,j+1,k+1]/(sx^i*sy^j*sz^k)
    end
    D, (M[1], M[2]/M[1], M[6]/M[1], M[16]/M[1], sx, sy, sz)
end
function _mn_S_to_raw(D, meta)
    M000,u,v,w,sx,sy,sz = meta
    C(i,j,k) = D[Symbol("S",i,j,k)]*sx^i*sy^j*sz^k
    out = C4toM4_3D(M000,u,v,w, sx^2,C(1,1,0),C(1,0,1),sy^2,C(0,1,1),sz^2,
        C(3,0,0),C(2,1,0),C(2,0,1),C(1,2,0),C(1,1,1),C(1,0,2),C(0,3,0),C(0,2,1),C(0,1,2),C(0,0,3),
        C(4,0,0),C(3,1,0),C(3,0,1),C(2,2,0),C(2,1,1),C(2,0,2),C(1,3,0),C(1,2,1),C(1,1,2),C(1,0,3),
        C(0,4,0),C(0,3,1),C(0,2,2),C(0,1,3),C(0,0,4))
    Float64[out[i+1,j+1,k+1] for (i,j,k) in _MN_IJK35]
end

# raw plane extractors (jacobian15 arg order) for the fallback firing check
@inline _mn_pl(M, idx) = ntuple(t->M[idx[t]], 15)
const _MN_PIDX = ((1,6,10,13,15,2,7,11,14,3,8,12,4,9,5),(1,16,20,23,25,2,17,21,24,3,18,22,4,19,5),
 (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),(1,16,20,23,25,6,26,32,34,10,29,35,13,31,15),
 (1,2,3,4,5,16,17,18,19,20,21,22,23,24,25),(1,6,10,13,15,16,26,29,31,20,32,35,23,34,25))
function _mn_plane_fires(M, p)
    J = jacobian15(_mn_pl(M, _MN_PIDX[p])...)
    any(!isfinite, J) && return true
    _, hc = eig3_realparts(J[13,13],J[13,14],J[13,15],J[14,13],J[14,14],J[14,15],J[15,13],J[15,14],J[15,15])
    hc
end

"""
    correct_moments_minnorm(M; w3=1.0, w4=0.25, mrg=1e-3, iters=40)

Minimal-norm hyperbolicity correction (opt-in). Returns the 35 corrected raw moments.
"""
function correct_moments_minnorm(M::AbstractVector; w3=1.0, w4=0.25, mrg=1e-3, iters=40)
    D, meta = _mn_raw_to_S(M)
    for k in (:S220,:S202,:S022); D[k] = max(D[k], 1.0/3.0); end
    # joint KKT projection (standardized frame) to discriminant >= mrg
    winv = Float64[i<=6 ? 1/w3 : 1/w4 for i in 1:12]
    x = Float64[D[k] for k in _MN_K12]; g = zeros(12)
    for _ in 1:iters
        dvals = ntuple(p->_mn_Dp(D,_MN_PLANES[p],x), 6)
        viol = findall(<(mrg), dvals)
        isempty(viol) && break
        A = zeros(length(viol), 12); dv = zeros(length(viol))
        for (r,p) in enumerate(viol); A[r,:] = _mn_gradDp!(g, D, _MN_PLANES[p], x); dv[r] = mrg - dvals[p]; end
        Mmat = A*Diagonal(winv)*A' + 1e-10I
        step = Diagonal(winv)*A'*(Mmat \ dv)
        base = maximum(mrg .- min.(dvals, mrg)); α = 1.0
        for _ in 1:20
            xt = x .+ α.*step
            dt = ntuple(p->_mn_Dp(D,_MN_PLANES[p],xt), 6)
            (maximum(mrg .- min.(dt,mrg)) < base*0.999 || α < 1e-4) && (x = xt; break)
            α *= 0.5
        end
    end
    for (i,k) in enumerate(_MN_K12); D[k] = x[i]; end
    Mout = _mn_S_to_raw(D, meta)
    # targeted fallback: zero the 3rd-order of any plane still firing in the raw frame
    refill = false
    for p in 1:6
        _mn_plane_fires(Mout, p) || continue
        pm = _MN_PLANES[p]; D[pm[6]] = 0.0; D[pm[7]] = 0.0; refill = true
    end
    refill ? _mn_S_to_raw(D, meta) : Mout
end

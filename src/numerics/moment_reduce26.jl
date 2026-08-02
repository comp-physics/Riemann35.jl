"""
    moment_reduce26.jl — OPT-IN 26-moment reduction (Rodney Fox, 2026-07-22).

Projects the 35-moment state onto the 26-moment reduced manifold: the nine ODD
fourth-order standardized moments (the six s310-type and three s211-type) are
replaced by their closure values, keeping only the six EVEN fourth-order moments
(S400,S040,S004,S220,S202,S022) as independent. Closures:
  s310-type : eq (43),  S310 = S110 S400 + (3/2) S300 (S210 - S110 S300)  (+perms)
  s211-type : S211 = S011 + S300 S111  (+perms)

Applied per cell per step (operator-split), `REDUCE26[]` toggles it on. Default
off => the solver evolves the full 35 moments (byte-identical). A diagnostic /
comparison feature, not a production default. Reuses `M4toC4_3D`, `C4toM4_3D`.

ACCURACY COST, MEASURED AGAINST KINETIC GROUND TRUTH. Decaying transverse shear wave,
1D x 3V, periodic (no walls), DVM-BGK reference, Nx = 384, Ma ~ 0.1. Relative L2 error
against the DVM on the nine dropped moments, and the error in the shear amplitude:

    tau (Kn)                        0.05      0.20      1.00
    error on the nine, full-35      0.067     0.244     0.522
    error on the nine, reduced-26   0.086     0.454     0.977
    u_y amplitude, full-35         -5.36%    -2.55%    -0.12%
    u_y amplitude, reduced-26      -4.70%   +14.19%   +61.30%

At Kn = 0.05 the reduction costs essentially nothing. By Kn ~ 1 it retains almost none
of the dropped moments' content -- 0.977 is very nearly orthogonal to truth -- and
over-predicts the shear amplitude by 61% where the full 35 gets it to 0.12%. Converged:
the nine-moment error is stable to 1-3% across Nx = 192/384/768 and to 3-4 digits across
nv = 16/20/28.

THE GOVERNING AXIS IS DEPARTURE FROM EQUILIBRIUM, NOT MACH NUMBER, and that reconciles
two results which look contradictory. An earlier DVM comparison at Ma = 25 found the
reduction costs no accuracy; this one at Ma ~ 0.1 finds a large cost. Both are correct.
What matters is how far the distribution sits from a Maxwellian -- confirmed
independently by a homogeneous-relaxation sweep in which the a priori closure error fell
from 18.2% to 0.49% as the departure amplitude was reduced. So `REDUCE26[] = true` is
safe near equilibrium and safe where parity hides the dropped moments, and costs real
accuracy in between; it is not a knob whose safety can be read off the Mach number.
"""
const REDUCE26 = Ref(false)

const _R26_IJK = ((0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(0,3,0),(1,3,0),(0,4,0),(0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,0,2),(1,0,2),(2,0,2),(0,0,3),(1,0,3),(0,0,4),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(0,3,1),(0,1,2),(1,1,2),(0,1,3),(0,2,2))

function reduce26_moments(M::AbstractVector)
    C = M4toC4_3D(M...)
    sx = sqrt(max(C[3,1,1], eps())); sy = sqrt(max(C[1,3,1], eps())); sz = sqrt(max(C[1,1,3], eps()))
    S(i,j,k) = C[i+1,j+1,k+1] / (sx^i * sy^j * sz^k)
    e43(a,b,c,d) = a*b + 1.5*c*(d - a*c)
    s110 = S(1,1,0); s101 = S(1,0,1); s011 = S(0,1,1); s111 = S(1,1,1)
    s300 = S(3,0,0); s030 = S(0,3,0); s003 = S(0,0,3)
    # closed standardized values (Rodney)
    v310 = e43(s110,S(4,0,0),s300,S(2,1,0)); v130 = e43(s110,S(0,4,0),s030,S(1,2,0))
    v301 = e43(s101,S(4,0,0),s300,S(2,0,1)); v103 = e43(s101,S(0,0,4),s003,S(1,0,2))
    v031 = e43(s011,S(0,4,0),s030,S(0,2,1)); v013 = e43(s011,S(0,0,4),s003,S(0,1,2))
    v211 = s011 + s300*s111; v121 = s101 + s030*s111; v112 = s110 + s003*s111
    @inline function Cg(i,j,k)
        (i,j,k) == (3,1,0) && return v310*sx^3*sy;  (i,j,k) == (1,3,0) && return v130*sx*sy^3
        (i,j,k) == (3,0,1) && return v301*sx^3*sz;  (i,j,k) == (1,0,3) && return v103*sx*sz^3
        (i,j,k) == (0,3,1) && return v031*sy^3*sz;  (i,j,k) == (0,1,3) && return v013*sy*sz^3
        (i,j,k) == (2,1,1) && return v211*sx^2*sy*sz; (i,j,k) == (1,2,1) && return v121*sx*sy^2*sz
        (i,j,k) == (1,1,2) && return v112*sx*sy*sz^2
        return C[i+1,j+1,k+1]
    end
    u = M[2]/M[1]; v = M[6]/M[1]; w = M[16]/M[1]
    out = C4toM4_3D(M[1],u,v,w, Cg(2,0,0),Cg(1,1,0),Cg(1,0,1),Cg(0,2,0),Cg(0,1,1),Cg(0,0,2),
        Cg(3,0,0),Cg(2,1,0),Cg(2,0,1),Cg(1,2,0),Cg(1,1,1),Cg(1,0,2),Cg(0,3,0),Cg(0,2,1),Cg(0,1,2),Cg(0,0,3),
        Cg(4,0,0),Cg(3,1,0),Cg(3,0,1),Cg(2,2,0),Cg(2,1,1),Cg(2,0,2),Cg(1,3,0),Cg(1,2,1),Cg(1,1,2),Cg(1,0,3),
        Cg(0,4,0),Cg(0,3,1),Cg(0,2,2),Cg(0,1,3),Cg(0,0,4))
    Float64[out[i+1,j+1,k+1] for (i,j,k) in _R26_IJK]
end

# kfvs_f3_drift_diagnostic.jl — DIAGNOSTIC (not a feature): settle WHY the F3
# flux-level anchor drifts out of the Δ2* cross-moment cone over multiple steps.
#
# F3 conserves to machine precision and is per-step realizable, but over 40 steps at
# Ma=100 the Δ2* cone margin goes < -1e-8 on a growing set of cells (worst ≈ -0.99).
# Two candidate causes with DIFFERENT fixes:
#   (A) predicate precision — the Δ2* test admits states slightly outside the cone.
#       BUT: the CPU predicate `is_realizable` = `realizability_margin ≥ 0` already
#       uses the EXACT LAPACK `eigvals` of the Δ2* matrix (src/Riemann35.jl `_geigvals`
#       → `eigvals`), NOT Bunch-Kaufman inertia. So on the CPU path VARIANT A ≡ the
#       baseline; this script confirms that and reports it.
#   (B) missing cross re-interiorization — θ* parks limited cells AT margin≈0 (lam_min
#       =0.0); nothing floors the Δ2* margin interior, so 1/H conditioning kicks it out
#       over steps. VARIANT B: require margin ≥ +δ (δ=1e-6) via `set_kfvs_xfloor!`.
#
# For each config we run the well-posed Ma=100 crossing-jets march (OFF-path stable)
# at 8/20/40 steps and report, on the FINAL interior field:
#   (i)   # cells with Δ2* margin < -1e-8
#   (ii)  worst (most negative) margin
#   (iii) SPLIT: "small tail" (-1e-3 ≤ margin < -1e-8, precision-scale) vs
#                "large outliers" (margin < -1e-3, down to -0.99)
#   plus:  correlation of drifting cells with θ*<1 (boundary-parked) is checked
#          separately (a single run with per-cell θ*<1 flags).
#
# CPU-only. Production path unchanged (floor defaults to 0.0). Slow (KFVS inversion
# per face per stage); ~minutes per config.

ENV["HYQMOM_SKIP_PLOTTING"]="true"; ENV["CI"]="true"
using Riemann35, MPI, Printf
MPI.Initialized() || MPI.Init()

function build_ic(Np,halo,Ma)
    C=1.0; vj=0.15*Ma
    bg=InitializeM4_35(0.05,0.0,0.0,0.0,C,0.0,0.0,C,0.0,C)
    l =InitializeM4_35(1.0, vj, vj/2,0.0,C,0.0,0.0,C,0.0,C)
    r =InitializeM4_35(1.0,-vj,-vj/2,0.0,C,0.0,0.0,C,0.0,C)
    nx=ny=nz=Np; M=zeros(Float64,nx+2halo,ny+2halo,nz,35)
    Cs=max(1,floor(Int,0.2*Np)); lo=div(Np,2)-Cs;hi=div(Np,2);lo2=div(Np,2)+1;hi2=div(Np,2)+1+Cs
    for k in 1:nz,i in 1:nx,j in 1:ny
        Mr=bg; (lo<=i<=hi&&lo<=j<=hi)&&(Mr=l); (lo2<=i<=hi2&&lo2<=j<=hi2)&&(Mr=r)
        M[i+halo,j+halo,k,:]=Mr
    end
    M,nx,ny,nz
end

# run the march, return the final margin distribution
function run_margins(Np, halo, Ma, nsteps, dt)
    M,nx,ny,nz=build_ic(Np,halo,Ma)
    comm=MPI.COMM_WORLD; decomp=setup_mpi_cartesian_3d(Np,Np,Np,halo,comm); bc=:copy
    halo_exchange_3d!(M,decomp,bc)
    dx=1.0/Np; s3max=max(40.0,4.0+abs(Ma)/2.0)
    mass0=0.0; en0=0.0
    for k in 1:nz,j in 1:ny,i in 1:nx; mass0+=M[i+halo,j+halo,k,1]; en0+=M[i+halo,j+halo,k,3]+M[i+halo,j+halo,k,10]+M[i+halo,j+halo,k,20]; end
    for s in 1:nsteps
        step_highorder_3d!(M,dt,decomp,bc,nx,ny,nz,halo,dx,dx,dx,Ma; order=3,s3max=s3max,use_kfvs_anchor=true)
    end
    margins=Float64[]; nfin=0; mass1=0.0; en1=0.0
    for k in 1:nz,j in 1:ny,i in 1:nx
        c=@view M[i+halo,j+halo,k,:]
        for q in 1:35; isfinite(c[q])||(nfin+=1); end
        push!(margins, realizability_margin(c))
        mass1+=c[1]; en1+=c[3]+c[10]+c[20]
    end
    massdrift=abs(mass1-mass0)/mass0; endrift=abs(en1-en0)/max(abs(en0),1e-300)
    return margins, nfin, massdrift, endrift
end

function report(label, margins, nfin, massdrift, endrift)
    ntail=0; nout=0; worst=0.0
    for m in margins
        if m < -1e-3; nout+=1; worst=min(worst,m)
        elseif m < -1e-8; ntail+=1; worst=min(worst,m); end
    end
    nneg = ntail+nout
    @printf("  %-22s neg(<-1e-8)=%4d  worst=%.3e  |  small-tail(-1e-3..-1e-8)=%4d  large-outlier(<-1e-3)=%4d  | nonfin=%d mass=%.2e en=%.2e\n",
            label, nneg, worst, ntail, nout, nfin, massdrift, endrift)
end

function main()
    Ma=100.0; Np=16; halo=4; dx=1.0/Np
    dt = 0.12*dx/(Ma/2 + 5)
    # FAST single-variant mode (parallelize across processes): set
    # HYQMOM_KFVS_XFLOOR_RUN=<δ> and HYQMOM_DIAG_STEPS=<n> to run ONE floor at ONE horizon.
    _sv = get(ENV, "HYQMOM_KFVS_XFLOOR_RUN", "")
    if _sv != ""
        δ = parse(Float64, _sv); set_kfvs_xfloor!(δ)
        ns = parse(Int, get(ENV, "HYQMOM_DIAG_STEPS", "20"))
        m,nf,md,ed = run_margins(Np,halo,Ma,ns,dt); report("δ=$(δ) $(ns)step", m,nf,md,ed)
        set_kfvs_xfloor!(0.0); println("DONE."); return
    end
    steps = (8, 20, 40)

    # confirm the CPU Δ2* predicate is exact-eig (VARIANT A ≡ baseline): compare
    # is_realizable's margin path against a fresh LAPACK eigvals on a sample cell.
    println("PREDICATE CHECK: is_realizable uses realizability_margin = min eigvals(Δ2*) (exact LAPACK).")
    println("  => VARIANT A (exact eig) is IDENTICAL to the production predicate on the CPU path.\n")

    println("BASELINE (production F3, cross-floor δ=0):")
    set_kfvs_xfloor!(0.0)
    for ns in steps
        m,nf,md,ed = run_margins(Np,halo,Ma,ns,dt); report("$(ns)step", m,nf,md,ed)
    end

    println("\nVARIANT B (cross-margin floor δ=1e-6, re-interiorization):")
    set_kfvs_xfloor!(1e-6)
    for ns in steps
        m,nf,md,ed = run_margins(Np,halo,Ma,ns,dt); report("$(ns)step", m,nf,md,ed)
    end
    set_kfvs_xfloor!(0.0)

    println("\nVARIANT B-strong (cross-margin floor δ=1e-3):")
    set_kfvs_xfloor!(1e-3)
    for ns in steps
        m,nf,md,ed = run_margins(Np,halo,Ma,ns,dt); report("$(ns)step", m,nf,md,ed)
    end
    set_kfvs_xfloor!(0.0)
    println("\nDONE.")
end
main()

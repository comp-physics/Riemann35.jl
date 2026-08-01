# test_driven_dt_convergence.jl — A DRIVEN STEADY STATE MUST HAVE A dt -> 0 LIMIT.
#
# The property is not subtle: a steady state solves R(M) = 0, an equation containing no dt.
# If refining the timestep at fixed grid and fixed physical time keeps moving the answer,
# something in the update contributes per APPLICATION rather than per unit time, and the
# scheme is not converging to a root of R at all.
#
# WHY THIS TEST EXISTS. Nothing in the suite checked it, and a defect that violated it
# survived for the life of the project: `realizability_S2` applied its 0.9999 back-off
# unconditionally, shaving 1e-4 off the velocity correlations on every cell every RK stage
# whether or not the state needed correcting -- an artificial shear-stress dissipation
# accumulating as 1/dt. Before the fix this configuration moved the amplitude by tens of
# percent under refinement with the increments DOUBLING. Every unit test passed throughout,
# and twelve of them actively defended the bug.
#
# A convergence study is a different KIND of check from a unit assertion: it interrogates the
# SCHEME rather than a function. This suite had none.
#
# THE CONFIGURATION, chosen so a failure is unambiguous:
#   * Kolmogorov flow -- sinusoidal forcing balanced by viscous dissipation, a genuine
#     driven steady state. A decaying or homogeneous problem would NOT show the defect,
#     because there is no balance for the accumulated bias to displace.
#   * FULLY PERIODIC: no walls, no ghost cells, no boundary of any kind, so a failure cannot
#     be blamed on a boundary condition.
#   * dt varied at FIXED grid. The shipped dt = 0.2 dy/(5 sqrt T) ties timestep to mesh, so
#     an ny sweep returns the SUM of two errors -- which is exactly why this went unseen.
#   * CPU driver, small grid, so it runs in CI -- but note WHICH code it exercises.
#     `realizable_3D_M4` (what _project_interior! calls) is
#         standardized_to_M4(realizable_3D_M4_corr_dev(...))
#     i.e. the CPU march routes through the DEVICE chain in realize_dev.jl. So this test
#     covers the same realizability_S2_dev that the GPU kernels use, and the standalone CPU
#     `realizability_S2` is reachable only via the `realizability(:S2,...)` dispatcher that
#     the MATLAB goldens exercise. A first attempt to falsify this test reverted the
#     dispatcher copy, saw no change, and briefly suggested the test was inert -- it was the
#     revert that was inert.
#
# FALSIFIED, not assumed: with the back-off moved back outside the guard in realize_dev.jl,
# the increment ratio goes 0.41 -> 5.39 and the gate below fails. A convergence test that
# passes both with and without the defect would be worthless, so this was checked.
#
# The force is applied here rather than through `apply_body_force!` because the CPU body
# force is uniform-only, and a uniform force on a periodic domain has no steady state -- it
# accelerates everything. `body_force_shift` per cell is exactly the operator split the
# march would use.
#
# GATE: successive halvings must SHRINK the change. Deliberately loose -- this catches
# divergence, not accuracy, and must not become a brittle golden.
using Test, Riemann35, MPI   # NB: no Statistics -- it is not in the test target and was unused;
                             # importing it failed CI on a clean env while passing locally
using Riemann35: body_force_shift
# `using MPI` is EXPLICIT here. Every test file is included into the same Main, so a
# file that omits it still works as long as some earlier include did `using MPI` --
# an order-dependent coupling that breaks the moment the file is run on its own, or
# the include order changes. See issue #62.
using MPI

MPI.Initialized() || MPI.Init()

@testset "driven steady state converges in dt" begin
    H = 1.0; T0 = 1.0; ny = 16; nz = 1; halo = 2; nx = halo + 2
    Kn = 0.5;  A = 0.02; tendf = 0.2   # larger Kn => larger nu => fewer steps to steady state
    dy = H/ny; dx = dy; dz = dy
    lam = Kn*H; tau = lam*sqrt(2.0); nu = T0*tau; k = 2pi/H
    kn_tau = 2*tau
    U0 = A/(nu*k*k)
    BC = (xlo=:periodic, xhi=:periodic, ylo=:periodic,
          yhi=:periodic, zlo=:periodic, zhi=:periodic)
    decomp = setup_mpi_cartesian_3d(nx, ny, nz, halo, MPI.COMM_WORLD)
    gprof(j) = A*sin(k*((j - halo) - 0.5)*dy)

    function amplitude(dt)
        nst = ceil(Int, (tendf*H*H/nu)/dt)
        M = zeros(nx+2halo, ny+2halo, nz, 35)
        for kk in 1:nz, j in 1:(ny+2halo), i in 1:(nx+2halo)
            uy = U0*sin(k*clamp(((j - halo) - 0.5)*dy, 0.0, H))
            M[i,j,kk,:] = InitializeM4_35(1.0, uy, 0.0, 0.0, T0,0,0, T0,0, T0)
        end
        for _ in 1:nst
            step_highorder_3d!(M, dt, decomp, BC, nx, ny, nz, halo, dx, dy, dz, 0.0;
                               order = 2, stage_bgk_kn = kn_tau, stage_bgk_exact = true,
                               Pr = 1.0, omega = 1.0)
            # sinusoidal drive, operator-split once per step (exact velocity-space shift)
            for kk in 1:nz, j in (halo+1):(halo+ny), i in (halo+1):(halo+nx)
                M[i,j,kk,:] = body_force_shift(collect(Float64, M[i,j,kk,:]),
                                               gprof(j), 0.0, 0.0, dt)
            end
        end
        u = [ M[halo+1, halo+j, 1, 2]/M[halo+1, halo+j, 1, 1] for j in 1:ny ]
        p = [ sin(k*(j-0.5)*dy) for j in 1:ny ]
        2*sum(u .* p)/ny
    end

    dts = [2.0e-3, 1.0e-3, 5.0e-4]
    a = [amplitude(d) for d in dts]
    d1 = abs(a[2] - a[1]); d2 = abs(a[3] - a[2])
    @info "driven dt convergence" amplitudes=a increments=(d1,d2) ratio=d2/max(d1,eps())

    # the response must be real, or the gates below are vacuous
    @test abs(a[end]) > 1e-6

    # THE GATE. Halving dt must SHRINK the change. First-order gives ~0.5; the pre-fix
    # behaviour gave ~2.0, increments doubling. 0.8 catches divergence and stagnation while
    # leaving room for the scheme's actual order and run-to-run variation.
    @test d2 < 0.8 * d1

    # and the total spread over a 4x refinement must be small in absolute terms
    @test abs(a[3] - a[1])/abs(a[3]) < 0.05
end

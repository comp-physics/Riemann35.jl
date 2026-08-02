"""
Main Test Entry Point for Riemann35.jl

This file is the entry point for Julia's package testing system (Pkg.test()).
It runs all unit tests and optionally integration tests.

For more control over test execution, use the shell scripts:
- ./test/run_tests.sh         # All tests with nice formatting
- ./test/run_mpi_tests.sh     # MPI-specific tests
- ./test/ci_test.jl           # CI-friendly runner for Julia 1.9

Environment Variables:
- TEST_INTEGRATION: Set to "false" to skip integration tests
"""

using Test
using Riemann35
using LinearAlgebra

# Test tolerance
const TOL = 1e-10

@testset "Riemann35.jl" begin
    @testset "Unit Tests" begin
        # Core functionality unit tests
        include("test_autogen.jl")
        include("test_moment_conversions.jl")
        include("test_adaptive_quadrature.jl")
        include("test_chyqmom_nodes.jl")
        include("test_initialization.jl")
        include("test_initial_conditions.jl")
        include("test_simulation_runner.jl")
        include("test_realizability.jl")
        include("test_realizability_oracle.jl")
        include("test_closures.jl")
        include("test_reduce26.jl")
        include("test_hll_realize_precondition.jl")  # issue #39: the pass is load-bearing
        include("test_hyqmom_closure_golden.jl")
        include("test_highorder_1d.jl")
        include("test_highorder_3d.jl")
        include("test_rodney_cases.jl")
        include("test_numerical_schemes.jl")
        include("test_scaling_limiter.jl")
        include("test_riemann_solver.jl")

        # Regression and bug fix tests
        include("test_z_eigenvalue_fix.jl")
        
        # Golden file comparison tests
        include("test_golden_files.jl")
        
        # Additional comprehensive unit tests
        include("test_timestep.jl")
        include("test_realizability_bounds.jl")
        include("test_conservation.jl")
        include("test_boundary_conditions.jl")
        include("test_eigenvalue_ordering.jl")
        include("test_initial_conditions_properties.jl")
        include("test_numerical_accuracy.jl")
        include("test_s3max.jl")
        include("test_roeps3.jl")
        include("test_moment_correction.jl")
        include("test_face_bc.jl")   # direction-agnostic per-face BC + sponge (byte-identical presets)
        include("test_esbgk.jl")     # ES-BGK + VHS transport (opt-in; defaults bitwise BGK)
        include("test_wall_bc.jl")   # Maxwell-accommodating wall ghost (opt-in :wall face)
        include("test_wall_conservation.jl")  # a wall must pass ZERO net mass (see header)
        include("test_body_force_dev.jl")     # device body force vs the CPU form
        include("test_run_params.jl")         # results headers must record what was READ
        include("test_projection_identity.jl")   # a projection must FIX every point already in the set
        include("test_driven_dt_convergence.jl") # a driven steady state must have a dt -> 0 limit
        include("test_dvm_reference.jl")         # the ground truth everything is measured against
        include("test_steady_state_time_convergence.jl")  # weak guard: a steady state must have a t_end limit
        include("test_kfvs_wall.jl")             # the wall must conserve mass exactly (#36)
        include("test_wall_tangential_convention.jl")  # uw1->t1, uw2->t2 (see header)

        # θ*-IDP limiter goldens — pins BOTH the bisection baseline (frozen,
        # bit-for-bit fallback guard via theta_closed=false) and the closed-form
        # default path (frozen golden + finite + realizable).
        include("test_theta_star_goldens.jl")
    end
    
    # Integration tests (Julia vs MATLAB golden files)
    # The test file handles its own skip logic if golden files are missing
    # or if TEST_INTEGRATION=false
    include("test_integration.jl")
end

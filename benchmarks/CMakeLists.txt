# Formulation convergence study (WP15, Fu et al. 2023 Fig. 2a,b): GMRES iteration counts of
# PMCHWT / ICTF / MCTF without and with the Jacobi preconditioner on Si and Ag rough surfaces,
# dense operator. Long-running (hours, ~14 GB per system): run by hand, see
# benchmarks/results/formulation_convergence.md.
add_executable(specklebem_formulation_convergence formulation_convergence.cpp)
target_link_libraries(specklebem_formulation_convergence PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)

# Dense assembly profile (WP-P2): per-class element_blocks timings, degree and touching-rule
# statistics, schedule makespan and timed DenseStrategy::build runs on rough boxes and spheres.
# Run by hand, see benchmarks/results/dense_assembly_profile.md.
add_executable(specklebem_dense_assembly_profile dense_assembly_profile.cpp)
target_link_libraries(specklebem_dense_assembly_profile PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Same flags as src/kernels/operators.cpp, so the isolated fastmath loops vectorise.
    set_source_files_properties(dense_assembly_profile.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-math-errno;-fno-trapping-math")
endif()

# Ag sphere with MLFMM vs Mie (WP22a, Phase 4 DoD): one case per process, run by hand, see
# benchmarks/results/ag_sphere_4um_mlfmm.md. Shares the case code with the validation-large test
# (tests/support/ag_sphere_mlfmm_support.hpp).
add_executable(specklebem_ag_sphere_mlfmm ag_sphere_mlfmm.cpp ${PROJECT_SOURCE_DIR}/tests/support/system_memory.cpp)
target_include_directories(specklebem_ag_sphere_mlfmm PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(specklebem_ag_sphere_mlfmm PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)

# MLFMM scaling study (WP22b1, Phase 4 DoD; Fig. 2(c) of the paper: wall time vs N for
# preconditioned MLFMM (Ag) and ICTF-MLFMM (Si) rough surfaces): one case per process, run by
# hand, see benchmarks/results/mlfmm_scaling.md. Shares the case code with the smoke test
# (tests/support/mlfmm_scaling_support.hpp); mlfmm_scaling_fit.py fits the exponents.
add_executable(specklebem_mlfmm_scaling mlfmm_scaling.cpp ${PROJECT_SOURCE_DIR}/tests/support/system_memory.cpp)
target_include_directories(specklebem_mlfmm_scaling PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(specklebem_mlfmm_scaling PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)

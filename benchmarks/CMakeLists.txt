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

# Benchmarks reproduce Fig. 2(c) of the paper: wall time vs N for dense LU,
# preconditioned MLFMM (Ag) and ICTF-MLFMM (Si). Added in Phase 4.
# add_executable(bench_matvec bench_matvec.cpp)

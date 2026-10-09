# Dense assembly profile (WP-P2): per-class element_blocks timings, degree and touching-rule
# statistics, schedule makespan and timed DenseStrategy::build runs on rough boxes and spheres.
# Run by hand, see benchmarks/results/dense_assembly_profile.md.
add_executable(specklebem_dense_assembly_profile dense_assembly_profile.cpp)
target_link_libraries(specklebem_dense_assembly_profile PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)

# Benchmarks reproduce Fig. 2(c) of the paper: wall time vs N for dense LU,
# preconditioned MLFMM (Ag) and ICTF-MLFMM (Si). Added in Phase 4.
# add_executable(bench_matvec bench_matvec.cpp)

# Formulation convergence study (WP15, Fu et al. 2023 Fig. 2a,b): GMRES iteration counts of
# PMCHWT / ICTF / MCTF without and with the Jacobi preconditioner on Si and Ag rough surfaces,
# dense operator. Long-running (hours, ~14 GB per system): run by hand, see
# benchmarks/results/formulation_convergence.md.
add_executable(specklebem_formulation_convergence formulation_convergence.cpp)
target_link_libraries(specklebem_formulation_convergence PRIVATE SpeckleBem::specklebem SpeckleBem::warnings)

# Benchmarks reproduce Fig. 2(c) of the paper: wall time vs N for dense LU,
# preconditioned MLFMM (Ag) and ICTF-MLFMM (Si). Added in Phase 4.
# add_executable(bench_matvec bench_matvec.cpp)

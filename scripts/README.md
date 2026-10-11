# Scripts

- `format.sh` – run clang-format / ruff over the tree
- `server_run_kit/` – (WP22d) large-area MLFMM runs on a Linux server: build instructions, NUMA
  settings, run list with memory/time estimates, what to send back (`README.md`), and
  `run_large_area.sh` (one case: estimate, memory check, solve with logging, summary row)
- `generate_reference_data.py` – (Phase 2) create Mie reference CSVs under tests/data
- `plot_convergence.py` – (Phase 3) residual-vs-iteration plots from solver logs

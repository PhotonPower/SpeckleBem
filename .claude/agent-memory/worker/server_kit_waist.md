---
name: server-kit-waist
description: WP22d server run kit pins w0 = L/4 for Si and Ag (forwarded explicitly) although ADR 0006 asks L/6 for quantitative Si work since WP-B3
metadata:
  type: project
---

The server run kit (`scripts/server_run_kit/run_large_area.sh`) always forwards
`--waist-factor` (default 4) and adds `--allow-wide-beam` for Si below a factor of 6
(coordinator decision 2026-10-12, WP-B3 review C1). The benchmark default for Si became L/6
with WP-B3, so any recorded Si command of `specklebem_mlfmm_scaling` / `specklebem_fresnel_flat`
made before WP-B3 needs `--waist-factor 4 --allow-wide-beam` (or `--allow-wide-beam` with an
explicit `--w0` above L/6) to reproduce.

**Why:** the kit is a scaling/feasibility measurement comparable with the desktop proxy and the
WP22b3 series (all at L/4); the lead runs it at a pre-WP-B3 commit.

**How to apply:** when a default changes in a benchmark support header, grep
`benchmarks/results/*.md`, `benchmarks/README.md` and the server kit for commands that relied on
the old default and make the value explicit. See [[rough-box-grading]].

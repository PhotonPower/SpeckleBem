# Server run kit: Si 30 × 30 µm² (WP22d)

Everything needed to run the Phase 4 target "Si 30×30 µm², σ = 100 nm, λ/10 mesh" (docs/01) on a
large Linux server and to send the results back, plus a server Ag series that closes the Ag part
of the scaling DoD. The proxy on the 128 GB development machine, the estimates behind the numbers
below and the record template are in
[`benchmarks/results/si_large_area.md`](../../benchmarks/results/si_large_area.md).

Contents:

- `README.md` — this file: build, machine settings, run list, what to send back.
- `run_large_area.sh` — one case per call (or a series: `--L a,b,c`): estimate, memory check,
  solve with logging, summary row (`--help` lists the options). POSIX sh (dash, bash, busybox
  ash).

## 1. Build (Linux, `release` preset)

Requirements: CMake ≥ 3.25, Ninja, a C++20 compiler (GCC ≥ 12 or Clang ≥ 16; CI uses GCC 13 and
Clang 18 on Ubuntu 24.04), OpenMP, Eigen 3.4, OpenBLAS + LAPACK, git. spdlog and Catch2 v3 are
fetched by CMake if they are not installed (needs internet access at configure time; offline
servers: see below).

Debian / Ubuntu 24.04:

```sh
sudo apt-get install -y build-essential cmake ninja-build git libeigen3-dev \
    libopenblas-dev liblapack-dev libspdlog-dev catch2 numactl
```

Ubuntu 22.04 has CMake 3.22 and GCC 11: `pip install --user cmake` (or the Kitware APT
repository), `sudo apt-get install g++-12` and configure with `CXX=g++-12 cmake --preset ...`;
its `catch2` package is v2 (CMake then fetches v3, or use `-DSPECKLEBEM_BUILD_TESTS=OFF`).

RHEL / Rocky / Alma 8 and 9 (Eigen, OpenBLAS, spdlog and Ninja come from EPEL / CRB; the system
GCC 8 / 11 is too old):

```sh
sudo dnf install -y epel-release && sudo dnf config-manager --set-enabled crb   # EL8: powertools
sudo dnf install -y gcc-toolset-13 cmake ninja-build git eigen3-devel openblas-devel \
    lapack-devel spdlog-devel numactl
scl enable gcc-toolset-13 bash      # then configure and build in this shell
```

**Offline servers** (no internet at configure time). `cmake/Dependencies.cmake` uses an installed
package first (`find_package`) and fetches only what is missing: Eigen 3.4.0, spdlog v1.15.3,
Catch2 v3.16.1 (tests only) and pybind11 v2.13.6 (Python bindings only, off below). Either
install the packages (`libeigen3-dev`, `libspdlog-dev`, `catch2` ≥ v3 / `eigen3-devel`,
`spdlog-devel`), or clone the sources on a machine with internet, copy them over and point CMake
at them, which skips the download:

```sh
git clone --depth 1 --branch v1.15.3 https://github.com/gabime/spdlog.git
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
git clone --depth 1 --branch v3.16.1 https://github.com/catchorg/Catch2.git
# on the server, added to the configure line below:
#   -DFETCHCONTENT_SOURCE_DIR_SPDLOG=/path/spdlog -DFETCHCONTENT_SOURCE_DIR_EIGEN=/path/eigen
#   -DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/Catch2   (or -DSPECKLEBEM_BUILD_TESTS=OFF: the
#   benchmark is still built, the unit tests are not)
#   (-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/path/pybind11 only with the Python bindings)
```

Configure, build and check (from the repository root; the Python bindings are not needed):

```sh
git checkout <commit given by the coordinator>
cmake --preset release -DSPECKLEBEM_BUILD_BENCHMARKS=ON -DSPECKLEBEM_BUILD_PYTHON=OFF
cmake --build --preset release -j
ctest --preset release -LE "validation|slow" -j 16        # unit tests, a few minutes
ctest --preset release -R mlfmm_scaling                   # the benchmark's own smoke tests
```

The binary is `build/release/benchmarks/specklebem_mlfmm_scaling`. CMake's log line
`-- Found BLAS` names the BLAS in use (also recorded by the script from `CMakeCache.txt`).

**BLAS.** OpenBLAS is the reference (the development machine uses it). The MLFMM solve itself
hardly uses BLAS (near field: sparse; far field: own OpenMP loops; GMRES CGS2: Eigen), so MKL
gives no measurable gain here. If MKL is preferred anyway: `source /opt/intel/oneapi/setvars.sh`,
`cmake --preset release-mkl -DSPECKLEBEM_BUILD_BENCHMARKS=ON -DSPECKLEBEM_BUILD_PYTHON=OFF`,
build with `--preset release-mkl`, set `MKL_THREADING_LAYER=GNU` (GCC's libgomp, not Intel's
libiomp5, otherwise two OpenMP runtimes compete) and pass `--build-dir build/release-mkl` to the
script.

## 2. Machine settings (large NUMA servers)

**Threads: use the physical core count.** The script takes `--threads`, else an
`OMP_NUM_THREADS` already set in the environment (by you or the batch system), else all hardware
threads (`nproc`, which counts both SMT siblings of a core). On the server, set it once to the
number of physical cores (`lscpu`: "Core(s) per socket" × "Socket(s)"), e.g.
`export OMP_NUM_THREADS=128` on a 2 × 64-core node, before calling the script. Why:

- the Si matvec is 91–97 % far-field matvec, which is floating-point bound (it streams only
  ~5 GB/s of radiation patterns at the proxy); the two SMT threads of a core share its
  floating-point units and caches, so a second thread per core adds little, while every OpenMP
  loop of the level sweeps then synchronises twice as many threads;
- the Ag solve and the CGS2 orthogonalisation are memory-bandwidth bound, which SMT does not
  raise;
- with at most one thread per core the script sets `OMP_PLACES=cores` (and
  `OMP_PROC_BIND=spread`): one thread on each core, spread over all sockets. With
  `OMP_PLACES=threads` and fewer threads than hardware threads, spread binding can put two
  threads on the siblings of one core (Linux usually numbers the siblings of cores 0…n−1 as
  n…2n−1) and leave other cores idle. More threads than cores keep `OMP_PLACES=threads`.
  `OMP_PLACES` / `OMP_PROC_BIND` set in the environment are kept.

The desktop reference runs used all 24 hardware threads of the 12 cores. Whether SMT helps on the
server is not known: if time allows, repeat run 2 once with `--threads $(nproc)` and record both
(`threads=` and `places=` are annotated in `row.txt`).

**NUMA.** `numactl --interleave=all` is used when the machine has more than one NUMA node and
`numactl` exists (`--numa auto`, the default; without `numactl` the nodes are counted in
`/sys/devices/system/node`, a warning is logged and the run uses first-touch). Reason: the large
arrays (near-field matrix, radiation patterns, Krylov basis) are partly first-touched by one
thread, so with the default first-touch policy they end up on one socket and every other socket
reads them remotely; interleaving spreads the pages and the bandwidth evenly. `--numa none`
keeps first-touch (try it once on a two-socket machine if time allows: the faster setting is
worth recording).

Example for a 2 × 64-core node with SMT (`nproc` = 256):

```sh
export OMP_NUM_THREADS=128                         # physical cores; script: OMP_PLACES=cores
scripts/server_run_kit/run_large_area.sh --L 20e-6 --label L20     # interleaved (auto)
```

Single-socket comparison (only if one socket's memory holds the case): bind the whole script to
socket 0 and switch off the script's own interleaving, which would otherwise override the
binding of the memory:

```sh
numactl --cpunodebind=0 --membind=0 scripts/server_run_kit/run_large_area.sh \
    --numa none --threads 64 --L 20e-6 --label L20_socket0
```

The memory check sees the whole node's MemAvailable, not the socket's: compare the "need" line
with `numactl -H` ("node 0 free") yourself.

**Other recommendations** (by hand): no other large jobs on the node; transparent huge pages
`madvise` or `always` (recorded, not changed); under a batch system request the whole node and
all of its memory. The script reads the cgroup memory limit and uses it in the memory check: for
cgroup v2 the smallest numeric `memory.max` on the path from its own cgroup up to the root (under
Slurm the limit sits on the job's cgroup, an ancestor, while the task's own cgroup says `max`),
for cgroup v1 `hierarchical_memory_limit` from `memory.stat` (includes the ancestors); every
value read is listed in `system.txt`.

## 3. Running a case

```sh
scripts/server_run_kit/run_large_area.sh --dry-run            # check, print the command
scripts/server_run_kit/run_large_area.sh --L 2e-6 --label smoke
scripts/server_run_kit/run_large_area.sh --matvecs 1 --label target   # Si L = 30 um (default)
scripts/server_run_kit/run_large_area.sh --material ag --L 4e-6,7e-6,11e-6,17.5e-6 \
    --max-iter 4000 --label agseries                          # a series: one case after another
```

Multi-hour runs: start them inside `tmux` / `screen` or with `nohup ... &` (the script stops
the solve on SIGHUP, so a dropped SSH session would otherwise end the run).

Per call the script

1. records the system (`system.txt`: date, `uname`, git commit and status, compiler and BLAS
   from `CMakeCache.txt`, OpenMP / BLAS environment, `lscpu`, `numactl -H`, `/proc/meminfo`,
   cgroup memory limits, transparent huge pages, load average; every item is optional);
2. runs `--estimate-only` (`estimate.log`: mesh, 2N, octree, near-field / pattern / table
   estimates, setup peak model; seconds, no large allocation; the far-field options are
   validated here too) and refuses to start if the setup peak model + the Krylov basis
   (`--krylov-iters`, default 150 for Si, `--max-iter` for Ag) + a reserve (`--margin-gb`,
   default max(16 GB, 3 % of the memory)) exceed the available memory (MemAvailable, or the
   cgroup limit if lower; exit code 3, also with `--dry-run`; `--force-memory` overrides and also
   lifts the benchmark's own guard). The model is conservative: the measured peak minus the Krylov
   basis was 0.72–0.86 × the model on the development machine;
3. runs the solve (`stdout.log`, `stderr.log`), sampling VmRSS / VmHWM of the process,
   MemAvailable and the load every `--sample` s, default 30 (`memory_samples.tsv`); the
   benchmark prints one `HIST` line per GMRES iteration (`gmres_history.tsv`: iteration,
   residual, seconds since the solve started). Before GMRES it times `--matvecs` rounds of
   (full, near, far) matvec after one untimed round, i.e. about 2 (matvecs + 1) full-matvec
   equivalents: 12 with the default 5, 4 with `--matvecs 1` (the large runs below);
4. writes `row.txt` (the benchmark's `ROW` line in the format of
   `benchmarks/results/mlfmm_scaling_rows.txt`, plus the annotations `gmres=cgs2
   machine=server threads= places= numa= matvecs= vmhwm_gb= rss_max_gb= load0= wall_s=`) and
   `summary.txt`. If the power summary after the solve fails (`--far-field`), the solve is kept:
   the `ROW` line is written without R / A and with `post_error=1`, the error is in
   `stderr.log`, exit status 4.

Results go to `server_runs/<date>_<time>_<material>_L<L>[_label]/` in the repository root
(ignored by git; `--out-dir` moves them). Exit status:

| status | meaning |
|---:|---|
| 0 | done (also a run that stopped at `--max-iter`: `conv=0` in `row.txt`) |
| 1 | error in the benchmark (`stderr.log`) |
| 2 | the benchmark's own memory guard, or an option / estimate error of the script |
| 3 | refused by the memory check (also with `--dry-run`) |
| 4 | solved, but the power summary failed (`row.txt` without R / A, `post_error=1`) |
| 130 | interrupted (Ctrl-C, `kill`, SIGHUP) |
| 137 | killed by SIGKILL, almost always the kernel's out-of-memory killer (`dmesg`) |

A series (`--L a,b,c`) runs the cases in order, one run directory each, and stops at the first
failed case (any status other than 0 and 4); with `--dry-run` it checks every case and returns
the largest status.

Interrupting: Ctrl-C or `kill <script pid>` stops the script and the solve (INT / TERM / HUP
are forwarded to the solve); `gmres_history.tsv` and `summary.txt` are still written. The solve
has no restart files; a run that is stopped must be repeated. In a series, Ctrl-C stops the
running case and with it the series.

## 4. Run list

Same settings as the proxy unless stated: Si, σ = 100 nm, Lc = 500 nm, h = 50 nm, uniform 50 nm
closing box with the fine band, rigorous beam w₀ = L/4, ICTF, d₀ = 3, GMRES tol 1e-3, seed 1.
**Memory and times are extrapolations from the development machine, not measurements**
(`si_large_area.md`, "Extrapolation"; Ag: the WP22b3 rows and projection in
`benchmarks/results/mlfmm_scaling.md`). "Check needs" = model + Krylov basis + margin, the figure
the script compares with the available memory (margin 33 GB assumed for a 1.1 TB node). Times
are for the 12-core / 24-thread development machine; the server's Si factor is what run 2
measures. Timing matvecs = the 2 (matvecs + 1) full-matvec equivalents before GMRES; post = the
`--far-field 1` power summary, which scales with directions × triangles (324 s at the proxy,
2N = 1.1·10⁶, so ≈ 324 s × 2N / 1.1·10⁶); total = setup + timing matvecs + solve + post.

| # | case | command (after `scripts/server_run_kit/run_large_area.sh`) | 2N | model [GB] | check needs [GB] | expected peak [GB] | setup | timing matvecs | solve (desktop) | post | **total (desktop)** | purpose |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| 1 | smoke: Si L = 2 µm | `--L 2e-6 --label smoke` | 127 680 | 14 | 47 | 10–12 | < 1 min | < 1 min | 2–5 min | ~40 s | **3–7 min** | build, environment, output files |
| 2 | Si L = 20 µm | `--L 20e-6 --label L20` | 3 004 800 | 285 | 325 | 210–250 | 8–15 min | 8–12 min (12 × 42–62 s) | 1.0–2.2 h | ~15 min | **1.5–2.9 h** | server speed vs desktop; checks the extrapolation |
| 3 | **Si L = 30 µm (target)** | `--matvecs 1 --label target` (L = 30 µm is the default) | 5 947 200 | 823 | 870 | 600–720 | 27–53 min | 5–11 min (4 × 82–158 s) | 2.2–5.6 h | ~30 min | **3.2–7.1 h** | the Phase 4 target |
| 3b | Si L = 30 µm, graded box, q = 0.8 (alternative) | `--L 30e-6 --box 100e-9 --lq 0.8 --matvecs 1 --label graded` | 3 999 600 | 627 | 670 | 460–550 | 18–36 min | 6–9 min (4 × 94–131 s) | 2.9–6.8 h | ~20 min | **3.6–7.9 h** | only if 3 does not fit (fewer unknowns, ~1.4 × the iterations) |
| 4 | **Ag series, server** (recommended): 4a–4d in one call | `--material ag --L 4e-6,7e-6,11e-6,17.5e-6 --max-iter 4000 --label agseries` | see below | | | | | | | | **5.2–7.0 h** | closes the Ag scaling DoD by measurement on one machine |
| 4a | Ag L = 4 µm | (series) | 69 600 | 4.5 | 42 | ≈ 4.2 | 13 s | 2 s | 348 s (1 394 it) | ~20 s | **6 min** | |
| 4b | Ag L = 7 µm | (series) | 186 480 | 9.1 | 54 | ≈ 10.9 | 33 s | 5 s | 993 s (1 518 it) | ~1 min | **18 min** | |
| 4c | Ag L = 11 µm | (series) | 422 400 | 39.5 | 100 | ≈ 43 | 85 s | 14 s | 4 748 s (2 272 it) | ~2 min | **1.4 h** | |
| 4d | Ag L = 17.5 µm | (series) | 1 017 450 | 64 | 162 | 87–101 | 3 min | 0.5 min | 3.3–5.1 h projected (2 530–3 360 it) | ~5 min | **3.4–5.2 h** | |
| 5 | Si L = 32 µm, q = 0.8 (optional) | `--L 32e-6 --lq 0.8 --matvecs 1 --label L32q08` | 6 650 880 | 546 | 595 | 400–480 | 13–26 min | 6–12 min (4 × 91–185 s) | 2.0–5.0 h | ~33 min | **2.9–6.2 h** | octree quantisation: larger L, less memory than 30 µm |

Notes:

- `--matvecs 1` (runs 3, 3b, 5) times one round of (full, near, far) after one untimed round
  instead of five: about 4 instead of 12 full-matvec equivalents, 11–21 min less at L = 30 µm.
  The `ROW` fields `matvec_s`, `near_mv_s` and `far_mv_s` are then single applies, not the
  median of 5 (`matvecs=1` in `row.txt`); run 2 keeps the default 5 and is the matvec
  calibration. Setup, iterations, solve and memory are unaffected.
- **Run 4, the server Ag series**, replaces the former single Ag L = 17.5 µm run: one server point
  cannot be fitted together with the desktop series, and the Ag solve is bandwidth-bound (its
  CGS2 orthogonalisation was 30–44 % of the WP22b3 Ag solves), so the Si server / desktop factor
  of run 2 does not carry over to it. The four sizes span 2N = 7.0·10⁴ … 1.02·10⁶ and give the
  `machine=server` series its own γ_solve (`benchmarks/mlfmm_scaling_fit.py` keeps it separate
  from the desktop rows). The options are those of the WP22b3 Ag series (ROW lines
  `material=ag box=100 fine=0 restart=0 h_nm=50 waist=4 beam=angular_spectrum d0=3 tol=0.001
  lq=1 seed=1`, σ = 50 nm: the script's Ag defaults `--sigma 50e-9 --box 100e-9 --lq 1`, the
  benchmark's defaults for the rest; ICTF + left Jacobi chosen automatically; `--matvecs 5`)
  with `--max-iter 4000`; `--far-field 1` only adds the power summary after the solve. Times in
  the table are the **desktop references** (WP22b3: 348 / 993 / 4 748 s measured for 4a–4c; 4d
  was capped at 1 450 iterations there, residual 4.1·10⁻³, and is projected at 2 530–3 360
  iterations, 3.3–5.1 h); the server times are unknown. The iteration counts do not depend on the
  machine up to rounding (a different thread count changes the summation order of the parallel
  reductions, so they may differ by a few iterations): compare them with the desktop counts
  1 394 / 1 518 / 2 272 as a consistency check. The memory check reserves the full
  4 000-iteration Krylov basis (65 GB at 4d); expected peaks: 4a–4c as measured on the desktop
  (the Krylov basis follows the iterations), 4d 87–101 GB. If 4d stops at the cap without
  converging (`conv=0` in `row.txt`, exit status 0), send it anyway and note it. The series
  stops at the first failed case.
- Run 3 needs ~870 GB available for the check. If the node has less MemAvailable than that but
  more than ~760 GB, `--force-memory` is justified: the measured peak minus Krylov never
  exceeded 0.86 × the model (≤ 708 GB + 14 GB Krylov). Below that, run 3b instead.
- The `ROW` line carries the reflectance R, R_coarse (half the far-field nodes), the absorbed
  fraction A and the post-processing time (`--far-field 1`; `--far-field 0` skips it). At
  tol 1e-3 these are consistency checks, not quantitative reflectances.

Order: 1 → 2 → 3 (each one checks the next estimate: compare its `row.txt` with the table),
then the Ag series 4, then optionally 5. One case at a time: two large cases on one node share
the bandwidth and falsify both timings.

## 5. What to send back

For every run, the run directory as a tarball (the script prints the command at the end):

```sh
tar czf "<run>.tar.gz" -C server_runs "<run>"
```

It contains only text files (`system.txt`, `run.log`, `estimate.log`, `stdout.log`,
`stderr.log`, `gmres_history.tsv`, `memory_samples.tsv`, `row.txt`, `summary.txt`; a few MB).
No current or mesh files are written; if the coordinator asks for currents later, that is a
separate run with `--extra` options agreed then. Please add a short note with: the server
(CPU model, sockets × cores, memory, NUMA nodes) if `system.txt` lacks it, anything unusual
(other jobs on the node, changed settings, interruptions, `--force-memory`), and for run 2 /
3 whether `--numa` or `--threads` were varied.

We record the results in `benchmarks/results/si_large_area.md` ("Server results": the ROW
fields, the server description) and the `row.txt` lines in
`benchmarks/results/mlfmm_scaling_rows.txt` (`machine=server` keeps them a separate series in
`benchmarks/mlfmm_scaling_fit.py`).

## 6. Troubleshooting

- `estimate failed`: run the printed estimate command by hand; an exception text names the cause
  (e.g. the waist rule, or a `--far-field` spacing above 25.7°).
- `REFUSED: need ... > available`: free memory or use the next smaller case; `--force-memory`
  only if you know the model is pessimistic for your machine (the measured peak minus Krylov
  was 14–28 % below the model on the development machine).
- Slow start: before the first `HIST` line the run does the setup and the timing matvecs. On the
  desktop that is 16–27 min at L = 20 µm (run 2: setup 8–15 min + 12 × 42–62 s) and 33–64 min
  at L = 30 µm (run 3: setup 27–53 min + 4 × 82–158 s); the server should be faster. If there is
  no `HIST` line after ~1.5 h at L = 30 µm, or an iteration takes longer than ~3 min (desktop
  1.4–2.9 min per iteration at 30 µm), the server is slower than the desktop: check in `run.log`
  the thread line (`OMP_NUM_THREADS` not 1, `OMP_PLACES`), the NUMA policy (`numactl -H`,
  interleave active?) and the load in `memory_samples.tsv`.
- Out-of-memory kill: exit status 137, `stderr.log` is empty and `dmesg` shows the OOM killer;
  send the run directory anyway (`memory_samples.tsv` shows how far it got).
- Exit status 4: the solve finished but the power summary failed; `row.txt` has the solve
  fields (`post_error=1`), `stderr.log` the error. Send it as it is.

## 7. How this kit was tested

On the Windows development machine (2026-10-11, `win-release` build of this branch, no Linux
machine available), with MSYS2's `dash` (POSIX sh) and `bash`, output to a scratch directory:

- `--dry-run` with the binary (estimate, memory check, printed command) and without a binary
  (check skipped with a warning); the memory refusal: L = 30 µm on the 128 GB machine, need
  853 GB > 101 GB available, exit 3 (also in `--dry-run`, review fix);
- option errors (review fixes): a missing value (`--L` at the end, `--L --dry-run`) and a
  non-numeric `--far-field` stop with exit 2; `--far-field 30` is refused by the benchmark at the
  estimate (the half-node grid would have n_φ = 3); `--far-field 0.0` switches the power summary
  off;
- the Ag series as one call, `--material ag --L 4e-6,7e-6,11e-6,17.5e-6 --dry-run`: four run
  directories, L = 17.5 µm refused locally (need 145 GB with the 16 GB margin), exit 3;
- a full Si run at L = 1 µm, σ = 100 nm (59 040 unknowns; 116 iterations, R = 0.311,
  A = 0.665, 159 s wall on a machine shared with another job): all files written, `row.txt`
  parsed by `benchmarks/mlfmm_scaling_fit.py`; repeated after the review fixes with
  `--matvecs 1` (same 116 iterations, R and A; peak 5.3 GB, 319 s wall at a load average of
  ~110 from other jobs; `row.txt` with `places= matvecs=1` parsed by the fit script);
- a full Ag run at L = 1 µm with the Ag defaults (8 400 unknowns, ICTF + left Jacobi, 525
  iterations, `--max-iter 4000`, 103 s wall);
- signal forwarding: TERM to the script 25 s into a run stopped the solve within a second,
  exit 130; after the review fixes TERM during GMRES (iteration 14) stopped the solve, exit 130,
  and `gmres_history.tsv` and `summary.txt` (with the last iteration) were still written.

Locally `/proc/<pid>/status` of the native Windows process has no VmHWM and the MSYS
`/proc/meminfo` no MemAvailable (`NA` columns; the benchmark's own `peak_gb` and memory query
are used instead), and there is no `numactl`, `lscpu` or cgroup. The cgroup code was run against
fake `/sys/fs/cgroup` trees with dash and bash: cgroup v2 under Slurm (limit on `job_<id>`, `max`
on the step and the task: the job's limit is used), a container (`0::/` and the limit at the
mount root), all levels `max` (no limit), a smaller limit below a larger one (the smaller is
used), cgroup v1 `hierarchical_memory_limit` (including the unlimited value 2^63 − 4096 and a
container whose own path is missing under the mount: the mount root is used), a hybrid v1 / v2
layout and no cgroup files at all. The other Linux-only branches were checked by reading:
VmRSS / VmHWM and MemAvailable are parsed in kB, `numactl -H` from its "available: N nodes"
line, `lscpu -p=CORE,SOCKET` for the physical cores. The C++ side the script relies on,
`tests/support/system_memory.cpp`, reads MemAvailable from `/proc/meminfo` (kB; fallback
`sysconf(_SC_AVPHYS_PAGES)`) for the benchmark's start guard and
`getrusage(RUSAGE_SELF).ru_maxrss` (KiB on Linux) for `peak_gb`; both are the standard Linux
interfaces and need no extra permission. The benchmark ignores cgroup limits; the script's own
check covers them.

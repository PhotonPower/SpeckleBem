#!/bin/sh
# SpeckleBem large-area run (WP22d server run kit): one MLFMM rough-surface case of
# specklebem_mlfmm_scaling with a memory check, full logging and a summary row.
#
# Usage: scripts/server_run_kit/run_large_area.sh [options]     (see --help; README.md)
#
# POSIX sh (dash, bash, busybox). Linux is the target; on other systems (Windows Git Bash) the
# missing /proc entries, numactl and lscpu are skipped. Nothing here is specific to one machine.
set -u

# ------------------------------------------------------------------------------------------
# Defaults (the Phase 4 Si 30 x 30 um target, docs/01; benchmarks/results/si_large_area.md)
material=si
L=30e-6
sigma=""          # empty: si 100e-9 (target), ag 50e-9 (the Ag scaling series)
box=""            # empty: si 50e-9 (uniform box), ag 100e-9 (graded lambda/5 box)
lq=1              # leaf radius quantile (1 = global leaf rule)
digits=3
tol=1e-3
max_iter=""       # empty: si 1000, ag 4000 (WP22b3 Ag L = 17.5 um: 2 530-3 360 projected)
far_field=1       # polar spacing of the reflection-hemisphere grid [deg]; 0 = off
matvecs=5         # timed matvec rounds before GMRES (each round: full + near + far apply)
threads=""        # empty: $OMP_NUM_THREADS if set, else all hardware threads (nproc)
krylov_iters=""   # iterations assumed for the Krylov memory: empty = si 150, ag = max_iter
margin_gb=""      # empty: max(16 GB, 3 % of the total memory)
numa=auto         # auto | interleave | none
build_dir=""      # empty: build/release
out_dir=""        # empty: server_runs (in the repository root)
label=""
extra=""
force_memory=0
dry_run=0
sample_s=30

usage() {
    cat <<'EOF'
run_large_area.sh - one large-area MLFMM case with memory check, logs and a summary row

Options (SI units, metres):
  --L <m>[,<m>...]      patch edge (default 30e-6); a comma-separated list runs the cases one
                        after another (one run directory each, stops at the first failure)
  --material si|ag      (default si)
  --sigma <m>           RMS height (default si 100e-9, ag 50e-9)
  --box <m>|auto        closing-box spacing (default si 50e-9 = uniform box, ag 100e-9)
  --lq <q>              leaf radius quantile, 1 = global leaf rule (default 1)
  --digits <d0>         MLFMM accuracy digits (default 3)
  --tol <t>             GMRES tolerance (default 1e-3)
  --max-iter <n>        GMRES iteration cap (default si 1000, ag 4000)
  --far-field <deg>     power summary grid spacing, 0 = off (default 1)
  --matvecs <n>         timed matvec rounds before GMRES, each full + near + far (default 5;
                        1 for the large cases: the ROW matvec times are then single applies)
  --threads <n>         OpenMP threads (default: $OMP_NUM_THREADS if set, else nproc; the
                        README recommends the physical core count)
  --krylov-iters <n>    iterations assumed in the memory check (default si 150, ag max-iter)
  --margin-gb <g>       memory reserve in the check (default max(16, 3 % of total))
  --numa auto|interleave|none   numactl --interleave=all (auto: if >1 node and numactl exists)
  --build-dir <dir>     build tree with benchmarks/ (default build/release)
  --out-dir <dir>       parent of the run directory (default server_runs)
  --label <text>        free text appended to the run directory name
  --extra "<args>"      further specklebem_mlfmm_scaling options, passed verbatim
  --sample <s>          memory sampling interval [s] (default 30)
  --force-memory        start even if the memory check fails (passes --force)
  --dry-run             estimate and check, print the commands, do not solve (exit 3 if the
                        memory check refuses)
  -h, --help            this text
EOF
}

die() {
    echo "run_large_area: $*" >&2
    exit 2
}

# Series: "--L a,b,c" runs this script once per value, in order (the last --L on a command line
# wins, so the original arguments are passed through unchanged). Stops at the first failed case
# (a refused or failed case, larger ones would fail too; status 4 = solved without the power
# summary does not stop it), except with --dry-run, which checks every case and returns the
# largest status.
series=""
series_dry=0
prev=""
for a in "$@"; do
    if [ "$prev" = "--L" ]; then
        case "$a" in *,*) series=$a ;; *) series="" ;; esac
    fi
    [ "$a" = "--dry-run" ] && series_dry=1
    prev=$a
done
if [ -n "$series" ]; then
    worst=0
    for one in $(printf '%s' "$series" | tr ',' ' '); do
        echo "run_large_area: series case L = $one" >&2
        sh "$0" "$@" --L "$one"
        rc=$?
        [ "$rc" -gt "$worst" ] && worst=$rc
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 4 ] && [ "$series_dry" -eq 0 ]; then
            echo "run_large_area: series stopped at L = $one (exit status $rc)" >&2
            exit "$rc"
        fi
    done
    exit "$worst"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --force-memory) force_memory=1; shift; continue ;;
        --dry-run) dry_run=1; shift; continue ;;
        -h | --help) usage; exit 0 ;;
        --L | --material | --sigma | --box | --lq | --digits | --tol | --max-iter | --far-field | \
            --matvecs | --threads | --krylov-iters | --margin-gb | --numa | --build-dir | \
            --out-dir | --label | --extra | --sample)
            [ $# -ge 2 ] || die "missing value after $1"
            # A value that looks like an option means the value was left out ("--L --dry-run").
            case "$2" in --*) [ "$1" = --extra ] || die "missing value after $1" ;; esac ;;
        *) usage >&2; die "unknown option $1" ;;
    esac
    case "$1" in
        --L) L=$2 ;;
        --material) material=$2 ;;
        --sigma) sigma=$2 ;;
        --box) box=$2 ;;
        --lq) lq=$2 ;;
        --digits) digits=$2 ;;
        --tol) tol=$2 ;;
        --max-iter) max_iter=$2 ;;
        --far-field) far_field=$2 ;;
        --matvecs) matvecs=$2 ;;
        --threads) threads=$2 ;;
        --krylov-iters) krylov_iters=$2 ;;
        --margin-gb) margin_gb=$2 ;;
        --numa) numa=$2 ;;
        --build-dir) build_dir=$2 ;;
        --out-dir) out_dir=$2 ;;
        --label) label=$2 ;;
        --extra) extra=$2 ;;
        --sample) sample_s=$2 ;;
        *) usage >&2; die "unknown option $1" ;;
    esac
    shift 2
done

case "$material" in
    si)
        : "${sigma:=100e-9}" "${box:=50e-9}" "${max_iter:=1000}"
        : "${krylov_iters:=150}" ;;
    ag)
        : "${sigma:=50e-9}" "${box:=100e-9}" "${max_iter:=4000}"
        : "${krylov_iters:=$max_iter}" ;;
    *) die "--material must be si or ag" ;;
esac
case "$numa" in auto | interleave | none) ;; *) die "--numa must be auto, interleave or none" ;; esac

# ------------------------------------------------------------------------------------------
# Locations (relative to the repository; no machine-specific paths)
script_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
: "${build_dir:=$repo/build/release}"
: "${out_dir:=$repo/server_runs}"
bin="$build_dir/benchmarks/specklebem_mlfmm_scaling"
[ -x "$bin" ] || [ -x "$bin.exe" ] || [ "$dry_run" -eq 1 ] ||
    die "benchmark not found: $bin (build with -DSPECKLEBEM_BUILD_BENCHMARKS=ON, README.md)"
[ -x "$bin" ] || { [ -x "$bin.exe" ] && bin="$bin.exe"; }

have() { command -v "$1" >/dev/null 2>&1; }

hw_threads=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
if [ -z "$threads" ]; then
    # An OMP_NUM_THREADS set by the user (or the batch system) wins; else all hardware threads,
    # as in the desktop reference runs (24 threads on 12 cores with SMT). README section 2
    # recommends the physical core count on the server.
    threads=${OMP_NUM_THREADS:-$hw_threads}
fi
[ "$threads" -ge 1 ] 2>/dev/null || die "--threads / OMP_NUM_THREADS must be a positive integer"
[ "$sample_s" -ge 1 ] 2>/dev/null || die "--sample must be a positive integer (seconds)"
[ "$matvecs" -ge 1 ] 2>/dev/null || die "--matvecs must be a positive integer"
physical_cores=""
if have lscpu; then
    physical_cores=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')
fi
export OMP_NUM_THREADS="$threads"
if [ -z "${OMP_PLACES:-}" ]; then
    # One thread per core when there are no more threads than cores: OMP_PLACES=threads with
    # spread binding can put pairs of threads on the two SMT siblings of one core (Linux
    # typically numbers the siblings of cores 0..n-1 as n..2n-1) and leave half the cores idle.
    if [ -n "$physical_cores" ] && [ "$threads" -le "$physical_cores" ] 2>/dev/null; then
        OMP_PLACES=cores
    else
        OMP_PLACES=threads
    fi
fi
: "${OMP_PROC_BIND:=spread}"
export OMP_PLACES OMP_PROC_BIND

numa_nodes=""
if have numactl; then
    numa_nodes=$(numactl -H 2>/dev/null | sed -n 's/^available: \([0-9][0-9]*\) nodes.*/\1/p')
fi
if [ -z "$numa_nodes" ]; then
    # Without numactl: count the nodes in sysfs (absent outside Linux: 1).
    numa_nodes=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l | tr -d ' ')
    [ "${numa_nodes:-0}" -ge 1 ] 2>/dev/null || numa_nodes=1
fi
numa_prefix=""
numa_mode=none
numa_note=""
if [ "$numa" = interleave ] || { [ "$numa" = auto ] && [ "$numa_nodes" -gt 1 ]; }; then
    if have numactl; then
        numa_prefix="numactl --interleave=all"
        numa_mode=interleave
    elif [ "$numa" = interleave ]; then
        die "--numa interleave needs numactl (install it, or pass --numa none)"
    else
        numa_note="WARNING: $numa_nodes NUMA nodes but no numactl: running with first-touch placement (install numactl for --interleave=all, README section 2)"
    fi
fi

case_args="--material $material --L $L --sigma $sigma --box-mesh-size $box \
--leaf-radius-quantile $lq --digits $digits --tol $tol --max-iter $max_iter \
--beam angular-spectrum"
# --far-field: numeric comparison (0, 0.0 and 0e0 are all "off"); the benchmark validates the
# spacing (both hemisphere grids) and the degree before the estimate, so the estimate call below
# gets the same post-processing options and a bad value fails in seconds.
post_args=""
awk -v f="$far_field" 'BEGIN {
    if (f !~ /^[+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/) exit 2
    exit !(f + 0 != 0)
}'
case $? in
    0) post_args="--far-field $far_field" ;;
    1) ;;
    *) die "--far-field must be a non-negative number of degrees (0 = off), got '$far_field'" ;;
esac
run_args="$case_args --history --matvecs $matvecs${post_args:+ $post_args}"
[ "$force_memory" -eq 1 ] && run_args="$run_args --force"
[ -n "$extra" ] && run_args="$run_args $extra"

stamp=$(date +%Y%m%d_%H%M%S)
name="${stamp}_${material}_L${L}"
[ -n "$label" ] && name="${name}_$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
run="$out_dir/$name"
mkdir -p "$run" || die "cannot create $run"

log() { echo "$*" | tee -a "$run/run.log"; }

# ------------------------------------------------------------------------------------------
# cgroup memory limit: the smallest limit that applies to this process. Under a batch system
# (Slurm with cgroup v2) the limit sits on an ancestor (e.g. .../job_<id>) while the process's
# own cgroup (.../step_batch/user/task_0) says "max", so every level counts.
#  - v2 (unified hierarchy, /sys/fs/cgroup/cgroup.controllers exists): walk from
#    /sys/fs/cgroup<path> (/proc/self/cgroup "0::<path>") up to /sys/fs/cgroup and take the
#    minimum numeric memory.max ("max" = no limit).
#  - v1 (memory controller line "<n>:...memory...:<path>"): memory.stat's
#    hierarchical_memory_limit, which already includes the ancestors; the process's own cgroup
#    first, then the controller's mount root (a container or job usually sees its own cgroup
#    there). Values near 2^63 mean no limit.
# Missing files are skipped; cg_report records every value read (system.txt).
cg_limit=""        # GB, empty: no limit found
cg_limit_bytes=""
cg_report=""
cg_take() {  # $1 = file, $2 = value in bytes or "max"
    cg_report="${cg_report}$1: $2
"
    case "$2" in '' | *[!0-9]*) return 0 ;; esac
    awk -v b="$2" 'BEGIN {exit !(b < 4e18)}' || return 0
    if [ -z "$cg_limit_bytes" ] || awk -v b="$2" -v m="$cg_limit_bytes" 'BEGIN {exit !(b < m)}'; then
        cg_limit_bytes=$2
    fi
}
cg2=$(sed -n 's/^0::\(.*\)$/\1/p' /proc/self/cgroup 2>/dev/null | head -1)
cg1=$(sed -n 's/^[0-9]*:[^:]*memory[^:]*:\(.*\)$/\1/p' /proc/self/cgroup 2>/dev/null | head -1)
if [ -r /sys/fs/cgroup/cgroup.controllers ]; then
    d="/sys/fs/cgroup${cg2%/}"
    while :; do
        [ -r "$d/memory.max" ] && cg_take "$d/memory.max" "$(cat "$d/memory.max" 2>/dev/null)"
        [ "$d" = /sys/fs/cgroup ] && break
        d=${d%/*}
        case "$d" in /sys/fs/cgroup/* | /sys/fs/cgroup) ;; *) break ;; esac
    done
fi
for f in "/sys/fs/cgroup/memory${cg1%/}/memory.stat" /sys/fs/cgroup/memory/memory.stat; do
    [ -n "$cg1" ] || break
    if [ -r "$f" ]; then
        cg_take "$f (hierarchical_memory_limit)" \
            "$(awk '$1 == "hierarchical_memory_limit" {print $2}' "$f" 2>/dev/null)"
        break
    fi
done
[ -n "$cg_limit_bytes" ] && cg_limit=$(awk -v b="$cg_limit_bytes" 'BEGIN {printf "%.1f", b / 1e9}')

# ------------------------------------------------------------------------------------------
# System information (each item optional)
{
    echo "== date"; date -u +%Y-%m-%dT%H:%M:%SZ
    echo "== uname"; uname -a
    echo "== git"
    if have git; then
        git -C "$repo" rev-parse HEAD 2>&1
        git -C "$repo" log -1 --format='%h %ad %s' --date=iso 2>&1
        git -C "$repo" status --short 2>&1 | head -20
    else
        echo "git not available"
    fi
    echo "== compiler / BLAS (CMakeCache.txt)"
    if [ -f "$build_dir/CMakeCache.txt" ]; then
        grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS|CMAKE_CXX_FLAGS_RELEASE|BLAS_LIBRARIES|LAPACK_LIBRARIES|BLA_VENDOR|SPECKLEBEM_[A-Z_]*|OpenMP_CXX_FLAGS|Eigen3_DIR|spdlog_DIR)[:=]' \
            "$build_dir/CMakeCache.txt"
        cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$build_dir/CMakeCache.txt")
        [ -n "$cxx" ] && "$cxx" --version 2>&1 | head -1
    else
        echo "no CMakeCache.txt in $build_dir"
    fi
    echo "== environment"
    env | grep -E '^(OMP_|OPENBLAS_|MKL_|GOMP_|KMP_|SLURM_JOB_ID|SLURM_MEM)' | sort
    echo "numa prefix: ${numa_prefix:-none} (nodes: $numa_nodes)"
    echo "== lscpu"; if have lscpu; then lscpu; else echo "lscpu not available"; fi
    echo "== numactl -H"; if have numactl; then numactl -H; else echo "numactl not available"; fi
    echo "== /proc/meminfo"; cat /proc/meminfo 2>/dev/null || echo "not available"
    echo "== cgroup memory limit"
    echo "/proc/self/cgroup: v2 path '${cg2:-none}', v1 memory path '${cg1:-none}'"
    printf '%s' "${cg_report:-no cgroup memory files found
}"
    if [ -n "$cg_limit" ]; then echo "effective limit: $cg_limit GB"; else echo "effective limit: none"; fi
    echo "== transparent huge pages"
    cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "not available"
    echo "== ulimit -v"; ulimit -v 2>/dev/null
    echo "== loadavg"; cat /proc/loadavg 2>/dev/null || echo "not available"
} >"$run/system.txt" 2>&1

log "run directory: $run"
log "case: $case_args"
log "threads: $OMP_NUM_THREADS (hardware threads $hw_threads, physical cores ${physical_cores:-?}; OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND), NUMA nodes: $numa_nodes, numa: ${numa_prefix:-none}"
[ -z "$numa_note" ] || log "$numa_note"

# ------------------------------------------------------------------------------------------
# Estimate and memory check
if [ -x "$bin" ]; then
    # shellcheck disable=SC2086
    "$bin" $case_args $post_args --estimate-only >"$run/estimate.log" 2>&1 ||
        { cat "$run/estimate.log" >&2; die "estimate failed (see $run/estimate.log)"; }
else
    echo "(dry run without a binary: no estimate)" >"$run/estimate.log"
fi
est=$(grep '^EST ' "$run/estimate.log" | tail -1)
field() { echo "$est" | tr ' ' '\n' | sed -n "s/^$1=//p"; }
n2=$(field N2)
peak_est=$(field peak_setup_est_gb)

# Available memory: the benchmark's own figure (MemAvailable on Linux), the cgroup limit if one
# applies (batch systems: MemAvailable shows the node, not the job), else /proc/meminfo.
avail=$(sed -n 's/^available memory at start: \([0-9.]*\) GB of \([0-9.]*\) GB.*/\1/p' "$run/estimate.log")
total=$(sed -n 's/^available memory at start: \([0-9.]*\) GB of \([0-9.]*\) GB.*/\2/p' "$run/estimate.log")
if [ -z "$avail" ] && [ -r /proc/meminfo ]; then
    avail=$(awk '/^MemAvailable:/ {printf "%.1f", $2 * 1024 / 1e9}' /proc/meminfo)
    [ -n "$avail" ] || avail=$(awk '/^MemFree:/ {printf "%.1f", $2 * 1024 / 1e9}' /proc/meminfo)
    total=$(awk '/^MemTotal:/ {printf "%.1f", $2 * 1024 / 1e9}' /proc/meminfo)
fi
# cgroup limit (cg_limit, computed before system.txt) caps the available memory.
if [ -n "$cg_limit" ] && [ -n "$avail" ]; then
    avail=$(awk -v a="$avail" -v c="$cg_limit" 'BEGIN {printf "%.1f", (c < a) ? c : a}')
fi
[ -n "$margin_gb" ] || margin_gb=$(awk -v t="${total:-0}" 'BEGIN {m = 0.03 * t; printf "%.0f", (m > 16) ? m : 16}')

refused=0
if [ -n "$n2" ] && [ -n "$peak_est" ]; then
    krylov_gb=$(awk -v n="$n2" -v k="$krylov_iters" 'BEGIN {printf "%.1f", (k + 1) * n * 16 / 1e9}')
    need=$(awk -v p="$peak_est" -v k="$krylov_gb" -v m="$margin_gb" 'BEGIN {printf "%.1f", p + k + m}')
    log "estimate: 2N = $n2, setup peak model $peak_est GB (measured peak - Krylov was 0.72-0.86 x the model), Krylov ($krylov_iters it) $krylov_gb GB, margin $margin_gb GB -> need $need GB"
    log "memory: available ${avail:-unknown} GB of ${total:-unknown} GB${cg_limit:+ (cgroup limit $cg_limit GB)}"
    if [ -z "$avail" ]; then
        log "WARNING: available memory unknown, check skipped"
    elif awk -v a="$avail" -v n="$need" 'BEGIN {exit !(n > a)}'; then
        if [ "$force_memory" -eq 1 ]; then
            log "WARNING: need $need GB > available $avail GB; starting anyway (--force-memory)"
        else
            log "REFUSED: need $need GB > available $avail GB (free memory, use a smaller --L, or --force-memory)"
            [ "$dry_run" -eq 1 ] || exit 3
            refused=1
        fi
    else
        log "memory check passed"
    fi
else
    log "WARNING: no EST line, memory check skipped"
fi

cmd="$numa_prefix $bin $run_args"
log "command: $cmd"
if [ "$dry_run" -eq 1 ]; then
    log "dry run: not solving. The run would write stdout.log, stderr.log, gmres_history.tsv,"
    log "memory_samples.tsv, row.txt and summary.txt to $run"
    if [ "$refused" -eq 1 ]; then
        log "dry run: the memory check refused this case (exit status 3)"
        exit 3
    fi
    exit 0
fi

# ------------------------------------------------------------------------------------------
# Solve with memory sampling
load0=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)
t_start=$(date +%s)
log "start: $(date -u +%Y-%m-%dT%H:%M:%SZ), load0 ${load0:-unknown}"
hwm_gb=""
rss_max_gb=""

# Extract the GMRES history, the summary row and the summary text from stdout.log ($1 = exit
# status). Also called on an interrupt, so a stopped run still leaves its history.
extract() {
    t_now=$(date +%s)
    printf 'it\tresidual\tt_s\n' >"$run/gmres_history.tsv"
    awk '/^HIST it=/ {
        for (i = 2; i <= NF; ++i) { split($i, kv, "="); v[kv[1]] = kv[2] }
        printf "%s\t%s\t%s\n", v["it"], v["res"], v["t_s"]
    }' "$run/stdout.log" >>"$run/gmres_history.tsv" 2>/dev/null
    row=$(grep '^ROW ' "$run/stdout.log" 2>/dev/null | tail -1)
    if [ -n "$row" ]; then
        # The ROW line of benchmarks/results/mlfmm_scaling_rows.txt plus annotations (not parsed
        # by the fit): machine=server, threads, OpenMP places, numa, timed matvec rounds, VmHWM,
        # largest sampled VmRSS, load0, wall time.
        echo "$row gmres=cgs2 machine=server threads=$OMP_NUM_THREADS places=$OMP_PLACES numa=$numa_mode matvecs=$matvecs vmhwm_gb=${hwm_gb:-NA} rss_max_gb=${rss_max_gb:-NA} load0=${load0:-NA} wall_s=$((t_now - t_start))" \
            >"$run/row.txt"
        log "row: $run/row.txt"
    else
        log "WARNING: no ROW line (failed or interrupted run); see stderr.log"
    fi
    {
        echo "case: $case_args"
        grep -E '^(RESULT|triangles|formulation|setup|memory|power)' "$run/stdout.log" 2>/dev/null
        last=$(tail -n 1 "$run/gmres_history.tsv")
        case "$last" in it*) ;; *) echo "last GMRES iteration (it, residual, t_s): $last" ;; esac
        echo "peak VmHWM: ${hwm_gb:-NA} GB, largest sampled VmRSS: ${rss_max_gb:-NA} GB; exit status $1; wall $((t_now - t_start)) s"
    } >"$run/summary.txt"
    cat "$run/summary.txt"
    log "pack for sending: tar czf \"$name.tar.gz\" -C \"$out_dir\" \"$name\""
}

# A non-interactive shell starts background jobs with SIGINT ignored, so Ctrl-C would leave the
# solve running: forward INT / TERM / HUP to it explicitly.
pid=""
on_signal() {
    trap - INT TERM HUP
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        rm -f "$run/pid"
    fi
    log "interrupted at $(date -u +%Y-%m-%dT%H:%M:%SZ) after $(($(date +%s) - t_start)) s: solve stopped (no restart files, repeat the run)"
    extract 130
    exit 130
}
trap on_signal INT TERM HUP
# shellcheck disable=SC2086
$numa_prefix "$bin" $run_args >"$run/stdout.log" 2>"$run/stderr.log" &
pid=$!
echo "$pid" >"$run/pid"
printf 't_s\tvmrss_gb\tvmhwm_gb\tmem_available_gb\tload1\n' >"$run/memory_samples.tsv"
tick=0
while kill -0 "$pid" 2>/dev/null; do
    # 1-s ticks (prompt end of the loop and signal handling), one sample every $sample_s s.
    if [ $((tick % sample_s)) -ne 0 ]; then
        tick=$((tick + 1))
        sleep 1
        continue
    fi
    st=/proc/$pid/status
    rss=NA
    hwm=NA
    if [ -r "$st" ]; then
        rss=$(awk '/^VmRSS:/ {printf "%.2f", $2 * 1024 / 1e9}' "$st")
        hwm=$(awk '/^VmHWM:/ {printf "%.2f", $2 * 1024 / 1e9}' "$st")
        [ -n "$hwm" ] && hwm_gb=$hwm
        # Largest sampled VmRSS: the fallback where the kernel reports no VmHWM.
        [ -n "$rss" ] && rss_max_gb=$(awk -v a="$rss" -v b="${rss_max_gb:-0}" 'BEGIN {print (a > b) ? a : b}')
    fi
    ma=$(awk '/^MemAvailable:/ {printf "%.1f", $2 * 1024 / 1e9}' /proc/meminfo 2>/dev/null)
    ld=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(($(date +%s) - t_start))" "${rss:-NA}" "${hwm:-NA}" \
        "${ma:-NA}" "${ld:-NA}" >>"$run/memory_samples.tsv"
    tick=$((tick + 1))
    sleep 1
done
wait "$pid"
status=$?
trap - INT TERM HUP
t_end=$(date +%s)
rm -f "$run/pid"
log "end: $(date -u +%Y-%m-%dT%H:%M:%SZ), exit status $status, wall $((t_end - t_start)) s"
[ "$status" -eq 137 ] && log "exit status 137 = SIGKILL, usually the out-of-memory killer (dmesg)"
[ "$status" -eq 4 ] && log "exit status 4: solved, but the power summary failed (row.txt without R / A, post_error=1; see stderr.log)"
extract "$status"
exit "$status"

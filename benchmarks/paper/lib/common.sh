#!/usr/bin/env bash
set -Eeuo pipefail

PAPER_SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

paper_die() {
  printf 'paper experiment error: %s\n' "$*" >&2
  exit 2
}

paper_require_file() {
  [[ -f "$1" ]] || paper_die "required file is missing: $1"
}

paper_require_dir() {
  [[ -d "$1" ]] || paper_die "required directory is missing: $1"
}

paper_require_complete() {
  local stage="$PAPER_RUN_ROOT/$1" checksum listed actual
  [[ -f "$stage/.complete" ]] || paper_die "required stage is incomplete: $1"
  checksum="$stage/SHA256SUMS"
  paper_require_file "$checksum"
  (cd / && sha256sum -c "$checksum" >/dev/null) || \
    paper_die "required stage checksum verification failed: $1"
  listed=$(cut -d' ' -f3- "$checksum" | LC_ALL=C sort)
  actual=$(find "$stage" \( -type f -o -type l \) ! -name SHA256SUMS ! -name .complete | LC_ALL=C sort)
  [[ "$listed" == "$actual" ]] || paper_die "required stage file inventory drifted: $1"
  if [[ "${PAPER_PHASE:-development}" == formal ]]; then
    grep -Fxq 'phase=formal' "$stage/logs/environment.txt" || \
      paper_die "formal run cannot consume a non-formal stage: $1"
    grep -Fxq 'eligibility=eligible' "$stage/logs/environment.txt" || \
      paper_die "formal run cannot consume an ineligible stage: $1"
  fi
}

paper_verify_script_bundle() {
  local checksum="$PAPER_SCRIPT_ROOT/SCRIPT_SHA256SUMS" listed actual
  paper_require_file "$checksum"
  if find "$PAPER_SCRIPT_ROOT" \( -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) -print -quit | grep -q .; then
    paper_die 'paper script bundle contains executable Python cache files'
  fi
  (cd "$PAPER_SCRIPT_ROOT" && sha256sum -c SCRIPT_SHA256SUMS >/dev/null) || \
    paper_die 'paper script bundle checksum verification failed'
  listed=$(cut -d' ' -f3- "$checksum" | LC_ALL=C sort)
  actual=$(cd "$PAPER_SCRIPT_ROOT" && find . -type f ! -name SCRIPT_SHA256SUMS | LC_ALL=C sort)
  [[ "$listed" == "$actual" ]] || paper_die 'paper script bundle file inventory drifted'
}

paper_require_canonical_library() {
  local configured=$1 expected=$2 label=$3 stage identity recorded_link recorded_resolved recorded_sha
  paper_require_file "$configured"
  paper_require_file "$expected"
  [[ "$(readlink -f "$configured")" == "$(readlink -f "$expected")" ]] || \
    paper_die "$label library is not the validated P0-A build: $configured"
  stage=$(dirname "$(dirname "$expected")")
  identity="$stage/derived/canonical-library.txt"
  paper_require_file "$identity"
  recorded_link=$(sed -n 's/^link_target=//p' "$identity")
  recorded_resolved=$(sed -n 's/^resolved_path=//p' "$identity")
  recorded_sha=$(sed -n 's/^sha256=//p' "$identity")
  [[ "$(readlink "$expected")" == "$recorded_link" ]] || \
    paper_die "$label canonical library symlink target drifted"
  [[ "$(readlink -f "$expected")" == "$recorded_resolved" ]] || \
    paper_die "$label canonical library resolved path drifted"
  paper_require_sha256 "$expected" "$recorded_sha" "$label canonical library"
}

paper_record_canonical_library() {
  local library=$1 output=$2
  paper_require_file "$library"
  [[ -L "$library" ]] || paper_die "canonical library is expected to be a CMake symlink: $library"
  {
    printf 'configured_path=%s\n' "$library"
    printf 'link_target=%s\n' "$(readlink "$library")"
    printf 'resolved_path=%s\n' "$(readlink -f "$library")"
    printf 'sha256=%s\n' "$(sha256sum "$library" | awk '{print $1}')"
  } >"$output"
}

paper_require_sha256() {
  local path=$1 expected=$2 label=$3 actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || paper_die "$label requires a frozen lowercase SHA-256"
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || paper_die "$label SHA-256 mismatch"
}

paper_require_numeric_equal() {
  local label=$1 value=$2 expected=$3
  awk -v value="$value" -v expected="$expected" 'BEGIN { exit !(value == expected) }' || \
    paper_die "formal paper contract fixes $label at $expected"
}

paper_require_integer_minimum() {
  local label=$1 value=$2 minimum=$3
  [[ "$value" =~ ^[0-9]+$ ]] && (( value >= minimum )) || \
    paper_die "formal paper contract requires $label >= $minimum"
}

paper_load_config() {
  local config_before config_after repo_root data_root numactl_command
  [[ -n "${PAPER_CONFIG:-}" ]] || paper_die 'PAPER_CONFIG is not set'
  paper_require_file "$PAPER_CONFIG"
  config_before=$(sha256sum "$PAPER_CONFIG" | awk '{print $1}')
  # shellcheck disable=SC1090
  source "$PAPER_CONFIG"
  config_after=$(sha256sum "$PAPER_CONFIG" | awk '{print $1}')
  [[ "$config_before" == "$config_after" ]] || paper_die 'config changed while it was being loaded'
  export PAPER_LOADED_CONFIG_SHA256=$config_before
  export PYTHONDONTWRITEBYTECODE=1
  : "${PAPER_PHASE:?}" "${PAPER_RUN_ID:?}" "${PAPER_REPO_ROOT:?}" \
    "${PAPER_DATA_ROOT:?}" "${PAPER_PYTHON:?}"
  [[ "$PAPER_CONFIG" = /* && "$PAPER_REPO_ROOT" = /* && "$PAPER_DATA_ROOT" = /* && "$PAPER_PYTHON" = /* ]] || \
    paper_die 'config, repository, data root and Python paths must be absolute'
  repo_root=$(realpath -m -- "$PAPER_REPO_ROOT")
  data_root=$(realpath -m -- "$PAPER_DATA_ROOT")
  [[ "$data_root" != / ]] || paper_die 'PAPER_DATA_ROOT cannot be /'
  case "$data_root/" in
    "$repo_root"/*) paper_die 'PAPER_DATA_ROOT cannot be inside PAPER_REPO_ROOT' ;;
  esac
  [[ "$PAPER_PHASE" == development || "$PAPER_PHASE" == formal ]] || \
    paper_die 'PAPER_PHASE must be development or formal'
  if [[ "$PAPER_PHASE" == formal ]]; then
    [[ "${PAPER_MAX_SCC_ITERATIONS:-}" == 500 ]] || \
      paper_die 'formal paper contract fixes max SCC iterations at 500'
    [[ "${PAPER_REQUIRE_DXTB:-}" == 1 ]] || \
      paper_die 'formal paper matrix requires the pinned dxtb diagnostic rows'
    paper_require_numeric_equal accuracy "${PAPER_ACCURACY:-}" 0.0001
    paper_require_numeric_equal electronic-temperature-K "${PAPER_ELECTRONIC_TEMPERATURE_K:-}" 300
    paper_require_numeric_equal energy-atol-Hartree "${PAPER_ENERGY_ATOL_HARTREE:-}" 5e-7
    paper_require_numeric_equal force-atol-Hartree-per-bohr "${PAPER_FORCE_ATOL_HARTREE_PER_BOHR:-}" 5e-6
    paper_require_numeric_equal charge-atol-e "${PAPER_CHARGE_ATOL_E:-}" 5e-7
    paper_require_numeric_equal CPU-CUDA-atol "${PAPER_CPU_CUDA_ATOL:-}" 1e-6
    paper_require_numeric_equal peer-unchanged-atol "${PAPER_PEER_UNCHANGED_ATOL:-}" 1e-10
    paper_require_numeric_equal finite-difference-atol "${PAPER_FD_ATOL_HARTREE_PER_BOHR:-}" 1e-5
    [[ "${PAPER_SEED:-}" == xtbloom-paper-20260821 ]] || \
      paper_die 'formal paper contract fixes PAPER_SEED'
    [[ "${PAPER_CPU_THREADS:-}" == 1,4,16,32 ]] || paper_die 'formal CPU thread grid drifted'
    [[ "${PAPER_CPU_BATCH_SIZES:-}" == 1,32,128 ]] || paper_die 'formal CPU batch grid drifted'
    [[ "${PAPER_GPU_BATCH_SIZES:-}" == 1,4,16,64,256 ]] || paper_die 'formal GPU batch grid drifted'
    [[ "${PAPER_QM9_GPU_CAPACITY_BATCH_SIZES:-}" == 256,512,1024,2048 ]] || \
      paper_die 'formal QM9 capacity grid drifted'
    [[ "${PAPER_OMOL25_GPU_CAPACITY_BATCH_SIZES:-}" == 256,512,1024,2048,4096 ]] || \
      paper_die 'formal OMol25 capacity grid drifted'
    [[ "${PAPER_AO_BINS:-}" == 1-64,65-128,129-256,257-512,513-1024,1025-inf ]] || \
      paper_die 'formal AO-bin grid drifted'
    paper_require_integer_minimum warmups "${PAPER_WARMUPS:-}" 10
    paper_require_integer_minimum repetitions "${PAPER_REPETITIONS:-}" 30
    paper_require_integer_minimum long-repetitions "${PAPER_LONG_REPETITIONS:-}" 5
    [[ "${PAPER_BOOTSTRAP_SAMPLES:-}" == 10000 ]] || \
      paper_die 'formal bootstrap sample count drifted'
    [[ -n "${PAPER_CPU_SCALING_GOVERNOR:-}" && "$PAPER_CPU_SCALING_GOVERNOR" != CHANGE-ME ]] || \
      paper_die 'formal CPU frequency policy requires an explicit site governor'
  fi
  [[ "$PAPER_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || paper_die 'unsafe PAPER_RUN_ID'
  paper_require_dir "$PAPER_REPO_ROOT"
  paper_require_file "$PAPER_PYTHON"
  numactl_command=${PAPER_NUMACTL:-numactl}
  if [[ "$numactl_command" == */* ]]; then
    [[ "$numactl_command" = /* && -x "$numactl_command" ]] || \
      paper_die 'PAPER_NUMACTL path must be absolute and executable'
  else
    numactl_command=$(command -v -- "$numactl_command") || \
      paper_die "PAPER_NUMACTL command is unavailable: ${PAPER_NUMACTL:-numactl}"
  fi
  export PAPER_NUMACTL_RESOLVED=$numactl_command
  paper_verify_script_bundle
}

paper_require_slurm() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || paper_die 'compute scripts must run inside Slurm'
}

paper_check_source() {
  local actual dirty
  actual=$(git -C "$PAPER_REPO_ROOT" rev-parse HEAD)
  dirty=$(git -C "$PAPER_REPO_ROOT" status --porcelain=v1)
  if [[ "$PAPER_PHASE" == formal ]]; then
    [[ "${PAPER_SUBMISSION_WRAPPER:-0}" == 1 ]] || paper_die 'formal jobs must be submitted through bin/submit.sh'
    [[ "$PAPER_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || \
      paper_die 'formal PAPER_SOURCE_COMMIT must be a full SHA'
    [[ "$actual" == "$PAPER_SOURCE_COMMIT" ]] || \
      paper_die "source commit mismatch: actual=$actual expected=$PAPER_SOURCE_COMMIT"
    [[ -z "$dirty" ]] || paper_die 'formal paper experiments require a clean repo'
    paper_require_sha256 "$PAPER_SCRIPT_ROOT/protocol/paper-experiment-plan.md" \
      52932a8124e74de05979758d3dbd89aa3eb0ecbf5367c2fae6b6311aefb2ad21 'paper experiment plan'
    paper_require_sha256 "$PAPER_SCRIPT_ROOT/protocol/xtbloom-paper-outline.md" \
      721ac2b9c42f5cf48528266a73b2ad6bf06b58279911cd174b018abad841e605 'paper outline'
    export PAPER_EVIDENCE_ELIGIBILITY=eligible
  else
    if [[ -n "$dirty" && "${PAPER_ALLOW_DIRTY:-0}" != 1 ]]; then
      paper_die 'dirty development source requires PAPER_ALLOW_DIRTY=1'
    fi
    export PAPER_EVIDENCE_ELIGIBILITY=development-ineligible
  fi
  if [[ "$PAPER_PHASE" == formal ]]; then
    paper_require_dir "${PAPER_DXTB_SOURCE:-}"
  fi
  if [[ -n "${PAPER_DXTB_SOURCE:-}" ]]; then
    paper_require_dir "$PAPER_DXTB_SOURCE"
    [[ "${PAPER_DXTB_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || paper_die 'configured dxtb source requires a full pinned commit'
    [[ "$(git -C "$PAPER_DXTB_SOURCE" rev-parse HEAD)" == "$PAPER_DXTB_COMMIT" ]] || \
      paper_die 'dxtb source commit does not match the frozen protocol'
    [[ -z "$(git -C "$PAPER_DXTB_SOURCE" status --porcelain=v1)" ]] || \
      paper_die 'configured dxtb source must be clean'
  fi
  if [[ "$PAPER_PHASE" == formal ]]; then
    paper_require_file "${PAPER_XTB_LIBRARY:-}"
    paper_require_file "${PAPER_TBLITE_LIBRARY:-}"
    [[ "$PAPER_XTB_LIBRARY" =~ ${PAPER_XTB_PATH_REGEX:?} ]] || paper_die 'xTB library identity does not match the frozen protocol'
    [[ "$PAPER_TBLITE_LIBRARY" =~ ${PAPER_TBLITE_PATH_REGEX:?} ]] || paper_die 'tblite library identity does not match the frozen protocol'
    paper_require_sha256 "$PAPER_XTB_LIBRARY" "${PAPER_XTB_SHA256:-}" xTB
    paper_require_sha256 "$PAPER_TBLITE_LIBRARY" "${PAPER_TBLITE_SHA256:-}" tblite
    paper_require_sha256 "$PAPER_CPU_LINALG_LIBRARY" "${PAPER_CPU_LINALG_SHA256:-}" CPU-linear-algebra-provider
  fi
}

paper_check_gpu() {
  command -v nvidia-smi >/dev/null || paper_die 'nvidia-smi is unavailable'
  local ordinal=${PAPER_DEVICE_ID:-0} token name uuid bus
  local -a cudart_args=()
  [[ "$ordinal" =~ ^[0-9]+$ ]] || paper_die 'PAPER_DEVICE_ID must be a visible CUDA ordinal'
  paper_require_dir "${PAPER_CUDA_ROOT:-}"
  [[ -z "${PAPER_CUDART_LIBRARY:-}" ]] || cudart_args+=(--cudart-library "$PAPER_CUDART_LIBRARY")
  bus=$("$PAPER_PYTHON" "$PAPER_SCRIPT_ROOT/python/resolve_visible_gpu.py" \
    --cuda-root "$PAPER_CUDA_ROOT" "${cudart_args[@]}" --device-id "$ordinal") || \
    paper_die 'cannot resolve visible CUDA ordinal through libcudart'
  token=$bus
  name=$(nvidia-smi -i "$token" --query-gpu=name --format=csv,noheader | head -n1)
  uuid=$(nvidia-smi -i "$token" --query-gpu=uuid --format=csv,noheader | head -n1)
  bus=$(nvidia-smi -i "$token" --query-gpu=pci.bus_id --format=csv,noheader | head -n1)
  [[ -n "$name" && -n "$uuid" && -n "$bus" ]] || paper_die 'Slurm allocation exposes no resolvable NVIDIA GPU'
  if [[ -n "${PAPER_EXPECT_GPU_REGEX:-}" ]]; then
    grep -Eq "$PAPER_EXPECT_GPU_REGEX" <<<"$name" || \
      paper_die "GPU does not match PAPER_EXPECT_GPU_REGEX: $name"
  fi
  export PAPER_RESOLVED_GPU_TOKEN=$token
  export PAPER_RESOLVED_GPU_UUID=$uuid
  export PAPER_RESOLVED_GPU_NAME=$name
  export PAPER_RESOLVED_GPU_PCI_BUS_ID=$bus
  if [[ -n "${PAPER_OUTPUT_DIR:-}" ]]; then
    printf 'visible_cuda_ordinal=%s\nresolved_gpu_token=%s\nresolved_gpu_uuid=%s\nresolved_gpu_name=%s\nresolved_gpu_pci_bus_id=%s\n' \
      "$ordinal" "$token" "$uuid" "$name" "$bus" >>"$PAPER_OUTPUT_DIR/logs/environment.txt"
  fi
}

paper_begin() {
  local experiment_id=$1
  local plan_section=$2
  local current_script_manifest frozen_script_manifest
  paper_load_config
  paper_require_slurm
  paper_check_source
  export PAPER_EXPERIMENT_ID=$experiment_id
  export PAPER_PLAN_SECTION=$plan_section
  export PAPER_RUN_ROOT="$PAPER_DATA_ROOT/$PAPER_RUN_ID"
  export PAPER_OUTPUT_DIR="$PAPER_RUN_ROOT/$experiment_id"
  if [[ "$PAPER_PHASE" == formal && "$experiment_id" != freeze-manifests ]]; then
    [[ "$PAPER_SELECTION_FILE" == "$PAPER_RUN_ROOT/freeze-manifests/derived/paper-selection.json" ]] || \
      paper_die 'formal jobs must consume the selection sealed inside freeze-manifests'
  fi
  if [[ "$experiment_id" != freeze-manifests ]]; then
    paper_require_complete freeze-manifests
    paper_require_file "$PAPER_RUN_ROOT/freeze-manifests/logs/paper.env.snapshot"
    cmp -s "$PAPER_CONFIG" "$PAPER_RUN_ROOT/freeze-manifests/logs/paper.env.snapshot" || \
      paper_die 'run config differs from the immutable freeze-manifests snapshot'
    paper_require_file "$PAPER_RUN_ROOT/freeze-manifests/logs/script-manifest-sha256.txt"
    current_script_manifest=$(sha256sum "$PAPER_SCRIPT_ROOT/SCRIPT_SHA256SUMS" | awk '{print $1}')
    frozen_script_manifest=$(awk '{print $1}' "$PAPER_RUN_ROOT/freeze-manifests/logs/script-manifest-sha256.txt")
    [[ "$current_script_manifest" == "$frozen_script_manifest" ]] || \
      paper_die 'script checksum manifest differs from the freeze-manifests stage'
  fi
  case "$experiment_id" in
    freeze-manifests) ;;
    p0a-canonical-cpu|p0a-canonical-gpu)
      paper_require_complete freeze-manifests ;;
    p0b-dataset-cpu|p0c-fd-cpu|p0e-degeneracy|si-gfn1-cpu|si-qmmm)
      paper_require_complete p0a-canonical-cpu ;;
    p0b-dataset-gpu|p0c-fd-gpu)
      paper_require_complete p0a-canonical-gpu ;;
    p0d-failure-cpu)
      paper_require_complete p0a-canonical-cpu; paper_require_complete p0b-dataset-cpu ;;
    p0d-failure-gpu)
      paper_require_complete p0a-canonical-gpu; paper_require_complete p0b-dataset-gpu ;;
    p0-gate)
      for required in p0a-canonical-cpu p0a-canonical-gpu p0b-dataset-cpu p0b-dataset-gpu \
        p0c-fd-cpu p0c-fd-gpu p0d-failure-cpu p0d-failure-gpu p0e-degeneracy; do
        paper_require_complete "$required"
      done ;;
    performance-reference)
      paper_require_complete p0-gate ;;
    exp2-gpu-profiler)
      paper_require_complete p0-gate; paper_require_complete performance-reference
      paper_require_complete exp2-gpu-crossover; paper_require_complete exp2-gpu-capacity ;;
    analyze-archive)
      for required in freeze-manifests p0a-canonical-cpu p0a-canonical-gpu \
        p0b-dataset-cpu p0b-dataset-gpu p0c-fd-cpu p0c-fd-gpu \
        p0d-failure-cpu p0d-failure-gpu p0e-degeneracy p0-gate \
        performance-reference exp1-cpu-native si-cpu-process-pool \
        exp2-gpu-crossover exp2-gpu-capacity exp2-gpu-profiler \
        exp3a-convergence exp3b-ragged-cpu exp3b-ragged-gpu \
        si-gfn1-cpu si-qmmm si-cuda-mixed si-energy-only; do
        paper_require_complete "$required"
      done
      [[ "${PAPER_RUN_WARM_SI:-0}" != 1 ]] || paper_require_complete si-warm-trajectory
      [[ "${PAPER_RUN_SECOND_HARDWARE:-0}" != 1 ]] || paper_require_complete si-second-hardware ;;
    *)
      paper_require_complete p0-gate; paper_require_complete performance-reference ;;
  esac
  case "$experiment_id" in
    freeze-manifests|p0a-canonical-cpu|p0a-canonical-gpu|performance-reference) ;;
    p0b-dataset-cpu|p0c-fd-cpu|p0d-failure-cpu|p0e-degeneracy|exp1-cpu-native|si-cpu-process-pool|exp3b-ragged-cpu|si-gfn1-cpu|si-qmmm)
      paper_require_canonical_library "$PAPER_CPU_LIBRARY" "$PAPER_RUN_ROOT/p0a-canonical-cpu/build/libxtbloom.so" CPU ;;
    p0b-dataset-gpu|p0c-fd-gpu|p0d-failure-gpu|exp2-gpu-capacity|exp2-gpu-profiler|exp3b-ragged-gpu|si-cuda-mixed|si-warm-trajectory)
      paper_require_canonical_library "$PAPER_CUDA_LIBRARY" "$PAPER_RUN_ROOT/p0a-canonical-gpu/build/libxtbloom.so" CUDA ;;
    si-second-hardware) ;;
    *)
      paper_require_canonical_library "$PAPER_CPU_LIBRARY" "$PAPER_RUN_ROOT/p0a-canonical-cpu/build/libxtbloom.so" CPU
      paper_require_canonical_library "$PAPER_CUDA_LIBRARY" "$PAPER_RUN_ROOT/p0a-canonical-gpu/build/libxtbloom.so" CUDA ;;
  esac
  # Evidence directories are immutable attempt records.  A failed or cancelled
  # attempt must use a new PAPER_RUN_ID instead of mixing new samples with an
  # incomplete directory from an earlier Slurm job.
  [[ ! -e "$PAPER_OUTPUT_DIR" ]] || \
    paper_die "experiment output already exists; choose a new PAPER_RUN_ID: $PAPER_OUTPUT_DIR"
  mkdir -p "$PAPER_OUTPUT_DIR/logs" "$PAPER_OUTPUT_DIR/raw" "$PAPER_OUTPUT_DIR/derived"
  local cpu_policy_command=("$PAPER_PYTHON" "$PAPER_SCRIPT_ROOT/python/resolve_cpu_policy.py")
  if [[ "$PAPER_PHASE" == formal ]]; then
    cpu_policy_command+=(--expected-governor "$PAPER_CPU_SCALING_GOVERNOR")
  fi
  local cpu_policy=()
  mapfile -t cpu_policy < <("${cpu_policy_command[@]}")
  [[ "${#cpu_policy[@]}" == 6 ]] || paper_die 'CPU policy resolver returned an incomplete record'
  [[ "${cpu_policy[0]}" =~ ^[0-9]+(,[0-9]+)*$ && "${cpu_policy[1]}" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    paper_die 'CPU policy resolver returned unsafe affinity/NUMA values'
  export PAPER_ALLOCATED_CPU_LIST=${cpu_policy[0]}
  export PAPER_ALLOCATED_NUMA_NODES=${cpu_policy[1]}
  {
    printf 'physical_cpu_affinity=%s\n' "${cpu_policy[0]}"
    printf 'numa_interleave_nodes=%s\n' "${cpu_policy[1]}"
    printf 'scaling_governors=%s\n' "${cpu_policy[2]}"
    printf 'scaling_drivers=%s\n' "${cpu_policy[3]}"
    printf 'scaling_min_freq_khz=%s\n' "${cpu_policy[4]}"
    printf 'scaling_max_freq_khz=%s\n' "${cpu_policy[5]}"
  } >"$PAPER_OUTPUT_DIR/logs/cpu-policy.txt"
  local runtime_command=("$PAPER_PYTHON" "$PAPER_SCRIPT_ROOT/python/runtime_identity.py")
  [[ -z "${PAPER_DXTB_SOURCE:-}" ]] || runtime_command+=(--dxtb-source "$PAPER_DXTB_SOURCE")
  if [[ "$PAPER_PHASE" == formal ]]; then
    runtime_command+=(--require numpy --require torch --require dxtb)
  fi
  paper_run_capture "$PAPER_OUTPUT_DIR/logs/python-runtime.json" "${runtime_command[@]}"
  : >"$PAPER_OUTPUT_DIR/logs/toolchain-identity.txt"
  local toolchain_path
  for toolchain_path in "${PAPER_CMAKE:-}" "${PAPER_NINJA:-}" \
    "${PAPER_CUDA_COMPILER:-}"; do
    [[ -z "$toolchain_path" ]] && continue
    paper_require_file "$toolchain_path"
    printf '%s\t%s\t%s\n' "$toolchain_path" "$(readlink -f "$toolchain_path")" \
      "$(sha256sum "$toolchain_path" | awk '{print $1}')" \
      >>"$PAPER_OUTPUT_DIR/logs/toolchain-identity.txt"
  done
  if [[ "$experiment_id" != freeze-manifests ]]; then
    cmp -s "$PAPER_OUTPUT_DIR/logs/python-runtime.json" \
      "$PAPER_RUN_ROOT/freeze-manifests/logs/python-runtime.json" || \
      paper_die 'Python executable, packages, or module origins drifted after manifest freeze'
    cmp -s "$PAPER_OUTPUT_DIR/logs/toolchain-identity.txt" \
      "$PAPER_RUN_ROOT/freeze-manifests/logs/toolchain-identity.txt" || \
      paper_die 'configured compiler/build-tool binary drifted after manifest freeze'
  fi
  {
    printf 'experiment_id=%s\n' "$PAPER_EXPERIMENT_ID"
    printf 'plan_section=%s\n' "$PAPER_PLAN_SECTION"
    printf 'phase=%s\n' "$PAPER_PHASE"
    printf 'eligibility=%s\n' "$PAPER_EVIDENCE_ELIGIBILITY"
    printf 'job_id=%s\n' "$SLURM_JOB_ID"
    printf 'hostname=%s\n' "$(hostname)"
    date --iso-8601=seconds
    git -C "$PAPER_REPO_ROOT" rev-parse HEAD
    git -C "$PAPER_REPO_ROOT" status --short --branch
    scontrol show job "$SLURM_JOB_ID" || true
    type -a sbatch || true
    type -a srun || true
    taskset -pc $$ || true
    "$PAPER_NUMACTL_RESOLVED" --hardware || true
    lscpu || true
    free -h || true
    grep . /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor || true
    "$PAPER_PYTHON" --version
    [[ -z "${PAPER_CMAKE:-}" ]] || "$PAPER_CMAKE" --version
    [[ -z "${PAPER_NINJA:-}" ]] || "$PAPER_NINJA" --version
    [[ -z "${PAPER_CUDA_COMPILER:-}" ]] || "$PAPER_CUDA_COMPILER" --version
    env | LC_ALL=C sort | grep -E '^(OMP|OPENBLAS|MKL|CUDA|XTB|XTBLOOM|SLURM|PAPER_)' || true
    if command -v nvidia-smi >/dev/null; then nvidia-smi -q; fi
    if command -v nvcc >/dev/null; then nvcc --version; fi
    local binary
    for binary in "${PAPER_CPU_LIBRARY:-}" "${PAPER_CUDA_LIBRARY:-}" \
      "${PAPER_XTB_LIBRARY:-}" "${PAPER_TBLITE_LIBRARY:-}" \
      "${PAPER_CPU_LINALG_LIBRARY:-}"; do
      [[ ! -f "$binary" ]] || { printf 'ldd %s\n' "$binary"; ldd "$binary" || true; }
    done
  } >"$PAPER_OUTPUT_DIR/logs/environment.txt" 2>&1
  cp -- "$PAPER_CONFIG" "$PAPER_OUTPUT_DIR/logs/paper.env.snapshot"
  [[ "$(sha256sum "$PAPER_OUTPUT_DIR/logs/paper.env.snapshot" | awk '{print $1}')" == "$PAPER_LOADED_CONFIG_SHA256" ]] || \
    paper_die 'config changed after it was loaded'
  sha256sum "$PAPER_SCRIPT_ROOT/SCRIPT_SHA256SUMS" \
    >"$PAPER_OUTPUT_DIR/logs/script-manifest-sha256.txt"
  printf '%q ' "$0" "$@" >"$PAPER_OUTPUT_DIR/logs/argv.txt"
  printf '\n' >>"$PAPER_OUTPUT_DIR/logs/argv.txt"
  paper_hash_inputs "$PAPER_CONFIG" \
    "$PAPER_SCRIPT_ROOT/protocol/paper-experiment-plan.md" \
    "$PAPER_SCRIPT_ROOT/protocol/xtbloom-paper-outline.md" \
    "${PAPER_QM9_MAIN_MANIFEST:-}" "${PAPER_QM9_PERFORMANCE_MANIFEST:-}" \
    "${PAPER_OMOL25_MANIFEST:-}" "${PAPER_CPU_LIBRARY:-}" \
    "${PAPER_CUDA_LIBRARY:-}" "${PAPER_XTB_LIBRARY:-}" \
    "${PAPER_TBLITE_LIBRARY:-}" "${PAPER_CPU_LINALG_LIBRARY:-}" \
    "${PAPER_TRAJECTORY_MANIFEST:-}"
  find "$PAPER_SCRIPT_ROOT" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
    >"$PAPER_OUTPUT_DIR/logs/script-sha256.txt"
  if [[ "$experiment_id" != freeze-manifests ]]; then
    paper_require_file "${PAPER_SELECTION_FILE:-}"
    paper_run_python "$PAPER_SCRIPT_ROOT/python/verify_frozen_selection.py" \
      --selection "$PAPER_SELECTION_FILE" \
      --qm9-main-manifest "$PAPER_QM9_MAIN_MANIFEST" \
      --qm9-performance-manifest "$PAPER_QM9_PERFORMANCE_MANIFEST" \
      --omol25-manifest "$PAPER_OMOL25_MANIFEST" \
      >"$PAPER_OUTPUT_DIR/logs/frozen-input-verification.txt"
  fi
}

paper_hash_inputs() {
  local output="$PAPER_OUTPUT_DIR/logs/input-sha256.txt"
  touch "$output"
  local path
  for path in "$@"; do
    if [[ -f "$path" ]]; then sha256sum "$path" >>"$output"; fi
  done
  LC_ALL=C sort -u -o "$output" "$output"
}

paper_finish() {
  local checksum="$PAPER_OUTPUT_DIR/SHA256SUMS"
  find "$PAPER_OUTPUT_DIR" \( -type f -o -type l \) ! -name SHA256SUMS ! -name .complete -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >"$checksum"
  (cd / && sha256sum -c "$checksum")
  date --iso-8601=seconds >"$PAPER_OUTPUT_DIR/.complete"
}

paper_run() {
  printf 'RUN:' | tee -a "$PAPER_OUTPUT_DIR/logs/commands.txt"
  printf ' %q' "$@" | tee -a "$PAPER_OUTPUT_DIR/logs/commands.txt"
  printf '\n' | tee -a "$PAPER_OUTPUT_DIR/logs/commands.txt"
  "$@"
}

paper_run_capture() {
  local output=$1
  shift
  printf 'RUN:' | tee -a "$PAPER_OUTPUT_DIR/logs/commands.txt"
  printf ' %q' "$@" | tee -a "$PAPER_OUTPUT_DIR/logs/commands.txt"
  printf ' > %q\n' "$output" | tee -a "$PAPER_OUTPUT_DIR/logs/commands.txt"
  "$@" >"$output"
}

paper_run_python() {
  [[ -n "${PAPER_NUMACTL_RESOLVED:-}" ]] || paper_die 'paper_load_config did not resolve numactl'
  paper_run "$PAPER_NUMACTL_RESOLVED" --physcpubind="$PAPER_ALLOCATED_CPU_LIST" \
    --interleave="$PAPER_ALLOCATED_NUMA_NODES" "$PAPER_PYTHON" "$@"
}

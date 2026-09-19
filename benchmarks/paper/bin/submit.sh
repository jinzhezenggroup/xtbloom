#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=
ALL=0
LIST=0
DEPENDENCY=
declare -a REQUESTED=()

usage() {
  cat <<'EOF'
Usage: submit.sh --config /absolute/paper.env [--list | --all | EXPERIMENT...]

--all submits the stage-gated paper plan in dependency order. Individual jobs
may also be submitted by ID. GPU partition/GRES/constraint come from config.
EOF
}

while (($#)); do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --all) ALL=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) REQUESTED+=("$1"); shift ;;
  esac
done

declare -A SCRIPT CLASS
register() { SCRIPT[$1]=$2; CLASS[$1]=$3; }
register freeze-manifests 00_freeze_manifests.sbatch cpu
register p0a-canonical-cpu 01_p0a_canonical_cpu.sbatch cpu
register p0a-canonical-gpu 02_p0a_canonical_gpu.sbatch gpu
register p0b-dataset-cpu 03_p0b_dataset_cpu.sbatch cpu
register p0b-dataset-gpu 04_p0b_dataset_gpu.sbatch gpu
register p0c-fd-cpu 05_p0c_finite_difference_cpu.sbatch cpu
register p0c-fd-gpu 06_p0c_finite_difference_gpu.sbatch gpu
register p0d-failure-cpu 07_p0d_failure_cpu.sbatch cpu
register p0d-failure-gpu 08_p0d_failure_gpu.sbatch gpu
register p0e-degeneracy 09_p0e_degeneracy.sbatch cpu
register p0-gate 09b_p0_gate.sbatch cpu
register performance-reference 12_performance_references.sbatch cpu
register exp1-cpu-native 10_exp1_cpu_native_batch.sbatch cpu
register si-cpu-process-pool 11_si_cpu_process_pool.sbatch cpu
register exp2-gpu-crossover 20_exp2_gpu_crossover.sbatch gpu
register exp2-gpu-capacity 21_exp2_gpu_capacity.sbatch gpu
register exp2-gpu-profiler 22_exp2_gpu_profiler.sbatch gpu
register exp3a-convergence 30_exp3a_convergence.sbatch cpu
register exp3b-ragged-cpu 31_exp3b_ragged_cpu.sbatch cpu
register exp3b-ragged-gpu 32_exp3b_ragged_gpu.sbatch gpu
register si-gfn1-cpu 40_si_gfn1_cpu.sbatch cpu
register si-qmmm 41_si_qmmm.sbatch cpu
register si-cuda-mixed 42_si_cuda_mixed.sbatch gpu
register si-energy-only 43_si_energy_only.sbatch gpu
register si-warm-trajectory 44_si_warm_trajectory.sbatch gpu
register si-second-hardware 45_si_second_hardware.sbatch gpu
register analyze-archive 90_analyze_archive.sbatch cpu

ORDER=(freeze-manifests p0a-canonical-cpu p0a-canonical-gpu p0b-dataset-cpu
  p0b-dataset-gpu p0c-fd-cpu p0c-fd-gpu p0d-failure-cpu p0d-failure-gpu
  p0e-degeneracy p0-gate performance-reference exp2-gpu-crossover exp2-gpu-capacity exp2-gpu-profiler
  exp1-cpu-native si-cpu-process-pool exp3a-convergence exp3b-ragged-cpu
  exp3b-ragged-gpu si-gfn1-cpu si-qmmm si-cuda-mixed si-energy-only
  si-warm-trajectory si-second-hardware analyze-archive)

if ((LIST)); then
  for id in "${ORDER[@]}"; do printf '%-28s %s\n' "$id" "${CLASS[$id]}"; done
  exit 0
fi
[[ -n "$CONFIG" && -f "$CONFIG" ]] || { usage >&2; exit 2; }
[[ "$CONFIG" = /* ]] || { echo 'config path must be absolute' >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONFIG"
: "${PAPER_RUN_ID:?}" "${PAPER_REPO_ROOT:?}" "${PAPER_DATA_ROOT:?}"
[[ "$PAPER_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'unsafe PAPER_RUN_ID' >&2; exit 2; }
[[ "$PAPER_REPO_ROOT" = /* && "$PAPER_DATA_ROOT" = /* ]] || {
  echo 'repository and data roots must be absolute' >&2
  exit 2
}
repo_root=$(realpath -m -- "$PAPER_REPO_ROOT")
data_root=$(realpath -m -- "$PAPER_DATA_ROOT")
[[ "$data_root" != / ]] || { echo 'PAPER_DATA_ROOT cannot be /' >&2; exit 2; }
case "$data_root/" in
  "$repo_root"/*) echo 'PAPER_DATA_ROOT cannot be inside PAPER_REPO_ROOT' >&2; exit 2 ;;
esac

sbatch_command=${PAPER_SBATCH:-sbatch}
if [[ "$sbatch_command" == */* ]]; then
  [[ "$sbatch_command" = /* && -x "$sbatch_command" ]] || {
    echo 'PAPER_SBATCH path must be absolute and executable' >&2
    exit 2
  }
else
  sbatch_command=$(command -v -- "$sbatch_command") || {
    echo "PAPER_SBATCH command is unavailable: ${PAPER_SBATCH:-sbatch}" >&2
    exit 2
  }
fi

# Freeze the submitted bytes once per RUN_ID. Queued afterok jobs must never
# re-source a later edit of the caller's config and silently mix protocols.
SUBMISSION_DIR="$PAPER_DATA_ROOT/$PAPER_RUN_ID/submission"
mkdir -p "$SUBMISSION_DIR"
FROZEN_CONFIG="$SUBMISSION_DIR/paper.env"
if [[ ! -e "$FROZEN_CONFIG" ]]; then
  temporary_config="$SUBMISSION_DIR/.paper.env.$$"
  cp -- "$CONFIG" "$temporary_config"
  chmod 0444 "$temporary_config"
  if ! ln "$temporary_config" "$FROZEN_CONFIG" 2>/dev/null; then
    cmp -s "$temporary_config" "$FROZEN_CONFIG" || {
      rm -f -- "$temporary_config"
      echo 'RUN_ID already has a different frozen config; choose a new RUN_ID' >&2
      exit 2
    }
  fi
  rm -f -- "$temporary_config"
fi
cmp -s "$CONFIG" "$FROZEN_CONFIG" || {
  echo 'submitted config differs from the frozen bytes for this RUN_ID; choose a new RUN_ID' >&2
  exit 2
}
CONFIG=$FROZEN_CONFIG
# Re-source the path exported to jobs, even though cmp proved byte equality.
# shellcheck disable=SC1090
source "$CONFIG"

if ((ALL)); then
  REQUESTED=()
  for id in "${ORDER[@]}"; do
    [[ "$id" != si-warm-trajectory || "${PAPER_RUN_WARM_SI:-0}" == 1 ]] || continue
    [[ "$id" != si-second-hardware || "${PAPER_RUN_SECOND_HARDWARE:-0}" == 1 ]] || continue
    REQUESTED+=("$id")
  done
fi
((${#REQUESTED[@]})) || { usage >&2; exit 2; }

LEDGER="$SUBMISSION_DIR/submission-ledger.tsv"
mkdir -p "$(dirname "$LEDGER")"
SLURM_LOG_DIR="$PAPER_DATA_ROOT/$PAPER_RUN_ID/slurm"
mkdir -p "$SLURM_LOG_DIR"
if [[ ! -e "$LEDGER" ]]; then
  printf 'timestamp\texperiment\tclass\tdependency\tcommand\tsbatch_output\n' >"$LEDGER"
fi

for id in "${REQUESTED[@]}"; do
  [[ -n "${SCRIPT[$id]:-}" ]] || { echo "unknown experiment: $id" >&2; exit 2; }
  args=("$sbatch_command" --export="ALL,PAPER_CONFIG=$CONFIG,PAPER_SUBMISSION_WRAPPER=1")
  if [[ "${CLASS[$id]}" == gpu ]]; then
    partition=${PAPER_GPU_PARTITION:-}; gres=${PAPER_GPU_GRES:-}; constraint=${PAPER_GPU_CONSTRAINT:-}
    if [[ "$id" == si-second-hardware ]]; then
      partition=${PAPER_SECOND_GPU_PARTITION:-}; gres=${PAPER_SECOND_GPU_GRES:-}
      constraint=${PAPER_SECOND_GPU_CONSTRAINT:-}
    fi
    [[ -z "$partition" ]] || args+=(--partition="$partition")
    [[ -z "$gres" ]] || args+=(--gres="$gres")
    [[ -z "$constraint" ]] || args+=(--constraint="$constraint")
  else
    [[ -z "${PAPER_CPU_PARTITION:-}" ]] || args+=(--partition="$PAPER_CPU_PARTITION")
  fi
  [[ -z "${PAPER_CPU_HINT:-}" ]] || args+=(--hint="$PAPER_CPU_HINT")
  args+=(--output="$SLURM_LOG_DIR/%x-%j.out" --error="$SLURM_LOG_DIR/%x-%j.err")
  [[ -z "$DEPENDENCY" ]] || args+=(--dependency="afterok:$DEPENDENCY")
  args+=("$ROOT/slurm/${SCRIPT[$id]}")
  printf -v command '%q ' "${args[@]}"
  output=$("${args[@]}")
  job_id=$(awk '{print $NF}' <<<"$output")
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "cannot parse sbatch output: $output" >&2; exit 2; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$id" \
    "${CLASS[$id]}" "${DEPENDENCY:-none}" "$command" "$output" >>"$LEDGER"
  printf '%s -> %s\n' "$id" "$job_id"
  if ((ALL)); then DEPENDENCY=$job_id; fi
done

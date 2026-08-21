#!/usr/bin/env bash
# Experiment 2 profiler: operational candidates for the three planned regimes.
# Raw profiler captures remain outside the repo; only derived reports are portable.
#SBATCH --job-name=paper-profiler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin exp2-gpu-profiler 'Experiment 2 three profiler candidates; causal labels require trace review'
paper_check_gpu
command -v nsys >/dev/null || paper_die 'nsys unavailable'
RAW="$PAPER_OUTPUT_DIR/raw-profiler"; mkdir -p "$RAW"
profile_point() {
  local name=$1 dataset=$2 ao=$3 batch=$4
  local report="$RAW/$name"
  local manifest=$PAPER_OMOL25_MANIFEST
  local -a cudart_args=()
  [[ "$dataset" != qm9 ]] || manifest=$PAPER_QM9_PERFORMANCE_MANIFEST
  [[ -z "${PAPER_CUDART_LIBRARY:-}" ]] || cudart_args+=(--cudart-library "$PAPER_CUDART_LIBRARY")
  paper_run nsys profile --force-overwrite=false --trace=cuda,nvtx,osrt --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$report" "$PAPER_NUMACTL_RESOLVED" --physcpubind="$PAPER_ALLOCATED_CPU_LIST" \
    --interleave="$PAPER_ALLOCATED_NUMA_NODES" "$PAPER_PYTHON" "$ROOT/python/profile_cell.py" \
    --repo "$PAPER_REPO_ROOT" --manifest "$manifest" --dataset "$dataset" \
    --library "$PAPER_CUDA_LIBRARY" --cuda-root "$PAPER_CUDA_ROOT" "${cudart_args[@]}" --ao-bin "$ao" \
    --batch-size "$batch" --device-id "$PAPER_DEVICE_ID" --warmups "$PAPER_WARMUPS" \
    --repetitions "$PAPER_LONG_REPETITIONS" --seed "$PAPER_SEED" \
    --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
    --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" --output "$PAPER_OUTPUT_DIR/derived/$name-timing.json"
  paper_run nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    --output "$PAPER_OUTPUT_DIR/derived/$name" "$report.nsys-rep"
}
paper_run_python "$ROOT/python/select_profiler_points.py" \
  --input-root "$PAPER_DATA_ROOT/$PAPER_RUN_ID/exp2-gpu-crossover/raw" \
  --input-root "$PAPER_DATA_ROOT/$PAPER_RUN_ID/exp2-gpu-capacity/raw" \
  --json-output "$PAPER_OUTPUT_DIR/derived/profiler-points.json" \
  --tsv-output "$PAPER_OUTPUT_DIR/derived/profiler-points.tsv"
while IFS=$'\t' read -r name dataset ao batch; do
  profile_point "$name" "$dataset" "$ao" "$batch"
done <"$PAPER_OUTPUT_DIR/derived/profiler-points.tsv"
nsys --version >"$PAPER_OUTPUT_DIR/logs/nsys-version.txt"
if command -v ncu >/dev/null; then
  ncu --version >"$PAPER_OUTPUT_DIR/logs/ncu-version.txt"
  printf 'ncu is available; counter collection requires a reviewed kernel filter and is intentionally not guessed.\n' >"$PAPER_OUTPUT_DIR/derived/ncu-status.txt"
else
  printf 'ncu unavailable\n' >"$PAPER_OUTPUT_DIR/derived/ncu-status.txt"
fi
printf '%s\n' 'Candidate names are operational selections, not bottleneck claims. Review the derived Nsight tables before assigning launch-bound, saturated, or eigensolver-bound labels.' \
  >"$PAPER_OUTPUT_DIR/derived/interpretation-required.txt"
paper_finish

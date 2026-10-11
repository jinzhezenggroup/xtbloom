#!/usr/bin/env bash
# P0-B/3-A CUDA raw data: all xTBloom descriptor modes; optional dxtb device row.
#SBATCH --job-name=paper-p0b-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/lib/common.sh"
paper_begin p0b-dataset-gpu 'P0-B dataset equivalence and Experiment 3-A CUDA convergence'
paper_check_gpu
paper_require_file "$PAPER_CUDA_LIBRARY"
DXTB_ARGS=(); [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]] && DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE")
run_one() {
  local dataset=$1 manifest=$2 subsets=$3 memory=$4 engines=$5 suffix=$6
  paper_run_python "$PAPER_REPO_ROOT/benchmarks/dataset_runner.py" \
    --manifest "$manifest" --output-dir "$PAPER_OUTPUT_DIR/raw/$dataset-$subsets-$suffix" \
    --dataset "$dataset" --subsets "$subsets" --engines "$engines" --batch-size 32 \
    --library "$PAPER_CUDA_LIBRARY" "${DXTB_ARGS[@]}" \
    --backend cuda --memory-mode "$memory" --device-id "$PAPER_DEVICE_ID" --cpu-threads 1 \
    --accuracy "$PAPER_ACCURACY" --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K"
}
for memory in host device mixed; do
  run_one qm9 "$PAPER_QM9_MAIN_MANIFEST" main "$memory" xtbloom "xtbloom-$memory"
  run_one omol25 "$PAPER_OMOL25_MANIFEST" main,stress "$memory" xtbloom "xtbloom-$memory"
done
if [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]]; then
  run_one qm9 "$PAPER_QM9_MAIN_MANIFEST" main device dxtb dxtb-device
  run_one omol25 "$PAPER_OMOL25_MANIFEST" main,stress device dxtb dxtb-device
else
  printf 'dxtb unavailable: no source/runtime\n' >"$PAPER_OUTPUT_DIR/derived/dxtb-unavailable.txt"
fi
paper_finish

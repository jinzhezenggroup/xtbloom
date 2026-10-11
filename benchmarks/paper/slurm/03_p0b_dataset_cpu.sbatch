#!/usr/bin/env bash
# P0-B and Experiment 3-A raw CPU data: QM9/OMol25 xTBloom/xTB/tblite/dxtb.
#SBATCH --job-name=paper-p0b-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/lib/common.sh"
paper_begin p0b-dataset-cpu 'P0-B dataset equivalence and Experiment 3-A CPU convergence'
: "${PAPER_QM9_MAIN_MANIFEST:?}" "${PAPER_OMOL25_MANIFEST:?}" "${PAPER_CPU_LIBRARY:?}"
paper_require_file "$PAPER_CPU_LIBRARY"
ENGINES=xtbloom,xtb,tblite
DXTB_ARGS=()
if [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]]; then
  ENGINES+=,dxtb
  DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE")
else
  printf 'dxtb unavailable: no source/runtime\n' >"$PAPER_OUTPUT_DIR/derived/dxtb-unavailable.txt"
fi
run_dataset() {
  local dataset=$1 manifest=$2 subsets=$3
  local out="$PAPER_OUTPUT_DIR/raw/$dataset-$subsets"
  paper_run_python "$PAPER_REPO_ROOT/benchmarks/dataset_runner.py" \
    --manifest "$manifest" --output-dir "$out" --dataset "$dataset" --subsets "$subsets" \
    --engines "$ENGINES" --batch-size 32 --library "$PAPER_CPU_LIBRARY" \
    --xtb-library "$PAPER_XTB_LIBRARY" --tblite-library "$PAPER_TBLITE_LIBRARY" \
    "${DXTB_ARGS[@]}" \
    --backend cpu --memory-mode host --cpu-threads 1 \
    --accuracy "$PAPER_ACCURACY" --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K"
}
run_dataset qm9 "$PAPER_QM9_MAIN_MANIFEST" main
run_dataset omol25 "$PAPER_OMOL25_MANIFEST" main,stress
paper_finish

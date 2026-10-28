#!/usr/bin/env bash
# Performance qualification prerequisite: xTB/tblite on the disjoint frozen
# QM9/OMol25 performance sets.  These are correctness references, not timings.
#SBATCH --job-name=paper-perf-ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin performance-reference 'Performance-set xTB/tblite correctness references'
ENGINES=xtb,tblite
DXTB_ARGS=()
if [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]]; then
  ENGINES+=,dxtb
  DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE")
elif [[ "${PAPER_REQUIRE_DXTB:-0}" == 1 ]]; then
  paper_die 'formal performance reference matrix requires dxtb'
fi
for spec in "qm9:$PAPER_QM9_PERFORMANCE_MANIFEST" "omol25:$PAPER_OMOL25_MANIFEST"; do
  dataset=${spec%%:*}; manifest=${spec#*:}
  paper_run_python "$PAPER_REPO_ROOT/benchmarks/dataset_runner.py" \
    --manifest "$manifest" --output-dir "$PAPER_OUTPUT_DIR/raw/$dataset" --dataset "$dataset" \
    --subsets performance --engines "$ENGINES" --batch-size 32 "${DXTB_ARGS[@]}" \
    --xtb-library "$PAPER_XTB_LIBRARY" --tblite-library "$PAPER_TBLITE_LIBRARY" \
    --backend cpu --memory-mode host --cpu-threads 1 --accuracy "$PAPER_ACCURACY" \
    --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K"
done
paper_finish

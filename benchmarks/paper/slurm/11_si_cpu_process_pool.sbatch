#!/usr/bin/env bash
# SI Experiment 1 sensitivity: persistent N-process x 1-thread xTB/tblite pools.
#SBATCH --job-name=paper-si-pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin si-cpu-process-pool 'SI persistent external process-pool sensitivity'
REFERENCE="$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference"
for spec in "qm9:$PAPER_QM9_PERFORMANCE_MANIFEST" "omol25:$PAPER_OMOL25_MANIFEST"; do
  dataset=${spec%%:*}; manifest=${spec#*:}
  for ao in 1-64 65-128; do
    for engine in xtb tblite; do
      library=$PAPER_XTB_LIBRARY; [[ "$engine" == tblite ]] && library=$PAPER_TBLITE_LIBRARY
      paper_run_python "$ROOT/python/process_pool_matrix.py" --repo "$PAPER_REPO_ROOT" \
        --manifest "$manifest" --dataset "$dataset" --subset performance --engine "$engine" \
        --library "$library" --reference-root "$REFERENCE" --processes "$PAPER_CPU_THREADS" \
        --ao-bin "$ao" --batch-size 128 \
        --warmups "$PAPER_WARMUPS" --repetitions "$PAPER_REPETITIONS" \
        --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" --seed "$PAPER_SEED" \
        --energy-atol "$PAPER_ENERGY_ATOL_HARTREE" --force-atol "$PAPER_FORCE_ATOL_HARTREE_PER_BOHR" \
        --charge-atol "$PAPER_CHARGE_ATOL_E" --accuracy "$PAPER_ACCURACY" \
        --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
        --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
        --output "$PAPER_OUTPUT_DIR/raw/$dataset-$ao-$engine.json"
    done
    paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
      --manifest "$manifest" --dataset "$dataset" --subset performance \
      --cpu-library "$PAPER_CPU_LIBRARY" --suite pool-baseline --ao-bins "$ao" \
      --batch-sizes 128 --cpu-threads "$PAPER_CPU_THREADS" --property force \
      --warmups "$PAPER_WARMUPS" --repetitions "$PAPER_REPETITIONS" \
      --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" --seed "$PAPER_SEED" \
      --reference-root "$REFERENCE" --require-correctness --charge-atol "$PAPER_CHARGE_ATOL_E" \
      --accuracy "$PAPER_ACCURACY" --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
      --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
      --output "$PAPER_OUTPUT_DIR/raw/$dataset-$ao-xtbloom.json"
  done
done
paper_finish

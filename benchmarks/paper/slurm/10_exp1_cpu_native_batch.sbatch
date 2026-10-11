#!/usr/bin/env bash
# Experiment 1 / Figure 2: native batch vs per-system public APIs under equal budgets.
#SBATCH --job-name=paper-exp1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin exp1-cpu-native 'Experiment 1, Figure 2 CPU native-interface matrix'
REFERENCE="$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference"
for spec in "qm9:$PAPER_QM9_PERFORMANCE_MANIFEST" "omol25:$PAPER_OMOL25_MANIFEST"; do
  dataset=${spec%%:*}; manifest=${spec#*:}
  paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
    --manifest "$manifest" --dataset "$dataset" --subset performance --cpu-library "$PAPER_CPU_LIBRARY" \
    --xtb-library "$PAPER_XTB_LIBRARY" --tblite-library "$PAPER_TBLITE_LIBRARY" \
    --suite cpu-native --ao-bins "$PAPER_AO_BINS" --batch-sizes "$PAPER_CPU_BATCH_SIZES" \
    --cpu-threads "$PAPER_CPU_THREADS" --property force --warmups "$PAPER_WARMUPS" \
    --repetitions "$PAPER_REPETITIONS" --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" \
    --seed "$PAPER_SEED" --reference-root "$REFERENCE" \
    --require-correctness --energy-atol "$PAPER_ENERGY_ATOL_HARTREE" \
    --force-atol "$PAPER_FORCE_ATOL_HARTREE_PER_BOHR" --charge-atol "$PAPER_CHARGE_ATOL_E" \
    --accuracy "$PAPER_ACCURACY" --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
    --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    --output "$PAPER_OUTPUT_DIR/raw/$dataset.json"
done
paper_finish

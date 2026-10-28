#!/usr/bin/env bash
# Experiment 2 / Figure 3: AO x batch CPU/CUDA host/device crossover matrix.
#SBATCH --job-name=paper-exp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); source "$ROOT/lib/common.sh"
paper_begin exp2-gpu-crossover 'Experiment 2, Figure 3 CPU-CUDA crossover'
paper_check_gpu
REFERENCE="$PAPER_DATA_ROOT/$PAPER_RUN_ID/performance-reference"
for spec in "qm9:$PAPER_QM9_PERFORMANCE_MANIFEST" "omol25:$PAPER_OMOL25_MANIFEST"; do
  dataset=${spec%%:*}; manifest=${spec#*:}
  DXTB_ARGS=(); [[ -n "${PAPER_DXTB_SOURCE:-}" && -d "$PAPER_DXTB_SOURCE" ]] && DXTB_ARGS=(--dxtb-source "$PAPER_DXTB_SOURCE")
  DXTB_GATE=(); [[ "${PAPER_REQUIRE_DXTB:-0}" == 1 ]] && DXTB_GATE=(--require-dxtb)
  paper_run_python "$ROOT/python/performance_matrix.py" --repo "$PAPER_REPO_ROOT" \
    --manifest "$manifest" --dataset "$dataset" --subset performance \
    --cpu-library "$PAPER_CPU_LIBRARY" --cuda-library "$PAPER_CUDA_LIBRARY" \
    "${DXTB_ARGS[@]}" --suite gpu-crossover --ao-bins "$PAPER_AO_BINS" \
    --batch-sizes "$PAPER_GPU_BATCH_SIZES" --cpu-threads 32 --device-id "$PAPER_DEVICE_ID" \
    --property force --warmups "$PAPER_WARMUPS" --repetitions "$PAPER_REPETITIONS" \
    --bootstrap-samples "$PAPER_BOOTSTRAP_SAMPLES" \
    --seed "$PAPER_SEED" --reference-root "$REFERENCE" --require-correctness \
    --energy-atol "$PAPER_ENERGY_ATOL_HARTREE" --force-atol "$PAPER_FORCE_ATOL_HARTREE_PER_BOHR" \
    --charge-atol "$PAPER_CHARGE_ATOL_E" --accuracy "$PAPER_ACCURACY" \
    --electronic-temperature-kelvin "$PAPER_ELECTRONIC_TEMPERATURE_K" \
    --max-scc-iterations "$PAPER_MAX_SCC_ITERATIONS" \
    "${DXTB_GATE[@]}" \
    --output "$PAPER_OUTPUT_DIR/raw/$dataset.json"
done
paper_finish

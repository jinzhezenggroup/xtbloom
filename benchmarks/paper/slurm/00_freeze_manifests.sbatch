#!/usr/bin/env bash
# Paper prerequisite: Sections 4 and 11. Freeze dataset roles and P0-C coordinates.
#SBATCH --job-name=paper-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/lib/common.sh"
paper_begin freeze-manifests 'Sections 4 and 11.1-2'
: "${PAPER_QM9_MAIN_MANIFEST:?}" "${PAPER_QM9_PERFORMANCE_MANIFEST:?}" "${PAPER_OMOL25_MANIFEST:?}" "${PAPER_SEED:?}"
paper_hash_inputs "$PAPER_QM9_MAIN_MANIFEST" "$PAPER_QM9_PERFORMANCE_MANIFEST" "$PAPER_OMOL25_MANIFEST"
paper_run_python "$ROOT/python/manifest_freeze.py" \
  --repo "$PAPER_REPO_ROOT" \
  --qm9-main-manifest "$PAPER_QM9_MAIN_MANIFEST" \
  --qm9-performance-manifest "$PAPER_QM9_PERFORMANCE_MANIFEST" \
  --omol25-manifest "$PAPER_OMOL25_MANIFEST" \
  --seed "$PAPER_SEED" --output "$PAPER_OUTPUT_DIR/derived/paper-selection.json"
paper_finish

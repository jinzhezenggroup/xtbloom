# Restricted one-body component evidence

This bundle is **diagnostic component evidence**, not a scientific endpoint,
public E2E performance qualification, accepted mode/threshold contract, or
empirical no-go decision. `claim_eligible` is false. The automatic 40-AO guard
is unchanged. Missing declining-activity, mixed-bucket, force/oracle, public
endpoint and sanitizer qualification remains owned by issue #515.

## Producing and analyzing identities

- Producing source: clean `d76e4b8f2a5fc99bace389bc996438593acaa581`.
- Native binary SHA-256:
  `6b305971a1b640e2c9936fcc3fa77f8b0e5fa43cdd948d261dc26cb21028044b`.
- Analyzer: clean `ade43e5c8ad669cc5eb78b5c455f69dd04377f91`, independently
  identified from the older producer; no replacement of its source identity.
- Slurm job6732, node1/main, assigned `CUDA_VISIBLE_DEVICES=0`, finite45min,
  one RTX5090 and requested4 CPUs; GNU13.3/CUDA12.9.86/sm120/driver580.95.05.
  Thread variables are1;
  observed CPU affinity spans128 logical CPUs, not a pinned exclusive core.
  GPU inventory does not establish native/NVML UUID association.

The complete grid is AO40/41/62/122/180 × B1/8/32/64/128/256 × static requested
active fractions1/.5/.25. There are90 replay-pass coordinates,2700 measured
pairs and450 excluded warmup pairs. AO62/122 use the existing alkane builders;
AO40/41/180 are synthetic probes. Materialized geometry hashes are unavailable;
fixture source hashes and exact producer/native/cache identities are indexed.

`component-table.csv` retains all90 coordinates, outcomes, activity, observed
counts, setup, known memory/executables and descriptive paired ratio quantiles.
Floats use six-significant-digit display precision; authoritative exact values,
failure reasons, limitations and complete ledger statistics stay in the
hash-pinned external full JSON/CSV. Quantiles are not confidence intervals.
Schema-complete failure distributions are descriptive only; malformed or
incomplete records never contribute usable statistics.
For AO122/B128 the chain/tail median ratios are about9.045 full,5.737 half and
3.590 quarter active. This is a component regression, not reproduction of the
historical4.28× result or a public-throughput conclusion. Zero-active and tiny
ratios below1 are not accepted benefit regions.

## Reproduction

At the clean producing checkout, in an allocated finite GPU job with scheduler
visibility preserved:

```bash
python3 -m benchmarks.cuda_crossover --run \
  --binary /absolute/producing-build/xtbloom_cuda_scc_iteration_production_test \
  --cmake-cache /absolute/producing-build/CMakeCache.txt \
  --warmups 5 --samples 30 --seed 515 --timeout 900 \
  --output /absolute/external/component-matrix.jsonl
```

At the clean analyzer checkout, without GPU execution:

```bash
python3 -m benchmarks.cuda_crossover_summary \
  --input /absolute/external/component-matrix.jsonl \
  --json-output /absolute/external/component-summary.json \
  --csv-output /absolute/external/component-summary.csv \
  --table-csv-output /absolute/external/component-table.csv
sha256sum -c SHA256SUMS
```

`artifact-index.json` pins the complete external raw matrix (13233337 bytes,
SHA-256 `c43b8c973c38d17763d2d512d3b908a38ea4cff9ee700de82f772fdb6710d3ab`),
full exact summaries, the projection and both source identities. Raw samples
are reproducible and deliberately omitted from Git; no evidence size gate or
raw-profiler exclusion is weakened. External files are retained on node1
under the indexed directory, with the original raw matrix separately retained.

These are own-repository generated observations. No original2048 molecule
bytes, new dependency, copied third-party code, vendor binary or raw profiler
capture enters this bundle. Existing source/data and CUDA/MKL additional-
permission boundaries remain unchanged; vendor runtimes are not GPL-covered
by this statement or redistributed here.

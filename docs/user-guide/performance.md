# Performance evidence

xTBloom targets workloads that evaluate many differently sized molecular
systems through one reusable public-API context. Performance claims are
published only with raw timing samples, environment and binary identity,
correctness gates, and explicit limitations.

## Cross-engine molecule-size scaling

![Cross-engine GFN2-xTB scaling benchmark](../assets/natoms_cross_engine.svg)

The public figure measures GFN2-xTB energy plus analytic-force latency for
distinct conformers of one alkane family.

CUDA curves use an NVIDIA GeForce RTX 5090 (driver 580.95.05). The xTBloom
CUDA rows were refreshed on 2026-08-21; CPU and third-party rows retain their
unchanged 2026-08-09 measurements.

On an AMD EPYC 7K62 with the same 16-thread CPU budget:

| Workload | xTBloom CPU | xTB | tblite | Measured xTBloom speedup |
| --- | ---: | ---: | ---: | ---: |
| 128 systems, 62 atoms, warm/persistent state | 182 ms | 1555 ms | 1384 ms | 8.6x / 7.6x |
| 512 systems, 62 atoms, cold state | 1.28 s | 11.47 s | 13.70 s | 8.9x / 10.7x |

The result reflects xTBloom's intended ragged-batch execution path: xTBloom
submits the complete batch, while the compared xTB and tblite public adapters
loop over per-structure calls. It does not establish a universal
single-molecule or cross-library ranking.

The batch-1 panel is latency context. Batch 128 uses an untimed cold seed
followed by warm/persistent calls; batches 1 and 512 use cold electronic state.
xTBloom CUDA uses host descriptors in this figure, while dxtb CUDA retains
device tensors, so no direct cross-library CUDA speedup is claimed.

Every timed dependent sample passed the panel-matched output gate. The evidence
is limited to the recorded alkane corpus, energy plus forces, three measured
samples per coordinate, and the stated hardware.

- [Methodology](../../benchmarks/cross-engine.md)
- [Latest machine-readable table](../../benchmarks/natoms_cross_engine_latest.csv)
- [RTX 5090 xTBloom CUDA evidence](../../benchmarks/evidence/issue-467/2026-08-21-node3/README.md)
- [Historical CPU and third-party evidence](../../benchmarks/evidence/issue-13/2026-08-09-node3-pr231/README.md)

## Warm-state CPU evidence

A separate single-system CPU study on an AMD EPYC 7K62 pins one logical CPU,
one xTBloom worker, and one-thread BLAS, with 30 samples per coordinate and
stricter conformance settings. For the measured 32–122 atom alkane corpus,
strict compatible warm starts reduce SCC work from 17–18 iterations to 2 and
run 1.09x–1.54x faster than the measured persistent tblite calculation.

This is a different protocol from the cross-engine figure and uses different
correctness thresholds. Do not mix its numbers or gates with that figure.

- [FRESH/WARM methodology](../../benchmarks/fresh-warm.md)
- [Archived evidence](../../benchmarks/evidence/issue-168/2026-08-06-epyc7k62/README.md)

## Reading performance claims

A result is eligible for publication only when the requested backend actually
ran, every required output passed its correctness gate, raw samples and
environment identity were retained, and skipped or unavailable coordinates
remain visible. A benchmark on one molecule family or device never becomes a
release-wide guarantee.

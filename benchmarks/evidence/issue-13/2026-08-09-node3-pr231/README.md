# Issue #13: cross-engine molecule-size scaling

This bundle is the public evidence used by PR #231. It measures end-to-end
GFN2-xTB energy plus analytic-force latency through each library's public
interface for distinct conformers of one alkane stoichiometry. The scope is a
selected high-throughput workload, not a general ranking of xTB-family
libraries.

All measured JSON/CSV artifacts were produced from clean xTBloom commit
`c9c0a432947f122d25cb91d0a4624af0a3e761ad` on 2026-08-09. JSON retains raw
samples, complete final force vectors, per-sample output checks, convergence
state, build/runtime identity, CPU affinity, and GPU UUID. Issue #348 removed
seven JSON files over the repository's 1 MiB evidence limit from the current
tree while retaining their compact CSV views. Their exact historical path,
byte count, SHA-256, and retrieval revision are recorded in
`benchmarks/evidence/legacy-large-artifacts.tsv`; under-limit JSON remains in
this bundle. The SVG was derived from the complete measured artifacts with
`benchmarks/plot_natoms_cross_engine.py` SHA-256
`deeaf58589cabc2b9ae71a314492be8a9420e3eeef1ddc00c9728ddeabac9aa9`.

## Result

Median public-call latency in milliseconds; each available coordinate has one
warmup and three timed samples. The SVG shows the observed min-max interval.

### Batch 1, cold electronic state

| atoms | xTBloom CPU | xTBloom CUDA | xTB | tblite | dxtb CPU | dxtb CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 2.747 | 14.528 | 3.400 | 4.286 | 147.156 | 331.464 |
| 32 | 11.534 | 41.952 | 7.242 | 12.478 | 199.967 | 380.566 |
| 62 | 35.958 | 68.936 | 21.832 | 24.918 | 328.366 | 428.167 |
| 122 | 134.229 | 194.324 | 87.339 | 100.471 | 750.878 | 496.256 |
| 242 | 607.670 | 945.004 | 449.544 | 469.164 | 3474.477 | 744.665 |
| 362 | 1612.963 | 2908.959 | 1194.935 | 1389.230 | 8939.077 | 1214.937 |

### Batch 128, warm after an untimed seed

xTBloom, xTB, and tblite continue from persistent electronic state. dxtb has no
equivalent continuation path in this adapter and resets inside every timed
call.

| atoms | xTBloom CPU | xTBloom CUDA | xTB | tblite | dxtb CPU | dxtb CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 16.876 | 28.119 | 236.419 | 152.118 | 1355.310 | 1082.919 |
| 32 | 52.282 | 60.693 | 527.083 | 476.014 | 3963.157 | 2316.508 |
| 62 | 181.693 | 181.694 | 1554.560 | 1384.229 | 15003.518 | 7732.841 |
| 122 | 682.283 | 637.054 | 6129.988 | 4972.355 | unavailable | OOM |

At the common 62-atom coordinate, xTBloom CPU is 8.56x faster than xTB and
7.62x faster than tblite per public call.

### Batch 512, cold electronic state

| atoms | xTBloom CPU | xTBloom CUDA | xTB | tblite | dxtb CPU | dxtb CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 70.065 | 214.515 | 1415.640 | 1277.834 | 5096.588 | 3869.492 |
| 32 | 350.098 | 307.800 | 3812.131 | 4648.710 | 25434.360 | 19504.037 |
| 62 | 1282.293 | 1152.002 | 11474.464 | 13697.168 | 104793.390 | OOM |
| 122 | 5528.880 | 4035.910 | 44094.246 | 50513.279 | unavailable | OOM |

At 62 atoms, xTBloom CPU is 8.95x faster than xTB and 10.68x faster than
tblite. xTBloom CUDA is 1.11x faster than xTBloom CPU at that coordinate and
1.37x faster at 122 atoms.

## Correctness qualification

`2e-3` is an owner-authorized output-compatibility gate for this public
benchmark. It is not the tblite convergence default and does not replace the
repository's primary scientific conformance gates.

Every timed dependent sample is compared with the panel-matched clean tblite
reference and must satisfy:

```text
max_s |Delta E_s| <= 2e-3 Eh per system
max_i |Delta F_i| <= 2e-3 Eh/bohr
```

All 65 available dependent rows pass; the remaining 14 available rows are the
panel-matched tblite references. The largest observed dependent deltas are:

| engine | max energy delta (Eh) | max force-component delta (Eh/bohr) |
| --- | ---: | ---: |
| xTBloom CPU | 4.5711e-7 | 2.9322e-5 |
| xTBloom CUDA | 4.5711e-7 | 2.9322e-5 |
| xTB | 1.3808e-4 | 1.0953e-5 |
| dxtb CPU | 1.2719e-4 | 1.0758e-5 |
| dxtb CUDA | 1.2719e-4 | 9.8941e-6 |

Five of 84 requested rows are explicitly unavailable rather than silently
dropped:

- dxtb CPU, 122 atoms at batch 128 and 512: LU solve rejects invalid pivots;
- dxtb CUDA, 122 atoms at batch 128: CUDA out of memory;
- dxtb CUDA, 62 and 122 atoms at batch 512: CUDA out of memory.

The native convergence controls retain each library's public meaning:

| library | controls used |
| --- | --- |
| xTBloom | charge `1e-4`; energy `1e-6`; maximum 500 iterations |
| xTB | public accuracy factor `1.0`; maximum 500 iterations |
| tblite | public accuracy factor `1.0`; maximum 500 iterations |
| dxtb | `x_atol=1e-4`; `x_atol_max=1e-5`; `f_atol=1e-4`; `force_convergence=true`; maximum 500 iterations |

At tblite source commit `133f91ef`, the C API initializes its accuracy factor
to `1.0`; the single-point implementation multiplies its internal convergence
settings by that factor. xTB 6.7.1 likewise documents `1.0` as its default
accuracy factor. Therefore the earlier PR data produced with
`accuracy=1e-4` is not reused here.

## Timing boundaries

| panel policy | xTBloom | xTB/tblite | dxtb |
| --- | --- | --- | --- |
| cold | `FRESH` initialization inside timed `xtbloom_compute` | calculator rebuild outside timing; public single point and getters timed | reset, single point, synchronization, and host tensor publication timed |
| auto-warm | untimed cold seed, then measured strict `WARM` calls | untimed cold seed, then measured persistent calls | remains cold/reset for every measured call |

Every interval ends with host-visible energy and forces. Python-list
normalization is outside timing. xTBloom CUDA uses host descriptors staged by
the public C ABI; dxtb CUDA retains device tensors. Their CUDA curves expose
observed behavior but are not used for a direct cross-library CUDA speedup
claim.

## Hardware and identities

- Host: `node3`, AMD EPYC 7K62, process affinity CPUs 0-15.
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, UUID
  `GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`, driver 580.95.05.
- CPU budget: 16 xTBloom workers / reference threads; OpenBLAS and MKL set to
  one internal thread to avoid nested oversubscription.
- xTBloom CPU library SHA-256:
  `eba5b40dd5cd9c156a3d4eb9a1fdb0da96e0d5beaa668bdeb51694ce67364ad2`.
- xTBloom CUDA library SHA-256:
  `d995eca4a864db2d7f5fd284e9288f6f5cdd6a1436537fb1bd4fe600dfc0c524`;
  CUDA architecture 120, nvcc 12.9.86.
- xTB 6.7.1 library SHA-256:
  `959ed711f85f3c84e5e9dffd15e1b49dfae1bd46783f544ce8756b5e148094dd`;
  clean source `b31754bf`.
- tblite 0.7.0 library SHA-256:
  `1c2fb4308b398851580af11ccad5eec22314ff45a34a553380277e776a44c3b5`;
  clean source `133f91ef`.
- dxtb 0.4.0 clean source `b529b5dd`; PyTorch 2.13.0+cu130;
  tad-libcint 0.3.0. JSON metadata verifies the installed RECORD payloads.

The JSON metadata contains the complete compiler, CMake cache, native
dependency, source-state, and installed-distribution fingerprints.

## Commands

The exact measured orchestration scripts are retained in this directory.
Phase 1 stopped after the first explicit dxtb unavailable row because the
runner intentionally exits with status 2 when any requested coordinate fails.
Phase 2 resumes the remaining cells and retains status-2 artifacts only when
both JSON and CSV were successfully published.

```bash
srun -n 1 --gres=gpu:1 -c 16 -w node3 \
  taskset -c 0-15 bash run-evidence-phase-1.sh

srun -n 1 --gres=gpu:1 -c 16 -w node3 \
  taskset -c 0-15 bash run-evidence-phase-2.sh
```

To render the figure again, first restore the complete historical bundle, then
pass every JSON artifact to the plotter:

```bash
restore=/tmp/issue-13-pr231-evidence
mkdir -p "$restore"
git archive cbdf755f27ab02b548783bce3573ecb4385ed167 -- \
  benchmarks/evidence/issue-13/2026-08-09-node3-pr231 | \
  tar -x -C "$restore"
cd "$restore/benchmarks/evidence/issue-13/2026-08-09-node3-pr231"
artifacts=()
for artifact in ./*.json; do
  artifacts+=(--artifact "$artifact")
done
uv run --script /path/to/xtbloom/benchmarks/plot_natoms_cross_engine.py \
  "${artifacts[@]}" \
  --output natoms_cross_engine.svg
```

## Scope and limitations

- One homogeneous alkane family with distinct seeded conformers.
- Energy plus analytic forces only; no energy-only, mixed-chemistry, QM/MM,
  memory-capacity, or profiler conclusion.
- Batch sizes 1, 128, and 512; three timed samples per coordinate.
- xTB/tblite adapters use their per-structure public APIs in a serial loop,
  while xTBloom accepts the complete ragged batch in one public call.
- The evidence supports the stated coordinates and hardware only; it is not a
  release-wide performance guarantee.

`SHA256SUMS` covers the JSON, CSV, scripts, README, and SVG retained in the
current tree. The legacy artifact table independently pins the removed raw
JSON bytes.

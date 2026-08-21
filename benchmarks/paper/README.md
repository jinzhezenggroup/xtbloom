# xTBloom paper experiment scripts (plan snapshot 2026-08-21)

This bundle implements the execution surface for the paper plan confirmed on
2026-08-21. It is tracked with the benchmark harness, while datasets, build
trees, and outputs live below the configured `PAPER_DATA_ROOT`. The selected
xTBloom checkout is read-only during an experiment.

The scripts do **not** turn an unavailable coordinate into a pass.  Formal mode
requires a clean repository at the configured commit, the exact plan hashes above, immutable manifests,
real Slurm allocation, and (for CUDA jobs) the configured GPU name.  Development
mode is only for syntax/import/smoke work and marks every bundle ineligible for
the paper.

## Plan identity

- `protocol/paper-experiment-plan.md`: SHA-256
  `52932a8124e74de05979758d3dbd89aa3eb0ecbf5367c2fae6b6311aefb2ad21`
- `protocol/xtbloom-paper-outline.md`: SHA-256
  `721ac2b9c42f5cf48528266a73b2ad6bf06b58279911cd174b018abad841e605`

These are byte-for-byte snapshots of the canonical documents under `docs/`.
Keeping the snapshots inside the deployable suite lets every run checksum the
scientific protocol independently of the surrounding checkout layout. The
OMol25 stress count is fixed at 2,000 in the plan, outline, and scripts.

## Script-to-experiment map

| Script | Paper experiment | Main output |
| --- | --- | --- |
| `slurm/00_freeze_manifests.sbatch` | Dataset/eligibility/manifest prerequisite | immutable paper selection and finite-difference coordinates |
| `slurm/01_p0a_canonical_cpu.sbatch` | P0-A CPU independent conformance | canonical CPU test log |
| `slurm/02_p0a_canonical_gpu.sbatch` | P0-A real-GPU host/device/mixed conformance | canonical CUDA test log |
| `slurm/03_p0b_dataset_cpu.sbatch` | P0-B QM9/OMol25 xTBloom CPU + xTB/tblite/dxtb | per-system JSONL |
| `slurm/04_p0b_dataset_gpu.sbatch` | P0-B xTBloom CUDA host/device + dxtb CUDA | per-system JSONL |
| `slurm/05_p0c_finite_difference_cpu.sbatch` | P0-C real-system derivative check, CPU | coordinate-level finite differences |
| `slurm/06_p0c_finite_difference_gpu.sbatch` | P0-C real-system derivative check, CUDA | coordinate-level finite differences |
| `slurm/07_p0d_failure_cpu.sbatch` | P0-D CPU failure/publication semantics | focused CTest evidence |
| `slurm/08_p0d_failure_gpu.sbatch` | P0-D CUDA failure/publication semantics | focused CTest evidence |
| `slurm/09_p0e_degeneracy.sbatch` | P0-E exact-degeneracy stress coordinate | pinned cross-engine records |
| `slurm/09b_p0_gate.sbatch` | Mandatory P0 stage gate before timing | fail-closed equivalence report |
| `slurm/12_performance_references.sbatch` | Disjoint performance-set qualification prerequisite | xTB/tblite per-system references |
| `slurm/10_exp1_cpu_native_batch.sbatch` | Experiment 1 native CPU batch matrix | raw timing JSONL |
| `slurm/11_si_cpu_process_pool.sbatch` | SI persistent process-pool sensitivity | wall/CPU/RSS/startup samples |
| `slurm/20_exp2_gpu_crossover.sbatch` | Experiment 2 AO x batch CPU/CUDA crossover | host/device timing JSONL |
| `slurm/21_exp2_gpu_capacity.sbatch` | Experiment 2 dataset-level distinct-real capacity sweep | explicit available/OOM/sample-censored rows and claim eligibility |
| `slurm/22_exp2_gpu_profiler.sbatch` | Experiment 2 three operational profiler candidates | derived CSV/text/JSON plus explicit interpretation-required record |
| `slurm/30_exp3a_convergence.sbatch` | Experiment 3-A paired convergence analysis | unweighted Wilson CI, weighted Kish-Wilson approximation, pair tables and SCC summaries |
| `slurm/31_exp3b_ragged_cpu.sbatch` | Experiment 3-B controlled/real ragged CPU | all-input and matched-success timings |
| `slurm/32_exp3b_ragged_gpu.sbatch` | Experiment 3-B controlled/real ragged CUDA | all-input/success/VRAM timings |
| `slurm/40_si_gfn1_cpu.sbatch` | SI GFN1 CPU energy/force/charge validation | focused CTest evidence |
| `slurm/41_si_qmmm.sbatch` | SI QM/MM energy and force accounting | conformance/invariance evidence |
| `slurm/42_si_cuda_mixed.sbatch` | SI CUDA mixed-descriptor correctness/spot check | focused tests + timing rows |
| `slurm/43_si_energy_only.sbatch` | SI representative energy-only coordinates | qualified timing rows |
| `slurm/44_si_warm_trajectory.sbatch` | Conditional SI WARM real-trajectory sanity | fresh/warm paired rows |
| `slurm/45_si_second_hardware.sbatch` | Recommended second-hardware three-point repeat | representative timing rows |
| `slurm/90_analyze_archive.sbatch` | Table 1 / Figures 2-4 inputs and archive integrity | compact summaries + SHA256SUMS |

P0-B and Experiment 3-A deliberately reuse the reviewed dataset runner so
manifest order, IDs, hashes, charge/spin, failure isolation, and unavailable
records stay authoritative. Runner and adapter code come from the configured
clean checkout at `PAPER_SOURCE_COMMIT`; stage environment records and hashes
bind that implementation to the evidence without maintaining a second copy.

The equivalence analyzer writes both compact aggregate tables and a
`raw/paired-errors.jsonl` system ledger.  The ledger retains energy-per-atom,
force-component, atom-force-vector, and charge RMSE metrics. Main probability
samples report both realized-sample and inverse-probability-weighted quantiles;
stress rows are explicitly non-population estimates. Charge equivalence is
provided by the tblite and CPU-CUDA comparisons because the pinned xTB adapter
does not expose an atomic-charge getter. The archive analysis also emits
`si-finite-difference.csv`, normalized SCC ECDF/stratum tables, and the
persistent-process-pool comparison with concurrent whole-process-group RSS.

## Usage

1. Copy `config/paper.env.example` to a run-specific file outside this bundle.
2. Fill every path and site scheduler setting, then set `PAPER_SOURCE_COMMIT`
   to the clean release candidate.
3. Set `PAPER_PHASE=formal` only after the manifests and commit are frozen.
4. List jobs with `bin/submit.sh --config /abs/paper.env --list`.
5. Submit one job with `bin/submit.sh --config /abs/paper.env p0b-dataset-cpu`.
6. Submit the dependency-ordered plan with `bin/submit.sh --config ... --all`.
   Formal jobs reject direct `sbatch`; the wrapper also puts stdout/stderr in
   the run-specific `slurm/` directory and records the exact submission. On
   first submission it freezes the config bytes under the RUN_ID; any later
   edit requires a new RUN_ID. Every stage also pins the script checksum
   manifest to the `freeze-manifests` stage.

`--all` excludes conditional WARM and second-hardware jobs unless their
corresponding `PAPER_RUN_*` flag is `1`.  Every other stage checks prerequisite
`.complete` markers itself; a manually requested later stage cannot bypass P0.

For a future FP64-oriented NVIDIA GPU, set the site Slurm partition/GRES and
`PAPER_EXPECT_GPU_REGEX` (for example `NVIDIA (A100|H100)`).  xTBloom remains
IEEE binary64; this switch selects hardware, not a different numerical mode.
Also set a new `PAPER_HARDWARE_ID`, update the CUDA compiler/root, and leave
`PAPER_CUDA_ARCH=auto` so the architecture is derived inside the allocation.
The separate `PAPER_SECOND_*` variables are only for the optional three-point
SI replicate and are required to identify a genuinely different machine.

## Formal-run prerequisites

The portable example deliberately leaves site and evidence identities
fail-closed. Before a formal run, configure and review:

- the disjoint frozen QM9 and OMol25 manifests and complete licensed bundles;
- pinned xTB, tblite, dxtb, Python-package, and linear-algebra identities;
- a clean xTBloom source commit and run-local canonical CPU/CUDA builds;
- the site scheduler command/resources, CPU policy, CUDA toolkit, and expected
  hardware identity.

Do not replace any item with an older runtime or a main-sample manifest.  The
manifest-freeze stage verifies every bundle `SHA256SUMS`, retained license,
eligibility/sampling metadata, disjointness, sample counts, probabilities and
input hashes.  Every later job re-verifies the frozen bytes before computing.

## Evidence boundaries

- `PAPER_PHASE=development`: dirty sources allowed only when
  `PAPER_ALLOW_DIRTY=1`; outputs are tagged ineligible.
- `PAPER_PHASE=formal`: clean exact commit, real Slurm job, complete configured
  libraries/manifests, and matching GPU are mandatory.
- Raw Nsight captures stay under the run directory in `raw-profiler/`.  Only
  derived reports may be copied into a repository evidence bundle.
- Every experiment writes environment logs, exact argv, source/binary hashes,
  raw samples, explicit failures/unavailable rows, and a checksum manifest.
- Formal stages require one frozen Python executable/package inventory, the
  configured linear-algebra provider hash, identical CPU governors on every
  Slurm-assigned CPU, fixed physical affinity, and an explicit NUMA interleave
  policy applied before Python loads inputs or creates adapters.
- Timed force rows use the common `energy + analytic forces` output contract.
  xTBloom atomic charges are correctness-qualified in a separate untimed call
  for force timing matrices, so a charge getter cannot distort xTB/tblite/dxtb
  timing comparisons. Capacity and energy-only sweeps intentionally retain
  only their declared timed output contract; charge evidence comes from P0.
- All formal timing coordinates, including capacity and ragged matrices, use
  the frozen minimum of 10 warm-ups and 30 measured samples. Profiler traces
  use five captured calls only as attribution evidence, never as a timing
  estimate.
- Each performance cell runs a separate untimed, three-invocation resource
  probe in a fresh spawned process. It covers adapter setup plus three
  synchronized calls, samples current RSS and NVML per-process VRAM at a 2 ms
  target interval, and retains every change plus periodic raw samples. Process
  exit prevents allocator state from leaking between cells; monitoring
  failures make required cells unavailable.
- Performance rows fail the job after retaining diagnostic rows when a
  required correctness-qualified cell is non-finite, missing, or outside the
  registered gate. Empty/sparse AO cells remain explicit `not-applicable` or
  `sample-censored` records. Capacity sweeps stop at the distinct frozen real
  structures; `sample-censored` closes the saturation/OOM claim until a larger
  capacity-only manifest is frozen. Expected capacity OOM and unavailable
  optional dxtb rows remain evidence rather than being relabelled as passes.

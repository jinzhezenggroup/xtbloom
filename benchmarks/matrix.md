# Public cross-library matrix

`run.py` measures end-to-end inference through the public interfaces of
xTBloom, xTB, tblite, and dxtb. It covers gas-phase and QM/MM workloads,
energy/force property sets, CPU/CUDA backends, and multiple batch sizes.
The default workloads remain homogeneous `gas,qmmm`. The optional
`heterogeneous-gas,heterogeneous-qmmm` rows construct one real ragged batch by
cycling named committed conformance cases in manifest order defined by the
runner; batches longer than the candidate list wrap deterministically.

## Timing boundary

For xTBloom, the context, ragged descriptors, and result buffers stay alive
between calls. Setup, first-call, and steady-state timings are reported
separately. CUDA calls are explicitly synchronized at the end of every measured
interval; result downloads used only for correctness checking remain outside
timing.

Steady-state samples keep coordinates unchanged and are labeled
`same_geometry_repeated_compute`. They still execute the full public compute
path and do not, by themselves, prove pair-list no-refresh reuse.

For QM/MM force rows, the timed xTBloom call requests and validates both QM
forces and external-point-charge forces.

The reference adapters preserve unavoidable public-API work:

- xTB keeps one environment, molecule, calculator, and result per logical
  system, then times its serial public-call loop. Its 6.7.1 C API has no
  energy-only flag, so an energy row still pays for the native full
  single-point calculation. QM/MM rows rebind the persistent external-charge
  arrays inside every measured call because libxtb 6.7.1 otherwise accumulates
  point-charge gradients; that required public-API work remains in latency.
- tblite likewise keeps persistent public objects and times geometry update,
  single point, and requested getters. Its public C API lacks discrete point
  charges, so those QM/MM coordinates remain explicitly unavailable.
- dxtb keeps one in-process PyTorch calculator and resets it for every measured
  call so tensor-identity caching cannot turn the workload into a cache hit.
  CUDA timing includes synchronization; unsupported dependencies or QM/MM
  mappings remain visible as unavailable rows.

## Example

Use paths and scheduler options for the evidence machine:

```bash
python3 benchmarks/run.py \
  --library /absolute/path/to/libxtbloom.so \
  --tblite-library /absolute/path/to/libtblite.so \
  --xtb-library /absolute/path/to/libxtb.so \
  --xtb-executable /absolute/path/to/xtb \
  --engines xtbloom,tblite,xtb,dxtb \
  --batch-sizes 1,8,32,128 \
  --properties energy,force \
  --workloads gas,qmmm \
  --fail-on-correctness

python3 benchmarks/run.py \
  --library /absolute/path/to/libxtbloom.so \
  --engines xtbloom,xtb,tblite,dxtb \
  --backends cuda --cuda-memory-modes host \
  --batch-sizes 8,32 \
  --properties energy,force \
  --workloads heterogeneous-gas,heterogeneous-qmmm \
  --fail-on-correctness
```

Homogeneous JSON/CSV rows retain scalar `case_id`. Heterogeneous rows retain
the complete ordered `case_ids` vector. xTB receives per-system source atomic
numbers when every selected QM/MM case exposes the element-hardness mapping;
gamma-only QM/MM cases remain explicit `unavailable` coordinates. tblite and
dxtb likewise remain explicit `unavailable` for any row containing point
charges because their adapters cannot express that public coordinate.

The default outputs are `build/benchmarks/matrix.json` and
`build/benchmarks/matrix.csv`. Artifacts are written before the runner returns
status 2 for a requested row that fails correctness or is unavailable.

## Validation

```bash
python3 -m unittest -v benchmarks.test_run
python3 -m unittest -v benchmarks.test_dxtb_adapter
```

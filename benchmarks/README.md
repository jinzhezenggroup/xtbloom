# Cross-library benchmark harness

`run.py` measures end-to-end inference through gpuxtb's public C API. It keeps
the context, ragged batch descriptors, and result buffers alive between calls,
separates setup, first-call (cold), and steady-state timings, and explicitly
synchronizes CUDA after every measured call. Device result downloads used for
correctness checking happen after timing.
For QM/MM force rows the measured public call requests both QM atomic forces
and external-point-charge forces; both are compared with committed goldens.

Run the complete requested gpuxtb matrix on one allocated GPU:

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/tmp/gpuxtb-reference-env.E0KcEA/lib:/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/run.py \
    --library build-cuda-shared/libgpuxtb.so.0.1.0 \
    --tblite-library /path/to/validated/libtblite.so \
    --xtb-library /tmp/gpuxtb-reference-env.E0KcEA/lib/libxtb.so.6.7.1 \
    --xtb-executable /tmp/gpuxtb-reference-env.E0KcEA/bin/xtb \
    --engines gpuxtb,tblite,xtb,dxtb \
    --batch-sizes 1,8,32,128 \
    --properties energy,force \
    --workloads gas,qmmm
```

For a quick CUDA smoke run:

```bash
srun --gres=gpu:1 env \
  LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib \
  MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL \
  python3 benchmarks/run.py \
    --library build-cuda-shared/libgpuxtb.so.0.1.0 \
    --engines gpuxtb --backends cuda --cuda-memory-modes host,device \
    --workloads gas --properties energy,force --batch-sizes 1,8 \
    --warmups 1 --repetitions 2
```

The default artifacts are `build/benchmarks/matrix.json` and
`build/benchmarks/matrix.csv`. JSON retains raw samples, environment variables,
hardware/runtime information, exact git revisions, library hashes, correctness
errors, SCC iterations, and memory snapshots. CSV contains the main comparison
columns and embeds raw timing and memory evidence as JSON fields.

The xTB baseline dynamically loads the public C API from `--xtb-library`.
Every logical batch system owns a persistent environment, molecule, calculator,
and result object; measured batch execution is an in-process serial loop over
`xtb_updateMolecule`, `xtb_singlepoint`, and the requested getters. Process
startup and object construction are excluded from steady state and reported as
setup. xTB 6.7.1 has no energy-only C API flag, so an `energy` row still pays
for its native full single-point calculation; this is recorded in each row and
must be considered when comparing it with gpuxtb energy-only inference.

QM/MM xTB rows use `xtb_setExternalCharges` with the corpus' source atomic
numbers, which exactly represents the selected element-hardness case. The
adapter rebinds the same persistent external-charge arrays inside each measured
call because libxtb 6.7.1 otherwise accumulates PC gradients when the object is
reused. This required C API work is included in latency.

The tblite baseline likewise uses its public C API through
`--tblite-library`, with one persistent context, structure, calculator, and
result per logical batch system. Its measured serial loop includes geometry
update, single-point execution, and requested getters, but excludes process
startup and object construction. The tblite 0.6 public C API does not expose
discrete external point charges, so requested QM/MM rows are retained as
explicit `unavailable` coordinates rather than silently omitted.

The dxtb baseline uses one persistent in-process PyTorch `Calculator` and
input tensors from `--dxtb-source`. CPU and CUDA rows are selected with
`--dxtb-backends`; CUDA timing ends with an explicit PyTorch synchronize and
host result download remains outside the measured interval. Every measured
call includes `Calculator.reset()` so dxtb's tensor-identity result cache
cannot turn repeated geometry into a cache-hit benchmark. The active Python
environment must provide dxtb's normal dependencies (including
`tad-libcint` for GFN2); otherwise requested rows remain explicitly
`unavailable`. Discrete QM/MM point-charge mapping is not implemented yet and
is likewise reported as unavailable rather than silently omitted.

Use `--fail-on-correctness` when the command should return status 2 if an
available row exceeds the committed primary conformance tolerance. Artifacts
are written before that status is returned.

Run the hardware-independent harness self-check with:

```bash
python3 -m unittest -v benchmarks.test_run

# Requires PyTorch; uses a fake differentiable dxtb runtime, not dxtb itself.
python3 -m unittest -v benchmarks.test_dxtb_adapter
```

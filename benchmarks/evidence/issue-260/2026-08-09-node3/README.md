# Issue #260 CUDA D4 ATM Evidence

This bundle records committed-head evidence for PR #261 after fix commit
`a31d31a000f18e00f8f6caf9fc2983b97e407e21`.

## Environment

- Host: `node3`, AMD EPYC 7K62, 48 CPUs allocated by Slurm.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, driver `580.95.05`.
- CUDA: toolkit `12.9.1`, `nvcc` V12.9.86, architecture `120`.
- Build: shared Release CMake build in `build/bench-cuda-shared`.
- Library SHA-256: `bf85c14b8a9d6df90670f436de86fa1b7c66aabc15bd41ea34f1e9f99e8b7254`.
- CMake cache SHA-256: `5d95a991b124f4293dda5d27dc91b6e4b68f9b4d60a223e1eb09eddf762a8e26`.
- Conformance manifest SHA-256: `0f7d222c089c1d03603239ebef8fa03522f77374ca3f403e3749e0937e52b1e9`.

## Public Timing

The retained timing artifact is `cuda-fresh.csv`. It contains every measured
latency sample plus the per-cell timing, energy, SCC-iteration, and numerical
drift summaries needed for the performance claim. The run used the repository
`benchmarks/natoms_scaling.py` harness through the public C ABI:

```text
srun --partition=main --gres=gpu:5090:1 --ntasks=1 --cpus-per-task=48 --wait=60 env PYTHONPATH="$PWD/python" LD_LIBRARY_PATH="$PWD/build/bench-cuda-shared:/group/software/cuda-12.9.1/lib64:/group/software/deepmd-kit-3.1.1/lib:$LD_LIBRARY_PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=SEQUENTIAL python3 benchmarks/natoms_scaling.py --engine gpuxtb --library "$PWD/build/bench-cuda-shared/libgpuxtb.so.0.1.0" --backend cuda --property force --natoms 62,122,242 --batch-sizes 1,128 --warmups 3 --repetitions 5 --start-mode fresh --output-json "$PWD/benchmarks/evidence/issue-260/2026-08-09-node3/cuda-fresh.json" --output-csv "$PWD/benchmarks/evidence/issue-260/2026-08-09-node3/cuda-fresh.csv"
```

All 30 measured samples were available and passed the harness finite-output
and repeated-FRESH correctness checks. The median public latency in
milliseconds was:

| Atoms | Batch 1 | Batch 128 |
| ---: | ---: | ---: |
| 62 | 151.865677 | 442.835534 |
| 122 | 449.934003 | 1638.566578 |
| 242 | 2155.627751 | 7974.954526 |

The batch-1 rows are the requested large-system scaling claim. Batch 128 is a
regression coordinate; its work is homogeneous and therefore retains the split
path. The public timing does not by itself prove an independent oracle; the
same commit passed the full CUDA CTest and public CUDA conformance/invariant
matrix.

The harness also generated a 34,903,491-byte full JSON artifact with complete
force vectors (SHA-256
`d8c45c56ed189539fbf4416bbda77289231e789f2060d791db64afedcdcbb063`).
It is intentionally omitted from the repository to keep the retained evidence
compact. Consequently, the CSV preserves all raw latency samples and recorded
drift maxima, while the separately executed CUDA conformance and CTest matrix
is the retained numerical-correctness evidence.

## Nsight Summary

`nsys-kernel_cuda_gpu_kern_sum.csv` is a sanitized derived report from Nsight
Systems `2025.1.3.140-251335620677v`, generated with:

```text
/group/software/cuda-12.9.1/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=true --force-overwrite=true -o /tmp/gpuxtb-261-d4-atm build/bench-cuda-shared/gpuxtb_cuda_d4_test
/group/software/cuda-12.9.1/bin/nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_kern_sum --format csv --output benchmarks/evidence/issue-260/2026-08-09-node3/nsys-kernel /tmp/gpuxtb-261-d4-atm.nsys-rep
```

The focused D4 test's split kernels measured `0.244289 ms` for
`atm_energy_split_kernel` and `0.531139 ms` for
`atm_gradient_split_kernel`. Raw `.nsys-rep`, `.sqlite`, and `.qdstrm` files
were kept outside the repository under `/tmp` and are intentionally not part
of this bundle.

This bundle records current-head results and the retained profiler extraction;
the historical pre-change values in the PR description were not regenerated
from a separate baseline binary in this run.

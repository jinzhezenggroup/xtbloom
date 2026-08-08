# Issue #253 CUDA solve-bucket evidence

This bundle compares current `main` with the final reviewed issue #253 code on
the same public-C-ABI FRESH energy workload. The unsafe density-contraction
optimization proposed by the original PR is absent from the final revision;
the measured change is the cooperative spin solve-bucket preparation.

## Environment

- Host: `node3`, AMD EPYC 7K62, 48 logical CPUs available to the process.
- GPU: NVIDIA GeForce RTX 5090, driver 580.95.05, compute capability 12.0.
- CUDA: toolkit 12.9.1, nvcc 12.9.86, `sm_120`.
- Build: CMake 4.2.1, Ninja, GCC 11.4.0, shared Release library, CUDA ON.
- LP64 provider: `/group/software/deepmd-kit-3.1.1/lib/libmkl_rt.so`, SHA-256
  `221e89c09644d546cdc6505fc1fdecdf6490a4c57f7da6ca3b48a1c96c4860bd`.
- Baseline revision: `e23ec9cdd5e4247a9312a1ca5c851e20720e888c` (`main`).
- Final code revision: `017e840ad0c700909b998fe9fc429b8b1b978ef8`.

Both source trees were clean. The JSON records the repository, library,
compiler, provider, CMake cache, workload, and environment identities. The
baseline library SHA-256 is
`de79cae54c7725bb00d58aa41799a0a3e4f07401ba456c77ee0bee772fbcfcc9`;
the final library SHA-256 is
`8eb1f78ddadf8aed9c9a2ca7b9f4a22e861ee5cc1fb7418d40de59d83f8f3def`.

## Protocol

`benchmarks/natoms_scaling.py` retained one public context, descriptor set,
and caller-owned result set for each run. The coordinate is CUDA with host
descriptors, FRESH energy, one homogeneous batch of 128 C20H42 systems (62
atoms and 122 AOs per system), five untimed warmups, and 50 measured calls.
Every measured call ends at the public synchronous completion boundary;
correctness inspection is outside the timed interval.

Baseline command, run from the clean `main` worktree:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64 \
python3 benchmarks/natoms_scaling.py \
  --engine gpuxtb \
  --library build/cuda-baseline/libgpuxtb.so.0.1.0 \
  --backend cuda --device-id 0 --property energy \
  --natoms 62 --batch-sizes 128 \
  --warmups 5 --repetitions 50 --start-mode fresh \
  --output-json /home/jzzeng/codes/gpuxtb4/benchmarks/evidence/issue-253/2026-08-08-node3/iter2-main-baseline.json \
  --output-csv /home/jzzeng/codes/gpuxtb4/benchmarks/evidence/issue-253/2026-08-08-node3/iter2-main-baseline.csv
```

Final command, run from the clean issue branch:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64 \
python3 benchmarks/natoms_scaling.py \
  --engine gpuxtb \
  --library /home/jzzeng/codes/gpuxtb4/build/pr254-cuda-review2/libgpuxtb.so.0.1.0 \
  --backend cuda --device-id 0 --property energy \
  --natoms 62 --batch-sizes 128 \
  --warmups 5 --repetitions 50 --start-mode fresh \
  --output-json benchmarks/evidence/issue-253/2026-08-08-node3/iter3-final.json \
  --output-csv benchmarks/evidence/issue-253/2026-08-08-node3/iter3-final.csv
```

## Results

| Revision | Median (ms) | Min (ms) | p95 (ms) | Samples |
| --- | ---: | ---: | ---: | ---: |
| `e23ec9c` main | 454.978 | 454.679 | 458.132 | 50 |
| `017e840` final | 420.338 | 419.993 | 422.703 | 50 |

The final median is 7.61% lower than current `main`. Both runs report
`-64.27950675607356 Eh` for every system, zero measured energy drift, 18 SCC
iterations for every system, successful per-system status, and a passing
correctness gate.

This is a narrow single-GPU result for one homogeneous molecule, one batch
size, host descriptors, FRESH start, and energy-only publication. It does not
establish a release-wide speedup across molecule sizes, ragged topology,
device/mixed descriptors, WARM starts, or force workloads. The JSON files are
the authoritative raw-sample artifacts; the CSV files are compact views.

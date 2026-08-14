# Issue #358 quick complete-Hessian matrix

This bundle records a bounded development benchmark for complete dense
62-atom Hessians. It corrects the earlier displacement-chunk interpretation:
`batch=1` means one complete `186 x 186` Hessian and `batch=128` means 128
complete Hessians. Every coordinate used the same `nthreads=16` budget;
xTBloom's displacement chunk size was independently fixed at 128.

The batch-1 xTB/xTBloom rows retain three samples. The batch-128 xTBloom rows
retain one complete-workload sample because one sample already produces 128
dense Hessians. These large-batch numbers are quick throughput diagnostics,
not publication-quality distributions.

## Result

Neutral-singlet C20H42, GFN2-xTB, float64, 0.005 bohr central differences,
AMD EPYC 7K62 cores 0-15, BLAS single-threaded, and NVIDIA GeForce RTX 5090:

| Engine | Complete Hessians | Whole batch | Per Hessian | Hessians/hour | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| xTB 6.7.1 native | 1 | 2.859 s median | 2.859 s | 1,259 | 3 |
| xTBloom CPU | 1 | 1.373 s median | 1.373 s | 2,622 | 3 |
| xTBloom CUDA | 1 | 2.339 s median | 2.339 s | 1,539 | 3 |
| xTBloom CPU | 128 | 145.630 s | 1.138 s | 3,164 | 1 |
| xTBloom CUDA | 128 | 105.841 s | 0.827 s | 4,354 | 1 |
| dxtb 0.4.0 CPU AD | 1 | 96.170 s | 96.170 s | 37.4 | 1 |
| xTB 6.7.1 native | 128 | unavailable | unavailable | unavailable | 0 |
| dxtb 0.4.0 CPU AD | 128 | unavailable | unavailable | unavailable | 0 |

xTBloom CUDA batch 128 was 1.376x faster than xTBloom CPU for the complete
workload. xTBloom CPU batch 1 was 2.083x faster than xTB. CUDA batch 1 was
slower than CPU because this small coordinate did not amortize GPU
SCC/eigensolver launch and synchronization overhead.

xTB and dxtb expose only single-system Hessian APIs. Their batch-128 rows are
therefore explicitly unavailable: executing 128 sequential complete Hessians
would not be a native batch comparison and was the source of the discarded
roughly 30-minute coordinate.

## Correctness

Every available retained sample was finite. Available rows passed the symmetry
and acoustic gates. Repeated batch-1 rows were bitwise repeatable. xTBloom and
dxtb slot zero passed comparison against the independent xTB Hessian:

| Engine/batch | Max absolute delta (Hartree/bohr²) | RMS delta (Hartree/bohr²) |
| --- | ---: | ---: |
| xTBloom CPU, 1 or 128 | 4.2184e-05 | 2.9762e-06 |
| xTBloom CUDA, 1 or 128 | 4.2184e-05 | 2.9762e-06 |
| dxtb CPU AD, 1 | 2.0133e-05 | 8.0746e-07 |

The compact JSONs retain all raw timing samples, correctness diagnostics,
commands, clean source and binary identities, environment metadata, and one
authenticated identity for every final Hessian. Reproducible dense payloads
are omitted according to the repository evidence-size policy; each compact
artifact authenticates its omitted raw JSON and matrices by SHA-256.

## Environment and identities

- Measured xTBloom revision: `2388c9e933188aa16f33c4916e6d58e2224f3d45`, clean.
- Final bounded-default implementation: `bfe51d69c7a99782802de60b3d19df6ec1bbfb01`.
- xTB source revision: `b31754bf3c7cccf8c242c469b03ae675e04bd608`, clean.
- dxtb source revision: `b529b5ddb75c0554274955082a189f9f88437cb2`, clean.
- xTB library SHA-256: `959ed711f85f3c84e5e9dffd15e1b49dfae1bd46783f544ce8756b5e148094dd`.
- xTBloom CPU library SHA-256: `b0c658b6be5a209aeb94673791f26d87171470587de77650562eddde4e15fc79`.
- xTBloom CUDA library SHA-256: `eebd3cafffc6585dc78e6f35e9852863294937a96fb91f2c26e2161b843d3b84`.
- CPU: AMD EPYC 7K62 48-Core Processor; affinity `0-15`.
- GPU: NVIDIA GeForce RTX 5090, driver 580.95.05, CUDA 12.9.1 toolkit.
- Thread boundary: `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`; every engine reported effective `nthreads=16`.
- CPU eigensolver provider: scipy-openblas32 LP64 runtime recorded in the
  compact artifact build metadata.

## Commands

All commands ran from the xTBloom source root using the Python 3.11 benchmark
environment. `${XTB_SOURCE_ROOT}` and `${DXTB_SOURCE_ROOT}` denote the clean
checkouts whose exact revisions are listed above. Common CPU commands used:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 taskset -c 0-15 \
  /tmp/xtbloom-issue358-py311/bin/python benchmarks/hessian.py \
  --engines xtb --batch-sizes 1,128 \
  --xtb-library /tmp/pr231-1fc8698-xtb-final/libxtb.so.6.7.1 \
  --xtb-source "${XTB_SOURCE_ROOT}" \
  --dxtb-source "${DXTB_SOURCE_ROOT}" --nthreads 16 \
  --max-serial-hessian-batch-size 1 --warmups 1 --repetitions 3 \
  --coordinate-timeout-seconds 60 --make-reference \
  --output-json /tmp/xtbloom-issue358-fast-2388c9e/xtb-reference.json \
  --compact-output-json /tmp/xtbloom-issue358-fast-2388c9e/xtb-reference-compact.json \
  --output-csv /tmp/xtbloom-issue358-fast-2388c9e/xtb-reference.csv

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 taskset -c 0-15 \
  /tmp/xtbloom-issue358-py311/bin/python benchmarks/hessian.py \
  --engines xtbloom-cpu --batch-sizes 1 \
  --library build/issue-358-corrected-cpu/libxtbloom.so \
  --reference-json /tmp/xtbloom-issue358-fast-2388c9e/xtb-reference.json \
  --xtb-source "${XTB_SOURCE_ROOT}" \
  --dxtb-source "${DXTB_SOURCE_ROOT}" --nthreads 16 \
  --displacement-chunk-size 128 --warmups 1 --repetitions 3 \
  --coordinate-timeout-seconds 60 --fail-on-correctness \
  --output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cpu-b1.json \
  --compact-output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cpu-b1-compact.json \
  --output-csv /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cpu-b1.csv

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 taskset -c 0-15 \
  /tmp/xtbloom-issue358-py311/bin/python benchmarks/hessian.py \
  --engines xtbloom-cpu --batch-sizes 128 \
  --library build/issue-358-corrected-cpu/libxtbloom.so \
  --reference-json /tmp/xtbloom-issue358-fast-2388c9e/xtb-reference.json \
  --xtb-source "${XTB_SOURCE_ROOT}" \
  --dxtb-source "${DXTB_SOURCE_ROOT}" --nthreads 16 \
  --displacement-chunk-size 128 --warmups 0 --repetitions 1 \
  --coordinate-timeout-seconds 300 --fail-on-correctness \
  --output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cpu-b128.json \
  --compact-output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cpu-b128-compact.json \
  --output-csv /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cpu-b128.csv
```

CUDA commands additionally used the actual toolkit runtime and visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
LD_LIBRARY_PATH=/group/software/cuda-12.9.1/targets/x86_64-linux/lib:$PWD/build/issue-358-corrected-cuda:/home/jzzeng/miniconda3/lib/python3.13/site-packages/scipy_openblas32/lib \
taskset -c 0-15 /tmp/xtbloom-issue358-py311/bin/python \
  benchmarks/hessian.py --engines xtbloom-cuda --batch-sizes 1 \
  --library build/issue-358-corrected-cuda/libxtbloom.so \
  --reference-json /tmp/xtbloom-issue358-fast-2388c9e/xtb-reference.json \
  --xtb-source "${XTB_SOURCE_ROOT}" \
  --dxtb-source "${DXTB_SOURCE_ROOT}" --nthreads 16 \
  --displacement-chunk-size 128 --warmups 1 --repetitions 3 \
  --coordinate-timeout-seconds 300 --fail-on-correctness \
  --output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cuda-b1-v2.json \
  --compact-output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cuda-b1-v2-compact.json \
  --output-csv /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cuda-b1-v2.csv

# The batch-128 command was identical except for these arguments:
--batch-sizes 128 --warmups 0 --repetitions 1 \
--output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cuda-b128.json \
--compact-output-json /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cuda-b128-compact.json \
--output-csv /tmp/xtbloom-issue358-fast-2388c9e/xtbloom-cuda-b128.csv
```

dxtb CPU AD used the same CPU affinity/thread environment, `--warmups 0
--repetitions 1 --coordinate-timeout-seconds 300`, and separate batch-1 and
batch-128 invocations recorded verbatim in their compact JSONs.

## Validation

- Focused benchmark/evidence tests after the final bounded-default change:
  `28/28` passed.
- Full prek: passed.
- `uv lock --check`: passed.
- `git diff --check`: passed.
- CPU and CUDA registered benchmark self-tests: `1/1` passed in each build.
- Full shared LP64 CPU CTest at the measured benchmark head: `53/53` passed.
- Non-editable Python suite at the measured benchmark head: `321 passed, 67 skipped`.
- Full real-GPU CUDA CTest: `127/127` passed with `CUDA_VISIBLE_DEVICES=0`,
  including host/device/mixed public conformance.

The final commit changes only default benchmark sampling/timeout policy and its
hardware-free tests; numerical implementation and measured binaries are
unchanged.

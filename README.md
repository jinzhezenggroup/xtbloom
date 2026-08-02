# gpuxtb

gpuxtb is a new C++ library for high-throughput GFN-xTB energy and force inference on CPUs and
GPUs. GFN2-xTB is the first implementation target. The public C ABI is designed for ragged batches,
host or CUDA device pointers, and external point charges that participate in SCC iterations.

> [!IMPORTANT]
> The repository currently contains the runtime and API scaffold. The GFN2-xTB physics kernels are
> not implemented yet, and `gpuxtb_compute` returns `GPUXTB_STATUS_NOT_IMPLEMENTED` deliberately.

## Build

```console
cmake -S . -B build -G Ninja -DGPUXTB_ENABLE_CUDA=AUTO
cmake --build build
ctest --test-dir build --output-on-failure
```

`GPUXTB_ENABLE_CUDA` accepts `AUTO`, `ON`, or `OFF`. `AUTO` enables CUDA when a CUDA compiler is
available and otherwise produces a CPU-only library. The ROCm enum is reserved in the ABI, but the
backend is not implemented yet.

On the current development machine, CUDA 12.9.1 can be selected explicitly with:

```console
cmake -S . -B build-cuda \
  -DGPUXTB_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

## Design goals

- Match established GFN2-xTB energies and analytic forces before performance tuning.
- Make batched inference the primary execution model, rather than a wrapper around serial calls.
- Accept caller-owned host and CUDA memory through the same stable C API.
- Include external point charges in SCC and return forces on both QM atoms and point charges.
- Reuse workspaces and immutable device-resident parameters on steady-state inference paths.
- Maintain an optimized CPU backend and a backend boundary suitable for a future ROCm port.

The detailed design and correctness/performance strategy are in [docs/architecture.md](docs/architecture.md).
Implementation progress is tracked in [GitHub Epic #1](https://github.com/njzjz/gpuxtb/issues/1)
and its sub-issues.

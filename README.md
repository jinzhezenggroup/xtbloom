# gpuxtb

gpuxtb is a new C++ library for high-throughput GFN-xTB energy and force inference on CPUs and
GPUs. GFN2-xTB is the first implementation target. The public C ABI is designed for ragged batches,
host or CUDA device pointers, and external point charges that participate in SCC iterations.

gpuxtb is licensed under GPL-3.0-or-later with a narrowly scoped
[CUDA and Intel MKL additional permission](CUDA_MKL_LINKING_EXCEPTION).
Redistributed tblite, dftd4, mctc-lib, and numerical-oracle material remains
under the separate terms documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with pinned provenance.

> [!IMPORTANT]
> Restricted and unrestricted GFN2-xTB inference is available through `gpuxtb_compute` on CPU and
> CUDA, including energies, analytic QM forces, atomic charges, external point charges in SCC,
> and point-charge forces. The CUDA backend accepts host, CUDA-device, or mixed input/output
> descriptors for ragged batches. Its fixed-topology
> runtime reuses device arenas across changed geometries and includes explicit point charges and
> caller-supplied periodic `b + A*q` operators in SCC. The public call is synchronous and rejects
> CUDA stream capture; a future asynchronous ABI remains a separate extension. GFN1-xTB and ROCm
> remain reserved but not implemented.

## Build

```console
cmake -S . -B build -G Ninja -DGPUXTB_ENABLE_CUDA=AUTO
cmake --build build
ctest --test-dir build --output-on-failure
```

`GPUXTB_ENABLE_CUDA` accepts `AUTO`, `ON`, or `OFF`. `AUTO` enables CUDA when a CUDA compiler is
available and otherwise produces a CPU-only library. The ROCm enum is reserved in the ABI, but the
backend is not implemented yet.

CPU contexts use `gpuxtb_context_options_t::cpu_threads` as the outer batch-parallelism ceiling.
Set it to one for deterministic serial execution, to a positive value for that many available CPU
workers, or to zero for an affinity-aware automatic choice (currently capped at 64). The CPU
runtime keeps its workers and numerical staging for the lifetime of the context; the selected LP64
BLAS remains one-thread-per-worker so batch parallelism does not create nested oversubscription.

On the current development machine, CUDA 12.9.1 can be selected explicitly with:

```console
cmake -S . -B build-cuda \
  -DGPUXTB_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
```

A CUDA-enabled install does not require a CUDA toolkit merely to link a CMake
consumer. At execution time, the backend dynamically resolves compatible CUDA
12 host runtime/math libraries and the system NVIDIA driver. Install the
`cuda12` Python extra to obtain the matching, separately distributed
`nvidia-*` runtime packages; the package preloads their SONAMEs.
gpuxtb does not embed build-host CUDA paths in the installed library's RPATH.
For a native deployment whose compatible CUDA host libraries are outside the
system loader path, expose that library directory through `LD_LIBRARY_PATH` or
preload the SONAMEs before loading libgpuxtb.
The device link explicitly disables the unused static `cudadevrt` input.
Compiler-inserted NVIDIA libdevice code may still be present and remains under
NVIDIA's terms through the project's additional permission.

## Python package

A Python package wrapping the public C ABI is provided under `python/` and is
packaged with scikit-build-core:

```console
pip install .             # CPU runtime
pip install ".[cuda12]"  # CUDA 12 host runtimes
```

It offers a tblite-like single-molecule interface (`gpuxtb.Calculator`,
`gpuxtb.Structure`, `gpuxtb.Result`), native ragged-batch inference
(`gpuxtb.BatchCalculator`), net charge and spin multiplicity (unpaired
electrons / spin channels), an ASE calculator (`gpuxtb.ase.GPUxtb`), and a
dpdata driver plugin registered as `gpuxtb`. Full details are in
[python/README.md](python/README.md).

## Development

Contributions must pass the repository pre-commit hooks (trailing whitespace,
end-of-file, YAML/JSON/TOML checks, ruff lint and format for Python, and
clang-format for C/C++/CUDA). Install [pre-commit](https://pre-commit.com), or
prek (its Rust reimplementation), and run:

```console
pre-commit install        # or: prek install
pre-commit run --all-files  # or: prek run --all-files
```

Pull requests run the same hooks in read-only CI. If a hook reports an
autofixable change, run the command locally and commit the resulting diff.

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

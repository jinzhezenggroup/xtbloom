# User guide

gpuxtb is a single-point GFN2-xTB inference library. It is most useful when an
application needs many independent molecules, reusable native state, direct
CUDA integration, or a stable C deployment boundary.

## Installation paths

### Python

gpuxtb is being prepared for its first PyPI release. Published releases use:

```console
python -m pip install gpuxtb
python -m pip install "gpuxtb[ase,dpdata]"  # optional integrations
python -m pip install "gpuxtb[cuda12]"      # optional CUDA 12 host libraries
```

See the [Python guide](python.md) for runtime requirements and examples.

### C and C++

Build a shared or static native SDK with CMake 3.24 or newer:

```console
cmake -S . -B build/release -G Ninja \
  -DGPUXTB_ENABLE_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
cmake --install build/release --prefix "$PWD/build/install"
```

`GPUXTB_ENABLE_CUDA` accepts `OFF`, `ON`, or `AUTO`. Use `ON` when a CUDA build
is required; `AUTO` is convenient for local exploration but can legitimately
produce a CPU-only build when `nvcc` is absent.

CPU inference requires one monolithic LP64 LAPACKE+CBLAS runtime that can be
opened dynamically. gpuxtb can discover a compatible system OpenBLAS or MKL,
or the build can record an absolute provider with
`GPUXTB_CPU_LINALG_LIBRARY`. The library still loads without that provider,
but CPU eigensolver-backed inference then reports a diagnostic rather than
silently using an incompatible BLAS.

Continue with the [C and C++ guide](c-api.md).

## Units and result meaning

The native ABI uses IEEE binary64 and atomic units:

| Quantity | Native and high-level Python unit |
| --- | --- |
| Positions | bohr |
| Energy | Hartree |
| Forces | Hartree/bohr |
| Charge | elementary-charge units |
| Point-charge screening `gamma` | Hartree |

The high-level Python API accepts `electronic_temperature` in kelvin and
converts it internally. The C ABI accepts the energy scale `k_B T` in Hartree.

At finite electronic temperature, `energy` is the variational electronic
Helmholtz free energy, including the Fermi-occupation entropy term. `forces`
is its negative coordinate derivative. Python also exposes `gradient`, which
is exactly `-forces`.

When periodic `b` or `A` operators are supplied, gpuxtb holds them fixed while
differentiating. The result flag
`GPUXTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES` tells native
callers that `db/dR` and `dA/dR` are not included.

## Backends and memory

- `CPU` runs systems in parallel at the outer batch layer and keeps the BLAS
  provider single-threaded within each worker.
- `CUDA` runs the same model equations and accepts host, CUDA-device, or mixed
  descriptors through the C ABI.
- `AUTO` prefers CUDA when the backend is compiled, a compatible device is
  present, and initialization succeeds; otherwise it selects CPU.

The high-level Python interface always submits host NumPy arrays. It can still
execute on CUDA, but the native runtime stages those arrays. Applications that
already own device memory should use the C ABI to avoid staging copies.

Public compute is synchronous. Active CUDA stream capture is rejected; an
asynchronous public ABI is not currently implemented.

## Failure behavior

gpuxtb separates request failures from per-system numerical failures.

- Invalid descriptors, unsupported settings, allocation failures, and other
  call-level errors return a failing `gpuxtb_status_t`. Before caller-output
  commit, outputs and result flags remain unchanged.
- SCC nonconvergence or eigensolver failure for one molecule is a data-level
  result. The call succeeds, diagnostics identify the failed molecule, and all
  requested floating-point slices for that molecule are filled with quiet
  NaNs. Successful peers remain available.

Python `BatchCalculator.compute()` preserves this peer-local behavior by
default. Call `BatchResult.raise_for_status()` or pass
`raise_on_failure=True` for exception-oriented control flow.

## Scope and limitations

Only GFN2-xTB is implemented. GFN1-xTB and ROCm are reserved so the ABI can
grow without reusing numeric tags, but requesting them returns an unsupported
or not-implemented status.

gpuxtb has no lattice input. Its periodic charge-response API accepts fields
computed by another electrostatics program; it does not make the QM calculation
periodic by itself. Geometry optimization, molecular dynamics, solvation,
vibrational analysis, and Hessians belong in calling applications or other xTB
implementations.

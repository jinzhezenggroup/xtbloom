# xTBloom Installation Reference

Use this reference for user installations. These instructions describe the current published and source-build boundaries; verify newer release documentation if the installed version differs.

## Choose the Artifact

xTBloom is published on PyPI. Choose among:

- install the reviewed PyPI wheel for ordinary Python use;
- build the Python package from source for development or an unsupported target;
- consume a native CMake install.

Do not substitute an unreviewed package with a similar name.

## PyPI Python Installation

For an agent-owned one-off program, avoid mutating the current environment or
requiring xTBloom to be preinstalled. Add PEP 723 metadata to the program:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["xtbloom>=0.1.1"]
# ///
```

Then let uv create the temporary environment using the user's configured index:

```bash
uv run --script your_program.py
```

If the user explicitly wants a persistent environment, install only the base
package and extras needed by that environment:

```bash
pip install xtbloom
pip install 'xtbloom[cuda12]'
pip install 'xtbloom[ase]'
pip install 'xtbloom[dpdata]'
```

Combine extras, such as `xtbloom[cuda12,ase]`, only when the same workflow
actually needs both.

Published Linux, macOS, and Windows wheels include a reviewed private LP64
OpenBLAS provider. Linux x86_64 and aarch64 wheels also include the CUDA backend.
The `cuda12` extra supplies supported CUDA 12 user-space libraries; it cannot
install a GPU or NVIDIA driver.

## Source-Checkout Python Installation

Python 3.10 or newer is supported by the installed package. Repository test configurations require Python 3.11 or newer. A source build needs CMake 3.24 or newer, Ninja for the documented commands, C11/C++17-capable compilers, and complete Git tag history except for the documented exact-tag shallow build.

Use the locked, non-editable project environment:

```bash
uv sync --locked --no-editable --no-default-groups \
  --reinstall-package xtbloom
```

Run applications with `uv run --no-sync` or activate `.venv`. The default native selection is `XTBLOOM_ENABLE_CUDA=AUTO`: it builds CUDA when a usable compiler is found and otherwise builds CPU-only.

Require a CPU-only build explicitly when CUDA must not be compiled:

```bash
XTBLOOM_ENABLE_CUDA=OFF \
  uv sync --locked --no-editable --no-default-groups \
  --reinstall-package xtbloom
```

Require CUDA compilation, select the actual target architecture, and install the supported CUDA 12 host-provider packages when the system does not supply them:

```bash
XTBLOOM_ENABLE_CUDA=ON \
XTBLOOM_CUDA_ARCHITECTURES=<actual-compute-capability> \
  uv sync --locked --no-editable --no-default-groups \
  --extra cuda12 --reinstall-package xtbloom
```

The CUDA source build currently targets 64-bit Linux ELF on x86_64 and aarch64. CUDA Toolkit 12.9 is the qualified baseline. `nvcc` must be on `PATH` or selected with `CUDACXX`. The `cuda12` extra supplies supported cudart, cuBLAS, cuSOLVER, cuSPARSE, and nvJitLink host packages; it does not install `nvcc`, a GPU, or the NVIDIA driver.

## CPU Linear Algebra

CPU inference needs one dynamically loadable, monolithic LP64 runtime exporting LAPACKE and CBLAS plus provider-local thread control. A split BLAS/LAPACK pair, an ILP64 provider, or a static archive does not satisfy this contract.

Source builds auto-discover a compatible system runtime. When discovery fails, pass an absolute reviewed provider path and reinstall:

```bash
CMAKE_ARGS="-DXTBLOOM_CPU_LINALG_LIBRARY=/absolute/path/to/provider.so" \
XTBLOOM_ENABLE_CUDA=OFF \
  uv sync --locked --no-editable --no-default-groups \
  --reinstall-package xtbloom
```

Published desktop wheels contain their reviewed private LP64 OpenBLAS provider. Do not install `scipy-openblas32` as an xTBloom runtime dependency; it is a build-only input to those wheels.

## Native-Library Resolution

The Python binding resolves `libxtbloom` in this order:

1. the explicit `XTBLOOM_LIBRARY` environment variable;
2. the native library bundled beside the installed Python package; and
3. the platform loader's system lookup for `xtbloom`.

An `XTBLOOM_LIBRARY` override must be ABI-compatible with the Python package. Its dependent libraries remain the deployer's responsibility. Prefer the bundled library for ordinary wheel use, and remove stale development overrides before diagnosing a packaged installation.

## Runtime Backend Choice

- `backend="cpu"` requires CPU context creation and fails if its provider is unusable.
- `backend="cuda"` requires a compiled CUDA backend, a real compatible NVIDIA GPU and driver, and the matching host libraries.
- `backend="auto"` prefers CUDA and may fall back to CPU. Use it only when that fallback is intended.

Importing the Python package does not create a backend. CUDA compilation, `nvidia-smi`, or presence of CUDA libraries does not prove that xTBloom can create and execute a CUDA context.

## Supported Scope

xTBloom implements GFN1-xTB on CPU and GFN2-xTB on CPU and CUDA. ROCm,
lattice/PBC inputs, solvation, native optimization, molecular dynamics, and
Hessians are not supported.

Authoritative online sources:

- <https://pypi.org/project/xtbloom/>
- <https://github.com/jinzhezenggroup/xtbloom/blob/main/docs/user-guide/index.md>
- <https://github.com/jinzhezenggroup/xtbloom/blob/main/python/README.md>

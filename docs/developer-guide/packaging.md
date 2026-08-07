# Packaging, dependencies, and licensing

gpuxtb has three related distribution surfaces:

1. native CMake installs for C and C++ consumers;
2. source distributions containing build inputs and provenance; and
3. Python wheels containing the Python package and one native `libgpuxtb`.

Changing one surface does not prove the other two remain correct.

## PyPI documentation

The repository root [`README.md`](../../README.md) is the GitHub-facing overview
for Python and native users. `pyproject.toml` deliberately points
`project.readme` at [`python/README.md`](../../python/README.md), so PyPI renders
a Python-only package page without native developer/test instructions.

Use absolute repository URLs in the PyPI-facing Markdown. Relative links are
resolved by PyPI rather than GitHub and otherwise lead to the wrong location.
Build metadata and inspect the generated `METADATA` payload whenever the
selected README or project URLs change.

## Native install

The install exports `gpuxtb::gpuxtb`, the public header, version/config files,
licenses, third-party notices, and applicable provenance manifests. Validate
both shared and static consumers when CMake export or dependency behavior
changes.

The shared library exports only versioned `gpuxtb_*` C symbols on Linux. CPU
BLAS/LAPACK and CUDA host providers are opened dynamically; they must not
become accidental `DT_NEEDED` dependencies merely to simplify discovery.

## Python wheels

scikit-build-core builds `libgpuxtb` through CMake and installs it under the
Python package. The ctypes binding is independent of the CPython extension ABI,
so one platform wheel can serve supported Python 3 versions.

Linux CUDA wheels contain compiled gpuxtb device code but do not bundle CUDA
host shared libraries or the NVIDIA driver. The optional `cuda12` extra installs
supported host providers separately from PyPI. CPU-only installations remain
usable without the proprietary stack.

Wheel checks cover size, license payload, native symbol and dynamic-dependency
policy, bundled-library discovery, and installed inference. Hosted wheel jobs
without an NVIDIA driver do not count as real CUDA runtime validation.

## Source distributions

Build and inspect a source archive with:

```console
uv build --sdist --out-dir build/dist-license
python3 tools/licensing/check_licenses.py --source-root . \
  build/dist-license/*.tar.gz
```

The archive must retain the license, CUDA/MKL additional permission,
third-party notices, required license texts, parameter and source provenance,
the pinned implib generator source, both README roles, and maintained
documentation.

## External material

Every new dependency, copied source, generated dataset, parameter table,
download, CI action, native link, or distributed payload needs an inventory:

- upstream project and immutable revision/version;
- retrieval or deterministic generation procedure;
- source and output digests;
- license and notice source;
- local destination; and
- build-only, runtime, install, sdist, and wheel boundaries.

Update [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md), `LICENSES/`,
manifests, packaging rules, and automated checks as applicable. Dynamic loading
does not itself decide license compatibility. CUDA and MKL use the precise
additional permission in
[`CUDA_MKL_LINKING_EXCEPTION`](../../CUDA_MKL_LINKING_EXCEPTION); provider code
remains under vendor terms.

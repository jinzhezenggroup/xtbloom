# Packaging, dependencies, and licensing

xTBloom has three related distribution surfaces:

1. native CMake installs for C and C++ consumers;
2. source distributions containing build inputs and provenance; and
3. Python wheels containing the Python package and one native `libxtbloom`.

Changing one surface does not prove the other two remain correct.

## Authoritative product version

Release tags use the strict form `vMAJOR.MINOR.PATCH`, with each component
written canonically (`0` or a non-zero digit followed by digits). The latest
tag reachable from `HEAD` is the product version until another release tag is
created; setuptools-scm's `only-version` scheme deliberately ignores commit
distance, object IDs, and worktree dirtiness. The tag without its leading `v`
feeds Python metadata, Python `__version__`, `xtbloom_version_string()`, CMake
`project(VERSION)`, target filenames, the installed CMake package version, and
the generated public version macros.

`XTBLOOM_API_VERSION`, the Linux symbol-version node, and the ELF `SOVERSION`
are ABI contracts, not product versions. They change only after an explicit ABI
decision; a product tag never changes them automatically.

Native Git checkouts read the nearest reachable strict tag from complete tag
history. Python package metadata comes directly from scikit-build-core's
built-in setuptools-scm provider; CMake consumes `SKBUILD_PROJECT_VERSION` in
wheel builds and resolves the same tag itself for native builds. Native CMake
configuration rejects shallow history; automated Python builds also fetch
complete history so branch builds can find the true nearest tag. An exact-tag
Python build may use that tag from a shallow checkout, following setuptools-scm
semantics. Repositories without a reachable strict tag fail configuration. An
sdist consumes the version frozen from that tag into `PKG-INFO`, while a Git
archive consumes the nearest tag expanded into `.git_archival.txt`. The `v*`
namespace is reserved for product versions, so a nearer malformed tag such as
`v1.2` is rejected instead of being silently skipped. There is intentionally no
fallback version.

Release automation must check out complete history and require a clean exact
tag. Wheel jobs build directly from that checkout and therefore resolve the
tag themselves. The independent sdist job verifies the frozen `PKG-INFO` path
used by unpacked source archives. Validate that the sdist/wheel metadata,
generated public header, CMake package, C API string, and Python `__version__`
agree before publishing artifacts.

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

The install exports `xtbloom::xtbloom`, the public header, version/config files,
licenses, third-party notices, and applicable provenance manifests. Validate
both shared and static consumers when CMake export or dependency behavior
changes.

The shared library exports only versioned `xtbloom_*` C symbols on Linux. CPU
BLAS/LAPACK and CUDA host providers are opened dynamically; they must not
become accidental `DT_NEEDED` dependencies merely to simplify discovery.

## Python wheels

scikit-build-core builds `libxtbloom` through CMake and installs it under the
Python package. The ctypes binding is independent of the CPython extension ABI,
so one platform wheel can serve supported Python 3 versions.

Linux CUDA wheels contain compiled xTBloom device code but do not bundle CUDA
host shared libraries or the NVIDIA driver. The optional `cuda12` extra installs
supported host providers separately from PyPI. CPU-only installations remain
usable without the proprietary stack.

Linux x86_64/aarch64 wheels do bundle one private LP64 OpenBLAS provider. The
upstream `scipy-openblas32` distribution is deliberately build-only: an
environment-gated scikit-build-core override installs the exact reviewed
version only for cibuildwheel's audited Linux wheel builds, while a non-default
uv dependency group retains its PyPI hashes in `uv.lock`. It must never appear
in project dependencies, extras, editable builds, or wheel `METADATA`.

CMake validates the installed distribution version, license, architecture, and
complete ELF inventory through `python/ci/resolve-openblas-wheel.py` without
importing the package. A private shim retains the only `DT_NEEDED` edge.
`auditwheel repair` then vendors and collision-renames OpenBLAS and its
architecture-specific GCC runtime closure. `python/ci/inspect-openblas-wheel.py`
checks the final metadata, exact cohort names, ELF machine, symbols, SONAMEs,
RPATHs, and dependency graph. `libxtbloom` itself must remain free of OpenBLAS,
libgfortran, libquadmath, and shim `DT_NEEDED` entries.

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

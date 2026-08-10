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
Python package. The ctypes binding and LibTorch Stable ABI integration are not
CPython extension modules, so one `py3-none-<platform>` wheel serves supported
Python 3 versions. Free-threaded `cp313t`/`cp314t` interpreters therefore need
runtime validation of that same wheel, not separately published wheel bytes.

Linux CUDA wheels contain compiled xTBloom device code but do not bundle CUDA
host shared libraries or the NVIDIA driver. The optional `cuda12` extra installs
supported host providers separately from PyPI. CPU-only installations remain
usable without the proprietary stack.

Linux x86_64/aarch64, macOS x86_64/arm64, and Windows AMD64/ARM64 wheels bundle
one private LP64 OpenBLAS provider. The upstream `scipy-openblas32`
distribution is deliberately build-only: an environment-gated
scikit-build-core override installs the exact reviewed version only for
cibuildwheel's audited native wheel builds, while a non-default uv dependency
group retains every platform artifact hash in `uv.lock`. It must never appear
in project dependencies, extras, editable builds, test environments, or wheel
`METADATA`.

CMake validates the installed distribution version, target, exact packaged
license, and complete native-library inventory through
`python/ci/resolve-openblas-wheel.py` without importing the package. The source
manifest pins every upstream wheel and native payload hash plus exact local
copies of each platform-specific packaged-license variant.

On Linux, a private shim retains the only `DT_NEEDED` edge. `auditwheel repair`
then vendors and collision-renames OpenBLAS and its architecture-specific GCC
runtime closure, and `libxtbloom` loads the shim in a new glibc link-map
namespace. On macOS, every OpenBLAS/libgfortran/libquadmath/libgcc image gets a
content-hash-qualified private filename and LC_ID; all intra-cohort load
commands are rewritten before every derived image is ad-hoc signed. On
Windows, the self-contained provider DLL gets a content-hash-qualified private
filename. macOS loads by canonical absolute sibling path and verifies the
dispatch image path; Windows loads by absolute sibling path and verifies every
dispatch symbol against the returned module handle. Desktop providers expose
only global thread control, so their already-private image is fixed to one
thread once during thread-safe initialization. System providers still require
worker-local thread control.

`python/ci/inspect-openblas-wheel.py` checks final metadata, exact cohort names,
machine architecture, required symbols/exports, private IDs and dependency
paths, signatures, and native dependency closure. `libxtbloom` itself must
remain free of hard OpenBLAS, libgfortran, libquadmath, and shim dependencies.

The compiled Torch extension is included in Linux wheels, macOS arm64 wheels,
and Windows AMD64 wheels. The isolated build uses a non-installed platform stub
with the real runtime identity: `libtorch_cpu.so`,
`@rpath/libtorch_cpu.dylib`, or `torch_cpu.dll` plus its generated import
library. Wheel repair excludes that unresolved external runtime edge, and
payload/linkage checks prove that no Torch library or stub is bundled. The
installed-wheel tests import the separately installed Torch 2.13 runtime first,
then run CPU forward, forces, and autograd through the compiled op.

PyTorch 2.10+ publishes no supported PyPI runtime wheel for macOS x86_64 or
Windows ARM64. xTBloom therefore does not ship an untested extension in those
two wheel cohorts; their payload checks require its absence. Pyodide likewise
has no LibTorch runtime and contains no Torch extension.

The Pyodide `cp314-pyodide_wasm32` wheel targets Pyodide 314.x and its stable
`pyemscripten_2026_0_wasm32` ABI, then is smoke-tested as a CI-only artifact.
PyPI accepts this platform tag, but xTBloom excludes the wheel from the PyPI
artifact prefix until the Python wheel has a production WebAssembly eigensolver
path; the existing Web demo uses its separate preloaded LAPACK design.
Windows/macOS wheels must pass installed GFN2 inference, missing-provider,
concurrency, and host numerical-coexistence tests before being release-eligible.

Final GitHub Releases publish the validated sdist and release-eligible native
wheels through PyPI Trusted Publishing. The upload job runs only for a
non-prerelease published release, uses the protected `pypi` environment, and
receives job-local `id-token: write`; build jobs retain read-only permissions.

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

# Vendored LibTorch Stable ABI headers

`src/bindings/torch/xtbloom_torch_ext.cpp` is compiled against the LibTorch Stable ABI. This
directory holds the exact `#include` closure of the stable-ABI headers the
extension uses, vendored from the PyPI `torch` wheel, together with the
`aoti_torch_*` / `torch_library_impl` / `torch_get_mutable_data_ptr` symbol
list needed to build the extension **without downloading torch**.

## Why vendor instead of a build-time torch dependency

A plain compiled torch extension normally adds `torch` to
`[build-system].requires`, so every PEP-517-isolated wheel build downloads the
~500 MB torch wheel just to compile. Vendoring the ~50 header files (less than
a megabyte) removes that download entirely. The extension still needs torch at
runtime — that is inherent to being a torch extension — but the build never
touches torch.

The extension also links a *build-time-only* stub `libtorch_cpu.so` instead of
the real one. The stub has the same `DT_NEEDED` name (`libtorch_cpu.so`), so
the shipped `libxtbloom_torch_ext.so` behaves exactly like one built against
real torch: when `torch.ops.load_library` loads it, the dependency resolves to
the torch the end user already imported. The stub is generated from
`aoti_symbols.txt` at configure time and is never installed or shipped.

## Provenance

- Upstream: https://github.com/pytorch/pytorch (BSD-3-Clause,
  `LICENSES/BSD-3-Clause.txt`)
- Pinned release: `torch 2.12.1` (see `manifest.json`, whose `files` entries
  pin per-file sha256).
- Source path inside the wheel: `torch/include`.
- The vendored set is the transitive `torch/`-prefixed include closure of the
  five `ROOT_HEADERS` in `tools/torch_stable_vendor.py`. The stable set is
  deliberately self-contained (it does not pull in c10/ATen), which is what
  keeps the vendor this small.
- The runtime floor is unchanged: `#define TORCH_TARGET_VERSION` in
  `xtbloom_torch_ext.cpp` still limits the emitted symbol set to torch 2.10, and
  the stable C ABI guarantees `libtorch_cpu.so` exports these symbols on every
  torch >= 2.10.

## Regenerating

```bash
python3 tools/torch_stable_vendor.py check --out cmake/3rdparty/torch-stable
```

followed by a regenerate from an installed torch, then rebuild the extension
symbol list from the supported compiled objects:

```bash
python3 tools/torch_stable_vendor.py generate \
  --torch-include <torch/include> --torch-version <x.y.z> \
  --out cmake/3rdparty/torch-stable
# Compile src/bindings/torch/xtbloom_torch_ext.cpp in every supported
# compiler/instrumentation mode, then pass each object to form their union:
python3 tools/torch_stable_vendor.py symbols \
  --extension-object <release-ext.o> \
  --extension-object <coverage-ext.o> \
  --out cmake/3rdparty/torch-stable/aoti_symbols.txt
```

The extension link uses each platform's strict undefined-symbol policy
(`-z defs` on ELF, `-undefined error` on Mach-O, and the ordinary MSVC import
link on PE), so a header/version change that introduces an unstubbed Torch
symbol fails the build loudly instead of producing a broken extension.

## Platform scope

The build-time stub supports Linux ELF (x86_64/aarch64), macOS arm64, and
Windows AMD64. Linux uses the real `libtorch_cpu.so` SONAME. macOS uses the
official `@rpath/libtorch_cpu.dylib` install name without adding an LC_RPATH to
the extension. Windows compiles the generated C stubs as a private
`torch_cpu.dll`; CMake produces the architecture-correct `torch_cpu.lib` that
the extension consumes. None of those build artifacts is installed.

The corresponding wheels contain and test the extension against a separately
installed Torch 2.13 runtime. PyTorch 2.10+ publishes no supported PyPI runtime
wheel for macOS x86_64 or Windows ARM64, so those xTBloom wheels deliberately
omit the extension rather than shipping an unvalidated plugin. Pyodide also
omits it because no LibTorch runtime exists there. The Python integration
reports the missing optional extension instead of silently copying tensors.

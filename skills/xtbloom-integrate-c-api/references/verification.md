# Native consumer verification

## Build the bundled template

Copy `assets/installed-consumer/` to a temporary application directory, then
configure it against the selected xTBloom install:

```console
cmake -S installed-consumer -B installed-consumer/build -G Ninja \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/xtbloom-install
cmake --build installed-consumer/build --parallel
installed-consumer/build/xtbloom_c_consumer cpu
```

The executable checks descriptor initialization, real GFN2 inference, required
diagnostics, explicit CPU selection, and one compatible `FRESH` to `WARM`
transition.

For a CUDA-enabled install, run on a real compatible NVIDIA GPU:

```console
installed-consumer/build/xtbloom_c_consumer cuda
```

This intentionally uses host descriptors with a CUDA context, proving public
host staging and publication without requiring the consumer to compile CUDA.
Failure because CUDA is unavailable is not a CUDA pass.

## Integration checklist

- Build the consumer as C11 with warnings enabled.
- If C++ consumers are supported, compile the public header from at least one
  C++17 translation unit and keep the calls within the C ABI.
- Test the installed package, not a source-tree include path or uninstalled
  library artifact.
- Require the intended backend and inspect the resolved backend.
- Check a successful system and intentionally exercise a per-system numerical
  failure if the application has recovery logic.
- Confirm undersized, missing, wrongly tagged, overlapping, and non-finite
  descriptors fail before outputs are changed.
- Cover empty optional fields and a ragged mixed-size batch when batching is
  part of the application.
- For CUDA device buffers, cover host-only, device-only, and mixed descriptors
  on the exact device selected by the context.
- Verify custom-stream synchronous behavior, active-capture rejection, and
  caller-device restoration when the application supplies streams or changes
  devices.
- Verify `WARM` success after a compatible converged call and rejection for a
  first call, changed topology/policy, or failed predecessor.
- If using asynchronous requests, use `xtbloom_plan_compute_enqueue`; verify
  that context-level `xtbloom_compute_enqueue` remains unavailable, then test
  repeated changed-geometry calls, FRESH-only admission, request lifetime,
  pending reuse rejection, wait/query status, and teardown order.

## Reporting

Record the exact install prefix, xTBloom version, compiler, backend, GPU and
driver/toolkit when applicable, commands, pass/fail counts, and unavailable or
skipped rows. Do not describe an AUTO CPU fallback, compile-only CUDA build, or
unexecuted GPU test as CUDA validation.

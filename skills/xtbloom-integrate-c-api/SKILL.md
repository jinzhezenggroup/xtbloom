---
name: xtbloom-integrate-c-api
description: Integrate xTBloom's stable public C ABI into installed C or C++ applications, including CMake package linking, initialized descriptors, caller-owned host or CUDA buffers, per-system diagnostics, synchronous CUDA behavior, and strict warm starts. Use when adding, reviewing, packaging, or troubleshooting native consumers that include xtbloom/xtbloom.h or call xtbloom context, compute, or plan APIs.
---

# Integrate the xTBloom C API

Build against the installed `xtbloom::xtbloom` target and treat the public
header as a borrowed-buffer ABI, not as a C++ object API.

## Use the Native Toolchain

Use CMake and the target project's compiler environment for native consumers;
uv does not replace an installed xTBloom CMake package, headers, linker inputs,
or a CUDA toolkit. Do not invent a temporary native SDK from a Python wheel or
an unrelated Python environment. If the task instead needs a Python helper,
route it to the matching Python skill and use its ephemeral `uv run` guidance.

## Workflow

1. Inspect the installed `xtbloom/xtbloom.h` and CMake package before coding.
   Use [references/c-api-contract.md](references/c-api-contract.md) as the
   baseline contract, but let the installed header govern additive fields in a
   newer release.
2. Copy `assets/installed-consumer/` into the target project when a minimal
   executable is useful. Keep its descriptor initializers, two-level status
   checks, explicit backend selection, and strict `FRESH` then `WARM` probe.
3. Convert application data to the exact public types and atomic units. Pack a
   ragged batch with signed 64-bit offsets; do not cast arbitrary application
   arrays to the ABI types without proving dtype, layout, extent, and lifetime.
4. Call every structure initializer with the caller's `sizeof(struct)` before
   overriding fields. Bind all required diagnostics even when requesting only
   one floating-point property.
5. Select `CPU` or `CUDA` when the application requires a backend. Use `AUTO`
   only when CUDA-to-CPU fallback is acceptable and inspect the resolved
   backend after context creation.
6. Keep every borrowed input and output alive for the complete operation. A
   synchronous call releases its views on return. For asynchronous work, use
   the connected fixed-topology CUDA `xtbloom_plan_compute_enqueue` path and
   retain device inputs plus all outputs until the request becomes complete;
   the context-level `xtbloom_compute_enqueue` symbol is not implemented on
   CUDA in the current API.
7. Separate call-level failure from per-system numerical failure. Read
   `xtbloom_get_last_error()` immediately after a failing synchronous API call,
   then inspect every `per_system_status` after a successful batch call.
8. Add reuse or warm starts only after the simple synchronous path is correct.
   `WARM` is strict state consumption, never an optimization hint with fallback.
9. Run the applicable checks in
   [references/verification.md](references/verification.md) and report CPU,
   CUDA, and unavailable configurations separately.

## Non-negotiable boundaries

- Use GFN1-xTB only with a CPU context. GFN2-xTB supports CPU and CUDA; ROCm
  remains reserved and unsupported.
- Use binary64 and atomic units: bohr, Hartree, Hartree/bohr, and `k_B T` in
  Hartree at the C boundary.
- Never omit `scc_iterations`, `scc_converged`, or `per_system_status` for a
  nonempty batch.
- Never infer whole-batch success from the return value alone. A successful
  call may contain peer-local SCC or eigensolver failures and NaN property
  slices for only those systems.
- Never free, resize, retag, or overlap borrowed buffers while xTBloom may use
  them.
- Never silently replace an explicitly requested CUDA backend or strict warm
  start with a CPU or fresh calculation.

## Resources

- `references/c-api-contract.md`: types, units, ownership, failures, CUDA, and
  warm-state semantics.
- `references/verification.md`: installed-consumer and behavior checks.
- `assets/installed-consumer/`: minimal C11 CMake consumer to copy and adapt.

---
name: gpuxtb-evolve-c-abi
description: Implement and review changes to gpuxtb's stable public C ABI across headers, initialization, validation, CPU/CUDA execution, Python ctypes, symbols, packaging, and install consumers. Use whenever changing `include/gpuxtb/gpuxtb.h`, public structures or tags, `gpuxtb_*` symbols, descriptors, result semantics, ABI suffixes, or their Python mirror.
---

# Evolve the gpuxtb C ABI

Preserve existing callers while carrying one public contract through every language and backend boundary. Read `AGENTS.md`, `include/gpuxtb/gpuxtb.h`, `src/api.cpp`, and both runtime validation layers before editing.

## Classify the Change

Write an ABI impact inventory before implementation:

- Name every structure, tag, flag, function, and buffer whose meaning changes.
- State whether the change is a new symbol, an appended structure suffix, or an internal-only change.
- Identify field ownership, units, memory spaces, null/empty rules, failure behavior, and the oldest structure prefix that must remain valid.
- Stop if the requested behavior requires reordering, deleting, resizing, or reinterpreting an existing public field. Design an appended suffix or a new API instead.

Do not bump or reinterpret the global API version by instinct. Follow the repository's existing size-gated suffix pattern.

## Implement the Contract End to End

Update every applicable row. Explain intentionally unaffected rows in the issue or PR.

| Boundary | Required work |
| --- | --- |
| Public header | Append fields only; keep tags fixed-width `int32_t`; document units, ownership, defaults, and failure semantics |
| Size contract | Add the versioned size macro using `offsetof` plus the final member size; retain old macros; add C and C++ layout assertions |
| Initializers | Initialize every recognized suffix default; accept valid short prefixes; never write beyond caller `struct_size` |
| Common validation | Validate complete extents, overflows, tags, aliases, null/empty combinations, finite values, and requested outputs before execution |
| CUDA validation | Mirror public validation without dereferencing device pointers on the host; validate pointer ownership and memory-space tags |
| CPU and CUDA execution | Consume the suffix only after a size guard and preserve identical semantics and failure publication |
| Python | Update exact ctypes layout, constants, low-level calls, high-level packing, result mapping, and tests |
| Symbols and install | Update the version script only for an intentional new public symbol; keep implementation symbols hidden; exercise external C consumers |

Search for every existing size and suffix gate before editing:

```bash
rg -n 'struct_size|api_version|_V[0-9]+_SIZE|gpuxtb_.*_init' \
  include src python tests cmake
```

## Prove Compatibility

Add focused tests for all applicable cases:

1. Exact offsets, sizes, alignment, fixed-width tags, and initializer defaults in C and C++.
2. The oldest supported short structure, each complete suffix, a size one byte short of a suffix, an undersized base, an unknown API version, and a larger future caller structure.
3. Hostile counts, offset overflow, invalid tags, aliases, non-finite inputs, null/empty buffers, and mismatched memory spaces.
4. Complete pre-execution validation with unchanged result flags and sentinel buffers on call-level failure.
5. Per-system numerical failure with full quiet-NaN slices and surviving successful peers.
6. CPU public execution plus CUDA host, device, and mixed descriptors when CUDA can observe the field.
7. Repeated calls, changed geometry/topology, cache and strict WARM identity when the field participates in execution policy.
8. Python low-level layout and high-level behavior from a non-editable installed wheel.
9. Shared symbol allowlist, static and shared installs, and the external install consumer.

Invoke `$gpuxtb-select-validation` and require a shared public CPU build with a verified LP64 runtime. For CUDA-visible changes, also invoke `$gpuxtb-validate-cuda-change`; compile-only coverage is incomplete.

## Review the Final Diff

Re-read every new suffix access and prove that `struct_size` dominates it. Compare header and ctypes offsets directly. Verify that all requested outputs remain caller-owned borrowed views and that validation completes before output commit. Do not describe GFN1 or ROCm as implemented while extending their reserved tags.

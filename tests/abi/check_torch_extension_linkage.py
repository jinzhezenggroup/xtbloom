"""Verify the PyTorch stable-ABI extension's runtime linkage contract.

The compiled `libxtbloom_torch_ext.so` is linked against a build-time-only
stub `libtorch_cpu.so` that carries the real library's SONAME (see
`cmake/3rdparty/torch-stable/README.md`).  The acceptance contract is:

- `DT_NEEDED` contains `libtorch_cpu.so` (the real torch library the end user
  already imported resolves it at `torch.ops.load_library` time) plus only the
  platform C/C++ support libraries needed by the extension;
- no other torch/c10/nvidia provider library is linked;
- there is no RPATH/RUNPATH, so the build-tree stub or any unrelated
  `libtorch_cpu.so` can never be discovered at runtime;
- the extension's only external torch symbols are the pinned stable-C set in
  `aoti_symbols.txt`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

NEEDED_EXPECTED = {
    "libtorch_cpu.so",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    # glibc < 2.34 still carries dlopen/dlsym/dladdr in a separate library.
    "libdl.so.2",
    # manylinux_2_28 keeps std::thread support in a separate system library.
    "libpthread.so.0",
}
FORBIDDEN_NEEDED_FRAGMENTS = (
    "libc10",
    "libtorch.so",
    "libtorch_python",
    "nvidia-",
    "cudart",
    "cublas",
)


def main() -> int:
    """Validate one extension shared library against the linkage contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readelf", required=True, help="readelf-compatible executable"
    )
    parser.add_argument("--nm", required=True, help="nm-compatible executable")
    parser.add_argument(
        "--symbol-list",
        required=True,
        type=Path,
        help="pinned stable-C symbol set (aoti_symbols.txt)",
    )
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()

    dynamic = subprocess.run(
        [args.readelf, "-dW", str(args.library)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    needed = set(re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dynamic))
    unexpected = needed - NEEDED_EXPECTED
    if unexpected:
        raise SystemExit(f"unexpected DT_NEEDED entries: {sorted(unexpected)}")
    if any(
        fragment in entry for entry in needed for fragment in FORBIDDEN_NEEDED_FRAGMENTS
    ):
        raise SystemExit("forbidden torch/provider DT_NEEDED entry found")
    if "libtorch_cpu.so" not in needed:
        raise SystemExit("extension must carry DT_NEEDED libtorch_cpu.so")
    if re.search(r"\(RPATH\)|\(RUNPATH\)", dynamic):
        raise SystemExit("extension must not carry RPATH/RUNPATH")

    undefined = subprocess.run(
        [args.nm, "-D", "--undefined-only", str(args.library)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pinned = set(args.symbol_list.read_text(encoding="utf-8").split())
    actual = set(
        re.findall(r"\b(aoti_torch_[A-Za-z0-9_]+|torch_[A-Za-z0-9_]+)\b", undefined)
    )
    stale = actual - pinned
    if stale:
        raise SystemExit(
            "extension references torch symbols absent from aoti_symbols.txt: "
            + ", ".join(sorted(stale))
        )
    print(  # noqa: T201 - CLI validation output
        f"torch extension linkage OK ({len(needed)} DT_NEEDED, no RPATH/RUNPATH)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

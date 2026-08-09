#!/usr/bin/env python3
"""Vendor the LibTorch Stable ABI headers used by the compiled torch extension.

The compiled gpuxtb PyTorch integration
(`src/bindings/torch/gpuxtb_torch_ext.cpp`) is built
against the LibTorch Stable ABI headers and a build-time-only stub
`libtorch_cpu.so`. Vendoring just the header closure here means building the
extension never requires downloading a torch wheel; the stub is linked instead
of `libtorch_cpu.so` and the shipped extension carries a plain `DT_NEEDED
libtorch_cpu.so` that binds to the torch the end user already imported at
runtime (see AGENTS.md and `cmake/3rdparty/torch-stable/README.md`).

The vendored set is the transitive `#include` closure of the handful of
stable-ABI headers the extension includes.  Only `torch/`-prefixed includes
are followed; the stable set is self-contained (it intentionally does not pull
in c10/ATen), which is what makes the small vendor feasible.

The manifest records the exact torch release the headers came from plus a
per-file sha256 and Git blob.  Regenerate with:

    python3 tools/torch_stable_vendor.py generate \
        --torch-include <torch/include> --torch-version <x.y.z> \
        --out cmake/3rdparty/torch-stable
    python3 tools/torch_stable_vendor.py check --out cmake/3rdparty/torch-stable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

MANIFEST_NAME = "manifest.json"
SYMBOLS_NAME = "aoti_symbols.txt"
VENDOR_RELPATH = "cmake/3rdparty/torch-stable"
INCLUDE_SUBDIR = "include"

# The extension's exact include list; anything these transitively include is
# part of the vendored, pinned set.
ROOT_HEADERS = (
    "torch/csrc/stable/library.h",
    "torch/csrc/stable/tensor.h",
    "torch/headeronly/core/ScalarType.h",
    "torch/headeronly/macros/Macros.h",
    "torch/headeronly/util/Exception.h",
)

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
_TORCH_PREFIXES = ("torch/",)


def _git_object_id(kind: str, data: bytes) -> str:
    """Return the Git SHA-1 object ID for canonical object bytes."""
    header = f"{kind} {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _git_tree_id(entries: dict[str, tuple[str, str]]) -> str:
    """Reconstruct a Git tree ID from relative paths and (mode, blob) pairs."""
    root: dict[str, object] = {}
    for path, leaf in entries.items():
        node = root
        for component in path.split("/")[:-1]:
            child = node.setdefault(component, {})
            node = child
        node[path.split("/")[-1]] = leaf

    def digest_tree(node: dict[str, object]) -> str:
        records: list[tuple[bytes, bytes]] = []
        for name, value in node.items():
            encoded_name = name.encode()
            if isinstance(value, dict):
                record_mode = "40000"
                object_id = digest_tree(value)
                sort_key = encoded_name + b"/"
            else:
                record_mode, object_id = value
                sort_key = encoded_name
            record = (
                f"{record_mode} ".encode()
                + encoded_name
                + b"\0"
                + bytes.fromhex(object_id)
            )
            records.append((sort_key, record))
        payload = b"".join(record for _key, record in sorted(records))
        return _git_object_id("tree", payload)

    return digest_tree(root)


def trace_closure(torch_include: Path) -> set[str]:
    """Return the transitive torch/-prefixed include closure of ROOT_HEADERS."""
    seen: set[str] = set()

    def visit(rel: str) -> None:
        if rel in seen:
            return
        full = torch_include / rel
        if not full.is_file():
            raise FileNotFoundError(f"missing torch include: {rel}")
        seen.add(rel)
        text = full.read_text(encoding="utf-8", errors="replace")
        for match in _INCLUDE_RE.finditer(text):
            inc = match.group(1)
            if not inc.startswith(_TORCH_PREFIXES):
                continue
            for candidate in (str(Path(rel).parent / inc), inc):
                if (torch_include / candidate).is_file():
                    visit(candidate)
                    break

    for root in ROOT_HEADERS:
        visit(root)
    return seen


def generate(args: argparse.Namespace) -> int:
    """Copy the header closure into the vendor tree and pin its manifest."""
    torch_include = Path(args.torch_include)
    out = Path(args.out)
    include_out = out / INCLUDE_SUBDIR

    rels = sorted(trace_closure(torch_include))
    include_out.mkdir(parents=True, exist_ok=True)
    for rel in rels:
        dst = include_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(torch_include / rel, dst)

    entries: dict[str, tuple[str, str]] = {}
    declared: list[dict[str, str]] = []
    for rel in rels:
        payload = (include_out / rel).read_bytes()
        blob = _git_object_id("blob", payload)
        entries[rel] = ("100644", blob)
        declared.append(
            {
                "path": rel,
                "mode": "100644",
                "git_blob": blob,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "license": "BSD-3-Clause",
        "upstream_repository": "https://github.com/pytorch/pytorch",
        "upstream_release": args.torch_version,
        "source_path": "torch/include",
        "tree": _git_tree_id(entries),
        "comment": (
            "LibTorch Stable ABI header closure vendored from the PyPI torch "
            "wheel; regenerate with tools/torch_stable_vendor.py. Build-time "
            "only: never copied into wheels or install trees."
        ),
        "files": declared,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(  # noqa: T201 - CLI progress output
        f"vendored {len(rels)} headers ({args.torch_version}) tree "
        f"{manifest['tree']} -> {out}"
    )
    return 0


def check(args: argparse.Namespace) -> int:
    """Verify the vendored tree matches the pinned manifest byte for byte."""
    out = Path(args.out)
    manifest_path = out / MANIFEST_NAME
    if not manifest_path.is_file():
        print(f"missing {manifest_path}", file=sys.stderr)  # noqa: T201
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    entries: dict[str, tuple[str, str]] = {}
    for entry in manifest.get("files", []):
        rel = entry["path"]
        tree_file = out / INCLUDE_SUBDIR / rel
        expected_blob = entry.get("git_blob")
        if not tree_file.is_file():
            errors.append(f"missing {rel}")
            continue
        payload = tree_file.read_bytes()
        if not isinstance(expected_blob, str) or len(expected_blob) != 40:
            errors.append(f"missing git_blob {rel}")
            continue
        if _git_object_id("blob", payload) != expected_blob:
            errors.append(f"blob mismatch {rel}")
        if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
            errors.append(f"hash mismatch {rel}")
        entries[rel] = (entry.get("mode", "100644"), expected_blob)
    observed = {
        p.relative_to(out / INCLUDE_SUBDIR).as_posix()
        for p in (out / INCLUDE_SUBDIR).rglob("*")
        if p.is_file()
    }
    declared = {entry["path"] for entry in manifest.get("files", [])}
    errors.extend(f"unexpected {extra}" for extra in sorted(observed - declared))
    if manifest.get("tree") != _git_tree_id(entries):
        errors.append("tree id mismatch")
    if errors:
        print("torch-stable vendor check failed:", file=sys.stderr)  # noqa: T201
        for error in errors:
            print(f"  {error}", file=sys.stderr)  # noqa: T201
        return 1
    print(  # noqa: T201 - CLI validation report
        f"torch-stable vendor OK: {len(declared)} files"
    )
    return 0


def build_symbol_list(args: argparse.Namespace) -> int:
    """Emit the union of stable symbols referenced by extension objects.

    Compiler and instrumentation choices can retain different inline stable
    shim calls. Accept multiple objects so the build-time stub covers every
    supported configuration instead of only the object used to refresh it.
    """
    object_paths = [Path(path) for path in args.extension_object]
    missing = [path for path in object_paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing object file: {path}", file=sys.stderr)  # noqa: T201
        return 1
    import subprocess

    outputs = [
        subprocess.run(
            ["nm", "-u", str(object_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for object_path in object_paths
    ]
    # The stable C shim couples aoti_torch_* operators with a small legacy
    # torch_* surface (library.h/tensor.h inline helpers call torch_library_impl
    # and torch_get_mutable_data_ptr); both families are exported by the real
    # libtorch_cpu.so and must be stubbed.
    symbols = sorted(
        set(
            re.findall(
                r"\b(aoti_torch_[A-Za-z0-9_]+|torch_get_mutable_data_ptr|"
                r"torch_library_impl)\b",
                "\n".join(outputs),
            )
        )
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(  # noqa: T201 - CLI progress output
        f"wrote {len(symbols)} torch stable symbols from "
        f"{len(object_paths)} object(s) to {out}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch the vendor subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="vendor the header closure")
    gen.add_argument("--torch-include", required=True, type=Path)
    gen.add_argument("--torch-version", required=True, help="e.g. 2.12.1")
    gen.add_argument("--out", default=VENDOR_RELPATH, type=Path)
    gen.set_defaults(func=generate)

    ck = sub.add_parser("check", help="verify the vendored tree matches the manifest")
    ck.add_argument("--out", default=VENDOR_RELPATH, type=Path)
    ck.set_defaults(func=check)

    syms = sub.add_parser("symbols", help="refresh the torch stable symbol list")
    syms.add_argument(
        "--extension-object",
        required=True,
        action="append",
        type=Path,
        help="compiled extension object; repeat for each supported build mode",
    )
    syms.add_argument("--out", default=f"{VENDOR_RELPATH}/{SYMBOLS_NAME}", type=Path)
    syms.set_defaults(func=build_symbol_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate lazy-dlopen trampoline stubs without scanning a real library.

gpuxtb dlopens the NVIDIA cuBLAS / cuSOLVER / CUDA-runtime / CUDA-driver
cohort at runtime through hand-rolled trampolines (see src/runtime/
cuda_dlopen.c).  When a full CUDA toolkit is present, the build derives
those trampolines from the real shared objects with the vendored
cmake/3rdparty/implib/implib-gen.py, which both pins the SONAME and filters
the exported symbol set against the toolkit.  The generated stub content,
however, depends only on the curated symbol list, the dlopen'd library name,
the target architecture, and implib's architecture templates, and never on
the library bytes themselves.

This script reproduces implib-gen.py's non-vtable output byte-for-byte from
the same vendored templates so the CUDA backend can be configured and built
from nvcc's compiler-support files and the cudart headers, with provider
shared libraries required only at run time.  It is original gpuxtb code;
implib assumes no input library because it never inspects one.

The build uses this stub path whenever the proprietary CUDA libraries are
absent from the build environment, and the byte-exact output keeps the two
paths (and the fixed trampoline names in cuda_dlopen.c) interchangeable.
"""

from __future__ import annotations

import argparse
import configparser
import os
import re
import string
import sys
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Sequence

_SUPPORTED_TARGETS = ("x86_64", "aarch64")


def _error(message: str) -> NoReturn:
    """Print a message to stderr and exit with a nonzero status."""
    sys.stderr.write(f"implib-stubgen: error: {message}\n")
    sys.exit(1)


def _resolve_target(target: str) -> str:
    """Map a CMake/uname-style target string to an implib architecture dir."""
    if target.startswith("arm"):
        candidate = "arm"
    elif re.match(r"^i[0-9]86", target):
        candidate = "i386"
    elif target.startswith("mips64"):
        candidate = "mips64"
    elif target.startswith("mips"):
        candidate = "mips"
    else:
        candidate = target.split("-")[0]
    if candidate not in _SUPPORTED_TARGETS:
        _error(f"unsupported stub generation target '{target}'")
    return candidate


def _pointer_size(arch_dir: str) -> int:
    """Read the arch-specific pointer size pinned in implib's config.ini."""
    parser = configparser.ConfigParser()
    parser.read(os.path.join(arch_dir, "config.ini"))
    try:
        return int(parser["Arch"]["PointerSize"])
    except (KeyError, ValueError):
        _error(f"cannot read pointer size from {arch_dir}/config.ini")


def generate(
    base_name: str,
    symbols: Sequence[str],
    load_name: str,
    target: str,
    implib_root: str,
    outdir: str,
) -> tuple[str, str]:
    """Write the trampoline and initializer for one library; return file names.

    Replacement variables mirror implib-gen.py exactly: string.Template
    ``$$`` escapes in the templates expand to literal ``$``, and the
    library-suffix symbol base normalizes every non-alphanumeric character
    in the basename to ``_`` (for example libcusolver.so -> libcusolver_so).
    """
    lib_suffix = re.sub(r"[^a-zA-Z_0-9]+", "_", base_name)
    ptr_size = _pointer_size(os.path.join(implib_root, "arch", target))

    tramp_path = os.path.join(outdir, f"{base_name}.tramp.S")
    with open(tramp_path, "w", encoding="utf-8") as handle:
        table_template_path = os.path.join(implib_root, "arch", target, "table.S.tpl")
        with open(table_template_path, encoding="utf-8") as table_source:
            template = string.Template(table_source.read())
        handle.write(
            template.substitute(
                lib_suffix=lib_suffix, table_size=ptr_size * (len(symbols) + 1)
            )
        )
        trampoline_template_path = os.path.join(
            implib_root, "arch", target, "trampoline.S.tpl"
        )
        with open(trampoline_template_path, encoding="utf-8") as trampoline_source:
            trampoline = string.Template(trampoline_source.read())
        for index, name in enumerate(symbols):
            handle.write(
                trampoline.substitute(
                    lib_suffix=lib_suffix,
                    sym=name,
                    offset=index * ptr_size,
                    number=index,
                )
            )

    init_path = os.path.join(outdir, f"{base_name}.init.c")
    with open(init_path, "w", encoding="utf-8") as handle:
        init_template_path = os.path.join(implib_root, "arch", "common", "init.c.tpl")
        with open(init_template_path, encoding="utf-8") as init_source:
            template = string.Template(init_source.read())
        sym_names = ",\n  ".join(f'"{name}"' for name in symbols)
        handle.write(
            template.substitute(
                lib_suffix=lib_suffix,
                load_name=load_name,
                dlopen_callback="",
                dlsym_callback="gpu_xtb_cuda_dlsym",
                has_dlopen_callback=0,
                has_dlsym_callback=1,
                no_dlopen=1,
                lazy_load=1,
                sym_names=sym_names + ",",
            )
        )
    return tramp_path, init_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CUDA dlopen trampolines from a curated symbol list."
    )
    parser.add_argument(
        "--base-name", required=True, help="library basename, e.g. libcusolver.so"
    )
    parser.add_argument(
        "--symbol-list",
        required=True,
        help="file with one dlsym'd symbol name per line",
    )
    parser.add_argument(
        "--load-name",
        required=True,
        help="dlopen'd library SONAME, e.g. libcusolver.so.11",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="implib target architecture, e.g. x86_64 or aarch64",
    )
    parser.add_argument(
        "--implib-root",
        required=True,
        help="path to the vendored cmake/3rdparty/implib tree",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="directory in which to write the generated files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    """Entry point."""
    args = parse_args(argv)
    target = _resolve_target(args.target)
    if not os.path.isdir(args.implib_root):
        _error(f"implib root does not exist: {args.implib_root}")
    os.makedirs(args.outdir, exist_ok=True)
    symbols: list[str] = []
    with open(args.symbol_list, encoding="utf-8") as handle:
        for raw in handle.read().splitlines():
            line = re.sub(r"#.*", "", raw).strip()
            if line:
                symbols.append(line)
    if not symbols:
        _error(f"symbol list '{args.symbol_list}' is empty")
    generate(
        args.base_name, symbols, args.load_name, target, args.implib_root, args.outdir
    )


if __name__ == "__main__":
    main(sys.argv[1:])

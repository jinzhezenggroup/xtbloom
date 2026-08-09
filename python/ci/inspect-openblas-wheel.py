#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify xTBloom's auditwheel-vendored private OpenBLAS provider cohort."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


class InspectionError(RuntimeError):
    """Report wheel metadata or ELF structure outside the reviewed policy."""


def _one(names: set[str], pattern: str, description: str) -> str:
    matches = sorted(name for name in names if re.fullmatch(pattern, name))
    if len(matches) != 1:
        raise InspectionError(
            f"wheel must contain exactly one {description}; found {len(matches)}"
        )
    return matches[0]


def _run_readelf(readelf: str, option: str, path: Path) -> str:
    completed = subprocess.run(
        [readelf, option, "--wide", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise InspectionError(
            f"readelf failed for {path.name}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _dynamic(readelf: str, path: Path) -> tuple[set[str], str | None, str | None]:
    output = _run_readelf(readelf, "--dynamic", path)
    needed = set(re.findall(r"Shared library: \[([^]]+)]", output))
    sonames = re.findall(r"Library soname: \[([^]]+)]", output)
    rpaths = re.findall(r"Library rpath: \[([^]]+)]", output)
    runpaths = re.findall(r"Library runpath: \[([^]]+)]", output)
    if len(sonames) > 1 or len(rpaths) > 1 or len(runpaths) > 1:
        raise InspectionError(f"ambiguous dynamic tags in {path.name}")
    if rpaths and runpaths:
        raise InspectionError(f"{path.name} carries both DT_RPATH and DT_RUNPATH")
    search_path = rpaths[0] if rpaths else (runpaths[0] if runpaths else None)
    return needed, sonames[0] if sonames else None, search_path


def _machine(readelf: str, path: Path) -> str:
    output = _run_readelf(readelf, "--file-header", path)
    match = re.search(r"^\s*Machine:\s*(.+?)\s*$", output, re.MULTILINE)
    if match is None:
        raise InspectionError(f"cannot determine ELF machine for {path.name}")
    return match.group(1)


def _expected_vendored_name(source_name: str, source_sha256: str) -> str:
    marker = ".so"
    index = source_name.find(marker)
    if index < 0:
        raise InspectionError(f"manifest payload is not an ELF DSO: {source_name}")
    return source_name[:index] + f"-{source_sha256[:8]}" + source_name[index:]


def inspect_wheel(wheel: Path, manifest_path: Path, readelf: str) -> None:
    """Validate metadata, provenance, and the repaired ELF dependency graph."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    architecture = "aarch64" if wheel.name.endswith("_aarch64.whl") else "x86_64"
    if not wheel.name.endswith(("_x86_64.whl", "_aarch64.whl")):
        raise InspectionError(
            f"unsupported wheel architecture in filename: {wheel.name}"
        )
    arch_manifest = manifest["architectures"][architecture]
    records = arch_manifest["files"]
    expected_machine = arch_manifest["elf_machine"]
    expected_vendor_names = {
        _expected_vendored_name(PurePosixPath(record["source"]).name, record["sha256"])
        for record in records
    }

    with zipfile.ZipFile(wheel) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        metadata_name = _one(names, r"[^/]+\.dist-info/METADATA", "METADATA")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        requirements = [
            value.lower() for value in metadata.get_all("Requires-Dist", [])
        ]
        if any(value.startswith("scipy-openblas32") for value in requirements):
            raise InspectionError(
                "wheel metadata publishes scipy-openblas32 as a dependency"
            )
        if any(name.startswith("scipy_openblas32/") for name in names):
            raise InspectionError(
                "wheel contains the upstream scipy_openblas32 Python package"
            )
        if any(PurePosixPath(name).name == "libscipy_openblas.so" for name in names):
            raise InspectionError("wheel retains the upstream generic OpenBLAS SONAME")

        provenance_name = _one(
            names,
            r"xtbloom/share/licenses/xtbloom/provenance/scipy_openblas32_manifest\.json",
            "installed scipy-openblas32 provenance manifest",
        )
        if archive.read(provenance_name) != manifest_path.read_bytes():
            raise InspectionError("wheel OpenBLAS provenance differs from source bytes")
        source_root = manifest_path.parents[2]
        license_bytes = (source_root / manifest["source"]["local_license"]).read_bytes()
        wheel_license = _one(
            names,
            r"xtbloom/share/licenses/xtbloom/third-party/scipy-openblas32-0\.3\.34\.0\.0\.txt",
            "installed scipy-openblas32 license",
        )
        if archive.read(wheel_license) != license_bytes:
            raise InspectionError("wheel OpenBLAS license differs from reviewed bytes")

        libxtbloom_name = _one(
            names, r"xtbloom/lib(?:64)?/libxtbloom\.so", "libxtbloom"
        )
        shim_name = _one(
            names,
            r"xtbloom/lib(?:64)?/libxtbloom_openblas_lp64_shim\.so",
            "private OpenBLAS shim",
        )
        vendor_members = {
            name
            for name in names
            if ".libs/" in name and (name.endswith(".so") or ".so." in name)
        }
        observed_vendor_names = {PurePosixPath(name).name for name in vendor_members}
        if observed_vendor_names != expected_vendor_names:
            raise InspectionError(
                "auditwheel OpenBLAS cohort differs: expected "
                f"{sorted(expected_vendor_names)}, found "
                f"{sorted(observed_vendor_names)}"
            )
        vendor_dirs = {str(PurePosixPath(name).parent) for name in vendor_members}
        if len(vendor_dirs) != 1:
            raise InspectionError(
                "auditwheel provider cohort is split across directories"
            )
        vendor_dir = next(iter(vendor_dirs))

        with tempfile.TemporaryDirectory(prefix="xtbloom-openblas-wheel-") as directory:
            root = Path(directory)
            selected = {libxtbloom_name, shim_name, *vendor_members}
            for name in selected:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))

            extracted = {name: root / name for name in selected}
            for name, path in extracted.items():
                if _machine(readelf, path) != expected_machine:
                    raise InspectionError(f"{name} has the wrong ELF machine")

            lib_needed, _, _ = _dynamic(readelf, extracted[libxtbloom_name])
            forbidden_direct = {
                name
                for name in lib_needed
                if re.search(r"openblas|gfortran|quadmath|xtbloom_openblas", name)
            }
            if forbidden_direct:
                raise InspectionError(
                    "libxtbloom has a hard private-provider dependency: "
                    + ", ".join(sorted(forbidden_direct))
                )

            provider_record = next(
                record
                for record in records
                if record["source"].endswith("libscipy_openblas.so")
            )
            provider_name = _expected_vendored_name(
                "libscipy_openblas.so", provider_record["sha256"]
            )
            shim_needed, shim_soname, shim_rpath = _dynamic(
                readelf, extracted[shim_name]
            )
            if (
                provider_name not in shim_needed
                or shim_soname != PurePosixPath(shim_name).name
            ):
                raise InspectionError(
                    "private shim does not bind the vendored OpenBLAS provider"
                )
            relative_vendor = os.path.relpath(
                vendor_dir, str(PurePosixPath(shim_name).parent)
            )
            if shim_rpath != f"$ORIGIN/{relative_vendor}":
                raise InspectionError(
                    "private shim does not use auditwheel's relative RPATH"
                )

            provider_member = next(
                name
                for name in vendor_members
                if PurePosixPath(name).name == provider_name
            )
            provider_needed, provider_soname, provider_rpath = _dynamic(
                readelf, extracted[provider_member]
            )
            if provider_soname != provider_name or provider_rpath != "$ORIGIN":
                raise InspectionError(
                    "vendored OpenBLAS SONAME/RPATH is not private and relocatable"
                )
            gfortran_name = next(
                name for name in expected_vendor_names if "gfortran" in name
            )
            if gfortran_name not in provider_needed:
                raise InspectionError(
                    "vendored OpenBLAS does not bind its reviewed libgfortran"
                )
            symbols = _run_readelf(readelf, "--dyn-syms", extracted[provider_member])
            for symbol in (
                "scipy_LAPACKE_dpotrf_work",
                "scipy_LAPACKE_dpocon_work",
                "scipy_LAPACKE_dsyevd_work",
                "scipy_cblas_dtrsm",
                "scipy_cblas_dgemm",
                "scipy_openblas_get_config",
                "openblas_set_num_threads_local",
            ):
                if symbol not in symbols:
                    raise InspectionError(
                        f"vendored OpenBLAS omits required symbol {symbol}"
                    )

            gfortran_member = next(
                name
                for name in vendor_members
                if PurePosixPath(name).name == gfortran_name
            )
            gfortran_needed, gfortran_soname, _ = _dynamic(
                readelf, extracted[gfortran_member]
            )
            if gfortran_soname != gfortran_name:
                raise InspectionError(
                    "vendored libgfortran SONAME is not collision-renamed"
                )
            quadmath_names = {
                name for name in expected_vendor_names if "quadmath" in name
            }
            if architecture == "x86_64":
                quadmath_name = next(iter(quadmath_names))
                if quadmath_name not in gfortran_needed:
                    raise InspectionError(
                        "x86_64 libgfortran does not bind vendored libquadmath"
                    )
            elif quadmath_names:
                raise InspectionError(
                    "aarch64 manifest unexpectedly includes libquadmath"
                )


def main() -> int:
    """Inspect one repaired wheel from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readelf", default="readelf")
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        inspect_wheel(args.wheel.resolve(), args.manifest.resolve(), args.readelf)
    except (
        InspectionError,
        OSError,
        KeyError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        parser.error(str(error))
    print(f"OpenBLAS wheel inspection passed: {args.wheel}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

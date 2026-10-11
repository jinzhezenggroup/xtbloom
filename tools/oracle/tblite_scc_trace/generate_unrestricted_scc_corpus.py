#!/usr/bin/env python3
"""Generate or verify the pinned unrestricted OH SCC trace corpus (issue #51).

This generator reuses the reviewed offline dependency and toolchain machinery
from the restricted v1 generator, but it owns a distinct v2 observer patch,
recorder, manifest, schema, and case.  The separation is intentional: changing
unrestricted evidence must never rewrite or reinterpret the five restricted
v1 goldens.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_scc_corpus as common
import validate_observer_patch as validator
import xtbloom_scc_trace as writer

TOOL_DIR = Path(__file__).resolve().parent
RECORDER_PATH = TOOL_DIR / "scc_trace_recorder_v2.f90"
MAIN_PATH = TOOL_DIR / "scc_trace_main_v2.f90"
METADATA_PATH = TOOL_DIR / "metadata-v2.json"
PATCH_PATH = TOOL_DIR / "tblite-e9abc395-scc-observer-v2.patch"

FORMAT = writer.FORMAT_V2
REVISION = common.REVISION
MANIFEST_SCHEMA = "xtbloom-scc-trace-corpus-manifest-v2"
CASES: dict[str, dict[str, object]] = {
    "oh_radical": {
        "atomic_numbers": [8, 1],
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.834]],
        "molecular_charge": 0.0,
        "unpaired_electrons": 1,
        "spin_channels": 2,
        "temperature_kelvin": 300.0,
        "mixer_memory": 2,
        "mixer_damping": 0.4,
        "maximum_iterations": 100,
    }
}


class CorpusError(common.CorpusError):
    """Raised when the unrestricted corpus violates its pinned contract."""


def sha256_file(path: Path) -> str:
    """Return one file's lowercase SHA-256 digest."""
    return common.sha256_file(path)


def serialize_spec(spec: dict[str, object]) -> str:
    """Append the v2 spin-channel selector to the backward-compatible spec."""
    return common.serialize_spec(spec) + f"{int(spec['spin_channels'])}\n"


def write_spec(spec: dict[str, object], path: Path) -> None:
    """Write the canonical unrestricted fixed-layout spec."""
    path.write_text(serialize_spec(spec), encoding="utf-8")


def _matrix_channels(
    take_floats: Callable[[int], list[float]], channels: int, nao: int
) -> list[list[list[float]]]:
    """Read channel-major Fortran matrices into logical row/column order."""
    return [
        common.column_major_to_rows(take_floats(nao * nao), nao)
        for _ in range(channels)
    ]


def _spectra(
    take_floats: Callable[[int], list[float]], channels: int, length: int
) -> list[list[float]]:
    return [take_floats(length) for _ in range(channels)]


def _multipoles(
    take_floats: Callable[[int], list[float]],
    channels: int,
    atoms: int,
    components: int,
) -> list[list[list[float]]]:
    result: list[list[list[float]]] = []
    for _ in range(channels):
        flat = take_floats(atoms * components)
        result.append(
            [flat[atom * components : (atom + 1) * components] for atom in range(atoms)]
        )
    return result


def _flatten_population(entry: dict[str, object], prefix: str) -> list[float]:
    """Flatten q/d/Q in the exact channel-major tblite mixer order."""
    values: list[float] = []
    for channel in entry[f"{prefix}_qsh"]:
        values.extend(channel)
    for channel in entry[f"{prefix}_dipoles"]:
        for atom in channel:
            values.extend(atom)
    for channel in entry[f"{prefix}_quadrupoles"]:
        for atom in channel:
            values.extend(atom)
    return values


def canonicalize(raw: str, spec: dict[str, object], command_line: str) -> dict:
    """Convert one v2 recorder stream to a validated canonical document."""
    lines = raw.splitlines()
    if not lines:
        raise CorpusError("raw stream is empty")
    header = lines[0].split()
    if (
        len(header) not in (10, 12)
        or header[:10:2] != ["nat", "nsh", "nao", "niterations", "terminal"]
        or (
            len(header) == 12
            and (header[10] != "failed_attempt" or header[11] not in ("0", "1"))
        )
    ):
        raise CorpusError("raw stream header is malformed")
    nat, nsh, nao, niterations, terminal = (int(value) for value in header[1:10:2])
    failed_attempt_present = len(header) == 12 and header[11] == "1"
    tokens = iter(token for line in lines[1:] for token in line.split())

    def take() -> str:
        try:
            return next(tokens)
        except StopIteration:
            raise CorpusError("raw stream ended prematurely") from None

    def expect(label: str) -> None:
        actual = take()
        if actual != label:
            raise CorpusError(f"raw stream expected {label!r}, got {actual!r}")

    def floats(count: int) -> list[float]:
        return [float(take()) for _ in range(count)]

    def ints(count: int) -> list[int]:
        return [int(take()) for _ in range(count)]

    expect("atomic_numbers")
    atomic_numbers = ints(nat)
    expect("positions")
    positions = floats(3 * nat)
    expect("molecular_charge")
    molecular_charge = float(take())
    expect("unpaired_electrons")
    unpaired = int(take())
    expect("spin_channels")
    channels = int(take())
    if channels != 2:
        raise CorpusError(f"unrestricted recorder reported {channels} channels")
    expect("temperature")
    temperature = float(take())
    expect("n_point_charges")
    npc = int(take())
    point_charges = [floats(5) for _ in range(npc)]
    expect("atom_to_shell_count")
    shell_counts = ints(nat)
    if sum(shell_counts) != nsh:
        raise CorpusError("atom_to_shell_count does not sum to n_shells")
    expect("overlap")
    overlap = common.column_major_to_rows(floats(nao * nao), nao)
    expect("core_hamiltonian")
    core_hamiltonian = common.column_major_to_rows(floats(nao * nao), nao)

    iterations = []
    for offset in range(niterations):
        expect("iteration")
        index = int(take())
        if index != offset + 1:
            raise CorpusError("raw iteration indices are not contiguous")
        expect("assembled_hamiltonian")
        assembled = _matrix_channels(floats, channels, nao)
        expect("solver_hamiltonian")
        solver = _matrix_channels(floats, channels, nao)
        expect("eigenvalues")
        eigenvalues = _spectra(floats, channels, nao)
        expect("occupations")
        occupations = _spectra(floats, channels, nao)
        expect("density")
        density = _matrix_channels(floats, channels, nao)
        entry: dict[str, object] = {
            "index": index,
            "assembled_hamiltonian": assembled,
            "solver_hamiltonian": solver,
            "eigenvalues": eigenvalues,
            "occupations": occupations,
            "density": density,
        }
        for prefix in ("mixed", "raw"):
            expect(f"{prefix}_qsh")
            entry[f"{prefix}_qsh"] = _spectra(floats, channels, nsh)
            expect(f"{prefix}_qat")
            entry[f"{prefix}_qat"] = _spectra(floats, channels, nat)
            expect(f"{prefix}_dipoles")
            entry[f"{prefix}_dipoles"] = _multipoles(floats, channels, nat, 3)
            expect(f"{prefix}_quadrupoles")
            entry[f"{prefix}_quadrupoles"] = _multipoles(floats, channels, nat, 6)
        if npc:
            expect("point_charge_shell_potential")
            entry["point_charge_shell_potential"] = [floats(nsh)]
            expect("point_charge_energy")
            entry["point_charge_energy"] = [floats(nsh)]
        expect("energy")
        entry["energy"] = float(take())
        expect("energy_delta")
        entry["energy_delta"] = float(take())
        expect("residual_rms")
        entry["residual_rms"] = float(take())
        expect("convergence")
        flags = [bool(int(take())) for _ in range(4)]
        entry["convergence"] = {
            "energy": flags[0],
            "population": flags[1],
            "temperature": flags[2],
            "overall": flags[3],
        }
        mixed = _flatten_population(entry, "mixed")
        raw_values = _flatten_population(entry, "raw")
        entry["residual"] = [
            raw_value - mixed_value
            for raw_value, mixed_value in zip(raw_values, mixed, strict=True)
        ]
        iterations.append(entry)

    failed_attempt = None
    if failed_attempt_present:
        expect("failed_attempt")
        failed_index = int(take())
        if failed_index != len(iterations) + 1:
            raise CorpusError(
                "failed attempt index does not follow completed iterations"
            )
        expect("assembled_hamiltonian")
        assembled = _matrix_channels(floats, channels, nao)
        expect("solver_hamiltonian")
        solver = _matrix_channels(floats, channels, nao)
        failed_attempt = {
            "index": failed_index,
            "assembled_hamiltonian": assembled,
            "solver_hamiltonian": solver,
        }
        for prefix in ("mixed",):
            expect(f"{prefix}_qsh")
            failed_attempt[f"{prefix}_qsh"] = _spectra(floats, channels, nsh)
            expect(f"{prefix}_qat")
            failed_attempt[f"{prefix}_qat"] = _spectra(floats, channels, nat)
            expect(f"{prefix}_dipoles")
            failed_attempt[f"{prefix}_dipoles"] = _multipoles(floats, channels, nat, 3)
            expect(f"{prefix}_quadrupoles")
            failed_attempt[f"{prefix}_quadrupoles"] = _multipoles(
                floats, channels, nat, 6
            )

    try:
        extra = next(tokens)
    except StopIteration:
        extra = None
    if extra is not None:
        raise CorpusError(f"raw stream has unexpected trailing token {extra!r}")

    trace: dict[str, object] = {
        "format": FORMAT,
        "provenance": {
            "tblite_revision": REVISION,
            "oracle_patch_sha256": sha256_file(PATCH_PATH),
            "oracle_command": command_line,
        },
        "input": {
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "molecular_charge": molecular_charge,
            "unpaired_electrons": unpaired,
            "spin_channels": channels,
            "temperature": temperature,
        },
        "basis": {
            "n_atoms": nat,
            "n_shells": nsh,
            "nao": nao,
            "atom_to_shell_count": shell_counts,
        },
        "statics": {
            "overlap": [overlap],
            "core_hamiltonian": [core_hamiltonian],
        },
        "residual_layout": {
            "shell_charges": channels * nsh,
            "atomic_dipoles": channels * 3 * nat,
            "atomic_quadrupoles": channels * 6 * nat,
        },
        "iterations": iterations,
        "terminal": {
            "status": terminal,
            "converged": terminal == writer.STATUS_CONVERGED,
            "iterations": niterations + (1 if failed_attempt is not None else 0),
        },
    }
    if failed_attempt is not None:
        trace["failed_attempt"] = failed_attempt
    if point_charges:
        trace["input"]["point_charges"] = {
            "positions": [value for row in point_charges for value in row[:3]],
            "charges": [row[3] for row in point_charges],
            "hardnesses": [row[4] for row in point_charges],
        }
    writer.validate(trace)
    return trace


def _manifest_path(case_dir: Path) -> Path:
    return case_dir / "manifest-v2.json"


def validate_manifest_document(manifest: object, case_dir: Path) -> None:
    """Validate every repository-owned byte and pin in the v2 bundle."""
    common.require_manifest(
        isinstance(manifest, dict), "manifest root must be an object"
    )
    expected_fields = {
        "cases",
        "command",
        "dependencies",
        "environment",
        "format",
        "generated_at",
        "host",
        "oracle_patch_sha256",
        "oracle_sources",
        "revision",
        "schema",
        "toolchain",
    }
    common.require_manifest(
        set(manifest) == expected_fields, "manifest fields mismatch"
    )
    common.require_manifest(
        manifest["schema"] == MANIFEST_SCHEMA, "manifest schema mismatch"
    )
    common.require_manifest(manifest["format"] == FORMAT, "manifest format mismatch")
    common.require_manifest(
        manifest["revision"] == REVISION, "manifest revision mismatch"
    )
    common.require_manifest(
        manifest["oracle_patch_sha256"] == sha256_file(PATCH_PATH),
        "observer patch digest mismatch",
    )
    metadata = validator.load_metadata(METADATA_PATH)
    validator.validate_bundle(metadata)
    expected_sources = {
        RECORDER_PATH.name: sha256_file(RECORDER_PATH),
        MAIN_PATH.name: sha256_file(MAIN_PATH),
    }
    common.require_manifest(
        metadata["oracle_sources"] == expected_sources,
        "observer metadata source digest mismatch",
    )
    sources = manifest["oracle_sources"]
    common.require_manifest(
        isinstance(sources, dict), "oracle_sources must be an object"
    )
    for name, digest in sources.items():
        common.require_sha256(digest, f"oracle_sources.{name}")
    common.require_manifest(
        sources == expected_sources, "oracle source digest mismatch"
    )
    dependencies = manifest["dependencies"]
    common.require_manifest(
        isinstance(dependencies, dict), "dependencies must be an object"
    )
    for name, revision in dependencies.items():
        common.require_manifest(
            isinstance(name, str)
            and isinstance(revision, str)
            and common.LOWER_HEX_40.fullmatch(revision) is not None,
            f"dependency {name!r} is not pinned to a lowercase commit",
        )
    common.require_manifest(
        dependencies == common.PINNED_DEPENDENCIES,
        "dependency pins mismatch",
    )
    common.require_manifest(
        manifest["environment"] == common.DETERMINISTIC_ENVIRONMENT,
        "deterministic environment mismatch",
    )
    command = manifest["command"]
    common.require_manifest(
        isinstance(command, str) and command, "command must be nonempty"
    )
    common.require_manifest(
        isinstance(manifest["host"], str) and manifest["host"],
        "host must be nonempty",
    )
    generated_at = manifest["generated_at"]
    common.require_manifest(
        isinstance(generated_at, str), "generated_at must be a string"
    )
    try:
        timestamp = datetime.datetime.fromisoformat(generated_at)
    except ValueError as error:
        raise CorpusError(f"generated_at is not ISO-8601: {error}") from error
    common.require_manifest(
        timestamp.tzinfo is not None, "generated_at must include a timezone"
    )
    common.validate_toolchain_provenance(manifest["toolchain"])
    cases = manifest["cases"]
    common.require_manifest(isinstance(cases, dict), "cases must be an object")
    common.require_manifest(set(cases) == set(CASES), "manifest case set mismatch")
    for case_id, spec in CASES.items():
        entry = cases[case_id]
        common.require_manifest(
            isinstance(entry, dict)
            and set(entry) == {"path", "sha256", "spec_path", "spec_sha256"},
            f"manifest entry for {case_id} is malformed",
        )
        expected = {
            "path": f"{case_id}.json",
            "spec_path": f"specs/{case_id}.spec",
        }
        common.require_manifest(
            entry["path"] == expected["path"]
            and entry["spec_path"] == expected["spec_path"],
            f"{case_id} manifest paths mismatch",
        )
        expected_digest = common.require_sha256(
            entry["sha256"], f"cases.{case_id}.sha256"
        )
        expected_spec_digest = common.require_sha256(
            entry["spec_sha256"], f"cases.{case_id}.spec_sha256"
        )
        trace_path = case_dir / entry["path"]
        spec_path = case_dir / entry["spec_path"]
        common.require_manifest(trace_path.is_file(), f"missing {trace_path.name}")
        common.require_manifest(spec_path.is_file(), f"missing {spec_path.name}")
        trace_bytes = trace_path.read_bytes()
        spec_bytes = spec_path.read_bytes()
        common.require_manifest(
            hashlib.sha256(trace_bytes).hexdigest() == expected_digest,
            f"{case_id} trace hash mismatch",
        )
        common.require_manifest(
            hashlib.sha256(spec_bytes).hexdigest() == expected_spec_digest,
            f"{case_id} spec hash mismatch",
        )
        common.require_manifest(
            spec_bytes == serialize_spec(spec).encode(),
            f"{case_id} spec is not canonical",
        )
        try:
            trace = json.loads(trace_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorpusError(f"{case_id} is not canonical JSON: {error}") from error
        try:
            writer.validate(trace)
            canonical = writer.dumps(trace).encode("utf-8")
        except writer.TraceError as error:
            raise CorpusError(f"{case_id} fails trace validation: {error}") from error
        common.require_manifest(
            canonical == trace_bytes,
            f"{case_id} trace is not canonical",
        )
        provenance = trace["provenance"]
        common.require_manifest(
            provenance["tblite_revision"] == REVISION
            and provenance["oracle_patch_sha256"] == manifest["oracle_patch_sha256"]
            and provenance.get("oracle_command") == command,
            f"{case_id} trace provenance mismatch",
        )


def verify_manifest(case_dir: Path) -> int:
    """Verify the committed v2 corpus without rebuilding tblite."""
    try:
        manifest = json.loads(_manifest_path(case_dir).read_text(encoding="utf-8"))
        validate_manifest_document(manifest, case_dir)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        common.CorpusError,
        writer.TraceError,
    ) as error:
        print(  # noqa: T201
            f"ERROR: invalid unrestricted corpus manifest: {error}", file=sys.stderr
        )
        return 2
    print(  # noqa: T201
        f"verified {len(CASES)} unrestricted golden and complete oracle provenance"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line contract for generation and offline checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--meson-command",
        default=f"{shlex.quote(sys.executable)} -m mesonbuild.mesonmain",
    )
    parser.add_argument(
        "--lapack",
        default="auto",
        choices=("auto", "mkl", "mkl-rt", "openblas", "netlib", "custom"),
    )
    parser.add_argument("--custom-library", action="append", default=[])
    parser.add_argument("--wrap-mode", default="nodownload", choices=("nodownload",))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the pinned v2 corpus or verify its committed provenance."""
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = build_parser().parse_args(raw_arguments)
    if arguments.check:
        return verify_manifest(arguments.corpus_dir)
    if arguments.lapack == "custom" and not arguments.custom_library:
        raise CorpusError("--lapack=custom requires --custom-library")
    if arguments.lapack != "custom" and arguments.custom_library:
        raise CorpusError("--custom-library is valid only with --lapack=custom")
    if any(
        re.fullmatch(r"[A-Za-z0-9_.+-]+", name) is None
        for name in arguments.custom_library
    ):
        raise CorpusError("invalid custom library name")

    metadata = validator.load_metadata(METADATA_PATH)
    validator.validate_bundle(metadata)
    _, source_status = validator.source_state(arguments.source_root)
    if source_status.strip():
        raise CorpusError("source checkout is dirty; provenance cannot be established")
    probe = subprocess.run(
        [
            "git",
            "-C",
            str(arguments.source_root),
            "cat-file",
            "-e",
            f"{REVISION}^{{commit}}",
        ],
        check=False,
    )
    if probe.returncode:
        raise CorpusError("pinned tblite revision is not reachable")
    dependencies = common.dependency_revisions(arguments.source_root)

    redacted: list[str] = []
    replace_next = False
    for argument in raw_arguments:
        if replace_next:
            redacted.append("<corpus-dir>")
            replace_next = False
        elif argument == "--corpus-dir":
            redacted.append(argument)
            replace_next = True
        elif argument.startswith("--corpus-dir="):
            redacted.append("--corpus-dir=<corpus-dir>")
        else:
            redacted.append(argument)
    command_line = shlex.join(["generate_unrestricted_scc_corpus.py", *redacted])

    case_dir = arguments.corpus_dir
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtbloom-scc-corpus-v2-") as directory:
        work = Path(directory)
        checkout = work / "checkout"
        validator.clone_and_apply(arguments.source_root, checkout, metadata)
        binary, toolchain = common.build_oracle(
            checkout,
            arguments.source_root,
            dependency_pins=dependencies,
            meson_command=shlex.split(arguments.meson_command),
            lapack=arguments.lapack,
            wrap_mode=arguments.wrap_mode,
            custom_libraries=arguments.custom_library,
            recorder_path=RECORDER_PATH,
            main_path=MAIN_PATH,
            executable_name="xtbloom-tblite-scc-trace-v2",
        )
        entries = {}
        specs_dir = case_dir / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        for case_id, spec in CASES.items():
            print(f"[corpus-v2] running {case_id}", file=sys.stderr)  # noqa: T201
            spec_path = work / f"{case_id}.spec"
            write_spec(spec, spec_path)
            raw = common.run_recorder(binary, spec_path, work)
            trace = canonicalize(raw, spec, command_line)
            output_path = case_dir / f"{case_id}.json"
            output_path.write_text(writer.dumps(trace), encoding="utf-8")
            committed_spec = specs_dir / f"{case_id}.spec"
            shutil.copy2(spec_path, committed_spec)
            entries[case_id] = {
                "path": output_path.name,
                "sha256": sha256_file(output_path),
                "spec_path": f"specs/{case_id}.spec",
                "spec_sha256": sha256_file(committed_spec),
            }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "format": FORMAT,
        "revision": REVISION,
        "oracle_patch_sha256": sha256_file(PATCH_PATH),
        "oracle_sources": {
            RECORDER_PATH.name: sha256_file(RECORDER_PATH),
            MAIN_PATH.name: sha256_file(MAIN_PATH),
        },
        "toolchain": toolchain,
        "dependencies": dependencies,
        "environment": common.DETERMINISTIC_ENVIRONMENT,
        "command": command_line,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "cases": entries,
    }
    _manifest_path(case_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(  # noqa: T201
        f"generated {len(entries)} unrestricted golden under {case_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        CorpusError,
        common.CorpusError,
        validator.ObserverPatchError,
        OSError,
        writer.TraceError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)  # noqa: T201
        sys.exit(2)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise an installed desktop wheel and its private eigensolver provider."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xtbloom
from xtbloom import Calculator

NUMBERS = np.array([8, 1, 1], dtype=np.int32)
POSITIONS = np.array(
    [
        [0.0, 0.0, -0.73578586109551],
        [1.44183152868459, 0.0, 0.36789293054775],
        [-1.44183152868459, 0.0, 0.36789293054775],
    ],
    dtype=np.float64,
)


def _singlepoint(model: str = "GFN2-xTB") -> tuple[float, np.ndarray]:
    calculator = Calculator(model, NUMBERS, POSITIONS, backend="cpu")
    result = calculator.singlepoint()
    if not result.scc_converged:
        raise RuntimeError(
            f"installed desktop wheel did not converge water with {model}"
        )
    if not np.isfinite(result.energy) or not np.isfinite(result.forces).all():
        raise RuntimeError(
            f"installed desktop wheel returned non-finite {model} results"
        )
    return float(result.energy), np.asarray(result.forces)


def _private_provider() -> Path:
    package = Path(xtbloom.__file__).resolve().parent
    candidates = [
        *sorted((package / "lib").glob("libxtbloom_blas-*.dylib")),
        *sorted((package / "bin").glob("xtbloom_openblas-*.dll")),
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise RuntimeError(
            "installed desktop wheel must contain one private provider; "
            f"found {present}"
        )
    return present[0]


def _prove_missing_provider_fails(provider: Path) -> None:
    disabled = provider.with_name(provider.name + ".disabled")
    provider.rename(disabled)
    try:
        code = """
import numpy as np
from xtbloom import Calculator

calculator = Calculator(
    "GFN2-xTB",
    np.array([8, 1, 1], dtype=np.int32),
    np.array(
        [
            [0.0, 0.0, -0.73578586109551],
            [1.44183152868459, 0.0, 0.36789293054775],
            [-1.44183152868459, 0.0, 0.36789293054775],
        ],
        dtype=np.float64,
    ),
    backend="cpu",
)
try:
    calculator.singlepoint()
except Exception as error:
    message = str(error).lower()
    if (
        "openblas" not in message
        and "linear-algebra" not in message
        and "backend" not in message
    ):
        raise
else:
    raise SystemExit(
        "CPU inference unexpectedly succeeded without the private provider"
    )
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "missing-provider child did not fail through the expected "
                "backend path:\n" + completed.stdout + completed.stderr
            )
    finally:
        disabled.rename(provider)


def _prove_torch_extension() -> None:
    """Run the installed compiled op first, including Windows Unicode paths."""
    try:
        distribution = importlib.metadata.distribution("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Torch runtime validation was requested but not installed"
        ) from error

    code = """
import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xtbloom
from xtbloom import Calculator, xtbloom_torch

expected_package = os.environ.get("XTBLOOM_EXPECT_PACKAGE")
if expected_package is not None:
    actual_package = Path(xtbloom.__file__).resolve().parent
    if actual_package != Path(expected_package).resolve():
        raise RuntimeError(
            f"expected copied package {expected_package}, loaded {actual_package}"
        )

if sys.platform == "win32":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
    handle = kernel32.GetModuleHandleW("torch_cpu.dll")
    if not handle:
        raise RuntimeError("import torch did not preload torch_cpu.dll")
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise OSError(ctypes.get_last_error(), "cannot resolve loaded torch_cpu.dll")
    loaded_torch_cpu = Path(buffer.value).resolve()
    expected_torch_lib = (Path(torch.__file__).resolve().parent / "lib").resolve()
    if loaded_torch_cpu.parent != expected_torch_lib:
        raise RuntimeError(
            f"torch_cpu.dll came from {loaded_torch_cpu}, not {expected_torch_lib}"
        )

numbers = np.array([8, 1, 1], dtype=np.int32)
positions = np.array(
    [
        [0.0, 0.0, -0.73578586109551],
        [1.44183152868459, 0.0, 0.36789293054775],
        [-1.44183152868459, 0.0, 0.36789293054775],
    ],
    dtype=np.float64,
)
torch_positions = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
energies, forces = xtbloom_torch(
    torch_positions,
    torch.tensor(numbers, dtype=torch.int32),
    torch.tensor([0, len(numbers)], dtype=torch.int64),
    torch.zeros(1, dtype=torch.float64),
    torch.zeros(1, dtype=torch.int32),
    torch.ones(1, dtype=torch.int32),
    backend="cpu",
)
energies.sum().backward()
assert torch.isfinite(energies).all()
assert torch.isfinite(forces).all()
torch.testing.assert_close(torch_positions.grad, -forces, atol=0.0, rtol=0.0)

# Compare with the ordinary public C-ABI path only after the Torch entry point
# has proven it can initialize the native provider itself.
reference = Calculator("GFN2-xTB", numbers, positions, backend="cpu").singlepoint()
torch.testing.assert_close(
    energies,
    torch.tensor([reference.energy], dtype=torch.float64),
    atol=1.0e-12,
    rtol=1.0e-12,
)
torch.testing.assert_close(
    forces,
    torch.from_numpy(np.ascontiguousarray(reference.forces)),
    atol=1.0e-12,
    rtol=1.0e-12,
)
print(f"xTBloom desktop Torch wheel smoke passed: torch {torch.__version__}")
"""

    def run_child(*, environment: dict[str, str] | None = None) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "installed desktop Torch extension smoke failed:\n"
                + completed.stdout
                + completed.stderr
            )

    run_child()
    if sys.platform == "win32":
        # Exercise the extension's Wide Win32 path handling, including loading
        # the adjacent xtbloom.dll and private provider from a Unicode path.
        with tempfile.TemporaryDirectory(prefix="xtbloom-路径-") as temporary:
            copied_root = Path(temporary) / "测试包"
            copied_package = copied_root / "xtbloom"
            shutil.copytree(Path(xtbloom.__file__).resolve().parent, copied_package)
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (str(copied_root), existing_pythonpath)
                if part is not None and part != ""
            )
            environment["XTBLOOM_EXPECT_PACKAGE"] = str(copied_package)
            run_child(environment=environment)
    print(  # noqa: T201 - CI validation report
        f"xTBloom desktop Torch runtime passed: {distribution.version}"
    )


def main() -> int:
    """Run provider, host-coexistence, correctness, and concurrency checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch",
        action="store_true",
        help="also require a separately installed Torch and run the compiled op",
    )
    args = parser.parse_args()

    try:
        importlib.metadata.distribution("scipy-openblas32")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("scipy-openblas32 leaked into the wheel test environment")

    provider = _private_provider()
    _prove_missing_provider_fails(provider)

    # Exercise the host array backend before and after xTBloom initializes its
    # separately verified private provider. This is a numerical coexistence
    # check; the runtime factory independently proves that every setter and
    # dispatch symbol belongs to the exact private image it opened.
    host_matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    host_before = np.linalg.eigvalsh(host_matrix)
    reference_energy, reference_forces = _singlepoint()
    gfn1_energy, gfn1_forces = _singlepoint("GFN1-xTB")
    host_after = np.linalg.eigvalsh(host_matrix)
    np.testing.assert_allclose(host_after, host_before, rtol=0.0, atol=0.0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: _singlepoint(), range(8)))
    for energy, forces in results:
        np.testing.assert_allclose(energy, reference_energy, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=1.0e-12)

    if args.torch:
        _prove_torch_extension()

    print(  # noqa: T201 - CI validation report
        f"xTBloom desktop wheel CPU inference passed: energy={reference_energy:.16g}; "
        f"gfn1_energy={gfn1_energy:.16g}; "
        f"gfn1_force_norm={np.linalg.norm(gfn1_forces):.16g}; "
        f"provider={provider.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

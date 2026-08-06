"""Unit tests for the CUDA dependency and symbol checker."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("check_cuda_dependencies.py")
SPEC = importlib.util.spec_from_file_location("check_cuda_dependencies", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CudaDependencyCheckerTests(unittest.TestCase):
    """Verify dependency and symbol parsing against representative readelf text."""

    def test_accepts_non_nvidia_dependencies(self) -> None:
        """Allow ordinary non-NVIDIA ELF dependencies."""
        dynamic = """
 0x0000000000000001 (NEEDED) Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]
"""
        self.assertEqual(CHECKER.find_forbidden_needed(dynamic), [])

    def test_rejects_full_nvidia_dependency_families(self) -> None:
        """Reject every NVIDIA dependency family covered by the gate."""
        dynamic = """
 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.12]
 0x0000000000000001 (NEEDED) Shared library: [libnvJitLink.so.12]
 0x0000000000000001 (NEEDED) Shared library: [libnvidia-ml.so.1]
 0x0000000000000001 (NEEDED) Shared library: [libcufile.so.0]
 0x0000000000000001 (NEEDED) Shared library: [libcudnn.so.9]
 0x0000000000000001 (NEEDED) Shared library: [libnccl.so.2]
 0x0000000000000001 (NEEDED) Shared library: [libcutensor.so.2]
 0x0000000000000001 (NEEDED) Shared library: [libnvshmem_host.so.3]
"""
        self.assertEqual(
            CHECKER.find_forbidden_needed(dynamic),
            [
                "libcudart.so.12",
                "libcudnn.so.9",
                "libcufile.so.0",
                "libcutensor.so.2",
                "libnccl.so.2",
                "libnvidia-ml.so.1",
                "libnvjitlink.so.12",
                "libnvshmem_host.so.3",
            ],
        )

    def test_rejects_undefined_versioned_cuda_alias(self) -> None:
        """Reject an unresolved versioned CUDA driver symbol."""
        symbols = """
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND cuMemGetAddressRange_v2@Base
     2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND memcpy@GLIBC_2.14
"""
        unresolved, exported = CHECKER.find_cuda_symbol_leaks(symbols)
        self.assertEqual(unresolved, ["cuMemGetAddressRange_v2"])
        self.assertEqual(exported, [])

    def test_rejects_public_loader_and_cuda_shim_exports(self) -> None:
        """Reject public CUDA API and internal loader shim exports."""
        symbols = """
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 0000000000001234    16 FUNC    GLOBAL DEFAULT   12 gpu_xtb_cuda_dlsym
     2: 0000000000002345    16 FUNC    GLOBAL DEFAULT   12 cudaMalloc
     3: 0000000000003456    16 FUNC    GLOBAL HIDDEN    12 gpu_xtb_cuda_bootstrap
"""
        unresolved, exported = CHECKER.find_cuda_symbol_leaks(symbols)
        self.assertEqual(unresolved, [])
        self.assertEqual(exported, ["cudaMalloc", "gpu_xtb_cuda_dlsym"])

    def test_rejects_embedded_cudadevrt_link_input(self) -> None:
        """Reject an ELF payload that records a cudadevrt link input."""
        self.assertEqual(
            CHECKER.find_forbidden_binary_tokens(b"elf metadata -l dl,cudadevrt"),
            ["cudadevrt"],
        )

    def test_accepts_binary_without_cudadevrt_token(self) -> None:
        """Accept an ELF payload without the forbidden device runtime token."""
        self.assertEqual(
            CHECKER.find_forbidden_binary_tokens(b"elf metadata -l dl"), []
        )


if __name__ == "__main__":
    unittest.main()

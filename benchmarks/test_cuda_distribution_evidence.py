"""Regression tests for exact CUDA distribution-evidence serialization."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    REPOSITORY_ROOT / "benchmarks/evidence/issue-220/2026-08-12-rtx5090-final-matrix"
)


class CudaDistributionEvidenceTests(unittest.TestCase):
    """Keep the compact archive reproducible without changing samples or values."""

    def test_summary_uses_compact_json_with_final_newline(self) -> None:
        """Avoid reintroducing indentation that exceeds the aggregate budget."""
        contents = (BUNDLE / "distribution-summary.json").read_text(encoding="utf-8")
        self.assertEqual(
            contents, json.dumps(json.loads(contents), separators=(",", ":")) + "\n"
        )

    def test_summary_regenerates_exact_json_and_csv(self) -> None:
        """Replay every retained distribution from the authoritative raw samples."""
        spec = importlib.util.spec_from_file_location(
            "cuda_evidence_aggregate", BUNDLE / "aggregate.py"
        )
        assert spec is not None and spec.loader is not None
        aggregate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aggregate)
        documents = {
            kind: json.loads(
                (BUNDLE / f"{kind}-term-matrix.json").read_text(encoding="utf-8")
            )
            for kind in ("d4", "aes2")
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            shutil.copyfile(
                BUNDLE / "pairlist-matrix.json", output / "pairlist-matrix.json"
            )
            names = aggregate.write_distribution_summary(output, documents)
            self.assertEqual(
                names, ["distribution-summary.json", "distribution-summary.csv"]
            )
            for name in names:
                with self.subTest(artifact=name):
                    self.assertEqual(
                        (output / name).read_bytes(), (BUNDLE / name).read_bytes()
                    )


if __name__ == "__main__":
    unittest.main()

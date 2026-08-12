"""Focused tests for the shared conformance-fixture helpers."""

from pathlib import Path

import _cases
import numpy as np


def test_parse_coord_normalizes_lowercase_heavy_elements(tmp_path: Path) -> None:
    """Turbomole lowercase labels cover the full public element mapping."""
    coord = tmp_path / "coord"
    coord.write_text("$coord\n0.0 0.0 0.0 i\n1.0 0.0 0.0 zn\n2.0 0.0 0.0 rn\n$end\n")

    numbers, positions = _cases._parse_coord(coord)

    assert numbers == [53, 30, 86]
    np.testing.assert_array_equal(
        positions,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

"""Focused tests for the shared conformance-fixture helpers."""

from pathlib import Path

import numpy as np

import _cases


def test_parse_coord_normalizes_lowercase_heavy_elements(tmp_path: Path) -> None:
    """Turbomole lowercase labels cover the full public element mapping."""
    coord = tmp_path / "coord"
    coord.write_text(
        "$coord\n"
        "0.0 0.0 0.0 i\n"
        "1.0 0.0 0.0 zn\n"
        "2.0 0.0 0.0 rn\n"
        "$end\n"
    )

    numbers, positions = _cases._parse_coord(coord)

    assert numbers == [53, 30, 86]
    np.testing.assert_array_equal(
        positions,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

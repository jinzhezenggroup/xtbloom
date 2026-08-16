"""Focused coverage for dpdata per-frame metadata resolution."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dpdata")
from xtbloom.dpdata import _frame_value  # noqa: E402
from xtbloom.exceptions import XTBloomValueError  # noqa: E402


def test_frame_value_accepts_scalar_and_per_frame_metadata() -> None:
    """Accept both documented metadata forms without changing their values."""
    assert _frame_value(
        {"charge": np.array(1.5)}, "charge", None, 2, 3, default=0.0
    ) == pytest.approx(1.5)
    assert _frame_value(
        {"charge": np.array([0.0, 1.0, 2.0])},
        "charge",
        None,
        2,
        3,
        default=0.0,
    ) == pytest.approx(2.0)


def test_frame_value_preserves_default_and_fixed_precedence() -> None:
    """Use defaults only for absent keys and keep explicit values authoritative."""
    assert _frame_value({}, "charge", None, 0, 3, default=0.0) == pytest.approx(0.0)
    assert _frame_value(
        {"charge": np.array([99.0])}, "charge", 1.0, 0, 3, default=0.0
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value",
    [np.array([0.0, 1.0]), np.zeros((3, 1)), np.array([0.0, 1.0, 2.0, 3.0])],
)
def test_frame_value_rejects_malformed_present_metadata(value: np.ndarray) -> None:
    """Never substitute a default for a present but malformed metadata array."""
    with pytest.raises(XTBloomValueError, match="one value per frame"):
        _frame_value({"charge": value}, "charge", None, 0, 3, default=0.0)

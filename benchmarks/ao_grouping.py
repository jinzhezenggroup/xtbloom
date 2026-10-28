"""Compatibility facade for the installed exact-AO batching implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_shared_batching() -> ModuleType:
    """Load the source module directly so this facade avoids package imports."""
    module_path = (
        Path(__file__).resolve().parents[1] / "python" / "xtbloom" / "batching.py"
    ).resolve()
    existing = sys.modules.get("xtbloom.batching")
    existing_path = getattr(existing, "__file__", None)
    if isinstance(existing_path, str):
        try:
            if Path(existing_path).resolve() == module_path:
                return existing
        except (OSError, RuntimeError):
            pass

    private_name = (
        "_xtbloom_shared_batching_"
        + hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    )
    existing = sys.modules.get(private_name)
    if existing is not None:
        private_path = getattr(existing, "__file__", None)
        if isinstance(private_path, str):
            try:
                if Path(private_path).resolve() == module_path:
                    return existing
            except (OSError, RuntimeError):
                pass
        del sys.modules[private_name]

    spec = importlib.util.spec_from_file_location(private_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared batching module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[private_name]
        raise
    return module


_shared_batching = _load_shared_batching()
globals().update(
    {
        name: value
        for name, value in vars(_shared_batching).items()
        if not name.startswith("__")
    }
)
del _shared_batching

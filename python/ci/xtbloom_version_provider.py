"""Resolve the dynamic product version directly from a strict Git tag."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

_NUMBER = r"(?:0|[1-9][0-9]*)"
_TAG_RE = re.compile(rf"^v(?P<version>{_NUMBER}\.{_NUMBER}\.{_NUMBER})$")
_VERSION_RE = re.compile(rf"^{_NUMBER}\.{_NUMBER}\.{_NUMBER}$")


def _run_git(source_root: Path, *arguments: str) -> str:
    """Run one read-only Git query and return its stripped standard output."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "Git metadata exists but the git executable is unavailable"
        ) from error

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _git_tag_version(source_root: Path) -> str:
    """Return the nearest reachable release tag after rejecting shallow history."""
    top_level = Path(_run_git(source_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != source_root:
        raise RuntimeError(
            f"source root {source_root} unexpectedly belongs to parent "
            f"repository {top_level}"
        )

    shallow = _run_git(source_root, "rev-parse", "--is-shallow-repository")
    if shallow == "true":
        raise RuntimeError(
            "xTBloom version resolution requires complete Git tag history; "
            "shallow clone rejected"
        )
    if shallow != "false":
        raise RuntimeError(f"Git returned invalid shallow-repository state {shallow!r}")

    tag = _run_git(
        source_root,
        "describe",
        "--tags",
        "--match",
        "v*",
        "--abbrev=0",
    )
    match = _TAG_RE.fullmatch(tag)
    if match is None:
        raise RuntimeError(f"Git tag {tag!r} does not match strict vMAJOR.MINOR.PATCH")
    return match.group("version")


def _frozen_tag_version(source_root: Path) -> str:
    """Read a tag-derived version frozen into an sdist or Git archive."""
    pkg_info = source_root / "PKG-INFO"
    if pkg_info.is_file():
        for line in pkg_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("Version: "):
                version = line.removeprefix("Version: ").strip()
                if _VERSION_RE.fullmatch(version) is None:
                    raise RuntimeError(f"PKG-INFO contains non-tag version {version!r}")
                return version
        raise RuntimeError("PKG-INFO does not contain a Version field")

    archival = source_root / ".git_archival.txt"
    if archival.is_file():
        for line in archival.read_text(encoding="utf-8").splitlines():
            if line.startswith("describe-name: "):
                tag = line.removeprefix("describe-name: ").strip()
                if "$Format:" in tag:
                    break
                match = _TAG_RE.fullmatch(tag)
                if match is None:
                    raise RuntimeError(
                        f"Git archive tag {tag!r} does not match strict "
                        "vMAJOR.MINOR.PATCH"
                    )
                return match.group("version")

    raise RuntimeError(
        "cannot resolve xTBloom version without a full Git checkout, "
        "sdist PKG-INFO, or expanded Git archive metadata"
    )


def dynamic_metadata(
    settings: Mapping[str, Any],
    project: Mapping[str, Any],
) -> dict[str, Any]:
    """Provide one bare tag version to scikit-build-core dynamic metadata."""
    source_root = Path("pyproject.toml").resolve().parent
    if (source_root / ".git").exists():
        expected = _git_tag_version(source_root)

        # Keep the reviewed setuptools-scm plugin as the Git parser, but
        # validate its output so derived version suffixes can never leak into
        # any product-version surface.
        from scikit_build_core.metadata.setuptools_scm import Provider

        resolved = Provider.dynamic_metadata(settings, project)
        observed = resolved.get("version")
        if observed != expected:
            raise RuntimeError(
                f"setuptools-scm produced {observed!r}; expected Git tag "
                f"version {expected!r}"
            )
        return resolved

    return {"version": _frozen_tag_version(source_root)}


def get_requires_for_dynamic_metadata(
    _settings: Mapping[str, Any],
) -> list[str]:
    """Keep the provider's Python dependency exact in isolated builds."""
    return ["setuptools-scm==10.2.1"]

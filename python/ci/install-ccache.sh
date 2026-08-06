#!/usr/bin/env bash
# Install a CUDA-capable ccache into the manylinux build container used by
# cibuildwheel. The wheel workflow caches both the ccache directory and
# cibuildwheel's own tool downloads, so repeated PR runs reuse compiled C++
# and CUDA objects instead of rebuilding them.
#
# Why a downloaded static binary instead of the distro package:
#   - AlmaLinux-8-based manylinux_2_28 images ship ccache 3.7 via yum, which
#     predates nvcc/CUDA support (added in ccache 4.1) and cannot cache the
#     CUDA backend at all.
#   - A ccache built for the Ubuntu host runner links glibc newer than the
#     manylinux_2_28 container, so it cannot execute inside the container.
# The musl-static release binaries from ccache/ccache are fully static and
# run on any Linux kernel, matching the pin/provenance style of the other
# pinned artifacts in this repository (see THIRD_PARTY_NOTICES.md).
#
# Must be run as root inside the container (before-build runs as root) and is
# idempotent: the container persists across per-Python-version before-build
# calls, so a second call detects the existing install and only preloads the
# cache directory and binary mode are already in place.
set -euo pipefail

ccache_version=4.13.6

# SHA-256 digests of the exact pinned release archives (GPL-3.0-or-later,
# upstream ccache build artifacts; recorded for provenance in
# THIRD_PARTY_NOTICES.md).
declare -A ccache_sha256=(
  [x86_64]=156ec57c5198cc849d92834023d09910b83dc5504c6cf405d09e6ae7b208a3e5
  [aarch64]=2098d561e4a8e36bd06a29aedce53ea90c7e365f9573a93d91c230efbf96a958
)

arch="$(uname -m)"
case "$arch" in
  x86_64 | aarch64) ;;
  *)
    echo "unsupported build architecture for ccache: $arch" >&2
    exit 1
    ;;
esac

if command -v ccache >/dev/null 2>&1 && ccache --version >/dev/null 2>&1; then
  echo "ccache is already installed: $(ccache --version | head -n1)"
else
  archive="ccache-${ccache_version}-linux-${arch}-musl-static.tar.xz"
  url="https://github.com/ccache/ccache/releases/download/v${ccache_version}/${archive}"
  download_dir="$(mktemp -d)"
  trap 'rm -rf "$download_dir"' EXIT
  # The manylinux_2_28 base (AlmaLinux 8) ships curl 7.61, which predates
  # --retry-all-errors (curl 7.71); keep to flags it understands.
  curl --fail --location --retry 3 \
    --output "$download_dir/${archive}" "$url"
  echo "${ccache_sha256[$arch]}  $download_dir/${archive}" | sha256sum --check -
  tar -xJf "$download_dir/${archive}" -C /usr/local/bin \
    --strip-components=1 --wildcards "ccache-${ccache_version}*/ccache"
  chmod 0755 /usr/local/bin/ccache
  echo "installed static ccache ${ccache_version} for ${arch} to /usr/local/bin"
fi

# Ensure the shared cache directory (CCACHE_DIR=/host/... inside the
# container) exists before the first compilation writes to it.
mkdir -p "${CCACHE_DIR:-/root/.cache/ccache}"

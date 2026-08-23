#!/usr/bin/env bash
set -euo pipefail

# Build the standalone probe with the exact compiler and module files used by
# the pinned tblite oracle.  Paths may be overridden for independent rebuilds.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
common_dir=$(git -C "$repo_root" rev-parse \
  --path-format=absolute --git-common-dir)
shared_root=$(cd -- "$common_dir/.." && pwd)
oracle_root=${XTBLOOM_TBLITE_ORACLE_ROOT:-$shared_root/build/oracles}
oracle_env=${XTBLOOM_TBLITE_ORACLE_ENV:-$oracle_root/tblite-133-env}
oracle_build=${XTBLOOM_TBLITE_ORACLE_BUILD:-$oracle_root/tblite-133-build-gf143}
output=${1:-$repo_root/build/periodic-gfn2-terms/probe}

mkdir -p -- "$(dirname -- "$output")"
"$oracle_env/bin/x86_64-conda-linux-gnu-gfortran" \
  -I"$oracle_build/libtblite.so.0.7.0.p" \
  -I"$oracle_env/include" \
  -I"$oracle_env/include/toml-f/modules" \
  -I"$oracle_env/include/mctc-lib/modules" \
  -I"$oracle_env/include/jonquil/modules" \
  -I"$oracle_env/include/multicharge/gcc-14.3.0" \
  -I"$oracle_env/include/dftd4/gcc-14.3.0" \
  -I"$oracle_env/include/s-dftd3/gcc-14.3.0" \
  -fopenmp \
  "$script_dir/probe.f90" \
  -Wl,--start-group \
  -L"$oracle_build" \
  -L"$oracle_env/lib" \
  -ltblite \
  "$oracle_env/lib/liblapack.so" \
  "$oracle_env/lib/libblas.so" \
  "$oracle_env/lib/libtoml-f.so" \
  "$oracle_env/lib/libmctc-lib.so" \
  "$oracle_env/lib/libmulticharge.so" \
  "$oracle_env/lib/libdftd4.so" \
  "$oracle_env/lib/libs-dftd3.so" \
  -Wl,--end-group \
  -o "$output"

printf '%s\n' "$output"

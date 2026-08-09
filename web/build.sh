#!/usr/bin/env bash
#
# Build the GitHub Pages WASM demo assets into web/dist.
#
# Produces:
#   dist/gpuxtb_web.js     Emscripten MODULARIZE glue (factory: createGpuxTbModule)
#   dist/gpuxtb_web.wasm   wasm64 main module (gpuxtb + web adapter)
#   dist/libscipy_openblas.so  dlopen-able wasm64 side module (minimal LP64 LAPACKe/CBLAS)
#
# Requirements (already set up by the calling CI step / local emsdk):
#   em++, emcmake, ninja in PATH; EM_CONFIG valid.
# Set GPUXTB_WASM_LIB to the cmake build dir that contains libgpuxtb.a.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB="$ROOT/web"
OUT="${GPUXTB_WASM_OUT:-$WEB/dist}"
LIB_A="${GPUXTB_WASM_LIB:-$ROOT/build/wasm-web/libgpuxtb.a}"
INCLUDE="$ROOT/include"

mkdir -p "$OUT"

echo "[1/4] LAPACKEe side module (wasm64, single-threaded)"
emcc "$WEB/wasm/linalg.c" -o "$OUT/libscipy_openblas.so" \
  -sSIDE_MODULE=2 -m64 -O2 \
  -sEXPORTED_FUNCTIONS='["_openblas_get_config","_openblas_set_num_threads_local","_LAPACKE_dpotrf_work","_LAPACKE_dpocon_work","_LAPACKE_dsyevd_work","_cblas_dgemm","_cblas_dtrsm"]'

echo "[2/4] web adapter object"
emcc -m64 -fPIC -O2 -I"$INCLUDE" -c "$WEB/gpuxtb_web.c" -o "$OUT/gpuxtb_web.o"

echo "[3/4] main module (wasm64, single-threaded, dlopens the side module)"
EMCC_FORCE_STDLIBS=1 em++ "$OUT/gpuxtb_web.o" "$LIB_A" -I"$INCLUDE" \
  -o "$OUT/gpuxtb_web.js" \
  -m64 -O2 -sMAIN_MODULE=2 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createGpuxTbModule \
  -sSTACK_SIZE=8MB -sALLOW_MEMORY_GROWTH \
  -sFORCE_FILESYSTEM \
  --preload-file "$OUT/libscipy_openblas.so@/libscipy_openblas.so" \
  -sEXPORTED_FUNCTIONS=_gpuxtb_web_compute,_gpuxtb_web_optimize,_gpuxtb_web_version,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToUTF8OnStack,ccall,cwrap

echo "[4/4] copy static assets"
cp "$WEB/index.html" "$WEB/style.css" "$WEB/app.js" "$WEB/worker.js" "$OUT/"
mkdir -p "$OUT/vendor"
cp "$WEB/vendor/3Dmol-min.js" "$OUT/vendor/"

echo "done -> $OUT"
ls -la "$OUT"

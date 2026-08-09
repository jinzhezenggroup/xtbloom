# gpuxtb GitHub Pages WASM web demo.
#
# Enabled with -DGPUXTB_BUILD_WEB_DEMO=ON under the Emscripten toolchain
# (emcmake). The whole pipeline — npm-fetch the 3Dmol.js browser bundle,
# build the dlopen-able LAPACKE side module, compile the web adapter, link the
# wasm64 main module, and stage the static assets — is expressed as custom
# commands so a single `cmake --build` produces the deployable site in
#   <build>/web/site
#
# The deterministic, version-pinned build inputs live in this directory:
#   package.json / package-lock.json   (3dmol.js 2.5.5, BSD-3-Clause)
#   wasm/linalg.c                      (minimal LP64 LAPACKE/CBLAS side module)
#   gpuxtb_web.c                       (XYZ/JSON C adapter + L-BFGS optimizer)
#   index.html / style.css / app.js / worker.js
# See THIRD_PARTY_NOTICES.md for provenance. Outputs under <build> are build
# artifacts and never committed.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message(FATAL_ERROR
    "GPUXTB_BUILD_WEB_DEMO=ON requires configure with the Emscripten toolchain "
    "(emcmake cmake ...).")
endif()

set(_gpuxtb_web_src "${CMAKE_CURRENT_SOURCE_DIR}")
set(_gpuxtb_web_site "${CMAKE_CURRENT_BINARY_DIR}/site")
set(_gpuxtb_web_obj "${CMAKE_CURRENT_BINARY_DIR}/gpuxtb_web.o")
set(_gpuxtb_web_include "-I${CMAKE_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# 0) Fetch build-time npm dependencies (3Dmol.js). Re-runs only when
#    package.json / package-lock.json are newer than the resolved bundle.
# ---------------------------------------------------------------------------
set(_gpuxtb_web_3dmol
    "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/3dmol/build/3Dmol-min.js")
add_custom_command(
  OUTPUT "${_gpuxtb_web_3dmol}"
  COMMAND npm ci --ignore-scripts --no-audit --no-fund
  WORKING_DIRECTORY "${_gpuxtb_web_src}"
  DEPENDS
    "${_gpuxtb_web_src}/package.json"
    "${_gpuxtb_web_src}/package-lock.json"
  COMMENT "Fetching npm build dependencies (3Dmol.js 2.5.5)"
  VERBATIM)

# ---------------------------------------------------------------------------
# 1) LAPACKE/CBLAS side module, dlopen'ed by gpuxtb's eigensolver as
#    "libscipy_openblas.so"; preloaded into the main module's virtual FS
#    below, so browser main-thread dlopen never needs a synchronous fetch.
# ---------------------------------------------------------------------------
set(_gpuxtb_web_side "${_gpuxtb_web_site}/libscipy_openblas.so")
add_custom_command(
  OUTPUT "${_gpuxtb_web_side}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_gpuxtb_web_site}"
  COMMAND ${CMAKE_C_COMPILER}
          "${_gpuxtb_web_src}/wasm/linalg.c"
          -o "${_gpuxtb_web_side}"
          -sSIDE_MODULE=2 -m64 -O2
          "-sEXPORTED_FUNCTIONS=[\"_openblas_get_config\",\"_openblas_set_num_threads_local\",\"_LAPACKE_dpotrf_work\",\"_LAPACKE_dpocon_work\",\"_LAPACKE_dsyevd_work\",\"_cblas_dgemm\",\"_cblas_dtrsm\"]"
  DEPENDS "${_gpuxtb_web_src}/wasm/linalg.c"
  COMMENT "Building wasm64 LAPACKE side module"
  VERBATIM)

# ---------------------------------------------------------------------------
# 2) Web adapter translation unit.
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT "${_gpuxtb_web_obj}"
  COMMAND ${CMAKE_C_COMPILER}
          -m64 -fPIC -O2
          "${_gpuxtb_web_include}"
          -c "${_gpuxtb_web_src}/gpuxtb_web.c"
          -o "${_gpuxtb_web_obj}"
  DEPENDS "${_gpuxtb_web_src}/gpuxtb_web.c"
  COMMENT "Compiling wasm64 web adapter"
  VERBATIM)

# ---------------------------------------------------------------------------
# 3) wasm64 main module: gpuxtb static library + web adapter, single-threaded
#    (no SharedArrayBuffer -> GitHub Pages works without COOP/COEP headers),
#    dynamic-linking main module, ES6 MODULARIZE factory, side module
#    preloaded into the virtual filesystem.
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT
    "${_gpuxtb_web_site}/gpuxtb_web.js"
    "${_gpuxtb_web_site}/gpuxtb_web.wasm"
    "${_gpuxtb_web_site}/gpuxtb_web.data"
  COMMAND ${CMAKE_COMMAND} -E env "EMCC_FORCE_STDLIBS=1"
          ${CMAKE_CXX_COMPILER}
          "${_gpuxtb_web_obj}"
          $<TARGET_FILE:gpuxtb>
          "${_gpuxtb_web_include}"
          -o "${_gpuxtb_web_site}/gpuxtb_web.js"
          -m64 -O2 -sMAIN_MODULE=2
          -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createGpuxTbModule
          -sSTACK_SIZE=8MB -sALLOW_MEMORY_GROWTH
          -sFORCE_FILESYSTEM
          "--preload-file=${_gpuxtb_web_side}@/libscipy_openblas.so"
          "-sEXPORTED_FUNCTIONS=_gpuxtb_web_compute,_gpuxtb_web_optimize,_gpuxtb_web_set_optimize_step_cb,_gpuxtb_web_version,_malloc,_free"
          "-sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToUTF8OnStack,ccall,cwrap,addFunction,wasmMemory"
  DEPENDS
    gpuxtb
    "${_gpuxtb_web_side}"
    "${_gpuxtb_web_obj}"
  COMMENT "Linking wasm64 main module (gpuxtb_web.js)"
  VERBATIM)

# ---------------------------------------------------------------------------
# 4) Static assets (front end + vendored 3Dmol.js browser bundle).
# ---------------------------------------------------------------------------
set(_gpuxtb_web_copy_deps
    "${_gpuxtb_web_src}/index.html"
    "${_gpuxtb_web_src}/style.css"
    "${_gpuxtb_web_src}/app.js"
    "${_gpuxtb_web_src}/worker.js"
    "${_gpuxtb_web_3dmol}")

add_custom_command(
  OUTPUT
    "${_gpuxtb_web_site}/index.html"
    "${_gpuxtb_web_site}/style.css"
    "${_gpuxtb_web_site}/app.js"
    "${_gpuxtb_web_site}/worker.js"
    "${_gpuxtb_web_site}/vendor/3Dmol-min.js"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_gpuxtb_web_site}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_gpuxtb_web_src}/index.html" "${_gpuxtb_web_site}/index.html"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_gpuxtb_web_src}/style.css" "${_gpuxtb_web_site}/style.css"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_gpuxtb_web_src}/app.js" "${_gpuxtb_web_site}/app.js"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_gpuxtb_web_src}/worker.js" "${_gpuxtb_web_site}/worker.js"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_gpuxtb_web_site}/vendor"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_gpuxtb_web_3dmol}" "${_gpuxtb_web_site}/vendor/3Dmol-min.js"
  DEPENDS ${_gpuxtb_web_copy_deps}
  COMMENT "Staging web static assets"
  VERBATIM)

# ---------------------------------------------------------------------------
# Aggregate target: build every site output with `cmake --build`.
# ---------------------------------------------------------------------------
add_custom_target(
  gpuxtb_web ALL
  DEPENDS
    "${_gpuxtb_web_site}/gpuxtb_web.js"
    "${_gpuxtb_web_site}/gpuxtb_web.wasm"
    "${_gpuxtb_web_site}/gpuxtb_web.data"
    "${_gpuxtb_web_site}/index.html"
    "${_gpuxtb_web_site}/style.css"
    "${_gpuxtb_web_site}/app.js"
    "${_gpuxtb_web_site}/worker.js"
    "${_gpuxtb_web_site}/vendor/3Dmol-min.js")

message(STATUS "gpuxtb web demo output: ${_gpuxtb_web_site}")

# xtbloom GitHub Pages WASM web demo.
#
# Enabled with -DXTBLOOM_BUILD_WEB_DEMO=ON under the Emscripten toolchain
# (emcmake). The whole pipeline — npm-fetch the 3Dmol.js browser bundle,
# build the dlopen-able LAPACKE side module, compile the web adapter, link the
# target-width main module, and stage the static assets — is expressed as custom
# commands so a single `cmake --build` produces the deployable site in
#   <build>/web/site
#
# The deterministic, version-pinned build inputs live in this directory:
#   package.json / package-lock.json   (3dmol.js 2.5.5, BSD-3-Clause)
#   wasm/linalg.c                      (minimal LP64 LAPACKE/CBLAS side module)
#   xtbloom_web.c                       (XYZ/JSON C adapter + L-BFGS optimizer)
#   index.html / style.css / app.js / app_helpers.js / worker.js
#   smiles_helpers.js / smiles_worker.js
# See THIRD_PARTY_NOTICES.md for provenance. Outputs under <build> are build
# artifacts and never committed.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message(FATAL_ERROR
    "XTBLOOM_BUILD_WEB_DEMO=ON requires configure with the Emscripten toolchain "
    "(emcmake cmake ...).")
endif()

set(_xtbloom_web_src "${CMAKE_CURRENT_SOURCE_DIR}")
set(_xtbloom_web_site "${CMAKE_CURRENT_BINARY_DIR}/site")
set(_xtbloom_web_obj "${CMAKE_CURRENT_BINARY_DIR}/xtbloom_web.o")
# The adapter is compiled by a custom emcc command rather than a CMake target,
# so propagate both halves of the public-header build interface explicitly.
set(_xtbloom_web_source_include "-I${CMAKE_SOURCE_DIR}/include")
set(_xtbloom_web_generated_include "-I${XTBLOOM_GENERATED_INCLUDE_DIR}")

# The core library's data model is selected before this subdirectory is
# processed, through the Emscripten compiler flags used at configure time.
# Derive every custom command from CMake's detected pointer size so the core,
# adapter, main module, and dlopen side module cannot accidentally mix widths.
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(_xtbloom_web_arch "wasm32")
  set(_xtbloom_web_arch_flag "-m32")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_xtbloom_web_arch "wasm64")
  set(_xtbloom_web_arch_flag "-m64")
else()
  message(FATAL_ERROR
    "The web demo supports only 32-bit or 64-bit WebAssembly pointers; "
    "CMAKE_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}.")
endif()

# ---------------------------------------------------------------------------
# 0) Fetch build-time npm dependencies (3Dmol.js). Re-runs only when
#    package.json / package-lock.json are newer than the resolved bundle.
# ---------------------------------------------------------------------------
set(_xtbloom_web_3dmol
    "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/3dmol/build/3Dmol-min.js")
add_custom_command(
  OUTPUT "${_xtbloom_web_3dmol}"
  COMMAND npm ci --ignore-scripts --no-audit --no-fund
  WORKING_DIRECTORY "${_xtbloom_web_src}"
  DEPENDS
    "${_xtbloom_web_src}/package.json"
    "${_xtbloom_web_src}/package-lock.json"
  COMMENT "Fetching npm build dependencies (3Dmol.js 2.5.5)"
  VERBATIM)

# ---------------------------------------------------------------------------
# 1) LAPACKE/CBLAS side module, dlopen'ed by xtbloom's eigensolver as
#    "libscipy_openblas.so"; preloaded into the main module's virtual FS
#    below, so browser main-thread dlopen never needs a synchronous fetch.
# ---------------------------------------------------------------------------
set(_xtbloom_web_side "${CMAKE_CURRENT_BINARY_DIR}/libscipy_openblas.so")
add_custom_command(
  OUTPUT "${_xtbloom_web_side}"
  COMMAND ${CMAKE_C_COMPILER}
          "${_xtbloom_web_src}/wasm/linalg.c"
          -o "${_xtbloom_web_side}"
          -sSIDE_MODULE=2 "${_xtbloom_web_arch_flag}" -O2
          "-sEXPORTED_FUNCTIONS=[\"_openblas_get_config\",\"_openblas_set_num_threads_local\",\"_LAPACKE_dpotrf_work\",\"_LAPACKE_dpocon_work\",\"_LAPACKE_dsyevd_work\",\"_cblas_dgemm\",\"_cblas_dtrsm\"]"
  DEPENDS "${_xtbloom_web_src}/wasm/linalg.c"
  COMMENT "Building ${_xtbloom_web_arch} LAPACKE side module"
  VERBATIM)

# ---------------------------------------------------------------------------
# 2) Web adapter translation unit.
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT "${_xtbloom_web_obj}"
  COMMAND ${CMAKE_C_COMPILER}
          "${_xtbloom_web_arch_flag}" -fPIC -O2
          "${_xtbloom_web_generated_include}"
          "${_xtbloom_web_source_include}"
          -c "${_xtbloom_web_src}/xtbloom_web.c"
          -o "${_xtbloom_web_obj}"
  DEPENDS "${_xtbloom_web_src}/xtbloom_web.c"
  COMMENT "Compiling ${_xtbloom_web_arch} web adapter"
  VERBATIM)

# ---------------------------------------------------------------------------
# 3) Target-width main module: xtbloom static library + web adapter,
#    single-threaded
#    (no SharedArrayBuffer -> GitHub Pages works without COOP/COEP headers),
#    dynamic-linking main module, ES6 MODULARIZE factory, side module
#    preloaded into the virtual filesystem.
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT
    "${_xtbloom_web_site}/xtbloom_web.js"
    "${_xtbloom_web_site}/xtbloom_web.wasm"
    "${_xtbloom_web_site}/xtbloom_web.data"
  COMMAND ${CMAKE_COMMAND} -E env "EMCC_FORCE_STDLIBS=1"
          ${CMAKE_CXX_COMPILER}
          "${_xtbloom_web_obj}"
          $<TARGET_FILE:xtbloom>
          "${_xtbloom_web_generated_include}"
          "${_xtbloom_web_source_include}"
          -o "${_xtbloom_web_site}/xtbloom_web.js"
          "${_xtbloom_web_arch_flag}" -O2 -sMAIN_MODULE=2
          -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createXTBloomModule
          -sSTACK_SIZE=8MB -sALLOW_MEMORY_GROWTH
          -sFORCE_FILESYSTEM
          "--preload-file=${_xtbloom_web_side}@/libscipy_openblas.so"
          "-sEXPORTED_FUNCTIONS=_xtbloom_web_compute,_xtbloom_web_optimize,_xtbloom_web_set_optimize_step_cb,_xtbloom_web_version,_malloc,_free"
          "-sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToUTF8OnStack,ccall,cwrap,addFunction,wasmMemory"
  DEPENDS
    xtbloom
    "${_xtbloom_web_side}"
    "${_xtbloom_web_obj}"
  COMMENT "Linking ${_xtbloom_web_arch} main module (xtbloom_web.js)"
  VERBATIM)

# ---------------------------------------------------------------------------
# 4) Static assets (front end + vendored 3Dmol.js browser bundle).
# ---------------------------------------------------------------------------
set(_xtbloom_web_copy_deps
    "${_xtbloom_web_src}/index.html"
    "${_xtbloom_web_src}/style.css"
    "${_xtbloom_web_src}/app.js"
    "${_xtbloom_web_src}/app_helpers.js"
    "${_xtbloom_web_src}/worker.js"
    "${_xtbloom_web_src}/smiles_helpers.js"
    "${_xtbloom_web_src}/smiles_worker.js"
    "${CMAKE_SOURCE_DIR}/docs/assets/xtbloom-mark.svg"
    "${_xtbloom_web_3dmol}")

add_custom_command(
  OUTPUT
    "${_xtbloom_web_site}/index.html"
    "${_xtbloom_web_site}/style.css"
    "${_xtbloom_web_site}/app.js"
    "${_xtbloom_web_site}/app_helpers.js"
    "${_xtbloom_web_site}/worker.js"
    "${_xtbloom_web_site}/smiles_helpers.js"
    "${_xtbloom_web_site}/smiles_worker.js"
    "${_xtbloom_web_site}/xtbloom-mark.svg"
    "${_xtbloom_web_site}/vendor/3Dmol-min.js"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_xtbloom_web_site}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/index.html" "${_xtbloom_web_site}/index.html"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/style.css" "${_xtbloom_web_site}/style.css"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/app.js" "${_xtbloom_web_site}/app.js"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/app_helpers.js" "${_xtbloom_web_site}/app_helpers.js"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/worker.js" "${_xtbloom_web_site}/worker.js"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/smiles_helpers.js"
          "${_xtbloom_web_site}/smiles_helpers.js"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/smiles_worker.js"
          "${_xtbloom_web_site}/smiles_worker.js"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/docs/assets/xtbloom-mark.svg"
          "${_xtbloom_web_site}/xtbloom-mark.svg"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${_xtbloom_web_site}/vendor"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_3dmol}" "${_xtbloom_web_site}/vendor/3Dmol-min.js"
  DEPENDS ${_xtbloom_web_copy_deps}
  COMMENT "Staging web static assets"
  VERBATIM)

# ---------------------------------------------------------------------------
# 5) Legal and provenance payload. The site conveys the GPL-covered WASM
#    engine, LGPL/Apache-derived parameter data, and a 3Dmol bundle containing
#    npm transitive code, so recipients must receive every applicable notice.
# ---------------------------------------------------------------------------
set(_xtbloom_web_legal_sources
    "${CMAKE_SOURCE_DIR}/LICENSE"
    "${CMAKE_SOURCE_DIR}/CUDA_MKL_LINKING_EXCEPTION"
    "${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    "${CMAKE_SOURCE_DIR}/LICENSES/3Dmol.js-BSD-3-Clause.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/openchemlib-BSD-3-Clause.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/Apache-2.0.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/LGPL-3.0-or-later.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/MIT.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/MPL-2.0.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/array-api-compat-MIT.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/iobuffer-MIT.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/netcdfjs-MIT.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/pako-MIT.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/pako-Zlib.txt"
    "${CMAKE_SOURCE_DIR}/LICENSES/upng-js-MIT.txt"
    "${CMAKE_SOURCE_DIR}/data/parameters/d4.NOTICE"
    "${CMAKE_SOURCE_DIR}/data/parameters/licenses/dftd4-COPYING"
    "${CMAKE_SOURCE_DIR}/data/parameters/licenses/dftd4-COPYING.LESSER"
    "${CMAKE_SOURCE_DIR}/data/parameters/licenses/mctc-lib-LICENSE"
    "${CMAKE_SOURCE_DIR}/data/parameters/manifest.json"
    "${CMAKE_SOURCE_DIR}/data/parameters/sto_manifest.json"
    "${CMAKE_SOURCE_DIR}/data/parameters/spin_manifest.json"
    "${CMAKE_SOURCE_DIR}/data/parameters/d4_manifest.json"
    "${CMAKE_SOURCE_DIR}/data/parameters/mctc_manifest.json"
    "${_xtbloom_web_src}/openchemlib_manifest.json")

set(_xtbloom_web_legal_outputs
    "${_xtbloom_web_site}/LICENSE"
    "${_xtbloom_web_site}/CUDA_MKL_LINKING_EXCEPTION"
    "${_xtbloom_web_site}/THIRD_PARTY_NOTICES.md"
    "${_xtbloom_web_site}/LICENSES/3Dmol.js-BSD-3-Clause.txt"
    "${_xtbloom_web_site}/LICENSES/openchemlib-BSD-3-Clause.txt"
    "${_xtbloom_web_site}/LICENSES/Apache-2.0.txt"
    "${_xtbloom_web_site}/LICENSES/LGPL-3.0-or-later.txt"
    "${_xtbloom_web_site}/LICENSES/MIT.txt"
    "${_xtbloom_web_site}/LICENSES/MPL-2.0.txt"
    "${_xtbloom_web_site}/LICENSES/array-api-compat-MIT.txt"
    "${_xtbloom_web_site}/LICENSES/pako-MIT.txt"
    "${_xtbloom_web_site}/LICENSES/pako-Zlib.txt"
    "${_xtbloom_web_site}/LICENSES/iobuffer-MIT.txt"
    "${_xtbloom_web_site}/LICENSES/netcdfjs-MIT.txt"
    "${_xtbloom_web_site}/LICENSES/upng-js-MIT.txt"
    "${_xtbloom_web_site}/LICENSES/parameters/d4.NOTICE"
    "${_xtbloom_web_site}/LICENSES/parameters/dftd4-COPYING"
    "${_xtbloom_web_site}/LICENSES/parameters/dftd4-COPYING.LESSER"
    "${_xtbloom_web_site}/LICENSES/parameters/mctc-lib-LICENSE"
    "${_xtbloom_web_site}/provenance/parameters/manifest.json"
    "${_xtbloom_web_site}/provenance/parameters/sto_manifest.json"
    "${_xtbloom_web_site}/provenance/parameters/spin_manifest.json"
    "${_xtbloom_web_site}/provenance/parameters/d4_manifest.json"
    "${_xtbloom_web_site}/provenance/parameters/mctc_manifest.json"
    "${_xtbloom_web_site}/provenance/openchemlib_manifest.json")

add_custom_command(
  OUTPUT ${_xtbloom_web_legal_outputs}
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${_xtbloom_web_site}/LICENSES/parameters"
          "${_xtbloom_web_site}/provenance/parameters"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/LICENSE" "${_xtbloom_web_site}/LICENSE"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/CUDA_MKL_LINKING_EXCEPTION"
          "${_xtbloom_web_site}/CUDA_MKL_LINKING_EXCEPTION"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
          "${_xtbloom_web_site}/THIRD_PARTY_NOTICES.md"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/LICENSES" "${_xtbloom_web_site}/LICENSES"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_SOURCE_DIR}/data/parameters/licenses"
          "${_xtbloom_web_site}/LICENSES/parameters"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/data/parameters/d4.NOTICE"
          "${_xtbloom_web_site}/LICENSES/parameters/d4.NOTICE"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_SOURCE_DIR}/data/parameters/manifest.json"
          "${CMAKE_SOURCE_DIR}/data/parameters/sto_manifest.json"
          "${CMAKE_SOURCE_DIR}/data/parameters/spin_manifest.json"
          "${CMAKE_SOURCE_DIR}/data/parameters/d4_manifest.json"
          "${CMAKE_SOURCE_DIR}/data/parameters/mctc_manifest.json"
          "${_xtbloom_web_site}/provenance/parameters"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_xtbloom_web_src}/openchemlib_manifest.json"
          "${_xtbloom_web_site}/provenance/openchemlib_manifest.json"
  DEPENDS ${_xtbloom_web_legal_sources}
  COMMENT "Staging web licenses, notices, and parameter provenance"
  VERBATIM)

# ---------------------------------------------------------------------------
# Aggregate target: build every site output with `cmake --build`.
# ---------------------------------------------------------------------------
add_custom_target(
  xtbloom_web ALL
  DEPENDS
    "${_xtbloom_web_site}/xtbloom_web.js"
    "${_xtbloom_web_site}/xtbloom_web.wasm"
    "${_xtbloom_web_site}/xtbloom_web.data"
    "${_xtbloom_web_site}/index.html"
    "${_xtbloom_web_site}/style.css"
    "${_xtbloom_web_site}/app.js"
    "${_xtbloom_web_site}/app_helpers.js"
    "${_xtbloom_web_site}/worker.js"
    "${_xtbloom_web_site}/smiles_helpers.js"
    "${_xtbloom_web_site}/smiles_worker.js"
    "${_xtbloom_web_site}/xtbloom-mark.svg"
    "${_xtbloom_web_site}/vendor/3Dmol-min.js"
    ${_xtbloom_web_legal_outputs})

message(STATUS "xTBloom ${_xtbloom_web_arch} web demo output: ${_xtbloom_web_site}")

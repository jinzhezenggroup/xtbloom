# Generate the small cache-validation manifest consumed by bootstrap.js and
# app.js. Sizes and hashes describe decoded file bytes, so browser progress
# remains exact even when the HTTP transport compresses JavaScript responses.
if(NOT DEFINED ASSET_DIR OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "ASSET_DIR and OUTPUT are required")
endif()

set(_asset_specs
    "app|app.js"
    "c60|c60_case.js"
    "worker|worker.js"
    "helpers|app_helpers.js"
    "smiles_worker|smiles_worker.js"
    "smiles_helpers|smiles_helpers.js"
    "module|xtbloom_web.js"
    "wasm|xtbloom_web.wasm"
    "data|xtbloom_web.data")
set(_entries "")
set(_version_material "")
set(_separator "")

foreach(_spec IN LISTS _asset_specs)
  string(REPLACE "|" ";" _fields "${_spec}")
  list(GET _fields 0 _id)
  list(GET _fields 1 _relative_path)
  set(_path "${ASSET_DIR}/${_relative_path}")
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "engine manifest input is missing: ${_path}")
  endif()
  file(SIZE "${_path}" _size)
  file(SHA256 "${_path}" _sha256)
  string(APPEND _version_material
         "${_id}:${_relative_path}:${_size}:${_sha256}\n")
  string(APPEND _entries
         "${_separator}    {\"id\": \"${_id}\", \"path\": \"${_relative_path}\", "
         "\"bytes\": ${_size}, \"sha256\": \"${_sha256}\"}")
  set(_separator ",\n")
endforeach()

string(SHA256 _version "${_version_material}")
file(WRITE "${OUTPUT}"
     "{\n"
     "  \"schema_version\": 1,\n"
     "  \"version\": \"${_version}\",\n"
     "  \"assets\": [\n${_entries}\n  ]\n"
     "}\n")

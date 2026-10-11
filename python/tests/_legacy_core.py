"""Build a minimal ABI-compatible core that predates compute-options V3."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def build_legacy_core(directory: Path) -> Path:
    """Compile a synchronous V2-only test core with the Python-required symbols."""
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise RuntimeError("a C compiler is required for the legacy-core test")
    source = directory / "legacy_xtbloom.c"
    library = directory / "libxtbloom_legacy.so"
    source.write_text(
        r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int init_full(void *value, size_t size) {
  memset(value, 0, size);
  ((uint32_t *)value)[0] = (uint32_t)size;
  ((uint32_t *)value)[1] = 1u;
  return 0;
}

const char *xtbloom_get_last_error(void) { return ""; }
const char *xtbloom_version_string(void) { return "0.1.1"; }
const char *xtbloom_status_string(int32_t status) {
  (void)status;
  return "success";
}
int xtbloom_context_options_init(void *value, size_t size) {
  return init_full(value, size);
}
int xtbloom_batch_init(void *value, size_t size) { return init_full(value, size); }
int xtbloom_compute_options_init(void *value, size_t size) {
  size_t legacy_size = size < 56u ? size : 56u;
  memset(value, 0, legacy_size);
  ((uint32_t *)value)[0] = (uint32_t)size;
  ((uint32_t *)value)[1] = 1u;
  return 0;
}
int xtbloom_batch_result_init(void *value, size_t size) {
  return init_full(value, size);
}
int xtbloom_workspace_query_init(void *value, size_t size) {
  return init_full(value, size);
}
int xtbloom_result_owner_options_init(void *value, size_t size) {
  return init_full(value, size);
}
int xtbloom_result_owner_create(const void *options, void **owner) {
  (void)options;
  *owner = (void *)(uintptr_t)1u;
  return 0;
}
int xtbloom_result_owner_buffer(void *owner, void *buffer) {
  (void)owner;
  return init_full(buffer, 24u);
}
void xtbloom_result_owner_retain(void *owner) { (void)owner; }
void xtbloom_result_owner_release(void *owner) { (void)owner; }
int xtbloom_result_owner_export_dltensor(
    void *owner, void *view, int stream, void **managed) {
  (void)owner; (void)view; (void)stream; (void)managed;
  return 0;
}
int xtbloom_plan_create(
    void *context, const void *batch, const void *options, void **plan) {
  (void)context; (void)batch; (void)options;
  *plan = (void *)(uintptr_t)1u;
  return 0;
}
void xtbloom_plan_destroy(void *plan) { (void)plan; }
int xtbloom_plan_query_workspace(void *plan, void *query) {
  (void)plan; (void)query;
  return 0;
}
int xtbloom_plan_compute(
    void *plan, const void *batch, const void *options, void *result) {
  (void)plan; (void)batch; (void)options; (void)result;
  return 0;
}
int xtbloom_context_create(const void *options, void **context) {
  (void)options;
  *context = (void *)(uintptr_t)1u;
  return 0;
}
void xtbloom_context_destroy(void *context) { (void)context; }
int32_t xtbloom_context_get_backend(const void *context) { (void)context; return 1; }
int32_t xtbloom_context_get_device_id(const void *context) { (void)context; return -1; }
int xtbloom_compute(
    void *context, const void *batch, const void *options, void *result) {
  (void)context; (void)batch; (void)options; (void)result;
  return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    return library

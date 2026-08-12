// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "runtime/model_registry.hpp"

#include <array>
#include <string>

namespace xtbloom::detail {
namespace {

constexpr std::array<ModelDescriptor, 2> kModels{{
    {XTBLOOM_MODEL_GFN1_XTB, ModelFamily::kGfn1, "GFN1-xTB", false, false},
    {XTBLOOM_MODEL_GFN2_XTB, ModelFamily::kGfn2, "GFN2-xTB", true, true},
}};

}  // namespace

const ModelDescriptor* find_model_descriptor(xtbloom_model_t model) noexcept {
  for (const ModelDescriptor& descriptor : kModels) {
    if (descriptor.tag == model) {
      return &descriptor;
    }
  }
  return nullptr;
}

xtbloom_status_t validate_model_dispatch(xtbloom_model_t model, xtbloom_backend_t backend,
                                         std::string& error) {
  const ModelDescriptor* descriptor = find_model_descriptor(model);
  if (descriptor == nullptr) {
    error = "compute options contain an unknown model value";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  bool implemented = false;
  if (backend == XTBLOOM_BACKEND_CPU) {
    implemented = descriptor->cpu_implemented;
  } else if (backend == XTBLOOM_BACKEND_CUDA) {
    implemented = descriptor->cuda_implemented;
  } else {
    error = "model dispatch requires a resolved CPU or CUDA backend";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  if (!implemented) {
    error = std::string(descriptor->canonical_name) +
            " is reserved by the ABI but is not implemented yet";
    return XTBLOOM_STATUS_NOT_SUPPORTED;
  }
  error.clear();
  return XTBLOOM_STATUS_SUCCESS;
}

}  // namespace xtbloom::detail

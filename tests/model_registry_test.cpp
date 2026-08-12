// SPDX-License-Identifier: GPL-3.0-or-later

#include "runtime/model_registry.hpp"

#include <cstring>
#include <string>

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

int main() {
  using xtbloom::detail::find_model_descriptor;
  using xtbloom::detail::ModelFamily;
  using xtbloom::detail::validate_model_dispatch;

  const auto* gfn1 = find_model_descriptor(XTBLOOM_MODEL_GFN1_XTB);
  CHECK(gfn1 != nullptr);
  CHECK(gfn1->family == ModelFamily::kGfn1);
  CHECK(std::strcmp(gfn1->canonical_name, "GFN1-xTB") == 0);
  CHECK(!gfn1->cpu_implemented);
  CHECK(!gfn1->cuda_implemented);

  const auto* gfn2 = find_model_descriptor(XTBLOOM_MODEL_GFN2_XTB);
  CHECK(gfn2 != nullptr);
  CHECK(gfn2->family == ModelFamily::kGfn2);
  CHECK(std::strcmp(gfn2->canonical_name, "GFN2-xTB") == 0);
  CHECK(gfn2->cpu_implemented);
  CHECK(gfn2->cuda_implemented);
  CHECK(find_model_descriptor(static_cast<xtbloom_model_t>(99)) == nullptr);

  std::string error;
  CHECK(validate_model_dispatch(XTBLOOM_MODEL_GFN2_XTB, XTBLOOM_BACKEND_CPU, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());
  CHECK(validate_model_dispatch(XTBLOOM_MODEL_GFN2_XTB, XTBLOOM_BACKEND_CUDA, error) ==
        XTBLOOM_STATUS_SUCCESS);
  CHECK(error.empty());

  CHECK(validate_model_dispatch(XTBLOOM_MODEL_GFN1_XTB, XTBLOOM_BACKEND_CPU, error) ==
        XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(error.find("GFN1-xTB") != std::string::npos);
  CHECK(validate_model_dispatch(XTBLOOM_MODEL_GFN1_XTB, XTBLOOM_BACKEND_CUDA, error) ==
        XTBLOOM_STATUS_NOT_SUPPORTED);
  CHECK(error.find("GFN1-xTB") != std::string::npos);

  CHECK(validate_model_dispatch(static_cast<xtbloom_model_t>(99), XTBLOOM_BACKEND_CPU, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("unknown model") != std::string::npos);
  CHECK(validate_model_dispatch(XTBLOOM_MODEL_GFN2_XTB, XTBLOOM_BACKEND_ROCM, error) ==
        XTBLOOM_STATUS_INVALID_ARGUMENT);
  CHECK(error.find("resolved CPU or CUDA") != std::string::npos);
  return 0;
}

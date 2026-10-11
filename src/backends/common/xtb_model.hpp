#ifndef XTBLOOM_BACKENDS_COMMON_XTB_MODEL_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_BACKENDS_COMMON_XTB_MODEL_HPP

#include <cstdint>

namespace xtbloom::detail {

/* Internal model selector shared by backend-neutral host/device descriptors. */
enum class XtbModelFlavor : std::uint32_t {
  kGfn1 = 1u,
  kGfn2 = 2u,
};

[[nodiscard]] constexpr bool valid_xtb_model_flavor(XtbModelFlavor model) noexcept {
  return model == XtbModelFlavor::kGfn1 || model == XtbModelFlavor::kGfn2;
}

}  // namespace xtbloom::detail

#endif  // XTBLOOM_BACKENDS_COMMON_XTB_MODEL_HPP

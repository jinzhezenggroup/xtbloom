#ifndef XTBLOOM_RUNTIME_MODEL_REGISTRY_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define XTBLOOM_RUNTIME_MODEL_REGISTRY_HPP

#include <cstdint>
#include <string>

#include "xtbloom/xtbloom.h"

namespace xtbloom::detail {

/*
 * Internal physics and parameter-table identity. This is deliberately
 * separate from public ABI tags and backend capability: registering a family
 * makes its metadata selectable without implying that it can execute yet.
 */
enum class ModelFamily : std::uint8_t {
  kGfn1,
  kGfn2,
};

/*
 * Internal model registration record.
 *
 * The stable ABI reserves model tags independently of implementation state.
 * Keeping the dispatch capability here prevents public validation, plans, and
 * synchronous/asynchronous execution from acquiring separate notions of what
 * a known or executable model means. A model becomes public only after its
 * backend entry is enabled together with the complete scientific path.
 */
struct ModelDescriptor {
  xtbloom_model_t tag;
  ModelFamily family;
  const char* canonical_name;
  bool cpu_implemented;
  bool cuda_implemented;
};

[[nodiscard]] const ModelDescriptor* find_model_descriptor(xtbloom_model_t model) noexcept;

/*
 * Validate only the model-to-backend dispatch boundary. Descriptor validation
 * remains responsible for request fields and pointer ownership. A reserved
 * but unfinished model returns NOT_SUPPORTED so callers can distinguish it
 * from an unknown ABI tag without any caller-output mutation.
 */
[[nodiscard]] xtbloom_status_t validate_model_dispatch(xtbloom_model_t model,
                                                       xtbloom_backend_t backend,
                                                       std::string& error);

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_MODEL_REGISTRY_HPP

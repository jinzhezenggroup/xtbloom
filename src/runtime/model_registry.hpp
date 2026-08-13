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
 * Concrete backend execution route. Capability is derived from this route,
 * rather than stored as an independent boolean that could be enabled while a
 * caller still falls through to another model's executor.
 */
enum class ModelBackendRoute : std::uint8_t {
  kUnavailable,
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
  ModelBackendRoute cpu_route;
  ModelBackendRoute cuda_route;
};

[[nodiscard]] const ModelDescriptor* find_model_descriptor(xtbloom_model_t model) noexcept;

/* Return the concrete executor route for one resolved backend. */
[[nodiscard]] ModelBackendRoute model_backend_route(const ModelDescriptor& descriptor,
                                                    xtbloom_backend_t backend) noexcept;

/*
 * Validate only the model-to-backend dispatch boundary. Descriptor validation
 * remains responsible for request fields and pointer ownership. A reserved
 * but unfinished model returns NOT_SUPPORTED so callers can distinguish it
 * from an unknown ABI tag without any caller-output mutation.
 */
[[nodiscard]] xtbloom_status_t validate_model_dispatch(xtbloom_model_t model,
                                                       xtbloom_backend_t backend,
                                                       std::string& error,
                                                       ModelBackendRoute* route = nullptr);

}  // namespace xtbloom::detail

#endif  // XTBLOOM_RUNTIME_MODEL_REGISTRY_HPP

// Compile-time boundary checks for the generated GFN1 host/device tables.
#include "data/parameters/gfn1.hpp"

using namespace xtbloom::parameters::gfn1;

static_assert(kElementCount == 86u);
static_assert(kShellCount == 237u);
static_assert(kPairScaleOverrides.size() == 869u);
static_assert(find_element(0u) == nullptr);
static_assert(find_element(1u)->atomic_number == 1u);
static_assert(find_element(86u)->atomic_number == 86u);
static_assert(find_element(87u) == nullptr);
static_assert(pair_scale(1u, 1u) == 0.96);
static_assert(pair_scale(1u, 5u) == 0.95);
static_assert(pair_scale(1u, 6u) == 1.0);
static_assert(pair_scale(78u, 1u) == 0.8);

int main() {
  const auto* hydrogen = find_element(1u);
  if (hydrogen == nullptr || hydrogen->shell_count != 2u) {
    return 1;
  }
  const auto& first = kShells[hydrogen->shell_offset];
  const auto& second = kShells[hydrogen->shell_offset + 1u];
  return first.angular_momentum == second.angular_momentum && first.is_valence &&
                 !second.is_valence && second.reference_occupation == 0.0
             ? 0
             : 1;
}

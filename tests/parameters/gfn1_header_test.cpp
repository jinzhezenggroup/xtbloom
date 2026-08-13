// Compile-time boundary checks for the generated GFN1 host/device tables.
#include "data/parameters/gfn1.hpp"

using namespace xtbloom::parameters::gfn1;

constexpr bool shell_ranges_are_contiguous_and_complete() {
  std::size_t next_offset = 0u;
  for (const auto& element : kElements) {
    if (element.shell_offset != next_offset ||
        static_cast<std::size_t>(element.shell_offset) + element.shell_count > kShells.size()) {
      return false;
    }
    next_offset += element.shell_count;
  }
  return next_offset == kShells.size();
}

static_assert(kElementCount == 86u);
static_assert(kShellCount == 237u);
static_assert(kPairScaleOverrides.size() == 869u);
static_assert(shell_ranges_are_contiguous_and_complete());
static_assert(kGlobal.coordination_number_model == 1u);
static_assert(kGlobal.coordination_steepness == 16.0);
static_assert(kGlobal.coordination_cutoff_bohr == 25.0);
static_assert(kGlobal.coordination_directed_factor == 1.0);
static_assert(!kGlobal.coordination_has_maximum_cn_cutoff);
static_assert(kGlobal.coordination_cutoff_inclusive);
static_assert(kGlobal.coordination_coincident_cutoff_inclusive);
static_assert(find_element(0u) == nullptr);
static_assert(find_element(1u)->atomic_number == 1u);
static_assert(find_element(1u)->electronegativity == 2.2);
static_assert(find_element(1u)->atomic_radius_bohr > 0.6);
static_assert(find_element(1u)->covalent_radius_bohr > 0.8);
static_assert(kShells[0].level_electronvolt == -10.923452);
static_assert(kShells[0].coordination_number_scale_electronvolt == 0.065540712);
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

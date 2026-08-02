// Compile-time boundary checks for the generated host/device parameter tables.
#include "data/parameters/gfn2.hpp"

using namespace gpuxtb::parameters::gfn2;

static_assert(kElementCount == 86u);
static_assert(find_element(0u) == nullptr);
static_assert(find_element(1u)->atomic_number == 1u);
static_assert(find_element(86u)->atomic_number == 86u);
static_assert(find_element(87u) == nullptr);
static_assert(pair_scale(1u, 86u) == 1.0);

int main() { return kShells.empty() ? 1 : 0; }

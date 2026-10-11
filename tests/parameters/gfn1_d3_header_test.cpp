// Compile-time boundary and symmetry checks for the generated GFN1-D3 tables.
#include "data/parameters/gfn1_d3.hpp"

using namespace xtbloom::parameters::gfn1_d3;

static_assert(kElementCount == 86u);
static_assert(kReferenceCount == 237u);
static_assert(kElementPairCount == 3741u);
static_assert(kReferenceC6Count == 28455u);
static_assert(find_element(0u) == nullptr);
static_assert(find_element(1u)->reference_count == 2u);
static_assert(find_element(86u)->reference_count == 1u);
static_assert(find_element(87u) == nullptr);
static_assert(find_pair(0u, 1u) == nullptr);
static_assert(find_pair(1u, 86u) == find_pair(86u, 1u));
static_assert(reference_cn(1u, 0u) == 0.9118);
static_assert(reference_cn(58u, 0u) == 2.7991);
static_assert(reference_cn(86u, 1u) == 0.0);
static_assert(reference_c6(1u, 0u, 1u, 0u) == 3.0267);
static_assert(reference_c6(1u, 1u, 2u, 0u) == 3.1287);
static_assert(reference_c6(1u, 1u, 86u, 0u) == 55.4345);
static_assert(reference_c6(1u, 1u, 86u, 0u) == reference_c6(86u, 0u, 1u, 1u));
static_assert(reference_c6(1u, 2u, 86u, 0u) == 0.0);
static_assert(vdw_radius(1u, 86u) == vdw_radius(86u, 1u));
static_assert(vdw_radius(0u, 1u) == 0.0);

int main() { return kR4R2.front() > 2.0 && kR4R2.back() > 6.0 && vdw_radius(1u, 1u) > 4.0 ? 0 : 1; }

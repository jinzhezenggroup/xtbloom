// SPDX-License-Identifier: GPL-3.0-or-later

/* Compile this internal runtime header in an otherwise empty translation unit.
 * Clean CI toolchains must not depend on another header defining std::int32_t
 * before the GFN1 cache constructor declaration is parsed. */
#include "runtime/gfn1_cpu_execution.hpp"

int main() { return 0; }

// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef XTBLOOM_DATA_PARAMETERS_GFN1_LEGACY_STO_HPP
#define XTBLOOM_DATA_PARAMETERS_GFN1_LEGACY_STO_HPP

#include <array>

namespace xtbloom::parameters::gfn1 {

/*
 * Legacy STO-6G 4s/4p rows used by xTB 6.7.1 GFN1. The exponent row is
 * shared by both angular momenta, while their contraction coefficients differ.
 * All other GFN1 shells retain the reviewed tblite Stewart table.
 */
inline constexpr std::array<double, 6> kLegacyAlpha6_4sp{{
    1.365346e+00,
    4.393213e-01,
    1.877069e-01,
    9.360270e-02,
    5.052263e-02,
    2.809354e-02,
}};
inline constexpr std::array<double, 6> kLegacyCoeff6_4s{{
    3.775056e-03,
    -5.585965e-02,
    -3.192946e-01,
    -2.764780e-02,
    9.049199e-01,
    3.406258e-01,
}};
inline constexpr std::array<double, 6> kLegacyCoeff6_4p{{
    -7.052075e-03,
    -5.259505e-02,
    -3.773450e-02,
    3.874773e-01,
    5.791672e-01,
    1.221817e-01,
}};

}  // namespace xtbloom::parameters::gfn1

#endif  // XTBLOOM_DATA_PARAMETERS_GFN1_LEGACY_STO_HPP

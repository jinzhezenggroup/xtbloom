// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_MODEL_GFN2_PERIODIC_TOPOLOGY_HPP
#define XTBLOOM_MODEL_GFN2_PERIODIC_TOPOLOGY_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model/gfn2/lattice.hpp"

namespace xtbloom::detail::gfn2 {

/*
 * Wigner--Seitz topology keeps only the closest periodic image or equally
 * close images of an atom pair. Every retained image receives weight 1/n,
 * where n is the number of closest images whose Cartesian distances differ
 * by strictly less than xTB's 0.01-bohr tolerance. This is a
 * topology/canonicalization weight, not a generic real-space pair-energy
 * factor.
 */
constexpr double kWignerSeitzDistanceTolerance = 0.01;
constexpr double kPeriodicTopologyMinimumDistanceSquared = 1.0e-12;

enum class WignerSeitzPairMode : std::int32_t {
  /* Store one lower-triangular atom pair (image_atom <= center_atom). */
  kUnique = 0,
  /* Store every ordered atom pair. */
  kDirected = 1,
};

/*
 * One canonical closest image. translation is applied to image_atom after
 * both input atoms have been wrapped into the central cell. displacement is
 * then r(image_atom + translation) - r(center_atom), in bohr.
 */
struct WignerSeitzImage {
  std::int64_t center_atom = 0;
  std::int64_t image_atom = 0;
  std::array<std::int64_t, 3> translation{};
  std::array<double, 3> displacement{};
  double distance_squared = 0.0;
  double weight = 0.0;
};

namespace periodic_topology_detail {

inline bool representable_geometry_size(std::int64_t atom_count) noexcept {
  if (atom_count < 0) return false;
  const auto count = static_cast<std::uint64_t>(atom_count);
  return count <= std::numeric_limits<std::size_t>::max() / 3u &&
         count <= static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 3u;
}

inline bool is_origin(const LatticeTranslation& translation) noexcept {
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0;
}

enum class ImageGeometryStatus {
  kInsideCutoff,
  kOutsideCutoff,
};

inline ImageGeometryStatus image_geometry(const std::array<double, 3>& center,
                                          const std::array<double, 3>& image,
                                          const LatticeTranslation& translation, double cutoff,
                                          double cutoff_squared,
                                          std::array<double, 3>& displacement,
                                          double& distance_squared) noexcept {
  using lattice_binary64_detail::absolute;
  using lattice_binary64_detail::finite;
  using lattice_binary64_detail::rounded_add;
  using lattice_binary64_detail::rounded_multiply;
  using lattice_binary64_detail::rounded_subtract;

  for (std::size_t component = 0; component < 3u; ++component) {
    const double translated = rounded_add(image[component], translation.cartesian[component]);
    displacement[component] = rounded_subtract(translated, center[component]);
    /*
     * The lattice generator deliberately returns a rectangular superset. A
     * far image can therefore overflow while squaring even though another
     * image of the same pair is safely inside the cutoff. An infinite
     * displacement produced from finite inputs is also provably outside every
     * accepted finite cutoff. All inputs have already been validated as
     * finite, so these operations can produce infinity through overflow but
     * cannot produce NaN.
     */
    if (!finite(displacement[component]) || absolute(displacement[component]) > cutoff) {
      return ImageGeometryStatus::kOutsideCutoff;
    }
  }
  distance_squared = rounded_add(rounded_add(rounded_multiply(displacement[0], displacement[0]),
                                             rounded_multiply(displacement[1], displacement[1])),
                                 rounded_multiply(displacement[2], displacement[2]));
  if (!finite(distance_squared) || distance_squared > cutoff_squared) {
    return ImageGeometryStatus::kOutsideCutoff;
  }
  return ImageGeometryStatus::kInsideCutoff;
}

inline bool within_wigner_seitz_distance_tolerance(double distance_squared,
                                                   double minimum_distance) noexcept {
  using lattice_binary64_detail::absolute;
  using lattice_binary64_detail::rounded_square_root;
  using lattice_binary64_detail::rounded_subtract;

  const double distance = rounded_square_root(distance_squared);
  return absolute(rounded_subtract(distance, minimum_distance)) < kWignerSeitzDistanceTolerance;
}

}  // namespace periodic_topology_detail

/*
 * Build a deterministic Wigner--Seitz image topology from atom-major Cartesian
 * coordinates in bohr. Inputs are wrapped internally, so integer-lattice
 * shifts of individual atoms produce the same canonical topology. For each
 * atom pair, only the closest image(s) inside cutoff are retained. Self pairs
 * exclude the zero translation but retain periodic self images.
 *
 * kUnique emits lower-triangular pairs in (center_atom, image_atom) order;
 * kDirected emits the complete ordered pair map. Within each pair, the origin
 * is first when retained and all other translations follow the lattice
 * generator's lexicographic integer-triplet order.
 *
 * The output is transactional: it remains unchanged on error.
 */
inline xtbloom_status_t make_wigner_seitz_topology(const Lattice3D& lattice,
                                                   std::int64_t atom_count, const double* positions,
                                                   double cutoff, WignerSeitzPairMode pair_mode,
                                                   std::vector<WignerSeitzImage>& topology,
                                                   std::string& error) {
  using namespace periodic_topology_detail;
  using lattice_binary64_detail::finite;
  using lattice_binary64_detail::rounded_multiply;
  using lattice_binary64_detail::rounded_square_root;

  if (!representable_geometry_size(atom_count)) {
    error = "periodic topology atom count is outside the supported host range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (atom_count > 0 && positions == nullptr) {
    error = "periodic topology positions must not be NULL";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (!(cutoff >= 0.0) || !finite(cutoff)) {
    error = "periodic topology cutoff must be finite and nonnegative";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  if (pair_mode != WignerSeitzPairMode::kUnique && pair_mode != WignerSeitzPairMode::kDirected) {
    error = "periodic topology pair mode is invalid";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }
  const double cutoff_squared = rounded_multiply(cutoff, cutoff);
  if (!finite(cutoff_squared)) {
    error = "periodic topology cutoff squared is outside the binary64 range";
    return XTBLOOM_STATUS_INVALID_ARGUMENT;
  }

  try {
    std::vector<LatticeTranslation> translations;
    xtbloom_status_t status = make_lattice_translations(
        lattice, cutoff, LatticeOriginPolicy::kInclude, translations, error);
    if (status != XTBLOOM_STATUS_SUCCESS) return status;

    const auto count = static_cast<std::size_t>(atom_count);
    std::vector<std::array<double, 3>> wrapped;
    if (count > wrapped.max_size()) {
      error = "periodic topology atom count exceeds the wrapped-coordinate vector limit";
      return XTBLOOM_STATUS_INVALID_ARGUMENT;
    }
    wrapped.resize(count);
    for (std::size_t atom = 0; atom < count; ++atom) {
      status = wrap_cartesian(lattice, positions + atom * 3u, wrapped[atom].data(), error);
      if (status != XTBLOOM_STATUS_SUCCESS) return status;
    }

    std::vector<WignerSeitzImage> created;
    for (std::size_t center = 0; center < count; ++center) {
      const std::size_t image_end =
          pair_mode == WignerSeitzPairMode::kDirected ? count : center + 1u;
      for (std::size_t image = 0; image < image_end; ++image) {
        double minimum = std::numeric_limits<double>::infinity();
        for (const auto& translation : translations) {
          if (center == image && is_origin(translation)) continue;
          std::array<double, 3> displacement{};
          double distance_squared = 0.0;
          const ImageGeometryStatus geometry =
              image_geometry(wrapped[center], wrapped[image], translation, cutoff, cutoff_squared,
                             displacement, distance_squared);
          if (geometry == ImageGeometryStatus::kInsideCutoff && distance_squared < minimum) {
            minimum = distance_squared;
          }
        }
        if (!finite(minimum)) continue;
        if (minimum < kPeriodicTopologyMinimumDistanceSquared) {
          error = "periodic topology is undefined for coincident or near-coincident images";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }

        const double minimum_distance = rounded_square_root(minimum);
        std::size_t multiplicity = 0u;
        for (const auto& translation : translations) {
          if (center == image && is_origin(translation)) continue;
          std::array<double, 3> displacement{};
          double distance_squared = 0.0;
          const ImageGeometryStatus geometry =
              image_geometry(wrapped[center], wrapped[image], translation, cutoff, cutoff_squared,
                             displacement, distance_squared);
          if (geometry == ImageGeometryStatus::kInsideCutoff &&
              within_wigner_seitz_distance_tolerance(distance_squared, minimum_distance)) {
            ++multiplicity;
          }
        }
        if (multiplicity == 0u) {
          error = "periodic topology lost its closest Wigner-Seitz image";
          return XTBLOOM_STATUS_INTERNAL_ERROR;
        }
        if (multiplicity > created.max_size() - created.size()) {
          error = "periodic topology image count exceeds the vector implementation limit";
          return XTBLOOM_STATUS_INVALID_ARGUMENT;
        }
        const double weight = 1.0 / static_cast<double>(multiplicity);
        for (const auto& translation : translations) {
          if (center == image && is_origin(translation)) continue;
          std::array<double, 3> displacement{};
          double distance_squared = 0.0;
          const ImageGeometryStatus geometry =
              image_geometry(wrapped[center], wrapped[image], translation, cutoff, cutoff_squared,
                             displacement, distance_squared);
          if (geometry == ImageGeometryStatus::kOutsideCutoff ||
              !within_wigner_seitz_distance_tolerance(distance_squared, minimum_distance)) {
            continue;
          }
          WignerSeitzImage entry;
          entry.center_atom = static_cast<std::int64_t>(center);
          entry.image_atom = static_cast<std::int64_t>(image);
          entry.translation = translation.index;
          entry.displacement = displacement;
          entry.distance_squared = distance_squared;
          entry.weight = weight;
          created.push_back(entry);
        }
      }
    }

    topology = std::move(created);
    error.clear();
    return XTBLOOM_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate periodic Wigner-Seitz topology";
    return XTBLOOM_STATUS_ALLOCATION_FAILED;
  }
}

}  // namespace xtbloom::detail::gfn2

#endif  // XTBLOOM_MODEL_GFN2_PERIODIC_TOPOLOGY_HPP

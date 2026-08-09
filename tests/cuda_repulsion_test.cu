#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/gfn2_repulsion.cuh"
#include "model/gfn2/repulsion.hpp"
#include "runtime/backend.hpp"

#define CHECK(condition) \
  do {                   \
    if (!(condition)) {  \
      return __LINE__;   \
    }                    \
  } while (false)

#define CUDA_CHECK(expression) CHECK((expression) == cudaSuccess)

namespace {

using xtbloom::detail::cuda::Gfn2RepulsionDeviceBatch;
using xtbloom::detail::cuda::Gfn2RepulsionDeviceError;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  cudaError_t allocate(std::size_t count) {
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  cudaError_t copy_from(const T* source, std::size_t count, cudaStream_t stream = nullptr) {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  }

  cudaError_t copy_to(T* destination, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_) {
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
  }

  T* get() { return data_; }
  const T* get() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

bool near(double actual, double expected, double absolute_tolerance,
          double relative_tolerance = 0.0) {
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return std::abs(actual - expected) <= absolute_tolerance + relative_tolerance * scale;
}

constexpr std::array<std::int32_t, 24> kClusterAtomicNumbers{
    6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 8, 7, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

/* xtb test/unit/test_repulsion.f90 coordinates, in bohr. */
constexpr std::array<double, 72> kClusterPositions{
    2.02799738646442,  0.09231312124713,  -0.14310895950963, 4.75011007621000,  0.02373496014051,
    -0.14324124033844, 6.33434307654413,  2.07098865582721,  -0.14235306905930, 8.72860718071825,
    1.38002919517619,  -0.14265542523943, 8.65318821103610,  -1.19324866489847, -0.14231527453678,
    6.23857175648671,  -2.08353643730276, -0.14218299370797, 5.63266886875962,  -4.69950321056008,
    -0.13940509630299, 3.44931709749015,  -5.48092386085491, -0.14318454855466, 7.77508917214346,
    -6.24427872938674, -0.13107140408805, 10.30229550927022, -5.39739796609292, -0.13672168520430,
    12.07410272485492, -6.91573621641911, -0.13666499342053, 10.70038521493902, -2.79078533715849,
    -0.14148379504141, 13.24597858727017, -1.76969072232377, -0.14218299370797, 7.40891694074004,
    -8.95905928176407, -0.11636933482904, 1.38702118184179,  2.05575746325296,  -0.14178615122154,
    1.34622199478497,  -0.86356704498496, 1.55590600570783,  1.34624089204623,  -0.86133716815647,
    -1.84340893849267, 5.65596919189118,  4.00172183859480,  -0.14131371969009, 14.67430918222276,
    -3.26230980007732, -0.14344911021228, 13.50897177220290, -0.60815166181684, 1.54898960808727,
    13.50780014200488, -0.60614855212345, -1.83214617078268, 5.41408424778406,  -9.49239668625902,
    -0.11022772492007, 8.31919801555568,  -9.74947502841788, 1.56539243085954,  8.31511620712388,
    -9.76854236502758, -1.79108242206824,
};

int test_heterogeneous_ragged_batch_and_stream() {
  constexpr std::array<std::int64_t, 5> offsets{0, 2, 5, 5, 6};
  constexpr std::array<std::int32_t, 6> atomic_numbers{1, 1, 1, 6, 8, 10};
  constexpr std::array<double, 18> positions{
      0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.4, 1.5, 0.2, 3.0, -2.0, 0.5,
  };

  xtbloom::detail::gfn2::RepulsionPlan cpu_plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(4, 6, offsets.data(), atomic_numbers.data(),
                                                   cpu_plan, error) == XTBLOOM_STATUS_SUCCESS);

  std::array<double, 4> expected_energies{0.25, -0.5, 1.5, 2.5};
  std::array<double, 18> expected_forces{};
  for (std::size_t index = 0; index < expected_forces.size(); ++index) {
    expected_forces[index] = 0.01 * static_cast<double>(index) - 0.08;
  }
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(cpu_plan, positions.data(),
                                                 expected_energies.data(), expected_forces.data(),
                                                 error) == XTBLOOM_STATUS_SUCCESS);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceBuffer<std::int64_t> device_offsets;
  DeviceBuffer<std::int32_t> device_atomic_numbers;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> device_energies;
  DeviceBuffer<double> device_forces;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(device_offsets.allocate(offsets.size()));
  CUDA_CHECK(device_atomic_numbers.allocate(atomic_numbers.size()));
  CUDA_CHECK(device_positions.allocate(positions.size()));
  CUDA_CHECK(device_energies.allocate(expected_energies.size()));
  CUDA_CHECK(device_forces.allocate(expected_forces.size()));
  CUDA_CHECK(device_error.allocate(1));

  std::array<double, 4> actual_energies{0.25, -0.5, 1.5, 2.5};
  std::array<double, 18> actual_forces{};
  for (std::size_t index = 0; index < actual_forces.size(); ++index) {
    actual_forces[index] = 0.01 * static_cast<double>(index) - 0.08;
  }
  CUDA_CHECK(device_offsets.copy_from(offsets.data(), offsets.size(), stream));
  CUDA_CHECK(device_atomic_numbers.copy_from(atomic_numbers.data(), atomic_numbers.size(), stream));
  CUDA_CHECK(device_positions.copy_from(positions.data(), positions.size(), stream));
  CUDA_CHECK(device_energies.copy_from(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device_forces.copy_from(actual_forces.data(), actual_forces.size(), stream));

  const Gfn2RepulsionDeviceBatch batch{4, 6, device_offsets.get(), device_atomic_numbers.get(),
                                       device_positions.get()};
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(
      batch, device_energies.get(), device_forces.get(), device_error.get(), stream));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device_energies.copy_to(actual_energies.data(), actual_energies.size(), stream));
  CUDA_CHECK(device_forces.copy_to(actual_forces.data(), actual_forces.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == static_cast<std::uint32_t>(Gfn2RepulsionDeviceError::kSuccess));
  for (std::size_t index = 0; index < actual_energies.size(); ++index) {
    CHECK(near(actual_energies[index], expected_energies[index], 3.0e-15));
  }
  for (std::size_t index = 0; index < actual_forces.size(); ++index) {
    CHECK(near(actual_forces[index], expected_forces[index], 4.0e-15));
  }

  /* Exercise the optional-force path and prove energy accumulation again. */
  std::array<double, 4> energy_only{0.125, 0.25, 0.5, 1.0};
  std::array<double, 4> expected_energy_only = energy_only;
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(cpu_plan, positions.data(),
                                                 expected_energy_only.data(), nullptr,
                                                 error) == XTBLOOM_STATUS_SUCCESS);
  CUDA_CHECK(device_energies.copy_from(energy_only.data(), energy_only.size(), stream));
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(batch, device_energies.get(), nullptr,
                                                            device_error.get(), stream));
  CUDA_CHECK(device_energies.copy_to(energy_only.data(), energy_only.size(), stream));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK(semantic_error == 0u);
  for (std::size_t index = 0; index < energy_only.size(); ++index) {
    CHECK(near(energy_only[index], expected_energy_only[index], 3.0e-15));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int test_xtb_24_atom_golden() {
  constexpr std::array<std::int64_t, 2> offsets{0, 24};
  xtbloom::detail::gfn2::RepulsionPlan cpu_plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(1, 24, offsets.data(),
                                                   kClusterAtomicNumbers.data(), cpu_plan,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 1> expected_energy{};
  std::array<double, 72> expected_forces{};
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(cpu_plan, kClusterPositions.data(),
                                                 expected_energy.data(), expected_forces.data(),
                                                 error) == XTBLOOM_STATUS_SUCCESS);
  CHECK(near(expected_energy[0], 0.49222837261241, 1.0e-13));

  DeviceBuffer<std::int64_t> device_offsets;
  DeviceBuffer<std::int32_t> device_atomic_numbers;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> device_energy;
  DeviceBuffer<double> device_forces;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(device_offsets.allocate(offsets.size()));
  CUDA_CHECK(device_atomic_numbers.allocate(kClusterAtomicNumbers.size()));
  CUDA_CHECK(device_positions.allocate(kClusterPositions.size()));
  CUDA_CHECK(device_energy.allocate(1));
  CUDA_CHECK(device_forces.allocate(kClusterPositions.size()));
  CUDA_CHECK(device_error.allocate(1));
  CUDA_CHECK(device_offsets.copy_from(offsets.data(), offsets.size()));
  CUDA_CHECK(
      device_atomic_numbers.copy_from(kClusterAtomicNumbers.data(), kClusterAtomicNumbers.size()));
  CUDA_CHECK(device_positions.copy_from(kClusterPositions.data(), kClusterPositions.size()));
  CUDA_CHECK(cudaMemset(device_energy.get(), 0, sizeof(double)));
  CUDA_CHECK(cudaMemset(device_forces.get(), 0, kClusterPositions.size() * sizeof(double)));

  const Gfn2RepulsionDeviceBatch batch{1, 24, device_offsets.get(), device_atomic_numbers.get(),
                                       device_positions.get()};
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(
      batch, device_energy.get(), device_forces.get(), device_error.get()));
  std::array<double, 1> actual_energy{};
  std::array<double, 72> actual_forces{};
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device_energy.copy_to(actual_energy.data(), 1));
  CUDA_CHECK(device_forces.copy_to(actual_forces.data(), actual_forces.size()));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  CHECK(near(actual_energy[0], expected_energy[0], 2.0e-15));
  CHECK(near(actual_energy[0], 0.49222837261241, 1.0e-13));
  for (std::size_t index = 0; index < actual_forces.size(); ++index) {
    CHECK(near(actual_forces[index], expected_forces[index], 2.0e-14));
  }
  return 0;
}

int test_large_all_element_ragged_batch() {
  /*
   * Include empty edge systems and a molecule larger than one CUDA block's
   * thread count. Cycling through all supported elements exercises every
   * repulsion parameter entry rather than only the common organic subset.
   */
  constexpr std::array<std::int64_t, 6> offsets{0, 0, 1, 3, 303, 303};
  constexpr std::size_t atom_count = 303;
  std::vector<std::int32_t> atomic_numbers(atom_count);
  std::vector<double> positions(atom_count * 3, 0.0);
  atomic_numbers[0] = 86;
  atomic_numbers[1] = 1;
  atomic_numbers[2] = 2;
  positions[5] = 1.4;
  for (std::size_t local = 0; local < 300; ++local) {
    const std::size_t atom = local + 3;
    atomic_numbers[atom] = static_cast<std::int32_t>(local % 86) + 1;
    positions[atom * 3] = 1.3 * static_cast<double>(local % 10);
    positions[atom * 3 + 1] = 1.4 * static_cast<double>((local / 10) % 10);
    positions[atom * 3 + 2] = 1.5 * static_cast<double>(local / 100);
  }

  xtbloom::detail::gfn2::RepulsionPlan cpu_plan;
  std::string error;
  CHECK(xtbloom::detail::gfn2::make_repulsion_plan(5, static_cast<std::int64_t>(atom_count),
                                                   offsets.data(), atomic_numbers.data(), cpu_plan,
                                                   error) == XTBLOOM_STATUS_SUCCESS);
  std::array<double, 5> expected_energies{0.0, 0.25, -0.5, 0.75, -1.0};
  std::vector<double> expected_forces(atom_count * 3);
  for (std::size_t coordinate = 0; coordinate < expected_forces.size(); ++coordinate) {
    expected_forces[coordinate] = 1.0e-4 * static_cast<double>(coordinate % 17) - 8.0e-4;
  }
  CHECK(xtbloom::detail::gfn2::add_repulsion_cpu(cpu_plan, positions.data(),
                                                 expected_energies.data(), expected_forces.data(),
                                                 error) == XTBLOOM_STATUS_SUCCESS);

  DeviceBuffer<std::int64_t> device_offsets;
  DeviceBuffer<std::int32_t> device_atomic_numbers;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> device_energies;
  DeviceBuffer<double> device_forces;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(device_offsets.allocate(offsets.size()));
  CUDA_CHECK(device_atomic_numbers.allocate(atomic_numbers.size()));
  CUDA_CHECK(device_positions.allocate(positions.size()));
  CUDA_CHECK(device_energies.allocate(expected_energies.size()));
  CUDA_CHECK(device_forces.allocate(expected_forces.size()));
  CUDA_CHECK(device_error.allocate(1));
  CUDA_CHECK(device_offsets.copy_from(offsets.data(), offsets.size()));
  CUDA_CHECK(device_atomic_numbers.copy_from(atomic_numbers.data(), atomic_numbers.size()));
  CUDA_CHECK(device_positions.copy_from(positions.data(), positions.size()));

  std::array<double, 5> actual_energies{0.0, 0.25, -0.5, 0.75, -1.0};
  std::vector<double> actual_forces(atom_count * 3);
  for (std::size_t coordinate = 0; coordinate < actual_forces.size(); ++coordinate) {
    actual_forces[coordinate] = 1.0e-4 * static_cast<double>(coordinate % 17) - 8.0e-4;
  }
  CUDA_CHECK(device_energies.copy_from(actual_energies.data(), actual_energies.size()));
  CUDA_CHECK(device_forces.copy_from(actual_forces.data(), actual_forces.size()));

  const Gfn2RepulsionDeviceBatch batch{5, static_cast<std::int64_t>(atom_count),
                                       device_offsets.get(), device_atomic_numbers.get(),
                                       device_positions.get()};
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(
      batch, device_energies.get(), device_forces.get(), device_error.get()));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device_energies.copy_to(actual_energies.data(), actual_energies.size()));
  CUDA_CHECK(device_forces.copy_to(actual_forces.data(), actual_forces.size()));
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  for (std::size_t system = 0; system < actual_energies.size(); ++system) {
    if (!near(actual_energies[system], expected_energies[system], 2.0e-11, 3.0e-14)) {
      std::cerr << "all-element energy mismatch at system " << system
                << ": actual=" << std::setprecision(17) << actual_energies[system]
                << ", expected=" << expected_energies[system] << '\n';
      return __LINE__;
    }
  }
  for (std::size_t coordinate = 0; coordinate < actual_forces.size(); ++coordinate) {
    if (!near(actual_forces[coordinate], expected_forces[coordinate], 2.0e-10, 2.0e-13)) {
      std::cerr << "all-element force mismatch at coordinate " << coordinate
                << ": actual=" << std::setprecision(17) << actual_forces[coordinate]
                << ", expected=" << expected_forces[coordinate] << '\n';
      return __LINE__;
    }
  }
  return 0;
}

int expect_device_error(const std::array<std::int64_t, 2>& offsets,
                        const std::array<std::int32_t, 2>& atomic_numbers,
                        const std::array<double, 6>& positions, Gfn2RepulsionDeviceError expected) {
  DeviceBuffer<std::int64_t> device_offsets;
  DeviceBuffer<std::int32_t> device_atomic_numbers;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> device_energy;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(device_offsets.allocate(offsets.size()));
  CUDA_CHECK(device_atomic_numbers.allocate(atomic_numbers.size()));
  CUDA_CHECK(device_positions.allocate(positions.size()));
  CUDA_CHECK(device_energy.allocate(1));
  CUDA_CHECK(device_error.allocate(1));
  CUDA_CHECK(device_offsets.copy_from(offsets.data(), offsets.size()));
  CUDA_CHECK(device_atomic_numbers.copy_from(atomic_numbers.data(), atomic_numbers.size()));
  CUDA_CHECK(device_positions.copy_from(positions.data(), positions.size()));
  CUDA_CHECK(cudaMemset(device_energy.get(), 0, sizeof(double)));

  const Gfn2RepulsionDeviceBatch batch{1, 2, device_offsets.get(), device_atomic_numbers.get(),
                                       device_positions.get()};
  CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(batch, device_energy.get(), nullptr,
                                                            device_error.get()));
  std::uint32_t actual = 0u;
  CUDA_CHECK(device_error.copy_to(&actual, 1));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(actual == static_cast<std::uint32_t>(expected));
  return 0;
}

int test_input_and_launch_errors() {
  Gfn2RepulsionDeviceBatch invalid{};
  CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(invalid, nullptr, nullptr, nullptr) ==
        cudaErrorInvalidValue);

  /* Non-null sentinels are safe because validation returns before enqueue. */
  invalid.batch_size = 1;
  invalid.total_atoms = 1;
  invalid.atom_offsets = reinterpret_cast<const std::int64_t*>(1);
  invalid.atomic_numbers = reinterpret_cast<const std::int32_t*>(1);
  invalid.positions = reinterpret_cast<const double*>(1);
  CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(invalid, nullptr, nullptr,
                                                       reinterpret_cast<std::uint32_t*>(1)) ==
        cudaErrorInvalidValue);
  invalid.total_atoms = std::numeric_limits<std::int64_t>::max();
  CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(
            invalid, reinterpret_cast<double*>(1), nullptr, reinterpret_cast<std::uint32_t*>(1)) ==
        cudaErrorInvalidValue);

  constexpr std::array<std::int64_t, 2> offsets{0, 2};
  constexpr std::array<std::int32_t, 2> atoms{1, 1};
  constexpr std::array<double, 6> positions{0.0, 0.0, 0.0, 1.4, 0.0, 0.0};
  constexpr std::array<std::int64_t, 2> bad_offsets{0, 3};
  CHECK(expect_device_error(bad_offsets, atoms, positions,
                            Gfn2RepulsionDeviceError::kInvalidOffsets) == 0);
  constexpr std::array<std::int32_t, 2> bad_atoms{1, 87};
  CHECK(expect_device_error(offsets, bad_atoms, positions,
                            Gfn2RepulsionDeviceError::kInvalidAtomicNumberOrParameter) == 0);
  std::array<double, 6> nonfinite_positions = positions;
  nonfinite_positions[3] = std::numeric_limits<double>::quiet_NaN();
  CHECK(expect_device_error(offsets, atoms, nonfinite_positions,
                            Gfn2RepulsionDeviceError::kNonfinitePosition) == 0);
  constexpr std::array<double, 6> coincident_positions{};
  CHECK(expect_device_error(offsets, atoms, coincident_positions,
                            Gfn2RepulsionDeviceError::kCoincidentAtoms) == 0);

  /* Reject an unrepresentable launch grid before enqueueing any device work. */
  DeviceBuffer<std::int64_t> device_offsets;
  DeviceBuffer<std::int32_t> device_atom;
  DeviceBuffer<double> device_position;
  DeviceBuffer<double> device_energy;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(device_offsets.allocate(2));
  CUDA_CHECK(device_atom.allocate(1));
  CUDA_CHECK(device_position.allocate(3));
  CUDA_CHECK(device_energy.allocate(1));
  CUDA_CHECK(device_error.allocate(1));
  invalid.batch_size = static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
  invalid.total_atoms = 1;
  invalid.atom_offsets = device_offsets.get();
  invalid.atomic_numbers = device_atom.get();
  invalid.positions = device_position.get();
  CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(invalid, device_energy.get(), nullptr,
                                                       device_error.get()) ==
        cudaErrorInvalidConfiguration);
  return 0;
}

int record_initial_throughput() {
  constexpr std::int64_t batch_size = 512;
  constexpr int warmups = 3;
  int iterations = 20;
  if (const char* configured = std::getenv("XTBLOOM_BENCHMARK_ITERATIONS")) {
    iterations = std::max(1, std::atoi(configured));
  }
  const std::size_t atom_count =
      static_cast<std::size_t>(batch_size) * kClusterAtomicNumbers.size();
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(batch_size) + 1);
  std::vector<std::int32_t> atomic_numbers(atom_count);
  std::vector<double> positions(atom_count * 3);
  for (std::int64_t system = 0; system < batch_size; ++system) {
    offsets[static_cast<std::size_t>(system)] =
        system * static_cast<std::int64_t>(kClusterAtomicNumbers.size());
    std::copy(kClusterAtomicNumbers.begin(), kClusterAtomicNumbers.end(),
              atomic_numbers.begin() + offsets[static_cast<std::size_t>(system)]);
    std::copy(kClusterPositions.begin(), kClusterPositions.end(),
              positions.begin() + offsets[static_cast<std::size_t>(system)] * 3);
  }
  offsets.back() = static_cast<std::int64_t>(atom_count);

  DeviceBuffer<std::int64_t> device_offsets;
  DeviceBuffer<std::int32_t> device_atomic_numbers;
  DeviceBuffer<double> device_positions;
  DeviceBuffer<double> device_energies;
  DeviceBuffer<double> device_forces;
  DeviceBuffer<std::uint32_t> device_error;
  CUDA_CHECK(device_offsets.allocate(offsets.size()));
  CUDA_CHECK(device_atomic_numbers.allocate(atomic_numbers.size()));
  CUDA_CHECK(device_positions.allocate(positions.size()));
  CUDA_CHECK(device_energies.allocate(static_cast<std::size_t>(batch_size)));
  CUDA_CHECK(device_forces.allocate(positions.size()));
  CUDA_CHECK(device_error.allocate(1));
  CUDA_CHECK(device_offsets.copy_from(offsets.data(), offsets.size()));
  CUDA_CHECK(device_atomic_numbers.copy_from(atomic_numbers.data(), atomic_numbers.size()));
  CUDA_CHECK(device_positions.copy_from(positions.data(), positions.size()));
  CUDA_CHECK(
      cudaMemset(device_energies.get(), 0, static_cast<std::size_t>(batch_size) * sizeof(double)));
  CUDA_CHECK(cudaMemset(device_forces.get(), 0, positions.size() * sizeof(double)));

  const Gfn2RepulsionDeviceBatch batch{batch_size, static_cast<std::int64_t>(atom_count),
                                       device_offsets.get(), device_atomic_numbers.get(),
                                       device_positions.get()};
  for (int iteration = 0; iteration < warmups; ++iteration) {
    CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(
        batch, device_energies.get(), device_forces.get(), device_error.get()));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    CUDA_CHECK(xtbloom::detail::cuda::add_gfn2_repulsion_cuda(
        batch, device_energies.get(), device_forces.get(), device_error.get()));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::uint32_t semantic_error = 99u;
  CUDA_CHECK(device_error.copy_to(&semantic_error, 1));
  CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(semantic_error == 0u);
  const double systems_per_second =
      static_cast<double>(batch_size) * iterations * 1000.0 / static_cast<double>(milliseconds);
  std::cout << "CUDA GFN2 repulsion baseline: " << std::fixed << std::setprecision(3)
            << milliseconds / iterations << " ms/batch, " << std::setprecision(0)
            << systems_per_second << " systems/s (batch=" << batch_size << ", atoms/system=24)\n";
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    /* CUDA builds remain testable on runners without a visible GPU. */
    std::cout << "CUDA GFN2 repulsion test skipped: "
              << (count_status == cudaSuccess ? "no visible device"
                                              : cudaGetErrorString(count_status))
              << '\n';
    return 0;
  }
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  std::string error;
  CHECK(xtbloom::detail::ensure_cuda_gfn2_parameters(device, error));

  if (const int status = test_heterogeneous_ragged_batch_and_stream(); status != 0) {
    return status;
  }
  if (const int status = test_xtb_24_atom_golden(); status != 0) {
    return status;
  }
  if (const int status = test_large_all_element_ragged_batch(); status != 0) {
    return status;
  }
  if (const int status = test_input_and_launch_errors(); status != 0) {
    return status;
  }
  return record_initial_throughput();
}

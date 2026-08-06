// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
//
// Focused #41 evidence: the per-iteration SCC mixer history transaction must
// scale with the number of active systems, not the total batch history.
//
// The benchmark builds one fixed ragged batch of N identical carbon systems and
// times, for K active systems in {1, N/4, N/2, N}:
//
//   copy:   prepare+commit of the first K systems (the per-system history they
//           copy in and out of the staged binding);
//   mix:    prepare, one Broyden transition, and commit for the first K systems
//           (the full transaction a driver iteration performs per active peer);
//   old:    the pre-change full mixer-state copy used when the whole batch
//           history was duplicated every iteration (two state_size memcpys).
//
// A large ("cold") read/write scrubber is touched between measured rounds so
// both the active-system transactions and the whole-state baseline are timed in
// the same cache regime rather than as a tiny hot loop.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpuxtb/gpuxtb.h"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"

namespace {

using gpuxtb::detail::gfn2::BasisPlan;
using gpuxtb::detail::gfn2::SccMixerPlan;
using gpuxtb::detail::gfn2::SccMixerState;
using gpuxtb::detail::gfn2::SccMixerWorkspace;
using gpuxtb::detail::gfn2::WavefunctionLayout;
using gpuxtb::detail::gfn2::WavefunctionView;

using Clock = std::chrono::steady_clock;

struct AlignedBuffer {
  void* data = nullptr;
  std::size_t size = 0u;

  explicit AlignedBuffer(std::size_t requested, double fill = 0.0) {
    size = (std::max<std::size_t>(requested, 1u) + 63u) & ~std::size_t{63u};
    data = std::aligned_alloc(64u, size);
    if (data != nullptr) {
      std::memset(data, fill == 0.0 ? 0 : static_cast<unsigned char>(0xab), size);
    }
  }
  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

std::size_t field_begin(const gpuxtb::detail::gfn2::WavefunctionFieldLayout& field,
                        std::size_t system) {
  return static_cast<std::size_t>(field.system_offsets[system]);
}

std::size_t field_end(const gpuxtb::detail::gfn2::WavefunctionFieldLayout& field,
                      std::size_t system) {
  return static_cast<std::size_t>(field.system_offsets[system + 1u]);
}

void set_system_vector(const WavefunctionLayout& layout, WavefunctionView& wavefunction,
                       std::size_t system, const double* values) {
  std::size_t packed = 0u;
  const std::array<double*, 3> fields{
      {wavefunction.qsh, wavefunction.dipole, wavefunction.quadrupole}};
  const std::array<const gpuxtb::detail::gfn2::WavefunctionFieldLayout*, 3> layouts{
      {&layout.qsh, &layout.dipole, &layout.quadrupole}};
  for (std::size_t field = 0u; field < fields.size(); ++field) {
    for (std::size_t destination = field_begin(*layouts[field], system);
         destination < field_end(*layouts[field], system); ++destination, ++packed) {
      fields[field][destination] = values[packed];
    }
  }
}

std::vector<double> median_samples(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2u];
  return {median, samples.front(), samples.back()};
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t batch = argc > 1 ? static_cast<std::size_t>(std::atoll(argv[1])) : 64u;
  const std::size_t atoms_per_system = argc > 2 ? static_cast<std::size_t>(std::atoll(argv[2])) : 6u;
  const std::int64_t history_size = argc > 3 ? std::atoll(argv[3]) : 64;
  const std::size_t repetitions = argc > 4 ? static_cast<std::size_t>(std::atoll(argv[4])) : 200u;
  const char* json_path = argc > 5 ? argv[5] : nullptr;
  const std::size_t scrub_bytes = batch * std::max<std::size_t>(atoms_per_system, 1u) * 4096u;

  std::string error;
  const std::int64_t total_atoms = static_cast<std::int64_t>(batch * atoms_per_system);
  std::vector<std::int64_t> atom_offsets(batch + 1u);
  std::vector<std::int32_t> atomic_numbers(static_cast<std::size_t>(total_atoms), 6);
  for (std::size_t system = 0u; system <= batch; ++system) {
    atom_offsets[system] = static_cast<std::int64_t>(system * atoms_per_system);
  }
  std::vector<double> charges(batch, 0.0);
  std::vector<std::int32_t> unpaired(batch, 0);
  std::vector<std::int32_t> spin_channels(batch, 1);

  BasisPlan basis;
  if (gpuxtb::detail::gfn2::make_basis_plan(
          static_cast<std::int64_t>(batch), total_atoms, atom_offsets.data(),
          atomic_numbers.data(), basis, error) != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "make_basis_plan: %s\n", error.c_str());
    return 1;
  }
  WavefunctionLayout layout;
  if (gpuxtb::detail::gfn2::make_wavefunction_layout(
          basis, atomic_numbers.data(), charges.data(), unpaired.data(), spin_channels.data(),
          layout, error) != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "make_wavefunction_layout: %s\n", error.c_str());
    return 1;
  }
  SccMixerPlan plan;
  if (gpuxtb::detail::gfn2::make_scc_mixer_plan(layout, history_size, 0.4, 1.0e-8, 1.0e-8, plan,
                                                error) != GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "make_scc_mixer_plan: %s\n", error.c_str());
    return 1;
  }

  AlignedBuffer wavefunction_storage(layout.workspace_size_bytes, 1.0);
  AlignedBuffer state_storage(plan.state_size_bytes(), 1.0);
  AlignedBuffer staged_storage(plan.state_size_bytes(), 1.0);
  AlignedBuffer scratch_storage(plan.workspace_size_bytes(), 1.0);
  AlignedBuffer scrubber(scrub_bytes, 1.0);
  if (wavefunction_storage.data == nullptr || state_storage.data == nullptr ||
      staged_storage.data == nullptr || scratch_storage.data == nullptr ||
      scrubber.data == nullptr) {
    std::fprintf(stderr, "benchmark allocation failed\n");
    return 1;
  }
  WavefunctionView wavefunction;
  SccMixerState state;
  SccMixerState staged;
  SccMixerWorkspace scratch;
  if (gpuxtb::detail::gfn2::bind_wavefunction_view(
          layout, wavefunction_storage.data, wavefunction_storage.size, wavefunction,
          error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::initialize_sad_multipole_state(layout, wavefunction, error) !=
          GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::bind_scc_mixer_state(plan, state_storage.data, state_storage.size,
                                                 state, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::bind_scc_mixer_state(plan, staged_storage.data, staged_storage.size,
                                                 staged, error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::bind_scc_mixer_workspace(plan, scratch_storage.data,
                                                     scratch_storage.size, scratch,
                                                     error) != GPUXTB_STATUS_SUCCESS ||
      gpuxtb::detail::gfn2::initialize_scc_mixer_state_cpu(plan, wavefunction, state, error) !=
          GPUXTB_STATUS_SUCCESS) {
    std::fprintf(stderr, "bind/initialize: %s\n", error.c_str());
    return 1;
  }

  const std::vector<std::int64_t>& offsets = plan.vector_offsets();
  std::vector<std::size_t> dimensions(batch);
  std::vector<std::size_t> vector_offsets(batch);
  std::vector<std::size_t> history_offsets(batch);
  std::size_t history_cursor = 0u;
  for (std::size_t system = 0u; system < batch; ++system) {
    const std::size_t dimension =
        static_cast<std::size_t>(offsets[system + 1u] - offsets[system]);
    dimensions[system] = dimension;
    vector_offsets[system] = static_cast<std::size_t>(offsets[system]);
    history_offsets[system] = history_cursor;
    history_cursor += dimension * static_cast<std::size_t>(history_size);
  }
  const std::size_t memory = static_cast<std::size_t>(history_size);

  /* Warm the Broyden path for every system so the measured $mix$ rows exercise
   * history reads and writes instead of only the damped first step. */
  for (int iteration = 0; iteration < 3; ++iteration) {
    for (std::size_t system = 0u; system < batch; ++system) {
      const std::size_t dimension = dimensions[system];
      const std::size_t vector_offset = vector_offsets[system];
      std::vector<double> raw(dimension);
      for (std::size_t component = 0u; component < dimension; ++component) {
        raw[component] = state.current_inputs[vector_offset + component] +
                         0.001 * static_cast<double>(component + 1u) +
                         0.0002 * static_cast<double>(system + 1u) * static_cast<double>(iteration);
      }
      set_system_vector(layout, wavefunction, system, raw.data());
      if (gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
              plan, static_cast<std::int64_t>(system), wavefunction, state, scratch,
              error) != GPUXTB_STATUS_SUCCESS) {
        std::fprintf(stderr, "warmup mix: %s\n", error.c_str());
        return 1;
      }
    }
  }

  const auto scrub = [&]() {
    volatile double sink = 0.0;
    volatile std::uint64_t* words = static_cast<volatile std::uint64_t*>(scrubber.data);
    for (std::size_t word = 0u; word < scrubber.size / sizeof(std::uint64_t); ++word) {
      sink += words[word];
    }
    (void)sink;
  };

  const auto active_counts = [&]() {
    std::vector<std::size_t> counts{batch / 8u, batch / 4u, batch / 2u, batch};
    if (counts.front() == 0u) {
      counts[0u] = 1u;
    }
    return counts;
  }();

  const auto timestamp = []() { return Clock::now(); };
  const auto elapsed_us = [](Clock::time_point start) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()) /
           1.0e3;
  };

  const std::size_t scalar_bytes =
      2u * sizeof(double) + 2u * sizeof(std::uint64_t) + sizeof(gpuxtb_status_t) +
      2u * sizeof(std::uint8_t);
  const std::size_t per_system_bytes =
      (3u * dimensions.front() + 2u * dimensions.front() * memory + memory) * sizeof(double) +
      scalar_bytes;

  struct Row {
    std::size_t active = 0u;
    std::vector<double> old_samples;
    std::vector<double> copy_samples;
    std::vector<double> mix_samples;
    std::size_t old_bytes = 0u;
    std::size_t new_bytes = 0u;
  };
  std::vector<Row> rows;
  rows.reserve(active_counts.size());

  std::printf("mixer_transaction_benchmark\n");
  std::printf("batch=%zu atoms_per_system=%zu history=%lld state_size_bytes=%zu "
              "vector_bytes=%zu history_bytes=%zu per_system_bytes=%zu\n",
              batch, atoms_per_system, static_cast<long long>(history_size),
              static_cast<size_t>(plan.state_size_bytes()),
              static_cast<size_t>(plan.total_vector_elements()) * sizeof(double),
              history_cursor * sizeof(double), per_system_bytes);
  std::printf("row,active_systems,old_copy_us,copy_transaction_us,mix_transaction_us,"
              "old_copy_bytes,new_transaction_bytes\n");

  for (const std::size_t active : active_counts) {
    Row row;
    row.active = active;
    row.old_bytes = 2u * static_cast<size_t>(plan.state_size_bytes());
    row.new_bytes = active * per_system_bytes;
    row.old_samples.reserve(repetitions);
    row.copy_samples.reserve(repetitions);
    row.mix_samples.reserve(repetitions);

    /* Old design baseline: the driver duplicated the complete mixer state on
     * every iteration regardless of how many systems were active. */
    for (std::size_t round = 0u; round < repetitions; ++round) {
      scrub();
      const auto start = timestamp();
      std::memcpy(staged.workspace_base, state.workspace_base, plan.state_size_bytes());
      std::memcpy(state.workspace_base, staged.workspace_base, plan.state_size_bytes());
      row.old_samples.push_back(elapsed_us(start));
    }

    /* New design: each active system stages and commits only its own history. */
    for (std::size_t round = 0u; round < repetitions; ++round) {
      scrub();
      auto start = timestamp();
      for (std::size_t system = 0u; system < active; ++system) {
        (void)gpuxtb::detail::gfn2::prepare_scc_mixer_system_transaction_cpu(
            plan, static_cast<std::int64_t>(system), state, staged, error);
        (void)gpuxtb::detail::gfn2::commit_scc_mixer_system_transaction_cpu(
            plan, static_cast<std::int64_t>(system), staged, state, error);
      }
      row.copy_samples.push_back(elapsed_us(start));

      /* A realistic mix transaction additionally runs one Broyden transition
       * against a freshly prepared raw output and commits it. */
      scrub();
      start = timestamp();
      for (std::size_t system = 0u; system < active; ++system) {
        (void)gpuxtb::detail::gfn2::prepare_scc_mixer_system_transaction_cpu(
            plan, static_cast<std::int64_t>(system), state, staged, error);
        const std::size_t dimension = dimensions[system];
        const std::size_t vector_offset = vector_offsets[system];
        std::vector<double> raw(dimension);
        for (std::size_t component = 0u; component < dimension; ++component) {
          raw[component] =
              state.current_inputs[vector_offset + component] +
              0.0005 * static_cast<double>(component + 1u) * static_cast<double>(round + 1u);
        }
        set_system_vector(layout, wavefunction, system, raw.data());
        (void)gpuxtb::detail::gfn2::mix_scc_broyden_system_cpu(
            plan, static_cast<std::int64_t>(system), wavefunction, staged, scratch, error);
        (void)gpuxtb::detail::gfn2::commit_scc_mixer_system_transaction_cpu(
            plan, static_cast<std::int64_t>(system), staged, state, error);
      }
      row.mix_samples.push_back(elapsed_us(start));
    }

    const std::vector<double> old_stats = median_samples(row.old_samples);
    const std::vector<double> copy_stats = median_samples(row.copy_samples);
    const std::vector<double> mix_stats = median_samples(row.mix_samples);
    std::printf("%zu,%zu,%.3f,%.3f,%.3f,%zu,%zu\n", row.active, row.active, old_stats[0],
                copy_stats[0], mix_stats[0], row.old_bytes, row.new_bytes);
    rows.push_back(std::move(row));
  }

  if (json_path != nullptr) {
    std::ofstream json(json_path);
    json << "{\n"
         << "  \"benchmark\": \"scc_mixer_transaction\",\n"
         << "  \"batch\": " << batch << ",\n"
         << "  \"atoms_per_system\": " << atoms_per_system << ",\n"
         << "  \"history_size\": " << history_size << ",\n"
         << "  \"state_size_bytes\": " << plan.state_size_bytes() << ",\n"
         << "  \"per_system_bytes\": " << per_system_bytes << ",\n"
         << "  \"repetitions\": " << repetitions << ",\n"
         << "  \"rows\": [\n";
    for (std::size_t index = 0u; index < rows.size(); ++index) {
      const Row& row = rows[index];
      json << "    {\n"
           << "      \"active_systems\": " << row.active << ",\n"
           << "      \"old_copy_bytes\": " << row.old_bytes << ",\n"
           << "      \"new_transaction_bytes\": " << row.new_bytes << ",\n"
           << "      \"old_copy_samples_us\": [";
      for (std::size_t sample = 0u; sample < row.old_samples.size(); ++sample) {
        json << (sample == 0u ? "" : ", ") << row.old_samples[sample];
      }
      json << "],\n"
           << "      \"copy_transaction_samples_us\": [";
      for (std::size_t sample = 0u; sample < row.copy_samples.size(); ++sample) {
        json << (sample == 0u ? "" : ", ") << row.copy_samples[sample];
      }
      json << "],\n"
           << "      \"mix_transaction_samples_us\": [";
      for (std::size_t sample = 0u; sample < row.mix_samples.size(); ++sample) {
        json << (sample == 0u ? "" : ", ") << row.mix_samples[sample];
      }
      json << "]\n"
           << "    }" << (index + 1u == rows.size() ? "" : ",") << "\n";
    }
    json << "  ]\n"
         << "}\n";
  }
  return 0;
}
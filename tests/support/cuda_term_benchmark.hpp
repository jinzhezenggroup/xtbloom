#ifndef XTBLOOM_TESTS_SUPPORT_CUDA_TERM_BENCHMARK_HPP
#define XTBLOOM_TESTS_SUPPORT_CUDA_TERM_BENCHMARK_HPP

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

/* Some minimal CUDA toolkit packages omit cuda_profiler_api.h while retaining
 * these stable cudart entry points.  The benchmark exposes them only for an
 * explicitly selected measured term, so nsys/ncu capture ranges do not include
 * fixture setup, warmup, downloads, or correctness checks. */
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);

namespace xtbloom::test::cuda_term_benchmark {

inline constexpr int kMinimumWarmups = 3;
inline constexpr int kMinimumSamples = 20;

enum class Topology {
  kCompact,
  kOpen,
};

inline const char* topology_name(Topology topology) {
  return topology == Topology::kCompact ? "compact" : "open";
}

struct Options {
  int warmups = kMinimumWarmups;
  int samples = kMinimumSamples;
  std::int64_t batch_size = 1;
  std::int64_t atoms_per_system = 32;
  Topology topology = Topology::kCompact;
  std::string json_path;
  std::string csv_path;
  std::string source_revision;
  std::string executable_sha256;
  std::string build_identity_sha256;
  std::string profile_term;
};

struct Samples {
  std::vector<float> values_ms;
  float minimum_ms = 0.0F;
  float median_ms = 0.0F;
  float maximum_ms = 0.0F;
};

struct Row {
  std::string term;
  std::string workload;
  std::int64_t batch_size = 0;
  std::int64_t atoms_per_system = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  Samples timing;
};

inline bool parse_positive_integer(const char* text, std::int64_t* value) {
  if (text == nullptr || value == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const long long parsed = std::strtoll(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed <= 0) return false;
  *value = static_cast<std::int64_t>(parsed);
  return true;
}

inline bool parse_options(int argc, char** argv, Options* options, std::string* error) {
  if (options == nullptr || error == nullptr) return false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      *error = "help";
      return false;
    }
    if (index + 1 >= argc) {
      *error = argument + " requires a value";
      return false;
    }
    const std::string value = argv[++index];
    std::int64_t parsed = 0;
    if (argument == "--json") {
      options->json_path = value;
    } else if (argument == "--csv") {
      options->csv_path = value;
    } else if (argument == "--source-revision") {
      options->source_revision = value;
    } else if (argument == "--executable-sha256") {
      options->executable_sha256 = value;
    } else if (argument == "--build-identity-sha256") {
      options->build_identity_sha256 = value;
    } else if (argument == "--profile-term") {
      options->profile_term = value;
    } else if (argument == "--topology") {
      if (value == "compact") {
        options->topology = Topology::kCompact;
      } else if (value == "open") {
        options->topology = Topology::kOpen;
      } else {
        *error = "--topology must be compact or open";
        return false;
      }
    } else if (argument == "--warmups" && parse_positive_integer(value.c_str(), &parsed) &&
               parsed <= INT_MAX) {
      options->warmups = static_cast<int>(parsed);
    } else if (argument == "--samples" && parse_positive_integer(value.c_str(), &parsed) &&
               parsed <= INT_MAX) {
      options->samples = static_cast<int>(parsed);
    } else if (argument == "--batch" && parse_positive_integer(value.c_str(), &parsed)) {
      options->batch_size = parsed;
    } else if (argument == "--atoms-per-system" && parse_positive_integer(value.c_str(), &parsed)) {
      options->atoms_per_system = parsed;
    } else {
      *error = "unknown or invalid benchmark option: " + argument;
      return false;
    }
  }
  if (options->warmups < kMinimumWarmups || options->samples < kMinimumSamples) {
    *error = "term evidence requires at least 3 warmups and 20 measured samples";
    return false;
  }
  if (!options->profile_term.empty() &&
      options->profile_term.find_first_of("\n\r\t") != std::string::npos) {
    *error = "--profile-term must be one printable term identifier";
    return false;
  }
  if ((!options->json_path.empty() || !options->csv_path.empty()) &&
      (options->source_revision.empty() || options->executable_sha256.empty() ||
       options->build_identity_sha256.empty())) {
    *error =
        "file evidence requires --source-revision, --executable-sha256, and "
        "--build-identity-sha256";
    return false;
  }
  return true;
}

inline void print_usage(const char* executable) {
  std::cerr << "usage: " << executable
            << " [--batch N] [--atoms-per-system N] [--topology compact|open]\n"
               "       [--warmups N>=3] [--samples N>=20]\n"
               "       [--json PATH] [--csv PATH] --source-revision SHA\n"
               "       --executable-sha256 SHA256 --build-identity-sha256 SHA256\n"
               "       [--profile-term TERM]\n";
}

inline std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2u);
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

inline Samples summarize(std::vector<float> values) {
  Samples result;
  result.values_ms = std::move(values);
  std::vector<float> ordered = result.values_ms;
  std::sort(ordered.begin(), ordered.end());
  if (!ordered.empty()) {
    result.minimum_ms = ordered.front();
    result.median_ms = ordered[ordered.size() / 2u];
    result.maximum_ms = ordered.back();
  }
  return result;
}

/* prepare runs before the start event on the same stream, and validate runs
 * only after all samples.  Callers use those hooks for error reset, output
 * seeds, and D2H correctness gates without contaminating the measured term. */
template <typename Prepare, typename Enqueue, typename Validate>
bool measure_term(const Options& options, const std::string& term, cudaStream_t stream,
                  Prepare&& prepare, Enqueue&& enqueue, Validate&& validate, Samples* timing,
                  std::string* error) {
  if (timing == nullptr || error == nullptr) return false;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
    if (start != nullptr) (void)cudaEventDestroy(start);
    if (stop != nullptr) (void)cudaEventDestroy(stop);
    *error = "failed to create CUDA timing events for " + term;
    return false;
  }
  const auto run_once = [&](bool retain, std::vector<float>* values) {
    if (!prepare()) return false;
    if (cudaEventRecord(start, stream) != cudaSuccess || !enqueue() ||
        cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess) {
      return false;
    }
    float elapsed_ms = 0.0F;
    if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) return false;
    if (retain) values->push_back(elapsed_ms);
    return true;
  };

  std::vector<float> values;
  values.reserve(static_cast<std::size_t>(options.samples));
  for (int warmup = 0; warmup < options.warmups; ++warmup) {
    if (!run_once(false, &values)) {
      *error = "warmup failed for " + term;
      (void)cudaEventDestroy(stop);
      (void)cudaEventDestroy(start);
      return false;
    }
  }
  const bool profile = options.profile_term == term;
  bool profile_active = false;
  for (int sample = 0; sample < options.samples; ++sample) {
    if (!prepare()) {
      *error = "measured preparation failed for " + term;
      (void)cudaEventDestroy(stop);
      (void)cudaEventDestroy(start);
      return false;
    }
    /* Capture exactly one prepared launch.  Keeping the profiler enabled for
     * all samples would also capture later error resets and output seeds. */
    if (profile && sample == 0 &&
        (cudaStreamSynchronize(stream) != cudaSuccess || cudaProfilerStart() != cudaSuccess)) {
      *error = "cudaProfilerStart failed for " + term;
      (void)cudaEventDestroy(stop);
      (void)cudaEventDestroy(start);
      return false;
    }
    profile_active = profile && sample == 0;
    if (cudaEventRecord(start, stream) != cudaSuccess || !enqueue() ||
        cudaEventRecord(stop, stream) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess) {
      if (profile_active) (void)cudaProfilerStop();
      *error = "measured sample failed for " + term;
      (void)cudaEventDestroy(stop);
      (void)cudaEventDestroy(start);
      return false;
    }
    if (profile_active && cudaProfilerStop() != cudaSuccess) {
      *error = "cudaProfilerStop failed for " + term;
      (void)cudaEventDestroy(stop);
      (void)cudaEventDestroy(start);
      return false;
    }
    profile_active = false;
    float elapsed_ms = 0.0F;
    if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
      *error = "CUDA event query failed for " + term;
      (void)cudaEventDestroy(stop);
      (void)cudaEventDestroy(start);
      return false;
    }
    values.push_back(elapsed_ms);
  }
  if (!validate()) {
    *error = "post-timing correctness/error validation failed for " + term;
    (void)cudaEventDestroy(stop);
    (void)cudaEventDestroy(start);
    return false;
  }
  (void)cudaEventDestroy(stop);
  (void)cudaEventDestroy(start);
  *timing = summarize(std::move(values));
  return true;
}

inline bool write_results(const char* benchmark, const Options& options, int argc, char** argv,
                          const std::vector<Row>& rows, std::string* error) {
  int device_id = 0;
  int runtime_version = 0;
  int driver_version = 0;
  cudaDeviceProp properties{};
  if (!options.profile_term.empty() && std::none_of(rows.begin(), rows.end(), [&](const Row& row) {
        return row.term == options.profile_term;
      })) {
    *error = "unknown --profile-term: " + options.profile_term;
    return false;
  }
  if (cudaGetDevice(&device_id) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device_id) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
      cudaDriverGetVersion(&driver_version) != cudaSuccess) {
    *error = "failed to query CUDA provenance";
    return false;
  }
  const auto write_json = [&](std::ostream& output) {
    output << std::setprecision(9);
    output << "{\n  \"schema_version\": 1,\n"
           << "  \"benchmark\": \"" << json_escape(benchmark) << "\",\n"
           << "  \"timing_scope\": \"cudaEvent elapsed time around one production term; "
              "prepare/download/validation excluded\",\n"
           << "  \"warmups\": " << options.warmups << ",\n"
           << "  \"samples_per_term\": " << options.samples << ",\n"
           << "  \"topology\": \"" << topology_name(options.topology) << "\",\n"
           << "  \"profile_term\": \"" << json_escape(options.profile_term) << "\",\n"
           << "  \"profile_range_scope\": \"one prepared production-term launch; setup, "
              "warmups, resets, seeds, downloads, and validation excluded\",\n"
           << "  \"source_revision\": \"" << json_escape(options.source_revision) << "\",\n"
           << "  \"executable_sha256\": \"" << json_escape(options.executable_sha256) << "\",\n"
           << "  \"build_identity_sha256\": \"" << json_escape(options.build_identity_sha256)
           << "\",\n"
           << "  \"cuda_header_version\": " << CUDART_VERSION << ",\n"
           << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
           << "  \"cuda_driver_version\": " << driver_version << ",\n"
           << "  \"device_id\": " << device_id << ",\n"
           << "  \"device_name\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << '.' << properties.minor
           << "\",\n"
           << "  \"argv\": [";
    for (int index = 0; index < argc; ++index) {
      if (index != 0) output << ", ";
      output << '"' << json_escape(argv[index]) << '"';
    }
    output << "],\n  \"rows\": [\n";
    for (std::size_t index = 0; index < rows.size(); ++index) {
      const Row& row = rows[index];
      if (index != 0u) output << ",\n";
      output << "    {\"term\": \"" << json_escape(row.term) << "\", \"workload\": \""
             << json_escape(row.workload) << "\", \"batch\": " << row.batch_size
             << ", \"atoms_per_system\": " << row.atoms_per_system
             << ", \"total_atoms\": " << row.total_atoms << ", \"total_pairs\": " << row.total_pairs
             << ", \"minimum_ms\": " << row.timing.minimum_ms
             << ", \"median_ms\": " << row.timing.median_ms
             << ", \"maximum_ms\": " << row.timing.maximum_ms << ", \"samples_ms\": [";
      for (std::size_t sample = 0; sample < row.timing.values_ms.size(); ++sample) {
        if (sample != 0u) output << ", ";
        output << row.timing.values_ms[sample];
      }
      output << "]}";
    }
    output << "\n  ]\n}\n";
  };
  const auto write_csv = [&](std::ostream& output) {
    output << "term,workload,batch,atoms_per_system,total_atoms,total_pairs,minimum_ms,median_ms,"
              "maximum_ms,samples\n";
    output << std::setprecision(9);
    for (const Row& row : rows) {
      output << row.term << ',' << row.workload << ',' << row.batch_size << ','
             << row.atoms_per_system << ',' << row.total_atoms << ',' << row.total_pairs << ','
             << row.timing.minimum_ms << ',' << row.timing.median_ms << ',' << row.timing.maximum_ms
             << ',' << row.timing.values_ms.size() << '\n';
    }
  };

  if (options.json_path.empty() && options.csv_path.empty()) {
    write_json(std::cout);
    return true;
  }
  if (!options.json_path.empty()) {
    std::ofstream output(options.json_path);
    if (!output) {
      *error = "failed to open JSON output: " + options.json_path;
      return false;
    }
    write_json(output);
  }
  if (!options.csv_path.empty()) {
    std::ofstream output(options.csv_path);
    if (!output) {
      *error = "failed to open CSV output: " + options.csv_path;
      return false;
    }
    write_csv(output);
  }
  return true;
}

}  // namespace xtbloom::test::cuda_term_benchmark

#endif  // XTBLOOM_TESTS_SUPPORT_CUDA_TERM_BENCHMARK_HPP

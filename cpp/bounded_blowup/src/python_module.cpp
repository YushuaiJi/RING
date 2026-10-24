// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

#include "bounded_blowup/index_view.hpp"
#include "bounded_blowup/hnsw_baseline.hpp"
#include "bounded_blowup/native_bulk_build.hpp"
#include "bounded_blowup/result.hpp"
#include "bounded_blowup/router.hpp"
#include "bounded_blowup/routing_layout.hpp"
#include "bounded_blowup/search.hpp"
#include "bounded_blowup/verified_corpus.hpp"

namespace py = pybind11;

#ifndef RFBB_CPP_SOURCE_TREE_SHA256
#define RFBB_CPP_SOURCE_TREE_SHA256 "UNSET"
#endif

namespace bounded_blowup {
namespace {

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      throw std::runtime_error("cannot open mmap asset " + path.string() +
                               ": " + std::strerror(errno));
    }
    struct stat status {};
    if (::fstat(fd_, &status) != 0) {
      const auto message = std::string(std::strerror(errno));
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("cannot stat mmap asset " + path.string() +
                               ": " + message);
    }
    if (status.st_size <= 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap asset is empty: " + path.string());
    }
    size_ = static_cast<std::size_t>(status.st_size);
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
      const auto message = std::string(std::strerror(errno));
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("cannot mmap asset " + path.string() + ": " +
                               message);
    }
    data_ = static_cast<const std::byte*>(mapped);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&& other) noexcept
      : path_(std::move(other.path_)),
        fd_(std::exchange(other.fd_, -1)),
        data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  MappedFile& operator=(MappedFile&& other) noexcept {
    if (this != &other) {
      reset();
      path_ = std::move(other.path_);
      fd_ = std::exchange(other.fd_, -1);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  ~MappedFile() { reset(); }

  const std::byte* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  void reset() noexcept {
    if (data_ != nullptr) {
      ::munmap(const_cast<std::byte*>(data_), size_);
      data_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    size_ = 0;
  }

  std::filesystem::path path_;
  int fd_ = -1;
  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

std::string required_string(const py::dict& mapping, const char* key) {
  if (!mapping.contains(key)) {
    throw std::invalid_argument(std::string("missing descriptor field: ") + key);
  }
  return py::cast<std::string>(mapping[key]);
}

std::uint64_t required_u64(const py::dict& mapping, const char* key) {
  if (!mapping.contains(key)) {
    throw std::invalid_argument(std::string("missing descriptor field: ") + key);
  }
  return py::cast<std::uint64_t>(mapping[key]);
}

bool optional_bool(const py::dict& mapping, const char* key,
                   bool default_value) {
  return mapping.contains(key) ? py::cast<bool>(mapping[key]) : default_value;
}

std::size_t dtype_item_size(const std::string& dtype) {
  if (dtype == "float32" || dtype == "<f4" || dtype == "f4") return 4;
  if (dtype == "float64" || dtype == "<f8" || dtype == "f8") return 8;
  if (dtype == "uint16" || dtype == "<u2" || dtype == "u2") return 2;
  if (dtype == "int64" || dtype == "<i8" || dtype == "i8" ||
      dtype == "uint64" || dtype == "<u8" || dtype == "u8") {
    return 8;
  }
  throw std::invalid_argument("unsupported native descriptor dtype: " + dtype);
}

template <typename T>
bool dtype_compatible(const std::string& dtype) {
  if constexpr (std::is_same_v<T, float>) {
    return dtype == "float32" || dtype == "<f4" || dtype == "f4";
  } else if constexpr (std::is_same_v<T, double>) {
    return dtype == "float64" || dtype == "<f8" || dtype == "f8";
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return dtype == "uint16" || dtype == "<u2" || dtype == "u2";
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return dtype == "int64" || dtype == "<i8" || dtype == "i8" ||
           dtype == "uint64" || dtype == "<u8" || dtype == "u8";
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return dtype == "uint64" || dtype == "<u8" || dtype == "u8" ||
           dtype == "int64" || dtype == "<i8" || dtype == "i8";
  }
  return false;
}

std::size_t element_count(const py::dict& section) {
  if (!section.contains("shape")) {
    throw std::invalid_argument("section descriptor is missing shape");
  }
  const auto shape = py::cast<std::vector<std::uint64_t>>(section["shape"]);
  std::size_t count = 1;
  for (const auto extent : shape) {
    if (extent != 0 && count > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::invalid_argument("section shape overflows size_t");
    }
    count *= static_cast<std::size_t>(extent);
  }
  return count;
}

template <typename T>
ArrayView<T> bind_region(const MappedFile& file, const py::dict& descriptor,
                         const std::string& label) {
  const auto dtype = required_string(descriptor, "dtype");
  if (!dtype_compatible<T>(dtype)) {
    throw std::invalid_argument(label + " has incompatible dtype " + dtype);
  }
  const auto offset = required_u64(descriptor, "offset");
  const auto nbytes = required_u64(descriptor, "nbytes");
  const auto count = element_count(descriptor);
  if (dtype_item_size(dtype) != sizeof(T) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
      count * sizeof(T) != nbytes) {
    throw std::invalid_argument(label + " shape/dtype/nbytes disagree");
  }
  if (offset > file.size() || nbytes > file.size() - offset) {
    throw CorruptIndexError(label + " lies outside mapped file");
  }
  if (offset % alignof(T) != 0) {
    throw CorruptIndexError(label + " offset violates native alignment");
  }
  const auto* address = file.data() + static_cast<std::size_t>(offset);
  return ArrayView<T>(reinterpret_cast<const T*>(address), count);
}

py::dict require_section(const py::dict& sections, const char* name) {
  if (!sections.contains(name)) {
    throw std::invalid_argument(std::string("manifest is missing section: ") +
                                name);
  }
  return py::cast<py::dict>(sections[name]);
}

struct NativeIndexOwner {
  MappedFile arrays;
  MappedFile vectors;
  MappedFile attrs;
  IndexView view;

  NativeIndexOwner(MappedFile arrays_file, MappedFile vectors_file,
                   MappedFile attrs_file)
      : arrays(std::move(arrays_file)),
        vectors(std::move(vectors_file)),
        attrs(std::move(attrs_file)) {}
};

IndexIdentity parse_identity(const py::dict& descriptor,
                             const py::dict& expected_hashes) {
  if (!descriptor.contains("identity")) {
    throw std::invalid_argument("native descriptor is missing identity");
  }
  const auto identity = py::cast<py::dict>(descriptor["identity"]);
  IndexIdentity output;
  output.algorithm_id = required_string(identity, "algorithm_id");
  output.algorithm_revision = required_string(identity, "algorithm_revision");
  output.config_sha256 = required_string(identity, "config_sha256");
  output.artifact_sha256 = required_string(identity, "artifact_sha256");
  output.corpus_vectors_sha256 =
      required_string(identity, "corpus_vectors_sha256");
  output.corpus_attrs_sha256 = required_string(identity, "corpus_attrs_sha256");
  output.python_source_tree_sha256 =
      required_string(identity, "python_source_tree_sha256");
  output.cpp_source_tree_sha256 =
      required_string(identity, "cpp_source_tree_sha256");
  output.cpp_binary_sha256 = required_string(identity, "cpp_binary_sha256");
  if (output.algorithm_id != "bounded_blowup_multiscale" ||
      output.algorithm_revision != "cpp_v1") {
    throw std::invalid_argument("native runtime only accepts bounded_blowup_multiscale/cpp_v1");
  }
  for (const auto item : expected_hashes) {
    const auto key = py::cast<std::string>(item.first);
    const auto expected = py::cast<std::string>(item.second);
    if (!identity.contains(py::str(key))) {
      throw std::invalid_argument("expected identity key is absent: " + key);
    }
    const auto observed = py::cast<std::string>(identity[py::str(key)]);
    if (observed != expected) {
      throw std::invalid_argument("native identity mismatch for " + key);
    }
  }
  return output;
}

std::shared_ptr<NativeIndexOwner> load_native_index(
    const std::filesystem::path& artifact_dir, const py::dict& corpus_assets,
    const py::dict& expected_hashes, const py::dict& verification) {
  if (required_string(verification, "endianness") != "little") {
    throw std::invalid_argument("only little-endian artifacts are supported");
  }
  const std::uint16_t endian_probe = 1;
  if (*reinterpret_cast<const std::uint8_t*>(&endian_probe) != 1) {
    throw std::runtime_error("native runtime requires a little-endian host");
  }
  if (!optional_bool(verification, "section_hashes_verified", false)) {
    throw std::invalid_argument(
        "Python adapter must verify every manifest section hash before mmap");
  }
  if (!optional_bool(verification, "corpus_hashes_verified", false)) {
    throw std::invalid_argument(
        "Python adapter must verify corpus identities before mmap");
  }
  if (!verification.contains("sections")) {
    throw std::invalid_argument("native descriptor is missing sections");
  }
  const auto sections = py::cast<py::dict>(verification["sections"]);
  const auto arrays_path = verification.contains("arrays_path")
                               ? std::filesystem::path(
                                     py::cast<std::string>(verification["arrays_path"]))
                               : artifact_dir / "arrays.bin";
  const auto vectors_desc = py::cast<py::dict>(corpus_assets["vectors"]);
  const auto attrs_desc = py::cast<py::dict>(corpus_assets["attrs_raw"]);
  MappedFile arrays_file(arrays_path);
  MappedFile vectors_file(required_string(vectors_desc, "path"));
  MappedFile attrs_file(required_string(attrs_desc, "path"));
  auto owner = std::make_shared<NativeIndexOwner>(
      std::move(arrays_file), std::move(vectors_file), std::move(attrs_file));
  auto& view = owner->view;

  view.object_count = required_u64(verification, "object_count");
  view.vector_dim = static_cast<std::uint32_t>(
      required_u64(verification, "vector_dim"));
  view.attr_dim =
      static_cast<std::uint16_t>(required_u64(verification, "attr_dim"));
  view.entries_per_query = static_cast<std::uint32_t>(
      required_u64(verification, "entries_per_query"));
  view.max_degree =
      static_cast<std::uint32_t>(required_u64(verification, "max_degree"));
  view.identity = parse_identity(verification, expected_hashes);

  view.vectors = bind_region<float>(owner->vectors, vectors_desc, "vectors");
  view.attrs_raw = bind_region<float>(owner->attrs, attrs_desc, "attrs_raw");

#define RFBB_BIND(type, field) \
  view.field = bind_region<type>(owner->arrays, require_section(sections, #field), #field)
  RFBB_BIND(double, domain_min);
  RFBB_BIND(double, domain_max);
  RFBB_BIND(double, lattice_unit);
  RFBB_BIND(std::uint64_t, family_dim_ptr);
  RFBB_BIND(std::uint16_t, family_dims);
  RFBB_BIND(std::uint16_t, family_level_count);
  RFBB_BIND(std::uint16_t, layout_template_aware);
  RFBB_BIND(std::uint64_t, family_template_ptr);
  RFBB_BIND(std::uint16_t, template_family);
  RFBB_BIND(std::uint16_t, template_local_id);
  RFBB_BIND(std::uint64_t, template_ratio_ptr);
  RFBB_BIND(double, template_envelope);
  RFBB_BIND(std::int64_t, level_service_width_coeff);
  RFBB_BIND(std::int64_t, level_side_length_coeff);
  RFBB_BIND(std::int64_t, level_stride_coeff);
  RFBB_BIND(std::uint64_t, family_level_block_ptr);
  RFBB_BIND(std::uint16_t, block_family);
  RFBB_BIND(std::uint16_t, block_template);
  RFBB_BIND(std::uint16_t, block_level);
  RFBB_BIND(std::uint64_t, block_grid_ptr);
  RFBB_BIND(std::int64_t, block_grid_coords);
  RFBB_BIND(std::int64_t, block_lattice_lo_coeff);
  RFBB_BIND(std::int64_t, block_lattice_hi_coeff);
  RFBB_BIND(std::uint64_t, sorted_block_lookup);
  RFBB_BIND(std::uint64_t, block_member_ptr);
  RFBB_BIND(std::int64_t, block_member_ids);
  RFBB_BIND(std::uint64_t, block_entry_ptr);
  RFBB_BIND(std::int64_t, block_entry_ids);
  RFBB_BIND(std::int64_t, block_anchor_id);
  RFBB_BIND(std::int64_t, physical_edge_src);
  RFBB_BIND(std::int64_t, physical_edge_dst);
  RFBB_BIND(std::uint64_t, view_row_ptr);
  RFBB_BIND(std::int64_t, row_object_id);
  RFBB_BIND(std::uint64_t, row_adj_ptr);
  RFBB_BIND(std::uint64_t, row_adj_target_row);
  RFBB_BIND(std::uint64_t, row_adj_edge_id);
#undef RFBB_BIND

  ValidationOptions options;
  options.verify_corpus_finite =
      optional_bool(verification, "verify_corpus_finite", true);
  options.verify_view_members_equal_rows =
      optional_bool(verification, "verify_view_members_equal_rows", true);
  options.verify_anchor_reachability =
      optional_bool(verification, "verify_anchor_reachability", true);
  options.verify_physical_edge_identity =
      optional_bool(verification, "verify_physical_edge_identity", true);
  {
    py::gil_scoped_release release;
    validate_index_view(view, options);
  }
  return owner;
}

QueryWindow parse_window(const py::object& object) {
  const auto mapping = py::cast<py::dict>(object);
  QueryWindow window;
  if (mapping.contains("dimensions")) {
    window.dimensions =
        py::cast<std::vector<std::uint16_t>>(mapping["dimensions"]);
    window.lower_closed =
        py::cast<std::vector<double>>(mapping["lower_closed"]);
    window.upper_closed =
        py::cast<std::vector<double>>(mapping["upper_closed"]);
    return window;
  }
  std::vector<std::pair<std::uint16_t, std::pair<double, double>>> bounds;
  bounds.reserve(mapping.size());
  for (const auto item : mapping) {
    const auto dim = py::cast<std::uint16_t>(item.first);
    const auto pair = py::cast<std::pair<double, double>>(item.second);
    bounds.push_back({dim, pair});
  }
  std::sort(bounds.begin(), bounds.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });
  for (const auto& item : bounds) {
    window.dimensions.push_back(item.first);
    window.lower_closed.push_back(item.second.first);
    window.upper_closed.push_back(item.second.second);
  }
  return window;
}

QueryWindow parse_packed_window(
    const py::array_t<std::uint16_t, py::array::c_style>& dimensions,
    const py::array_t<double, py::array::c_style>& lower_closed,
    const py::array_t<double, py::array::c_style>& upper_closed) {
  const auto dim_buffer = dimensions.request();
  const auto lower_buffer = lower_closed.request();
  const auto upper_buffer = upper_closed.request();
  if (dim_buffer.ndim != 1 || lower_buffer.ndim != 1 ||
      upper_buffer.ndim != 1) {
    throw std::invalid_argument("packed window arrays must be one-dimensional");
  }
  if (dim_buffer.shape[0] <= 0 ||
      dim_buffer.shape[0] != lower_buffer.shape[0] ||
      dim_buffer.shape[0] != upper_buffer.shape[0]) {
    throw std::invalid_argument(
        "packed window arrays must be non-empty and have equal lengths");
  }
  const auto count = static_cast<std::size_t>(dim_buffer.shape[0]);
  const auto* dim_ptr = static_cast<const std::uint16_t*>(dim_buffer.ptr);
  const auto* lower_ptr = static_cast<const double*>(lower_buffer.ptr);
  const auto* upper_ptr = static_cast<const double*>(upper_buffer.ptr);
  QueryWindow window;
  window.dimensions.assign(dim_ptr, dim_ptr + count);
  window.lower_closed.assign(lower_ptr, lower_ptr + count);
  window.upper_closed.assign(upper_ptr, upper_ptr + count);
  return window;
}

py::dict stats_to_dict(const QueryStats& stats) {
  py::dict output;
  output["family_id"] = stats.family_id;
  output["template_id"] = stats.template_id;
  output["block_id"] = stats.block_id;
  output["level"] = stats.level;
  output["block_grid_coords"] = stats.block_grid_coords;
  output["normalized_query_widths"] = stats.normalized_query_widths;
  output["normalized_block_widths"] = stats.normalized_block_widths;
  output["side_blowups"] = stats.side_blowups;
  output["template_ratio"] = stats.template_ratio;
  output["shape_tau"] = stats.shape_tau;
  output["selected_scale"] = stats.selected_scale;
  output["routing_clipped"] = stats.routing_clipped;
  output["block_object_count"] = stats.block_object_count;
  output["num_graphs_searched"] = stats.num_graphs_searched;
  output["requested_budget"] = stats.requested_budget;
  output["actual_budget"] = stats.actual_budget;
  output["unique_node_expansions"] = stats.actual_budget;
  output["entry_pool_considered"] = stats.entry_pool_considered;
  output["entries_selected"] = stats.entries_selected;
  output["entry_planning_distance_evaluations"] =
      stats.planning_distance_evaluations;
  output["planning_distance_evaluations"] =
      stats.planning_distance_evaluations;
  output["traversal_distance_evaluations"] =
      stats.traversal_distance_evaluations;
  output["distance_evaluations"] = stats.distance_evaluations;
  output["candidate_pushes"] = stats.candidate_pushes;
  output["candidate_pops"] = stats.candidate_pops;
  output["post_filter_checks"] = stats.post_filter_checks;
  output["post_filter_dimension_checks"] =
      stats.post_filter_dimension_checks;
  output["qualifying_expansions"] = stats.qualifying_expansions;
  output["result_updates"] = stats.result_updates;
  output["termination_reason"] =
      termination_reason_name(stats.termination_reason);
  output["fallback"] = stats.fallback;
  output["algorithm_id"] = stats.algorithm_id;
  output["algorithm_revision"] = stats.algorithm_revision;
  output["config_sha256"] = stats.config_sha256;
  output["artifact_sha256"] = stats.artifact_sha256;
  output["corpus_vectors_sha256"] = stats.corpus_vectors_sha256;
  output["corpus_attrs_sha256"] = stats.corpus_attrs_sha256;
  output["cpp_source_tree_sha256"] = stats.cpp_source_tree_sha256;
  output["cpp_binary_sha256"] = stats.cpp_binary_sha256;
  return output;
}

py::dict result_to_dict(const SearchResult& result) {
  py::list hits;
  for (const auto& hit : result.hits) {
    hits.append(py::make_tuple(hit.object_id, hit.distance));
  }
  py::dict output;
  output["hits"] = py::tuple(hits);
  output["stats"] = stats_to_dict(result.stats);
  return output;
}

py::tuple compact_stats_to_tuple(const QueryStats& stats) {
  // This order is the private compact ABI mirrored by
  // native_adapter._COMPACT_STATS_FIELDS.  Keep it scalar-only: geometry
  // vectors and immutable identity strings belong to the diagnostic path.
  py::tuple output(24);
  output[0] = py::int_(stats.family_id);
  output[1] = py::int_(stats.template_id);
  output[2] = py::int_(stats.block_id);
  output[3] = py::int_(stats.level);
  output[4] = py::float_(stats.shape_tau);
  output[5] = py::float_(stats.selected_scale);
  output[6] = py::bool_(stats.routing_clipped);
  output[7] = py::int_(stats.block_object_count);
  output[8] = py::int_(stats.num_graphs_searched);
  output[9] = py::int_(stats.requested_budget);
  output[10] = py::int_(stats.actual_budget);
  output[11] = py::int_(stats.entry_pool_considered);
  output[12] = py::int_(stats.entries_selected);
  output[13] = py::int_(stats.planning_distance_evaluations);
  output[14] = py::int_(stats.traversal_distance_evaluations);
  output[15] = py::int_(stats.distance_evaluations);
  output[16] = py::int_(stats.candidate_pushes);
  output[17] = py::int_(stats.candidate_pops);
  output[18] = py::int_(stats.post_filter_checks);
  output[19] = py::int_(stats.post_filter_dimension_checks);
  output[20] = py::int_(stats.qualifying_expansions);
  output[21] = py::int_(stats.result_updates);
  output[22] = py::int_(static_cast<std::uint8_t>(stats.termination_reason));
  output[23] = py::bool_(stats.fallback);
  return output;
}

py::tuple compact_result_to_tuple(const SearchResult& result) {
  py::tuple hits(result.hits.size());
  for (std::size_t index = 0; index < result.hits.size(); ++index) {
    const auto& hit = result.hits[index];
    hits[index] = py::make_tuple(hit.object_id, hit.distance);
  }
  return py::make_tuple(std::move(hits), compact_stats_to_tuple(result.stats));
}

template <typename T>
std::vector<T> required_vector(const py::dict& mapping, const char* key) {
  if (!mapping.contains(key)) {
    throw std::invalid_argument(std::string("native bulk input is missing ") +
                                key);
  }
  return py::cast<std::vector<T>>(mapping[key]);
}

template <typename T>
ArrayView<T> vector_view(const std::vector<T>& values) noexcept {
  return ArrayView<T>(values.data(), values.size());
}

Rank2Shape required_shape(const py::dict& asset, const char* label) {
  const auto shape = required_vector<std::uint64_t>(asset, "shape");
  if (shape.size() != 2U || shape[0] == 0U || shape[1] == 0U) {
    throw std::invalid_argument(std::string(label) +
                                " shape must contain two positive values");
  }
  return Rank2Shape{shape[0], shape[1]};
}

NativeBulkBlockDescriptor parse_native_bulk_descriptor(
    const py::dict& block) {
  NativeBulkBlockDescriptor output;
#define RFBB_BLOCK_VECTOR(type, field) \
  output.field = required_vector<type>(block, #field)
  RFBB_BLOCK_VECTOR(std::uint64_t, block_id);
  RFBB_BLOCK_VECTOR(std::uint32_t, block_level);
  RFBB_BLOCK_VECTOR(LogicalViewId, block_view_id);
  RFBB_BLOCK_VECTOR(std::uint64_t, block_component_sort_key);
  RFBB_BLOCK_VECTOR(std::uint8_t, block_kind);
  RFBB_BLOCK_VECTOR(std::int64_t, block_anchor_id);
  RFBB_BLOCK_VECTOR(std::uint64_t, block_member_ptr);
  RFBB_BLOCK_VECTOR(std::int64_t, block_member_ids);
  RFBB_BLOCK_VECTOR(std::uint64_t, block_entry_ptr);
  RFBB_BLOCK_VECTOR(std::int64_t, block_entry_ids);
  RFBB_BLOCK_VECTOR(std::uint64_t, block_child_ptr);
  RFBB_BLOCK_VECTOR(LogicalViewId, block_child_view_id);
#undef RFBB_BLOCK_VECTOR
  if (block.contains("block_child_ids")) {
    output.block_child_ids =
        py::cast<std::vector<LogicalViewId>>(block["block_child_ids"]);
  }
  return output;
}

RoutingLayoutSnapshot parse_routing_layout(const py::dict& routing) {
  IndexView view;
  view.object_count = required_u64(routing, "object_count");
  const auto vector_dim = required_u64(routing, "vector_dim");
  const auto attr_dim = required_u64(routing, "attr_dim");
  const auto entries_per_query = required_u64(routing, "entries_per_query");
  if (vector_dim > std::numeric_limits<std::uint32_t>::max() ||
      attr_dim > std::numeric_limits<std::uint16_t>::max() ||
      entries_per_query > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("native bulk routing scalar is out of range");
  }
  view.vector_dim = static_cast<std::uint32_t>(vector_dim);
  view.attr_dim = static_cast<std::uint16_t>(attr_dim);
  view.entries_per_query = static_cast<std::uint32_t>(entries_per_query);

#define RFBB_ROUTE_VECTOR(type, field) \
  const auto field = required_vector<type>(routing, #field); \
  view.field = vector_view(field)
  RFBB_ROUTE_VECTOR(double, domain_min);
  RFBB_ROUTE_VECTOR(double, domain_max);
  RFBB_ROUTE_VECTOR(double, lattice_unit);
  RFBB_ROUTE_VECTOR(std::uint64_t, family_dim_ptr);
  RFBB_ROUTE_VECTOR(std::uint16_t, family_dims);
  RFBB_ROUTE_VECTOR(std::uint16_t, family_level_count);
  RFBB_ROUTE_VECTOR(std::uint16_t, layout_template_aware);
  RFBB_ROUTE_VECTOR(std::uint64_t, family_template_ptr);
  RFBB_ROUTE_VECTOR(std::uint16_t, template_family);
  RFBB_ROUTE_VECTOR(std::uint16_t, template_local_id);
  RFBB_ROUTE_VECTOR(std::uint64_t, template_ratio_ptr);
  RFBB_ROUTE_VECTOR(double, template_envelope);
  RFBB_ROUTE_VECTOR(std::int64_t, level_service_width_coeff);
  RFBB_ROUTE_VECTOR(std::int64_t, level_side_length_coeff);
  RFBB_ROUTE_VECTOR(std::int64_t, level_stride_coeff);
  RFBB_ROUTE_VECTOR(std::uint64_t, family_level_block_ptr);
  RFBB_ROUTE_VECTOR(std::uint16_t, block_family);
  RFBB_ROUTE_VECTOR(std::uint16_t, block_template);
  RFBB_ROUTE_VECTOR(std::uint16_t, block_level);
  RFBB_ROUTE_VECTOR(std::uint64_t, block_grid_ptr);
  RFBB_ROUTE_VECTOR(std::int64_t, block_grid_coords);
  RFBB_ROUTE_VECTOR(std::int64_t, block_lattice_lo_coeff);
  RFBB_ROUTE_VECTOR(std::int64_t, block_lattice_hi_coeff);
  RFBB_ROUTE_VECTOR(std::uint64_t, sorted_block_lookup);
#undef RFBB_ROUTE_VECTOR
  return RoutingLayoutSnapshot::from_index_view(view);
}

NativeBulkBuildOptions parse_native_bulk_options(const py::dict& options) {
  NativeBulkBuildOptions output;
  if (options.contains("outer_worker_count")) {
    output.outer_worker_count =
        py::cast<std::uint32_t>(options["outer_worker_count"]);
    output.thread_count = output.outer_worker_count;
  }
  if (options.contains("timeout_milliseconds")) {
    output.timeout = std::chrono::milliseconds(
        py::cast<std::int64_t>(options["timeout_milliseconds"]));
  }
  if (options.contains("leaf_candidate_size")) {
    output.leaf_candidate_size =
        py::cast<std::uint32_t>(options["leaf_candidate_size"]);
    output.candidate_size = output.leaf_candidate_size;
  }
  return output;
}

const char* native_bulk_state_name(NativeBulkBuildState state) noexcept {
  switch (state) {
    case NativeBulkBuildState::kFailed:
      return "FAILED";
    case NativeBulkBuildState::kComplete:
      return "COMPLETE";
    case NativeBulkBuildState::kCancelled:
      return "CANCELLED";
    case NativeBulkBuildState::kTimedOut:
      return "TIMEOUT";
  }
  return "FAILED";
}

class PyNativeBulkHandle {
 public:
  explicit PyNativeBulkHandle(NativeBulkBuildResult result)
      : result_(std::move(result)) {
    result_.query_handle().prepare(workspace_, 10U);
  }

  static std::shared_ptr<PyNativeBulkHandle> build(
      const py::dict& corpus_assets, const py::dict& routing,
      const py::dict& block_descriptor, const py::dict& options) {
    if (!corpus_assets.contains("vectors") ||
        !corpus_assets.contains("attrs_raw")) {
      throw std::invalid_argument(
          "native bulk corpus_assets requires vectors and attrs_raw");
    }
    const auto vectors = py::cast<py::dict>(corpus_assets["vectors"]);
    const auto attrs = py::cast<py::dict>(corpus_assets["attrs_raw"]);
    const std::filesystem::path vectors_path(required_string(vectors, "path"));
    const std::filesystem::path attrs_path(required_string(attrs, "path"));
    const std::filesystem::path corpus_root =
        corpus_assets.contains("corpus_root")
            ? std::filesystem::path(required_string(corpus_assets, "corpus_root"))
            : vectors_path.parent_path();
    const auto relative_asset = [&](const std::filesystem::path& path) {
      if (!path.is_absolute()) return path;
      const auto relative = path.lexically_relative(corpus_root);
      if (relative.empty() || *relative.begin() == "..") {
        throw std::invalid_argument(
            "native bulk corpus asset lies outside corpus_root");
      }
      return relative;
    };
    VerifiedCorpusOpenRequest request(
        corpus_root,
        CorpusAssetClaim::from_hex(relative_asset(vectors_path),
                                   required_string(vectors, "sha256"),
                                   required_shape(vectors, "vectors")),
        CorpusAssetClaim::from_hex(relative_asset(attrs_path), required_string(attrs, "sha256"),
                                   required_shape(attrs, "attrs_raw")));
    auto descriptor = parse_native_bulk_descriptor(block_descriptor);
    auto route = parse_routing_layout(routing);
    const auto native_options = parse_native_bulk_options(options);
    const bool nonreportable_test_only =
        optional_bool(options, "nonreportable_test_only", false);
    const bool shadow_only = optional_bool(options, "shadow_only", false);
    if (nonreportable_test_only && shadow_only) {
      throw std::invalid_argument(
          "native bulk corpus mode must be exactly one of formal, shadow, or test");
    }
    std::unique_ptr<NativeBulkBuildResult> result;
    {
      py::gil_scoped_release release;
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
      auto corpus = nonreportable_test_only
                        ? VerifiedCorpusSnapshot::open_for_testing(
                              request, kVerifiedCorpusTestOnly)
                    : shadow_only
                        ? VerifiedCorpusSnapshot::open_shadow(
                              request, kVerifiedCorpusShadowOnly)
                        : VerifiedCorpusSnapshot::open(request);
#else
      if (nonreportable_test_only) {
        throw std::invalid_argument(
            "nonreportable_test_only requires the isolated local test target");
      }
      auto corpus = shadow_only
                        ? VerifiedCorpusSnapshot::open_shadow(
                              request, kVerifiedCorpusShadowOnly)
                        : VerifiedCorpusSnapshot::open(request);
#endif
      result = std::make_unique<NativeBulkBuildResult>(native_bulk_build(
          corpus, std::move(descriptor), std::move(route), native_options));
    }
    if (!result->valid()) {
      throw NativeBulkBuildError("native bulk build did not produce a valid handle");
    }
    return std::make_shared<PyNativeBulkHandle>(std::move(*result));
  }

  py::dict receipt() const {
    const auto& receipt = result_.receipt();
    py::dict output;
    output["status"] = native_bulk_state_name(receipt.state);
    output["complete"] = receipt.complete();
    output["reportable"] = false;
    output["algorithm_revision"] = receipt.algorithm_revision;
    output["numeric_revision"] = receipt.numeric_revision;
    output["build_identity"] = receipt.build_identity_hex;
    output["corpus_sha256"] = receipt.corpus_sha256;
    output["corpus_vectors_sha256"] = receipt.corpus_vectors_sha256;
    output["corpus_attrs_sha256"] = receipt.corpus_attrs_sha256;
    output["artifact_schema"] = receipt.artifact_schema;
    output["leaf_candidate_size"] = receipt.leaf_candidate_size;
    output["leaf_count"] = receipt.leaf_count;
    output["parent_count"] = receipt.parent_count;
    output["leaf_memberships"] = receipt.leaf_memberships;
    output["parent_memberships"] = receipt.parent_memberships;
    output["candidate_evaluations"] = receipt.candidate_evaluations;
    output["alpha_distance_evaluations"] =
        receipt.alpha_distance_evaluations;
    output["repair_dirty_rows"] = receipt.repair_dirty_rows;
    output["full_stitch_count"] = receipt.full_stitch_count;
    output["hnsw_count"] = receipt.hnsw_count;
    output["faiss_count"] = receipt.faiss_count;
    output["cuda_count"] = receipt.cuda_count;
    output["requested_outer_workers"] = receipt.requested_outer_workers;
    output["outer_workers"] = receipt.outer_workers;
    output["max_inner_workers"] = receipt.max_inner_workers;
    output["nested_worker_count"] = receipt.nested_worker_count;
    output["max_observed_degree"] = receipt.max_observed_degree;
    output["overflow_rows"] = receipt.overflow_rows;
    output["logical_rows"] = receipt.logical_rows;
    output["logical_edges"] = receipt.logical_edges;
    output["stored_rows"] = receipt.stored_rows;
    output["stored_edges"] = receipt.stored_edges;
    output["parent_stored_rows"] = receipt.parent_stored_rows;
    output["parent_stored_edges"] = receipt.parent_stored_edges;
    output["clean_child_ref_rows"] = receipt.clean_child_ref_rows;
    output["child_ref_count"] = receipt.child_ref_count;
    output["projection_entries"] = receipt.projection_entries;
    output["physical_edge_count"] = receipt.physical_edge_count;
    output["logical_section_bytes"] = receipt.logical_section_bytes;
    output["route_section_bytes"] = receipt.route_section_bytes;
    output["routing_section_bytes"] = receipt.routing_section_bytes;
    output["section_bytes"] = receipt.section_bytes;
    return output;
  }

  py::tuple search_compact(
      py::array_t<float, py::array::c_style> query_vector,
      py::array_t<std::uint16_t, py::array::c_style> dimensions,
      py::array_t<double, py::array::c_style> lower_closed,
      py::array_t<double, py::array::c_style> upper_closed,
      std::uint32_t top_k, std::uint64_t budget) {
    const auto query = query_vector.request();
    if (query.ndim != 1) {
      throw std::invalid_argument(
          "native bulk query vector must be a C-contiguous 1D float32 array");
    }
    const auto window =
        parse_packed_window(dimensions, lower_closed, upper_closed);
    LogicalAnnSearchResultView result;
    {
      std::lock_guard<std::mutex> lock(workspace_mutex_);
      py::gil_scoped_release release;
      result = result_.query_handle().search_into(
          static_cast<const float*>(query.ptr),
          static_cast<std::size_t>(query.shape[0]), window, top_k, budget,
          workspace_);
    }
    py::tuple hits(result.hits.size());
    for (std::size_t index = 0U; index < result.hits.size(); ++index) {
      hits[index] = py::make_tuple(result.hits[index].object_id,
                                   result.hits[index].distance);
    }
    return py::make_tuple(std::move(hits), result.compact_stats());
  }

  py::tuple search_compact_adaptive(
      py::array_t<float, py::array::c_style> query_vector,
      py::array_t<std::uint16_t, py::array::c_style> dimensions,
      py::array_t<double, py::array::c_style> lower_closed,
      py::array_t<double, py::array::c_style> upper_closed,
      std::uint32_t top_k, std::uint64_t budget,
      std::uint64_t minimum_expansions = 200U,
      std::uint64_t no_improvement_patience = 64U,
      bool require_frontier_bound = true) {
    const auto query = query_vector.request();
    if (query.ndim != 1) {
      throw std::invalid_argument(
          "native bulk query vector must be a C-contiguous 1D float32 array");
    }
    const auto window =
        parse_packed_window(dimensions, lower_closed, upper_closed);
    LogicalAnnSearchResultView result;
    {
      std::lock_guard<std::mutex> lock(workspace_mutex_);
      py::gil_scoped_release release;
      result = result_.query_handle().search_into_adaptive(
          static_cast<const float*>(query.ptr),
          static_cast<std::size_t>(query.shape[0]), window, top_k, budget,
          minimum_expansions, no_improvement_patience,
          require_frontier_bound, workspace_);
    }
    py::tuple hits(result.hits.size());
    for (std::size_t index = 0U; index < result.hits.size(); ++index) {
      hits[index] = py::make_tuple(result.hits[index].object_id,
                                   result.hits[index].distance);
    }
    return py::make_tuple(std::move(hits), result.compact_stats());
  }

 private:
  NativeBulkBuildResult result_;
  LogicalAnnWorkspace workspace_;
  std::mutex workspace_mutex_;
};

class PyIndexHandle {
 public:
  explicit PyIndexHandle(std::shared_ptr<NativeIndexOwner> owner)
      : owner_(std::move(owner)) {}

  static std::shared_ptr<PyIndexHandle> open(
      const std::string& artifact_dir, const py::dict& corpus_assets,
      const py::dict& expected_hashes, const py::dict& verification) {
    return std::make_shared<PyIndexHandle>(load_native_index(
        std::filesystem::path(artifact_dir), corpus_assets, expected_hashes,
        verification));
  }

  py::dict search(
      py::array_t<float, py::array::c_style> query_vector,
      const py::object& window_object, std::uint32_t top_k,
      std::uint64_t budget) const {
    const auto buffer = query_vector.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument("query vector must be a C-contiguous 1D float32 array");
    }
    const auto window = parse_window(window_object);
    SearchResult result;
    {
      py::gil_scoped_release release;
      result = search_index(owner_->view, static_cast<const float*>(buffer.ptr),
                            static_cast<std::size_t>(buffer.shape[0]), window,
                            top_k, budget, QueryOutputMode::kRich);
    }
    return result_to_dict(result);
  }

  py::tuple search_compact(
      py::array_t<float, py::array::c_style> query_vector,
      py::array_t<std::uint16_t, py::array::c_style> dimensions,
      py::array_t<double, py::array::c_style> lower_closed,
      py::array_t<double, py::array::c_style> upper_closed,
      std::uint32_t top_k, std::uint64_t budget) const {
    const auto buffer = query_vector.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument(
          "query vector must be a C-contiguous 1D float32 array");
    }
    const auto window =
        parse_packed_window(dimensions, lower_closed, upper_closed);
    SearchResult result;
    {
      py::gil_scoped_release release;
      result = search_index(owner_->view, static_cast<const float*>(buffer.ptr),
                            static_cast<std::size_t>(buffer.shape[0]), window,
                            top_k, budget, QueryOutputMode::kCompact);
    }
    return compact_result_to_tuple(result);
  }

  py::dict metadata() const {
    const auto& view = owner_->view;
    py::dict output;
    output["algorithm_id"] = view.identity.algorithm_id;
    output["algorithm_revision"] = view.identity.algorithm_revision;
    output["config_sha256"] = view.identity.config_sha256;
    output["artifact_sha256"] = view.identity.artifact_sha256;
    output["corpus_vectors_sha256"] = view.identity.corpus_vectors_sha256;
    output["corpus_attrs_sha256"] = view.identity.corpus_attrs_sha256;
    output["python_source_tree_sha256"] =
        view.identity.python_source_tree_sha256;
    output["cpp_source_tree_sha256"] = view.identity.cpp_source_tree_sha256;
    output["cpp_binary_sha256"] = view.identity.cpp_binary_sha256;
    output["object_count"] = view.object_count;
    output["vector_dim"] = view.vector_dim;
    output["attr_dim"] = view.attr_dim;
    output["family_count"] = view.family_count();
    output["template_count"] = view.template_family.size();
    output["template_aware"] = view.layout_template_aware[0] != 0;
    output["block_count"] = view.block_count();
    output["row_count"] = view.row_count();
    output["physical_edge_count"] = view.physical_edge_src.size();
    output["edge_reference_count"] = view.edge_reference_count();
    output["entries_per_query"] = view.entries_per_query;
    output["max_degree"] = view.max_degree;
    output["immutable"] = true;
    return output;
  }

 private:
  std::shared_ptr<NativeIndexOwner> owner_;
};

}  // namespace
}  // namespace bounded_blowup

PYBIND11_MODULE(_bounded_blowup_cpp, module) {
  module.doc() =
      "Private production runtime for bounded_blowup_multiscale/cpp_v1";
  module.attr("__cpp_source_tree_sha256__") = RFBB_CPP_SOURCE_TREE_SHA256;
  py::register_exception<bounded_blowup::CorruptIndexError>(
      module, "CorruptArtifactError");
  py::register_exception<bounded_blowup::UnsupportedDimensionSetError>(
      module, "UnsupportedDimensionSetError");
  py::register_exception<bounded_blowup::NoContainingBlockError>(
      module, "QueryInvariantViolation");
  py::class_<bounded_blowup::PyIndexHandle,
             std::shared_ptr<bounded_blowup::PyIndexHandle>>(
      module, "_CppIndexHandle")
      .def_static("open", &bounded_blowup::PyIndexHandle::open,
                  py::arg("artifact_dir"), py::arg("corpus_assets"),
                  py::arg("expected_hashes"), py::arg("verification"))
      .def("search", &bounded_blowup::PyIndexHandle::search,
           py::arg("query_vector"), py::arg("window"), py::arg("top_k"),
           py::arg("budget"))
      .def("search_compact", &bounded_blowup::PyIndexHandle::search_compact,
           py::arg("query_vector"), py::arg("dimensions"),
           py::arg("lower_closed"), py::arg("upper_closed"),
           py::arg("top_k"), py::arg("budget"))
      .def("metadata", &bounded_blowup::PyIndexHandle::metadata);
  py::class_<bounded_blowup::PyNativeBulkHandle,
             std::shared_ptr<bounded_blowup::PyNativeBulkHandle>>(
      module, "_NativeBulkHandle")
      .def_static("build", &bounded_blowup::PyNativeBulkHandle::build,
                  py::arg("corpus_assets"), py::arg("routing_layout"),
                  py::arg("block_descriptor"), py::arg("options"))
      .def("receipt", &bounded_blowup::PyNativeBulkHandle::receipt)
      .def("search_compact", &bounded_blowup::PyNativeBulkHandle::search_compact,
           py::arg("query_vector"), py::arg("dimensions"),
           py::arg("lower_closed"), py::arg("upper_closed"),
           py::arg("top_k"), py::arg("budget"))
      .def("search_compact_adaptive",
           &bounded_blowup::PyNativeBulkHandle::search_compact_adaptive,
           py::arg("query_vector"), py::arg("dimensions"),
           py::arg("lower_closed"), py::arg("upper_closed"),
           py::arg("top_k"), py::arg("budget"),
           py::arg("minimum_expansions") = 200U,
           py::arg("no_improvement_patience") = 64U,
           py::arg("require_frontier_bound") = true);

  // Independent standard-HNSW baseline.  This binding intentionally exposes
  // only the local implementation; it does not depend on hnswlib, Faiss, or
  // DiskANN.  The graph is built once and then shared by HNSW-In, HNSW-Post,
  // and VBASE-style query policies in the experiment driver.
  using HnswArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
  py::class_<bounded_blowup::StandardHnswIndex,
             std::shared_ptr<bounded_blowup::StandardHnswIndex>>(
      module, "_StandardHNSW")
      .def_static(
          "build",
          [](HnswArray vectors, std::uint32_t m, std::uint32_t ef_construction,
             std::uint64_t seed) {
            const py::buffer_info info = vectors.request();
            if (info.ndim != 2 || info.shape[0] <= 0 || info.shape[1] <= 0) {
              throw std::invalid_argument("HNSW build vectors must be a non-empty 2D array");
            }
            bounded_blowup::HnswBuildOptions options;
            options.m = m;
            options.ef_construction = ef_construction;
            options.seed = seed;
            std::shared_ptr<bounded_blowup::StandardHnswIndex> index;
            {
              py::gil_scoped_release release;
              index = bounded_blowup::StandardHnswIndex::build(
                  static_cast<const float*>(info.ptr),
                  static_cast<std::size_t>(info.shape[0]),
                  static_cast<std::size_t>(info.shape[1]), options);
            }
            return index;
          },
          py::arg("vectors"), py::arg("m") = 16U,
          py::arg("ef_construction") = 200U, py::arg("seed") = 0U)
      .def_static("load", [](const std::string& path) {
        py::gil_scoped_release release;
        return bounded_blowup::StandardHnswIndex::load(path);
      }, py::arg("path"))
      .def("save", [](const bounded_blowup::StandardHnswIndex& index,
                       const std::string& path) {
        py::gil_scoped_release release;
        index.save(path);
      }, py::arg("path"))
      .def("search", [](const bounded_blowup::StandardHnswIndex& index,
                         HnswArray query, std::size_t k, std::size_t ef,
                         py::object allowed_object) {
        const py::buffer_info query_info = query.request();
        if (query_info.ndim != 1 ||
            static_cast<std::size_t>(query_info.shape[0]) != index.dimension()) {
          throw std::invalid_argument("HNSW query must be a 1D float32 vector with matching dimension");
        }
        py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
            allowed_array;
        const std::uint8_t* allowed = nullptr;
        if (!allowed_object.is_none()) {
          allowed_array = allowed_object.cast<decltype(allowed_array)>();
          const py::buffer_info allowed_info = allowed_array.request();
          if (allowed_info.ndim != 1 ||
              static_cast<std::size_t>(allowed_info.shape[0]) != index.size()) {
            throw std::invalid_argument("HNSW allowed mask must have one byte per object");
          }
          allowed = static_cast<const std::uint8_t*>(allowed_info.ptr);
        }
        std::vector<bounded_blowup::HnswSearchHit> hits;
        {
          py::gil_scoped_release release;
          hits = index.search(static_cast<const float*>(query_info.ptr), k, ef, allowed);
        }
        py::list output;
        for (const auto& hit : hits) {
          output.append(py::make_tuple(hit.id, hit.distance));
        }
        return output;
      }, py::arg("query"), py::arg("k"), py::arg("ef"),
         py::arg("allowed") = py::none())
      .def("size", &bounded_blowup::StandardHnswIndex::size)
      .def("dimension", &bounded_blowup::StandardHnswIndex::dimension)
      .def("metadata", [](const bounded_blowup::StandardHnswIndex& index) {
        py::dict output;
        output["implementation_id"] = "rfbb_standard_hnsw_cpp_v1";
        output["backend_mode"] = "standard_hnsw_cpp";
        output["count"] = index.size();
        output["dimensions"] = index.dimension();
        output["M"] = index.m();
        output["ef_construction"] = index.ef_construction();
        output["seed"] = index.seed();
        output["uses_hnswlib"] = false;
        output["uses_faiss"] = false;
        output["uses_diskann"] = false;
        return output;
      });
}

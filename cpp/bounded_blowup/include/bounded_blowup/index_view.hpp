// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bounded_blowup {

template <typename T>
class ArrayView {
 public:
  using value_type = T;

  constexpr ArrayView() noexcept = default;
  constexpr ArrayView(const T* data, std::size_t size) noexcept
      : data_(data), size_(size) {}

  constexpr const T* data() const noexcept { return data_; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr const T& operator[](std::size_t index) const noexcept {
    return data_[index];
  }
  constexpr const T* begin() const noexcept { return data_; }
  constexpr const T* end() const noexcept { return data_ + size_; }

  constexpr ArrayView subview(std::size_t offset,
                              std::size_t count) const noexcept {
    return ArrayView(data_ + offset, count);
  }

 private:
  const T* data_ = nullptr;
  std::size_t size_ = 0;
};

struct IndexIdentity {
  std::string algorithm_id;
  std::string algorithm_revision;
  std::string config_sha256;
  std::string artifact_sha256;
  std::string corpus_vectors_sha256;
  std::string corpus_attrs_sha256;
  std::string python_source_tree_sha256;
  std::string cpp_source_tree_sha256;
  std::string cpp_binary_sha256;
};

// Non-owning, immutable Structure-of-Arrays view over the mmap-backed artifact
// and corpus assets. The owning native handle guarantees the pointed-to storage
// outlives every query.
struct IndexView {
  std::uint64_t object_count = 0;
  std::uint32_t vector_dim = 0;
  std::uint16_t attr_dim = 0;
  std::uint32_t entries_per_query = 1;
  std::uint32_t max_degree = 0;

  ArrayView<float> vectors;
  ArrayView<float> attrs_raw;
  ArrayView<double> domain_min;
  ArrayView<double> domain_max;
  // One scalar for the current cpp_v1 layout, or one value per attribute.
  ArrayView<double> lattice_unit;

  ArrayView<std::uint64_t> family_dim_ptr;
  ArrayView<std::uint16_t> family_dims;
  ArrayView<std::uint16_t> family_level_count;
  ArrayView<std::uint16_t> layout_template_aware;
  ArrayView<std::uint64_t> family_template_ptr;
  ArrayView<std::uint16_t> template_family;
  ArrayView<std::uint16_t> template_local_id;
  ArrayView<std::uint64_t> template_ratio_ptr;
  ArrayView<double> template_envelope;
  // Geometry is shared by all dimension families; coefficients are indexed
  // only by level. family_level_block_ptr remains flattened by family/level.
  ArrayView<std::int64_t> level_service_width_coeff;
  ArrayView<std::int64_t> level_side_length_coeff;
  ArrayView<std::int64_t> level_stride_coeff;
  ArrayView<std::uint64_t> family_level_block_ptr;

  ArrayView<std::uint16_t> block_family;
  ArrayView<std::uint16_t> block_template;
  ArrayView<std::uint16_t> block_level;
  ArrayView<std::uint64_t> block_grid_ptr;
  ArrayView<std::int64_t> block_grid_coords;
  ArrayView<std::int64_t> block_lattice_lo_coeff;
  ArrayView<std::int64_t> block_lattice_hi_coeff;
  // A permutation of block IDs. Each family-level range is lexicographically
  // sorted by block_grid_coords and therefore supports binary search.
  ArrayView<std::uint64_t> sorted_block_lookup;

  ArrayView<std::uint64_t> block_member_ptr;
  ArrayView<std::int64_t> block_member_ids;
  ArrayView<std::uint64_t> block_entry_ptr;
  ArrayView<std::int64_t> block_entry_ids;
  ArrayView<std::int64_t> block_anchor_id;

  ArrayView<std::int64_t> physical_edge_src;
  ArrayView<std::int64_t> physical_edge_dst;
  ArrayView<std::uint64_t> view_row_ptr;
  ArrayView<std::int64_t> row_object_id;
  ArrayView<std::uint64_t> row_adj_ptr;
  ArrayView<std::uint64_t> row_adj_target_row;
  ArrayView<std::uint64_t> row_adj_edge_id;

  IndexIdentity identity;

  std::size_t family_count() const noexcept {
    return family_level_count.size();
  }
  std::size_t block_count() const noexcept { return block_family.size(); }
  std::size_t row_count() const noexcept { return row_object_id.size(); }
  std::size_t edge_reference_count() const noexcept {
    return row_adj_target_row.size();
  }

  std::uint64_t flattened_level_offset(std::uint16_t family_id,
                                       std::uint16_t template_id = 0) const;
  ArrayView<std::uint16_t> dimensions(std::uint16_t family_id) const;
  std::size_t template_count(std::uint16_t family_id) const;
  std::uint64_t global_template_id(std::uint16_t family_id,
                                   std::uint16_t template_id) const;
  ArrayView<double> template_ratio(std::uint16_t family_id,
                                   std::uint16_t template_id) const;
  ArrayView<std::int64_t> grid_coords(std::uint64_t block_id) const;
  ArrayView<std::int64_t> lattice_lo(std::uint64_t block_id) const;
  ArrayView<std::int64_t> lattice_hi(std::uint64_t block_id) const;
  ArrayView<std::int64_t> members(std::uint64_t block_id) const;
  ArrayView<std::int64_t> extra_entries(std::uint64_t block_id) const;
};

struct ValidationOptions {
  bool verify_corpus_finite = true;
  bool verify_view_members_equal_rows = true;
  bool verify_anchor_reachability = true;
  bool verify_physical_edge_identity = true;
};

class CorruptIndexError : public std::runtime_error {
 public:
  explicit CorruptIndexError(const std::string& message)
      : std::runtime_error(message) {}
};

void validate_index_view(const IndexView& index,
                         const ValidationOptions& options = {});

}  // namespace bounded_blowup

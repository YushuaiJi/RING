// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#include "bounded_blowup/index_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

namespace bounded_blowup {
namespace {

[[noreturn]] void corrupt(const std::string& message) {
  throw CorruptIndexError(message);
}

std::uint16_t checked_template_id(std::size_t value) {
  if (value > std::numeric_limits<std::uint16_t>::max()) {
    corrupt("template ID exceeds uint16 sentinel");
  }
  return static_cast<std::uint16_t>(value);
}

void require(bool condition, const std::string& message) {
  if (!condition) {
    corrupt(message);
  }
}

std::size_t checked_product(std::uint64_t a, std::uint64_t b,
                            const char* name) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    corrupt(std::string(name) + " size overflows size_t");
  }
  return static_cast<std::size_t>(a * b);
}

void validate_ptr(ArrayView<std::uint64_t> ptr, std::size_t expected_length,
                  std::size_t terminal, const char* name) {
  require(ptr.size() == expected_length,
          std::string(name) + " has unexpected length");
  require(!ptr.empty() && ptr[0] == 0,
          std::string(name) + " must start at zero");
  for (std::size_t i = 1; i < ptr.size(); ++i) {
    require(ptr[i - 1] <= ptr[i],
            std::string(name) + " is not monotone");
  }
  require(ptr[ptr.size() - 1] == terminal,
          std::string(name) + " terminal does not match payload length");
}

template <typename T>
bool strictly_increasing(ArrayView<T> values) {
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (!(values[i - 1] < values[i])) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool contains_sorted(ArrayView<T> values, T value) {
  return std::binary_search(values.begin(), values.end(), value);
}

int compare_grid(const IndexView& index, std::uint64_t lhs,
                 std::uint64_t rhs) {
  const auto a = index.grid_coords(lhs);
  const auto b = index.grid_coords(rhs);
  require(a.size() == b.size(), "block grid arity mismatch");
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return 0;
}

}  // namespace

std::uint64_t IndexView::flattened_level_offset(
    std::uint16_t family_id, std::uint16_t template_id) const {
  if (family_id >= family_count()) {
    throw CorruptIndexError("family id is out of range");
  }
  if (template_id >= template_count(family_id)) {
    throw CorruptIndexError("template id is out of range");
  }
  std::uint64_t offset = 0;
  for (std::uint16_t family = 0; family < family_id; ++family) {
    offset += template_count(family) * family_level_count[family];
  }
  return offset + template_id * family_level_count[family_id];
}

ArrayView<std::uint16_t> IndexView::dimensions(
    std::uint16_t family_id) const {
  if (family_id >= family_count() || family_dim_ptr.size() != family_count() + 1) {
    throw CorruptIndexError("family dimension lookup is out of range");
  }
  const auto begin = family_dim_ptr[family_id];
  const auto end = family_dim_ptr[family_id + 1];
  if (begin > end || end > family_dims.size()) {
    throw CorruptIndexError("family dimension pointer is corrupt");
  }
  return family_dims.subview(static_cast<std::size_t>(begin),
                             static_cast<std::size_t>(end - begin));
}

std::size_t IndexView::template_count(std::uint16_t family_id) const {
  if (family_id >= family_count() ||
      family_template_ptr.size() != family_count() + 1) {
    throw CorruptIndexError("family template lookup is out of range");
  }
  const auto begin = family_template_ptr[family_id];
  const auto end = family_template_ptr[family_id + 1];
  if (begin > end || end > template_family.size()) {
    throw CorruptIndexError("family template pointer is corrupt");
  }
  return static_cast<std::size_t>(end - begin);
}

std::uint64_t IndexView::global_template_id(
    std::uint16_t family_id, std::uint16_t template_id) const {
  if (template_id >= template_count(family_id)) {
    throw CorruptIndexError("template id is out of range");
  }
  return family_template_ptr[family_id] + template_id;
}

ArrayView<double> IndexView::template_ratio(
    std::uint16_t family_id, std::uint16_t template_id) const {
  const auto global = global_template_id(family_id, template_id);
  if (template_ratio_ptr.size() != template_family.size() + 1) {
    throw CorruptIndexError("template ratio pointer has the wrong length");
  }
  const auto begin = template_ratio_ptr[global];
  const auto end = template_ratio_ptr[global + 1];
  if (begin > end || end > template_envelope.size()) {
    throw CorruptIndexError("template ratio pointer is corrupt");
  }
  return template_envelope.subview(static_cast<std::size_t>(begin),
                                   static_cast<std::size_t>(end - begin));
}

ArrayView<std::int64_t> IndexView::grid_coords(
    std::uint64_t block_id) const {
  if (block_id >= block_count() || block_grid_ptr.size() != block_count() + 1) {
    throw CorruptIndexError("block grid lookup is out of range");
  }
  const auto begin = block_grid_ptr[block_id];
  const auto end = block_grid_ptr[block_id + 1];
  if (begin > end || end > block_grid_coords.size()) {
    throw CorruptIndexError("block grid pointer is corrupt");
  }
  return block_grid_coords.subview(static_cast<std::size_t>(begin),
                                   static_cast<std::size_t>(end - begin));
}

ArrayView<std::int64_t> IndexView::lattice_lo(
    std::uint64_t block_id) const {
  const auto grid = grid_coords(block_id);
  const auto begin = block_grid_ptr[block_id];
  return block_lattice_lo_coeff.subview(static_cast<std::size_t>(begin),
                                        grid.size());
}

ArrayView<std::int64_t> IndexView::lattice_hi(
    std::uint64_t block_id) const {
  const auto grid = grid_coords(block_id);
  const auto begin = block_grid_ptr[block_id];
  return block_lattice_hi_coeff.subview(static_cast<std::size_t>(begin),
                                        grid.size());
}

ArrayView<std::int64_t> IndexView::members(std::uint64_t block_id) const {
  if (block_id >= block_count() || block_member_ptr.size() != block_count() + 1) {
    throw CorruptIndexError("block member lookup is out of range");
  }
  const auto begin = block_member_ptr[block_id];
  const auto end = block_member_ptr[block_id + 1];
  if (begin > end || end > block_member_ids.size()) {
    throw CorruptIndexError("block member pointer is corrupt");
  }
  return block_member_ids.subview(static_cast<std::size_t>(begin),
                                  static_cast<std::size_t>(end - begin));
}

ArrayView<std::int64_t> IndexView::extra_entries(
    std::uint64_t block_id) const {
  if (block_id >= block_count() || block_entry_ptr.size() != block_count() + 1) {
    throw CorruptIndexError("block entry lookup is out of range");
  }
  const auto begin = block_entry_ptr[block_id];
  const auto end = block_entry_ptr[block_id + 1];
  if (begin > end || end > block_entry_ids.size()) {
    throw CorruptIndexError("block entry pointer is corrupt");
  }
  return block_entry_ids.subview(static_cast<std::size_t>(begin),
                                 static_cast<std::size_t>(end - begin));
}

void validate_index_view(const IndexView& index,
                         const ValidationOptions& options) {
  require(index.object_count > 0, "object_count must be positive");
  require(index.vector_dim > 0, "vector_dim must be positive");
  require(index.attr_dim > 0, "attr_dim must be positive");
  require(index.entries_per_query > 0, "entries_per_query must be positive");
  require(index.max_degree > 0, "max_degree must be positive");
  require(index.vectors.size() ==
              checked_product(index.object_count, index.vector_dim, "vectors"),
          "vectors shape does not match object_count x vector_dim");
  require(index.attrs_raw.size() ==
              checked_product(index.object_count, index.attr_dim, "attrs_raw"),
          "attrs_raw shape does not match object_count x attr_dim");
  require(index.domain_min.size() == index.attr_dim,
          "domain_min length does not match attr_dim");
  require(index.domain_max.size() == index.attr_dim,
          "domain_max length does not match attr_dim");
  require(index.lattice_unit.size() == 1 ||
              index.lattice_unit.size() == index.attr_dim,
          "lattice_unit must be scalar or per-attribute");

  for (std::size_t dim = 0; dim < index.attr_dim; ++dim) {
    require(std::isfinite(index.domain_min[dim]) &&
                std::isfinite(index.domain_max[dim]),
            "domain bounds must be finite");
    require(index.domain_min[dim] <= index.domain_max[dim],
            "domain_min exceeds domain_max");
    const double unit = index.lattice_unit.size() == 1
                            ? index.lattice_unit[0]
                            : index.lattice_unit[dim];
    require(std::isfinite(unit) && unit > 0.0,
            "lattice_unit must be finite and positive");
  }

  if (options.verify_corpus_finite) {
    for (const float value : index.vectors) {
      require(std::isfinite(value), "vectors contain NaN or Inf");
    }
    for (const float value : index.attrs_raw) {
      require(std::isfinite(value), "attrs_raw contain NaN or Inf");
    }
  }

  const std::size_t family_count = index.family_count();
  require(family_count > 0, "at least one dimension family is required");
  require(family_count <= std::numeric_limits<std::uint16_t>::max(),
          "family count exceeds uint16 method identity");
  validate_ptr(index.family_dim_ptr, family_count + 1,
               index.family_dims.size(), "family_dim_ptr");
  require(index.layout_template_aware.size() == 1 &&
              index.layout_template_aware[0] <= 1,
          "layout_template_aware must contain one boolean value");
  validate_ptr(index.family_template_ptr, family_count + 1,
               index.template_family.size(), "family_template_ptr");
  require(index.template_local_id.size() == index.template_family.size(),
          "template identity arrays have different lengths");
  validate_ptr(index.template_ratio_ptr, index.template_family.size() + 1,
               index.template_envelope.size(), "template_ratio_ptr");
  std::uint64_t total_levels = 0;
  std::uint16_t maximum_level_count = 0;
  ArrayView<std::uint16_t> previous_family;
  for (std::uint16_t family = 0; family < family_count; ++family) {
    const auto dims = index.dimensions(family);
    require(!dims.empty(), "dimension family cannot be empty");
    require(strictly_increasing(dims),
            "dimension family must be sorted and unique");
    for (const auto dim : dims) {
      require(dim < index.attr_dim, "family dimension is out of range");
    }
    if (family > 0) {
      const bool canonical_order = std::lexicographical_compare(
          previous_family.begin(), previous_family.end(), dims.begin(), dims.end());
      require(canonical_order, "dimension families are not in canonical order");
    }
    previous_family = dims;
    require(index.family_level_count[family] > 0,
            "every family must have at least one level");
    const auto template_count = index.template_count(family);
    require(template_count > 0,
            "every family must have at least one shape template");
    require(template_count <= std::numeric_limits<std::uint16_t>::max(),
            "family template count exceeds uint16 sentinel");
    for (std::size_t local = 0U; local < template_count; ++local) {
      const std::uint16_t local_id = checked_template_id(local);
      const auto global = index.global_template_id(family, local_id);
      require(index.template_family[global] == family &&
                  index.template_local_id[global] == local_id,
              "template family/local identity is not canonical");
      const auto ratio = index.template_ratio(family, local_id);
      require(ratio.size() == dims.size(),
              "template ratio rank does not match its family");
      for (const auto value : ratio) {
        require(std::isfinite(value) && value > 0.0,
                "template ratio must be finite and positive");
      }
    }
    const std::uint64_t template_count_u64 =
        static_cast<std::uint64_t>(template_count);
    const std::uint64_t level_count_u64 =
        static_cast<std::uint64_t>(index.family_level_count[family]);
    require(level_count_u64 != 0U &&
                template_count_u64 <=
                    std::numeric_limits<std::uint64_t>::max() /
                        level_count_u64,
            "family level count overflows uint64");
    const std::uint64_t family_levels = template_count_u64 * level_count_u64;
    require(total_levels <=
                std::numeric_limits<std::uint64_t>::max() - family_levels,
            "total level count overflows uint64");
    total_levels += family_levels;
    maximum_level_count =
        std::max(maximum_level_count, index.family_level_count[family]);
  }
  require(index.level_service_width_coeff.size() == maximum_level_count,
          "level_service_width_coeff has unexpected length");
  require(index.level_side_length_coeff.size() == maximum_level_count,
          "level_side_length_coeff has unexpected length");
  require(index.level_stride_coeff.size() == maximum_level_count,
          "level_stride_coeff has unexpected length");
  for (std::size_t i = 0; i < maximum_level_count; ++i) {
    require(index.level_service_width_coeff[i] > 0,
            "service width coefficient must be positive");
    require(index.level_side_length_coeff[i] > 0,
            "side length coefficient must be positive");
    require(index.level_stride_coeff[i] > 0,
            "stride coefficient must be positive");
  }

  const std::size_t block_count = index.block_count();
  require(block_count > 0, "artifact must contain a nonempty block");
  require(index.block_level.size() == block_count,
          "block_level length mismatch");
  require(index.block_template.size() == block_count,
          "block_template length mismatch");
  require(index.block_anchor_id.size() == block_count,
          "block_anchor_id length mismatch");
  require(total_levels < static_cast<std::uint64_t>(
                             std::numeric_limits<std::size_t>::max()),
          "total level count exceeds size_t");
  const std::size_t total_levels_size = static_cast<std::size_t>(total_levels);
  validate_ptr(index.family_level_block_ptr,
               total_levels_size + 1U, block_count,
               "family_level_block_ptr");
  validate_ptr(index.block_grid_ptr, block_count + 1,
               index.block_grid_coords.size(), "block_grid_ptr");
  require(index.block_lattice_lo_coeff.size() ==
              index.block_grid_coords.size(),
          "block_lattice_lo_coeff length mismatch");
  require(index.block_lattice_hi_coeff.size() ==
              index.block_grid_coords.size(),
          "block_lattice_hi_coeff length mismatch");
  require(index.sorted_block_lookup.size() == block_count,
          "sorted_block_lookup length mismatch");
  validate_ptr(index.block_member_ptr, block_count + 1,
               index.block_member_ids.size(), "block_member_ptr");
  validate_ptr(index.block_entry_ptr, block_count + 1,
               index.block_entry_ids.size(), "block_entry_ptr");
  validate_ptr(index.view_row_ptr, block_count + 1,
               index.row_object_id.size(), "view_row_ptr");
  validate_ptr(index.row_adj_ptr, index.row_object_id.size() + 1,
               index.row_adj_target_row.size(), "row_adj_ptr");
  require(index.row_adj_edge_id.size() == index.row_adj_target_row.size(),
          "row adjacency arrays have different lengths");
  require(index.physical_edge_src.size() == index.physical_edge_dst.size(),
          "physical edge source/destination lengths differ");

  std::vector<std::uint8_t> lookup_seen(block_count, 0);
  std::vector<std::uint64_t> lookup_position(block_count, 0);
  for (std::size_t flat_level = 0; flat_level < total_levels_size;
       ++flat_level) {
    const auto begin = index.family_level_block_ptr[flat_level];
    const auto end = index.family_level_block_ptr[flat_level + 1];
    std::uint64_t previous_block = std::numeric_limits<std::uint64_t>::max();
    for (auto pos = begin; pos < end; ++pos) {
      const auto block = index.sorted_block_lookup[pos];
      require(block < block_count, "sorted_block_lookup contains invalid block");
      require(lookup_seen[block] == 0,
              "sorted_block_lookup is not a permutation");
      lookup_seen[block] = 1;
      lookup_position[block] = pos;
      if (previous_block != std::numeric_limits<std::uint64_t>::max()) {
        require(compare_grid(index, previous_block, block) < 0,
                "family-level block keys are not strictly sorted");
      }
      previous_block = block;
    }
  }
  require(std::all_of(lookup_seen.begin(), lookup_seen.end(),
                      [](std::uint8_t value) { return value == 1; }),
          "sorted_block_lookup omits a block");

  for (std::uint64_t block = 0; block < block_count; ++block) {
    const auto family = index.block_family[block];
    const auto template_id = index.block_template[block];
    const auto level = index.block_level[block];
    require(family < family_count, "block family is out of range");
    require(template_id < index.template_count(family),
            "block template is out of range");
    require(level < index.family_level_count[family],
            "block level is out of range");
    const auto flat_level_range =
        index.flattened_level_offset(family, template_id) + level;
    require(lookup_position[block] >=
                index.family_level_block_ptr[flat_level_range] &&
                lookup_position[block] <
                    index.family_level_block_ptr[flat_level_range + 1],
            "block appears outside its declared family-level lookup range");
    const auto dims = index.dimensions(family);
    const auto grid = index.grid_coords(block);
    const auto lo = index.lattice_lo(block);
    const auto hi = index.lattice_hi(block);
    require(grid.size() == dims.size(), "block grid arity mismatch");
    const auto stride = index.level_stride_coeff[level];
    const auto side = index.level_side_length_coeff[level];
    for (std::size_t d = 0; d < grid.size(); ++d) {
      const __int128 expected_lo =
          static_cast<__int128>(grid[d]) * static_cast<__int128>(stride);
      const __int128 expected_hi = expected_lo + static_cast<__int128>(side);
      require(expected_lo >= std::numeric_limits<std::int64_t>::min() &&
                  expected_lo <= std::numeric_limits<std::int64_t>::max() &&
                  expected_hi >= std::numeric_limits<std::int64_t>::min() &&
                  expected_hi <= std::numeric_limits<std::int64_t>::max(),
              "block lattice coefficient overflows int64");
      require(lo[d] == static_cast<std::int64_t>(expected_lo) &&
                  hi[d] == static_cast<std::int64_t>(expected_hi),
              "block lattice bounds do not match grid/level coefficients");
      require(lo[d] < hi[d], "block lattice bounds are not positive width");
    }

    const auto members = index.members(block);
    require(!members.empty(), "only nonempty blocks may be materialized");
    require(strictly_increasing(members),
            "block membership must be strictly increasing");
    for (const auto object : members) {
      require(object >= 0 && static_cast<std::uint64_t>(object) < index.object_count,
              "block membership object is out of range");
    }
    const auto anchor = index.block_anchor_id[block];
    require(contains_sorted(members, anchor),
            "mandatory anchor is not a block member");
    const auto entries = index.extra_entries(block);
    std::vector<std::int64_t> canonical_entries(entries.begin(), entries.end());
    std::sort(canonical_entries.begin(), canonical_entries.end());
    require(std::adjacent_find(canonical_entries.begin(),
                               canonical_entries.end()) ==
                canonical_entries.end(),
            "extra entries must not contain duplicates");
    for (const auto object : entries) {
      require(contains_sorted(members, object),
              "extra entry is not a block member");
      require(object != anchor,
              "mandatory anchor must not be repeated as an extra entry");
    }

    const auto row_begin = index.view_row_ptr[block];
    const auto row_end = index.view_row_ptr[block + 1];
    require(row_end - row_begin == members.size(),
            "view row count does not match membership count");
    require(row_end - row_begin <=
                std::numeric_limits<std::uint32_t>::max(),
            "view row count exceeds compact local-row addressing");
    const auto rows = index.row_object_id.subview(
        static_cast<std::size_t>(row_begin),
        static_cast<std::size_t>(row_end - row_begin));
    require(strictly_increasing(rows),
            "view rows must be strictly increasing by object ID");
    for (const auto object : rows) {
      require(object >= 0 &&
                  static_cast<std::uint64_t>(object) < index.object_count,
              "view row object is out of range");
    }
    if (options.verify_view_members_equal_rows) {
      for (std::size_t i = 0; i < members.size(); ++i) {
        require(index.row_object_id[row_begin + i] == members[i],
                "view rows do not equal sorted membership");
      }
    }

    for (auto row = row_begin; row < row_end; ++row) {
      const auto adj_begin = index.row_adj_ptr[row];
      const auto adj_end = index.row_adj_ptr[row + 1];
      require(adj_end - adj_begin <= index.max_degree,
              "view row exceeds max_degree");
      for (auto ref = adj_begin; ref < adj_end; ++ref) {
        const auto target_row = index.row_adj_target_row[ref];
        require(target_row >= row_begin && target_row < row_end,
                "adjacency target escapes its logical view");
        const auto edge_id = index.row_adj_edge_id[ref];
        require(edge_id < index.physical_edge_src.size(),
                "row adjacency references an invalid physical edge");
        if (options.verify_physical_edge_identity) {
          require(index.physical_edge_src[edge_id] == index.row_object_id[row] &&
                      index.physical_edge_dst[edge_id] ==
                          index.row_object_id[target_row],
                  "physical edge identity disagrees with active view reference");
        }
      }
    }

    if (options.verify_anchor_reachability) {
      const auto* rows_begin = index.row_object_id.data() + row_begin;
      const auto* rows_end = index.row_object_id.data() + row_end;
      const auto* anchor_it = std::lower_bound(rows_begin, rows_end, anchor);
      require(anchor_it != rows_end && *anchor_it == anchor,
              "mandatory anchor row is missing");
      const auto anchor_row =
          row_begin + static_cast<std::uint64_t>(anchor_it - rows_begin);
      std::vector<std::uint8_t> reached(static_cast<std::size_t>(row_end - row_begin),
                                        0);
      std::queue<std::uint64_t> pending;
      reached[static_cast<std::size_t>(anchor_row - row_begin)] = 1;
      pending.push(anchor_row);
      std::size_t reached_count = 1;
      while (!pending.empty()) {
        const auto row = pending.front();
        pending.pop();
        for (auto ref = index.row_adj_ptr[row]; ref < index.row_adj_ptr[row + 1];
             ++ref) {
          const auto target = index.row_adj_target_row[ref];
          const auto local = static_cast<std::size_t>(target - row_begin);
          if (reached[local] == 0) {
            reached[local] = 1;
            ++reached_count;
            pending.push(target);
          }
        }
      }
      require(reached_count == members.size(),
              "mandatory anchor cannot reach every member in its directed view");
    }
  }
}

}  // namespace bounded_blowup

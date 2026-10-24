// SPDX-License-Identifier: MIT
#include "bounded_blowup/routing_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "bounded_blowup/digest.hpp"
#include "bounded_blowup/logical_graph.hpp"

namespace bounded_blowup {

struct RoutingLayoutSnapshot::State {
  std::uint64_t object_count = 0U;
  std::uint32_t vector_dim = 0U;
  std::uint16_t attr_dim = 0U;
  std::uint32_t entries_per_query = 0U;

  std::vector<double> domain_min;
  std::vector<double> domain_max;
  std::vector<double> lattice_unit;

  std::vector<std::uint64_t> family_dim_ptr;
  std::vector<std::uint16_t> family_dims;
  std::vector<std::uint16_t> family_level_count;
  std::vector<std::uint16_t> layout_template_aware;
  std::vector<std::uint64_t> family_template_ptr;
  std::vector<std::uint16_t> template_family;
  std::vector<std::uint16_t> template_local_id;
  std::vector<std::uint64_t> template_ratio_ptr;
  std::vector<double> template_envelope;

  std::vector<std::int64_t> level_service_width_coeff;
  std::vector<std::int64_t> level_side_length_coeff;
  std::vector<std::int64_t> level_stride_coeff;
  std::vector<std::uint64_t> family_level_block_ptr;

  std::vector<std::uint16_t> block_family;
  std::vector<std::uint16_t> block_template;
  std::vector<std::uint16_t> block_level;
  std::vector<std::uint64_t> block_grid_ptr;
  std::vector<std::int64_t> block_grid_coords;
  std::vector<std::int64_t> block_lattice_lo_coeff;
  std::vector<std::int64_t> block_lattice_hi_coeff;
  std::vector<std::uint64_t> sorted_block_lookup;
};

namespace {

using Byte = std::uint8_t;
constexpr Byte kMagic[] = {'R', 'F', 'B', 'B', 'R', 'T', 'L', '1'};

template <typename T>
std::vector<T> copy_array(ArrayView<T> source) {
  if (source.size() != 0U && source.data() == nullptr) {
    throw CorruptIndexError(
        "routing-layout source column has a null data pointer");
  }
  return std::vector<T>(source.begin(), source.end());
}

template <typename T>
ArrayView<T> array_view(const std::vector<T>& values) noexcept {
  return ArrayView<T>(values.data(), values.size());
}

std::uint64_t double_bits(const double value) noexcept {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double canonical_double(const double value, const char* label) {
  if (!std::isfinite(value)) {
    throw CorruptIndexError(std::string(label) + " must be finite");
  }
  // Equality catches both +0 and -0 without depending on the host sign bit.
  return value == 0.0 ? 0.0 : value;
}

void append_u32(std::vector<Byte>* bytes, std::uint32_t value) {
  for (unsigned shift = 0U; shift < 32U; shift += 8U) {
    bytes->push_back(static_cast<Byte>((value >> shift) & 0xffU));
  }
}

void append_u64(std::vector<Byte>* bytes, std::uint64_t value) {
  for (unsigned shift = 0U; shift < 64U; shift += 8U) {
    bytes->push_back(static_cast<Byte>((value >> shift) & UINT64_C(0xff)));
  }
}

void append_i64(std::vector<Byte>* bytes, std::int64_t value) {
  append_u64(bytes, static_cast<std::uint64_t>(value));
}

void append_double(std::vector<Byte>* bytes, double value, const char* label) {
  value = canonical_double(value, label);
  append_u64(bytes, double_bits(value));
}

template <typename T>
void append_vector(std::vector<Byte>* bytes, const std::vector<T>& values) {
  append_u64(bytes, static_cast<std::uint64_t>(values.size()));
  for (const T value : values) {
    if constexpr (std::is_same<T, std::uint16_t>::value) {
      // uint16 columns are still encoded as fixed-width little-endian values.
      bytes->push_back(static_cast<Byte>(value & 0xffU));
      bytes->push_back(static_cast<Byte>(value >> 8U));
    } else if constexpr (std::is_same<T, std::uint32_t>::value) {
      append_u32(bytes, value);
    } else if constexpr (std::is_same<T, std::uint64_t>::value) {
      append_u64(bytes, value);
    } else if constexpr (std::is_same<T, std::int64_t>::value) {
      append_i64(bytes, value);
    } else if constexpr (std::is_same<T, double>::value) {
      append_double(bytes, value, "routing-layout double");
    } else {
      static_assert(!sizeof(T), "unsupported routing-layout column type");
    }
  }
}

class Reader final {
 public:
  explicit Reader(const std::vector<Byte>& bytes) : bytes_(bytes) {}

  Byte byte() {
    require(1U);
    return bytes_[offset_++];
  }

  std::uint16_t u16() {
    require(2U);
    const std::uint16_t value = static_cast<std::uint16_t>(bytes_[offset_]) |
                                static_cast<std::uint16_t>(
                                    bytes_[offset_ + 1U] << 8U);
    offset_ += 2U;
    return value;
  }

  std::uint32_t u32() {
    require(4U);
    const std::uint32_t value = static_cast<std::uint32_t>(bytes_[offset_]) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 1U]) << 8U) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 2U]) << 16U) |
        (static_cast<std::uint32_t>(bytes_[offset_ + 3U]) << 24U);
    offset_ += 4U;
    return value;
  }

  std::uint64_t u64() {
    require(8U);
    std::uint64_t value = 0U;
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
      value |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
    }
    return value;
  }

  std::int64_t i64() { return static_cast<std::int64_t>(u64()); }

  double floating() {
    const std::uint64_t bits = u64();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
      throw CorruptIndexError("routing-layout double is NaN or infinity");
    }
    return value == 0.0 ? 0.0 : value;
  }

  bool at_end() const noexcept { return offset_ == bytes_.size(); }
  std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

 private:
  void require(std::size_t amount) {
    if (offset_ > bytes_.size() || amount > bytes_.size() - offset_) {
      throw CorruptIndexError("routing-layout canonical image is truncated");
    }
  }

  const std::vector<Byte>& bytes_;
  std::size_t offset_ = 0U;
};

template <typename T>
std::vector<T> read_vector(Reader* reader) {
  const std::uint64_t count64 = reader->u64();
  if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw CorruptIndexError("routing-layout column exceeds host size_t");
  }
  const std::size_t count = static_cast<std::size_t>(count64);
  constexpr std::size_t kWireElementBytes =
      std::is_same<T, std::uint16_t>::value ? 2U
      : (std::is_same<T, std::uint32_t>::value ? 4U : 8U);
  if (count > reader->remaining() / kWireElementBytes) {
    throw CorruptIndexError("routing-layout column is truncated");
  }
  std::vector<T> values;
  values.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    if constexpr (std::is_same<T, std::uint16_t>::value) {
      values.push_back(reader->u16());
    } else if constexpr (std::is_same<T, std::uint32_t>::value) {
      values.push_back(reader->u32());
    } else if constexpr (std::is_same<T, std::uint64_t>::value) {
      values.push_back(reader->u64());
    } else if constexpr (std::is_same<T, std::int64_t>::value) {
      values.push_back(reader->i64());
    } else if constexpr (std::is_same<T, double>::value) {
      values.push_back(reader->floating());
    } else {
      static_assert(!sizeof(T), "unsupported routing-layout column type");
    }
  }
  return values;
}

std::size_t checked_size(std::uint64_t value, std::size_t upper_bound,
                         const char* message) {
  if (value > static_cast<std::uint64_t>(upper_bound)) {
    throw CorruptIndexError(message);
  }
  return static_cast<std::size_t>(value);
}

int compare_dimension_ranges(const RoutingLayoutSnapshot::State& route,
                             std::size_t lhs_begin, std::size_t lhs_end,
                             std::size_t rhs_begin, std::size_t rhs_end) {
  const std::size_t lhs_size = lhs_end - lhs_begin;
  const std::size_t rhs_size = rhs_end - rhs_begin;
  const std::size_t common = std::min(lhs_size, rhs_size);
  for (std::size_t i = 0U; i < common; ++i) {
    if (route.family_dims[lhs_begin + i] < route.family_dims[rhs_begin + i]) {
      return -1;
    }
    if (route.family_dims[lhs_begin + i] > route.family_dims[rhs_begin + i]) {
      return 1;
    }
  }
  if (lhs_size < rhs_size) return -1;
  if (lhs_size > rhs_size) return 1;
  return 0;
}

void validate_state(const RoutingLayoutSnapshot::State& route) {
  const std::uint32_t attr_dim = route.attr_dim;
  const std::uint64_t object_count = route.object_count;
  const bool family_count_fits =
      route.family_level_count.size() < std::numeric_limits<std::size_t>::max();
  const bool template_count_fits =
      route.template_family.size() < std::numeric_limits<std::size_t>::max();
  if (object_count == 0U || route.vector_dim == 0U || attr_dim == 0U ||
      route.entries_per_query == 0U ||
      route.domain_min.size() != attr_dim ||
      route.domain_max.size() != attr_dim ||
      (route.lattice_unit.size() != 1U &&
       route.lattice_unit.size() != attr_dim) ||
      route.family_level_count.empty() || !family_count_fits ||
      route.family_dim_ptr.size() != route.family_level_count.size() + 1U ||
      route.family_dim_ptr.empty() || route.family_dim_ptr.front() != 0U ||
      route.family_dim_ptr.back() != route.family_dims.size() ||
      route.family_template_ptr.size() != route.family_level_count.size() + 1U ||
      route.family_template_ptr.empty() || route.family_template_ptr.front() != 0U ||
      route.family_template_ptr.back() != route.template_family.size() ||
      route.template_local_id.size() != route.template_family.size() ||
      !template_count_fits ||
      route.template_ratio_ptr.size() != route.template_family.size() + 1U ||
      route.template_ratio_ptr.empty() || route.template_ratio_ptr.front() != 0U ||
      route.template_ratio_ptr.back() != route.template_envelope.size() ||
      route.layout_template_aware.size() != 1U ||
      route.level_service_width_coeff.empty() ||
      route.level_service_width_coeff.size() != route.level_side_length_coeff.size() ||
      route.level_service_width_coeff.size() != route.level_stride_coeff.size()) {
    throw CorruptIndexError("logical routing snapshot has invalid shape");
  }
  for (std::size_t i = 1U; i < route.family_dim_ptr.size(); ++i) {
    if (route.family_dim_ptr[i - 1U] > route.family_dim_ptr[i] ||
        route.family_dim_ptr[i] > route.family_dims.size()) {
      throw CorruptIndexError("logical routing family pointers are invalid");
    }
  }
  for (std::size_t i = 1U; i < route.family_template_ptr.size(); ++i) {
    if (route.family_template_ptr[i - 1U] > route.family_template_ptr[i] ||
        route.family_template_ptr[i] > route.template_family.size()) {
      throw CorruptIndexError("logical routing family templates are invalid");
    }
  }
  for (std::size_t i = 1U; i < route.template_ratio_ptr.size(); ++i) {
    if (route.template_ratio_ptr[i - 1U] > route.template_ratio_ptr[i] ||
        route.template_ratio_ptr[i] > route.template_envelope.size()) {
      throw CorruptIndexError("logical routing template ratios are invalid");
    }
  }
  if (route.layout_template_aware[0] > 1U) {
    throw CorruptIndexError("logical routing template-aware flag is invalid");
  }
  const std::size_t family_count = route.family_level_count.size();
  if (family_count > std::numeric_limits<std::uint16_t>::max()) {
    throw CorruptIndexError("logical routing family count exceeds uint16");
  }
  std::size_t flattened_levels = 0U;
  std::size_t maximum_level_count = 0U;
  std::size_t previous_dimension_begin = 0U;
  std::size_t previous_dimension_end = 0U;
  for (std::size_t family = 0U; family < family_count; ++family) {
    const std::size_t dim_begin = checked_size(
        route.family_dim_ptr[family], route.family_dims.size(),
        "logical routing family dimension pointer is out of bounds");
    const std::size_t dim_end = checked_size(
        route.family_dim_ptr[family + 1U], route.family_dims.size(),
        "logical routing family dimension pointer is out of bounds");
    if (dim_begin == dim_end) {
      throw CorruptIndexError("logical routing dimension family is empty");
    }
    for (std::size_t i = dim_begin; i < dim_end; ++i) {
      if (route.family_dims[i] >= attr_dim ||
          (i != dim_begin && route.family_dims[i - 1U] >= route.family_dims[i])) {
        throw CorruptIndexError("logical routing family dimensions are invalid");
      }
    }
    if (family != 0U && compare_dimension_ranges(
            route, previous_dimension_begin, previous_dimension_end,
            dim_begin, dim_end) >= 0) {
      throw CorruptIndexError(
          "logical routing dimension families are not canonical");
    }
    previous_dimension_begin = dim_begin;
    previous_dimension_end = dim_end;

    const std::size_t template_begin = checked_size(
        route.family_template_ptr[family], route.template_family.size(),
        "logical routing family template pointer is out of bounds");
    const std::size_t template_end = checked_size(
        route.family_template_ptr[family + 1U], route.template_family.size(),
        "logical routing family template pointer is out of bounds");
    const std::size_t templates = template_end - template_begin;
    if (templates == 0U || templates > std::numeric_limits<std::uint16_t>::max()) {
      throw CorruptIndexError("logical routing family template count is invalid");
    }
    const std::size_t levels = route.family_level_count[family];
    if (levels == 0U || levels > kLogicalMaxDepth ||
        templates > std::numeric_limits<std::size_t>::max() / levels ||
        flattened_levels > std::numeric_limits<std::size_t>::max() - templates * levels) {
      throw CorruptIndexError("logical routing level count overflows");
    }
    flattened_levels += templates * levels;
    maximum_level_count = std::max(maximum_level_count, levels);
  }
  if (route.level_service_width_coeff.size() != maximum_level_count) {
    throw CorruptIndexError(
        "logical routing level coefficients have unexpected length");
  }
  if (flattened_levels == std::numeric_limits<std::size_t>::max() ||
      route.family_level_block_ptr.size() != flattened_levels + 1U ||
      route.family_level_block_ptr.front() != 0U ||
      route.family_level_block_ptr.back() != route.block_family.size()) {
    throw CorruptIndexError("logical routing level block pointers are invalid");
  }
  for (std::size_t i = 1U; i < route.family_level_block_ptr.size(); ++i) {
    if (route.family_level_block_ptr[i - 1U] > route.family_level_block_ptr[i] ||
        route.family_level_block_ptr[i] > route.block_family.size()) {
      throw CorruptIndexError(
          "logical routing level block pointers are not monotone");
    }
  }
  for (std::size_t dimension = 0U; dimension < attr_dim; ++dimension) {
    if (!std::isfinite(route.domain_min[dimension]) ||
        !std::isfinite(route.domain_max[dimension]) ||
        route.domain_min[dimension] > route.domain_max[dimension]) {
      throw CorruptIndexError("logical routing domain bounds are invalid");
    }
    const double unit = route.lattice_unit.size() == 1U
                            ? route.lattice_unit[0]
                            : route.lattice_unit[dimension];
    if (!std::isfinite(unit) || !(unit > 0.0)) {
      throw CorruptIndexError("logical routing lattice unit is invalid");
    }
  }
  for (std::size_t global_template = 0U;
       global_template < route.template_family.size(); ++global_template) {
    const std::uint16_t family = route.template_family[global_template];
    if (family >= family_count) {
      throw CorruptIndexError("logical routing template family is invalid");
    }
    const std::size_t family_template_begin = checked_size(
        route.family_template_ptr[family], route.template_family.size(),
        "logical routing family template pointer is out of bounds");
    const std::size_t family_template_end = checked_size(
        route.family_template_ptr[family + 1U], route.template_family.size(),
        "logical routing family template pointer is out of bounds");
    if (global_template < family_template_begin || global_template >= family_template_end) {
      throw CorruptIndexError("logical routing template partition is invalid");
    }
    const std::size_t local_template = global_template - family_template_begin;
    if (local_template > std::numeric_limits<std::uint16_t>::max() ||
        route.template_local_id[global_template] != static_cast<std::uint16_t>(local_template)) {
      throw CorruptIndexError("logical routing template local ID is invalid");
    }
    const std::size_t ratio_begin = checked_size(
        route.template_ratio_ptr[global_template], route.template_envelope.size(),
        "logical routing template ratio pointer is out of bounds");
    const std::size_t ratio_end = checked_size(
        route.template_ratio_ptr[global_template + 1U], route.template_envelope.size(),
        "logical routing template ratio pointer is out of bounds");
    const std::size_t dim_begin = checked_size(
        route.family_dim_ptr[family], route.family_dims.size(),
        "logical routing family dimension pointer is out of bounds");
    const std::size_t dim_end = checked_size(
        route.family_dim_ptr[family + 1U], route.family_dims.size(),
        "logical routing family dimension pointer is out of bounds");
    if (ratio_end - ratio_begin != dim_end - dim_begin) {
      throw CorruptIndexError("logical routing template rank is invalid");
    }
    for (std::size_t ratio = ratio_begin; ratio < ratio_end; ++ratio) {
      if (!std::isfinite(route.template_envelope[ratio]) ||
          !(route.template_envelope[ratio] > 0.0)) {
        throw CorruptIndexError("logical routing template ratio is invalid");
      }
    }
  }
  const std::size_t block_count = route.block_family.size();
  if (block_count == 0U || block_count == std::numeric_limits<std::size_t>::max() ||
      route.block_template.size() != block_count ||
      route.block_level.size() != block_count ||
      route.block_grid_ptr.size() != block_count + 1U ||
      route.block_grid_ptr.front() != 0U ||
      route.block_grid_ptr.back() != route.block_grid_coords.size() ||
      route.block_lattice_lo_coeff.size() != route.block_grid_coords.size() ||
      route.block_lattice_hi_coeff.size() != route.block_grid_coords.size() ||
      route.sorted_block_lookup.size() != block_count) {
    throw CorruptIndexError("logical routing block geometry is invalid");
  }
  for (std::size_t i = 1U; i < route.block_grid_ptr.size(); ++i) {
    if (route.block_grid_ptr[i - 1U] > route.block_grid_ptr[i] ||
        route.block_grid_ptr[i] > route.block_grid_coords.size()) {
      throw CorruptIndexError("logical routing block-grid pointers are invalid");
    }
  }
  std::vector<std::uint8_t> sorted_seen(block_count, 0U);
  for (const std::uint64_t block : route.sorted_block_lookup) {
    if (block >= block_count || sorted_seen[static_cast<std::size_t>(block)] != 0U) {
      throw CorruptIndexError("logical routing sorted lookup is out of range");
    }
    sorted_seen[static_cast<std::size_t>(block)] = 1U;
  }
  for (const std::uint8_t seen : sorted_seen) {
    if (seen == 0U) {
      throw CorruptIndexError("logical routing sorted lookup is not a permutation");
    }
  }
  for (std::size_t block = 0U; block < block_count; ++block) {
    const std::uint16_t family = route.block_family[block];
    const std::uint16_t template_id = route.block_template[block];
    const std::uint16_t level = route.block_level[block];
    if (family >= family_count) {
      throw CorruptIndexError("logical routing block family is invalid");
    }
    const std::uint64_t template_count =
        route.family_template_ptr[family + 1U] - route.family_template_ptr[family];
    if (template_id >= template_count || level > kLogicalMaxDepth ||
        level >= route.family_level_count[family]) {
      throw CorruptIndexError("logical routing block identity is invalid");
    }
    const std::size_t grid_begin = static_cast<std::size_t>(route.block_grid_ptr[block]);
    const std::size_t grid_end = static_cast<std::size_t>(route.block_grid_ptr[block + 1U]);
    if (grid_end - grid_begin != route.family_dim_ptr[family + 1U] - route.family_dim_ptr[family]) {
      throw CorruptIndexError("logical routing block grid rank is invalid");
    }
    const std::int64_t stride = route.level_stride_coeff[level];
    const std::int64_t side = route.level_side_length_coeff[level];
    for (std::size_t dimension = 0U; dimension < grid_end - grid_begin; ++dimension) {
      const std::int64_t grid = route.block_grid_coords[grid_begin + dimension];
      const __int128 expected_lo = static_cast<__int128>(grid) * stride;
      const __int128 expected_hi = expected_lo + side;
      if (expected_lo < std::numeric_limits<std::int64_t>::min() ||
          expected_lo > std::numeric_limits<std::int64_t>::max() ||
          expected_hi < std::numeric_limits<std::int64_t>::min() ||
          expected_hi > std::numeric_limits<std::int64_t>::max() ||
          route.block_lattice_lo_coeff[grid_begin + dimension] != static_cast<std::int64_t>(expected_lo) ||
          route.block_lattice_hi_coeff[grid_begin + dimension] != static_cast<std::int64_t>(expected_hi)) {
        throw CorruptIndexError("logical routing block lattice bounds are not canonical");
      }
    }
  }
  const auto compare_grid = [&](std::size_t lhs, std::size_t rhs) {
    const std::size_t lhs_begin = static_cast<std::size_t>(route.block_grid_ptr[lhs]);
    const std::size_t rhs_begin = static_cast<std::size_t>(route.block_grid_ptr[rhs]);
    const std::size_t rank = static_cast<std::size_t>(route.block_grid_ptr[lhs + 1U] - route.block_grid_ptr[lhs]);
    for (std::size_t dimension = 0U; dimension < rank; ++dimension) {
      if (route.block_grid_coords[lhs_begin + dimension] < route.block_grid_coords[rhs_begin + dimension]) return -1;
      if (route.block_grid_coords[lhs_begin + dimension] > route.block_grid_coords[rhs_begin + dimension]) return 1;
    }
    return 0;
  };
  std::size_t flattened_level = 0U;
  for (std::size_t family = 0U; family < family_count; ++family) {
    const std::size_t template_begin = checked_size(
        route.family_template_ptr[family], route.template_family.size(),
        "logical routing family template pointer is out of bounds");
    const std::size_t template_end = checked_size(
        route.family_template_ptr[family + 1U], route.template_family.size(),
        "logical routing family template pointer is out of bounds");
    for (std::size_t local_template = 0U; local_template < template_end - template_begin; ++local_template) {
      for (std::size_t level = 0U; level < route.family_level_count[family]; ++level) {
        const std::size_t section_begin = checked_size(
            route.family_level_block_ptr[flattened_level], block_count,
            "logical routing level block pointer is out of bounds");
        const std::size_t section_end = checked_size(
            route.family_level_block_ptr[flattened_level + 1U], block_count,
            "logical routing level block pointer is out of bounds");
        for (std::size_t position = section_begin; position < section_end; ++position) {
          const std::size_t block = static_cast<std::size_t>(route.sorted_block_lookup[position]);
          if (route.block_family[block] != family || route.block_template[block] != local_template || route.block_level[block] != level) {
            throw CorruptIndexError("logical routing sorted lookup has the wrong partition");
          }
          if (position != section_begin && compare_grid(static_cast<std::size_t>(route.sorted_block_lookup[position - 1U]), block) >= 0) {
            throw CorruptIndexError("logical routing sorted lookup is not canonical");
          }
        }
        ++flattened_level;
      }
    }
  }
  if (flattened_level != flattened_levels) {
    throw CorruptIndexError("logical routing level partition shape is invalid");
  }
  for (std::size_t level = 0U; level < route.level_service_width_coeff.size(); ++level) {
    if (route.level_service_width_coeff[level] <= 0 ||
        route.level_side_length_coeff[level] <= 0 ||
        route.level_stride_coeff[level] <= 0) {
      throw CorruptIndexError("logical routing level coefficients are invalid");
    }
  }
}

std::vector<Byte> encode_state(const RoutingLayoutSnapshot::State& route) {
  std::vector<Byte> bytes;
  bytes.insert(bytes.end(), std::begin(kMagic), std::end(kMagic));
  append_u32(&bytes, kRoutingLayoutSchemaVersion);
  append_u64(&bytes, static_cast<std::uint64_t>(route.block_family.size()));
  append_u64(&bytes, route.object_count);
  append_u32(&bytes, route.vector_dim);
  bytes.push_back(0U);
  bytes.push_back(static_cast<Byte>(route.attr_dim & 0xffU));
  bytes.push_back(static_cast<Byte>(route.attr_dim >> 8U));
  append_u32(&bytes, route.entries_per_query);
  append_vector(&bytes, route.domain_min);
  append_vector(&bytes, route.domain_max);
  append_vector(&bytes, route.lattice_unit);
  append_vector(&bytes, route.family_dim_ptr);
  append_vector(&bytes, route.family_dims);
  append_vector(&bytes, route.family_level_count);
  append_vector(&bytes, route.layout_template_aware);
  append_vector(&bytes, route.family_template_ptr);
  append_vector(&bytes, route.template_family);
  append_vector(&bytes, route.template_local_id);
  append_vector(&bytes, route.template_ratio_ptr);
  append_vector(&bytes, route.template_envelope);
  append_vector(&bytes, route.level_service_width_coeff);
  append_vector(&bytes, route.level_side_length_coeff);
  append_vector(&bytes, route.level_stride_coeff);
  append_vector(&bytes, route.family_level_block_ptr);
  append_vector(&bytes, route.block_family);
  append_vector(&bytes, route.block_template);
  append_vector(&bytes, route.block_level);
  append_vector(&bytes, route.block_grid_ptr);
  append_vector(&bytes, route.block_grid_coords);
  append_vector(&bytes, route.block_lattice_lo_coeff);
  append_vector(&bytes, route.block_lattice_hi_coeff);
  append_vector(&bytes, route.sorted_block_lookup);
  return bytes;
}

std::shared_ptr<RoutingLayoutSnapshot::State> decode_state(
    const std::vector<Byte>& bytes) {
  Reader reader(bytes);
  for (const Byte expected : kMagic) {
    if (reader.byte() != expected) {
      throw CorruptIndexError("routing-layout canonical magic is invalid");
    }
  }
  if (reader.u32() != kRoutingLayoutSchemaVersion) {
    throw CorruptIndexError("routing-layout schema version is unsupported");
  }
  const std::uint64_t block_count = reader.u64();
  const std::uint64_t object_count = reader.u64();
  const std::uint32_t vector_dim = reader.u32();
  if (reader.byte() != 0U) {
    throw CorruptIndexError("routing-layout reserved header byte is nonzero");
  }
  const std::uint16_t attr_dim = reader.u16();
  const std::uint32_t entries_per_query = reader.u32();
  auto state = std::make_shared<RoutingLayoutSnapshot::State>();
  state->object_count = object_count;
  state->vector_dim = vector_dim;
  state->attr_dim = attr_dim;
  state->entries_per_query = entries_per_query;
  state->domain_min = read_vector<double>(&reader);
  state->domain_max = read_vector<double>(&reader);
  state->lattice_unit = read_vector<double>(&reader);
  state->family_dim_ptr = read_vector<std::uint64_t>(&reader);
  state->family_dims = read_vector<std::uint16_t>(&reader);
  state->family_level_count = read_vector<std::uint16_t>(&reader);
  state->layout_template_aware = read_vector<std::uint16_t>(&reader);
  state->family_template_ptr = read_vector<std::uint64_t>(&reader);
  state->template_family = read_vector<std::uint16_t>(&reader);
  state->template_local_id = read_vector<std::uint16_t>(&reader);
  state->template_ratio_ptr = read_vector<std::uint64_t>(&reader);
  state->template_envelope = read_vector<double>(&reader);
  state->level_service_width_coeff = read_vector<std::int64_t>(&reader);
  state->level_side_length_coeff = read_vector<std::int64_t>(&reader);
  state->level_stride_coeff = read_vector<std::int64_t>(&reader);
  state->family_level_block_ptr = read_vector<std::uint64_t>(&reader);
  state->block_family = read_vector<std::uint16_t>(&reader);
  state->block_template = read_vector<std::uint16_t>(&reader);
  state->block_level = read_vector<std::uint16_t>(&reader);
  state->block_grid_ptr = read_vector<std::uint64_t>(&reader);
  state->block_grid_coords = read_vector<std::int64_t>(&reader);
  state->block_lattice_lo_coeff = read_vector<std::int64_t>(&reader);
  state->block_lattice_hi_coeff = read_vector<std::int64_t>(&reader);
  state->sorted_block_lookup = read_vector<std::uint64_t>(&reader);
  if (!reader.at_end()) {
    throw CorruptIndexError("routing-layout canonical image has trailing data");
  }
  if (block_count != state->block_family.size()) {
    throw CorruptIndexError("routing-layout block count does not match columns");
  }
  validate_state(*state);
  if (encode_state(*state) != bytes) {
    throw CorruptIndexError("routing-layout image is not canonical");
  }
  return state;
}

}  // namespace

RoutingLayoutSnapshot::RoutingLayoutSnapshot(
    std::shared_ptr<const State> state)
    : state_(std::move(state)) {
  if (!state_) throw std::invalid_argument("routing-layout state is null");
}

RoutingLayoutSnapshot::RoutingLayoutSnapshot(
    const RoutingLayoutSnapshot& other)
    : state_(std::make_shared<const State>(*other.state_)) {}

RoutingLayoutSnapshot& RoutingLayoutSnapshot::operator=(
    const RoutingLayoutSnapshot& other) {
  if (this != &other) state_ = std::make_shared<const State>(*other.state_);
  return *this;
}

RoutingLayoutSnapshot::RoutingLayoutSnapshot(
    RoutingLayoutSnapshot&& other) noexcept = default;
RoutingLayoutSnapshot& RoutingLayoutSnapshot::operator=(
    RoutingLayoutSnapshot&& other) noexcept = default;
RoutingLayoutSnapshot::~RoutingLayoutSnapshot() = default;

RoutingLayoutSnapshot RoutingLayoutSnapshot::from_index_view(
    const IndexView& index) {
  auto state = std::make_shared<State>();
  state->object_count = index.object_count;
  state->vector_dim = index.vector_dim;
  state->attr_dim = index.attr_dim;
  state->entries_per_query = index.entries_per_query;
  state->domain_min = copy_array(index.domain_min);
  state->domain_max = copy_array(index.domain_max);
  state->lattice_unit = copy_array(index.lattice_unit);
  state->family_dim_ptr = copy_array(index.family_dim_ptr);
  state->family_dims = copy_array(index.family_dims);
  state->family_level_count = copy_array(index.family_level_count);
  state->layout_template_aware = copy_array(index.layout_template_aware);
  state->family_template_ptr = copy_array(index.family_template_ptr);
  state->template_family = copy_array(index.template_family);
  state->template_local_id = copy_array(index.template_local_id);
  state->template_ratio_ptr = copy_array(index.template_ratio_ptr);
  state->template_envelope = copy_array(index.template_envelope);
  state->level_service_width_coeff = copy_array(index.level_service_width_coeff);
  state->level_side_length_coeff = copy_array(index.level_side_length_coeff);
  state->level_stride_coeff = copy_array(index.level_stride_coeff);
  state->family_level_block_ptr = copy_array(index.family_level_block_ptr);
  state->block_family = copy_array(index.block_family);
  state->block_template = copy_array(index.block_template);
  state->block_level = copy_array(index.block_level);
  state->block_grid_ptr = copy_array(index.block_grid_ptr);
  state->block_grid_coords = copy_array(index.block_grid_coords);
  state->block_lattice_lo_coeff = copy_array(index.block_lattice_lo_coeff);
  state->block_lattice_hi_coeff = copy_array(index.block_lattice_hi_coeff);
  state->sorted_block_lookup = copy_array(index.sorted_block_lookup);
  try {
    validate_state(*state);
  } catch (const CorruptIndexError&) {
    throw;
  } catch (const std::exception& error) {
    throw CorruptIndexError(error.what());
  }
  return RoutingLayoutSnapshot(std::move(state));
}

RoutingLayoutSnapshot RoutingLayoutSnapshot::from_canonical_bytes(
    std::vector<std::uint8_t> canonical_bytes) {
  return RoutingLayoutSnapshot(decode_state(canonical_bytes));
}

std::vector<std::uint8_t> RoutingLayoutSnapshot::canonical_bytes() const {
  return encode_state(*state_);
}

RoutingLayoutDigest RoutingLayoutSnapshot::digest() const noexcept {
  const std::vector<Byte> bytes = encode_state(*state_);
  return sha256_bytes(bytes.data(), bytes.size()).bytes();
}

std::size_t RoutingLayoutSnapshot::block_count() const noexcept {
  return state_->block_family.size();
}

std::uint16_t RoutingLayoutSnapshot::block_level(std::size_t block_id) const {
  if (block_id >= state_->block_level.size()) {
    throw CorruptIndexError("routing-layout block ID is outside layout");
  }
  return state_->block_level[block_id];
}

std::uint64_t RoutingLayoutSnapshot::object_count() const noexcept {
  return state_->object_count;
}

std::uint32_t RoutingLayoutSnapshot::vector_dim() const noexcept {
  return state_->vector_dim;
}

std::uint16_t RoutingLayoutSnapshot::attr_dim() const noexcept {
  return state_->attr_dim;
}

std::uint32_t RoutingLayoutSnapshot::entries_per_query() const noexcept {
  return state_->entries_per_query;
}

IndexView RoutingLayoutSnapshot::materialize() const noexcept {
  IndexView index;
  index.object_count = state_->object_count;
  index.vector_dim = state_->vector_dim;
  index.attr_dim = state_->attr_dim;
  index.entries_per_query = state_->entries_per_query;
  index.max_degree = kLogicalMaxDegree;
  index.domain_min = array_view(state_->domain_min);
  index.domain_max = array_view(state_->domain_max);
  index.lattice_unit = array_view(state_->lattice_unit);
  index.family_dim_ptr = array_view(state_->family_dim_ptr);
  index.family_dims = array_view(state_->family_dims);
  index.family_level_count = array_view(state_->family_level_count);
  index.layout_template_aware = array_view(state_->layout_template_aware);
  index.family_template_ptr = array_view(state_->family_template_ptr);
  index.template_family = array_view(state_->template_family);
  index.template_local_id = array_view(state_->template_local_id);
  index.template_ratio_ptr = array_view(state_->template_ratio_ptr);
  index.template_envelope = array_view(state_->template_envelope);
  index.level_service_width_coeff = array_view(state_->level_service_width_coeff);
  index.level_side_length_coeff = array_view(state_->level_side_length_coeff);
  index.level_stride_coeff = array_view(state_->level_stride_coeff);
  index.family_level_block_ptr = array_view(state_->family_level_block_ptr);
  index.block_family = array_view(state_->block_family);
  index.block_template = array_view(state_->block_template);
  index.block_level = array_view(state_->block_level);
  index.block_grid_ptr = array_view(state_->block_grid_ptr);
  index.block_grid_coords = array_view(state_->block_grid_coords);
  index.block_lattice_lo_coeff = array_view(state_->block_lattice_lo_coeff);
  index.block_lattice_hi_coeff = array_view(state_->block_lattice_hi_coeff);
  index.sorted_block_lookup = array_view(state_->sorted_block_lookup);
  return index;
}

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::vector<std::uint8_t> routing_layout_canonical_bytes_for_testing(
    const RoutingLayoutSnapshot& snapshot) {
  return snapshot.canonical_bytes();
}

RoutingLayoutSnapshot routing_layout_from_canonical_bytes_for_testing(
    std::vector<std::uint8_t> canonical_bytes) {
  return RoutingLayoutSnapshot::from_canonical_bytes(std::move(canonical_bytes));
}

RoutingLayoutDigest routing_layout_sha256_for_testing(
    const RoutingLayoutSnapshot& snapshot) noexcept {
  return snapshot.digest();
}
#endif

}  // namespace bounded_blowup

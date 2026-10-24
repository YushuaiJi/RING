// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#include "bounded_blowup/router.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace bounded_blowup {
namespace {

constexpr std::uint64_t kMissingBlock =
    std::numeric_limits<std::uint64_t>::max();

std::uint16_t checked_template_id(std::size_t value) {
  if (value > std::numeric_limits<std::uint16_t>::max()) {
    throw CorruptIndexError("template ID exceeds uint16 sentinel");
  }
  return static_cast<std::uint16_t>(value);
}

int compare_dimensions(ArrayView<std::uint16_t> lhs,
                       ArrayView<std::uint16_t> rhs) {
  const auto common = std::min(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < common; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  if (lhs.size() < rhs.size()) return -1;
  if (lhs.size() > rhs.size()) return 1;
  return 0;
}

int compare_grid(ArrayView<std::int64_t> lhs,
                 ArrayView<std::int64_t> rhs) {
  const auto common = std::min(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < common; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  if (lhs.size() < rhs.size()) return -1;
  if (lhs.size() > rhs.size()) return 1;
  return 0;
}

double lattice_unit_for(const IndexView& index, std::uint16_t dimension) {
  return index.lattice_unit.size() == 1 ? index.lattice_unit[0]
                                         : index.lattice_unit[dimension];
}

double shape_distance(ArrayView<double> left, const std::vector<double>& right) {
  if (left.size() != right.size() || left.empty()) {
    throw CorruptIndexError("template/query shape ranks disagree");
  }
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < left.size(); ++i) {
    if (!(left[i] > 0.0) || !(right[i] > 0.0)) {
      throw std::invalid_argument("template-aware routing requires positive widths");
    }
    const double value = std::log(right[i] / left[i]);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return maximum - minimum;
}

std::int64_t checked_floor_to_i64(double value) {
  const long double wide = static_cast<long double>(value);
  if (!std::isfinite(value) ||
      wide < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
      wide > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("routing coordinate is outside int64 range");
  }
  return static_cast<std::int64_t>(std::floor(value));
}

std::int64_t checked_add(std::int64_t value, int delta) {
  if (delta < 0 && value == std::numeric_limits<std::int64_t>::min()) {
    return value;
  }
  if (delta > 0 && value == std::numeric_limits<std::int64_t>::max()) {
    return value;
  }
  return value + delta;
}

double coeff_to_bound(std::int64_t coefficient, double unit) {
  const long double bound = static_cast<long double>(coefficient) *
                            static_cast<long double>(unit);
  if (!std::isfinite(static_cast<double>(bound))) {
    throw NoContainingBlockError("lattice bound overflows float64");
  }
  return static_cast<double>(bound);
}

}  // namespace

std::uint16_t lookup_exact_dimension_family(
    const IndexView& index, ArrayView<std::uint16_t> dimensions) {
  std::size_t lo = 0;
  std::size_t hi = index.family_count();
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    const int comparison = compare_dimensions(
        index.dimensions(static_cast<std::uint16_t>(mid)), dimensions);
    if (comparison < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo >= index.family_count() ||
      compare_dimensions(index.dimensions(static_cast<std::uint16_t>(lo)),
                         dimensions) != 0) {
    throw UnsupportedDimensionSetError(
        "query dimension set was not frozen in this index artifact");
  }
  return static_cast<std::uint16_t>(lo);
}

std::uint64_t lookup_materialized_block(
    const IndexView& index, std::uint16_t family_id,
    std::uint16_t template_id, std::uint16_t level,
    ArrayView<std::int64_t> grid_coords) {
  if (family_id >= index.family_count() ||
      template_id >= index.template_count(family_id) ||
      level >= index.family_level_count[family_id]) {
    throw CorruptIndexError("family-template-level lookup is out of range");
  }
  const auto flat_level =
      index.flattened_level_offset(family_id, template_id) + level;
  std::uint64_t lo = index.family_level_block_ptr[flat_level];
  std::uint64_t hi = index.family_level_block_ptr[flat_level + 1];
  while (lo < hi) {
    const auto mid = lo + (hi - lo) / 2;
    const auto block_id = index.sorted_block_lookup[mid];
    const int comparison = compare_grid(index.grid_coords(block_id), grid_coords);
    if (comparison < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const auto end = index.family_level_block_ptr[flat_level + 1];
  if (lo >= end) return kMissingBlock;
  const auto block_id = index.sorted_block_lookup[lo];
  if (compare_grid(index.grid_coords(block_id), grid_coords) != 0) {
    return kMissingBlock;
  }
  if (index.block_family[block_id] != family_id ||
      index.block_template[block_id] != template_id ||
      index.block_level[block_id] != level) {
    throw CorruptIndexError(
        "sorted block lookup crosses family-template-level range");
  }
  return block_id;
}

RoutingResult route_first_actual_containing_block(const IndexView& index,
                                                  const QueryWindow& window) {
  if (window.dimensions.empty()) {
    throw std::invalid_argument("query dimension set cannot be empty");
  }
  if (window.dimensions.size() != window.lower_closed.size() ||
      window.dimensions.size() != window.upper_closed.size()) {
    throw std::invalid_argument("window dimension/lower/upper lengths differ");
  }
  for (std::size_t i = 0; i < window.dimensions.size(); ++i) {
    if (window.dimensions[i] >= index.attr_dim) {
      throw std::invalid_argument("query dimension is out of range");
    }
    if (i > 0 && window.dimensions[i - 1] >= window.dimensions[i]) {
      throw std::invalid_argument(
          "query dimensions must be strictly increasing and unique");
    }
    if (!std::isfinite(window.lower_closed[i]) ||
        !std::isfinite(window.upper_closed[i])) {
      throw std::invalid_argument("query bounds must be finite");
    }
    if (window.lower_closed[i] > window.upper_closed[i]) {
      throw std::invalid_argument("query lower bound exceeds upper bound");
    }
  }

  const ArrayView<std::uint16_t> query_dims(window.dimensions.data(),
                                            window.dimensions.size());
  RoutingResult result;
  result.family_id = lookup_exact_dimension_family(index, query_dims);
  result.normalized_lower.resize(window.dimensions.size());
  result.normalized_upper.resize(window.dimensions.size());

  for (std::size_t i = 0; i < window.dimensions.size(); ++i) {
    const auto dim = window.dimensions[i];
    const double domain_lo = index.domain_min[dim];
    const double domain_hi = index.domain_max[dim];
    const double query_lo = window.lower_closed[i];
    const double query_hi = window.upper_closed[i];
    if (query_hi < domain_lo || query_lo > domain_hi) {
      result.outcome = RouteOutcome::kOutOfDomain;
      return result;
    }
    const double clipped_lo = std::max(query_lo, domain_lo);
    const double clipped_hi = std::min(query_hi, domain_hi);
    result.routing_clipped = result.routing_clipped || clipped_lo != query_lo ||
                             clipped_hi != query_hi;
    if (domain_lo == domain_hi) {
      // Intersection above proves the original closed interval contains the
      // corpus constant; its normalized coordinate is exactly zero.
      result.normalized_lower[i] = 0.0;
      result.normalized_upper[i] = 0.0;
    } else {
      const double scale = domain_hi - domain_lo;
      result.normalized_lower[i] = (clipped_lo - domain_lo) / scale;
      result.normalized_upper[i] = (clipped_hi - domain_lo) / scale;
    }
  }

  const auto level_count = index.family_level_count[result.family_id];
  const auto template_count = index.template_count(result.family_id);
  if (template_count == 0U ||
      template_count > std::numeric_limits<std::uint16_t>::max()) {
    throw CorruptIndexError(
        "template count exceeds uint16 sentinel in routing");
  }
  std::vector<double> query_widths(window.dimensions.size());
  double maximum_width = 0.0;
  for (std::size_t i = 0; i < window.dimensions.size(); ++i) {
    query_widths[i] = result.normalized_upper[i] - result.normalized_lower[i];
    maximum_width = std::max(maximum_width, query_widths[i]);
  }
  const bool template_aware = index.layout_template_aware[0] != 0;
  std::uint16_t selected_template = 0;
  double selected_tau = 0.0;
  if (template_aware) {
    if (std::any_of(query_widths.begin(), query_widths.end(),
                    [](double value) { return value < 0.0; })) {
      throw std::invalid_argument("query widths must be non-negative");
    }
    // Equality predicates have a legitimate zero-width coordinate, but the
    // log-ratio shape distance is defined only for positive values.  Replace
    // zeros by a fixed limiting proxy solely while choosing a template.  The
    // actual zero widths below still determine scale, containment and strict
    // filtering.  An all-equality query uses the balanced limiting shape.
    if (maximum_width > 0.0) {
      constexpr double kZeroWidthShapeFloor = 0x1p-24;
      for (double& value : query_widths) {
        value = value > 0.0 ? value / maximum_width : kZeroWidthShapeFloor;
      }
    } else {
      std::fill(query_widths.begin(), query_widths.end(), 1.0);
    }
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t candidate = 0U; candidate < template_count; ++candidate) {
      const std::uint16_t candidate_id = checked_template_id(candidate);
      const double distance =
          shape_distance(index.template_ratio(result.family_id, candidate_id),
                         query_widths);
      if (distance < best) {
        best = distance;
        selected_template = candidate_id;
      }
    }
    selected_tau = best;
  }
  result.template_id = selected_template;
  const auto ratio_view =
      index.template_ratio(result.family_id, selected_template);
  result.template_ratio.assign(ratio_view.begin(), ratio_view.end());
  result.shape_tau = selected_tau;
  result.required_scale = 0.0;
  for (std::size_t i = 0; i < window.dimensions.size(); ++i) {
    const double width =
        result.normalized_upper[i] - result.normalized_lower[i];
    result.required_scale =
        std::max(result.required_scale, width / ratio_view[i]);
  }
  std::uint16_t first_level = 0;
  if (template_aware) {
    const double base_unit = index.lattice_unit[0];
    while (first_level < level_count &&
           static_cast<double>(index.level_service_width_coeff[first_level]) *
                   base_unit <
               result.required_scale) {
      ++first_level;
    }
    if (first_level == level_count) {
      throw NoContainingBlockError(
          "template scale coverage invariant was not materialized");
    }
  }

  for (std::uint16_t level = first_level; level < level_count; ++level) {
    const auto stride_coeff = index.level_stride_coeff[level];
    const auto side_coeff = index.level_side_length_coeff[level];
    std::vector<std::int64_t> selected_grid(window.dimensions.size());
    std::vector<double> selected_lower(window.dimensions.size());
    std::vector<double> selected_upper(window.dimensions.size());
    bool contains_all_dimensions = true;

    for (std::size_t i = 0; i < window.dimensions.size(); ++i) {
      const auto dim = window.dimensions[i];
      const double unit = lattice_unit_for(index, dim) * ratio_view[i];
      const double stride = static_cast<double>(stride_coeff) * unit;
      const auto base_grid =
          checked_floor_to_i64(result.normalized_lower[i] / stride);
      const std::array<std::int64_t, 3> candidates = {
          checked_add(base_grid, -1), base_grid, checked_add(base_grid, 1)};

      bool found_start = false;
      std::int64_t rightmost_grid = 0;
      double rightmost_start = -std::numeric_limits<double>::infinity();
      double rightmost_end = -std::numeric_limits<double>::infinity();
      for (const auto grid : candidates) {
        const __int128 lo_coeff = static_cast<__int128>(grid) * stride_coeff;
        const __int128 hi_coeff = lo_coeff + side_coeff;
        if (lo_coeff < std::numeric_limits<std::int64_t>::min() ||
            lo_coeff > std::numeric_limits<std::int64_t>::max() ||
            hi_coeff < std::numeric_limits<std::int64_t>::min() ||
            hi_coeff > std::numeric_limits<std::int64_t>::max()) {
          continue;
        }
        const double start =
            coeff_to_bound(static_cast<std::int64_t>(lo_coeff), unit);
        const double end =
            coeff_to_bound(static_cast<std::int64_t>(hi_coeff), unit);
        if (start <= result.normalized_lower[i] &&
            (!found_start || start > rightmost_start ||
             (start == rightmost_start && grid > rightmost_grid))) {
          found_start = true;
          rightmost_grid = grid;
          rightmost_start = start;
          rightmost_end = end;
        }
      }
      if (!found_start || rightmost_end < result.normalized_upper[i]) {
        contains_all_dimensions = false;
        break;
      }
      selected_grid[i] = rightmost_grid;
      selected_lower[i] = rightmost_start;
      selected_upper[i] = rightmost_end;
    }

    if (!contains_all_dimensions) continue;

    result.level = level;
    result.selected_scale =
        static_cast<double>(index.level_service_width_coeff[level]) *
        index.lattice_unit[0];
    result.grid_coords = std::move(selected_grid);
    result.normalized_block_lower = std::move(selected_lower);
    result.normalized_block_upper = std::move(selected_upper);
    const ArrayView<std::int64_t> grid_view(result.grid_coords.data(),
                                            result.grid_coords.size());
    result.block_id = lookup_materialized_block(
        index, result.family_id, selected_template, level, grid_view);
    result.outcome = result.block_id == kMissingBlock
                         ? RouteOutcome::kEmptyTheoreticalBlock
                         : RouteOutcome::kMaterializedBlock;
    return result;
  }

  throw NoContainingBlockError(
      "validated artifact has no level that actually contains the query window");
}

}  // namespace bounded_blowup

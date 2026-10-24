// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bounded_blowup/index_view.hpp"

namespace bounded_blowup {

struct QueryWindow {
  std::vector<std::uint16_t> dimensions;
  std::vector<double> lower_closed;
  std::vector<double> upper_closed;
};

enum class RouteOutcome : std::uint8_t {
  kMaterializedBlock = 0,
  kEmptyTheoreticalBlock = 1,
  kOutOfDomain = 2,
};

struct RoutingResult {
  RouteOutcome outcome = RouteOutcome::kOutOfDomain;
  std::uint16_t family_id = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t template_id = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t level = std::numeric_limits<std::uint16_t>::max();
  std::uint64_t block_id = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::int64_t> grid_coords;
  std::vector<double> normalized_lower;
  std::vector<double> normalized_upper;
  std::vector<double> normalized_block_lower;
  std::vector<double> normalized_block_upper;
  std::vector<double> template_ratio;
  double shape_tau = std::numeric_limits<double>::quiet_NaN();
  double required_scale = std::numeric_limits<double>::quiet_NaN();
  double selected_scale = std::numeric_limits<double>::quiet_NaN();
  bool routing_clipped = false;
};

class UnsupportedDimensionSetError : public std::invalid_argument {
 public:
  explicit UnsupportedDimensionSetError(const std::string& message)
      : std::invalid_argument(message) {}
};

class NoContainingBlockError : public std::runtime_error {
 public:
  explicit NoContainingBlockError(const std::string& message)
      : std::runtime_error(message) {}
};

std::uint16_t lookup_exact_dimension_family(
    const IndexView& index, ArrayView<std::uint16_t> dimensions);

std::uint64_t lookup_materialized_block(
    const IndexView& index, std::uint16_t family_id,
    std::uint16_t template_id, std::uint16_t level,
    ArrayView<std::int64_t> grid_coords);

inline std::uint64_t lookup_materialized_block(
    const IndexView& index, std::uint16_t family_id, std::uint16_t level,
    ArrayView<std::int64_t> grid_coords) {
  return lookup_materialized_block(index, family_id, 0, level, grid_coords);
}

RoutingResult route_first_actual_containing_block(const IndexView& index,
                                                  const QueryWindow& window);

}  // namespace bounded_blowup

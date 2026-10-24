// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace bounded_blowup {

enum class TerminationReason : std::uint8_t {
  kEmptyBlock = 0,
  kOutOfDomain = 1,
  kBudgetExhausted = 2,
  kFrontierExhausted = 3,
};

inline const char* termination_reason_name(TerminationReason reason) noexcept {
  switch (reason) {
    case TerminationReason::kEmptyBlock:
      return "empty_block";
    case TerminationReason::kOutOfDomain:
      return "out_of_domain";
    case TerminationReason::kBudgetExhausted:
      return "budget_exhausted";
    case TerminationReason::kFrontierExhausted:
      return "frontier_exhausted";
  }
  return "unknown";
}

struct SearchHit {
  std::int64_t object_id = -1;
  float distance = std::numeric_limits<float>::infinity();
};

struct QueryStats {
  std::uint16_t family_id = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t template_id = std::numeric_limits<std::uint16_t>::max();
  std::uint64_t block_id = std::numeric_limits<std::uint64_t>::max();
  std::uint16_t level = std::numeric_limits<std::uint16_t>::max();
  std::vector<std::int64_t> block_grid_coords;
  std::vector<double> normalized_query_widths;
  std::vector<double> normalized_block_widths;
  std::vector<double> side_blowups;
  std::vector<double> template_ratio;
  double shape_tau = std::numeric_limits<double>::quiet_NaN();
  double selected_scale = std::numeric_limits<double>::quiet_NaN();

  bool routing_clipped = false;
  std::uint64_t block_object_count = 0;
  std::uint32_t num_graphs_searched = 0;
  std::uint64_t requested_budget = 0;
  std::uint64_t actual_budget = 0;

  std::uint64_t entry_pool_considered = 0;
  std::uint64_t entries_selected = 0;
  std::uint64_t planning_distance_evaluations = 0;
  std::uint64_t traversal_distance_evaluations = 0;
  std::uint64_t distance_evaluations = 0;
  std::uint64_t candidate_pushes = 0;
  std::uint64_t candidate_pops = 0;
  std::uint64_t post_filter_checks = 0;
  std::uint64_t post_filter_dimension_checks = 0;
  std::uint64_t qualifying_expansions = 0;
  std::uint64_t result_updates = 0;

  TerminationReason termination_reason = TerminationReason::kFrontierExhausted;
  bool fallback = false;

  std::string algorithm_id;
  std::string algorithm_revision;
  std::string config_sha256;
  std::string artifact_sha256;
  std::string corpus_vectors_sha256;
  std::string corpus_attrs_sha256;
  std::string cpp_source_tree_sha256;
  std::string cpp_binary_sha256;
};

struct SearchResult {
  std::vector<SearchHit> hits;
  QueryStats stats;
};

}  // namespace bounded_blowup

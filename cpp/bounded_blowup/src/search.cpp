// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#include "bounded_blowup/search.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace bounded_blowup {
namespace {

bool hit_better(const SearchHit& lhs, const SearchHit& rhs) {
  if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
  return lhs.object_id < rhs.object_id;
}

void copy_identity(const IndexIdentity& identity, QueryStats* stats) {
  stats->algorithm_id = identity.algorithm_id;
  stats->algorithm_revision = identity.algorithm_revision;
  stats->config_sha256 = identity.config_sha256;
  stats->artifact_sha256 = identity.artifact_sha256;
  stats->corpus_vectors_sha256 = identity.corpus_vectors_sha256;
  stats->corpus_attrs_sha256 = identity.corpus_attrs_sha256;
  stats->cpp_source_tree_sha256 = identity.cpp_source_tree_sha256;
  stats->cpp_binary_sha256 = identity.cpp_binary_sha256;
}

}  // namespace

void QueryWorkspace::begin_query(std::size_t local_row_count) {
  if (seen_epoch_.size() < local_row_count) {
    seen_epoch_.resize(local_row_count, 0);
    planned_distance_epoch_.resize(local_row_count, 0);
    planned_distance_cache_.resize(local_row_count, 0.0F);
  }
  ++current_epoch_;
  if (current_epoch_ == 0) {
    std::fill(seen_epoch_.begin(), seen_epoch_.end(), 0);
    std::fill(planned_distance_epoch_.begin(), planned_distance_epoch_.end(), 0);
    current_epoch_ = 1;
  }
  candidate_heap_.clear();
  result_heap_.clear();
  entry_rank_.clear();
  predicate_order_.clear();
}

SearchResult search_index(const IndexView& index, const float* query_vector,
                          std::size_t query_dim, const QueryWindow& window,
                          std::uint32_t top_k, std::uint64_t budget,
                          QueryWorkspace& workspace,
                          QueryOutputMode output_mode) {
  if (query_vector == nullptr) {
    throw std::invalid_argument("query_vector cannot be null");
  }
  if (query_dim != index.vector_dim) {
    throw std::invalid_argument("query vector dimension does not match index");
  }
  if (top_k == 0) {
    throw std::invalid_argument("top_k must be positive");
  }
  if (budget == 0) {
    throw std::invalid_argument("B_total must be positive");
  }
  for (std::size_t dim = 0; dim < query_dim; ++dim) {
    if (!std::isfinite(query_vector[dim])) {
      throw std::invalid_argument("query vector contains NaN or Inf");
    }
  }

  SearchResult output;
  output.stats.requested_budget = budget;
  const bool rich_output = output_mode == QueryOutputMode::kRich;
  if (rich_output) copy_identity(index.identity, &output.stats);
  const RoutingResult route =
      route_first_actual_containing_block(index, window);
  output.stats.family_id = route.family_id;
  output.stats.template_id = route.template_id;
  output.stats.level = route.level;
  output.stats.block_id = route.block_id;
  output.stats.routing_clipped = route.routing_clipped;
  output.stats.shape_tau = route.shape_tau;
  output.stats.selected_scale = route.selected_scale;
  if (rich_output) {
    output.stats.block_grid_coords = route.grid_coords;
    output.stats.template_ratio = route.template_ratio;
    output.stats.normalized_query_widths.reserve(route.normalized_lower.size());
    output.stats.normalized_block_widths.reserve(
        route.normalized_block_lower.size());
    output.stats.side_blowups.reserve(route.normalized_lower.size());
    for (std::size_t i = 0; i < route.normalized_lower.size(); ++i) {
      const double query_width =
          route.normalized_upper[i] - route.normalized_lower[i];
      output.stats.normalized_query_widths.push_back(query_width);
      if (i < route.normalized_block_lower.size()) {
        const double block_width =
            route.normalized_block_upper[i] - route.normalized_block_lower[i];
        output.stats.normalized_block_widths.push_back(block_width);
        output.stats.side_blowups.push_back(
            query_width == 0.0 ? std::numeric_limits<double>::infinity()
                               : block_width / query_width);
      }
    }
  }

  if (route.outcome == RouteOutcome::kOutOfDomain) {
    output.stats.termination_reason = TerminationReason::kOutOfDomain;
    return output;
  }
  if (route.outcome == RouteOutcome::kEmptyTheoreticalBlock) {
    output.stats.termination_reason = TerminationReason::kEmptyBlock;
    return output;
  }

  const auto block_id = route.block_id;
  const auto row_begin = index.view_row_ptr[block_id];
  const auto row_end = index.view_row_ptr[block_id + 1];
  const auto local_row_count = row_end - row_begin;
  if (local_row_count > std::numeric_limits<std::uint32_t>::max()) {
    throw CorruptIndexError(
        "selected view exceeds the compact local-row query limit");
  }
  output.stats.block_object_count = row_end - row_begin;
  output.stats.num_graphs_searched = 1;
  workspace.begin_query(static_cast<std::size_t>(local_row_count));

  const auto budget_bound =
      budget > (std::numeric_limits<std::uint64_t>::max() -
                index.entries_per_query) /
                   index.max_degree
          ? local_row_count
          : std::min(local_row_count,
                     budget * index.max_degree + index.entries_per_query);
  workspace.candidate_heap_.reserve(static_cast<std::size_t>(budget_bound));
  workspace.result_heap_.reserve(
      static_cast<std::size_t>(std::min<std::uint64_t>(top_k, local_row_count)));

  auto candidate_less = [](const QueryWorkspace::CandidateRecord& lhs,
                           const QueryWorkspace::CandidateRecord& rhs) {
    if (lhs.distance != rhs.distance) return lhs.distance > rhs.distance;
    return lhs.local_row > rhs.local_row;
  };
  auto entry_better = [](const QueryWorkspace::CandidateRecord& lhs,
                         const QueryWorkspace::CandidateRecord& rhs) {
    if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
    return lhs.local_row < rhs.local_row;
  };
  // With hit_better as the heap comparator, the worst retained hit is at the
  // front, making replacement deterministic for both distance and ID ties.
  auto result_heap_compare = [](const SearchHit& lhs, const SearchHit& rhs) {
    return hit_better(lhs, rhs);
  };

  auto distance_for = [&](std::int64_t object_id, bool planning) -> float {
    if (object_id < 0 ||
        static_cast<std::uint64_t>(object_id) >= index.object_count) {
      throw CorruptIndexError("query path referenced an invalid object ID");
    }
    const auto object = static_cast<std::size_t>(object_id);
    const auto vector_offset = object * index.vector_dim;
    double squared = 0.0;
    for (std::size_t dim = 0; dim < index.vector_dim; ++dim) {
      const double delta = static_cast<double>(query_vector[dim]) -
                           static_cast<double>(index.vectors[vector_offset + dim]);
      squared += delta * delta;
    }
    const auto distance = static_cast<float>(std::sqrt(squared));
    if (!std::isfinite(distance)) {
      throw CorruptIndexError(
          "L2 distance is non-finite for validated corpus/query values");
    }
    if (planning) {
      ++output.stats.planning_distance_evaluations;
    } else {
      ++output.stats.traversal_distance_evaluations;
    }
    return distance;
  };

  auto local_row_for = [&](std::int64_t object_id) -> std::uint32_t {
    const auto* begin = index.row_object_id.data() + row_begin;
    const auto* end = index.row_object_id.data() + row_end;
    const auto* found = std::lower_bound(begin, end, object_id);
    if (found == end || *found != object_id) {
      throw CorruptIndexError("entry object has no row in the selected view");
    }
    return static_cast<std::uint32_t>(found - begin);
  };

  auto admit_with_distance = [&](std::uint32_t local_row, float distance) {
    const auto local = static_cast<std::size_t>(local_row);
    if (workspace.seen_epoch_[local] == workspace.current_epoch_) return false;
    workspace.seen_epoch_[local] = workspace.current_epoch_;
    workspace.candidate_heap_.push_back({distance, local_row});
    std::push_heap(workspace.candidate_heap_.begin(),
                   workspace.candidate_heap_.end(), candidate_less);
    ++output.stats.candidate_pushes;
    return true;
  };

  auto admit_candidate = [&](std::uint32_t local_row, bool planning) {
    const auto local = static_cast<std::size_t>(local_row);
    if (workspace.seen_epoch_[local] == workspace.current_epoch_) return false;
    if (!planning &&
        workspace.planned_distance_epoch_[local] == workspace.current_epoch_) {
      return admit_with_distance(
          local_row, workspace.planned_distance_cache_[local]);
    }
    const auto object_id = index.row_object_id[row_begin + local_row];
    return admit_with_distance(local_row, distance_for(object_id, planning));
  };

  const auto anchor = index.block_anchor_id[block_id];
  const auto anchor_local_row = local_row_for(anchor);
  admit_candidate(anchor_local_row, true);
  output.stats.entries_selected = 1;

  const auto extras = index.extra_entries(block_id);
  output.stats.entry_pool_considered = extras.size();
  workspace.entry_rank_.reserve(extras.size());
  for (const auto object_id : extras) {
    if (object_id == anchor) continue;
    const auto local_row = local_row_for(object_id);
    const auto distance = distance_for(object_id, true);
    workspace.planned_distance_epoch_[local_row] = workspace.current_epoch_;
    workspace.planned_distance_cache_[local_row] = distance;
    workspace.entry_rank_.push_back({distance, local_row});
  }
  const auto additional_entry_count = std::min<std::size_t>(
      workspace.entry_rank_.size(),
      static_cast<std::size_t>(index.entries_per_query - 1));
  const auto selected_entries_end =
      workspace.entry_rank_.begin() +
      static_cast<std::vector<QueryWorkspace::CandidateRecord>::difference_type>(
          additional_entry_count);
  std::partial_sort(workspace.entry_rank_.begin(), selected_entries_end,
                    workspace.entry_rank_.end(), entry_better);
  for (auto entry_it = workspace.entry_rank_.begin();
       entry_it != selected_entries_end; ++entry_it) {
    const auto& entry = *entry_it;
    // Ranking already computed this exact distance.  Passing it through avoids
    // a global per-object cache lookup while preserving the reference key.
    if (admit_with_distance(entry.local_row, entry.distance)) {
      ++output.stats.entries_selected;
    }
  }

  workspace.predicate_order_.resize(window.dimensions.size());
  for (std::size_t i = 0; i < window.dimensions.size(); ++i) {
    workspace.predicate_order_[i] = i;
  }
  std::stable_sort(
      workspace.predicate_order_.begin(), workspace.predicate_order_.end(),
      [&](std::size_t lhs, std::size_t rhs) {
        const double left_width =
            route.normalized_upper[lhs] - route.normalized_lower[lhs];
        const double right_width =
            route.normalized_upper[rhs] - route.normalized_lower[rhs];
        return left_width < right_width;
      });

  auto satisfies_original_window = [&](std::int64_t object_id) {
    ++output.stats.post_filter_checks;
    const auto attr_offset =
        static_cast<std::size_t>(object_id) * index.attr_dim;
    for (const auto i : workspace.predicate_order_) {
      ++output.stats.post_filter_dimension_checks;
      const double value = index.attrs_raw[attr_offset + window.dimensions[i]];
      if (value < window.lower_closed[i] || value > window.upper_closed[i]) {
        return false;
      }
    }
    return true;
  };

  auto retain_hit = [&](std::int64_t object_id, float distance) {
    const SearchHit hit{object_id, distance};
    if (workspace.result_heap_.size() < top_k) {
      workspace.result_heap_.push_back(hit);
      std::push_heap(workspace.result_heap_.begin(), workspace.result_heap_.end(),
                     result_heap_compare);
      ++output.stats.result_updates;
      return;
    }
    if (!hit_better(hit, workspace.result_heap_.front())) return;
    std::pop_heap(workspace.result_heap_.begin(), workspace.result_heap_.end(),
                  result_heap_compare);
    workspace.result_heap_.back() = hit;
    std::push_heap(workspace.result_heap_.begin(), workspace.result_heap_.end(),
                   result_heap_compare);
    ++output.stats.result_updates;
  };

  while (!workspace.candidate_heap_.empty() &&
         output.stats.actual_budget < budget) {
    std::pop_heap(workspace.candidate_heap_.begin(),
                  workspace.candidate_heap_.end(), candidate_less);
    const auto candidate = workspace.candidate_heap_.back();
    workspace.candidate_heap_.pop_back();
    ++output.stats.candidate_pops;
    const auto row = row_begin + candidate.local_row;
    const auto object_id = index.row_object_id[row];
    ++output.stats.actual_budget;

    if (satisfies_original_window(object_id)) {
      ++output.stats.qualifying_expansions;
      retain_hit(object_id, candidate.distance);
    }

    const auto adj_end = index.row_adj_ptr[row + 1];
    for (auto ref = index.row_adj_ptr[row]; ref < adj_end; ++ref) {
#if defined(__GNUC__) || defined(__clang__)
      constexpr std::uint64_t kPrefetchLookahead = 2;
      if (index.vector_dim >= 32 && ref + kPrefetchLookahead < adj_end) {
        const auto future_row =
            index.row_adj_target_row[ref + kPrefetchLookahead];
        const auto future_id = index.row_object_id[future_row];
        const auto future_offset =
            static_cast<std::size_t>(future_id) * index.vector_dim;
        __builtin_prefetch(index.vectors.data() + future_offset, 0, 1);
        const auto future_local =
            static_cast<std::size_t>(future_row - row_begin);
        __builtin_prefetch(workspace.seen_epoch_.data() + future_local, 0, 1);
      }
#endif
      const auto target_row = index.row_adj_target_row[ref];
      const auto target_local =
          static_cast<std::uint32_t>(target_row - row_begin);
      admit_candidate(target_local, false);
    }
  }

  output.stats.distance_evaluations =
      output.stats.planning_distance_evaluations +
      output.stats.traversal_distance_evaluations;
  output.stats.termination_reason =
      output.stats.actual_budget == budget && !workspace.candidate_heap_.empty()
          ? TerminationReason::kBudgetExhausted
          : TerminationReason::kFrontierExhausted;
  output.hits = workspace.result_heap_;
  std::sort(output.hits.begin(), output.hits.end(), hit_better);
  return output;
}

SearchResult search_index(const IndexView& index, const float* query_vector,
                          std::size_t query_dim, const QueryWindow& window,
                          std::uint32_t top_k, std::uint64_t budget,
                          QueryOutputMode output_mode) {
  thread_local QueryWorkspace workspace;
  return search_index(index, query_vector, query_dim, window, top_k, budget,
                      workspace, output_mode);
}

}  // namespace bounded_blowup

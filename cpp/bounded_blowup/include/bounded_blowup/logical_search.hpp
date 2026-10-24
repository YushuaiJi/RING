// SPDX-License-Identifier: MIT
// Native Logical Artifact V3 ANN query adapter.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "bounded_blowup/index_view.hpp"
#include "bounded_blowup/logical_artifact.hpp"
#include "bounded_blowup/result.hpp"
#include "bounded_blowup/router.hpp"
#include "bounded_blowup/routing_layout.hpp"

namespace bounded_blowup {

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
// Test-only seam for proving that member-to-local-row lookup is paid during
// bind rather than repeated by the query hot path.
std::uint64_t logical_search_member_comparisons_for_testing() noexcept;
void reset_logical_search_member_comparisons_for_testing() noexcept;
#endif

class LogicalQueryHandle;

// Scalar counters intentionally mirror the measured native search contract.
// The returned hit view and selected route remain backed by the caller-owned
// LogicalAnnWorkspace; no result vector is copied by the query seam.
struct LogicalAnnSearchStats {
  std::uint16_t family_id = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t template_id = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t level = std::numeric_limits<std::uint16_t>::max();
  std::uint64_t block_id = std::numeric_limits<std::uint64_t>::max();
  LogicalViewId view_id = std::numeric_limits<LogicalViewId>::max();
  double shape_tau = std::numeric_limits<double>::quiet_NaN();
  double selected_scale = std::numeric_limits<double>::quiet_NaN();

  bool routing_clipped = false;
  std::uint64_t block_object_count = 0U;
  std::uint32_t num_graphs_searched = 0U;
  std::uint64_t requested_budget = 0U;
  std::uint64_t actual_budget = 0U;

  std::uint64_t entry_pool_considered = 0U;
  std::uint64_t entries_selected = 0U;
  std::uint64_t planning_distance_evaluations = 0U;
  std::uint64_t traversal_distance_evaluations = 0U;
  std::uint64_t distance_evaluations = 0U;
  std::uint64_t candidate_pushes = 0U;
  std::uint64_t candidate_pops = 0U;
  std::uint64_t post_filter_checks = 0U;
  std::uint64_t post_filter_dimension_checks = 0U;
  std::uint64_t qualifying_expansions = 0U;
  std::uint64_t result_updates = 0U;

  // Number of variable-length route payload elements inspected after bind.
  // The immutable artifact route table is copied and checked once at bind, so
  // the hot query path contributes zero.
  std::uint64_t route_validation_element_checks = 0U;

  // Number of prepared block_id -> route-table accesses for this query.  The
  // handle path performs one direct indexed lookup after routing, independent
  // of the number of routes in the artifact.
  std::uint64_t route_lookup_checks = 0U;

  TerminationReason termination_reason = TerminationReason::kFrontierExhausted;
  bool fallback = false;
};

// Exact scalar compact ABI consumed by native_adapter._COMPACT_STATS_FIELDS.
// Keep this tuple's order and arity frozen: it is intentionally independent of
// the richer LogicalAnnSearchStats layout and contains no geometry vectors or
// identity strings.
using LogicalAnnCompactStats = std::tuple<
    std::uint16_t,  // family_id
    std::uint16_t,  // template_id
    std::uint64_t,  // block_id
    std::uint16_t,  // level
    double,         // shape_tau
    double,         // selected_scale
    bool,           // routing_clipped
    std::uint64_t,  // block_object_count
    std::uint32_t,  // num_graphs_searched
    std::uint64_t,  // requested_budget
    std::uint64_t,  // actual_budget
    std::uint64_t,  // entry_pool_considered
    std::uint64_t,  // entries_selected
    std::uint64_t,  // planning_distance_evaluations
    std::uint64_t,  // traversal_distance_evaluations
    std::uint64_t,  // distance_evaluations
    std::uint64_t,  // candidate_pushes
    std::uint64_t,  // candidate_pops
    std::uint64_t,  // post_filter_checks
    std::uint64_t,  // post_filter_dimension_checks
    std::uint64_t,  // qualifying_expansions
    std::uint64_t,  // result_updates
    std::uint8_t,   // termination_reason_code
    bool>;          // fallback
static_assert(std::tuple_size<LogicalAnnCompactStats>::value == 24U,
              "logical compact ABI must retain exactly 24 fields");

struct LogicalAnnSearchResultView {
  ArrayView<SearchHit> hits;
  LogicalAnnSearchStats stats;

  LogicalAnnCompactStats compact_stats() const noexcept {
    return LogicalAnnCompactStats{
        stats.family_id,
        stats.template_id,
        stats.block_id,
        stats.level,
        stats.shape_tau,
        stats.selected_scale,
        stats.routing_clipped,
        stats.block_object_count,
        stats.num_graphs_searched,
        stats.requested_budget,
        stats.actual_budget,
        stats.entry_pool_considered,
        stats.entries_selected,
        stats.planning_distance_evaluations,
        stats.traversal_distance_evaluations,
        stats.distance_evaluations,
        stats.candidate_pushes,
        stats.candidate_pops,
        stats.post_filter_checks,
        stats.post_filter_dimension_checks,
        stats.qualifying_expansions,
        stats.result_updates,
        static_cast<std::uint8_t>(stats.termination_reason),
        stats.fallback};
  }
};

// Reusable, block-local query storage. prepare() is the only operation that
// may allocate. A query owns exactly one candidate heap, one result heap, one
// epoch domain, one routing scratch, and one fixed-size LogicalScratch.
class LogicalAnnWorkspace {
 public:
  LogicalAnnWorkspace() = default;
  // A workspace owns mutable query scratch and is intentionally thread-local.
  // It must never be copied, moved, or shared between concurrent queries.
  LogicalAnnWorkspace(const LogicalAnnWorkspace&) = delete;
  LogicalAnnWorkspace& operator=(const LogicalAnnWorkspace&) = delete;
  LogicalAnnWorkspace(LogicalAnnWorkspace&&) = delete;
  LogicalAnnWorkspace& operator=(LogicalAnnWorkspace&&) = delete;

  // Derive capacities from the immutable artifact captured by a handle. The
  // workspace retains the handle state token and rejects every other handle.
  void prepare(const LogicalQueryHandle& handle, std::uint32_t max_top_k);

 private:
  struct RouteState {
    RouteOutcome outcome = RouteOutcome::kOutOfDomain;
    std::uint16_t family_id = std::numeric_limits<std::uint16_t>::max();
    std::uint16_t template_id = std::numeric_limits<std::uint16_t>::max();
    std::uint16_t level = std::numeric_limits<std::uint16_t>::max();
    std::uint64_t block_id = std::numeric_limits<std::uint64_t>::max();
    bool routing_clipped = false;
    double shape_tau = std::numeric_limits<double>::quiet_NaN();
    double selected_scale = std::numeric_limits<double>::quiet_NaN();
  };

  struct CandidateRecord {
    float distance = 0.0F;
    std::uint32_t local_row = 0U;
  };
  static_assert(sizeof(CandidateRecord) == 8U,
                "logical candidate record must remain compact");

  std::vector<std::uint32_t> seen_epoch_;
  std::vector<CandidateRecord> candidate_heap_;
  std::vector<SearchHit> result_heap_;
  std::vector<CandidateRecord> entry_rank_;
  std::vector<std::size_t> predicate_order_;

  // Reusable routing scratch. The vectors are resized, never grown, by a
  // prepared query.
  std::vector<double> normalized_lower_;
  std::vector<double> normalized_upper_;
  std::vector<double> normalized_block_lower_;
  std::vector<double> normalized_block_upper_;
  std::vector<double> query_widths_;
  std::vector<double> template_ratio_;
  std::vector<std::int64_t> grid_coords_;

  std::size_t row_capacity_ = 0U;
  std::size_t dimension_capacity_ = 0U;
  std::size_t entry_capacity_ = 0U;
  std::uint32_t max_top_k_ = 0U;
  std::uint32_t current_epoch_ = 0U;
  bool prepared_ = false;
  LogicalScratch logical_scratch_;

  // A handle preparation retains its immutable state so queries remain valid
  // even when the originating handle is destroyed.
  std::shared_ptr<const void> prepared_handle_state_;
  void validate_storage_invariants() const;
  LogicalAnnSearchResultView search_bound(
      const LogicalQueryHandle& handle, const float* query_vector,
      std::size_t query_dim, const QueryWindow& window, std::uint32_t top_k,
      std::uint64_t budget, bool adaptive_early_stop,
      std::uint64_t adaptive_minimum_expansions,
      std::uint64_t adaptive_no_improvement_patience,
      bool adaptive_require_frontier_bound);
  static RouteState route_into(const LogicalQueryHandle&, const QueryWindow&,
                               LogicalAnnWorkspace*);
  friend class LogicalQueryHandle;
};

// Opaque immutable logical query owner.  A handle captures one artifact and
// one route-only layout snapshot at bind time.  All public query operations
// take this handle rather than an ordinary IndexView/independent artifact.
class LogicalQueryHandle final {
 public:
  // Opaque implementation state; callers cannot inspect or construct it.
  struct State;

  LogicalQueryHandle() = default;
  LogicalQueryHandle(const LogicalQueryHandle&) = default;
  LogicalQueryHandle& operator=(const LogicalQueryHandle&) = default;
  LogicalQueryHandle(LogicalQueryHandle&&) noexcept = default;
  LogicalQueryHandle& operator=(LogicalQueryHandle&&) noexcept = default;
  ~LogicalQueryHandle() = default;

  static LogicalQueryHandle bind(LogicalArtifactIndex&& artifact);

  bool valid() const noexcept { return static_cast<bool>(state_); }

  // prepare is the only handle operation allowed to allocate workspace
  // storage.  The state token is retained by workspace and compared in O(1)
  // by every subsequent search.
  void prepare(LogicalAnnWorkspace& workspace,
               std::uint32_t max_top_k) const;

  LogicalAnnSearchResultView search_into(
      const float* query_vector, std::size_t query_dim,
      const QueryWindow& window, std::uint32_t top_k, std::uint64_t budget,
      LogicalAnnWorkspace& workspace) const;

  // Experimental compact-search mode.  Once at least 200 nodes have been
  // expanded and top_k is full, stop after 64 consecutive expansions without
  // a result improvement when the best remaining frontier distance cannot
  // improve the current worst result.  The ordinary search_into path never
  // enables this policy and retains its frozen full-budget behavior.
  LogicalAnnSearchResultView search_into_adaptive(
      const float* query_vector, std::size_t query_dim,
      const QueryWindow& window, std::uint32_t top_k, std::uint64_t budget,
      LogicalAnnWorkspace& workspace) const;

  // Parameterized experimental policy used to measure a small Recall--QPS
  // frontier without rebuilding the same graph for every stopping rule.
  LogicalAnnSearchResultView search_into_adaptive(
      const float* query_vector, std::size_t query_dim,
      const QueryWindow& window, std::uint32_t top_k, std::uint64_t budget,
      std::uint64_t minimum_expansions,
      std::uint64_t no_improvement_patience,
      bool require_frontier_bound,
      LogicalAnnWorkspace& workspace) const;

  LogicalAnnSearchResultView search_into(
      ArrayView<float> query_vector, const QueryWindow& window,
      std::uint32_t top_k, std::uint64_t budget,
      LogicalAnnWorkspace& workspace) const {
    return search_into(query_vector.data(), query_vector.size(), window, top_k,
                       budget, workspace);
  }

  LogicalAnnSearchResultView search_into(
      ArrayView<const float> query_vector, const QueryWindow& window,
      std::uint32_t top_k, std::uint64_t budget,
      LogicalAnnWorkspace& workspace) const {
    return search_into(query_vector.data(), query_vector.size(), window, top_k,
                       budget, workspace);
  }

  LogicalAnnSearchResultView search_into(
      const std::vector<float>& query_vector, const QueryWindow& window,
      std::uint32_t top_k, std::uint64_t budget,
      LogicalAnnWorkspace& workspace) const {
    return search_into(query_vector.data(), query_vector.size(), window, top_k,
                       budget, workspace);
  }

  LogicalAnnSearchResultView search_into(
      const float* query_vector, const QueryWindow& window,
      std::uint32_t top_k, std::uint64_t budget,
      LogicalAnnWorkspace& workspace) const;

 private:
  std::shared_ptr<const State> state_;

  struct BoundIndex;
  static BoundIndex make_bound_index(const State* state);

  explicit LogicalQueryHandle(std::shared_ptr<const State> state)
      : state_(std::move(state)) {}

  friend class LogicalAnnWorkspace;
};

}  // namespace bounded_blowup

// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstdint>
#include <vector>

#include "bounded_blowup/index_view.hpp"
#include "bounded_blowup/result.hpp"
#include "bounded_blowup/router.hpp"

namespace bounded_blowup {

// Rich mode preserves the diagnostic payload used by audits and plotting.
// Compact mode keeps hits and traversal counters but avoids copying immutable
// identity strings and query-geometry vectors on the measured hot path.
enum class QueryOutputMode : std::uint8_t {
  kRich = 0,
  kCompact = 1,
};

// Scratch storage is owned per query thread. Epoch arrays are indexed by the
// selected block's local rows (not the whole corpus), avoiding O(N) clearing
// while preserving exact unique-expansion accounting.
class QueryWorkspace {
 public:
  void begin_query(std::size_t local_row_count);

 private:
  friend SearchResult search_index(const IndexView&, const float*,
                                   std::size_t, const QueryWindow&,
                                   std::uint32_t, std::uint64_t,
                                   QueryWorkspace&, QueryOutputMode);

  struct CandidateRecord {
    float distance = 0.0F;
    // The selected view is strictly sorted by object ID, so a local row is
    // both the adjacency address and the deterministic object-ID tie breaker.
    std::uint32_t local_row = 0;
  };
  static_assert(sizeof(CandidateRecord) == 8,
                "hot-path candidate record must remain compact");

  std::vector<std::uint32_t> seen_epoch_;
  // Only entry-pool distances need reuse: the planner scores every extra but
  // may admit only a subset before traversal reaches the remaining entries.
  // Keeping this cache block-local preserves reference counters without the
  // former whole-corpus distance-cache probe on the normal path.
  std::vector<std::uint32_t> planned_distance_epoch_;
  std::vector<float> planned_distance_cache_;
  std::vector<CandidateRecord> candidate_heap_;
  std::vector<SearchHit> result_heap_;
  std::vector<CandidateRecord> entry_rank_;
  std::vector<std::size_t> predicate_order_;
  std::uint32_t current_epoch_ = 0;
};

SearchResult search_index(const IndexView& index, const float* query_vector,
                          std::size_t query_dim, const QueryWindow& window,
                          std::uint32_t top_k, std::uint64_t budget,
                          QueryWorkspace& workspace,
                          QueryOutputMode output_mode = QueryOutputMode::kRich);

// Convenience entry for C++ callers; the workspace is thread-local and is
// never shared between concurrent queries.
SearchResult search_index(const IndexView& index, const float* query_vector,
                          std::size_t query_dim, const QueryWindow& window,
                          std::uint32_t top_k, std::uint64_t budget,
                          QueryOutputMode output_mode = QueryOutputMode::kRich);

}  // namespace bounded_blowup

// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_blowup/construction.hpp"

namespace bounded_blowup {

#if defined(RFBB_LEAF_BUILD_TESTING)
// Test-only observability for the worker-launch policy.  This declaration is
// intentionally absent from production headers/builds.
std::uint64_t leaf_build_thread_spawn_count_for_testing() noexcept;
void reset_leaf_build_thread_spawn_count_for_testing() noexcept;
#endif

class LeafBuildError : public std::runtime_error {
 public:
  explicit LeafBuildError(const std::string& message)
      : std::runtime_error(message) {}
};

enum class LeafCandidatePolicy : std::uint8_t {
  kAffineSampleV1 = 0U,
  kProjectionSymmetricPropagationV1 = 1U,
};

struct LeafBuildOptions {
  // The native leaf contract deliberately has a small closed candidate-size
  // set.  This keeps Python/native fixtures and cost accounting comparable.
  std::uint32_t candidate_size = 32;
  LeafCandidatePolicy candidate_policy = LeafCandidatePolicy::kAffineSampleV1;
  std::uint32_t max_degree = 24;
  double alpha = 1.2;
  std::uint64_t seed = 0;
  std::uint32_t thread_count = 1;
};

struct LeafBuildEvidence {
  std::uint64_t rows = 0;
  std::uint64_t candidate_visited = 0;
  std::uint64_t candidate_distance_evaluations = 0;
  std::uint64_t candidate_budget_exhausted_rows = 0;
  std::uint64_t candidate_membership_exhausted_rows = 0;
  std::uint64_t alpha_distance_evaluations = 0;
  std::uint64_t distance_evaluations = 0;
  std::uint64_t alpha_rejected = 0;
  std::uint64_t pruned_rows = 0;
  std::uint64_t pruned_edges = 0;

  bool operator==(const LeafBuildEvidence& other) const noexcept {
    return rows == other.rows &&
           candidate_visited == other.candidate_visited &&
           candidate_distance_evaluations ==
               other.candidate_distance_evaluations &&
           candidate_budget_exhausted_rows ==
               other.candidate_budget_exhausted_rows &&
           candidate_membership_exhausted_rows ==
               other.candidate_membership_exhausted_rows &&
           alpha_distance_evaluations == other.alpha_distance_evaluations &&
           distance_evaluations == other.distance_evaluations &&
           alpha_rejected == other.alpha_rejected &&
           pruned_rows == other.pruned_rows &&
           pruned_edges == other.pruned_edges;
  }
};

struct LeafBuildResult {
  // Rows are ordered exactly as row_object_ids (the supplied sorted member
  // list).  The candidate CSR is retained as audit evidence; it is not an
  // artifact-store write and contains no repair or parent-compose edges.
  std::vector<std::int64_t> row_object_ids;
  std::vector<std::uint64_t> candidate_row_ptr;
  std::vector<std::int64_t> candidate_target_ids;

  // Final CSR after the bounded Alpha prune.  Every target is a corpus ID,
  // is unique within its row, and is never the row's source ID.
  std::vector<std::uint64_t> row_ptr;
  std::vector<std::int64_t> target_ids;
  LeafBuildEvidence evidence;
};

// Build all leaf rows for a sorted, unique member list.  The corpus view must
// carry an already-verified finite-values contract; this function checks shape
// and ID bounds but intentionally does not rescan corpus values.  No HNSW,
// FAISS, reachability repair, artifact integration, or Python callback is
// involved in this seam.
LeafBuildResult build_leaf(const CorpusView& corpus,
                           ArrayView<std::int64_t> members,
                           const LeafBuildOptions& options = {});

}  // namespace bounded_blowup

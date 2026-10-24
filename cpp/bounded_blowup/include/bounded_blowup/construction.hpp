// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_blowup/index_view.hpp"

namespace bounded_blowup {

class ParentComposeError : public std::runtime_error {
 public:
  explicit ParentComposeError(const std::string& message)
      : std::runtime_error(message) {}
};

// CorpusView is an explicit prevalidation contract.  compose_parent checks
// shape and every object-ID bound, but intentionally does not rescan the
// corpus for finite values once this immutable view has been validated by the
// caller.
struct CorpusView {
  ArrayView<float> vectors;
  std::uint32_t vector_dim = 0;
  std::uint64_t object_count = 0;
  bool finite_values_prevalidated = false;
};

inline constexpr std::uint64_t kNoWitnessParent =
    std::numeric_limits<std::uint64_t>::max();

// Read-only child-view CSR supplied by the existing Shared View-CSR store.
// Row pointers and target-row references are local to this child view.  The
// witness parent for every non-anchor row is explicit and must be an actual
// directed child edge; the native seam never invents a child connectivity
// proof.
struct ChildView {
  std::int64_t child_id = -1;
  std::uint64_t component_sort_key = 0;
  ArrayView<std::int64_t> members;
  std::int64_t anchor = -1;
  ArrayView<std::int64_t> row_object_ids;
  ArrayView<std::uint64_t> row_adj_ptr;
  ArrayView<std::uint64_t> row_adj_target_row;
  ArrayView<std::uint64_t> witness_parent_row;
};

struct ParentComposeOptions {
  std::uint32_t max_degree = 24;
  double alpha = 1.2;
  std::uint32_t sample_count = 16;
  std::uint64_t seed = 0;
  std::uint64_t parent_identity = 0;
  std::uint32_t thread_count = 1;
};

enum ParentRowReason : std::uint8_t {
  kRowReasonOverlap = 1U << 0U,
  kRowReasonWitness = 1U << 1U,
  kRowReasonBridge = 1U << 2U,
  kRowReasonPruned = 1U << 3U,
};

struct ParentConnectionEvidence {
  std::uint64_t from_group = 0;
  std::uint64_t to_group = 0;
  std::int64_t source_id = -1;
  std::int64_t target_id = -1;
  std::uint64_t sample_distance_evaluations = 0;

  bool operator==(const ParentConnectionEvidence& other) const noexcept {
    return from_group == other.from_group && to_group == other.to_group &&
           source_id == other.source_id && target_id == other.target_id &&
           sample_distance_evaluations == other.sample_distance_evaluations;
  }
};

struct ParentComposeEvidence {
  std::uint64_t owner_rows = 0;
  std::uint64_t overlap_rows = 0;
  std::uint64_t non_dirty_rows = 0;
  std::uint64_t dirty_rows = 0;
  std::uint64_t pruned_rows = 0;
  std::uint64_t inherited_edge_refs = 0;
  std::uint64_t natural_unreachable = 0;
  std::uint64_t disconnected_parts = 0;
  std::uint64_t group_count = 0;
  std::uint64_t connection_pairs = 0;
  std::uint64_t sample_distance_evaluations = 0;
  std::uint64_t sample_reverse_verified = 0;
  std::uint64_t unreachable_after_audit = 0;
  std::uint64_t max_observed_degree = 0;
  std::uint64_t overflow_rows = 0;
  std::vector<ParentConnectionEvidence> connections;
};

struct ParentChildIdentity {
  std::int64_t child_id = -1;
  std::uint64_t component_sort_key = 0;

  bool operator==(const ParentChildIdentity& other) const noexcept {
    return child_id == other.child_id &&
           component_sort_key == other.component_sort_key;
  }
};

struct ParentComposeResult {
  // Child identity and owner mapping are sufficient to reconstruct an
  // ordinary row from child artifacts.  owner_ptr is indexed by
  // parent_members order; each flattened owner reference stores the child
  // identity index and its local row index.
  std::vector<ParentChildIdentity> child_identities;
  std::vector<std::uint64_t> owner_ptr;
  std::vector<std::uint64_t> owner_child_index;
  std::vector<std::uint64_t> owner_child_row;

  // Sparse replacement rows only.  A row with reason mask zero is inherited
  // verbatim from its owner and is intentionally absent from this overlay.
  std::vector<std::int64_t> overlay_source_ids;
  std::vector<std::uint64_t> overlay_row_ptr;
  std::vector<std::int64_t> overlay_target_ids;
  std::vector<std::uint8_t> row_reason_mask;
  ParentComposeEvidence evidence;
};

ParentComposeResult compose_parent(
    const CorpusView& corpus, ArrayView<std::int64_t> parent_members,
    std::int64_t parent_anchor, ArrayView<ChildView> children,
    const ParentComposeOptions& options = {});

}  // namespace bounded_blowup

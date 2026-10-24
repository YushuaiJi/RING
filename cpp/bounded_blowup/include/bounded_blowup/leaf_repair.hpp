// SPDX-License-Identifier: MIT
// Native leaf reachability repair is a separate seam from leaf construction.
#pragma once

#include <array>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bounded_blowup/leaf_build.hpp"

namespace bounded_blowup {

// The numeric revision below is meaningful only for one explicitly frozen
// scalar representation.  Reject a heterogeneous build at compile time
// instead of allowing two hosts to publish the same revision with different
// distance or tie semantics.
static_assert(std::numeric_limits<float>::is_iec559,
              "leaf repair requires IEC 60559 binary32 floats");
static_assert(std::numeric_limits<double>::is_iec559,
              "leaf repair requires IEC 60559 binary64 doubles");
static_assert(std::numeric_limits<float>::radix == 2,
              "leaf repair requires radix-2 binary32 floats");
static_assert(std::numeric_limits<double>::radix == 2,
              "leaf repair requires radix-2 binary64 doubles");
static_assert(std::numeric_limits<float>::digits == 24,
              "leaf repair requires 24-bit binary32 significands");
static_assert(std::numeric_limits<double>::digits == 53,
              "leaf repair requires 53-bit binary64 significands");
static_assert(sizeof(float) == 4U &&
                  sizeof(float) * static_cast<std::size_t>(CHAR_BIT) == 32U,
              "leaf repair requires 32-bit float objects");
static_assert(sizeof(double) == 8U &&
                  sizeof(double) * static_cast<std::size_t>(CHAR_BIT) == 64U,
              "leaf repair requires 64-bit double objects");

class LeafRepairError : public std::runtime_error {
 public:
  explicit LeafRepairError(const std::string& message)
      : std::runtime_error(message) {}
};

inline constexpr std::uint32_t kLeafRepairMaxDegree = 24U;
inline constexpr std::uint32_t kLeafRepairSampleBudget = 8U;
inline constexpr std::string_view kLeafRepairAlgorithmRevision =
    "native_leaf_reachability_repair_v1";
inline constexpr std::string_view kLeafRepairNumericRevision =
    "f32_f64_sq_l2_seq_rne_alpha2_le_v1";
inline constexpr std::string_view kLeafRepairTieOrder =
    "distance_source_id_target_id_v1";
inline constexpr std::string_view kLeafRepairTieOrderRevision =
    kLeafRepairTieOrder;
inline constexpr std::string_view kLeafRepairCanonicalDigestRevision =
    "fixed_u64x4_le_fnv1a_v1";

struct LeafRepairOptions {
  // These values are deliberately closed: they are part of the construction
  // identity, not tunable ANN knobs.  Repair itself is serial; an outer bulk
  // builder owns any worker-level parallelism.
  std::uint32_t max_degree = kLeafRepairMaxDegree;
  std::uint32_t sample_budget = kLeafRepairSampleBudget;
  double alpha = 1.2;
};

// This receipt is deliberately value-only.  Revision labels point at static
// string literals, while alpha is bound by its exact binary64 bit patterns;
// a bulk builder can persist it without mutable strings or implementation-
// defined hashes.
struct LeafRepairIdentityReceipt {
  std::int64_t frozen_anchor = -1;
  std::uint32_t max_degree = kLeafRepairMaxDegree;
  std::uint32_t sample_budget = kLeafRepairSampleBudget;
  std::uint64_t alpha_bits = 0U;
  std::uint64_t alpha_squared_bits = 0U;
  std::string_view algorithm_revision = kLeafRepairAlgorithmRevision;
  std::string_view numeric_revision = kLeafRepairNumericRevision;
  std::string_view tie_order = kLeafRepairTieOrder;
  bool thread_independent = true;

  bool operator==(const LeafRepairIdentityReceipt& other) const noexcept {
    return frozen_anchor == other.frozen_anchor &&
           max_degree == other.max_degree &&
           sample_budget == other.sample_budget && alpha_bits == other.alpha_bits &&
           alpha_squared_bits == other.alpha_squared_bits &&
           algorithm_revision == other.algorithm_revision &&
           numeric_revision == other.numeric_revision &&
           tie_order == other.tie_order &&
           thread_independent == other.thread_independent;
  }
};

using LeafRepairDigest = std::array<std::uint64_t, 4>;

struct LeafRepairBridgeEvidence {
  std::int64_t source_id = -1;
  std::int64_t target_id = -1;
  double squared_distance = 0.0;
  std::uint64_t sample_distance_evaluations = 0U;

  bool operator==(const LeafRepairBridgeEvidence& other) const noexcept {
    return source_id == other.source_id && target_id == other.target_id &&
           squared_distance == other.squared_distance &&
           sample_distance_evaluations == other.sample_distance_evaluations;
  }
};

struct LeafRepairEvidence {
  std::uint64_t rows = 0U;
  // The finite-values receipt is trusted at this seam; no full corpus value
  // scan is part of leaf repair.
  std::uint64_t corpus_value_scans = 0U;
  std::uint64_t initial_reachable = 0U;
  std::uint64_t initial_unreachable = 0U;
  std::uint64_t unreachable_source_scc_count = 0U;
  std::uint64_t repair_bridges = 0U;
  // Bridge sampling is the fixed 8-by-8 work budget.  Dirty-row Alpha
  // pruning has its own counter because it is bounded by one row, not by the
  // bridge sample budget.
  std::uint64_t repair_distance_evaluations = 0U;
  std::uint64_t repair_alpha_distance_evaluations = 0U;
  std::uint64_t repair_alpha_rejected = 0U;
  std::uint64_t dirty_source_rows = 0U;
  std::uint64_t bfs_edge_scans = 0U;
  std::uint64_t final_audit_edge_scans = 0U;
  std::vector<LeafRepairBridgeEvidence> bridges;

  bool operator==(const LeafRepairEvidence& other) const noexcept {
    return rows == other.rows &&
           corpus_value_scans == other.corpus_value_scans &&
           initial_reachable == other.initial_reachable &&
           initial_unreachable == other.initial_unreachable &&
           unreachable_source_scc_count == other.unreachable_source_scc_count &&
           repair_bridges == other.repair_bridges &&
           repair_distance_evaluations == other.repair_distance_evaluations &&
           repair_alpha_distance_evaluations ==
               other.repair_alpha_distance_evaluations &&
           repair_alpha_rejected == other.repair_alpha_rejected &&
           dirty_source_rows == other.dirty_source_rows &&
           bfs_edge_scans == other.bfs_edge_scans &&
           final_audit_edge_scans == other.final_audit_edge_scans &&
           bridges == other.bridges;
  }
};

struct LeafRepairResult {
  // Rows retain the exact sorted-member order supplied to build_leaf.  The
  // CSR vectors are owned snapshots suitable for immutable publication.  A
  // disconnected repair changes only bridge source rows; a connected leaf is
  // returned with the original CSR vectors unchanged.
  std::vector<std::int64_t> row_object_ids;
  std::vector<std::uint64_t> row_ptr;
  std::vector<std::int64_t> target_ids;

  // For every row, either kNoWitnessParent for the frozen anchor or the local
  // row number of a directed parent edge in target_ids.
  std::vector<std::uint64_t> witness_parent_row;
  LeafRepairIdentityReceipt identity;
  LeafRepairEvidence evidence;
};

// Validate and repair one complete, already-pruned leaf result in one call.
// The corpus is a prevalidated immutable view.  No external ANN or
// whole-graph fallback is available at this seam; an unsatisfiable contract
// throws LeafRepairError before a partial result can be published.
LeafRepairResult repair_leaf(const CorpusView& corpus,
                             const LeafBuildResult& leaf,
                             std::int64_t frozen_anchor,
                             const LeafRepairOptions& options = {});

// Canonical fixed-width digest over identity, final CSR, witness, and all
// immutable evidence.  It is independent of worker count and host hash
// randomization (a fixed non-cryptographic digest, not std::hash), and is
// suitable for a later bulk-builder receipt.
LeafRepairDigest leaf_repair_canonical_digest(const LeafRepairResult& result);

}  // namespace bounded_blowup

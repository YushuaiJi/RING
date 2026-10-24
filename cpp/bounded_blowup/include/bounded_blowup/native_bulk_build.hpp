// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Stage-A native coarse bulk-build seam.  This module owns all graph rows.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bounded_blowup/logical_search.hpp"
#include "bounded_blowup/verified_corpus.hpp"

namespace bounded_blowup {

// An owned structure-of-arrays descriptor.  It contains only block metadata,
// memberships, entries, and child references.  It deliberately has no graph
// CSR, candidate row, repair row, or parent overlay column.
struct NativeBulkBlockDescriptor final {
  std::vector<std::uint64_t> block_id;
  std::vector<std::uint32_t> block_level;
  std::vector<LogicalViewId> block_view_id;
  std::vector<std::uint64_t> block_component_sort_key;
  std::vector<std::uint8_t> block_kind;
  std::vector<std::int64_t> block_anchor_id;

  std::vector<std::uint64_t> block_member_ptr;
  std::vector<std::int64_t> block_member_ids;
  std::vector<std::uint64_t> block_entry_ptr;
  std::vector<std::int64_t> block_entry_ids;
  std::vector<std::uint64_t> block_child_ptr;
  std::vector<LogicalViewId> block_child_view_id;
  // Alias accepted for catalog producers that call these canonical route
  // references simply child IDs.  If both columns are supplied they must be
  // byte-identical; the builder never treats them as graph rows.
  std::vector<LogicalViewId> block_child_ids;

  // Copy/move ownership is intentional: the coarse call snapshots all SoA
  // values before any worker starts.  Validation is repeated by build() even
  // when callers have already called this method.
  static NativeBulkBlockDescriptor own(
      NativeBulkBlockDescriptor source) {
    return source;
  }

  static NativeBulkBlockDescriptor from_soa(
      NativeBulkBlockDescriptor source) {
    return source;
  }

  static NativeBulkBlockDescriptor copy_from(
      NativeBulkBlockDescriptor source) {
    return source;
  }

  std::size_t block_count() const noexcept { return block_id.size(); }

  // Throws NativeBulkBuildError on malformed offsets, IDs, ordering, kinds,
  // or a child edge that is not earlier in canonical (level, block_id) order.
  void validate(const VerifiedCorpusSnapshot& corpus) const;
};

// Compatibility spelling for callers that name the value after its SoA role.
using NativeBulkBlockSoA = NativeBulkBlockDescriptor;

enum class NativeBulkBuildState : std::uint8_t {
  kFailed = 0U,
  kComplete = 1U,
  kCancelled = 2U,
  kTimedOut = 3U,
};

enum class NativeBulkReportability : std::uint8_t {
  kNonreportableTestOnly = 0U,
};

inline constexpr std::uint32_t kNativeBulkMaxDegree = 24U;
inline constexpr std::uint32_t kNativeBulkSampleCount = 8U;
inline constexpr double kNativeBulkAlpha = 1.2;
inline constexpr std::uint32_t kNativeBulkLeafCandidateSize = 32U;
inline constexpr std::uint64_t kNativeBulkSeed = 0U;

inline constexpr const char* kNativeBulkAlgorithmRevision =
    "native_bulk_build_stage_a_mp8_symprop1_v1";
inline constexpr const char* kNativeBulkNumericRevision =
    "f32_f64_sq_l2_seq_rne_alpha2_le_v1";

class NativeBulkBuildError : public std::runtime_error {
 public:
  explicit NativeBulkBuildError(const std::string& message)
      : std::runtime_error(message) {}
};

// Scalar construction controls.  Only the first Stage-A algorithm revision is
// accepted: degree 24, alpha 1.2, sample count 8, one explicitly selected
// candidate size (32 or 64), and seed zero.  outer_worker_count is the one
// permitted parallelism knob; each leaf, repair pass, parent compose, and
// artifact assembly uses one worker.
struct NativeBulkBuildOptions final {
  std::uint32_t max_degree = kNativeBulkMaxDegree;
  double alpha = kNativeBulkAlpha;
  std::uint32_t sample_count = kNativeBulkSampleCount;
  std::uint32_t leaf_candidate_size = kNativeBulkLeafCandidateSize;
  std::uint64_t seed = kNativeBulkSeed;
  std::uint32_t outer_worker_count = 16U;
  // Vocabulary aliases retained for the existing native construction options.
  // A non-default alias may be used in place of the Stage-A spelling; setting
  // both aliases to different values fails closed.
  std::uint32_t thread_count = 16U;
  std::uint32_t candidate_size = kNativeBulkLeafCandidateSize;

  // The production hard wall is supplied by the runner.  The in-memory
  // Stage-A seam checks this deadline cooperatively between leaf tasks and
  // assembly phases; it does not preempt a single in-flight leaf.
  std::chrono::milliseconds timeout = std::chrono::minutes(5);
  const std::atomic<bool>* cancel_token = nullptr;
  std::function<bool()> cancellation_requested;

  // Reserved for the later disk-backed Stage-A/pybind slice.  No file is
  // written by this implementation, and receipt.persisted remains false.
  std::filesystem::path output_directory;
};

struct NativeBulkBuildReceipt final {
  NativeBulkBuildState state = NativeBulkBuildState::kFailed;
  NativeBulkReportability reportability =
      NativeBulkReportability::kNonreportableTestOnly;
  std::string failure_reason;

  std::string algorithm_revision = kNativeBulkAlgorithmRevision;
  std::string numeric_revision = kNativeBulkNumericRevision;
  std::uint32_t max_degree = kNativeBulkMaxDegree;
  double alpha = kNativeBulkAlpha;
  std::uint32_t sample_count = kNativeBulkSampleCount;
  std::uint32_t leaf_candidate_size = kNativeBulkLeafCandidateSize;
  std::uint64_t seed = kNativeBulkSeed;

  // Deterministic identity excludes worker count and timing/cancellation.
  LogicalSectionDigest build_identity{};
  std::string build_identity_hex;
  std::string corpus_sha256;
  std::string corpus_vectors_sha256;
  std::string corpus_attrs_sha256;
  std::uint32_t schema = kLogicalArtifactSchemaVersion;
  std::uint32_t artifact_schema = kLogicalArtifactSchemaVersion;
  LogicalSectionDigest logical_sha256{};
  LogicalSectionDigest route_sha256{};
  LogicalSectionDigest routing_sha256{};
  LogicalSectionDigest joint_sha256{};

  std::uint64_t leaf_count = 0U;
  std::uint64_t parent_count = 0U;
  std::uint64_t leaf_memberships = 0U;
  std::uint64_t parent_memberships = 0U;
  std::uint64_t candidate_visited = 0U;
  std::uint64_t candidate_evaluations = 0U;
  std::uint64_t candidate_budget_exhausted_rows = 0U;
  std::uint64_t candidate_membership_exhausted_rows = 0U;
  std::uint64_t leaf_alpha_distance_evaluations = 0U;
  std::uint64_t leaf_distance_evaluations = 0U;
  std::uint64_t leaf_alpha_rejected = 0U;
  std::uint64_t leaf_pruned_rows = 0U;
  std::uint64_t leaf_pruned_edges = 0U;
  std::uint64_t alpha_distance_evaluations = 0U;
  std::uint64_t alpha_pruned_rows = 0U;
  std::uint64_t alpha_pruned_edges = 0U;
  std::uint64_t repair_alpha_distance_evaluations = 0U;
  std::uint64_t repair_alpha_rejected = 0U;
  std::uint64_t repair_dirty_rows = 0U;

  // The V3 build does not return parent SCC/sample
  // evidence.  Aggregate fields remain zero unless their *_complete flag is
  // true; leaf repair values are exposed explicitly below.
  std::uint64_t scc_count = 0U;
  std::uint64_t bridge_count = 0U;
  std::uint64_t leaf_scc_count = 0U;
  std::uint64_t leaf_bridge_count = 0U;
  std::uint64_t parent_bridge_count = 0U;
  std::uint64_t leaf_initial_reachable = 0U;
  std::uint64_t leaf_initial_unreachable = 0U;
  std::uint64_t repair_bfs_edge_scans = 0U;
  std::uint64_t repair_final_audit_edge_scans = 0U;
  std::uint64_t sampled_bridge_distance_evaluations = 0U;
  std::uint64_t leaf_sampled_bridge_distance_evaluations = 0U;
  bool scc_count_complete = false;
  bool bridge_count_complete = false;
  bool sampled_bridge_distance_complete = false;
  bool parent_bridge_count_complete = false;
  bool alpha_aggregate_complete = false;

  std::uint64_t inherited_rows = 0U;
  std::uint64_t dirty_rows = 0U;
  std::uint64_t overlay_rows = 0U;
  std::uint64_t logical_rows = 0U;
  std::uint64_t logical_edges = 0U;
  std::uint64_t stored_rows = 0U;
  std::uint64_t stored_edges = 0U;
  std::uint64_t parent_stored_rows = 0U;
  std::uint64_t parent_stored_edges = 0U;
  std::uint64_t max_observed_degree = 0U;
  std::uint64_t overflow_rows = 0U;
  std::uint64_t clean_child_ref_rows = 0U;
  std::uint64_t child_ref_count = 0U;
  std::uint64_t projection_entries = 0U;
  std::uint64_t physical_edge_count = 0U;
  std::uint64_t section_bytes = 0U;
  std::uint64_t logical_section_bytes = 0U;
  std::uint64_t route_section_bytes = 0U;
  std::uint64_t routing_section_bytes = 0U;

  std::uint32_t requested_outer_workers = 0U;
  std::uint32_t outer_workers = 0U;
  std::uint32_t max_inner_workers = 1U;
  std::uint32_t nested_worker_count = 1U;

  bool reopened = false;
  bool audited = false;
  bool query_bound = false;
  bool persisted = false;
  bool complete_marker_published = false;

  // These paths are forbidden in Stage-A and are explicit receipt evidence.
  std::uint64_t full_stitch_count = 0U;
  std::uint64_t hnsw_count = 0U;
  std::uint64_t faiss_count = 0U;
  std::uint64_t cuda_count = 0U;

  bool complete() const noexcept {
    return state == NativeBulkBuildState::kComplete && reopened && audited &&
           query_bound && !complete_marker_published;
  }
};

// The result intentionally exposes only the immutable query handle and scalar
// receipt.  LogicalArtifactIndex, GraphSections, CSR rows, and parent overlays
// remain owned by the handle and never cross this seam.
class NativeBulkBuildResult final {
 public:
  NativeBulkBuildResult() = default;

  const NativeBulkBuildReceipt& receipt() const noexcept { return receipt_; }
  bool valid() const noexcept { return receipt_.complete() && query_.valid(); }

  const LogicalQueryHandle& query_handle() const noexcept { return query_; }
  const LogicalQueryHandle& handle() const noexcept { return query_; }

 private:
  NativeBulkBuildReceipt receipt_;
  LogicalQueryHandle query_;

  friend NativeBulkBuildResult native_bulk_build(
      VerifiedCorpusSnapshot, NativeBulkBlockDescriptor,
      RoutingLayoutSnapshot, const NativeBulkBuildOptions&);
};

// One coarse native operation.  Invalid descriptors/options throw before any
// artifact can be published.  A deadline/cancel observed during construction
// returns a failed/cancelled receipt with no valid query handle.
NativeBulkBuildResult native_bulk_build(
    VerifiedCorpusSnapshot corpus, NativeBulkBlockDescriptor descriptor,
    RoutingLayoutSnapshot routing,
    const NativeBulkBuildOptions& options = {});

// Explicitly named alias for callers that prefer an imperative spelling.
inline NativeBulkBuildResult build_native_bulk(
    VerifiedCorpusSnapshot corpus, NativeBulkBlockDescriptor descriptor,
    RoutingLayoutSnapshot routing,
    const NativeBulkBuildOptions& options = {}) {
  // VerifiedCorpusSnapshot deliberately deletes its move constructor; passing
  // by value still retains the immutable shared state without moving it.
  return native_bulk_build(corpus, std::move(descriptor),
                           std::move(routing), options);
}

}  // namespace bounded_blowup

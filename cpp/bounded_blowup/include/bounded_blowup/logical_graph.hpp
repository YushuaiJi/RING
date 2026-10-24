// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bounded_blowup/leaf_build.hpp"

namespace bounded_blowup {

using LogicalViewId = std::uint32_t;

inline constexpr std::uint32_t kLogicalMaxDegree = 24U;
inline constexpr std::uint32_t kLogicalMaxDepth = 16U;
inline constexpr std::uint32_t kLogicalSectionSchemaVersion = 1U;

class LogicalGraphError : public std::runtime_error {
 public:
  explicit LogicalGraphError(const std::string& message)
      : std::runtime_error(message) {}
};

struct LogicalGraphOptions {
  // The on-disk/query scratch layout has a frozen maximum depth.  A smaller
  // limit is allowed for a workload, but a caller cannot enlarge the layout.
  std::uint32_t max_depth = kLogicalMaxDepth;
};

enum class LogicalBlockKind : std::uint8_t { kLeaf = 0U, kParent = 1U };

enum class LogicalRowKind : std::uint8_t {
  kStored = 0U,
  kChildRef = 1U,
  kOverlay = 2U,
};

using LogicalSectionDigest = std::array<std::uint8_t, 32>;
using LogicalSectionFingerprint = LogicalSectionDigest;

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
// Test-only audit hook. Production builds do not expose or count this seam;
// the counter proves that one canonical section owner performs one SHA-256.
std::uint64_t logical_graph_sha256_call_count_for_testing() noexcept;
void reset_logical_graph_sha256_call_count_for_testing() noexcept;
std::uint64_t logical_graph_resolve_call_count_for_testing() noexcept;
void reset_logical_graph_resolve_call_count_for_testing() noexcept;
#endif

// Stable, host-independent logical-graph columns. Pointer arrays use global
// offsets and every vector is serialized with an explicit little-endian length.
// Callers may copy this value to construct malformed fixtures; a frozen
// LogicalGraphSections exposes only a const view of its copy.
struct LogicalGraphSectionData {
  std::uint32_t schema_version = kLogicalSectionSchemaVersion;
  std::uint32_t max_depth = kLogicalMaxDepth;
  std::uint64_t corpus_object_count = 0U;
  std::uint32_t corpus_vector_dim = 0U;
  std::uint8_t corpus_finite_values_prevalidated = 0U;

  // Blocks are ordered by (depth, view ID); their member and binding slices are
  // ordered by object/child ID. All *_ptr values below are global offsets.
  std::vector<LogicalViewId> block_view_id;
  std::vector<std::uint64_t> block_component_sort_key;
  std::vector<std::uint8_t> block_kind;
  std::vector<std::uint32_t> block_depth;
  std::vector<std::uint32_t> block_anchor_local_row;
  std::vector<std::uint64_t> block_member_ptr;
  std::vector<std::int64_t> member_ids;
  std::vector<std::uint64_t> block_entry_ptr;
  std::vector<std::int64_t> entry_ids;

  std::vector<std::uint64_t> block_binding_ptr;
  std::vector<LogicalViewId> binding_child_view_id;
  std::vector<std::uint64_t> binding_projection_ptr;
  std::vector<std::uint32_t> projection_parent_row;

  // Row columns align one-for-one with member_ids. owner_binding values index
  // the global binding columns and each row's owner refs are increasing.
  std::vector<std::uint64_t> row_owner_ptr;
  std::vector<std::uint32_t> owner_binding;
  std::vector<std::uint32_t> owner_child_row;
  std::vector<std::uint8_t> row_kind;
  std::vector<std::uint8_t> row_reason_mask;

  // Only leaf STORED and parent OVERLAY rows appear here. Sources are increasing
  // global row offsets. A physical edge ID is the index of its lexicographically
  // sorted, unique directed (src,dst) pair in the final two columns.
  std::vector<std::uint64_t> stored_source_global_row;
  std::vector<std::uint64_t> stored_row_ptr;
  std::vector<std::uint32_t> stored_target_local_row;
  std::vector<std::uint64_t> stored_physical_edge_id;
  std::vector<std::int64_t> physical_edge_src;
  std::vector<std::int64_t> physical_edge_dst;
};

struct LogicalGraphSectionStorage;

// Immutable owner for one canonical logical-graph section image. The untrusted
// factories check byte framing only; LogicalGraphView::open performs the full
// DAG, projection, row-kind, adjacency, physical-edge, and reachability audit.
class LogicalGraphSections {
 public:
  LogicalGraphSections() = default;

  static LogicalGraphSections from_untrusted_data(
      LogicalGraphSectionData data);
  static LogicalGraphSections from_canonical_bytes(
      std::vector<std::uint8_t> canonical_bytes);

  bool empty() const noexcept;
  // Returned views remain valid while this sections owner (or a copy) lives.
  const LogicalGraphSectionData& data() const;
  ArrayView<std::uint8_t> canonical_bytes() const noexcept;
  LogicalSectionDigest sha256() const noexcept;

 private:
  explicit LogicalGraphSections(
      std::shared_ptr<const LogicalGraphSectionStorage> storage);
  std::shared_ptr<const LogicalGraphSectionStorage> storage_;

  friend class LogicalGraphBuilder;
  friend class LogicalGraphView;
};

struct LogicalGraphStats {
  std::uint64_t block_count = 0;
  std::uint64_t leaf_count = 0;
  std::uint64_t parent_count = 0;
  std::uint64_t logical_rows = 0;
  std::uint64_t logical_edges = 0;
  std::uint64_t stored_rows = 0;
  std::uint64_t stored_edges = 0;
  std::uint64_t parent_stored_rows = 0;
  std::uint64_t parent_stored_edges = 0;
  std::uint64_t clean_child_ref_rows = 0;
  std::uint64_t dirty_overlay_rows = 0;
  std::uint64_t child_ref_count = 0;
  std::uint64_t projection_entries = 0;
  std::uint64_t physical_edge_count = 0;
  std::uint64_t section_bytes = 0;

  bool operator==(const LogicalGraphStats& other) const noexcept {
    return block_count == other.block_count && leaf_count == other.leaf_count &&
           parent_count == other.parent_count &&
           logical_rows == other.logical_rows &&
           logical_edges == other.logical_edges &&
           stored_rows == other.stored_rows &&
           stored_edges == other.stored_edges &&
           parent_stored_rows == other.parent_stored_rows &&
           parent_stored_edges == other.parent_stored_edges &&
           clean_child_ref_rows == other.clean_child_ref_rows &&
           dirty_overlay_rows == other.dirty_overlay_rows &&
           child_ref_count == other.child_ref_count &&
           projection_entries == other.projection_entries &&
           physical_edge_count == other.physical_edge_count &&
           section_bytes == other.section_bytes;
  }
};

struct LogicalNeighbors {
  // Both views point into the caller-owned LogicalScratch and are valid until
  // the next neighbors() call using that scratch. local_rows are indices in
  // the root view; object_ids are the same targets in corpus-ID space and
  // physical_edge_ids identify their canonical directed corpus-edge pairs.
  ArrayView<std::uint32_t> local_rows;
  ArrayView<std::int64_t> object_ids;
  ArrayView<std::uint64_t> physical_edge_ids;

  std::size_t size() const noexcept { return local_rows.size(); }
  bool empty() const noexcept { return local_rows.empty(); }
};

// Query scratch is fixed-size by contract.  It is supplied by the caller and
// is never allocated or resized by LogicalGraphView::neighbors().  Effective
// parent adjacency is prepared in the immutable graph owner, so neighbors()
// only copies a bounded CSR row into this scratch and does not recurse.
struct LogicalScratch {
  std::array<std::array<std::uint32_t, kLogicalMaxDegree>,
             kLogicalMaxDepth + 1U>
      frame_rows{};
  std::array<std::array<std::uint64_t, kLogicalMaxDegree>,
             kLogicalMaxDepth + 1U>
      frame_physical_edge_ids{};
  std::array<std::uint16_t, kLogicalMaxDepth + 1U> frame_counts{};
  std::array<std::int64_t, kLogicalMaxDegree> object_ids{};
};

struct LogicalGraphStorage;

class LogicalGraphView {
 public:
  LogicalGraphView() = default;
  LogicalGraphView(const LogicalGraphView&) = default;
  LogicalGraphView& operator=(const LogicalGraphView&) = default;
  LogicalGraphView(LogicalGraphView&&) noexcept = default;
  LogicalGraphView& operator=(LogicalGraphView&&) noexcept = default;
  ~LogicalGraphView();

  LogicalNeighbors neighbors(LogicalViewId root_view,
                              std::uint32_t root_local_row,
                              LogicalScratch& scratch) const;
  static LogicalGraphView open(const LogicalGraphSections& sections,
                               const CorpusView& corpus);
  void validate() const;
  ArrayView<std::int64_t> members(LogicalViewId view) const;
  LogicalGraphStats stats() const noexcept;
  LogicalSectionFingerprint section_fingerprint() const noexcept;

 private:
  explicit LogicalGraphView(std::shared_ptr<const LogicalGraphStorage> storage);
  std::shared_ptr<const LogicalGraphStorage> storage_;

  friend class LogicalGraphBuilder;
};

class LogicalGraphBuilder {
 public:
  // The corpus view is a non-owning construction dependency and must outlive
  // this builder.  Its identity is sealed into the logical sections; all leaf
  // and parent IDs are checked against the same bound corpus.
  explicit LogicalGraphBuilder(
      const CorpusView& corpus,
      const LogicalGraphOptions& options = LogicalGraphOptions{});
  ~LogicalGraphBuilder();

  LogicalGraphBuilder(const LogicalGraphBuilder&) = delete;
  LogicalGraphBuilder& operator=(const LogicalGraphBuilder&) = delete;
  LogicalGraphBuilder(LogicalGraphBuilder&&) noexcept;
  LogicalGraphBuilder& operator=(LogicalGraphBuilder&&) noexcept;

  // Leaf rows are copied as physical STORED rows.  The input result must be a
  // complete final CSR from build_leaf; candidate CSR is audit-only here.
  void add_leaf(LogicalViewId view_id, std::uint64_t component_sort_key,
                std::int64_t anchor, ArrayView<std::int64_t> entries,
                const LeafBuildResult& leaf);

  // children are previously added, immutable views.  The builder creates
  // ChildViews from their logical rows and invokes compose_parent itself;
  // callers cannot smuggle an arbitrary ParentComposeResult into storage.
  void add_parent(LogicalViewId view_id, std::uint64_t component_sort_key,
                  ArrayView<std::int64_t> members, std::int64_t anchor,
                  ArrayView<std::int64_t> entries,
                  ArrayView<LogicalViewId> children,
                  const ParentComposeOptions& options = {});

  // Both operations seal the builder exactly once. freeze() round-trips the
  // canonical bytes through the same untrusted-section loader used by open().
  LogicalGraphView freeze();
  LogicalGraphSections freeze_sections();

 private:
  CorpusView corpus_;
  std::shared_ptr<LogicalGraphStorage> storage_;
  bool frozen_ = false;
};

}  // namespace bounded_blowup

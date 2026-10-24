// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bounded_blowup/leaf_repair.hpp"
#include "bounded_blowup/logical_graph.hpp"
#include "bounded_blowup/routing_layout.hpp"
#include "bounded_blowup/verified_corpus.hpp"

namespace bounded_blowup {

class LogicalQueryHandle;

inline constexpr std::uint32_t kLogicalArtifactSchemaVersion = 3U;
inline constexpr const char* kLogicalStoreKind = "logical_child_ref_v3";
inline constexpr const char* kLogicalAlgorithmRevision = "cpp_v3";
inline constexpr const char* kLogicalNumericRevision =
    "f32_f64_sq_l2_seq_rne_alpha2_le_v1";

struct LogicalArtifactIdentity {
  std::string corpus_sha256;
  std::string corpus_vectors_sha256;
  std::string corpus_attrs_sha256;

  bool operator==(const LogicalArtifactIdentity& other) const noexcept {
    return corpus_sha256 == other.corpus_sha256 &&
           corpus_vectors_sha256 == other.corpus_vectors_sha256 &&
           corpus_attrs_sha256 == other.corpus_attrs_sha256;
  }
};

// One block-level descriptor is the only construction seam.  It is deliberately
// a complete bulk value: callers cannot add individual logical rows or edges.
struct LogicalArtifactBlockInput {
  std::uint64_t block_id = 0U;
  std::uint32_t level = 0U;
  LogicalViewId view_id = 0U;
  std::uint64_t component_sort_key = 0U;
  LogicalBlockKind kind = LogicalBlockKind::kLeaf;
  std::vector<std::int64_t> members;
  std::int64_t anchor = 0;
  std::vector<std::int64_t> entries;
  std::vector<LogicalViewId> children;
  // Meaningful only for a leaf. Only the immutable, repaired result is
  // accepted at the Artifact V3 seam; candidate LeafBuildResult values never
  // cross into artifact construction.
  LeafRepairResult leaf_repair;
  ParentComposeOptions parent_options;
};

struct LogicalArtifactBuildOptions {
  std::uint32_t thread_count = 1U;
};

struct LogicalArtifactRouteBinding {
  std::uint64_t block_id = 0U;
  LogicalViewId view_id = 0U;
  std::uint32_t level = 0U;
  std::uint64_t component_sort_key = 0U;
  std::int64_t anchor = 0;
  std::vector<std::int64_t> members;
  std::vector<std::int64_t> entries;
  std::vector<LogicalViewId> children;

  bool operator==(const LogicalArtifactRouteBinding& other) const noexcept {
    return block_id == other.block_id && view_id == other.view_id &&
           level == other.level && component_sort_key == other.component_sort_key &&
           anchor == other.anchor && members == other.members &&
           entries == other.entries && children == other.children;
  }
};

struct LogicalArtifactMetadata {
  std::uint32_t schema_version = kLogicalArtifactSchemaVersion;
  std::string store_kind = kLogicalStoreKind;
  std::string algorithm_revision = kLogicalAlgorithmRevision;
  std::string numeric_revision = kLogicalNumericRevision;
  LogicalArtifactIdentity identity;
  std::uint64_t corpus_object_count = 0U;
  std::uint32_t corpus_vector_dim = 0U;
  LogicalSectionDigest logical_sha256{};
  LogicalSectionDigest route_sha256{};
  LogicalSectionDigest routing_sha256{};
  LogicalSectionDigest joint_sha256{};
  std::uint64_t logical_section_bytes = 0U;
  std::uint64_t route_section_bytes = 0U;
  std::uint64_t routing_section_bytes = 0U;
  LogicalGraphStats logical_stats;
};

enum class LogicalArtifactSectionKind : std::uint8_t {
  kLogicalGraphCanonical = 0U,
  kBlockLogicalViewId = 1U,
  kRoutingLayoutCanonical = 2U,
};

// The image is an immutable, opaque three-section artifact.  Its only
// production construction seam is build(); no caller can supply section bytes,
// sidecar routes, or digest strings.
class LogicalArtifactImage final {
 public:
  struct State;

  LogicalArtifactImage(const LogicalArtifactImage& other);
  LogicalArtifactImage& operator=(const LogicalArtifactImage& other);
  LogicalArtifactImage(LogicalArtifactImage&& other) noexcept;
  LogicalArtifactImage& operator=(LogicalArtifactImage&& other) noexcept;
  ~LogicalArtifactImage();

  static LogicalArtifactImage build(
      VerifiedCorpusSnapshot corpus, RoutingLayoutSnapshot routing,
      std::vector<LogicalArtifactBlockInput> blocks,
      const LogicalArtifactBuildOptions& options = {});

  std::uint32_t schema_version() const noexcept;
  const LogicalArtifactMetadata& metadata() const noexcept;

 private:
  std::shared_ptr<const State> state_;

  explicit LogicalArtifactImage(std::shared_ptr<const State> state);
  const std::vector<std::uint8_t>& section(
      LogicalArtifactSectionKind kind) const noexcept;
  LogicalSectionDigest section_digest(
      LogicalArtifactSectionKind kind) const noexcept;
  LogicalSectionDigest joint_digest() const noexcept;

  friend class LogicalArtifactIndex;
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  friend LogicalArtifactImage logical_artifact_image_corrupt_copy_for_testing(
      const LogicalArtifactImage&, LogicalArtifactSectionKind,
      std::vector<std::uint8_t>, bool);
  friend std::vector<std::uint8_t> logical_artifact_section_bytes_for_testing(
      const LogicalArtifactImage&, LogicalArtifactSectionKind);
#endif
};

class LogicalArtifactError : public std::runtime_error {
 public:
  explicit LogicalArtifactError(const std::string& message)
      : std::runtime_error(message) {}
};

struct LogicalSearchWorkspace {
  // The vectors are allocated only by prepare(). A repeated search after
  // prepare() does not allocate; graph resolution uses the frozen fixed-size
  // LogicalScratch supplied by this workspace.
  std::vector<std::uint32_t> queue;
  std::vector<std::uint32_t> seen_epoch;
  std::vector<std::int64_t> output_object_ids;
  std::uint32_t epoch = 0U;
  LogicalScratch logical_scratch;

  void prepare(std::size_t local_row_count);
};

struct LogicalSearchResultView {
  ArrayView<std::int64_t> object_ids;
  std::uint64_t requested_budget = 0U;
  std::uint64_t actual_budget = 0U;
  std::uint64_t num_graphs_searched = 0U;
  bool fallback = false;
};

LogicalSectionDigest logical_artifact_sha256(ArrayView<std::uint8_t> bytes);

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::uint64_t logical_artifact_sha256_call_count_for_testing() noexcept;
void reset_logical_artifact_sha256_call_count_for_testing() noexcept;
#endif

class LogicalArtifactIndex {
 public:
  LogicalArtifactIndex() = delete;
  LogicalArtifactIndex(const LogicalArtifactIndex&) = default;
  LogicalArtifactIndex& operator=(const LogicalArtifactIndex&) = default;
  LogicalArtifactIndex(LogicalArtifactIndex&&) noexcept = default;
  LogicalArtifactIndex& operator=(LogicalArtifactIndex&&) noexcept = default;
  ~LogicalArtifactIndex() = default;

  static LogicalArtifactIndex open(VerifiedCorpusSnapshot corpus,
                                   const LogicalArtifactImage& image);

  const LogicalArtifactImage& image() const noexcept { return image_; }
  const LogicalArtifactMetadata& metadata() const noexcept {
    return image_.metadata();
  }
  const LogicalGraphSections& logical_sections() const noexcept {
    return logical_sections_;
  }
  const LogicalGraphView& logical_view() const noexcept { return logical_view_; }

  LogicalViewId route(std::uint64_t block_id) const;
  LogicalNeighbors neighbors(std::uint64_t block_id,
                             std::uint32_t root_local_row,
                             LogicalScratch& scratch) const;
  LogicalSearchResultView search_into(std::uint64_t block_id,
                                      std::uint32_t root_local_row,
                                      std::uint64_t budget,
                                      LogicalSearchWorkspace& workspace) const;

 private:
  friend class LogicalQueryHandle;

  LogicalArtifactIndex(std::shared_ptr<const VerifiedCorpusSnapshot> corpus,
                       LogicalArtifactImage image,
                       RoutingLayoutSnapshot routing)
      : corpus_snapshot_(std::move(corpus)),
        image_(std::move(image)),
        routing_(std::move(routing)) {}

  // Declare the owner before every borrower so destruction keeps the
  // verified mappings alive until the logical graph and artifact copies are
  // gone as well.
  std::shared_ptr<const VerifiedCorpusSnapshot> corpus_snapshot_;
  LogicalArtifactImage image_;
  RoutingLayoutSnapshot routing_;
  std::vector<LogicalArtifactRouteBinding> route_bindings_;
  LogicalGraphSections logical_sections_;
  LogicalGraphView logical_view_;
};

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
// Fixture-only corruption seam.  The optional section-digest rewrite models
// an attacker who edits a section and recomputes that descriptor; open() must
// still reject it because the immutable joint root covers all three images.
LogicalArtifactImage logical_artifact_image_corrupt_copy_for_testing(
    const LogicalArtifactImage& image, LogicalArtifactSectionKind section,
    std::vector<std::uint8_t> bytes, bool recompute_section_digest);
std::vector<std::uint8_t> logical_artifact_section_bytes_for_testing(
    const LogicalArtifactImage& image, LogicalArtifactSectionKind section);
#endif

}  // namespace bounded_blowup

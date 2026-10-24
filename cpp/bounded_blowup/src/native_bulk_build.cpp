// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
#include "bounded_blowup/native_bulk_build.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <utility>

#include "bounded_blowup/construction.hpp"
#include "bounded_blowup/leaf_build.hpp"
#include "bounded_blowup/leaf_repair.hpp"
#include "bounded_blowup/logical_artifact.hpp"

namespace bounded_blowup {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t double_bits(const double value) noexcept {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void append_u32(std::vector<std::uint8_t>* bytes, std::uint32_t value) {
  for (unsigned shift = 0U; shift < 32U; shift += 8U) {
    bytes->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64(std::vector<std::uint8_t>* bytes, std::uint64_t value) {
  for (unsigned shift = 0U; shift < 64U; shift += 8U) {
    bytes->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_string(std::vector<std::uint8_t>* bytes, const std::string& value) {
  append_u64(bytes, static_cast<std::uint64_t>(value.size()));
  bytes->insert(bytes->end(), value.begin(), value.end());
}

std::string digest_hex(const LogicalSectionDigest& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output;
  output.reserve(digest.size() * 2U);
  for (const std::uint8_t byte : digest) {
    output.push_back(kHex[(byte >> 4U) & 0x0fU]);
    output.push_back(kHex[byte & 0x0fU]);
  }
  return output;
}

template <typename T>
void require_parallel_size(const std::vector<T>& values, std::size_t count,
                           const char* label) {
  if (values.size() != count) {
    throw NativeBulkBuildError(std::string(label) + " length does not match block count");
  }
}

void require_offsets(const std::vector<std::uint64_t>& offsets,
                     std::size_t block_count, std::size_t payload_size,
                     const char* label) {
  if (offsets.size() != block_count + 1U || offsets.empty() ||
      offsets.front() != 0U || offsets.back() != payload_size) {
    throw NativeBulkBuildError(std::string(label) +
                               " has an invalid shape or terminal");
  }
  for (std::size_t index = 1U; index < offsets.size(); ++index) {
    if (offsets[index - 1U] > offsets[index] ||
        offsets[index] > static_cast<std::uint64_t>(payload_size)) {
      throw NativeBulkBuildError(std::string(label) +
                                 " is not monotone and in range");
    }
  }
}

template <typename T>
std::vector<T> copy_slice(const std::vector<T>& values,
                          const std::vector<std::uint64_t>& offsets,
                          std::size_t row, const char* label) {
  const std::uint64_t begin = offsets[row];
  const std::uint64_t end = offsets[row + 1U];
  if (end < begin || begin > static_cast<std::uint64_t>(values.size()) ||
      end > static_cast<std::uint64_t>(values.size()) ||
      begin > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
      end > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw NativeBulkBuildError(std::string(label) + " range is outside payload");
  }
  return std::vector<T>(values.begin() + static_cast<std::ptrdiff_t>(begin),
                        values.begin() + static_cast<std::ptrdiff_t>(end));
}

template <typename T>
ArrayView<T> slice_view(const std::vector<T>& values,
                       const std::vector<std::uint64_t>& offsets,
                       std::size_t row, const char* label) {
  const std::uint64_t begin = offsets[row];
  const std::uint64_t end = offsets[row + 1U];
  if (end < begin || begin > static_cast<std::uint64_t>(values.size()) ||
      end > static_cast<std::uint64_t>(values.size())) {
    throw NativeBulkBuildError(std::string(label) + " range is outside payload");
  }
  return ArrayView<T>(values.data() + static_cast<std::size_t>(begin),
                      static_cast<std::size_t>(end - begin));
}

bool includes_sorted(ArrayView<std::int64_t> parent,
                     ArrayView<std::int64_t> child) {
  std::size_t parent_row = 0U;
  for (const std::int64_t value : child) {
    while (parent_row < parent.size() && parent[parent_row] < value) {
      ++parent_row;
    }
    if (parent_row == parent.size() || parent[parent_row] != value) return false;
  }
  return true;
}

bool contains_sorted(const std::vector<std::int64_t>& values,
                     std::int64_t target) {
  return std::binary_search(values.begin(), values.end(), target);
}

CorpusView corpus_view(const VerifiedCorpusSnapshot& snapshot) {
  if (snapshot.object_count() == 0U || snapshot.vector_dim() == 0U ||
      snapshot.vector_dim() >
          static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
      snapshot.object_count() >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw NativeBulkBuildError("verified corpus shape is outside native bulk limits");
  }
  if (snapshot.object_count() >
          std::numeric_limits<std::uint64_t>::max() / snapshot.vector_dim()) {
    throw NativeBulkBuildError("verified corpus vector shape overflows");
  }
  const std::uint64_t values = snapshot.object_count() * snapshot.vector_dim();
  if (
      values != static_cast<std::uint64_t>(snapshot.vectors().size())) {
    throw NativeBulkBuildError("verified corpus vectors do not match shape");
  }
  return CorpusView{snapshot.vectors(), static_cast<std::uint32_t>(snapshot.vector_dim()),
                    snapshot.object_count(), true};
}

void validate_options(const NativeBulkBuildOptions& options) {
  if (options.max_degree != kNativeBulkMaxDegree) {
    throw NativeBulkBuildError("native bulk max_degree is fixed at 24");
  }
  if (!std::isfinite(options.alpha) ||
      double_bits(options.alpha) != double_bits(kNativeBulkAlpha)) {
    throw NativeBulkBuildError("native bulk alpha is fixed at 1.2");
  }
  if (options.sample_count != kNativeBulkSampleCount) {
    throw NativeBulkBuildError("native bulk sample_count is fixed at 8");
  }
  if (options.leaf_candidate_size != options.candidate_size) {
    throw NativeBulkBuildError("native bulk candidate-size aliases disagree");
  }
  if (options.leaf_candidate_size != 32U &&
      options.leaf_candidate_size != 64U) {
    throw NativeBulkBuildError(
        "native bulk leaf candidate size must be 32 or 64");
  }
  if (options.seed != kNativeBulkSeed) {
    throw NativeBulkBuildError("native bulk seed is fixed at zero");
  }
  const bool outer_alias_changed = options.outer_worker_count != 16U;
  const bool thread_alias_changed = options.thread_count != 16U;
  if (outer_alias_changed && thread_alias_changed &&
      options.outer_worker_count != options.thread_count) {
    throw NativeBulkBuildError(
        "native bulk outer worker aliases disagree");
  }
  const std::uint32_t workers = outer_alias_changed
                                    ? options.outer_worker_count
                                    : options.thread_count;
  if (workers == 0U || workers > 16U) {
    throw NativeBulkBuildError("native bulk outer worker count must be in [1, 16]");
  }
  if (options.timeout.count() < 0) {
    throw NativeBulkBuildError("native bulk timeout must be non-negative");
  }
  if (!options.output_directory.empty()) {
    throw NativeBulkBuildError(
        "native bulk Stage-A memory seam does not persist an output directory");
  }
}

std::uint32_t requested_outer_workers(const NativeBulkBuildOptions& options) {
  return options.outer_worker_count != 16U ? options.outer_worker_count
                                           : options.thread_count;
}

const std::vector<LogicalViewId>& child_values(
    const NativeBulkBlockDescriptor& descriptor) {
  if (!descriptor.block_child_view_id.empty() &&
      !descriptor.block_child_ids.empty() &&
      descriptor.block_child_view_id != descriptor.block_child_ids) {
    throw NativeBulkBuildError(
        "native bulk child alias columns disagree");
  }
  return descriptor.block_child_view_id.empty()
             ? descriptor.block_child_ids
             : descriptor.block_child_view_id;
}

void validate_canonical_child(const NativeBulkBlockDescriptor& descriptor,
                              const std::map<LogicalViewId, std::size_t>&
                                  view_index,
                              std::size_t parent, LogicalViewId child_view) {
  const auto child_iterator = view_index.find(child_view);
  if (child_iterator == view_index.end()) {
    throw NativeBulkBuildError("parent references an unknown child view");
  }
  const std::size_t child = child_iterator->second;
  const auto child_key = std::make_tuple(descriptor.block_level[child],
                                         descriptor.block_id[child]);
  const auto parent_key = std::make_tuple(descriptor.block_level[parent],
                                          descriptor.block_id[parent]);
  if (!(child_key < parent_key)) {
    throw NativeBulkBuildError(
        "parent child is not earlier in canonical (level, block_id) order");
  }
}

void add_leaf_counters(const LeafRepairResult& repaired,
                       NativeBulkBuildReceipt* receipt) {
  const LeafRepairEvidence& evidence = repaired.evidence;
  receipt->leaf_memberships += evidence.rows;
  receipt->repair_alpha_distance_evaluations +=
      evidence.repair_alpha_distance_evaluations;
  receipt->repair_alpha_rejected += evidence.repair_alpha_rejected;
  receipt->repair_dirty_rows += evidence.dirty_source_rows;
  receipt->leaf_scc_count += evidence.unreachable_source_scc_count;
  receipt->leaf_bridge_count += evidence.repair_bridges;
  receipt->leaf_initial_reachable += evidence.initial_reachable;
  receipt->leaf_initial_unreachable += evidence.initial_unreachable;
  receipt->repair_bfs_edge_scans += evidence.bfs_edge_scans;
  receipt->repair_final_audit_edge_scans += evidence.final_audit_edge_scans;
  receipt->leaf_sampled_bridge_distance_evaluations +=
      evidence.repair_distance_evaluations;
}

void collect_parent_counters(const LogicalGraphSections& sections,
                             NativeBulkBuildReceipt* receipt) {
  const LogicalGraphSectionData& data = sections.data();
  if (data.block_member_ptr.size() != data.block_view_id.size() + 1U ||
      data.block_kind.size() != data.block_view_id.size()) {
    throw NativeBulkBuildError("reopened logical section has invalid block columns");
  }
  for (std::size_t block = 0U; block < data.block_view_id.size(); ++block) {
    if (data.block_kind[block] !=
        static_cast<std::uint8_t>(LogicalBlockKind::kParent)) {
      continue;
    }
    const std::size_t begin = static_cast<std::size_t>(data.block_member_ptr[block]);
    const std::size_t end =
        static_cast<std::size_t>(data.block_member_ptr[block + 1U]);
    if (end < begin || end > data.member_ids.size() ||
        end > data.row_kind.size() || end > data.row_reason_mask.size()) {
      throw NativeBulkBuildError("reopened parent row columns are out of range");
    }
    receipt->parent_memberships += static_cast<std::uint64_t>(end - begin);
    for (std::size_t row = begin; row < end; ++row) {
      if (data.row_kind[row] ==
          static_cast<std::uint8_t>(LogicalRowKind::kChildRef)) {
        ++receipt->inherited_rows;
      } else if (data.row_kind[row] ==
                 static_cast<std::uint8_t>(LogicalRowKind::kOverlay)) {
        ++receipt->dirty_rows;
        ++receipt->overlay_rows;
      }
    }
  }
  for (std::size_t stored = 0U;
       stored < data.stored_source_global_row.size(); ++stored) {
    const std::uint64_t degree = data.stored_row_ptr[stored + 1U] -
                                 data.stored_row_ptr[stored];
    receipt->max_observed_degree =
        std::max(receipt->max_observed_degree, degree);
    if (degree > static_cast<std::uint64_t>(kLogicalMaxDegree)) {
      ++receipt->overflow_rows;
    }
  }
  // Parent compose evidence is not retained in the immutable V3 image.  Do
  // not infer a bridge count from endpoint reason bits: that is not an exact
  // count when one bridge can contribute multiple marked rows.
  receipt->parent_bridge_count = 0U;
  receipt->parent_bridge_count_complete = false;
}

void fill_identity(NativeBulkBuildReceipt* receipt) {
  std::vector<std::uint8_t> bytes;
  static constexpr std::array<std::uint8_t, 32> kMagic = {
      'R', 'F', 'B', 'B', '-', 'N', 'A', 'T', 'I', 'V', 'E', '-', 'B', 'U',
      'L', 'K', '-', 'S', 'T', 'A', 'G', 'E', '-', 'A', '-', 'V', '1', 0U, 0U,
      0U, 0U, 0U};
  bytes.insert(bytes.end(), kMagic.begin(), kMagic.end());
  append_string(&bytes, receipt->algorithm_revision);
  append_string(&bytes, receipt->numeric_revision);
  append_string(&bytes, receipt->corpus_sha256);
  append_string(&bytes, receipt->corpus_vectors_sha256);
  append_string(&bytes, receipt->corpus_attrs_sha256);
  append_u32(&bytes, receipt->schema);
  append_u32(&bytes, receipt->max_degree);
  append_u64(&bytes, double_bits(receipt->alpha));
  append_u32(&bytes, receipt->sample_count);
  append_u32(&bytes, receipt->leaf_candidate_size);
  append_u64(&bytes, receipt->seed);
  bytes.insert(bytes.end(), receipt->logical_sha256.begin(),
               receipt->logical_sha256.end());
  bytes.insert(bytes.end(), receipt->route_sha256.begin(),
               receipt->route_sha256.end());
  bytes.insert(bytes.end(), receipt->routing_sha256.begin(),
               receipt->routing_sha256.end());
  bytes.insert(bytes.end(), receipt->joint_sha256.begin(),
               receipt->joint_sha256.end());
  append_u64(&bytes, receipt->logical_rows);
  append_u64(&bytes, receipt->logical_edges);
  append_u64(&bytes, receipt->stored_rows);
  append_u64(&bytes, receipt->stored_edges);
  append_u64(&bytes, receipt->logical_section_bytes);
  append_u64(&bytes, receipt->route_section_bytes);
  append_u64(&bytes, receipt->routing_section_bytes);
  receipt->build_identity = logical_artifact_sha256(
      ArrayView<std::uint8_t>(bytes.data(), bytes.size()));
  receipt->build_identity_hex = digest_hex(receipt->build_identity);
}

NativeBulkBuildReceipt base_receipt(const NativeBulkBuildOptions& options,
                                    std::size_t leaf_count,
                                    std::uint32_t requested_workers) {
  NativeBulkBuildReceipt receipt;
  receipt.max_degree = options.max_degree;
  receipt.alpha = options.alpha;
  receipt.sample_count = options.sample_count;
  receipt.leaf_candidate_size = options.leaf_candidate_size;
  receipt.seed = options.seed;
  receipt.leaf_count = static_cast<std::uint64_t>(leaf_count);
  receipt.requested_outer_workers = requested_workers;
  receipt.outer_workers = static_cast<std::uint32_t>(std::min<std::size_t>(
      std::min<std::size_t>(16U, leaf_count), requested_workers));
  receipt.max_inner_workers = 1U;
  receipt.nested_worker_count = 1U;
  receipt.complete_marker_published = false;
  receipt.persisted = false;
  return receipt;
}

enum class AbortReason : std::uint8_t { kNone = 0U, kCancelled = 1U, kTimedOut = 2U,
                                        kError = 3U };

}  // namespace

void NativeBulkBlockDescriptor::validate(
    const VerifiedCorpusSnapshot& corpus) const {
  const std::size_t count = block_id.size();
  if (count == 0U) {
    throw NativeBulkBuildError("native bulk descriptor has no blocks");
  }
  require_parallel_size(block_level, count, "block_level");
  require_parallel_size(block_view_id, count, "block_view_id");
  require_parallel_size(block_component_sort_key, count,
                        "block_component_sort_key");
  require_parallel_size(block_kind, count, "block_kind");
  require_parallel_size(block_anchor_id, count, "block_anchor_id");
  require_offsets(block_member_ptr, count, block_member_ids.size(),
                  "block_member_ptr");
  require_offsets(block_entry_ptr, count, block_entry_ids.size(),
                  "block_entry_ptr");
  const std::vector<LogicalViewId>& children_payload = child_values(*this);
  require_offsets(block_child_ptr, count, children_payload.size(),
                  "block_child_ptr");

  const std::uint64_t object_count = corpus.object_count();
  if (object_count == 0U || object_count >
                                   static_cast<std::uint64_t>(
                                       std::numeric_limits<std::int64_t>::max())) {
    throw NativeBulkBuildError("native bulk corpus object count is invalid");
  }

  std::map<LogicalViewId, std::size_t> view_index;
  for (std::size_t block = 0U; block < count; ++block) {
    if (!view_index.emplace(block_view_id[block], block).second) {
      throw NativeBulkBuildError("native bulk descriptor repeats a view ID");
    }
  }
  std::size_t leaf_count = 0U;
  for (std::size_t block = 0U; block < count; ++block) {
    if (block_id[block] != static_cast<std::uint64_t>(block)) {
      throw NativeBulkBuildError(
          "native bulk block IDs must be contiguous and canonical");
    }
    if (block_level[block] > kLogicalMaxDepth) {
      throw NativeBulkBuildError("native bulk block level exceeds 16");
    }

    const std::vector<std::int64_t> members =
        copy_slice(block_member_ids, block_member_ptr, block, "membership");
    if (members.empty()) {
      throw NativeBulkBuildError("native bulk block membership is empty");
    }
    for (std::size_t row = 0U; row < members.size(); ++row) {
      if (members[row] < 0 || static_cast<std::uint64_t>(members[row]) >=
                                  object_count ||
          (row != 0U && members[row - 1U] >= members[row])) {
        throw NativeBulkBuildError(
            "native bulk block members must be sorted, unique, and in corpus");
      }
    }
    if (!contains_sorted(members, block_anchor_id[block])) {
      throw NativeBulkBuildError("native bulk block anchor is not a member");
    }

    const std::vector<std::int64_t> entries =
        copy_slice(block_entry_ids, block_entry_ptr, block, "entries");
    std::set<std::int64_t> seen_entries;
    for (std::size_t entry_index = 0U; entry_index < entries.size();
         ++entry_index) {
      const std::int64_t entry = entries[entry_index];
      if (entry < 0 || static_cast<std::uint64_t>(entry) >= object_count ||
          !contains_sorted(members, entry) ||
          !seen_entries.insert(entry).second ||
          (entry_index != 0U && entries[entry_index - 1U] >= entry)) {
        throw NativeBulkBuildError(
            "native bulk entries must be valid, unique members of their block");
      }
      if (entry == block_anchor_id[block]) {
        throw NativeBulkBuildError(
            "native bulk entries must not repeat their anchor");
      }
    }

    const auto kind = static_cast<LogicalBlockKind>(block_kind[block]);
    switch (kind) {
      case LogicalBlockKind::kLeaf: {
        ++leaf_count;
        if (block_level[block] != 0U || block_child_ptr[block] !=
                                               block_child_ptr[block + 1U]) {
          throw NativeBulkBuildError(
              "native bulk leaf must be level zero and child-free");
        }
        break;
      }
      case LogicalBlockKind::kParent: {
        if (block_level[block] == 0U || block_child_ptr[block] ==
                                             block_child_ptr[block + 1U]) {
          throw NativeBulkBuildError(
              "native bulk parent must have a positive level and children");
        }
        const std::vector<LogicalViewId> children = copy_slice(
            children_payload, block_child_ptr, block, "child references");
        std::set<LogicalViewId> child_set;
        for (std::size_t child_index = 0U; child_index < children.size();
             ++child_index) {
          const LogicalViewId child = children[child_index];
          if (!child_set.insert(child).second) {
            throw NativeBulkBuildError(
                "native bulk parent child list contains a duplicate");
          }
          if (child_index != 0U && children[child_index - 1U] >= child) {
            throw NativeBulkBuildError(
                "native bulk parent children must be sorted");
          }
          validate_canonical_child(*this, view_index, block, child);
          const std::size_t child_block = view_index.find(child)->second;
          const ArrayView<std::int64_t> child_members = slice_view(
              block_member_ids, block_member_ptr, child_block, "child membership");
          if (!includes_sorted(ArrayView<std::int64_t>(members.data(), members.size()),
                               child_members)) {
            throw NativeBulkBuildError(
                "parent membership does not contain child membership");
          }
        }
        break;
      }
      default:
        throw NativeBulkBuildError("native bulk block kind is invalid");
    }
  }
  if (leaf_count == 0U) {
    throw NativeBulkBuildError("native bulk descriptor has no leaves");
  }
}

NativeBulkBuildResult native_bulk_build(
    VerifiedCorpusSnapshot corpus, NativeBulkBlockDescriptor descriptor,
    RoutingLayoutSnapshot routing, const NativeBulkBuildOptions& options) {
  validate_options(options);
  descriptor.validate(corpus);
  // Resolve the optional child alias columns once during preflight.  The
  // assembly pass only slices this already-resolved payload and never repeats
  // a full-column equality check for every block.
  const std::vector<LogicalViewId>& resolved_children =
      descriptor.block_child_view_id.empty() ? descriptor.block_child_ids
                                             : descriptor.block_child_view_id;
  const CorpusView bound_corpus = corpus_view(corpus);
  const std::uint32_t requested_workers = requested_outer_workers(options);

  std::size_t leaf_count = 0U;
  for (const std::uint8_t kind : descriptor.block_kind) {
    if (kind == static_cast<std::uint8_t>(LogicalBlockKind::kLeaf)) ++leaf_count;
  }
  NativeBulkBuildResult output;
  output.receipt_ = base_receipt(options, leaf_count, requested_workers);
  output.receipt_.parent_count = static_cast<std::uint64_t>(
      descriptor.block_count() - leaf_count);
  const auto deadline = Clock::now() + options.timeout;
  std::mutex callback_mutex;
  auto external_abort = [&]() -> AbortReason {
    if (options.cancel_token != nullptr &&
        options.cancel_token->load(std::memory_order_relaxed)) {
      return AbortReason::kCancelled;
    }
    if (options.cancellation_requested) {
      std::lock_guard<std::mutex> lock(callback_mutex);
      if (options.cancellation_requested()) return AbortReason::kCancelled;
    }
    if (Clock::now() >= deadline) return AbortReason::kTimedOut;
    return AbortReason::kNone;
  };
  auto failed_result = [&](AbortReason reason, const char* message) {
    output.receipt_.state = reason == AbortReason::kCancelled
                                ? NativeBulkBuildState::kCancelled
                                : NativeBulkBuildState::kTimedOut;
    output.receipt_.failure_reason = message;
    output.receipt_.complete_marker_published = false;
    return output;
  };

  AbortReason initial_abort = external_abort();
  if (initial_abort != AbortReason::kNone) {
    return failed_result(initial_abort,
                         initial_abort == AbortReason::kCancelled
                             ? "native bulk build cancelled before leaf pool"
                             : "native bulk build timed out before leaf pool");
  }

  std::vector<LeafRepairResult> repaired(descriptor.block_count());
  std::vector<LeafBuildEvidence> leaf_evidence(descriptor.block_count());
  std::atomic<std::size_t> next_leaf{0U};
  std::atomic<bool> aborted{false};
  std::atomic<std::uint8_t> abort_reason{static_cast<std::uint8_t>(AbortReason::kNone)};
  std::mutex error_mutex;
  std::exception_ptr first_error;
  const std::size_t worker_count = output.receipt_.outer_workers;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  auto note_abort = [&](AbortReason reason) {
    aborted.store(true, std::memory_order_relaxed);
    std::uint8_t expected = static_cast<std::uint8_t>(AbortReason::kNone);
    (void)abort_reason.compare_exchange_strong(
        expected, static_cast<std::uint8_t>(reason),
        std::memory_order_relaxed, std::memory_order_relaxed);
  };
  auto worker = [&](std::size_t) {
    try {
      LeafBuildOptions leaf_options;
      leaf_options.candidate_size = options.leaf_candidate_size;
      leaf_options.candidate_policy =
          LeafCandidatePolicy::kProjectionSymmetricPropagationV1;
      leaf_options.max_degree = options.max_degree;
      leaf_options.alpha = options.alpha;
      leaf_options.seed = options.seed;
      leaf_options.thread_count = 1U;
      LeafRepairOptions repair_options;
      repair_options.max_degree = options.max_degree;
      repair_options.sample_budget = options.sample_count;
      repair_options.alpha = options.alpha;
      while (!aborted.load(std::memory_order_relaxed)) {
        const AbortReason reason = external_abort();
        if (reason != AbortReason::kNone) {
          note_abort(reason);
          break;
        }
        std::size_t leaf =
            next_leaf.fetch_add(1U, std::memory_order_relaxed);
        while (leaf < descriptor.block_count() &&
               descriptor.block_kind[leaf] !=
                   static_cast<std::uint8_t>(LogicalBlockKind::kLeaf)) {
          // The fetch index is the descriptor ordinal; skipping non-leaves
          // keeps the result slot stable and avoids a second scheduling queue.
          leaf = next_leaf.fetch_add(1U, std::memory_order_relaxed);
        }
        if (leaf >= descriptor.block_count()) break;
        if (descriptor.block_kind[leaf] !=
            static_cast<std::uint8_t>(LogicalBlockKind::kLeaf)) {
          continue;
        }
        const std::vector<std::int64_t> members = copy_slice(
            descriptor.block_member_ids, descriptor.block_member_ptr, leaf,
            "leaf membership");
        const LeafBuildResult built = build_leaf(
            bound_corpus, ArrayView<std::int64_t>(members.data(), members.size()),
            leaf_options);
        leaf_evidence[leaf] = built.evidence;
        repaired[leaf] = repair_leaf(bound_corpus, built,
                                     descriptor.block_anchor_id[leaf],
                                     repair_options);
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) first_error = std::current_exception();
      }
      note_abort(AbortReason::kError);
    }
  };
  try {
    for (std::size_t index = 0U; index < worker_count; ++index) {
      workers.emplace_back(worker, index);
    }
  } catch (...) {
    note_abort(AbortReason::kError);
    for (std::thread& thread : workers) {
      if (thread.joinable()) thread.join();
    }
    throw;
  }
  for (std::thread& thread : workers) {
    if (thread.joinable()) thread.join();
  }
  const AbortReason pool_abort = static_cast<AbortReason>(
      abort_reason.load(std::memory_order_relaxed));
  if (pool_abort == AbortReason::kError) {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (first_error != nullptr) std::rethrow_exception(first_error);
    throw NativeBulkBuildError("native bulk leaf worker failed");
  }
  if (pool_abort == AbortReason::kCancelled || pool_abort == AbortReason::kTimedOut) {
    return failed_result(pool_abort,
                         pool_abort == AbortReason::kCancelled
                             ? "native bulk build cancelled in leaf pool"
                             : "native bulk build timed out in leaf pool");
  }

  for (std::size_t block = 0U; block < descriptor.block_count(); ++block) {
    if (descriptor.block_kind[block] !=
        static_cast<std::uint8_t>(LogicalBlockKind::kLeaf)) {
      continue;
    }
    add_leaf_counters(repaired[block], &output.receipt_);
    const LeafBuildEvidence& evidence = leaf_evidence[block];
    output.receipt_.candidate_visited += evidence.candidate_visited;
    output.receipt_.candidate_evaluations +=
        evidence.candidate_distance_evaluations;
    output.receipt_.candidate_budget_exhausted_rows +=
        evidence.candidate_budget_exhausted_rows;
    output.receipt_.candidate_membership_exhausted_rows +=
        evidence.candidate_membership_exhausted_rows;
    output.receipt_.leaf_alpha_distance_evaluations +=
        evidence.alpha_distance_evaluations;
    output.receipt_.leaf_distance_evaluations += evidence.distance_evaluations;
    output.receipt_.leaf_alpha_rejected += evidence.alpha_rejected;
    output.receipt_.leaf_pruned_rows += evidence.pruned_rows;
    output.receipt_.leaf_pruned_edges += evidence.pruned_edges;
  }
  // The immutable V3 graph stores no parent-compose evidence.  Therefore an
  // aggregate alpha counter is exact only for a leaf-only descriptor; never
  // publish a leaf/repair subtotal as if it covered parent composition too.
  output.receipt_.alpha_aggregate_complete =
      output.receipt_.parent_count == 0U;
  if (output.receipt_.alpha_aggregate_complete) {
    output.receipt_.alpha_distance_evaluations =
        output.receipt_.leaf_alpha_distance_evaluations +
        output.receipt_.repair_alpha_distance_evaluations;
    output.receipt_.alpha_pruned_rows = output.receipt_.leaf_pruned_rows;
    output.receipt_.alpha_pruned_edges = output.receipt_.leaf_pruned_edges;
  } else {
    output.receipt_.alpha_distance_evaluations = 0U;
    output.receipt_.alpha_pruned_rows = 0U;
    output.receipt_.alpha_pruned_edges = 0U;
  }

  initial_abort = external_abort();
  if (initial_abort != AbortReason::kNone) {
    return failed_result(initial_abort,
                         initial_abort == AbortReason::kCancelled
                             ? "native bulk build cancelled before artifact assembly"
                             : "native bulk build timed out before artifact assembly");
  }

  std::vector<LogicalArtifactBlockInput> blocks;
  blocks.reserve(descriptor.block_count());
  for (std::size_t block = 0U; block < descriptor.block_count(); ++block) {
    LogicalArtifactBlockInput input;
    input.block_id = descriptor.block_id[block];
    input.level = descriptor.block_level[block];
    input.view_id = descriptor.block_view_id[block];
    input.component_sort_key = descriptor.block_component_sort_key[block];
    input.kind = static_cast<LogicalBlockKind>(descriptor.block_kind[block]);
    input.anchor = descriptor.block_anchor_id[block];
    input.members = copy_slice(descriptor.block_member_ids,
                               descriptor.block_member_ptr, block,
                               "block membership");
    input.entries = copy_slice(descriptor.block_entry_ids,
                               descriptor.block_entry_ptr, block, "block entries");
    input.children = copy_slice(resolved_children,
                                descriptor.block_child_ptr, block,
                                "block children");
    if (input.kind == LogicalBlockKind::kLeaf) {
      input.leaf_repair = std::move(repaired[block]);
    } else {
      input.parent_options.max_degree = options.max_degree;
      input.parent_options.alpha = options.alpha;
      input.parent_options.sample_count = options.sample_count;
      input.parent_options.seed = options.seed;
      input.parent_options.parent_identity = input.block_id;
      // Artifact V3 forwards this value to compose_parent.  Keep it one even
      // when the outer leaf pool has sixteen workers.
      input.parent_options.thread_count = 1U;
    }
    blocks.push_back(std::move(input));
  }

  LogicalArtifactBuildOptions artifact_options;
  artifact_options.thread_count = 1U;
  LogicalArtifactImage image = LogicalArtifactImage::build(
      corpus, routing, std::move(blocks), artifact_options);
  initial_abort = external_abort();
  if (initial_abort != AbortReason::kNone) {
    return failed_result(initial_abort,
                         initial_abort == AbortReason::kCancelled
                             ? "native bulk build cancelled after artifact assembly"
                             : "native bulk build timed out after artifact assembly");
  }

  LogicalArtifactIndex reopened = LogicalArtifactIndex::open(corpus, image);
  output.receipt_.reopened = true;
  output.receipt_.audited = true;

  const LogicalGraphStats stats = reopened.logical_view().stats();
  output.receipt_.logical_rows = stats.logical_rows;
  output.receipt_.logical_edges = stats.logical_edges;
  output.receipt_.stored_rows = stats.stored_rows;
  output.receipt_.stored_edges = stats.stored_edges;
  output.receipt_.parent_stored_rows = stats.parent_stored_rows;
  output.receipt_.parent_stored_edges = stats.parent_stored_edges;
  output.receipt_.clean_child_ref_rows = stats.clean_child_ref_rows;
  output.receipt_.child_ref_count = stats.child_ref_count;
  output.receipt_.projection_entries = stats.projection_entries;
  output.receipt_.physical_edge_count = stats.physical_edge_count;
  const auto& artifact_metadata = image.metadata();
  output.receipt_.schema = artifact_metadata.schema_version;
  output.receipt_.artifact_schema = artifact_metadata.schema_version;
  output.receipt_.logical_section_bytes = artifact_metadata.logical_section_bytes;
  output.receipt_.route_section_bytes = artifact_metadata.route_section_bytes;
  output.receipt_.routing_section_bytes = artifact_metadata.routing_section_bytes;
  output.receipt_.section_bytes = output.receipt_.logical_section_bytes +
                                  output.receipt_.route_section_bytes +
                                  output.receipt_.routing_section_bytes;
  output.receipt_.logical_sha256 = artifact_metadata.logical_sha256;
  output.receipt_.route_sha256 = artifact_metadata.route_sha256;
  output.receipt_.routing_sha256 = artifact_metadata.routing_sha256;
  output.receipt_.joint_sha256 = artifact_metadata.joint_sha256;
  output.receipt_.corpus_sha256 = corpus.identity().canonical_sha256().hex();
  output.receipt_.corpus_vectors_sha256 =
      corpus.identity().vectors().file_sha256().hex();
  output.receipt_.corpus_attrs_sha256 =
      corpus.identity().attrs_raw().file_sha256().hex();
  collect_parent_counters(reopened.logical_sections(), &output.receipt_);
  // Parent compose evidence is intentionally opaque in the frozen Artifact
  // V3 seam.  Do not label a leaf-only sum as a total; consumers can inspect
  // leaf_* fields and the explicit availability flags.
  output.receipt_.scc_count = 0U;
  output.receipt_.bridge_count = 0U;
  output.receipt_.sampled_bridge_distance_evaluations = 0U;
  output.receipt_.scc_count_complete = false;
  output.receipt_.bridge_count_complete = false;
  output.receipt_.sampled_bridge_distance_complete = false;

  initial_abort = external_abort();
  if (initial_abort != AbortReason::kNone) {
    return failed_result(initial_abort,
                         initial_abort == AbortReason::kCancelled
                             ? "native bulk build cancelled before query bind"
                             : "native bulk build timed out before query bind");
  }
  output.query_ = LogicalQueryHandle::bind(std::move(reopened));
  output.receipt_.query_bound = output.query_.valid();
  fill_identity(&output.receipt_);
  initial_abort = external_abort();
  if (initial_abort != AbortReason::kNone) {
    return failed_result(initial_abort,
                         initial_abort == AbortReason::kCancelled
                             ? "native bulk build cancelled after query bind"
                             : "native bulk build timed out after query bind");
  }
  output.receipt_.state = NativeBulkBuildState::kComplete;
  output.receipt_.complete_marker_published = false;
  return output;
}

}  // namespace bounded_blowup

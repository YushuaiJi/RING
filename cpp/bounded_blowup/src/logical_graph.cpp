// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#include "bounded_blowup/logical_graph.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace bounded_blowup {

namespace {

constexpr std::uint32_t kNoLocalRow = std::numeric_limits<std::uint32_t>::max();

using BlockKind = LogicalBlockKind;

struct ChildBinding {
  std::size_t child_index = 0U;
  std::vector<std::uint32_t> child_to_parent;
};

struct LogicalBlock {
  LogicalViewId view_id = 0U;
  std::uint64_t component_sort_key = 0U;
  BlockKind kind = BlockKind::kLeaf;
  std::uint32_t depth = 0U;
  std::vector<std::int64_t> members;
  std::uint32_t anchor_local_row = 0U;
  std::vector<std::int64_t> entries;

  // Leaf-only physical CSR. Targets are local rows, not corpus IDs.
  std::vector<std::uint64_t> leaf_row_ptr;
  std::vector<std::uint32_t> leaf_target_rows;
  std::vector<std::uint64_t> leaf_physical_edge_ids;

  // Parent-only logical sections. Ordinary rows have only these child refs;
  // dirty rows additionally have a replacement overlay.
  std::vector<ChildBinding> bindings;
  std::vector<std::uint64_t> owner_ptr;
  std::vector<std::uint32_t> owner_binding;
  std::vector<std::uint32_t> owner_child_row;
  std::vector<std::uint8_t> reason_mask;
  std::vector<std::uint32_t> overlay_source_rows;
  std::vector<std::uint32_t> overlay_index_by_source;
  std::vector<std::uint64_t> overlay_row_ptr;
  std::vector<std::uint32_t> overlay_target_rows;
  std::vector<std::uint64_t> overlay_physical_edge_ids;

  // Query-ready effective adjacency.  Parent child references and overlays
  // are resolved once when the immutable graph is opened/frozen.  Keeping the
  // result in CSR form removes recursive view resolution from neighbors(),
  // while preserving the resolver's exact target and physical-edge order.
  std::vector<std::uint64_t> effective_row_ptr;
  std::vector<std::uint32_t> effective_target_rows;
  std::vector<std::uint64_t> effective_physical_edge_ids;
};

}  // namespace

struct LogicalGraphStorage {
  LogicalGraphOptions options;
  std::uint64_t corpus_object_count = 0U;
  std::uint32_t corpus_vector_dim = 0U;
  bool corpus_finite_values_prevalidated = false;
  std::vector<LogicalBlock> blocks;
  std::vector<LogicalViewId> ordered_ids;
  std::vector<std::size_t> ordered_indices;
  LogicalGraphStats stats;
  LogicalSectionFingerprint fingerprint{};
  LogicalGraphSections sections;
};

struct LogicalGraphSectionStorage {
  LogicalGraphSectionData data;
  std::vector<std::uint8_t> canonical_bytes;
  LogicalSectionDigest digest{};
};

namespace {

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::atomic<std::uint64_t> logical_graph_sha256_calls{0U};
std::atomic<std::uint64_t> logical_graph_resolve_calls{0U};
#endif

std::string error_prefix(const char* prefix, const std::string& message) {
  return std::string(prefix) + message;
}

std::size_t find_member(const std::vector<std::int64_t>& members,
                        std::int64_t object_id) {
  const auto iterator =
      std::lower_bound(members.begin(), members.end(), object_id);
  if (iterator == members.end() || *iterator != object_id) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(iterator - members.begin());
}

void require_sorted_unique_members(const std::vector<std::int64_t>& members,
                                   const char* label) {
  if (members.empty()) {
    throw LogicalGraphError(std::string(label) + " must not be empty");
  }
  for (std::size_t row = 1U; row < members.size(); ++row) {
    if (members[row - 1U] >= members[row]) {
      throw LogicalGraphError(std::string(label) +
                              " must be strictly increasing");
    }
  }
  if (members.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw LogicalGraphError(std::string(label) +
                            " exceeds the frozen local-row range");
  }
}

std::uint32_t require_local_member(const std::vector<std::int64_t>& members,
                                   std::int64_t object_id,
                                   const char* label) {
  const std::size_t row = find_member(members, object_id);
  if (row == std::numeric_limits<std::size_t>::max()) {
    throw LogicalGraphError(std::string(label) + " is outside block members");
  }
  return static_cast<std::uint32_t>(row);
}

void validate_entries(const std::vector<std::int64_t>& members,
                      std::int64_t anchor,
                      const std::vector<std::int64_t>& entries) {
  (void)require_local_member(members, anchor, "anchor");
  for (std::size_t index = 0U; index < entries.size(); ++index) {
    (void)require_local_member(members, entries[index], "entry");
    if (index != 0U && entries[index - 1U] >= entries[index]) {
      throw LogicalGraphError("entries must be strictly increasing");
    }
  }
}

void checked_add(std::uint64_t* destination, std::uint64_t amount,
                 const char* label) {
  if (*destination > std::numeric_limits<std::uint64_t>::max() - amount) {
    throw LogicalGraphError(std::string("logical graph counter overflow: ") +
                            label);
  }
  *destination += amount;
}

std::size_t find_block_linear(const LogicalGraphStorage& storage,
                              LogicalViewId view_id) {
  for (std::size_t index = 0U; index < storage.blocks.size(); ++index) {
    if (storage.blocks[index].view_id == view_id) return index;
  }
  return std::numeric_limits<std::size_t>::max();
}

std::size_t find_block_ordered(const LogicalGraphStorage& storage,
                               LogicalViewId view_id) {
  const auto iterator =
      std::lower_bound(storage.ordered_ids.begin(), storage.ordered_ids.end(),
                       view_id);
  if (iterator == storage.ordered_ids.end() || *iterator != view_id) {
    throw LogicalGraphError("logical root/child view is outside frozen artifact");
  }
  const std::size_t ordered =
      static_cast<std::size_t>(iterator - storage.ordered_ids.begin());
  return storage.ordered_indices[ordered];
}

void append_unique(
    std::array<std::uint32_t, kLogicalMaxDegree>& values,
    std::array<std::uint64_t, kLogicalMaxDegree>& physical_edge_ids,
    std::uint16_t* count, std::uint32_t value,
    std::uint64_t physical_edge_id) {
  for (std::uint16_t index = 0U; index < *count; ++index) {
    if (values[index] == value) {
      if (physical_edge_ids[index] != physical_edge_id) {
        throw LogicalGraphError(
            "one logical edge resolved to conflicting physical identities");
      }
      return;
    }
  }
  if (*count >= static_cast<std::uint16_t>(kLogicalMaxDegree)) {
    throw LogicalGraphError("logical resolver produced degree greater than 24");
  }
  values[*count] = value;
  physical_edge_ids[*count] = physical_edge_id;
  ++(*count);
}

void resolve_frame(const LogicalGraphStorage& storage, std::size_t block_index,
                   std::uint32_t local_row, LogicalScratch& scratch,
                   std::uint32_t depth) {
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  logical_graph_resolve_calls.fetch_add(1U, std::memory_order_relaxed);
#endif
  if (depth > storage.options.max_depth || depth > kLogicalMaxDepth) {
    throw LogicalGraphError("logical graph query exceeded frozen max depth");
  }
  const LogicalBlock& block = storage.blocks[block_index];
  if (local_row >= block.members.size()) {
    throw LogicalGraphError("logical row is outside its block");
  }
  std::uint16_t& count = scratch.frame_counts[depth];
  count = 0U;
  std::array<std::uint32_t, kLogicalMaxDegree>& output =
      scratch.frame_rows[depth];
  std::array<std::uint64_t, kLogicalMaxDegree>& output_edge_ids =
      scratch.frame_physical_edge_ids[depth];
  if (block.kind == BlockKind::kLeaf) {
    const std::uint64_t begin = block.leaf_row_ptr[local_row];
    const std::uint64_t end = block.leaf_row_ptr[local_row + 1U];
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::size_t stored_offset = static_cast<std::size_t>(offset);
      const std::uint64_t edge_id = block.leaf_physical_edge_ids.empty()
                                        ? 0U
                                        : block.leaf_physical_edge_ids[stored_offset];
      append_unique(output, output_edge_ids, &count,
                    block.leaf_target_rows[stored_offset], edge_id);
    }
    return;
  }

  if (block.reason_mask[local_row] != 0U) {
    const std::uint32_t overlay_index =
        block.overlay_index_by_source[local_row];
    if (overlay_index == kNoLocalRow) {
      throw LogicalGraphError("dirty logical row has no replacement overlay");
    }
    const std::uint64_t begin = block.overlay_row_ptr[overlay_index];
    const std::uint64_t end = block.overlay_row_ptr[overlay_index + 1U];
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::size_t stored_offset = static_cast<std::size_t>(offset);
      const std::uint64_t edge_id = block.overlay_physical_edge_ids.empty()
                                        ? 0U
                                        : block.overlay_physical_edge_ids[stored_offset];
      append_unique(output, output_edge_ids, &count,
                    block.overlay_target_rows[stored_offset], edge_id);
    }
    return;
  }

  const std::uint64_t begin = block.owner_ptr[local_row];
  const std::uint64_t end = block.owner_ptr[local_row + 1U];
  if (begin == end) {
    throw LogicalGraphError("clean logical row has no child reference");
  }
  for (std::uint64_t offset = begin; offset < end; ++offset) {
    const std::size_t owner_offset = static_cast<std::size_t>(offset);
    const std::uint32_t binding_index = block.owner_binding[owner_offset];
    if (binding_index >= block.bindings.size()) {
      throw LogicalGraphError("logical owner binding is outside its block");
    }
    const ChildBinding& binding = block.bindings[binding_index];
    const std::uint32_t child_row = block.owner_child_row[owner_offset];
    if (child_row >= binding.child_to_parent.size()) {
      throw LogicalGraphError("logical child row is outside projection");
    }
    const std::size_t child_index = binding.child_index;
    resolve_frame(storage, child_index, child_row, scratch, depth + 1U);
    const std::uint16_t child_count = scratch.frame_counts[depth + 1U];
    for (std::uint16_t child_offset = 0U; child_offset < child_count;
         ++child_offset) {
      const std::uint32_t child_target =
          scratch.frame_rows[depth + 1U][child_offset];
      if (child_target >= binding.child_to_parent.size()) {
        throw LogicalGraphError("logical child target is outside projection");
      }
      append_unique(
          output, output_edge_ids, &count,
          binding.child_to_parent[child_target],
          scratch.frame_physical_edge_ids[depth + 1U][child_offset]);
    }
  }
}

void prepare_effective_adjacency(LogicalGraphStorage* storage) {
  if (storage == nullptr) {
    throw LogicalGraphError("cannot prepare an empty logical graph");
  }
  LogicalScratch scratch;
  for (std::size_t block_index = 0U;
       block_index < storage->blocks.size(); ++block_index) {
    LogicalBlock& block = storage->blocks[block_index];
    block.effective_row_ptr.clear();
    block.effective_target_rows.clear();
    block.effective_physical_edge_ids.clear();
    // Leaf CSR is already query-ready and is retained as the single stored
    // copy.  Only parent rows need a recursively prepared effective view.
    if (block.kind == BlockKind::kLeaf) continue;
    block.effective_row_ptr.reserve(block.members.size() + 1U);
    block.effective_row_ptr.push_back(0U);
    // Keep the prepared copy bounded for the common one-edge-per-row case;
    // rows wider than this grow geometrically while they are materialized.
    block.effective_target_rows.reserve(block.members.size());
    block.effective_physical_edge_ids.reserve(block.members.size());

    for (std::size_t row = 0U; row < block.members.size(); ++row) {
      resolve_frame(*storage, block_index, static_cast<std::uint32_t>(row),
                    scratch, 0U);
      const std::uint16_t count = scratch.frame_counts[0U];
      block.effective_target_rows.insert(
          block.effective_target_rows.end(), scratch.frame_rows[0U].begin(),
          scratch.frame_rows[0U].begin() + count);
      block.effective_physical_edge_ids.insert(
          block.effective_physical_edge_ids.end(),
          scratch.frame_physical_edge_ids[0U].begin(),
          scratch.frame_physical_edge_ids[0U].begin() + count);
      block.effective_row_ptr.push_back(
          static_cast<std::uint64_t>(block.effective_target_rows.size()));
    }
  }
}

std::vector<std::vector<std::uint32_t>> materialize_rows(
    const LogicalGraphStorage& storage, std::size_t block_index) {
  const LogicalBlock& block = storage.blocks[block_index];
  std::vector<std::vector<std::uint32_t>> rows(block.members.size());
  LogicalScratch scratch;
  for (std::size_t row = 0U; row < block.members.size(); ++row) {
    resolve_frame(storage, block_index, static_cast<std::uint32_t>(row), scratch,
                  0U);
    const std::uint16_t count = scratch.frame_counts[0U];
    rows[row].assign(scratch.frame_rows[0U].begin(),
                     scratch.frame_rows[0U].begin() + count);
  }
  return rows;
}

std::vector<std::uint64_t> make_row_ptr(
    const std::vector<std::vector<std::uint32_t>>& rows) {
  std::vector<std::uint64_t> row_ptr;
  row_ptr.reserve(rows.size() + 1U);
  row_ptr.push_back(0U);
  for (const auto& row : rows) {
    if (row_ptr.back() >
        std::numeric_limits<std::uint64_t>::max() - row.size()) {
      throw LogicalGraphError("materialized child CSR offset overflow");
    }
    row_ptr.push_back(row_ptr.back() + static_cast<std::uint64_t>(row.size()));
  }
  return row_ptr;
}

std::vector<std::uint64_t> make_witness(
    const std::vector<std::vector<std::uint32_t>>& rows,
    std::uint32_t anchor_row) {
  std::vector<std::uint64_t> witness(
      rows.size(), kNoWitnessParent);
  std::vector<unsigned char> reached(rows.size(), 0U);
  std::queue<std::uint32_t> pending;
  reached[anchor_row] = 1U;
  pending.push(anchor_row);
  std::size_t reached_count = 1U;
  while (!pending.empty()) {
    const std::uint32_t source = pending.front();
    pending.pop();
    for (const std::uint32_t target : rows[source]) {
      if (reached[target] == 0U) {
        reached[target] = 1U;
        witness[target] = source;
        ++reached_count;
        pending.push(target);
      }
    }
  }
  if (reached_count != rows.size()) {
    throw LogicalGraphError("child rows are not directed-reachable from anchor");
  }
  return witness;
}

struct MaterializedChild {
  std::size_t block_index = 0U;
  std::vector<std::vector<std::uint32_t>> rows;
  std::vector<std::uint64_t> row_ptr;
  std::vector<std::uint64_t> witness_parent_row;
  std::vector<std::uint64_t> target_rows;
};

void validate_directed_rows(
    const std::vector<std::vector<std::uint32_t>>& rows,
    std::uint32_t anchor_row, const char* label) {
  if (anchor_row >= rows.size()) {
    throw LogicalGraphError(std::string(label) + " anchor is outside rows");
  }
  std::vector<unsigned char> reached(rows.size(), 0U);
  std::queue<std::uint32_t> pending;
  reached[anchor_row] = 1U;
  pending.push(anchor_row);
  std::size_t reached_count = 1U;
  while (!pending.empty()) {
    const std::uint32_t source = pending.front();
    pending.pop();
    for (const std::uint32_t target : rows[source]) {
      if (target >= rows.size()) {
        throw LogicalGraphError(std::string(label) + " target is outside rows");
      }
      if (reached[target] == 0U) {
        reached[target] = 1U;
        ++reached_count;
        pending.push(target);
      }
    }
  }
  if (reached_count != rows.size()) {
    throw LogicalGraphError(std::string(label) + " failed directed audit");
  }
}

void validate_leaf_block(const LogicalBlock& block) {
  require_sorted_unique_members(block.members, "leaf members");
  if (block.anchor_local_row >= block.members.size()) {
    throw LogicalGraphError("leaf anchor is outside members");
  }
  if (block.leaf_row_ptr.size() != block.members.size() + 1U ||
      block.leaf_row_ptr.empty() || block.leaf_row_ptr.front() != 0U ||
      block.leaf_row_ptr.back() != block.leaf_target_rows.size()) {
    throw LogicalGraphError("leaf CSR shape is invalid");
  }
  if (!block.leaf_physical_edge_ids.empty() &&
      block.leaf_physical_edge_ids.size() != block.leaf_target_rows.size()) {
    throw LogicalGraphError("leaf physical-edge section has invalid shape");
  }
  std::vector<std::vector<std::uint32_t>> rows(block.members.size());
  for (std::size_t row = 0U; row < block.members.size(); ++row) {
    const std::uint64_t begin = block.leaf_row_ptr[row];
    const std::uint64_t end = block.leaf_row_ptr[row + 1U];
    if (begin > end || end > block.leaf_target_rows.size() ||
        end - begin > kLogicalMaxDegree) {
      throw LogicalGraphError("leaf row exceeds degree 24");
    }
    std::set<std::uint32_t> seen;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::uint32_t target =
          block.leaf_target_rows[static_cast<std::size_t>(offset)];
      if (target >= block.members.size() || target == row ||
          !seen.insert(target).second) {
        throw LogicalGraphError("leaf row has duplicate/self/outside target");
      }
      rows[row].push_back(target);
    }
  }
  validate_directed_rows(rows, block.anchor_local_row, "leaf graph");
  if (!block.bindings.empty() || !block.owner_ptr.empty() ||
      !block.overlay_target_rows.empty() ||
      !block.overlay_physical_edge_ids.empty()) {
    throw LogicalGraphError("leaf carries parent-only logical sections");
  }
}

void validate_parent_block(const LogicalGraphStorage& storage,
                           std::size_t block_index) {
  const LogicalBlock& block = storage.blocks[block_index];
  require_sorted_unique_members(block.members, "parent members");
  if (block.anchor_local_row >= block.members.size()) {
    throw LogicalGraphError("parent anchor is outside members");
  }
  if (!block.leaf_row_ptr.empty() || !block.leaf_target_rows.empty() ||
      !block.leaf_physical_edge_ids.empty()) {
    throw LogicalGraphError("ordinary parent rows were stored as full CSR");
  }
  if (block.bindings.empty()) {
    throw LogicalGraphError("parent has no child bindings");
  }
  std::set<LogicalViewId> child_ids;
  for (const ChildBinding& binding : block.bindings) {
    if (binding.child_index >= block_index ||
        binding.child_index >= storage.blocks.size()) {
      throw LogicalGraphError("parent has future/out-of-artifact child ref");
    }
    const LogicalBlock& child = storage.blocks[binding.child_index];
    if (!child_ids.insert(child.view_id).second) {
      throw LogicalGraphError("parent has duplicate child ref");
    }
    if (binding.child_to_parent.size() != child.members.size()) {
      throw LogicalGraphError("child-local projection has the wrong length");
    }
    for (std::size_t child_row = 0U; child_row < child.members.size();
         ++child_row) {
      const std::uint32_t parent_row = binding.child_to_parent[child_row];
      if (parent_row >= block.members.size() ||
          block.members[parent_row] != child.members[child_row]) {
        throw LogicalGraphError("child-local projection has wrong target");
      }
    }
  }
  if (block.owner_ptr.size() != block.members.size() + 1U ||
      block.owner_ptr.front() != 0U ||
      block.owner_ptr.back() != block.owner_binding.size() ||
      block.owner_binding.size() != block.owner_child_row.size()) {
    throw LogicalGraphError("parent owner section has invalid shape");
  }
  if (block.reason_mask.size() != block.members.size() ||
      block.overlay_index_by_source.size() != block.members.size()) {
    throw LogicalGraphError("parent reason section has invalid shape");
  }
  for (std::size_t row = 0U; row < block.members.size(); ++row) {
    const std::uint64_t begin = block.owner_ptr[row];
    const std::uint64_t end = block.owner_ptr[row + 1U];
    if (begin >= end) {
      throw LogicalGraphError("parent row has no child reference");
    }
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::size_t owner_offset = static_cast<std::size_t>(offset);
      const std::uint32_t binding_index = block.owner_binding[owner_offset];
      if (binding_index >= block.bindings.size()) {
        throw LogicalGraphError("parent owner ref has invalid binding");
      }
      const ChildBinding& binding = block.bindings[binding_index];
      const std::uint32_t child_row = block.owner_child_row[owner_offset];
      if (child_row >= binding.child_to_parent.size() ||
          binding.child_to_parent[child_row] != row) {
        throw LogicalGraphError("parent owner ref has invalid projection");
      }
    }
  }
  if (block.overlay_row_ptr.size() != block.overlay_source_rows.size() + 1U ||
      block.overlay_row_ptr.empty() || block.overlay_row_ptr.front() != 0U ||
      block.overlay_row_ptr.back() != block.overlay_target_rows.size()) {
    throw LogicalGraphError("parent overlay CSR has invalid shape");
  }
  if (!block.overlay_physical_edge_ids.empty() &&
      block.overlay_physical_edge_ids.size() !=
          block.overlay_target_rows.size()) {
    throw LogicalGraphError("parent overlay physical-edge shape is invalid");
  }
  std::set<std::uint32_t> overlay_sources;
  for (std::size_t overlay = 0U;
       overlay < block.overlay_source_rows.size(); ++overlay) {
    const std::uint32_t source = block.overlay_source_rows[overlay];
    if (source >= block.members.size() ||
        !overlay_sources.insert(source).second ||
        block.reason_mask[source] == 0U ||
        block.overlay_index_by_source[source] != overlay) {
      throw LogicalGraphError("parent overlay source/reason mismatch");
    }
    const std::uint64_t begin = block.overlay_row_ptr[overlay];
    const std::uint64_t end = block.overlay_row_ptr[overlay + 1U];
    if (begin > end || end > block.overlay_target_rows.size() ||
        end - begin > kLogicalMaxDegree) {
      throw LogicalGraphError("parent overlay row exceeds degree 24");
    }
    std::set<std::uint32_t> seen;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::uint32_t target =
          block.overlay_target_rows[static_cast<std::size_t>(offset)];
      if (target >= block.members.size() || target == source ||
          !seen.insert(target).second) {
        throw LogicalGraphError(
            "parent overlay has duplicate/self/outside target");
      }
    }
  }
  for (std::size_t row = 0U; row < block.members.size(); ++row) {
    if (block.reason_mask[row] == 0U) {
      if (block.overlay_index_by_source[row] != kNoLocalRow) {
        throw LogicalGraphError("clean parent row has an overlay reason missing");
      }
    } else if (block.overlay_index_by_source[row] == kNoLocalRow) {
      throw LogicalGraphError("dirty parent row is missing its overlay");
    }
  }
}

void validate_storage_shape(const LogicalGraphStorage& storage) {
  if (storage.options.max_depth == 0U ||
      storage.options.max_depth > kLogicalMaxDepth) {
    throw LogicalGraphError("frozen logical max depth exceeds scratch layout");
  }
  if (storage.blocks.empty()) {
    throw LogicalGraphError("logical graph artifact has no blocks");
  }
  std::set<LogicalViewId> ids;
  for (std::size_t index = 0U; index < storage.blocks.size(); ++index) {
    const LogicalBlock& block = storage.blocks[index];
    if (!ids.insert(block.view_id).second) {
      throw LogicalGraphError("logical view IDs must be unique");
    }
    if (block.depth > storage.options.max_depth) {
      throw LogicalGraphError("logical graph exceeds frozen max depth");
    }
    if (block.kind == BlockKind::kLeaf) {
      validate_leaf_block(block);
    } else {
      validate_parent_block(storage, index);
      std::uint32_t max_child_depth = 0U;
      for (const ChildBinding& binding : block.bindings) {
        max_child_depth =
            std::max(max_child_depth,
                     storage.blocks[binding.child_index].depth);
      }
      if (block.depth != max_child_depth + 1U) {
        throw LogicalGraphError("parent depth is inconsistent with child DAG");
      }
    }
  }
  if (storage.ordered_ids.size() != storage.blocks.size() ||
      storage.ordered_indices.size() != storage.blocks.size()) {
    throw LogicalGraphError("logical view index section has invalid shape");
  }
  for (std::size_t order = 0U; order < storage.ordered_ids.size(); ++order) {
    if (storage.ordered_indices[order] >= storage.blocks.size() ||
        storage.blocks[storage.ordered_indices[order]].view_id !=
            storage.ordered_ids[order]) {
      throw LogicalGraphError("logical view index section is inconsistent");
    }
    if (order != 0U && storage.ordered_ids[order - 1U] >=
                           storage.ordered_ids[order]) {
      throw LogicalGraphError("logical view index section is not sorted");
    }
  }
}

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t amount) noexcept {
  return (value >> amount) | (value << (32U - amount));
}

LogicalSectionDigest sha256_bytes(
    const std::vector<std::uint8_t>& input) {
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  logical_graph_sha256_calls.fetch_add(1U, std::memory_order_relaxed);
#endif
  if (input.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max() / 8U)) {
    throw LogicalGraphError("logical section is too large for SHA-256");
  }
  std::vector<std::uint8_t> message(input);
  message.push_back(0x80U);
  while (message.size() % 64U != 56U) message.push_back(0U);
  const std::uint64_t bit_length =
      static_cast<std::uint64_t>(input.size()) * 8U;
  for (std::uint32_t shift = 56U;; shift -= 8U) {
    message.push_back(
        static_cast<std::uint8_t>((bit_length >> shift) & UINT64_C(0xff)));
    if (shift == 0U) break;
  }

  static constexpr std::uint32_t constants[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::array<std::uint32_t, 8> state{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  for (std::size_t block = 0U; block < message.size(); block += 64U) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t word = 0U; word < 16U; ++word) {
      const std::size_t offset = block + word * 4U;
      schedule[word] =
          (static_cast<std::uint32_t>(message[offset]) << 24U) |
          (static_cast<std::uint32_t>(message[offset + 1U]) << 16U) |
          (static_cast<std::uint32_t>(message[offset + 2U]) << 8U) |
          static_cast<std::uint32_t>(message[offset + 3U]);
    }
    for (std::size_t word = 16U; word < schedule.size(); ++word) {
      const std::uint32_t s0 = rotate_right(schedule[word - 15U], 7U) ^
                               rotate_right(schedule[word - 15U], 18U) ^
                               (schedule[word - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(schedule[word - 2U], 17U) ^
                               rotate_right(schedule[word - 2U], 19U) ^
                               (schedule[word - 2U] >> 10U);
      schedule[word] = schedule[word - 16U] + s0 +
                       schedule[word - 7U] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t round = 0U; round < schedule.size(); ++round) {
      const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                 rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sum1 + choice + constants[round] + schedule[round];
      const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                 rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  LogicalSectionDigest result{};
  for (std::size_t word = 0U; word < state.size(); ++word) {
    for (std::size_t byte = 0U; byte < 4U; ++byte) {
      result[word * 4U + byte] = static_cast<std::uint8_t>(
          (state[word] >> static_cast<std::uint32_t>(24U - byte * 8U)) &
          0xffU);
    }
  }
  return result;
}

void append_u8(std::vector<std::uint8_t>* bytes, std::uint8_t value) {
  bytes->push_back(value);
}

void append_u32(std::vector<std::uint8_t>* bytes, std::uint32_t value) {
  for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
    bytes->push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void append_u64(std::vector<std::uint8_t>* bytes, std::uint64_t value) {
  for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
    bytes->push_back(static_cast<std::uint8_t>(
        (value >> shift) & UINT64_C(0xff)));
  }
}

void append_i64(std::vector<std::uint8_t>* bytes, std::int64_t value) {
  append_u64(bytes, static_cast<std::uint64_t>(value));
}

template <typename T>
void append_vector(std::vector<std::uint8_t>* bytes,
                   const std::vector<T>& values) {
  append_u64(bytes, static_cast<std::uint64_t>(values.size()));
  for (const T value : values) {
    if constexpr (std::is_same<T, std::uint8_t>::value) {
      append_u8(bytes, value);
    } else if constexpr (std::is_same<T, std::uint32_t>::value) {
      append_u32(bytes, value);
    } else if constexpr (std::is_same<T, std::uint64_t>::value) {
      append_u64(bytes, value);
    } else if constexpr (std::is_same<T, std::int64_t>::value) {
      append_i64(bytes, value);
    }
  }
}

std::vector<std::uint8_t> encode_section_data(
    const LogicalGraphSectionData& data) {
  std::vector<std::uint8_t> bytes;
  static constexpr std::uint8_t magic[8] = {'R', 'F', 'B', 'B', 'L', 'G',
                                             '0', '1'};
  bytes.insert(bytes.end(), std::begin(magic), std::end(magic));
  append_u32(&bytes, data.schema_version);
  append_u32(&bytes, data.max_depth);
  append_u64(&bytes, data.corpus_object_count);
  append_u32(&bytes, data.corpus_vector_dim);
  append_u8(&bytes, data.corpus_finite_values_prevalidated);
  append_vector(&bytes, data.block_view_id);
  append_vector(&bytes, data.block_component_sort_key);
  append_vector(&bytes, data.block_kind);
  append_vector(&bytes, data.block_depth);
  append_vector(&bytes, data.block_anchor_local_row);
  append_vector(&bytes, data.block_member_ptr);
  append_vector(&bytes, data.member_ids);
  append_vector(&bytes, data.block_entry_ptr);
  append_vector(&bytes, data.entry_ids);
  append_vector(&bytes, data.block_binding_ptr);
  append_vector(&bytes, data.binding_child_view_id);
  append_vector(&bytes, data.binding_projection_ptr);
  append_vector(&bytes, data.projection_parent_row);
  append_vector(&bytes, data.row_owner_ptr);
  append_vector(&bytes, data.owner_binding);
  append_vector(&bytes, data.owner_child_row);
  append_vector(&bytes, data.row_kind);
  append_vector(&bytes, data.row_reason_mask);
  append_vector(&bytes, data.stored_source_global_row);
  append_vector(&bytes, data.stored_row_ptr);
  append_vector(&bytes, data.stored_target_local_row);
  append_vector(&bytes, data.stored_physical_edge_id);
  append_vector(&bytes, data.physical_edge_src);
  append_vector(&bytes, data.physical_edge_dst);
  return bytes;
}

class SectionReader {
 public:
  explicit SectionReader(const std::vector<std::uint8_t>& bytes)
      : bytes_(bytes) {}

  std::uint8_t read_u8() {
    require_bytes(1U);
    return bytes_[position_++];
  }

  std::uint32_t read_u32() {
    require_bytes(4U);
    std::uint32_t value = 0U;
    for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
      value |= static_cast<std::uint32_t>(bytes_[position_++]) << (byte * 8U);
    }
    return value;
  }

  std::uint64_t read_u64() {
    require_bytes(8U);
    std::uint64_t value = 0U;
    for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
      value |= static_cast<std::uint64_t>(bytes_[position_++]) << (byte * 8U);
    }
    return value;
  }

  std::int64_t read_i64() {
    const std::uint64_t raw = read_u64();
    if (raw <= static_cast<std::uint64_t>(
                   std::numeric_limits<std::int64_t>::max())) {
      return static_cast<std::int64_t>(raw);
    }
    const std::uint64_t magnitude = (~raw) + 1U;
    if (magnitude == (UINT64_C(1) << 63U)) {
      return std::numeric_limits<std::int64_t>::min();
    }
    return -static_cast<std::int64_t>(magnitude);
  }

  template <typename T>
  std::vector<T> read_vector() {
    const std::uint64_t count64 = read_u64();
    if (count64 > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
      throw LogicalGraphError("logical section vector exceeds host size_t");
    }
    const std::size_t count = static_cast<std::size_t>(count64);
    constexpr std::size_t width = sizeof(T);
    if (count > remaining() / width) {
      throw LogicalGraphError("logical section vector is truncated");
    }
    std::vector<T> values;
    values.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
      if constexpr (std::is_same<T, std::uint8_t>::value) {
        values.push_back(read_u8());
      } else if constexpr (std::is_same<T, std::uint32_t>::value) {
        values.push_back(read_u32());
      } else if constexpr (std::is_same<T, std::uint64_t>::value) {
        values.push_back(read_u64());
      } else if constexpr (std::is_same<T, std::int64_t>::value) {
        values.push_back(read_i64());
      }
    }
    return values;
  }

  std::size_t remaining() const noexcept { return bytes_.size() - position_; }

 private:
  void require_bytes(std::size_t count) const {
    if (count > remaining()) {
      throw LogicalGraphError("logical section byte image is truncated");
    }
  }

  const std::vector<std::uint8_t>& bytes_;
  std::size_t position_ = 0U;
};

LogicalGraphSectionData decode_section_data(
    const std::vector<std::uint8_t>& bytes) {
  SectionReader reader(bytes);
  static constexpr std::uint8_t magic[8] = {'R', 'F', 'B', 'B', 'L', 'G',
                                             '0', '1'};
  for (const std::uint8_t expected : magic) {
    if (reader.read_u8() != expected) {
      throw LogicalGraphError("logical section magic is invalid");
    }
  }
  LogicalGraphSectionData data;
  data.schema_version = reader.read_u32();
  if (data.schema_version != kLogicalSectionSchemaVersion) {
    throw LogicalGraphError("logical section schema version is unsupported");
  }
  data.max_depth = reader.read_u32();
  data.corpus_object_count = reader.read_u64();
  data.corpus_vector_dim = reader.read_u32();
  data.corpus_finite_values_prevalidated = reader.read_u8();
  data.block_view_id = reader.read_vector<std::uint32_t>();
  data.block_component_sort_key = reader.read_vector<std::uint64_t>();
  data.block_kind = reader.read_vector<std::uint8_t>();
  data.block_depth = reader.read_vector<std::uint32_t>();
  data.block_anchor_local_row = reader.read_vector<std::uint32_t>();
  data.block_member_ptr = reader.read_vector<std::uint64_t>();
  data.member_ids = reader.read_vector<std::int64_t>();
  data.block_entry_ptr = reader.read_vector<std::uint64_t>();
  data.entry_ids = reader.read_vector<std::int64_t>();
  data.block_binding_ptr = reader.read_vector<std::uint64_t>();
  data.binding_child_view_id = reader.read_vector<std::uint32_t>();
  data.binding_projection_ptr = reader.read_vector<std::uint64_t>();
  data.projection_parent_row = reader.read_vector<std::uint32_t>();
  data.row_owner_ptr = reader.read_vector<std::uint64_t>();
  data.owner_binding = reader.read_vector<std::uint32_t>();
  data.owner_child_row = reader.read_vector<std::uint32_t>();
  data.row_kind = reader.read_vector<std::uint8_t>();
  data.row_reason_mask = reader.read_vector<std::uint8_t>();
  data.stored_source_global_row = reader.read_vector<std::uint64_t>();
  data.stored_row_ptr = reader.read_vector<std::uint64_t>();
  data.stored_target_local_row = reader.read_vector<std::uint32_t>();
  data.stored_physical_edge_id = reader.read_vector<std::uint64_t>();
  data.physical_edge_src = reader.read_vector<std::int64_t>();
  data.physical_edge_dst = reader.read_vector<std::int64_t>();
  if (reader.remaining() != 0U) {
    throw LogicalGraphError("logical section byte image has trailing data");
  }
  return data;
}

LogicalGraphStats calculate_stats(const LogicalGraphStorage& storage) {
  LogicalGraphStats result;
  result.block_count = static_cast<std::uint64_t>(storage.blocks.size());
  LogicalScratch scratch;
  for (std::size_t block_index = 0U; block_index < storage.blocks.size();
       ++block_index) {
    const LogicalBlock& block = storage.blocks[block_index];
    if (block.kind == BlockKind::kLeaf) {
      ++result.leaf_count;
      checked_add(&result.stored_rows,
                  static_cast<std::uint64_t>(block.members.size()),
                  "stored_rows");
      checked_add(&result.stored_edges,
                  static_cast<std::uint64_t>(block.leaf_target_rows.size()),
                  "stored_edges");
    } else {
      ++result.parent_count;
      checked_add(&result.child_ref_count,
                  static_cast<std::uint64_t>(block.owner_binding.size()),
                  "child_ref_count");
      checked_add(&result.projection_entries,
                  [&block] {
                    std::uint64_t count = 0U;
                    for (const ChildBinding& binding : block.bindings) {
                      count += static_cast<std::uint64_t>(
                          binding.child_to_parent.size());
                    }
                    return count;
                  }(),
                  "projection_entries");
      for (const std::uint8_t reason : block.reason_mask) {
        if (reason == 0U) {
          ++result.clean_child_ref_rows;
        } else {
          ++result.dirty_overlay_rows;
        }
      }
      checked_add(&result.stored_rows,
                  static_cast<std::uint64_t>(block.overlay_source_rows.size()),
                  "stored_rows");
      checked_add(&result.stored_edges,
                  static_cast<std::uint64_t>(block.overlay_target_rows.size()),
                  "stored_edges");
      checked_add(&result.parent_stored_rows,
                  static_cast<std::uint64_t>(block.overlay_source_rows.size()),
                  "parent_stored_rows");
      checked_add(&result.parent_stored_edges,
                  static_cast<std::uint64_t>(block.overlay_target_rows.size()),
                  "parent_stored_edges");
    }
    checked_add(&result.logical_rows,
                static_cast<std::uint64_t>(block.members.size()),
                "logical_rows");

    // Resolving each effective row once supplies both the logical-edge stats
    // and the directed reachability proof. Keeping these audits in one BFS is
    // important for clean parents, whose rows recursively expand to children.
    std::vector<unsigned char> reached(block.members.size(), 0U);
    std::queue<std::uint32_t> pending;
    reached[block.anchor_local_row] = 1U;
    pending.push(block.anchor_local_row);
    std::size_t reached_count = 1U;
    while (!pending.empty()) {
      const std::uint32_t source = pending.front();
      pending.pop();
      resolve_frame(storage, block_index, source, scratch, 0U);
      const std::uint16_t count = scratch.frame_counts[0U];
      checked_add(&result.logical_edges,
                  static_cast<std::uint64_t>(count),
                  "logical_edges");
      for (std::uint16_t offset = 0U; offset < count; ++offset) {
        const std::uint32_t target = scratch.frame_rows[0U][offset];
        if (target >= block.members.size()) {
          throw LogicalGraphError("logical resolver returned an outside target");
        }
        if (reached[target] == 0U) {
          reached[target] = 1U;
          ++reached_count;
          pending.push(target);
        }
      }
    }
    if (reached_count != block.members.size()) {
      throw LogicalGraphError("logical graph failed directed reachability audit");
    }
  }
  return result;
}

void prepare_index_and_validate(LogicalGraphStorage* storage) {
  storage->ordered_ids.clear();
  storage->ordered_indices.clear();
  storage->ordered_ids.reserve(storage->blocks.size());
  storage->ordered_indices.reserve(storage->blocks.size());
  std::vector<std::size_t> indices(storage->blocks.size());
  for (std::size_t index = 0U; index < indices.size(); ++index) indices[index] = index;
  std::sort(indices.begin(), indices.end(), [&](std::size_t lhs, std::size_t rhs) {
    return storage->blocks[lhs].view_id < storage->blocks[rhs].view_id;
  });
  for (const std::size_t index : indices) {
    storage->ordered_indices.push_back(index);
    storage->ordered_ids.push_back(storage->blocks[index].view_id);
  }
  validate_storage_shape(*storage);
  // Build a read-only per-block effective CSR once.  This is intentionally
  // after all block/binding validation, so query traversal only reads frozen
  // arrays and cannot observe partially prepared parent state.
  prepare_effective_adjacency(storage);
  storage->stats = calculate_stats(*storage);
}

LogicalGraphSectionData make_section_data(
    const LogicalGraphStorage& storage, const CorpusView& corpus) {
  LogicalGraphSectionData data;
  data.max_depth = storage.options.max_depth;
  data.corpus_object_count = corpus.object_count;
  data.corpus_vector_dim = corpus.vector_dim;
  data.corpus_finite_values_prevalidated =
      corpus.finite_values_prevalidated ? 1U : 0U;

  std::vector<std::size_t> canonical_indices(storage.blocks.size());
  for (std::size_t index = 0U; index < canonical_indices.size(); ++index) {
    canonical_indices[index] = index;
  }
  std::sort(canonical_indices.begin(), canonical_indices.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              const LogicalBlock& left = storage.blocks[lhs];
              const LogicalBlock& right = storage.blocks[rhs];
              return std::tie(left.depth, left.view_id) <
                     std::tie(right.depth, right.view_id);
            });

  data.block_member_ptr.push_back(0U);
  data.block_entry_ptr.push_back(0U);
  data.block_binding_ptr.push_back(0U);
  data.binding_projection_ptr.push_back(0U);
  data.row_owner_ptr.push_back(0U);
  data.stored_row_ptr.push_back(0U);
  std::vector<std::pair<std::int64_t, std::int64_t>> stored_edge_pairs;

  for (const std::size_t block_index : canonical_indices) {
    const LogicalBlock& block = storage.blocks[block_index];
    data.block_view_id.push_back(block.view_id);
    data.block_component_sort_key.push_back(block.component_sort_key);
    data.block_kind.push_back(static_cast<std::uint8_t>(block.kind));
    data.block_depth.push_back(block.depth);
    data.block_anchor_local_row.push_back(block.anchor_local_row);
    data.member_ids.insert(data.member_ids.end(), block.members.begin(),
                           block.members.end());
    data.block_member_ptr.push_back(
        static_cast<std::uint64_t>(data.member_ids.size()));
    data.entry_ids.insert(data.entry_ids.end(), block.entries.begin(),
                          block.entries.end());
    data.block_entry_ptr.push_back(
        static_cast<std::uint64_t>(data.entry_ids.size()));

    const std::size_t binding_base = data.binding_child_view_id.size();
    for (const ChildBinding& binding : block.bindings) {
      data.binding_child_view_id.push_back(
          storage.blocks[binding.child_index].view_id);
      data.projection_parent_row.insert(data.projection_parent_row.end(),
                                        binding.child_to_parent.begin(),
                                        binding.child_to_parent.end());
      data.binding_projection_ptr.push_back(
          static_cast<std::uint64_t>(data.projection_parent_row.size()));
    }
    data.block_binding_ptr.push_back(
        static_cast<std::uint64_t>(data.binding_child_view_id.size()));

    const std::uint64_t global_row_base =
        data.block_member_ptr[data.block_member_ptr.size() - 2U];
    for (std::size_t row = 0U; row < block.members.size(); ++row) {
      if (block.kind == BlockKind::kLeaf) {
        data.row_kind.push_back(
            static_cast<std::uint8_t>(LogicalRowKind::kStored));
        data.row_reason_mask.push_back(0U);
        data.row_owner_ptr.push_back(
            static_cast<std::uint64_t>(data.owner_binding.size()));
      } else {
        const std::uint64_t owner_begin = block.owner_ptr[row];
        const std::uint64_t owner_end = block.owner_ptr[row + 1U];
        for (std::uint64_t offset = owner_begin; offset < owner_end; ++offset) {
          const std::size_t owner_offset = static_cast<std::size_t>(offset);
          const std::size_t global_binding =
              binding_base + block.owner_binding[owner_offset];
          if (global_binding > static_cast<std::size_t>(
                                   std::numeric_limits<std::uint32_t>::max())) {
            throw LogicalGraphError(
                "logical section owner binding exceeds uint32 range");
          }
          data.owner_binding.push_back(
              static_cast<std::uint32_t>(global_binding));
          data.owner_child_row.push_back(block.owner_child_row[owner_offset]);
        }
        data.row_owner_ptr.push_back(
            static_cast<std::uint64_t>(data.owner_binding.size()));
        const std::uint8_t reason = block.reason_mask[row];
        data.row_reason_mask.push_back(reason);
        data.row_kind.push_back(static_cast<std::uint8_t>(
            reason == 0U ? LogicalRowKind::kChildRef
                         : LogicalRowKind::kOverlay));
      }

      bool stored = block.kind == BlockKind::kLeaf;
      std::uint64_t begin = 0U;
      std::uint64_t end = 0U;
      const std::vector<std::uint32_t>* targets = nullptr;
      if (block.kind == BlockKind::kLeaf) {
        begin = block.leaf_row_ptr[row];
        end = block.leaf_row_ptr[row + 1U];
        targets = &block.leaf_target_rows;
      } else if (block.reason_mask[row] != 0U) {
        stored = true;
        const std::uint32_t overlay = block.overlay_index_by_source[row];
        if (overlay == kNoLocalRow) {
          throw LogicalGraphError(
              "dirty builder row has no replacement overlay");
        }
        begin = block.overlay_row_ptr[overlay];
        end = block.overlay_row_ptr[overlay + 1U];
        targets = &block.overlay_target_rows;
      }
      if (stored) {
        data.stored_source_global_row.push_back(
            global_row_base + static_cast<std::uint64_t>(row));
        for (std::uint64_t offset = begin; offset < end; ++offset) {
          const std::uint32_t target =
              (*targets)[static_cast<std::size_t>(offset)];
          data.stored_target_local_row.push_back(target);
          stored_edge_pairs.emplace_back(block.members[row],
                                         block.members[target]);
        }
        data.stored_row_ptr.push_back(
            static_cast<std::uint64_t>(
                data.stored_target_local_row.size()));
      }
    }
  }

  std::set<std::pair<std::int64_t, std::int64_t>> unique_edges(
      stored_edge_pairs.begin(), stored_edge_pairs.end());
  std::map<std::pair<std::int64_t, std::int64_t>, std::uint64_t> edge_ids;
  for (const auto& edge : unique_edges) {
    const std::uint64_t edge_id =
        static_cast<std::uint64_t>(data.physical_edge_src.size());
    edge_ids.emplace(edge, edge_id);
    data.physical_edge_src.push_back(edge.first);
    data.physical_edge_dst.push_back(edge.second);
  }
  data.stored_physical_edge_id.reserve(stored_edge_pairs.size());
  for (const auto& edge : stored_edge_pairs) {
    data.stored_physical_edge_id.push_back(edge_ids.at(edge));
  }
  return data;
}

void validate_pointer_section(const std::vector<std::uint64_t>& pointers,
                              std::size_t segment_count,
                              std::size_t terminal,
                              const char* label) {
  if (pointers.size() != segment_count + 1U || pointers.empty() ||
      pointers.front() != 0U ||
      pointers.back() != static_cast<std::uint64_t>(terminal)) {
    throw LogicalGraphError(std::string(label) + " has invalid shape");
  }
  for (std::size_t index = 1U; index < pointers.size(); ++index) {
    if (pointers[index - 1U] > pointers[index] ||
        pointers[index] > static_cast<std::uint64_t>(terminal)) {
      throw LogicalGraphError(std::string(label) + " is not monotone");
    }
  }
}

std::ptrdiff_t checked_iterator_offset(std::size_t offset,
                                       const char* label) {
  if (offset > static_cast<std::size_t>(
                   std::numeric_limits<std::ptrdiff_t>::max())) {
    throw LogicalGraphError(std::string(label) +
                            " exceeds iterator difference range");
  }
  return static_cast<std::ptrdiff_t>(offset);
}

std::shared_ptr<LogicalGraphStorage> storage_from_section_data(
    const LogicalGraphSectionData& data) {
  if (data.schema_version != kLogicalSectionSchemaVersion) {
    throw LogicalGraphError("logical section schema version is unsupported");
  }
  if (data.max_depth == 0U || data.max_depth > kLogicalMaxDepth) {
    throw LogicalGraphError("logical section max depth is invalid");
  }
  if (data.corpus_object_count == 0U || data.corpus_vector_dim == 0U ||
      data.corpus_finite_values_prevalidated != 1U) {
    throw LogicalGraphError("logical section corpus identity is invalid");
  }
  const std::size_t block_count = data.block_view_id.size();
  if (block_count == 0U || data.block_component_sort_key.size() != block_count ||
      data.block_kind.size() != block_count ||
      data.block_depth.size() != block_count ||
      data.block_anchor_local_row.size() != block_count) {
    throw LogicalGraphError("logical section block columns have invalid shape");
  }
  validate_pointer_section(data.block_member_ptr, block_count,
                           data.member_ids.size(), "block-member pointers");
  validate_pointer_section(data.block_entry_ptr, block_count,
                           data.entry_ids.size(), "block-entry pointers");
  validate_pointer_section(data.block_binding_ptr, block_count,
                           data.binding_child_view_id.size(),
                           "block-binding pointers");
  const std::size_t binding_count = data.binding_child_view_id.size();
  validate_pointer_section(data.binding_projection_ptr, binding_count,
                           data.projection_parent_row.size(),
                           "binding-projection pointers");
  const std::size_t row_count = data.member_ids.size();
  validate_pointer_section(data.row_owner_ptr, row_count,
                           data.owner_binding.size(), "row-owner pointers");
  if (data.owner_binding.size() != data.owner_child_row.size() ||
      data.row_kind.size() != row_count ||
      data.row_reason_mask.size() != row_count) {
    throw LogicalGraphError("logical section row columns have invalid shape");
  }
  validate_pointer_section(data.stored_row_ptr,
                           data.stored_source_global_row.size(),
                           data.stored_target_local_row.size(),
                           "stored-row pointers");
  if (data.stored_target_local_row.size() !=
          data.stored_physical_edge_id.size() ||
      data.physical_edge_src.size() != data.physical_edge_dst.size()) {
    throw LogicalGraphError(
        "logical section stored-edge columns have invalid shape");
  }

  auto storage = std::make_shared<LogicalGraphStorage>();
  storage->options.max_depth = data.max_depth;
  storage->corpus_object_count = data.corpus_object_count;
  storage->corpus_vector_dim = data.corpus_vector_dim;
  storage->corpus_finite_values_prevalidated = true;
  storage->blocks.resize(block_count);
  std::map<LogicalViewId, std::size_t> block_by_id;
  for (std::size_t block_index = 0U; block_index < block_count; ++block_index) {
    LogicalBlock& block = storage->blocks[block_index];
    block.view_id = data.block_view_id[block_index];
    block.component_sort_key = data.block_component_sort_key[block_index];
    if (data.block_kind[block_index] ==
        static_cast<std::uint8_t>(BlockKind::kLeaf)) {
      block.kind = BlockKind::kLeaf;
    } else if (data.block_kind[block_index] ==
               static_cast<std::uint8_t>(BlockKind::kParent)) {
      block.kind = BlockKind::kParent;
    } else {
      throw LogicalGraphError("logical section block kind is invalid");
    }
    block.depth = data.block_depth[block_index];
    if ((block.kind == BlockKind::kLeaf && block.depth != 0U) ||
        (block.kind == BlockKind::kParent && block.depth == 0U)) {
      throw LogicalGraphError(
          "logical section block kind has an inconsistent depth");
    }
    if (block.depth > data.max_depth ||
        (block_index != 0U &&
         std::tie(data.block_depth[block_index - 1U],
                  data.block_view_id[block_index - 1U]) >=
             std::tie(data.block_depth[block_index],
                      data.block_view_id[block_index]))) {
      throw LogicalGraphError("logical section block order/depth is invalid");
    }
    if (!block_by_id.emplace(block.view_id, block_index).second) {
      throw LogicalGraphError("logical section view IDs are not unique");
    }
    const std::size_t member_begin =
        static_cast<std::size_t>(data.block_member_ptr[block_index]);
    const std::size_t member_end =
        static_cast<std::size_t>(data.block_member_ptr[block_index + 1U]);
    block.members.assign(
        data.member_ids.begin() +
            checked_iterator_offset(member_begin, "member offset"),
        data.member_ids.begin() +
            checked_iterator_offset(member_end, "member offset"));
    require_sorted_unique_members(block.members, "section block members");
    for (const std::int64_t object_id : block.members) {
      if (object_id < 0 ||
          static_cast<std::uint64_t>(object_id) >= data.corpus_object_count) {
        throw LogicalGraphError(
            "logical section member is outside corpus identity");
      }
    }
    block.anchor_local_row = data.block_anchor_local_row[block_index];
    if (block.anchor_local_row >= block.members.size()) {
      throw LogicalGraphError("logical section anchor is outside members");
    }
    const std::size_t entry_begin =
        static_cast<std::size_t>(data.block_entry_ptr[block_index]);
    const std::size_t entry_end =
        static_cast<std::size_t>(data.block_entry_ptr[block_index + 1U]);
    block.entries.assign(
        data.entry_ids.begin() +
            checked_iterator_offset(entry_begin, "entry offset"),
        data.entry_ids.begin() +
            checked_iterator_offset(entry_end, "entry offset"));
    validate_entries(block.members, block.members[block.anchor_local_row],
                     block.entries);
  }

  for (std::size_t block_index = 0U; block_index < block_count; ++block_index) {
    LogicalBlock& block = storage->blocks[block_index];
    const std::size_t binding_begin =
        static_cast<std::size_t>(data.block_binding_ptr[block_index]);
    const std::size_t binding_end =
        static_cast<std::size_t>(data.block_binding_ptr[block_index + 1U]);
    std::set<LogicalViewId> child_ids;
    LogicalViewId previous_child_id = 0U;
    bool has_previous_child = false;
    for (std::size_t binding_index = binding_begin;
         binding_index < binding_end; ++binding_index) {
      const LogicalViewId child_id =
          data.binding_child_view_id[binding_index];
      const auto child_iterator = block_by_id.find(child_id);
      if (child_iterator == block_by_id.end() ||
          child_iterator->second >= block_index ||
          !child_ids.insert(child_id).second) {
        throw LogicalGraphError(
            "logical section child DAG is future, missing, or duplicate");
      }
      if (has_previous_child && previous_child_id >= child_id) {
        throw LogicalGraphError(
            "logical section child bindings are not canonical");
      }
      previous_child_id = child_id;
      has_previous_child = true;
      ChildBinding binding;
      binding.child_index = child_iterator->second;
      const std::size_t projection_begin = static_cast<std::size_t>(
          data.binding_projection_ptr[binding_index]);
      const std::size_t projection_end = static_cast<std::size_t>(
          data.binding_projection_ptr[binding_index + 1U]);
      binding.child_to_parent.assign(
          data.projection_parent_row.begin() +
              checked_iterator_offset(projection_begin, "projection offset"),
          data.projection_parent_row.begin() +
              checked_iterator_offset(projection_end, "projection offset"));
      const LogicalBlock& child = storage->blocks[binding.child_index];
      if (binding.child_to_parent.size() != child.members.size()) {
        throw LogicalGraphError(
            "logical section projection length does not match child");
      }
      for (std::size_t child_row = 0U; child_row < child.members.size();
           ++child_row) {
        const std::uint32_t parent_row =
            binding.child_to_parent[child_row];
        if (parent_row >= block.members.size() ||
            block.members[parent_row] != child.members[child_row]) {
          throw LogicalGraphError(
              "logical section projection has an object-ID mismatch");
        }
      }
      block.bindings.push_back(std::move(binding));
    }
    if (block.kind == BlockKind::kLeaf && binding_begin != binding_end) {
      throw LogicalGraphError("logical leaf has child bindings");
    }
    if (block.kind == BlockKind::kParent && binding_begin == binding_end) {
      throw LogicalGraphError("logical parent has no child binding");
    }

    std::vector<unsigned char> binding_referenced(block.bindings.size(), 0U);
    if (block.kind == BlockKind::kParent) block.owner_ptr.push_back(0U);
    const std::size_t global_row_begin =
        static_cast<std::size_t>(data.block_member_ptr[block_index]);
    for (std::size_t row = 0U; row < block.members.size(); ++row) {
      const std::size_t global_row = global_row_begin + row;
      const std::uint64_t owner_begin = data.row_owner_ptr[global_row];
      const std::uint64_t owner_end = data.row_owner_ptr[global_row + 1U];
      if (block.kind == BlockKind::kLeaf && owner_begin != owner_end) {
        throw LogicalGraphError("logical leaf row has an owner reference");
      }
      if (block.kind == BlockKind::kParent && owner_begin == owner_end) {
        throw LogicalGraphError("logical parent row has no owner reference");
      }
      std::uint32_t previous_owner_binding = 0U;
      bool has_previous_owner = false;
      for (std::uint64_t offset = owner_begin; offset < owner_end; ++offset) {
        const std::size_t owner_offset = static_cast<std::size_t>(offset);
        const std::uint32_t global_binding = data.owner_binding[owner_offset];
        if (global_binding < binding_begin || global_binding >= binding_end) {
          throw LogicalGraphError(
              "logical section owner references another block's binding");
        }
        if (has_previous_owner && previous_owner_binding >= global_binding) {
          throw LogicalGraphError(
              "logical section row owners are not canonical unique refs");
        }
        previous_owner_binding = global_binding;
        has_previous_owner = true;
        const std::size_t local_binding =
            static_cast<std::size_t>(global_binding) - binding_begin;
        binding_referenced[local_binding] = 1U;
        const std::uint32_t child_row = data.owner_child_row[owner_offset];
        if (child_row >= block.bindings[local_binding].child_to_parent.size() ||
            block.bindings[local_binding].child_to_parent[child_row] != row) {
          throw LogicalGraphError(
              "logical section owner projection is inconsistent");
        }
        if (block.kind == BlockKind::kParent) {
          block.owner_binding.push_back(
              static_cast<std::uint32_t>(local_binding));
          block.owner_child_row.push_back(child_row);
        }
      }
      if (block.kind == BlockKind::kParent) {
        block.owner_ptr.push_back(
            static_cast<std::uint64_t>(block.owner_binding.size()));
      }
      const std::uint8_t reason = data.row_reason_mask[global_row];
      if ((reason & static_cast<std::uint8_t>(~0x0fU)) != 0U) {
        throw LogicalGraphError("logical section row reason has unknown bits");
      }
      if (block.kind == BlockKind::kLeaf && reason != 0U) {
        throw LogicalGraphError("logical leaf row carries a parent reason");
      }
      const std::size_t owner_count =
          static_cast<std::size_t>(owner_end - owner_begin);
      const bool overlap = (reason & kRowReasonOverlap) != 0U;
      if (block.kind == BlockKind::kParent &&
          (overlap != (owner_count > 1U))) {
        throw LogicalGraphError(
            "logical section overlap reason disagrees with owners");
      }
      if (block.kind == BlockKind::kParent) block.reason_mask.push_back(reason);
      const std::uint8_t expected_kind = static_cast<std::uint8_t>(
          block.kind == BlockKind::kLeaf
              ? LogicalRowKind::kStored
              : (reason == 0U ? LogicalRowKind::kChildRef
                              : LogicalRowKind::kOverlay));
      if (data.row_kind[global_row] != expected_kind) {
        throw LogicalGraphError(
            "logical section row kind disagrees with block/reason");
      }
    }
    for (const unsigned char referenced : binding_referenced) {
      if (referenced == 0U) {
        throw LogicalGraphError(
            "logical section has an unreferenced child binding");
      }
    }
  }

  std::vector<std::size_t> stored_index_by_row(
      row_count, std::numeric_limits<std::size_t>::max());
  for (std::size_t stored = 0U;
       stored < data.stored_source_global_row.size(); ++stored) {
    const std::uint64_t source = data.stored_source_global_row[stored];
    if (source >= row_count ||
        (stored != 0U &&
         data.stored_source_global_row[stored - 1U] >= source)) {
      throw LogicalGraphError(
          "logical section stored sources are not sorted unique rows");
    }
    stored_index_by_row[static_cast<std::size_t>(source)] = stored;
  }
  std::vector<unsigned char> physical_edge_referenced(
      data.physical_edge_src.size(), 0U);
  for (std::size_t edge_id = 0U; edge_id < data.physical_edge_src.size();
       ++edge_id) {
    const std::int64_t source = data.physical_edge_src[edge_id];
    const std::int64_t target = data.physical_edge_dst[edge_id];
    if (source < 0 || target < 0 || source == target ||
        static_cast<std::uint64_t>(source) >= data.corpus_object_count ||
        static_cast<std::uint64_t>(target) >= data.corpus_object_count ||
        (edge_id != 0U &&
         std::tie(data.physical_edge_src[edge_id - 1U],
                  data.physical_edge_dst[edge_id - 1U]) >=
             std::tie(source, target))) {
      throw LogicalGraphError(
          "logical physical-edge table is not canonical and valid");
    }
  }

  for (std::size_t block_index = 0U; block_index < block_count; ++block_index) {
    LogicalBlock& block = storage->blocks[block_index];
    const std::size_t global_row_begin =
        static_cast<std::size_t>(data.block_member_ptr[block_index]);
    if (block.kind == BlockKind::kLeaf) block.leaf_row_ptr.push_back(0U);
    block.overlay_index_by_source.assign(block.members.size(), kNoLocalRow);
    for (std::size_t row = 0U; row < block.members.size(); ++row) {
      const std::size_t global_row = global_row_begin + row;
      const std::size_t stored = stored_index_by_row[global_row];
      const bool should_store =
          block.kind == BlockKind::kLeaf || block.reason_mask[row] != 0U;
      if (should_store !=
          (stored != std::numeric_limits<std::size_t>::max())) {
        throw LogicalGraphError(
            "logical stored adjacency disagrees with row kind");
      }
      if (!should_store) continue;
      const std::uint64_t begin = data.stored_row_ptr[stored];
      const std::uint64_t end = data.stored_row_ptr[stored + 1U];
      if (end - begin > kLogicalMaxDegree) {
        throw LogicalGraphError("logical stored row exceeds degree 24");
      }
      if (block.kind == BlockKind::kParent) {
        block.overlay_index_by_source[row] =
            static_cast<std::uint32_t>(block.overlay_source_rows.size());
        block.overlay_source_rows.push_back(static_cast<std::uint32_t>(row));
        block.overlay_row_ptr.push_back(
            static_cast<std::uint64_t>(block.overlay_target_rows.size()));
      }
      std::set<std::uint32_t> seen_targets;
      for (std::uint64_t offset = begin; offset < end; ++offset) {
        const std::size_t stored_offset = static_cast<std::size_t>(offset);
        const std::uint32_t target =
            data.stored_target_local_row[stored_offset];
        const std::uint64_t edge_id =
            data.stored_physical_edge_id[stored_offset];
        if (target >= block.members.size() || target == row ||
            !seen_targets.insert(target).second ||
            edge_id >= data.physical_edge_src.size() ||
            data.physical_edge_src[static_cast<std::size_t>(edge_id)] !=
                block.members[row] ||
            data.physical_edge_dst[static_cast<std::size_t>(edge_id)] !=
                block.members[target]) {
          throw LogicalGraphError(
              "logical stored adjacency/physical identity is invalid");
        }
        physical_edge_referenced[static_cast<std::size_t>(edge_id)] = 1U;
        if (block.kind == BlockKind::kLeaf) {
          block.leaf_target_rows.push_back(target);
          block.leaf_physical_edge_ids.push_back(edge_id);
        } else {
          block.overlay_target_rows.push_back(target);
          block.overlay_physical_edge_ids.push_back(edge_id);
        }
      }
      if (block.kind == BlockKind::kLeaf) {
        block.leaf_row_ptr.push_back(
            static_cast<std::uint64_t>(block.leaf_target_rows.size()));
      }
    }
    if (block.kind == BlockKind::kParent) {
      block.overlay_row_ptr.push_back(
          static_cast<std::uint64_t>(block.overlay_target_rows.size()));
    }
  }
  for (const unsigned char referenced : physical_edge_referenced) {
    if (referenced == 0U) {
      throw LogicalGraphError("logical physical-edge table has an unused row");
    }
  }
  prepare_index_and_validate(storage.get());
  return storage;
}

void validate_builder_options(const LogicalGraphOptions& options) {
  if (options.max_depth == 0U || options.max_depth > kLogicalMaxDepth) {
    throw LogicalGraphError("max_depth must be in [1, frozen logical limit]");
  }
}

void validate_builder_corpus(const CorpusView& corpus) {
  if (!corpus.finite_values_prevalidated) {
    throw LogicalGraphError(
        "logical graph corpus finite-values contract is not prevalidated");
  }
  if (corpus.vector_dim == 0U || corpus.object_count == 0U) {
    throw LogicalGraphError("logical graph corpus shape must be non-empty");
  }
  const std::uint64_t dimension = static_cast<std::uint64_t>(corpus.vector_dim);
  if (corpus.object_count > std::numeric_limits<std::uint64_t>::max() /
                                dimension) {
    throw LogicalGraphError("logical graph corpus vector shape overflows");
  }
  const std::uint64_t expected = corpus.object_count * dimension;
  if (expected > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max()) ||
      corpus.vectors.size() != static_cast<std::size_t>(expected) ||
      (expected != 0U && corpus.vectors.data() == nullptr)) {
    throw LogicalGraphError(
        "logical graph corpus vectors do not match their declared shape");
  }
}

void validate_corpus_members(const CorpusView& corpus,
                             const std::vector<std::int64_t>& members,
                             const char* label) {
  for (const std::int64_t object_id : members) {
    if (object_id < 0 ||
        static_cast<std::uint64_t>(object_id) >= corpus.object_count) {
      throw LogicalGraphError(std::string(label) +
                              " contains an out-of-corpus object ID");
    }
  }
}

void require_builder_open(
    const std::shared_ptr<LogicalGraphStorage>& storage, bool frozen) {
  if (!storage) {
    throw LogicalGraphError("logical graph builder was moved from");
  }
  if (frozen) throw LogicalGraphError("logical graph builder is already frozen");
}

}  // namespace

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::uint64_t logical_graph_sha256_call_count_for_testing() noexcept {
  return logical_graph_sha256_calls.load(std::memory_order_relaxed);
}

void reset_logical_graph_sha256_call_count_for_testing() noexcept {
  logical_graph_sha256_calls.store(0U, std::memory_order_relaxed);
}

std::uint64_t logical_graph_resolve_call_count_for_testing() noexcept {
  return logical_graph_resolve_calls.load(std::memory_order_relaxed);
}

void reset_logical_graph_resolve_call_count_for_testing() noexcept {
  logical_graph_resolve_calls.store(0U, std::memory_order_relaxed);
}
#endif

LogicalGraphSections::LogicalGraphSections(
    std::shared_ptr<const LogicalGraphSectionStorage> storage)
    : storage_(std::move(storage)) {}

LogicalGraphSections LogicalGraphSections::from_untrusted_data(
    LogicalGraphSectionData data) {
  return from_canonical_bytes(encode_section_data(data));
}

LogicalGraphSections LogicalGraphSections::from_canonical_bytes(
    std::vector<std::uint8_t> canonical_bytes) {
  LogicalGraphSectionData data = decode_section_data(canonical_bytes);
  if (encode_section_data(data) != canonical_bytes) {
    throw LogicalGraphError("logical section image is not canonical");
  }
  auto storage = std::make_shared<LogicalGraphSectionStorage>();
  storage->data = std::move(data);
  storage->digest = sha256_bytes(canonical_bytes);
  storage->canonical_bytes = std::move(canonical_bytes);
  return LogicalGraphSections(
      std::shared_ptr<const LogicalGraphSectionStorage>(std::move(storage)));
}

bool LogicalGraphSections::empty() const noexcept { return !storage_; }

const LogicalGraphSectionData& LogicalGraphSections::data() const {
  if (!storage_) throw LogicalGraphError("logical graph sections are empty");
  return storage_->data;
}

ArrayView<std::uint8_t> LogicalGraphSections::canonical_bytes() const noexcept {
  if (!storage_) return ArrayView<std::uint8_t>();
  return ArrayView<std::uint8_t>(storage_->canonical_bytes.data(),
                                 storage_->canonical_bytes.size());
}

LogicalSectionDigest LogicalGraphSections::sha256() const noexcept {
  if (!storage_) return LogicalSectionDigest{};
  return storage_->digest;
}

LogicalGraphView::LogicalGraphView(
    std::shared_ptr<const LogicalGraphStorage> storage)
    : storage_(std::move(storage)) {}

LogicalGraphView::~LogicalGraphView() = default;

LogicalGraphView LogicalGraphView::open(const LogicalGraphSections& sections,
                                        const CorpusView& corpus) {
  validate_builder_corpus(corpus);
  if (!sections.storage_) {
    throw LogicalGraphError("cannot open empty logical graph sections");
  }
  // The only section factory is private-owner-backed and computes digest after
  // canonical framing validation. canonical_bytes() is a read-only ArrayView,
  // so this immutable owner can safely carry the cached digest through open.
  const LogicalGraphSectionData& data = sections.storage_->data;
  if (data.corpus_object_count != corpus.object_count ||
      data.corpus_vector_dim != corpus.vector_dim ||
      data.corpus_finite_values_prevalidated !=
          static_cast<std::uint8_t>(
              corpus.finite_values_prevalidated ? 1U : 0U)) {
    throw LogicalGraphError("logical section corpus identity mismatch");
  }
  std::shared_ptr<LogicalGraphStorage> storage =
      storage_from_section_data(data);
  storage->sections = sections;
  storage->fingerprint = sections.storage_->digest;
  storage->stats.physical_edge_count =
      static_cast<std::uint64_t>(data.physical_edge_src.size());
  storage->stats.section_bytes = static_cast<std::uint64_t>(
      sections.storage_->canonical_bytes.size());
  return LogicalGraphView(
      std::shared_ptr<const LogicalGraphStorage>(std::move(storage)));
}

LogicalNeighbors LogicalGraphView::neighbors(LogicalViewId root_view,
                                             std::uint32_t root_local_row,
                                             LogicalScratch& scratch) const {
  if (!storage_) throw LogicalGraphError("logical graph view is empty");
  const std::size_t block_index = find_block_ordered(*storage_, root_view);
  const LogicalBlock& block = storage_->blocks[block_index];
  if (root_local_row >= block.members.size()) {
    throw LogicalGraphError("root local row is outside frozen view");
  }
  const std::vector<std::uint64_t>* row_ptr = nullptr;
  const std::vector<std::uint32_t>* target_rows = nullptr;
  const std::vector<std::uint64_t>* physical_edge_ids = nullptr;
  if (block.kind == LogicalBlockKind::kLeaf) {
    row_ptr = &block.leaf_row_ptr;
    target_rows = &block.leaf_target_rows;
    physical_edge_ids = &block.leaf_physical_edge_ids;
  } else {
    if (block.effective_row_ptr.size() != block.members.size() + 1U ||
        block.effective_row_ptr.empty() ||
        block.effective_row_ptr.back() != block.effective_target_rows.size() ||
        block.effective_target_rows.size() !=
            block.effective_physical_edge_ids.size()) {
      throw LogicalGraphError("frozen effective adjacency has invalid shape");
    }
    row_ptr = &block.effective_row_ptr;
    target_rows = &block.effective_target_rows;
    physical_edge_ids = &block.effective_physical_edge_ids;
  }
  if (row_ptr->size() != block.members.size() + 1U || row_ptr->empty() ||
      row_ptr->back() != target_rows->size() ||
      (!physical_edge_ids->empty() &&
       target_rows->size() != physical_edge_ids->size())) {
    throw LogicalGraphError("frozen adjacency has invalid shape");
  }
  const std::uint64_t begin = (*row_ptr)[root_local_row];
  const std::uint64_t end = (*row_ptr)[root_local_row + 1U];
  if (begin > end || end > target_rows->size() ||
      end - begin > kLogicalMaxDegree) {
    throw LogicalGraphError("frozen effective adjacency row exceeds degree 24");
  }
  const std::uint16_t count = static_cast<std::uint16_t>(end - begin);
  for (std::uint16_t index = 0U; index < count; ++index) {
    const std::size_t effective_offset =
        static_cast<std::size_t>(begin) + index;
    const std::uint32_t local_row = (*target_rows)[effective_offset];
    if (local_row >= block.members.size()) {
      throw LogicalGraphError("effective adjacency returned an outside row");
    }
    scratch.frame_rows[0U][index] = local_row;
    scratch.frame_physical_edge_ids[0U][index] =
        physical_edge_ids->empty() ? 0U : (*physical_edge_ids)[effective_offset];
    scratch.object_ids[index] = block.members[local_row];
  }
  return LogicalNeighbors{
      ArrayView<std::uint32_t>(scratch.frame_rows[0U].data(), count),
      ArrayView<std::int64_t>(scratch.object_ids.data(), count),
      ArrayView<std::uint64_t>(
          scratch.frame_physical_edge_ids[0U].data(), count)};
}

void LogicalGraphView::validate() const {
  if (!storage_) throw LogicalGraphError("logical graph view is empty");
  validate_storage_shape(*storage_);
  (void)calculate_stats(*storage_);
}

ArrayView<std::int64_t> LogicalGraphView::members(LogicalViewId view) const {
  if (!storage_) throw LogicalGraphError("logical graph view is empty");
  const LogicalBlock& block = storage_->blocks[find_block_ordered(*storage_, view)];
  return ArrayView<std::int64_t>(block.members.data(), block.members.size());
}

LogicalGraphStats LogicalGraphView::stats() const noexcept {
  if (!storage_) return LogicalGraphStats{};
  return storage_->stats;
}

LogicalSectionFingerprint LogicalGraphView::section_fingerprint() const noexcept {
  if (!storage_) return LogicalSectionFingerprint{};
  return storage_->fingerprint;
}

LogicalGraphBuilder::LogicalGraphBuilder(const CorpusView& corpus,
                                         const LogicalGraphOptions& options)
    : corpus_(corpus), storage_(std::make_shared<LogicalGraphStorage>()) {
  validate_builder_corpus(corpus);
  validate_builder_options(options);
  storage_->options = options;
  storage_->corpus_object_count = corpus.object_count;
  storage_->corpus_vector_dim = corpus.vector_dim;
  storage_->corpus_finite_values_prevalidated =
      corpus.finite_values_prevalidated;
}

LogicalGraphBuilder::~LogicalGraphBuilder() = default;

LogicalGraphBuilder::LogicalGraphBuilder(LogicalGraphBuilder&& other) noexcept
    : corpus_(other.corpus_),
      storage_(std::move(other.storage_)),
      frozen_(other.frozen_) {
  other.corpus_ = CorpusView{};
  other.frozen_ = true;
}

LogicalGraphBuilder& LogicalGraphBuilder::operator=(
    LogicalGraphBuilder&& other) noexcept {
  if (this == &other) return *this;
  corpus_ = other.corpus_;
  storage_ = std::move(other.storage_);
  frozen_ = other.frozen_;
  other.corpus_ = CorpusView{};
  other.frozen_ = true;
  return *this;
}

void LogicalGraphBuilder::add_leaf(LogicalViewId view_id,
                                    std::uint64_t component_sort_key,
                                    std::int64_t anchor,
                                    ArrayView<std::int64_t> entries,
                                    const LeafBuildResult& leaf) {
  require_builder_open(storage_, frozen_);
  if (find_block_linear(*storage_, view_id) !=
      std::numeric_limits<std::size_t>::max()) {
    throw LogicalGraphError("logical view ID is already present");
  }
  std::vector<std::int64_t> members(leaf.row_object_ids.begin(),
                                    leaf.row_object_ids.end());
  require_sorted_unique_members(members, "leaf members");
  validate_corpus_members(corpus_, members, "leaf members");
  const std::vector<std::int64_t> copied_entries(entries.begin(), entries.end());
  validate_entries(members, anchor, copied_entries);
  if (leaf.row_ptr.size() != members.size() + 1U ||
      leaf.row_ptr.empty() || leaf.row_ptr.front() != 0U ||
      leaf.row_ptr.back() != leaf.target_ids.size()) {
    throw LogicalGraphError("leaf_build result has invalid final CSR shape");
  }

  LogicalBlock block;
  block.view_id = view_id;
  block.component_sort_key = component_sort_key;
  block.kind = BlockKind::kLeaf;
  block.depth = 0U;
  block.members = std::move(members);
  block.anchor_local_row = require_local_member(block.members, anchor, "anchor");
  block.entries = copied_entries;
  block.leaf_row_ptr = leaf.row_ptr;
  block.leaf_target_rows.reserve(leaf.target_ids.size());
  std::vector<std::vector<std::uint32_t>> rows(block.members.size());
  for (std::size_t row = 0U; row < block.members.size(); ++row) {
    const std::uint64_t begin = block.leaf_row_ptr[row];
    const std::uint64_t end = block.leaf_row_ptr[row + 1U];
    if (begin > end || end > leaf.target_ids.size() ||
        end - begin > kLogicalMaxDegree) {
      throw LogicalGraphError("leaf row exceeds degree 24");
    }
    std::set<std::uint32_t> seen;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::int64_t target_id =
          leaf.target_ids[static_cast<std::size_t>(offset)];
      const std::uint32_t target =
          require_local_member(block.members, target_id, "leaf target");
      if (target == row || !seen.insert(target).second) {
        throw LogicalGraphError("leaf row has duplicate/self target");
      }
      block.leaf_target_rows.push_back(target);
      rows[row].push_back(target);
    }
  }
  validate_directed_rows(rows, block.anchor_local_row, "leaf graph");
  storage_->blocks.push_back(std::move(block));
}

void LogicalGraphBuilder::add_parent(
    LogicalViewId view_id, std::uint64_t component_sort_key,
    ArrayView<std::int64_t> members, std::int64_t anchor,
    ArrayView<std::int64_t> entries, ArrayView<LogicalViewId> children,
    const ParentComposeOptions& options) {
  require_builder_open(storage_, frozen_);
  if (find_block_linear(*storage_, view_id) !=
      std::numeric_limits<std::size_t>::max()) {
    throw LogicalGraphError("logical view ID is already present");
  }
  if (children.empty()) {
    throw LogicalGraphError("parent must have at least one child");
  }
  std::vector<std::int64_t> copied_members(members.begin(), members.end());
  require_sorted_unique_members(copied_members, "parent members");
  validate_corpus_members(corpus_, copied_members, "parent members");
  const std::vector<std::int64_t> copied_entries(entries.begin(), entries.end());
  validate_entries(copied_members, anchor, copied_entries);

  std::vector<std::size_t> child_indices;
  child_indices.reserve(children.size());
  std::set<LogicalViewId> seen_child_ids;
  for (const LogicalViewId child_id : children) {
    if (!seen_child_ids.insert(child_id).second) {
      throw LogicalGraphError("parent child list contains a duplicate");
    }
    const std::size_t child_index = find_block_linear(*storage_, child_id);
    if (child_index == std::numeric_limits<std::size_t>::max()) {
      throw LogicalGraphError("parent references a future/out-of-artifact child");
    }
    child_indices.push_back(child_index);
  }
  std::sort(child_indices.begin(), child_indices.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              return storage_->blocks[lhs].view_id < storage_->blocks[rhs].view_id;
            });

  std::vector<MaterializedChild> materialized;
  materialized.reserve(child_indices.size());
  for (const std::size_t child_index : child_indices) {
    MaterializedChild child;
    child.block_index = child_index;
    child.rows = materialize_rows(*storage_, child_index);
    child.row_ptr = make_row_ptr(child.rows);
    child.witness_parent_row =
        make_witness(child.rows,
                     storage_->blocks[child_index].anchor_local_row);
    child.target_rows.reserve(child.row_ptr.back());
    for (const auto& row : child.rows) {
      child.target_rows.insert(child.target_rows.end(), row.begin(), row.end());
    }
    materialized.push_back(std::move(child));
  }
  std::vector<ChildView> child_views;
  child_views.reserve(materialized.size());
  for (std::size_t child_number = 0U; child_number < materialized.size();
       ++child_number) {
    const MaterializedChild& materialized_child = materialized[child_number];
    const LogicalBlock& child = storage_->blocks[materialized_child.block_index];
    child_views.push_back(ChildView{
        static_cast<std::int64_t>(child.view_id),
        child.component_sort_key,
        ArrayView<std::int64_t>(child.members.data(), child.members.size()),
        child.members[child.anchor_local_row],
        ArrayView<std::int64_t>(child.members.data(), child.members.size()),
        ArrayView<std::uint64_t>(materialized_child.row_ptr.data(),
                                 materialized_child.row_ptr.size()),
        ArrayView<std::uint64_t>(materialized_child.target_rows.data(),
                                 materialized_child.target_rows.size()),
        ArrayView<std::uint64_t>(materialized_child.witness_parent_row.data(),
                                 materialized_child.witness_parent_row.size())});
  }

  ParentComposeResult composed;
  try {
    composed = compose_parent(
        corpus_, ArrayView<std::int64_t>(copied_members.data(), copied_members.size()),
        anchor, ArrayView<ChildView>(child_views.data(), child_views.size()),
        options);
  } catch (const ParentComposeError& error) {
    throw LogicalGraphError(error_prefix("compose_parent rejected logical parent: ",
                                         error.what()));
  }
  if (composed.child_identities.size() != child_indices.size()) {
    throw LogicalGraphError("compose_parent changed the child identity set");
  }
  for (std::size_t child_number = 0U; child_number < child_indices.size();
       ++child_number) {
    const LogicalBlock& child = storage_->blocks[child_indices[child_number]];
    if (composed.child_identities[child_number].child_id !=
            static_cast<std::int64_t>(child.view_id) ||
        composed.child_identities[child_number].component_sort_key !=
            child.component_sort_key) {
      throw LogicalGraphError(
          "compose_parent returned inconsistent child identity ordering");
    }
  }

  LogicalBlock block;
  block.view_id = view_id;
  block.component_sort_key = component_sort_key;
  block.kind = BlockKind::kParent;
  block.members = std::move(copied_members);
  block.anchor_local_row = require_local_member(block.members, anchor, "anchor");
  block.entries = copied_entries;
  block.bindings.reserve(child_indices.size());
  for (const std::size_t child_index : child_indices) {
    const LogicalBlock& child = storage_->blocks[child_index];
    ChildBinding binding;
    binding.child_index = child_index;
    binding.child_to_parent.reserve(child.members.size());
    for (const std::int64_t object_id : child.members) {
      binding.child_to_parent.push_back(
          require_local_member(block.members, object_id, "child projection"));
    }
    block.bindings.push_back(std::move(binding));
    block.depth = std::max(block.depth, child.depth + 1U);
  }
  if (block.depth > storage_->options.max_depth) {
    throw LogicalGraphError("parent exceeds frozen logical max depth");
  }
  block.owner_ptr = composed.owner_ptr;
  block.owner_binding.reserve(composed.owner_child_index.size());
  block.owner_child_row.reserve(composed.owner_child_row.size());
  for (std::size_t index = 0U; index < composed.owner_child_index.size(); ++index) {
    const std::uint64_t binding = composed.owner_child_index[index];
    const std::uint64_t child_row = composed.owner_child_row[index];
    if (binding >= block.bindings.size() ||
        child_row >= block.bindings[static_cast<std::size_t>(binding)]
                          .child_to_parent.size()) {
      throw LogicalGraphError("compose_parent returned an invalid owner projection");
    }
    block.owner_binding.push_back(static_cast<std::uint32_t>(binding));
    block.owner_child_row.push_back(static_cast<std::uint32_t>(child_row));
  }
  block.reason_mask = composed.row_reason_mask;
  if (block.reason_mask.size() != block.members.size()) {
    throw LogicalGraphError("compose_parent omitted parent row reasons");
  }
  block.overlay_index_by_source.assign(block.members.size(), kNoLocalRow);
  if (composed.overlay_row_ptr.size() !=
      composed.overlay_source_ids.size() + 1U ||
      composed.overlay_row_ptr.empty() ||
      composed.overlay_row_ptr.back() != composed.overlay_target_ids.size()) {
    throw LogicalGraphError("compose_parent returned malformed overlay CSR");
  }
  for (std::size_t overlay = 0U;
       overlay < composed.overlay_source_ids.size(); ++overlay) {
    const std::uint32_t source = require_local_member(
        block.members, composed.overlay_source_ids[overlay],
        "overlay source");
    if (block.reason_mask[source] == 0U ||
        block.overlay_index_by_source[source] != kNoLocalRow) {
      throw LogicalGraphError("overlay source has no unique dirty reason");
    }
    block.overlay_index_by_source[source] = static_cast<std::uint32_t>(overlay);
    block.overlay_source_rows.push_back(source);
    const std::uint64_t begin = composed.overlay_row_ptr[overlay];
    const std::uint64_t end = composed.overlay_row_ptr[overlay + 1U];
    if (begin > end || end > composed.overlay_target_ids.size() ||
        end - begin > kLogicalMaxDegree) {
      throw LogicalGraphError("overlay row exceeds degree 24");
    }
    block.overlay_row_ptr.push_back(
        block.overlay_target_rows.size());
    std::set<std::uint32_t> seen_targets;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::uint32_t target = require_local_member(
          block.members,
          composed.overlay_target_ids[static_cast<std::size_t>(offset)],
          "overlay target");
      if (target == source || !seen_targets.insert(target).second) {
        throw LogicalGraphError("overlay has duplicate/self target");
      }
      block.overlay_target_rows.push_back(target);
    }
  }
  block.overlay_row_ptr.push_back(block.overlay_target_rows.size());
  for (std::size_t row = 0U; row < block.reason_mask.size(); ++row) {
    if (block.reason_mask[row] == 0U &&
        block.overlay_index_by_source[row] != kNoLocalRow) {
      throw LogicalGraphError("ordinary parent row was stored in overlay");
    }
    if (block.reason_mask[row] != 0U &&
        block.overlay_index_by_source[row] == kNoLocalRow) {
      throw LogicalGraphError("dirty parent row has no overlay replacement");
    }
  }
  storage_->blocks.push_back(std::move(block));
  try {
    validate_parent_block(*storage_, storage_->blocks.size() - 1U);
  } catch (...) {
    storage_->blocks.pop_back();
    throw;
  }
}

LogicalGraphView LogicalGraphBuilder::freeze() {
  require_builder_open(storage_, frozen_);
  prepare_index_and_validate(storage_.get());
  const LogicalGraphSections sections =
      LogicalGraphSections::from_untrusted_data(
          make_section_data(*storage_, corpus_));
  LogicalGraphView view = LogicalGraphView::open(sections, corpus_);
  frozen_ = true;
  return view;
}

LogicalGraphSections LogicalGraphBuilder::freeze_sections() {
  require_builder_open(storage_, frozen_);
  prepare_index_and_validate(storage_.get());
  const LogicalGraphSections sections =
      LogicalGraphSections::from_untrusted_data(
          make_section_data(*storage_, corpus_));
  (void)LogicalGraphView::open(sections, corpus_);
  frozen_ = true;
  return sections;
}

}  // namespace bounded_blowup

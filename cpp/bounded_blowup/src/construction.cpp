// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.

#include "bounded_blowup/construction.hpp"

#include "bounded_blowup/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <queue>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace bounded_blowup {
namespace {

constexpr std::size_t kNpos = std::numeric_limits<std::size_t>::max();
constexpr std::uint32_t kProductionMaxDegree = 24U;

struct NormalizedChild {
  ChildView view;
};

struct Reachability {
  std::vector<unsigned char> reached;
  std::vector<std::vector<std::int64_t>> witness_children;
  std::size_t reached_count = 0;
};

struct DisconnectedPart {
  std::vector<std::size_t> rows;
  std::uint64_t sort_key = std::numeric_limits<std::uint64_t>::max();
  std::int64_t sort_child = std::numeric_limits<std::int64_t>::max();
  std::int64_t entry = std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_object = std::numeric_limits<std::int64_t>::max();
};

struct Group {
  std::vector<std::size_t> rows;
  std::vector<std::size_t> reverse_eligible_rows;
  std::uint64_t sort_key = std::numeric_limits<std::uint64_t>::max();
  std::int64_t sort_child = std::numeric_limits<std::int64_t>::max();
  std::int64_t entry = std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_object = std::numeric_limits<std::int64_t>::max();
};

std::size_t find_sorted_id(ArrayView<std::int64_t> values,
                           std::int64_t object_id) {
  const auto iterator = std::lower_bound(values.begin(), values.end(), object_id);
  if (iterator == values.end() || *iterator != object_id) return kNpos;
  return static_cast<std::size_t>(iterator - values.begin());
}

std::size_t corpus_object_count(const CorpusView& corpus) {
  if (!corpus.finite_values_prevalidated) {
    throw ParentComposeError(
        "CorpusView must carry a prevalidated finite-value contract");
  }
  if (corpus.vector_dim == 0U || corpus.object_count == 0U) {
    throw ParentComposeError("CorpusView shape must be non-empty");
  }
  if (corpus.object_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw ParentComposeError("CorpusView object_count exceeds host size_t");
  }
  const std::size_t object_count = static_cast<std::size_t>(corpus.object_count);
  const std::size_t dimension = static_cast<std::size_t>(corpus.vector_dim);
  if (object_count > std::numeric_limits<std::size_t>::max() / dimension ||
      corpus.vectors.size() != object_count * dimension) {
    throw ParentComposeError("CorpusView vectors do not match its shape");
  }
  return object_count;
}

void validate_options(const ParentComposeOptions& options) {
  if (options.max_degree != kProductionMaxDegree) {
    throw ParentComposeError("max_degree is fixed at 24 for this seam");
  }
  if (!std::isfinite(options.alpha) || options.alpha <= 0.0) {
    throw ParentComposeError("alpha must be finite and positive");
  }
  if (options.sample_count != 8U && options.sample_count != 16U &&
      options.sample_count != 32U) {
    throw ParentComposeError("sample_count must be one of 8, 16, or 32");
  }
  if (options.thread_count == 0U || options.thread_count > 16U) {
    throw ParentComposeError("thread_count must be in the range [1, 16]");
  }
}

void validate_parent(ArrayView<std::int64_t> parent_members,
                     std::int64_t parent_anchor,
                     std::size_t corpus_count) {
  if (parent_members.empty()) {
    throw ParentComposeError("parent_members must not be empty");
  }
  for (std::size_t index = 0; index < parent_members.size(); ++index) {
    const std::int64_t object_id = parent_members[index];
    if (object_id < 0 || static_cast<std::uint64_t>(object_id) >=
                             static_cast<std::uint64_t>(corpus_count)) {
      throw ParentComposeError("parent member is outside CorpusView bounds");
    }
    if (index != 0U && parent_members[index - 1U] >= object_id) {
      throw ParentComposeError("parent_members must be strictly increasing");
    }
  }
  if (find_sorted_id(parent_members, parent_anchor) == kNpos) {
    throw ParentComposeError("parent_anchor is not a parent member");
  }
}

bool has_child_edge(const ChildView& child, std::size_t source_row,
                    std::size_t target_row) {
  const std::uint64_t begin = child.row_adj_ptr[source_row];
  const std::uint64_t end = child.row_adj_ptr[source_row + 1U];
  for (std::uint64_t offset = begin; offset < end; ++offset) {
    if (child.row_adj_target_row[static_cast<std::size_t>(offset)] ==
        static_cast<std::uint64_t>(target_row)) {
      return true;
    }
  }
  return false;
}

void validate_child(const ChildView& child,
                    ArrayView<std::int64_t> parent_members,
                    std::size_t corpus_count,
                    std::uint64_t* inherited_edge_refs) {
  if (child.child_id < 0) {
    throw ParentComposeError("child_id must be non-negative");
  }
  if (child.members.empty()) {
    throw ParentComposeError("child membership must not be empty");
  }
  if (child.row_object_ids.size() != child.members.size() ||
      child.row_adj_ptr.size() != child.members.size() + 1U ||
      child.witness_parent_row.size() != child.members.size()) {
    throw ParentComposeError(
        "child row and witness arrays do not match membership");
  }
  const std::size_t anchor_row = find_sorted_id(child.members, child.anchor);
  if (child.anchor < 0 || anchor_row == kNpos) {
    throw ParentComposeError("child anchor is not a child member");
  }
  if (child.row_adj_ptr[0] != 0U ||
      child.row_adj_ptr[child.members.size()] !=
          static_cast<std::uint64_t>(child.row_adj_target_row.size())) {
    throw ParentComposeError("child row_adj_ptr has an invalid terminal");
  }

  for (std::size_t index = 0; index < child.members.size(); ++index) {
    const std::int64_t object_id = child.members[index];
    if (object_id < 0 || static_cast<std::uint64_t>(object_id) >=
                             static_cast<std::uint64_t>(corpus_count)) {
      throw ParentComposeError("child member is outside CorpusView bounds");
    }
    if (find_sorted_id(parent_members, object_id) == kNpos) {
      throw ParentComposeError("child membership is not contained by parent");
    }
    if (index != 0U && child.members[index - 1U] >= object_id) {
      throw ParentComposeError("child members must be strictly increasing");
    }
    if (child.row_object_ids[index] != object_id) {
      throw ParentComposeError("child row_object_ids must match members");
    }
    const std::uint64_t begin = child.row_adj_ptr[index];
    const std::uint64_t end = child.row_adj_ptr[index + 1U];
    if (begin > end || end > static_cast<std::uint64_t>(
                             child.row_adj_target_row.size()) ||
        end - begin > static_cast<std::uint64_t>(kProductionMaxDegree)) {
      throw ParentComposeError("child row exceeds the fixed degree bound");
    }
    std::set<std::uint64_t> seen_targets;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::uint64_t target_row = child.row_adj_target_row[
          static_cast<std::size_t>(offset)];
      if (target_row >= static_cast<std::uint64_t>(child.members.size()) ||
          target_row == static_cast<std::uint64_t>(index)) {
        throw ParentComposeError("child edge target is invalid");
      }
      if (!seen_targets.insert(target_row).second) {
        throw ParentComposeError("child graph contains a duplicate edge");
      }
    }
  }

  if (child.witness_parent_row[anchor_row] != kNoWitnessParent) {
    throw ParentComposeError("child witness root must be its anchor");
  }
  std::vector<std::vector<std::size_t>> witness_children(child.members.size());
  for (std::size_t row = 0; row < child.members.size(); ++row) {
    if (row == anchor_row) continue;
    const std::uint64_t parent_row = child.witness_parent_row[row];
    if (parent_row >= static_cast<std::uint64_t>(child.members.size()) ||
        parent_row == static_cast<std::uint64_t>(row) ||
        !has_child_edge(child, static_cast<std::size_t>(parent_row), row)) {
      throw ParentComposeError("child witness parent is not an actual edge");
    }
    witness_children[static_cast<std::size_t>(parent_row)].push_back(row);
  }
  // Every witness edge was checked above.  A single BFS from the explicit
  // anchor proves all parent chains reach that root in O(n), avoiding the
  // former per-node parent backtrace (which was quadratic on long chains).
  std::vector<unsigned char> witness_reached(child.members.size(), 0U);
  std::queue<std::size_t> witness_pending;
  witness_reached[anchor_row] = 1U;
  witness_pending.push(anchor_row);
  std::size_t witness_reached_count = 1U;
  while (!witness_pending.empty()) {
    const std::size_t source = witness_pending.front();
    witness_pending.pop();
    for (const std::size_t target : witness_children[source]) {
      if (witness_reached[target] != 0U) {
        throw ParentComposeError("child witness is not a tree");
      }
      witness_reached[target] = 1U;
      ++witness_reached_count;
      witness_pending.push(target);
    }
  }
  if (witness_reached_count != child.members.size()) {
    throw ParentComposeError("child witness does not reach every member");
  }
  *inherited_edge_refs += static_cast<std::uint64_t>(
      child.row_adj_target_row.size());
}

std::vector<NormalizedChild> normalize_children(
    ArrayView<ChildView> children, ArrayView<std::int64_t> parent_members,
    std::size_t corpus_count, std::uint64_t* inherited_edge_refs) {
  if (children.empty()) {
    throw ParentComposeError("children must not be empty");
  }
  std::vector<NormalizedChild> normalized;
  normalized.reserve(children.size());
  for (const ChildView& child : children) {
    validate_child(child, parent_members, corpus_count, inherited_edge_refs);
    normalized.push_back(NormalizedChild{child});
  }
  std::sort(normalized.begin(), normalized.end(),
            [](const NormalizedChild& lhs, const NormalizedChild& rhs) {
              if (lhs.view.child_id != rhs.view.child_id) {
                return lhs.view.child_id < rhs.view.child_id;
              }
              return lhs.view.component_sort_key < rhs.view.component_sort_key;
            });
  for (std::size_t index = 1; index < normalized.size(); ++index) {
    if (normalized[index - 1U].view.child_id == normalized[index].view.child_id) {
      throw ParentComposeError("child IDs must be unique");
    }
  }
  return normalized;
}

std::vector<std::vector<std::size_t>> build_owners(
    ArrayView<std::int64_t> parent_members,
    const std::vector<NormalizedChild>& children) {
  std::vector<std::vector<std::size_t>> owners(parent_members.size());
  for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
    for (const std::int64_t object_id : children[child_index].view.members) {
      const std::size_t parent_row = find_sorted_id(parent_members, object_id);
      if (parent_row == kNpos) {
        throw ParentComposeError("child member has no parent row");
      }
      owners[parent_row].push_back(child_index);
    }
  }
  for (const auto& owner_list : owners) {
    if (owner_list.empty()) {
      throw ParentComposeError("every parent member needs an owner child");
    }
  }
  return owners;
}

std::vector<std::int64_t> child_row(const ChildView& child,
                                    std::int64_t object_id) {
  const std::size_t row = find_sorted_id(child.row_object_ids, object_id);
  if (row == kNpos) {
    throw ParentComposeError("child row is missing its member object");
  }
  const std::uint64_t begin = child.row_adj_ptr[row];
  const std::uint64_t end = child.row_adj_ptr[row + 1U];
  std::vector<std::int64_t> result;
  result.reserve(static_cast<std::size_t>(end - begin));
  for (std::uint64_t offset = begin; offset < end; ++offset) {
    const std::uint64_t target_row = child.row_adj_target_row[
        static_cast<std::size_t>(offset)];
    result.push_back(child.row_object_ids[static_cast<std::size_t>(target_row)]);
  }
  return result;
}

std::vector<std::int64_t> merge_row(
    std::int64_t object_id, const std::vector<std::size_t>& owner_list,
    const std::vector<NormalizedChild>& children) {
  std::vector<std::int64_t> result;
  std::set<std::int64_t> seen;
  for (const std::size_t child_index : owner_list) {
    const std::vector<std::int64_t> inherited =
        child_row(children[child_index].view, object_id);
    for (const std::int64_t target : inherited) {
      if (seen.insert(target).second) result.push_back(target);
    }
  }
  return result;
}

Reachability compute_reachability(
    ArrayView<std::int64_t> parent_members,
    const std::vector<std::vector<std::int64_t>>& rows,
    std::int64_t anchor) {
  Reachability result;
  result.reached.assign(parent_members.size(), 0U);
  result.witness_children.resize(parent_members.size());
  const std::size_t anchor_row = find_sorted_id(parent_members, anchor);
  if (anchor_row == kNpos) return result;
  std::queue<std::size_t> pending;
  result.reached[anchor_row] = 1U;
  result.reached_count = 1U;
  pending.push(anchor_row);
  while (!pending.empty()) {
    const std::size_t source_row = pending.front();
    pending.pop();
    for (const std::int64_t target : rows[source_row]) {
      const std::size_t target_row = find_sorted_id(parent_members, target);
      if (target_row == kNpos) {
        throw ParentComposeError("parent row contains an out-of-range target");
      }
      if (result.reached[target_row] == 0U) {
        result.reached[target_row] = 1U;
        result.witness_children[source_row].push_back(target);
        ++result.reached_count;
        pending.push(target_row);
      }
    }
  }
  return result;
}

void extend_reachability(
    ArrayView<std::int64_t> parent_members,
    const std::vector<std::vector<std::int64_t>>& rows,
    std::size_t seed_row, Reachability* reachability) {
  std::queue<std::size_t> pending;
  pending.push(seed_row);
  while (!pending.empty()) {
    const std::size_t source_row = pending.front();
    pending.pop();
    for (const std::int64_t target : rows[source_row]) {
      const std::size_t target_row = find_sorted_id(parent_members, target);
      if (target_row == kNpos) {
        throw ParentComposeError("incremental reachability saw an invalid target");
      }
      if (reachability->reached[target_row] == 0U) {
        reachability->reached[target_row] = 1U;
        reachability->witness_children[source_row].push_back(target);
        ++reachability->reached_count;
        pending.push(target_row);
      }
    }
  }
}

std::vector<DisconnectedPart> find_source_parts(
    ArrayView<std::int64_t> parent_members,
    const std::vector<std::vector<std::int64_t>>& rows,
    const Reachability& reachability,
    const std::vector<NormalizedChild>& children,
    const std::vector<std::vector<std::size_t>>& owners) {
  const std::size_t object_count = parent_members.size();
  std::vector<std::vector<std::size_t>> forward(object_count);
  std::vector<std::vector<std::size_t>> reverse(object_count);
  for (std::size_t source = 0; source < object_count; ++source) {
    if (reachability.reached[source] != 0U) continue;
    for (const std::int64_t target : rows[source]) {
      const std::size_t target_row = find_sorted_id(parent_members, target);
      if (target_row != kNpos && reachability.reached[target_row] == 0U) {
        forward[source].push_back(target_row);
        reverse[target_row].push_back(source);
      }
    }
  }

  std::vector<unsigned char> seen(object_count, 0U);
  std::vector<std::size_t> finish;
  finish.reserve(object_count - reachability.reached_count);
  for (std::size_t root = 0; root < object_count; ++root) {
    if (reachability.reached[root] != 0U || seen[root] != 0U) continue;
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    stack.emplace_back(root, 0U);
    seen[root] = 1U;
    while (!stack.empty()) {
      const std::size_t source = stack.back().first;
      std::size_t& position = stack.back().second;
      if (position < forward[source].size()) {
        const std::size_t target = forward[source][position];
        ++position;
        if (seen[target] == 0U) {
          seen[target] = 1U;
          stack.emplace_back(target, 0U);
        }
      } else {
        finish.push_back(source);
        stack.pop_back();
      }
    }
  }

  std::vector<std::size_t> component_of(object_count, kNpos);
  std::vector<std::vector<std::size_t>> components;
  for (auto iterator = finish.rbegin(); iterator != finish.rend(); ++iterator) {
    const std::size_t root = *iterator;
    if (component_of[root] != kNpos) continue;
    const std::size_t component_id = components.size();
    components.emplace_back();
    std::vector<std::size_t> stack{root};
    component_of[root] = component_id;
    while (!stack.empty()) {
      const std::size_t source = stack.back();
      stack.pop_back();
      components.back().push_back(source);
      for (const std::size_t target : reverse[source]) {
        if (component_of[target] == kNpos) {
          component_of[target] = component_id;
          stack.push_back(target);
        }
      }
    }
    std::sort(components.back().begin(), components.back().end());
  }

  std::vector<unsigned char> incoming(components.size(), 0U);
  for (std::size_t source = 0; source < object_count; ++source) {
    if (reachability.reached[source] != 0U) continue;
    for (const std::size_t target : forward[source]) {
      const std::size_t source_component = component_of[source];
      const std::size_t target_component = component_of[target];
      if (source_component != target_component) incoming[target_component] = 1U;
    }
  }

  std::vector<DisconnectedPart> parts;
  for (std::size_t component_id = 0; component_id < components.size();
       ++component_id) {
    if (incoming[component_id] != 0U) continue;
    DisconnectedPart part;
    part.rows = components[component_id];
    for (const std::size_t row : part.rows) {
      part.minimum_object = std::min(part.minimum_object, parent_members[row]);
      for (const std::size_t child_index : owners[row]) {
        const ChildView& child = children[child_index].view;
        const auto owner_key =
            std::make_tuple(child.component_sort_key, child.child_id);
        const auto current_key = std::make_tuple(part.sort_key, part.sort_child);
        if (owner_key < current_key) {
          part.sort_key = child.component_sort_key;
          part.sort_child = child.child_id;
        }
        const std::size_t anchor_row =
            find_sorted_id(parent_members, child.anchor);
        if (anchor_row != kNpos && component_of[anchor_row] == component_id &&
            child.anchor < part.entry) {
          part.entry = child.anchor;
        }
      }
    }
    if (part.entry == std::numeric_limits<std::int64_t>::max()) {
      part.entry = part.minimum_object;
    }
    parts.push_back(std::move(part));
  }
  std::sort(parts.begin(), parts.end(), [](const DisconnectedPart& lhs,
                                           const DisconnectedPart& rhs) {
    return std::make_tuple(lhs.sort_key, lhs.sort_child, lhs.entry,
                           lhs.minimum_object) <
           std::make_tuple(rhs.sort_key, rhs.sort_child, rhs.entry,
                           rhs.minimum_object);
  });
  return parts;
}

Group main_group(ArrayView<std::int64_t> parent_members,
                 std::int64_t parent_anchor, const Reachability& reachability,
                 const std::vector<NormalizedChild>& children,
                 const std::vector<std::vector<std::size_t>>& owners) {
  Group group;
  group.entry = parent_anchor;
  for (std::size_t row = 0; row < parent_members.size(); ++row) {
    if (reachability.reached[row] == 0U) continue;
    group.rows.push_back(row);
    group.minimum_object = std::min(group.minimum_object, parent_members[row]);
    for (const std::size_t child_index : owners[row]) {
      const ChildView& child = children[child_index].view;
      const auto candidate = std::make_tuple(child.component_sort_key,
                                              child.child_id);
      const auto current =
          std::make_tuple(group.sort_key, group.sort_child);
      if (candidate < current) {
        group.sort_key = child.component_sort_key;
        group.sort_child = child.child_id;
      }
    }
  }
  return group;
}

Group group_from_part(const DisconnectedPart& part) {
  Group group;
  group.rows = part.rows;
  group.sort_key = part.sort_key;
  group.sort_child = part.sort_child;
  group.entry = part.entry;
  group.minimum_object = part.minimum_object;
  return group;
}

void build_reverse_eligible(
    ArrayView<std::int64_t> parent_members,
    const std::vector<std::vector<std::int64_t>>& rows, Group* group) {
  const std::size_t entry_row = find_sorted_id(parent_members, group->entry);
  const auto entry_iterator =
      std::lower_bound(group->rows.begin(), group->rows.end(), entry_row);
  if (entry_row == kNpos || entry_iterator == group->rows.end() ||
      *entry_iterator != entry_row) {
    throw ParentComposeError("group entry is not in its group");
  }
  // Keep this graph local to the group.  No parent-sized reverse graph or
  // full-parent BFS is allocated once per disconnected component.
  std::vector<std::vector<std::size_t>> reverse(group->rows.size());
  for (std::size_t source_index = 0; source_index < group->rows.size();
       ++source_index) {
    const std::size_t source = group->rows[source_index];
    for (const std::int64_t target : rows[source]) {
      const std::size_t target_row = find_sorted_id(parent_members, target);
      const auto target_iterator = std::lower_bound(
          group->rows.begin(), group->rows.end(), target_row);
      if (target_row != kNpos && target_iterator != group->rows.end() &&
          *target_iterator == target_row) {
        const std::size_t target_index = static_cast<std::size_t>(
            target_iterator - group->rows.begin());
        reverse[target_index].push_back(source_index);
      }
    }
  }
  std::vector<unsigned char> seen(group->rows.size(), 0U);
  std::queue<std::size_t> pending;
  const std::size_t entry_index = static_cast<std::size_t>(
      entry_iterator - group->rows.begin());
  seen[entry_index] = 1U;
  pending.push(entry_index);
  while (!pending.empty()) {
    const std::size_t target = pending.front();
    pending.pop();
    for (const std::size_t source_index : reverse[target]) {
      if (seen[source_index] == 0U) {
        seen[source_index] = 1U;
        pending.push(source_index);
      }
    }
  }
  for (std::size_t row_index = 0; row_index < group->rows.size(); ++row_index) {
    if (seen[row_index] != 0U) {
      group->reverse_eligible_rows.push_back(group->rows[row_index]);
    }
  }
  if (group->reverse_eligible_rows.empty()) {
    throw ParentComposeError("group has no reverse-reachable sample");
  }
}

std::uint64_t splitmix64(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

std::uint64_t stable_priority(std::uint64_t seed, std::uint64_t parent_identity,
                              std::int64_t object_id,
                              std::uint64_t purpose) {
  std::uint64_t value = splitmix64(seed ^ UINT64_C(0x243f6a8885a308d3));
  value ^= splitmix64(parent_identity + UINT64_C(0x13198a2e03707344));
  value ^= splitmix64(static_cast<std::uint64_t>(object_id) + purpose);
  return splitmix64(value ^ purpose);
}

std::vector<std::int64_t> select_samples(
    const std::vector<std::int64_t>& candidates, std::int64_t mandatory,
    std::uint32_t sample_count, std::uint64_t seed,
    std::uint64_t parent_identity, std::uint64_t purpose) {
  std::vector<std::int64_t> unique = candidates;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  if (unique.empty() ||
      !std::binary_search(unique.begin(), unique.end(), mandatory)) {
    throw ParentComposeError("sample mandatory object is not in its group");
  }
  std::vector<std::int64_t> result{mandatory};
  std::vector<std::int64_t> remainder;
  for (const std::int64_t object_id : unique) {
    if (object_id != mandatory) remainder.push_back(object_id);
  }
  std::sort(remainder.begin(), remainder.end(),
            [seed, parent_identity, purpose](std::int64_t lhs,
                                              std::int64_t rhs) {
              const auto left = stable_priority(seed, parent_identity, lhs, purpose);
              const auto right = stable_priority(seed, parent_identity, rhs, purpose);
              if (left != right) return left < right;
              return lhs < rhs;
            });
  const std::size_t limit = static_cast<std::size_t>(sample_count);
  for (const std::int64_t object_id : remainder) {
    if (result.size() >= limit) break;
    result.push_back(object_id);
  }
  return result;
}

double squared_distance(const numeric_detail::NumericHotContext& numeric_context,
                        std::int64_t lhs,
                        std::int64_t rhs) {
  return numeric_detail::squared_l2_f32_f64_seq_hot(numeric_context, lhs, rhs);
}

std::vector<std::int64_t> robust_prune(
    std::int64_t source, const std::vector<std::int64_t>& candidates,
    const std::set<std::int64_t>& protected_targets,
    const numeric_detail::NumericHotContext& numeric_context,
    double alpha_squared_value) {
  std::set<std::int64_t> unique(candidates.begin(), candidates.end());
  unique.erase(source);
  for (const std::int64_t target : protected_targets) {
    if (target != source) unique.insert(target);
  }
  if (protected_targets.size() >
      static_cast<std::size_t>(kProductionMaxDegree)) {
    throw ParentComposeError("protected edges exceed max_degree=24");
  }
  std::vector<std::int64_t> ranked(unique.begin(), unique.end());
  std::sort(ranked.begin(), ranked.end(),
            [source, &numeric_context](std::int64_t lhs,
                                       std::int64_t rhs) {
              const double left =
                  squared_distance(numeric_context, source, lhs);
              const double right =
                  squared_distance(numeric_context, source, rhs);
              if (left != right) return left < right;
              return lhs < rhs;
            });

  // Protected witness/bridge edges consume the fixed row budget first.  A
  // caller must check the protected set before reaching this helper; keeping
  // the subtraction guarded makes the ordinary (non-protected) candidate
  // selection fail closed instead of wrapping size_t when a row is full.
  const std::size_t slots =
      protected_targets.size() >= static_cast<std::size_t>(kProductionMaxDegree)
          ? 0U
          : static_cast<std::size_t>(kProductionMaxDegree) -
                protected_targets.size();
  std::vector<std::int64_t> selected;
  selected.reserve(slots);
  for (const std::int64_t target : ranked) {
    if (protected_targets.find(target) != protected_targets.end()) continue;
    const double source_target =
        squared_distance(numeric_context, source, target);
    bool occluded = false;
    for (const std::int64_t chosen : selected) {
      // Python's L2Distance compares Euclidean distances.  This code stores
      // squared L2, so the equivalent alpha factor is alpha^2.
      if (numeric_detail::alpha_product_hot(
              alpha_squared_value,
              squared_distance(numeric_context, chosen, target)) <=
          source_target) {
        occluded = true;
        break;
      }
    }
    if (!occluded) selected.push_back(target);
    if (selected.size() >= slots) break;
  }

  std::vector<std::int64_t> result;
  result.reserve(protected_targets.size() + selected.size());
  result.insert(result.end(), protected_targets.begin(), protected_targets.end());
  result.insert(result.end(), selected.begin(), selected.end());
  std::sort(result.begin(), result.end(),
            [source, &numeric_context](std::int64_t lhs,
                                       std::int64_t rhs) {
              const double left =
                  squared_distance(numeric_context, source, lhs);
              const double right =
                  squared_distance(numeric_context, source, rhs);
              if (left != right) return left < right;
              return lhs < rhs;
            });
  return result;
}

bool has_protected_slot(
    std::size_t row, std::int64_t target, const Reachability& reachability,
    const std::vector<std::set<std::int64_t>>& protected_bridges) {
  std::set<std::int64_t> protected_targets;
  protected_targets.insert(reachability.witness_children[row].begin(),
                           reachability.witness_children[row].end());
  protected_targets.insert(protected_bridges[row].begin(),
                           protected_bridges[row].end());
  if (protected_targets.find(target) != protected_targets.end()) return true;
  return protected_targets.size() <
         static_cast<std::size_t>(kProductionMaxDegree);
}

void audit_rows(const CorpusView& corpus, ArrayView<std::int64_t> parent_members,
                const std::vector<std::vector<std::int64_t>>& rows,
                std::int64_t anchor, std::uint64_t* unreachable_after_audit) {
  for (std::size_t source = 0; source < rows.size(); ++source) {
    if (rows[source].size() > static_cast<std::size_t>(kProductionMaxDegree)) {
      throw ParentComposeError("parent row exceeds max_degree after prune");
    }
    std::set<std::int64_t> seen;
    for (const std::int64_t target : rows[source]) {
      if (target == parent_members[source] ||
          target < 0 || static_cast<std::uint64_t>(target) >= corpus.object_count ||
          find_sorted_id(parent_members, target) == kNpos ||
          !seen.insert(target).second) {
        throw ParentComposeError("parent graph failed range or duplicate audit");
      }
    }
  }
  const Reachability reachability =
      compute_reachability(parent_members, rows, anchor);
  *unreachable_after_audit = static_cast<std::uint64_t>(
      parent_members.size() - reachability.reached_count);
  if (*unreachable_after_audit != 0U) {
    throw ParentComposeError("parent graph failed directed reachability audit");
  }
}

}  // namespace

ParentComposeResult compose_parent(
    const CorpusView& corpus, ArrayView<std::int64_t> parent_members,
    std::int64_t parent_anchor, ArrayView<ChildView> children,
    const ParentComposeOptions& options) {
  validate_options(options);
  const numeric_detail::NumericHotContext numeric_context =
      numeric_detail::make_numeric_hot_context(
          corpus.vectors, corpus.vector_dim, corpus.object_count,
          corpus.finite_values_prevalidated);
  const double alpha_squared_value =
      bounded_blowup::alpha_squared(options.alpha);
  const std::size_t corpus_count = corpus_object_count(corpus);
  validate_parent(parent_members, parent_anchor, corpus_count);

  ParentComposeResult result;
  result.evidence.owner_rows = static_cast<std::uint64_t>(parent_members.size());
  std::uint64_t inherited_edge_refs = 0U;
  const std::vector<NormalizedChild> normalized = normalize_children(
      children, parent_members, corpus_count, &inherited_edge_refs);
  result.evidence.inherited_edge_refs = inherited_edge_refs;
  const auto owners = build_owners(parent_members, normalized);

  result.child_identities.reserve(normalized.size());
  for (const NormalizedChild& child : normalized) {
    result.child_identities.push_back(
        ParentChildIdentity{child.view.child_id, child.view.component_sort_key});
  }
  result.owner_ptr.push_back(0U);
  for (std::size_t row = 0; row < parent_members.size(); ++row) {
    for (const std::size_t child_index : owners[row]) {
      const std::size_t child_row = find_sorted_id(
          normalized[child_index].view.row_object_ids, parent_members[row]);
      if (child_row == kNpos) {
        throw ParentComposeError("owner child has no local row");
      }
      result.owner_child_index.push_back(static_cast<std::uint64_t>(child_index));
      result.owner_child_row.push_back(static_cast<std::uint64_t>(child_row));
    }
    result.owner_ptr.push_back(
        static_cast<std::uint64_t>(result.owner_child_index.size()));
  }
  for (const auto& owner_list : owners) {
    if (owner_list.size() > 1U) ++result.evidence.overlap_rows;
  }

  std::vector<std::vector<std::int64_t>> rows(parent_members.size());
  std::vector<std::uint8_t> reasons(parent_members.size(), 0U);

  const std::size_t worker_count = std::min(
      static_cast<std::size_t>(options.thread_count), parent_members.size());
  std::vector<std::exception_ptr> worker_errors(worker_count);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  try {
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([worker, worker_count, &parent_members, &owners,
                            &normalized, &rows, &reasons, &worker_errors,
                            &numeric_context]() {
        try {
          // The parent merge itself is logical, but keeping the same contract
          // boundary on every worker prevents a future numeric row operation
          // from reintroducing per-distance FE/shape scans.
          numeric_detail::validate_numeric_hot_context(numeric_context);
          for (std::size_t row = worker; row < parent_members.size();
               row += worker_count) {
            rows[row] = merge_row(parent_members[row], owners[row], normalized);
            if (owners[row].size() > 1U) {
              reasons[row] |= kRowReasonOverlap;
            }
          }
        } catch (...) {
          worker_errors[worker] = std::current_exception();
        }
      });
    }
  } catch (...) {
    for (std::thread& worker : workers) worker.join();
    throw;
  }
  for (std::thread& worker : workers) worker.join();
  for (const std::exception_ptr& error : worker_errors) {
    if (error) std::rethrow_exception(error);
  }

  // A point shared by several children inherits the union of their neighbour
  // rows.  Reduce only that merged row to the production degree before the
  // parent-level reachability proof.  Child witnesses validate each input
  // graph, but they are not all simultaneously protected in the parent: the
  // parent constructs and protects its own witness after this competitive
  // prune.
  for (std::size_t row = 0; row < rows.size(); ++row) {
    if (rows[row].size() <= static_cast<std::size_t>(kProductionMaxDegree)) {
      continue;
    }
    rows[row] = robust_prune(parent_members[row], rows[row], {}, numeric_context,
                             alpha_squared_value);
    reasons[row] |= kRowReasonWitness;
    reasons[row] |= kRowReasonPruned;
  }

  const Reachability natural_reachability =
      compute_reachability(parent_members, rows, parent_anchor);
  result.evidence.natural_unreachable = static_cast<std::uint64_t>(
      parent_members.size() - natural_reachability.reached_count);
  const std::vector<DisconnectedPart> parts = find_source_parts(
      parent_members, rows, natural_reachability, normalized, owners);
  result.evidence.disconnected_parts = static_cast<std::uint64_t>(parts.size());

  std::vector<Group> groups;
  groups.reserve(parts.size() + 1U);
  groups.push_back(main_group(parent_members, parent_anchor, natural_reachability,
                              normalized, owners));
  for (const DisconnectedPart& part : parts) groups.push_back(group_from_part(part));
  // The anchor group is always first so every bridge is an adjacent monotone
  // expansion from the already reachable prefix.  Disconnected groups retain
  // their frozen component_sort_key/child-ID order.
  std::sort(groups.begin() + 1U, groups.end(), [](const Group& lhs,
                                                  const Group& rhs) {
    return std::make_tuple(lhs.sort_key, lhs.sort_child, lhs.entry,
                           lhs.minimum_object) <
           std::make_tuple(rhs.sort_key, rhs.sort_child, rhs.entry,
                           rhs.minimum_object);
  });
  result.evidence.group_count = static_cast<std::uint64_t>(groups.size());
  for (Group& group : groups) {
    build_reverse_eligible(parent_members, rows, &group);
  }

  std::vector<std::set<std::int64_t>> protected_bridges(parent_members.size());
  Reachability incremental_reachability = natural_reachability;
  for (std::size_t group_index = 1U; group_index < groups.size(); ++group_index) {
    const Group& source_group = groups[group_index - 1U];
    const Group& target_group = groups[group_index];
    std::vector<std::int64_t> source_pool;
    std::vector<std::int64_t> target_pool;
    for (const std::size_t row : source_group.reverse_eligible_rows) {
      source_pool.push_back(parent_members[row]);
    }
    for (const std::size_t row : target_group.reverse_eligible_rows) {
      target_pool.push_back(parent_members[row]);
    }
    const std::vector<std::int64_t> source_samples = select_samples(
        source_pool, source_group.entry, options.sample_count, options.seed,
        options.parent_identity,
        UINT64_C(0x5352430000000000) + static_cast<std::uint64_t>(group_index));
    const std::vector<std::int64_t> target_samples = select_samples(
        target_pool, target_group.entry, options.sample_count, options.seed,
        options.parent_identity,
        UINT64_C(0x5447520000000000) + static_cast<std::uint64_t>(group_index));
    result.evidence.sample_reverse_verified += static_cast<std::uint64_t>(
        source_samples.size() + target_samples.size());

    bool found_pair = false;
    double best_distance = std::numeric_limits<double>::infinity();
    std::int64_t best_source = -1;
    std::int64_t best_target = -1;
    std::uint64_t pair_evaluations = 0U;
    for (const std::int64_t source : source_samples) {
      for (const std::int64_t target : target_samples) {
        const std::size_t source_candidate_row =
            find_sorted_id(parent_members, source);
        const std::size_t target_candidate_row =
            find_sorted_id(parent_members, target);
        if (source_candidate_row == kNpos || target_candidate_row == kNpos ||
            !has_protected_slot(source_candidate_row, target,
                                incremental_reachability, protected_bridges) ||
            !has_protected_slot(target_candidate_row, source,
                                incremental_reachability, protected_bridges)) {
          continue;
        }
        ++pair_evaluations;
        const double distance =
            squared_distance(numeric_context, source, target);
        if (!found_pair || std::tie(distance, source, target) <
                               std::tie(best_distance, best_source, best_target)) {
          found_pair = true;
          best_distance = distance;
          best_source = source;
          best_target = target;
        }
      }
    }
    if (!found_pair) {
      // The sampled pool is a speed bound, not a reason to overfill a
      // protected row.  If every sampled pair uses a full witness/bridge
      // endpoint, choose the first deterministic pair with two free slots from
      // the already verified reverse-eligible rows.  This is a linear scan of
      // the group endpoints and performs no additional distance search.
      for (const std::size_t source_candidate_row :
           source_group.reverse_eligible_rows) {
        if (source_candidate_row >= parent_members.size()) continue;
        const std::int64_t source = parent_members[source_candidate_row];
        for (const std::size_t target_candidate_row :
             target_group.reverse_eligible_rows) {
          if (target_candidate_row >= parent_members.size()) continue;
          const std::int64_t target = parent_members[target_candidate_row];
          if (!has_protected_slot(source_candidate_row, target,
                                  incremental_reachability,
                                  protected_bridges) ||
              !has_protected_slot(target_candidate_row, source,
                                  incremental_reachability,
                                  protected_bridges)) {
            continue;
          }
          found_pair = true;
          best_source = source;
          best_target = target;
          break;
        }
        if (found_pair) break;
      }
    }
    if (!found_pair) {
      throw ParentComposeError(
          "adjacent group has no bridge pair with protected degree slots");
    }
    result.evidence.sample_distance_evaluations += pair_evaluations;
    const std::size_t source_row = find_sorted_id(parent_members, best_source);
    const std::size_t target_row = find_sorted_id(parent_members, best_target);
    if (source_row == kNpos || target_row == kNpos) {
      throw ParentComposeError("sampled bridge is outside parent membership");
    }
    if (!has_protected_slot(source_row, best_target,
                           incremental_reachability, protected_bridges) ||
        !has_protected_slot(target_row, best_source,
                            incremental_reachability, protected_bridges)) {
      throw ParentComposeError(
          "selected bridge pair has no protected degree slots");
    }
    if (std::find(rows[source_row].begin(), rows[source_row].end(), best_target) ==
        rows[source_row].end()) {
      rows[source_row].push_back(best_target);
    }
    if (std::find(rows[target_row].begin(), rows[target_row].end(), best_source) ==
        rows[target_row].end()) {
      rows[target_row].push_back(best_source);
    }
    protected_bridges[source_row].insert(best_target);
    protected_bridges[target_row].insert(best_source);
    reasons[source_row] |= kRowReasonBridge;
    reasons[target_row] |= kRowReasonBridge;
    extend_reachability(parent_members, rows, source_row,
                        &incremental_reachability);
    extend_reachability(parent_members, rows, target_row,
                        &incremental_reachability);
    result.evidence.connections.push_back(ParentConnectionEvidence{
        static_cast<std::uint64_t>(group_index - 1U),
        static_cast<std::uint64_t>(group_index), best_source, best_target,
        pair_evaluations});
  }
  result.evidence.connection_pairs =
      static_cast<std::uint64_t>(result.evidence.connections.size());

  // Only rows with a reason are ever re-pruned.  Protect the witness of the
  // already-pruned parent graph plus the new bridge edges; protecting every
  // historical child witness here would recreate the unbounded overlap row.
  for (std::size_t row = 0; row < rows.size(); ++row) {
    if (reasons[row] == 0U ||
        rows[row].size() <= static_cast<std::size_t>(kProductionMaxDegree)) {
      continue;
    }
    std::set<std::int64_t> protected_targets;
    protected_targets.insert(incremental_reachability.witness_children[row].begin(),
                             incremental_reachability.witness_children[row].end());
    protected_targets.insert(protected_bridges[row].begin(),
                             protected_bridges[row].end());
    if (protected_targets.size() >
        static_cast<std::size_t>(kProductionMaxDegree)) {
      throw ParentComposeError("final protected witness/bridge set exceeds 24");
    }
    rows[row] = robust_prune(parent_members[row], rows[row], protected_targets,
                             numeric_context, alpha_squared_value);
    reasons[row] |= kRowReasonPruned;
  }

  // Evidence is derived from the final reason mask, so a bridge endpoint is
  // always dirty even when its bridge happened to duplicate an inherited edge.
  result.evidence.non_dirty_rows = 0U;
  result.evidence.dirty_rows = 0U;
  result.evidence.pruned_rows = 0U;
  for (const std::uint8_t reason : reasons) {
    if (reason == 0U) {
      ++result.evidence.non_dirty_rows;
    } else {
      ++result.evidence.dirty_rows;
    }
    if ((reason & kRowReasonPruned) != 0U) ++result.evidence.pruned_rows;
  }
  result.row_reason_mask = reasons;

  result.evidence.max_observed_degree = 0U;
  result.evidence.overflow_rows = 0U;
  for (const std::vector<std::int64_t>& row : rows) {
    result.evidence.max_observed_degree = std::max(
        result.evidence.max_observed_degree,
        static_cast<std::uint64_t>(row.size()));
    if (row.size() > static_cast<std::size_t>(kProductionMaxDegree)) {
      ++result.evidence.overflow_rows;
    }
  }

  audit_rows(corpus, parent_members, rows, parent_anchor,
             &result.evidence.unreachable_after_audit);

  result.overlay_row_ptr.push_back(0U);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    if (reasons[row] == 0U) continue;
    result.overlay_source_ids.push_back(parent_members[row]);
    result.overlay_target_ids.insert(result.overlay_target_ids.end(), rows[row].begin(),
                                     rows[row].end());
    result.overlay_row_ptr.push_back(
        static_cast<std::uint64_t>(result.overlay_target_ids.size()));
  }
  return result;
}

}  // namespace bounded_blowup

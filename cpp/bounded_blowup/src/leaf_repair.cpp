// SPDX-License-Identifier: MIT
#include "bounded_blowup/leaf_repair.hpp"

#include "bounded_blowup/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bounded_blowup {
namespace {

using Row = std::vector<std::size_t>;
using MemberRowMap = std::unordered_map<std::int64_t, std::size_t>;

struct ValidatedLeaf {
  // Target rows are stored in the input CSR's edge order.  Keeping this flat
  // lets the connected fast path avoid materializing one heap-backed Row per
  // input row; mutable rows are materialized only when a bridge is needed.
  std::vector<std::size_t> target_rows;
};

std::uint64_t double_bits(const double value) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "leaf repair requires binary64 doubles");
  std::uint64_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct SccResult {
  std::vector<std::size_t> component;
  std::vector<std::vector<std::size_t>> members;
};

[[noreturn]] void fail(const char* message) {
  throw LeafRepairError(message);
}

void validate_options(const LeafRepairOptions& options) {
  if (options.max_degree != kLeafRepairMaxDegree) {
    fail("leaf repair max_degree must be exactly 24");
  }
  if (options.sample_budget != kLeafRepairSampleBudget) {
    fail("leaf repair sample_budget must be exactly 8");
  }
  if (!std::isfinite(options.alpha) || options.alpha <= 0.0) {
    fail("leaf repair alpha must be finite and positive");
  }
}

void validate_corpus(const CorpusView& corpus) {
  numeric_detail::require_round_to_nearest();
  if (!corpus.finite_values_prevalidated) {
    fail("leaf repair requires a prevalidated finite corpus");
  }
  if (corpus.vector_dim == 0U || corpus.object_count == 0U) {
    fail("leaf repair corpus shape must be non-empty");
  }
  if (corpus.object_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    fail("leaf repair corpus IDs exceed signed ID range");
  }
  const std::uint64_t dimension = static_cast<std::uint64_t>(corpus.vector_dim);
  if (corpus.object_count > std::numeric_limits<std::uint64_t>::max() /
                                dimension) {
    fail("leaf repair corpus shape overflows");
  }
  const std::uint64_t expected = corpus.object_count * dimension;
  if (expected > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      corpus.vectors.size() != static_cast<std::size_t>(expected) ||
      (corpus.vectors.data() == nullptr && expected != 0U)) {
    fail("leaf repair corpus vector shape is invalid");
  }
  // Values are covered by the caller's immutable finite-values receipt.  A
  // repair must not turn that receipt into an O(corpus_size * dimension)
  // scan for every leaf.
}

std::size_t row_for_id(const std::vector<std::int64_t>& members,
                       std::int64_t object_id) {
  const auto found = std::lower_bound(members.begin(), members.end(), object_id);
  if (found == members.end() || *found != object_id) {
    fail("leaf repair edge target is outside the leaf members");
  }
  return static_cast<std::size_t>(found - members.begin());
}

ValidatedLeaf validate_leaf(const CorpusView& corpus,
                            const LeafBuildResult& leaf) {
  const std::size_t row_count = leaf.row_object_ids.size();
  if (row_count == 0U || leaf.row_object_ids.data() == nullptr) {
    fail("leaf repair members must be non-empty");
  }
  if (leaf.row_ptr.size() != row_count + 1U || leaf.row_ptr.empty() ||
      leaf.row_ptr.front() != 0U ||
      leaf.row_ptr.back() != leaf.target_ids.size()) {
      fail("leaf repair input CSR shape is invalid");
  }

  MemberRowMap member_rows;
  member_rows.reserve(row_count);
  std::int64_t previous = -1;
  for (std::size_t row = 0U; row < row_count; ++row) {
    const std::int64_t object_id = leaf.row_object_ids[row];
    if (object_id < 0 ||
        static_cast<std::uint64_t>(object_id) >= corpus.object_count ||
        (row != 0U && object_id <= previous)) {
      fail("leaf repair members must be sorted, unique, and in corpus bounds");
    }
    previous = object_id;
    member_rows.emplace(object_id, row);
  }

  ValidatedLeaf validated;
  validated.target_rows.resize(leaf.target_ids.size());
  const std::size_t no_row = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> seen_target(row_count, no_row);
  for (std::size_t row = 0U; row < row_count; ++row) {
    const std::int64_t object_id = leaf.row_object_ids[row];
    if (leaf.row_ptr[row] > leaf.row_ptr[row + 1U] ||
        leaf.row_ptr[row + 1U] > leaf.target_ids.size() ||
        leaf.row_ptr[row + 1U] - leaf.row_ptr[row] > kLeafRepairMaxDegree) {
      fail("leaf repair input CSR row is invalid or exceeds degree 24");
    }
    for (std::uint64_t offset = leaf.row_ptr[row];
         offset < leaf.row_ptr[row + 1U]; ++offset) {
      const std::size_t edge = static_cast<std::size_t>(offset);
      const std::int64_t target_id = leaf.target_ids[edge];
      if (target_id == object_id ||
          target_id < 0 ||
          static_cast<std::uint64_t>(target_id) >= corpus.object_count) {
        fail("leaf repair input CSR has an invalid or self edge");
      }
      const auto found = member_rows.find(target_id);
      if (found == member_rows.end()) {
        fail("leaf repair edge target is outside the leaf members");
      }
      const std::size_t target_row = found->second;
      if (seen_target[target_row] == row) {
        fail("leaf repair input CSR has a duplicate edge");
      }
      seen_target[target_row] = row;
      validated.target_rows[edge] = target_row;
    }
  }
  return validated;
}

std::vector<Row> materialize_adjacency(const std::vector<std::uint64_t>& row_ptr,
                                       const ValidatedLeaf& validated) {
  if (row_ptr.empty()) fail("leaf repair input CSR row pointer is empty");
  const std::size_t row_count = row_ptr.size() - 1U;
  std::vector<Row> adjacency(row_count);
  for (std::size_t row = 0U; row < row_count; ++row) {
    const std::size_t begin = static_cast<std::size_t>(row_ptr[row]);
    const std::size_t end = static_cast<std::size_t>(row_ptr[row + 1U]);
    adjacency[row].reserve(end - begin);
    for (std::size_t edge = begin; edge < end; ++edge) {
      adjacency[row].push_back(validated.target_rows[edge]);
    }
  }
  return adjacency;
}

std::vector<std::uint8_t> directed_bfs(
    const std::vector<Row>& adjacency, std::size_t anchor,
    std::vector<std::uint64_t>* witness_parent,
    std::uint64_t* edge_scans) {
  const std::size_t row_count = adjacency.size();
  std::vector<std::uint8_t> reached(row_count, 0U);
  witness_parent->assign(row_count, kNoWitnessParent);
  std::vector<std::size_t> pending;
  pending.reserve(row_count);
  reached[anchor] = 1U;
  pending.push_back(anchor);
  for (std::size_t cursor = 0U; cursor < pending.size(); ++cursor) {
    const std::size_t source = pending[cursor];
    for (const std::size_t target : adjacency[source]) {
      ++(*edge_scans);
      if (reached[target] == 0U) {
        reached[target] = 1U;
        (*witness_parent)[target] = static_cast<std::uint64_t>(source);
        pending.push_back(target);
      }
    }
  }
  return reached;
}

std::vector<std::uint8_t> directed_bfs_flat(
    const std::vector<std::uint64_t>& row_ptr,
    const std::vector<std::size_t>& target_rows, std::size_t anchor,
    std::vector<std::uint64_t>* witness_parent,
    std::uint64_t* edge_scans) {
  const std::size_t row_count = row_ptr.size() - 1U;
  std::vector<std::uint8_t> reached(row_count, 0U);
  witness_parent->assign(row_count, kNoWitnessParent);
  std::vector<std::size_t> pending;
  pending.reserve(row_count);
  reached[anchor] = 1U;
  pending.push_back(anchor);
  for (std::size_t cursor = 0U; cursor < pending.size(); ++cursor) {
    const std::size_t source = pending[cursor];
    const std::size_t begin = static_cast<std::size_t>(row_ptr[source]);
    const std::size_t end = static_cast<std::size_t>(row_ptr[source + 1U]);
    for (std::size_t edge = begin; edge < end; ++edge) {
      ++(*edge_scans);
      const std::size_t target = target_rows[edge];
      if (reached[target] == 0U) {
        reached[target] = 1U;
        (*witness_parent)[target] = static_cast<std::uint64_t>(source);
        pending.push_back(target);
      }
    }
  }
  return reached;
}

std::vector<std::vector<std::size_t>> witness_children(
    const std::vector<std::uint64_t>& witness_parent) {
  std::vector<std::vector<std::size_t>> children(witness_parent.size());
  for (std::size_t row = 0U; row < witness_parent.size(); ++row) {
    if (witness_parent[row] != kNoWitnessParent) {
      const std::uint64_t parent = witness_parent[row];
      if (parent >= witness_parent.size()) {
        fail("leaf repair witness parent is out of bounds");
      }
      children[static_cast<std::size_t>(parent)].push_back(row);
    }
  }
  return children;
}

SccResult unreachable_sccs(const std::vector<Row>& adjacency,
                           const std::vector<std::uint8_t>& reached) {
  const std::size_t row_count = adjacency.size();
  std::vector<Row> reverse(row_count);
  for (std::size_t source = 0U; source < row_count; ++source) {
    if (reached[source] != 0U) continue;
    for (const std::size_t target : adjacency[source]) {
      if (reached[target] == 0U) reverse[target].push_back(source);
    }
  }

  std::vector<std::uint8_t> seen(row_count, 0U);
  std::vector<std::size_t> finish;
  finish.reserve(row_count);
  for (std::size_t root = 0U; root < row_count; ++root) {
    if (reached[root] != 0U || seen[root] != 0U) continue;
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    stack.emplace_back(root, 0U);
    seen[root] = 1U;
    while (!stack.empty()) {
      std::size_t& next_edge = stack.back().second;
      const std::size_t source = stack.back().first;
      if (next_edge == adjacency[source].size()) {
        finish.push_back(source);
        stack.pop_back();
        continue;
      }
      const std::size_t target = adjacency[source][next_edge];
      ++next_edge;
      if (reached[target] == 0U && seen[target] == 0U) {
        seen[target] = 1U;
        stack.emplace_back(target, 0U);
      }
    }
  }

  SccResult result;
  result.component.assign(row_count, std::numeric_limits<std::size_t>::max());
  for (std::size_t finish_index = finish.size(); finish_index > 0U;
       --finish_index) {
    const std::size_t root = finish[finish_index - 1U];
    if (result.component[root] != std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    const std::size_t component_id = result.members.size();
    result.members.emplace_back();
    std::vector<std::size_t> stack{root};
    result.component[root] = component_id;
    while (!stack.empty()) {
      const std::size_t source = stack.back();
      stack.pop_back();
      result.members[component_id].push_back(source);
      for (const std::size_t target : reverse[source]) {
        if (result.component[target] ==
            std::numeric_limits<std::size_t>::max()) {
          result.component[target] = component_id;
          stack.push_back(target);
        }
      }
    }
    // Keep the iterative SCC traversal order.  Canonical bridge samples are
    // selected with a bounded eight-slot insertion below, so a full member
    // sort is unnecessary.
  }
  return result;
}

std::vector<std::size_t> source_components(
    const std::vector<Row>& adjacency, const std::vector<std::uint8_t>& reached,
    const SccResult& scc) {
  const std::size_t component_count = scc.members.size();
  std::vector<std::uint8_t> has_incoming(component_count, 0U);
  for (std::size_t source = 0U; source < adjacency.size(); ++source) {
    if (reached[source] != 0U) continue;
    for (const std::size_t target : adjacency[source]) {
      if (reached[target] == 0U && scc.component[source] != scc.component[target]) {
        has_incoming[scc.component[target]] = 1U;
      }
    }
  }
  std::vector<std::size_t> output;
  for (std::size_t component = 0U; component < component_count; ++component) {
    if (has_incoming[component] == 0U) output.push_back(component);
  }
  std::sort(output.begin(), output.end(), [&](std::size_t left,
                                              std::size_t right) {
    return *std::min_element(scc.members[left].begin(), scc.members[left].end()) <
           *std::min_element(scc.members[right].begin(), scc.members[right].end());
  });
  return output;
}

std::vector<std::size_t> deterministic_target_samples(
    const std::vector<std::size_t>& component) {
  if (component.empty()) fail("leaf repair source SCC is empty");
  const std::size_t lowest =
      *std::min_element(component.begin(), component.end());
  std::vector<std::size_t> samples{lowest};
  for (const std::size_t row : component) {
    if (row == lowest) continue;
    const auto position = std::lower_bound(samples.begin(), samples.end(), row);
    if (samples.size() < kLeafRepairSampleBudget) {
      samples.insert(position, row);
    } else if (row < samples.back()) {
      samples.insert(position, row);
      samples.pop_back();
    }
  }
  return samples;
}

void extend_witness(const std::vector<Row>& adjacency, std::size_t source,
                    std::size_t target, std::vector<std::uint8_t>* reached,
                    std::vector<std::uint64_t>* witness_parent,
                    std::vector<std::vector<std::size_t>>* children,
                    std::set<std::size_t>* eligible_sources,
                    std::uint64_t* bfs_edge_scans) {
  if ((*reached)[source] == 0U || (*reached)[target] != 0U) {
    fail("leaf repair bridge does not connect a reachable source to an unknown target");
  }
  (*reached)[target] = 1U;
  (*witness_parent)[target] = static_cast<std::uint64_t>(source);
  (*children)[source].push_back(target);
  if ((*children)[source].size() >= kLeafRepairMaxDegree) {
    eligible_sources->erase(source);
  }
  eligible_sources->insert(target);
  std::vector<std::size_t> pending{target};
  for (std::size_t cursor = 0U; cursor < pending.size(); ++cursor) {
    const std::size_t current = pending[cursor];
    for (const std::size_t next : adjacency[current]) {
      ++(*bfs_edge_scans);
      if ((*reached)[next] == 0U) {
        (*reached)[next] = 1U;
        (*witness_parent)[next] = static_cast<std::uint64_t>(current);
        (*children)[current].push_back(next);
        if ((*children)[current].size() >= kLeafRepairMaxDegree) {
          eligible_sources->erase(current);
        }
        eligible_sources->insert(next);
        pending.push_back(next);
      }
    }
  }
}

struct RankedTarget {
  std::size_t row = 0U;
  double squared_distance = 0.0;
};

void add_and_prune_bridge(
    std::vector<Row>* adjacency, std::size_t source, std::size_t target,
    const std::vector<std::int64_t>& members, const CorpusView& corpus,
    double alpha_squared, const std::vector<std::vector<std::size_t>>& children,
    const std::vector<std::vector<std::size_t>>& prior_repair_targets,
    std::uint64_t* alpha_distance_evaluations,
    std::uint64_t* alpha_rejected) {
  Row& row = (*adjacency)[source];
  if (std::find(row.begin(), row.end(), target) != row.end()) {
    fail("leaf repair selected a duplicate bridge edge");
  }
  std::vector<std::size_t> protected_targets;
  protected_targets.reserve(kLeafRepairMaxDegree);
  const auto protect = [&](std::size_t protected_target) {
    if (std::find(protected_targets.begin(), protected_targets.end(),
                  protected_target) == protected_targets.end()) {
      protected_targets.push_back(protected_target);
    }
  };
  for (const std::size_t child : children[source]) protect(child);
  for (const std::size_t prior_target : prior_repair_targets[source]) {
    protect(prior_target);
  }
  protect(target);
  const std::size_t protected_count = protected_targets.size();
  if (protected_count > kLeafRepairMaxDegree) {
    fail("leaf repair protected witness set exceeds degree 24");
  }

  std::vector<std::size_t> candidates = row;
  candidates.push_back(target);
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  if (std::find(candidates.begin(), candidates.end(), source) !=
      candidates.end()) {
    fail("leaf repair robust prune received a self edge");
  }
  for (const std::size_t protected_target : protected_targets) {
    if (std::find(candidates.begin(), candidates.end(), protected_target) ==
        candidates.end()) {
      fail("leaf repair lost a protected edge before robust prune");
    }
  }

  // This is the only dirty-row operation.  Rank by the same sequential
  // squared-L2 seam as leaf construction, then apply inclusive
  // alpha_squared * chosen_to_candidate <= source_to_candidate occlusion.
  // The protected witness set bypasses occlusion but remains in the same
  // deterministic source-distance order in the published row.
  std::vector<RankedTarget> ranked;
  ranked.reserve(candidates.size());
  for (const std::size_t candidate : candidates) {
    const double distance = squared_l2_f32_f64_seq(
        corpus.vectors, corpus.vector_dim, corpus.object_count,
        members[source], members[candidate]);
    ++(*alpha_distance_evaluations);
    ranked.push_back(RankedTarget{candidate, distance});
  }
  std::sort(ranked.begin(), ranked.end(), [&](const RankedTarget& left,
                                              const RankedTarget& right) {
    if (left.squared_distance != right.squared_distance) {
      return left.squared_distance < right.squared_distance;
    }
    return members[left.row] < members[right.row];
  });

  const auto is_protected = [&](std::size_t candidate) {
    return std::find(protected_targets.begin(), protected_targets.end(),
                     candidate) != protected_targets.end();
  };
  std::vector<std::size_t> selected;
  selected.reserve(kLeafRepairMaxDegree);
  std::size_t selected_count = 0U;
  const std::size_t slots =
      static_cast<std::size_t>(kLeafRepairMaxDegree) - protected_count;
  if (slots != 0U) {
    for (const RankedTarget& candidate : ranked) {
      if (is_protected(candidate.row)) continue;
      bool rejected = false;
      for (const RankedTarget& chosen : ranked) {
        if (std::find(selected.begin(), selected.end(), chosen.row) ==
            selected.end()) {
          continue;
        }
        const double chosen_to_candidate = squared_l2_f32_f64_seq(
            corpus.vectors, corpus.vector_dim, corpus.object_count,
            members[chosen.row], members[candidate.row]);
        ++(*alpha_distance_evaluations);
        if (alpha_product(alpha_squared, chosen_to_candidate) <=
            candidate.squared_distance) {
          rejected = true;
          break;
        }
      }
      if (rejected) {
        ++(*alpha_rejected);
        continue;
      }
      selected.push_back(candidate.row);
      ++selected_count;
      if (selected_count == slots) break;
    }
  }

  Row replacement;
  replacement.reserve(protected_count + selected_count);
  for (const RankedTarget& candidate : ranked) {
    if (is_protected(candidate.row) ||
        std::find(selected.begin(), selected.end(), candidate.row) !=
            selected.end()) {
      replacement.push_back(candidate.row);
    }
  }
  if (replacement.size() > kLeafRepairMaxDegree ||
      replacement.size() < protected_count ||
      std::find(replacement.begin(), replacement.end(), target) ==
          replacement.end()) {
    fail("leaf repair robust prune failed to preserve mandatory edges");
  }
  row = std::move(replacement);
}

struct BridgeChoice {
  std::size_t source_row = 0U;
  std::size_t target_row = 0U;
  double squared_distance = 0.0;
};

BridgeChoice choose_bridge(
    const CorpusView& corpus, const std::vector<std::int64_t>& members,
    const std::set<std::size_t>& eligible_sources,
    const std::vector<std::size_t>& target_component,
    std::uint64_t* distance_evaluations) {
  std::vector<std::size_t> source_samples;
  for (const std::size_t row : eligible_sources) {
    source_samples.push_back(row);
    if (source_samples.size() == kLeafRepairSampleBudget) break;
  }
  if (source_samples.empty()) {
    fail("leaf repair has no reachable row with a protected degree slot");
  }
  if (source_samples.size() > kLeafRepairSampleBudget) {
    source_samples.resize(kLeafRepairSampleBudget);
  }

  const std::vector<std::size_t> target_samples =
      deterministic_target_samples(target_component);

  bool have_best = false;
  BridgeChoice best;
  for (const std::size_t source : source_samples) {
    for (const std::size_t target : target_samples) {
      const double distance = squared_l2_f32_f64_seq(
          corpus.vectors, corpus.vector_dim, corpus.object_count,
          members[source], members[target]);
      ++(*distance_evaluations);
      const bool better =
          !have_best || distance < best.squared_distance ||
          (distance == best.squared_distance &&
           (members[source] < members[best.source_row] ||
            (members[source] == members[best.source_row] &&
             members[target] < members[best.target_row])));
      if (better) {
        have_best = true;
        best = BridgeChoice{source, target, distance};
      }
    }
  }
  if (!have_best) fail("leaf repair could not choose a finite bridge");
  return best;
}

void audit_witness_edges(
    const std::vector<Row>& adjacency,
    const std::vector<std::int64_t>& members,
    std::size_t anchor,
    const std::vector<std::uint64_t>& witness_parent) {
  if (witness_parent.size() != members.size() ||
      witness_parent[anchor] != kNoWitnessParent) {
    fail("leaf repair witness anchor entry is invalid");
  }
  for (std::size_t row = 0U; row < members.size(); ++row) {
    if (row == anchor) continue;
    const std::uint64_t parent = witness_parent[row];
    if (parent >= members.size() ||
        std::find(adjacency[static_cast<std::size_t>(parent)].begin(),
                  adjacency[static_cast<std::size_t>(parent)].end(), row) ==
            adjacency[static_cast<std::size_t>(parent)].end()) {
      fail("leaf repair witness edge is not physically present");
    }
  }
}

void audit_witness_edges_flat(
    const std::vector<std::uint64_t>& row_ptr,
    const std::vector<std::size_t>& target_rows, std::size_t anchor,
    const std::vector<std::uint64_t>& witness_parent) {
  const std::size_t row_count = row_ptr.size() - 1U;
  if (witness_parent.size() != row_count ||
      witness_parent[anchor] != kNoWitnessParent) {
    fail("leaf repair witness anchor entry is invalid");
  }
  for (std::size_t row = 0U; row < row_count; ++row) {
    if (row == anchor) continue;
    const std::uint64_t parent = witness_parent[row];
    if (parent >= row_count) {
      fail("leaf repair witness parent is out of bounds");
    }
    const std::size_t begin =
        static_cast<std::size_t>(row_ptr[static_cast<std::size_t>(parent)]);
    const std::size_t end = static_cast<std::size_t>(
        row_ptr[static_cast<std::size_t>(parent) + 1U]);
    if (std::find(target_rows.begin() + static_cast<std::ptrdiff_t>(begin),
                 target_rows.begin() + static_cast<std::ptrdiff_t>(end), row) ==
        target_rows.begin() + static_cast<std::ptrdiff_t>(end)) {
      fail("leaf repair witness edge is not physically present");
    }
  }
}

void serialize_csr(const std::vector<Row>& adjacency,
                   const std::vector<std::int64_t>& members,
                   std::vector<std::uint64_t>* row_ptr,
                   std::vector<std::int64_t>* target_ids) {
  row_ptr->clear();
  target_ids->clear();
  row_ptr->reserve(adjacency.size() + 1U);
  row_ptr->push_back(0U);
  for (const Row& row : adjacency) {
    if (row.size() > kLeafRepairMaxDegree ||
        target_ids->size() > std::numeric_limits<std::uint64_t>::max() -
                                  row.size()) {
      fail("leaf repair output degree or CSR size exceeds its contract");
    }
    for (const std::size_t target : row) target_ids->push_back(members[target]);
    row_ptr->push_back(static_cast<std::uint64_t>(target_ids->size()));
  }
}

std::uint64_t audit_csr(const std::vector<Row>& adjacency,
                        const std::vector<std::int64_t>& members,
                        std::size_t anchor,
                        const std::vector<std::uint64_t>& witness_parent) {
  std::uint64_t edge_scans = 0U;
  for (std::size_t source = 0U; source < adjacency.size(); ++source) {
    if (adjacency[source].size() > kLeafRepairMaxDegree) {
      fail("leaf repair output exceeds degree 24");
    }
    std::vector<std::size_t> seen;
    seen.reserve(adjacency[source].size());
    for (const std::size_t target : adjacency[source]) {
      ++edge_scans;
      if (target >= members.size() || target == source ||
          std::find(seen.begin(), seen.end(), target) != seen.end()) {
        fail("leaf repair output has an invalid, self, or duplicate edge");
      }
      seen.push_back(target);
    }
  }
  audit_witness_edges(adjacency, members, anchor, witness_parent);
  std::uint64_t reach_scans = 0U;
  std::vector<std::uint64_t> audit_parent;
  std::vector<std::uint8_t> reached =
      directed_bfs(adjacency, anchor, &audit_parent, &reach_scans);
  const std::size_t reached_count = static_cast<std::size_t>(std::count(
      reached.begin(), reached.end(), static_cast<std::uint8_t>(1U)));
  if (reached_count != adjacency.size()) {
    fail("leaf repair final directed anchor audit failed");
  }
  return edge_scans + reach_scans;
}

// The initial flat BFS already proves reachability for an untouched leaf.
// The clean path therefore needs only one final edge/witness pass, rather
// than allocating mutable rows and running a second full BFS over the same
// bytes.  Repaired leaves still use audit_csr because their rows were
// intentionally mutated and require an independent final reachability pass.
std::uint64_t audit_flat(const std::vector<std::uint64_t>& row_ptr,
                         const std::vector<std::size_t>& target_rows,
                         std::size_t anchor,
                         const std::vector<std::uint64_t>& witness_parent) {
  const std::size_t row_count = row_ptr.size() - 1U;
  const std::size_t no_row = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> seen_target(row_count, no_row);
  std::uint64_t edge_scans = 0U;
  for (std::size_t source = 0U; source < row_count; ++source) {
    const std::size_t begin = static_cast<std::size_t>(row_ptr[source]);
    const std::size_t end = static_cast<std::size_t>(row_ptr[source + 1U]);
    if (end - begin > kLeafRepairMaxDegree) {
      fail("leaf repair output exceeds degree 24");
    }
    for (std::size_t edge = begin; edge < end; ++edge) {
      ++edge_scans;
      const std::size_t target = target_rows[edge];
      if (target >= row_count || target == source ||
          seen_target[target] == source) {
        fail("leaf repair output has an invalid, self, or duplicate edge");
      }
      seen_target[target] = source;
    }
  }
  audit_witness_edges_flat(row_ptr, target_rows, anchor, witness_parent);
  return edge_scans;
}

}  // namespace

static LeafRepairResult repair_leaf_impl(const CorpusView& corpus,
                                         const LeafBuildResult& leaf,
                                         std::int64_t frozen_anchor,
                                         const LeafRepairOptions& options) {
  validate_options(options);
  validate_corpus(corpus);
  const ValidatedLeaf validated = validate_leaf(corpus, leaf);
  const double alpha_squared_value = alpha_squared(options.alpha);
  LeafRepairIdentityReceipt identity;
  identity.frozen_anchor = frozen_anchor;
  identity.max_degree = options.max_degree;
  identity.sample_budget = options.sample_budget;
  identity.alpha_bits = double_bits(options.alpha);
  identity.alpha_squared_bits = double_bits(alpha_squared_value);
  const std::size_t anchor = row_for_id(leaf.row_object_ids, frozen_anchor);

  std::uint64_t bfs_edge_scans = 0U;
  std::vector<std::uint64_t> witness_parent;
  std::vector<std::uint8_t> reached =
      directed_bfs_flat(leaf.row_ptr, validated.target_rows, anchor,
                        &witness_parent, &bfs_edge_scans);
  const std::uint64_t reached_count = static_cast<std::uint64_t>(std::count(
      reached.begin(), reached.end(), static_cast<std::uint8_t>(1U)));

  LeafRepairEvidence evidence;
  evidence.rows = static_cast<std::uint64_t>(leaf.row_object_ids.size());
  evidence.initial_reachable = reached_count;
  evidence.initial_unreachable = evidence.rows - reached_count;
  evidence.bfs_edge_scans = bfs_edge_scans;

  std::vector<Row> adjacency;
  if (reached_count != evidence.rows) {
    adjacency = materialize_adjacency(leaf.row_ptr, validated);
    const SccResult scc = unreachable_sccs(adjacency, reached);
    const std::vector<std::size_t> sources =
        source_components(adjacency, reached, scc);
    evidence.unreachable_source_scc_count =
        static_cast<std::uint64_t>(sources.size());
    if (sources.empty()) {
      fail("leaf repair found unreachable rows without an unreachable SCC");
    }

    std::vector<std::vector<std::size_t>> children =
        witness_children(witness_parent);
    // This ordered set is intentional: it gives the first eight eligible
    // source IDs independent of worker scheduling.  The module makes no
    // blanket strict-O(V) claim for this ordering layer; the 50K fixture is a
    // measured gate in the public test seam.
    std::set<std::size_t> eligible_sources;
    for (std::size_t row = 0U; row < reached.size(); ++row) {
      if (reached[row] != 0U &&
          children[row].size() < kLeafRepairMaxDegree) {
        eligible_sources.insert(row);
      }
    }
    std::vector<std::vector<std::size_t>> prior_repair_targets(
        adjacency.size());
    std::vector<std::uint8_t> dirty(adjacency.size(), 0U);
    for (const std::size_t component : sources) {
      const std::uint64_t distance_evaluations_before =
          evidence.repair_distance_evaluations;
      const BridgeChoice bridge = choose_bridge(
          corpus, leaf.row_object_ids, eligible_sources, scc.members[component],
          &evidence.repair_distance_evaluations);
      evidence.bridges.push_back(LeafRepairBridgeEvidence{
          leaf.row_object_ids[bridge.source_row],
          leaf.row_object_ids[bridge.target_row], bridge.squared_distance,
          evidence.repair_distance_evaluations -
              distance_evaluations_before});
      if (reached[bridge.target_row] != 0U) {
        fail("leaf repair source SCC became reachable before its bridge");
      }
      add_and_prune_bridge(
          &adjacency, bridge.source_row, bridge.target_row,
          leaf.row_object_ids, corpus, alpha_squared_value, children,
          prior_repair_targets, &evidence.repair_alpha_distance_evaluations,
          &evidence.repair_alpha_rejected);
      prior_repair_targets[bridge.source_row].push_back(bridge.target_row);
      if (dirty[bridge.source_row] == 0U) {
        dirty[bridge.source_row] = 1U;
        ++evidence.dirty_source_rows;
      }
      extend_witness(adjacency, bridge.source_row, bridge.target_row, &reached,
                     &witness_parent, &children, &eligible_sources,
                     &evidence.bfs_edge_scans);
      ++evidence.repair_bridges;
    }

    if (evidence.repair_bridges != evidence.unreachable_source_scc_count) {
      fail("leaf repair bridge count is not the source-SCC minimum");
    }
    if (evidence.repair_bridges >
            std::numeric_limits<std::uint64_t>::max() / 64U ||
        evidence.repair_distance_evaluations >
            evidence.repair_bridges * 64U ||
        evidence.dirty_source_rows > evidence.repair_bridges ||
        evidence.bridges.size() !=
            static_cast<std::size_t>(evidence.repair_bridges)) {
      fail("leaf repair exceeded its deterministic bridge work contract");
    }
  }

  LeafRepairResult result;
  result.row_object_ids = leaf.row_object_ids;
  if (evidence.repair_bridges == 0U) {
    // Preserve the original vectors byte-for-byte on a naturally connected
    // leaf.  This also avoids publishing a different but equivalent CSR.
    result.row_ptr = leaf.row_ptr;
    result.target_ids = leaf.target_ids;
  } else {
    serialize_csr(adjacency, leaf.row_object_ids, &result.row_ptr,
                  &result.target_ids);
  }
  result.witness_parent_row = std::move(witness_parent);
  result.identity = identity;
  evidence.final_audit_edge_scans =
      evidence.repair_bridges == 0U
          ? audit_flat(leaf.row_ptr, validated.target_rows, anchor,
                       result.witness_parent_row)
          : audit_csr(adjacency, leaf.row_object_ids, anchor,
                      result.witness_parent_row);
  result.evidence = evidence;
  return result;
}

LeafRepairResult repair_leaf(const CorpusView& corpus,
                             const LeafBuildResult& leaf,
                             std::int64_t frozen_anchor,
                             const LeafRepairOptions& options) {
  try {
    return repair_leaf_impl(corpus, leaf, frozen_anchor, options);
  } catch (const LeafRepairError&) {
    throw;
  } catch (const NumericError& error) {
    throw LeafRepairError(std::string("leaf repair numeric failure: ") +
                          error.what());
  } catch (const std::exception& error) {
    throw LeafRepairError(std::string("leaf repair failure: ") +
                          error.what());
  }
}

namespace {

struct CanonicalDigestBuilder {
  std::array<std::uint64_t, 4> state{
      UINT64_C(0x243f6a8885a308d3), UINT64_C(0x13198a2e03707344),
      UINT64_C(0xa4093822299f31d0), UINT64_C(0x082efa98ec4e6c89)};

  void byte(const std::uint8_t value) noexcept {
    constexpr std::array<std::uint64_t, 4> multipliers{
        UINT64_C(0x100000001b3), UINT64_C(0x9e3779b185ebca87),
        UINT64_C(0xc2b2ae3d27d4eb4f), UINT64_C(0x165667b19e3779f9)};
    for (std::size_t lane = 0U; lane < state.size(); ++lane) {
      const std::uint64_t mixed =
          static_cast<std::uint64_t>(value) +
          UINT64_C(0x9e3779b97f4a7c15) * static_cast<std::uint64_t>(lane + 1U);
      state[lane] ^= mixed;
      state[lane] *= multipliers[lane];
      state[lane] ^= state[(lane + 1U) % state.size()] >> 29U;
    }
  }

  void u64(const std::uint64_t value) noexcept {
    for (std::size_t shift = 0U; shift < 64U; shift += 8U) {
      byte(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
  }

  void u32(const std::uint32_t value) noexcept { u64(value); }

  void i64(const std::int64_t value) noexcept {
    u64(static_cast<std::uint64_t>(value));
  }

  void boolean(const bool value) noexcept { byte(value ? 1U : 0U); }

  void text(const std::string_view value) noexcept {
    u64(static_cast<std::uint64_t>(value.size()));
    for (const char character : value) {
      byte(static_cast<std::uint8_t>(static_cast<unsigned char>(character)));
    }
  }

  template <typename T>
  void i64_vector(const std::vector<T>& values) noexcept {
    u64(static_cast<std::uint64_t>(values.size()));
    for (const T value : values) i64(static_cast<std::int64_t>(value));
  }

  void u64_vector(const std::vector<std::uint64_t>& values) noexcept {
    u64(static_cast<std::uint64_t>(values.size()));
    for (const std::uint64_t value : values) u64(value);
  }

  LeafRepairDigest finish() noexcept {
    for (std::size_t round = 0U; round < 4U; ++round) {
      for (std::size_t lane = 0U; lane < state.size(); ++lane) {
        std::uint64_t value = state[lane];
        value ^= value >> 33U;
        value *= UINT64_C(0xff51afd7ed558ccd);
        value ^= value >> 33U;
        value *= UINT64_C(0xc4ceb9fe1a85ec53);
        value ^= value >> 33U;
        state[lane] = value + state[(lane + 1U) % state.size()];
      }
    }
    return state;
  }
};

}  // namespace

LeafRepairDigest leaf_repair_canonical_digest(const LeafRepairResult& result) {
  CanonicalDigestBuilder digest;
  digest.text(kLeafRepairCanonicalDigestRevision);
  digest.i64(result.identity.frozen_anchor);
  digest.u32(result.identity.max_degree);
  digest.u32(result.identity.sample_budget);
  digest.u64(result.identity.alpha_bits);
  digest.u64(result.identity.alpha_squared_bits);
  digest.text(result.identity.algorithm_revision);
  digest.text(result.identity.numeric_revision);
  digest.text(result.identity.tie_order);
  digest.boolean(result.identity.thread_independent);

  digest.i64_vector(result.row_object_ids);
  digest.u64_vector(result.row_ptr);
  digest.i64_vector(result.target_ids);
  digest.u64_vector(result.witness_parent_row);

  digest.u64(result.evidence.rows);
  digest.u64(result.evidence.corpus_value_scans);
  digest.u64(result.evidence.initial_reachable);
  digest.u64(result.evidence.initial_unreachable);
  digest.u64(result.evidence.unreachable_source_scc_count);
  digest.u64(result.evidence.repair_bridges);
  digest.u64(result.evidence.repair_distance_evaluations);
  digest.u64(result.evidence.repair_alpha_distance_evaluations);
  digest.u64(result.evidence.repair_alpha_rejected);
  digest.u64(result.evidence.dirty_source_rows);
  digest.u64(result.evidence.bfs_edge_scans);
  digest.u64(result.evidence.final_audit_edge_scans);
  digest.u64(static_cast<std::uint64_t>(result.evidence.bridges.size()));
  for (const LeafRepairBridgeEvidence& bridge : result.evidence.bridges) {
    digest.i64(bridge.source_id);
    digest.i64(bridge.target_id);
    digest.u64(double_bits(bridge.squared_distance));
    digest.u64(bridge.sample_distance_evaluations);
  }
  return digest.finish();
}

}  // namespace bounded_blowup

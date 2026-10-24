// SPDX-License-Identifier: MIT
// Native Logical Artifact V3 ANN query adapter.
#include "bounded_blowup/logical_search.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace bounded_blowup {
namespace {

constexpr std::uint64_t kMissingBlock =
    std::numeric_limits<std::uint64_t>::max();

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::atomic<std::uint64_t> g_member_comparisons{0U};
#endif

std::uint16_t checked_template_id(std::size_t value) {
  if (value > std::numeric_limits<std::uint16_t>::max()) {
    throw CorruptIndexError("template ID exceeds uint16 sentinel");
  }
  return static_cast<std::uint16_t>(value);
}

std::uint32_t member_local_row(ArrayView<std::int64_t> members,
                               std::int64_t object_id) {
  std::size_t lo = 0U;
  std::size_t hi = members.size();
  while (lo < hi) {
    const std::size_t middle = lo + (hi - lo) / 2U;
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
    g_member_comparisons.fetch_add(1U, std::memory_order_relaxed);
#endif
    if (members[middle] < object_id) {
      lo = middle + 1U;
    } else {
      hi = middle;
    }
  }
  if (lo == members.size()) {
    throw CorruptIndexError(
        "logical route entry/anchor has no row in selected view");
  }
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  g_member_comparisons.fetch_add(1U, std::memory_order_relaxed);
#endif
  if (members[lo] != object_id) {
    throw CorruptIndexError(
        "logical route entry/anchor has no row in selected view");
  }
  return static_cast<std::uint32_t>(lo);
}

bool hit_better(const SearchHit& lhs, const SearchHit& rhs) {
  if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
  return lhs.object_id < rhs.object_id;
}

std::int64_t checked_floor_to_i64(double value) {
  const long double wide = static_cast<long double>(value);
  if (!std::isfinite(value) ||
      wide < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
      wide > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("routing coordinate is outside int64 range");
  }
  return static_cast<std::int64_t>(std::floor(value));
}

std::int64_t checked_grid_add(std::int64_t value, int delta) {
  if (delta < 0 && value == std::numeric_limits<std::int64_t>::min()) {
    return value;
  }
  if (delta > 0 && value == std::numeric_limits<std::int64_t>::max()) {
    return value;
  }
  return value + delta;
}

double lattice_unit_for(const IndexView& index, std::uint16_t dimension) {
  return index.lattice_unit.size() == 1U ? index.lattice_unit[0]
                                         : index.lattice_unit[dimension];
}

double coefficient_bound(std::int64_t coefficient, double unit) {
  const long double bound = static_cast<long double>(coefficient) *
                            static_cast<long double>(unit);
  if (!std::isfinite(static_cast<double>(bound))) {
    throw NoContainingBlockError("lattice bound overflows float64");
  }
  return static_cast<double>(bound);
}

double shape_distance(ArrayView<double> left,
                      const std::vector<double>& right) {
  if (left.size() != right.size() || left.empty()) {
    throw CorruptIndexError("template/query shape ranks disagree");
  }
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0U; i < left.size(); ++i) {
    if (!(left[i] > 0.0) || !(right[i] > 0.0)) {
      throw std::invalid_argument(
          "template-aware routing requires positive widths");
    }
    const double value = std::log(right[i] / left[i]);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return maximum - minimum;
}

template <typename T>
ArrayView<T> array_view(const std::vector<T>& values) noexcept {
  return ArrayView<T>(values.data(), values.size());
}

// A small fixed lookahead hides the corpus-vector and visited-epoch loads for
// the next logical neighbors.  This is deliberately only a cache hint: it
// never changes admission, ordering, or the visited-state decision.
constexpr std::size_t kLogicalNeighborPrefetchLookahead = 4U;

inline void prefetch_logical_neighbor(const IndexView& index,
                                      std::int64_t object_id,
                                      std::uint32_t local_row,
                                      const std::vector<std::uint32_t>&
                                          seen_epoch) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  if (object_id < 0 ||
      static_cast<std::uint64_t>(object_id) >= index.object_count ||
      local_row >= seen_epoch.size()) {
    return;
  }
  const std::size_t object = static_cast<std::size_t>(object_id);
  const std::size_t vector_offset =
      object * static_cast<std::size_t>(index.vector_dim);
  __builtin_prefetch(index.vectors.data() + vector_offset, 0, 1);
  __builtin_prefetch(seen_epoch.data() + static_cast<std::size_t>(local_row),
                     0, 1);
#else
  (void)index;
  (void)object_id;
  (void)local_row;
  (void)seen_epoch;
#endif
}

}  // namespace

struct LogicalQueryHandle::State {
  struct BoundRoute {
    const LogicalArtifactRouteBinding* binding = nullptr;
    LogicalViewId view_id = std::numeric_limits<LogicalViewId>::max();
    std::uint16_t level = std::numeric_limits<std::uint16_t>::max();
    std::int64_t anchor = 0;
    std::uint32_t anchor_local_row = 0U;
    std::size_t index_members_offset = 0U;
    std::size_t index_members_size = 0U;
    std::size_t index_entries_offset = 0U;
    std::size_t index_entries_size = 0U;
    std::size_t entry_local_rows_offset = 0U;
  };

  explicit State(LogicalArtifactIndex value) : artifact(std::move(value)) {}

  LogicalArtifactIndex artifact;

  // All route arrays below are copied from artifact.route_bindings.  No
  // caller-owned IndexView storage is retained by the handle.
  std::vector<std::uint64_t> block_member_ptr;
  std::vector<std::int64_t> block_member_ids;
  std::vector<std::uint64_t> block_entry_ptr;
  std::vector<std::int64_t> block_entry_ids;
  std::vector<std::int64_t> block_anchor_id;
  std::vector<std::uint32_t> entry_local_rows;
  std::vector<BoundRoute> routes;
  std::size_t max_rows = 0U;
  std::size_t max_entries = 0U;
};

struct LogicalQueryHandle::BoundIndex {
  IndexView view;
};

namespace {

/* routing-layout validation and canonicalization live in routing_layout.cpp. */
/* The query owner only consumes the already sealed route state. */
/* (The former bind-time validator was intentionally moved here to the
 * routing-layout owner.) */

}  // namespace

LogicalQueryHandle::BoundIndex LogicalQueryHandle::make_bound_index(
    const State* state) {
  // The friend seam is intentionally narrow: the query owner may borrow only
  // immutable verified corpus arrays, never an external IndexView.
  const VerifiedCorpusSnapshot& corpus = *state->artifact.corpus_snapshot_;

  BoundIndex bound;
  IndexView& index = bound.view;
  index = state->artifact.routing_.materialize();
  index.vectors = corpus.vectors();
  index.attrs_raw = corpus.attrs_raw();
  index.block_member_ptr = array_view(state->block_member_ptr);
  index.block_member_ids = array_view(state->block_member_ids);
  index.block_entry_ptr = array_view(state->block_entry_ptr);
  index.block_entry_ids = array_view(state->block_entry_ids);
  index.block_anchor_id = array_view(state->block_anchor_id);
  return bound;
}

LogicalQueryHandle LogicalQueryHandle::bind(LogicalArtifactIndex&& artifact) {
  auto state = std::make_shared<State>(std::move(artifact));
  if (!state->artifact.corpus_snapshot_) {
    throw LogicalArtifactError("logical query artifact has no corpus snapshot");
  }
  const VerifiedCorpusSnapshot& corpus = *state->artifact.corpus_snapshot_;
  if (corpus.object_count() == 0U || corpus.vector_dim() == 0U ||
      corpus.attr_dim() == 0U || corpus.vector_dim() >
          static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
      corpus.attr_dim() >
          static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max())) {
    throw LogicalArtifactError("logical query artifact corpus shape is invalid");
  }
  const auto& metadata = state->artifact.metadata();
  if (metadata.corpus_object_count != corpus.object_count() ||
      metadata.corpus_vector_dim != corpus.vector_dim() ||
      metadata.identity.corpus_vectors_sha256 !=
          corpus.identity().vectors().file_sha256().hex() ||
      metadata.identity.corpus_attrs_sha256 !=
          corpus.identity().attrs_raw().file_sha256().hex()) {
    throw LogicalArtifactError(
        "logical query artifact identity differs from corpus snapshot");
  }
  const std::size_t block_count = state->artifact.routing_.block_count();
  const auto& routes = state->artifact.route_bindings_;
  if (routes.empty()) {
    throw LogicalArtifactError("logical query artifact has no routes");
  }
  std::vector<const LogicalArtifactRouteBinding*> by_block(block_count, nullptr);
  state->routes.assign(block_count, State::BoundRoute{});
  for (const auto& route : routes) {
    if (route.block_id >= block_count || by_block[route.block_id] != nullptr) {
      throw LogicalArtifactError("logical query route map is not one-to-one");
    }
    by_block[route.block_id] = &route;
    if (route.level != state->artifact.routing_.block_level(route.block_id)) {
      throw LogicalArtifactError("logical query route level differs from layout");
    }
    const ArrayView<std::int64_t> logical_members =
        state->artifact.logical_view().members(route.view_id);
    if (logical_members.size() != route.members.size() ||
        !std::equal(route.members.begin(), route.members.end(),
                    logical_members.begin())) {
      throw LogicalArtifactError(
          "logical query route members differ from immutable graph view");
    }
    if (!std::is_sorted(route.members.begin(), route.members.end())) {
      throw LogicalArtifactError("logical query route members are not sorted");
    }
    if (route.members.empty() || route.members.size() >
                                    std::numeric_limits<std::uint32_t>::max() ||
        route.anchor < 0 ||
        static_cast<std::uint64_t>(route.anchor) >= corpus.object_count()) {
      throw LogicalArtifactError("logical query route has invalid members");
    }
    const auto valid_object_id = [&](std::int64_t object_id) {
      return object_id >= 0 &&
             static_cast<std::uint64_t>(object_id) < corpus.object_count();
    };
    if (!std::all_of(route.members.begin(), route.members.end(),
                     valid_object_id) ||
        !std::all_of(route.entries.begin(), route.entries.end(),
                     valid_object_id)) {
      throw LogicalArtifactError("logical query route references invalid object");
    }
    State::BoundRoute& bound = state->routes[route.block_id];
    bound.binding = &route;
    bound.view_id = route.view_id;
    if (route.level > std::numeric_limits<std::uint16_t>::max()) {
      throw LogicalArtifactError("logical query route level is out of range");
    }
    bound.level = static_cast<std::uint16_t>(route.level);
    bound.anchor = route.anchor;
    const auto anchor = std::lower_bound(route.members.begin(),
                                         route.members.end(), route.anchor);
    if (anchor == route.members.end() || *anchor != route.anchor) {
      throw LogicalArtifactError("logical query route anchor is not a member");
    }
    bound.anchor_local_row = static_cast<std::uint32_t>(
        anchor - route.members.begin());
    bound.entry_local_rows_offset = state->entry_local_rows.size();
    const ArrayView<std::int64_t> route_members(route.members.data(),
                                                route.members.size());
    for (const std::int64_t object_id : route.entries) {
      state->entry_local_rows.push_back(
          member_local_row(route_members, object_id));
    }
    state->max_rows = std::max(state->max_rows, route.members.size());
    state->max_entries = std::max(state->max_entries, route.entries.size());
  }
  for (std::size_t block = 0U; block < block_count; ++block) {
    if (by_block[block] == nullptr) {
      throw LogicalArtifactError("logical query route map omits a layout block");
    }
  }
  state->block_member_ptr.assign(block_count + 1U, 0U);
  state->block_entry_ptr.assign(block_count + 1U, 0U);
  state->block_anchor_id.assign(block_count, 0);
  for (std::size_t block = 0U; block < block_count; ++block) {
    const LogicalArtifactRouteBinding* route = by_block[block];
    if (route != nullptr) {
      state->block_anchor_id[block] = route->anchor;
      State::BoundRoute& bound = state->routes[block];
      bound.index_members_offset = state->block_member_ids.size();
      bound.index_members_size = route->members.size();
      state->block_member_ids.insert(state->block_member_ids.end(),
                                     route->members.begin(), route->members.end());
      bound.index_entries_offset = state->block_entry_ids.size();
      bound.index_entries_size = route->entries.size();
      state->block_entry_ids.insert(state->block_entry_ids.end(),
                                    route->entries.begin(), route->entries.end());
    }
    state->block_member_ptr[block + 1U] = state->block_member_ids.size();
    state->block_entry_ptr[block + 1U] = state->block_entry_ids.size();
  }
  return LogicalQueryHandle(std::move(state));
}

void LogicalQueryHandle::prepare(LogicalAnnWorkspace& workspace,
                                 std::uint32_t max_top_k) const {
  if (!state_) {
    throw std::invalid_argument("logical query handle is not bound");
  }
  if (max_top_k == 0U) {
    throw std::invalid_argument("max_top_k must be positive");
  }
  const std::size_t attr_dim = static_cast<std::size_t>(
      state_->artifact.corpus_snapshot_->attr_dim());
  if (state_->max_rows == 0U || attr_dim == 0U ||
      state_->artifact.routing_.entries_per_query() == 0U) {
    throw CorruptIndexError("logical query state has no workspace geometry");
  }

  // Bind has already built one direct block_id -> route record table.  This
  // preparation path only sizes reusable query storage; it does not scan a
  // logical section or route payload.
  workspace.seen_epoch_.assign(state_->max_rows, 0U);
  workspace.candidate_heap_.clear();
  workspace.candidate_heap_.reserve(state_->max_rows);
  workspace.result_heap_.clear();
  workspace.result_heap_.reserve(max_top_k);
  workspace.entry_rank_.clear();
  workspace.entry_rank_.reserve(state_->max_entries);
  workspace.predicate_order_.clear();
  workspace.predicate_order_.reserve(attr_dim);

  workspace.dimension_capacity_ = attr_dim;
  workspace.normalized_lower_.clear();
  workspace.normalized_lower_.reserve(workspace.dimension_capacity_);
  workspace.normalized_upper_.clear();
  workspace.normalized_upper_.reserve(workspace.dimension_capacity_);
  workspace.normalized_block_lower_.clear();
  workspace.normalized_block_lower_.reserve(workspace.dimension_capacity_);
  workspace.normalized_block_upper_.clear();
  workspace.normalized_block_upper_.reserve(workspace.dimension_capacity_);
  workspace.query_widths_.clear();
  workspace.query_widths_.reserve(workspace.dimension_capacity_);
  workspace.template_ratio_.clear();
  workspace.template_ratio_.reserve(workspace.dimension_capacity_);
  workspace.grid_coords_.clear();
  workspace.grid_coords_.reserve(workspace.dimension_capacity_);

  workspace.row_capacity_ = state_->max_rows;
  workspace.entry_capacity_ = state_->max_entries;
  workspace.max_top_k_ = max_top_k;
  workspace.current_epoch_ = 0U;
  workspace.prepared_ = true;
  workspace.prepared_handle_state_ = state_;
}

void LogicalAnnWorkspace::prepare(const LogicalQueryHandle& handle,
                                  std::uint32_t max_top_k) {
  handle.prepare(*this, max_top_k);
}

LogicalAnnSearchResultView LogicalQueryHandle::search_into(
    const float* query_vector, std::size_t query_dim,
    const QueryWindow& window, std::uint32_t top_k, std::uint64_t budget,
    LogicalAnnWorkspace& workspace) const {
  if (!state_) {
    throw std::invalid_argument("logical query handle is not bound");
  }
  if (workspace.prepared_handle_state_.get() != state_.get()) {
    throw std::invalid_argument(
        "logical query workspace was prepared for a different handle state");
  }
  return workspace.search_bound(*this, query_vector, query_dim, window, top_k,
                                budget, false, 0U, 0U, false);
}

LogicalAnnSearchResultView LogicalQueryHandle::search_into_adaptive(
    const float* query_vector, std::size_t query_dim,
    const QueryWindow& window, std::uint32_t top_k, std::uint64_t budget,
    LogicalAnnWorkspace& workspace) const {
  return search_into_adaptive(query_vector, query_dim, window, top_k, budget,
                              200U, 64U, true, workspace);
}

LogicalAnnSearchResultView LogicalQueryHandle::search_into_adaptive(
    const float* query_vector, std::size_t query_dim,
    const QueryWindow& window, std::uint32_t top_k, std::uint64_t budget,
    std::uint64_t minimum_expansions,
    std::uint64_t no_improvement_patience,
    bool require_frontier_bound,
    LogicalAnnWorkspace& workspace) const {
  if (!state_) {
    throw std::invalid_argument("logical query handle is not bound");
  }
  if (workspace.prepared_handle_state_.get() != state_.get()) {
    throw std::invalid_argument(
        "logical query workspace was prepared for a different handle state");
  }
  if (minimum_expansions == 0U || no_improvement_patience == 0U) {
    throw std::invalid_argument(
        "adaptive minimum expansions and patience must be positive");
  }
  return workspace.search_bound(*this, query_vector, query_dim, window, top_k,
                                budget, true, minimum_expansions,
                                no_improvement_patience,
                                require_frontier_bound);
}

LogicalAnnSearchResultView LogicalQueryHandle::search_into(
    const float* query_vector, const QueryWindow& window,
    std::uint32_t top_k, std::uint64_t budget,
    LogicalAnnWorkspace& workspace) const {
  if (!state_) {
    throw std::invalid_argument("logical query handle is not bound");
  }
  return search_into(query_vector,
                     static_cast<std::size_t>(
                         state_->artifact.corpus_snapshot_->vector_dim()),
                     window, top_k, budget, workspace);
}

// TODO(P0): replace this copied scalar/vector router with one shared
// allocation-free seam. The permanent 20,005-case differential test pins its
// behavior until that seam is available; this LOCAL_FIX intentionally does
// not expand the router module.
// This is the frozen router's scalar/vector logic expressed over reusable
// workspace vectors. It deliberately does not call route_first_actual...,
// whose value-returning vectors would allocate on every query.
LogicalAnnWorkspace::RouteState LogicalAnnWorkspace::route_into(
    const LogicalQueryHandle& handle, const QueryWindow& window,
    LogicalAnnWorkspace* workspace) {
  const auto bound_index =
      LogicalQueryHandle::make_bound_index(handle.state_.get());
  const IndexView& index = bound_index.view;
  if (window.dimensions.empty()) {
    throw std::invalid_argument("query dimension set cannot be empty");
  }
  if (window.dimensions.size() != window.lower_closed.size() ||
      window.dimensions.size() != window.upper_closed.size()) {
    throw std::invalid_argument("window dimension/lower/upper lengths differ");
  }
  if (window.dimensions.size() > workspace->dimension_capacity_) {
    throw std::invalid_argument(
        "query dimensions exceed prepared logical workspace capacity");
  }

  for (std::size_t i = 0U; i < window.dimensions.size(); ++i) {
    if (window.dimensions[i] >= index.attr_dim) {
      throw std::invalid_argument("query dimension is out of range");
    }
    if (i > 0U && window.dimensions[i - 1U] >= window.dimensions[i]) {
      throw std::invalid_argument(
          "query dimensions must be strictly increasing and unique");
    }
    if (!std::isfinite(window.lower_closed[i]) ||
        !std::isfinite(window.upper_closed[i])) {
      throw std::invalid_argument("query bounds must be finite");
    }
    if (window.lower_closed[i] > window.upper_closed[i]) {
      throw std::invalid_argument("query lower bound exceeds upper bound");
    }
  }

  LogicalAnnWorkspace::RouteState result;
  const ArrayView<std::uint16_t> query_dims(window.dimensions.data(),
                                            window.dimensions.size());
  result.family_id = lookup_exact_dimension_family(index, query_dims);
  workspace->normalized_lower_.resize(window.dimensions.size());
  workspace->normalized_upper_.resize(window.dimensions.size());
  workspace->query_widths_.resize(window.dimensions.size());
  workspace->grid_coords_.resize(window.dimensions.size());
  workspace->normalized_block_lower_.resize(window.dimensions.size());
  workspace->normalized_block_upper_.resize(window.dimensions.size());

  for (std::size_t i = 0U; i < window.dimensions.size(); ++i) {
    const std::uint16_t dimension = window.dimensions[i];
    const double domain_lo = index.domain_min[dimension];
    const double domain_hi = index.domain_max[dimension];
    const double query_lo = window.lower_closed[i];
    const double query_hi = window.upper_closed[i];
    if (query_hi < domain_lo || query_lo > domain_hi) {
      result.outcome = RouteOutcome::kOutOfDomain;
      return result;
    }
    const double clipped_lo = std::max(query_lo, domain_lo);
    const double clipped_hi = std::min(query_hi, domain_hi);
    result.routing_clipped = result.routing_clipped ||
                             clipped_lo != query_lo || clipped_hi != query_hi;
    if (domain_lo == domain_hi) {
      workspace->normalized_lower_[i] = 0.0;
      workspace->normalized_upper_[i] = 0.0;
    } else {
      const double scale = domain_hi - domain_lo;
      workspace->normalized_lower_[i] = (clipped_lo - domain_lo) / scale;
      workspace->normalized_upper_[i] = (clipped_hi - domain_lo) / scale;
    }
    workspace->query_widths_[i] = workspace->normalized_upper_[i] -
                                   workspace->normalized_lower_[i];
  }

  const std::uint16_t level_count = index.family_level_count[result.family_id];
  const bool template_aware = index.layout_template_aware[0] != 0U;
  std::uint16_t selected_template = 0U;
  double selected_tau = 0.0;
  double maximum_width = 0.0;
  for (const double width : workspace->query_widths_) {
    maximum_width = std::max(maximum_width, width);
  }
  if (template_aware) {
    if (maximum_width > 0.0) {
      constexpr double kZeroWidthShapeFloor = 0x1p-24;
      for (double& width : workspace->query_widths_) {
        width = width > 0.0 ? width / maximum_width : kZeroWidthShapeFloor;
      }
    } else {
      std::fill(workspace->query_widths_.begin(), workspace->query_widths_.end(),
                1.0);
    }
    double best = std::numeric_limits<double>::infinity();
    const std::size_t template_count = index.template_count(result.family_id);
    if (template_count == 0U ||
        template_count > std::numeric_limits<std::uint16_t>::max()) {
      throw CorruptIndexError(
          "template count exceeds uint16 sentinel in logical routing");
    }
    for (std::size_t candidate = 0U; candidate < template_count; ++candidate) {
      const std::uint16_t candidate_id = checked_template_id(candidate);
      const ArrayView<double> ratio =
          index.template_ratio(result.family_id, candidate_id);
      workspace->template_ratio_.assign(ratio.begin(), ratio.end());
      const double distance =
          shape_distance(ratio, workspace->query_widths_);
      if (distance < best) {
        best = distance;
        selected_template = candidate_id;
      }
    }
    selected_tau = best;
  }
  result.template_id = selected_template;
  result.shape_tau = selected_tau;
  const ArrayView<double> selected_ratio =
      index.template_ratio(result.family_id, selected_template);
  workspace->template_ratio_.assign(selected_ratio.begin(),
                                     selected_ratio.end());
  if (workspace->template_ratio_.size() != window.dimensions.size()) {
    throw CorruptIndexError("template/query shape ranks disagree");
  }

  double required_scale = 0.0;
  for (std::size_t i = 0U; i < window.dimensions.size(); ++i) {
    const double width = workspace->normalized_upper_[i] -
                         workspace->normalized_lower_[i];
    required_scale = std::max(required_scale,
                              width / workspace->template_ratio_[i]);
  }
  std::uint16_t first_level = 0U;
  if (template_aware) {
    const double base_unit = index.lattice_unit[0];
    while (first_level < level_count &&
           static_cast<double>(index.level_service_width_coeff[first_level]) *
                   base_unit < required_scale) {
      ++first_level;
    }
    if (first_level == level_count) {
      throw NoContainingBlockError(
          "template scale coverage invariant was not materialized");
    }
  }

  for (std::uint16_t level = first_level; level < level_count; ++level) {
    const std::int64_t stride_coeff = index.level_stride_coeff[level];
    const std::int64_t side_coeff = index.level_side_length_coeff[level];
    bool contains_all_dimensions = true;
    for (std::size_t i = 0U; i < window.dimensions.size(); ++i) {
      const std::uint16_t dimension = window.dimensions[i];
      const double unit = lattice_unit_for(index, dimension) *
                          workspace->template_ratio_[i];
      const double stride = static_cast<double>(stride_coeff) * unit;
      const std::int64_t base_grid = checked_floor_to_i64(
          workspace->normalized_lower_[i] / stride);
      const std::array<std::int64_t, 3U> candidates = {
          checked_grid_add(base_grid, -1), base_grid,
          checked_grid_add(base_grid, 1)};
      bool found_start = false;
      std::int64_t rightmost_grid = 0;
      double rightmost_start = -std::numeric_limits<double>::infinity();
      double rightmost_end = -std::numeric_limits<double>::infinity();
      for (const std::int64_t grid : candidates) {
        const __int128 lo_coeff = static_cast<__int128>(grid) * stride_coeff;
        const __int128 hi_coeff = lo_coeff + side_coeff;
        if (lo_coeff < std::numeric_limits<std::int64_t>::min() ||
            lo_coeff > std::numeric_limits<std::int64_t>::max() ||
            hi_coeff < std::numeric_limits<std::int64_t>::min() ||
            hi_coeff > std::numeric_limits<std::int64_t>::max()) {
          continue;
        }
        const double start = coefficient_bound(
            static_cast<std::int64_t>(lo_coeff), unit);
        const double end = coefficient_bound(
            static_cast<std::int64_t>(hi_coeff), unit);
        if (start <= workspace->normalized_lower_[i] &&
            (!found_start || start > rightmost_start ||
             (start == rightmost_start && grid > rightmost_grid))) {
          found_start = true;
          rightmost_grid = grid;
          rightmost_start = start;
          rightmost_end = end;
        }
      }
      if (!found_start || rightmost_end < workspace->normalized_upper_[i]) {
        contains_all_dimensions = false;
        break;
      }
      workspace->grid_coords_[i] = rightmost_grid;
      workspace->normalized_block_lower_[i] = rightmost_start;
      workspace->normalized_block_upper_[i] = rightmost_end;
    }
    if (!contains_all_dimensions) continue;

    result.level = level;
    result.selected_scale =
        static_cast<double>(index.level_service_width_coeff[level]) *
        index.lattice_unit[0];
    const ArrayView<std::int64_t> grid(workspace->grid_coords_.data(),
                                       window.dimensions.size());
    result.block_id = lookup_materialized_block(
        index, result.family_id, selected_template, level, grid);
    result.outcome = result.block_id == kMissingBlock
                         ? RouteOutcome::kEmptyTheoreticalBlock
                         : RouteOutcome::kMaterializedBlock;
    return result;
  }

  throw NoContainingBlockError(
      "validated artifact has no level that actually contains the query window");
}

void LogicalAnnWorkspace::validate_storage_invariants() const {
  const auto require_storage = [](const char* name, std::size_t size,
                                  std::size_t capacity,
                                  std::size_t required_capacity) {
    if (size > required_capacity || capacity < required_capacity) {
      throw CorruptIndexError(std::string("logical workspace ") + name +
                              " size/capacity invariant failed");
    }
  };
  if (seen_epoch_.size() != row_capacity_ ||
      seen_epoch_.capacity() < row_capacity_) {
    throw CorruptIndexError(
        "logical workspace seen_epoch size/capacity invariant failed");
  }
  if (!prepared_handle_state_) {
    throw CorruptIndexError(
        "logical workspace has no prepared handle state");
  }
  require_storage("candidate_heap", candidate_heap_.size(),
                  candidate_heap_.capacity(), row_capacity_);
  require_storage("result_heap", result_heap_.size(), result_heap_.capacity(),
                  max_top_k_);
  require_storage("entry_rank", entry_rank_.size(), entry_rank_.capacity(),
                  entry_capacity_);
  require_storage("predicate_order", predicate_order_.size(),
                  predicate_order_.capacity(), dimension_capacity_);
  require_storage("normalized_lower", normalized_lower_.size(),
                  normalized_lower_.capacity(), dimension_capacity_);
  require_storage("normalized_upper", normalized_upper_.size(),
                  normalized_upper_.capacity(), dimension_capacity_);
  require_storage("normalized_block_lower", normalized_block_lower_.size(),
                  normalized_block_lower_.capacity(), dimension_capacity_);
  require_storage("normalized_block_upper", normalized_block_upper_.size(),
                  normalized_block_upper_.capacity(), dimension_capacity_);
  require_storage("query_widths", query_widths_.size(),
                  query_widths_.capacity(), dimension_capacity_);
  require_storage("template_ratio", template_ratio_.size(),
                  template_ratio_.capacity(), dimension_capacity_);
  require_storage("grid_coords", grid_coords_.size(), grid_coords_.capacity(),
                  dimension_capacity_);
}

LogicalAnnSearchResultView LogicalAnnWorkspace::search_bound(
    const LogicalQueryHandle& handle, const float* query_vector,
    std::size_t query_dim, const QueryWindow& window, std::uint32_t top_k,
    std::uint64_t budget, bool adaptive_early_stop,
    std::uint64_t adaptive_minimum_expansions,
    std::uint64_t adaptive_no_improvement_patience,
    bool adaptive_require_frontier_bound) {
  if (!prepared_) {
    throw std::invalid_argument("logical search workspace is not prepared");
  }
  if (prepared_handle_state_.get() != handle.state_.get()) {
    throw std::invalid_argument(
        "logical search workspace was prepared for a different handle state");
  }
  validate_storage_invariants();
  const auto bound_index =
      LogicalQueryHandle::make_bound_index(handle.state_.get());
  const IndexView& index = bound_index.view;
  const LogicalArtifactIndex& logical = handle.state_->artifact;
  if (query_vector == nullptr) {
    throw std::invalid_argument("query_vector cannot be null");
  }
  if (query_dim != index.vector_dim) {
    throw std::invalid_argument("query vector dimension does not match index");
  }
  if (top_k == 0U || top_k > this->max_top_k_) {
    throw std::invalid_argument("top_k exceeds prepared logical workspace");
  }
  if (budget == 0U) {
    throw std::invalid_argument("B_total must be positive");
  }
  for (std::size_t dimension = 0U; dimension < query_dim; ++dimension) {
    if (!std::isfinite(query_vector[dimension])) {
      throw std::invalid_argument("query vector contains NaN or Inf");
    }
  }

  LogicalAnnSearchStats stats;
  stats.requested_budget = budget;
  const LogicalAnnWorkspace::RouteState route =
      LogicalAnnWorkspace::route_into(handle, window, this);
  stats.family_id = route.family_id;
  stats.template_id = route.template_id;
  stats.level = route.level;
  stats.block_id = route.block_id;
  stats.routing_clipped = route.routing_clipped;
  stats.shape_tau = route.shape_tau;
  stats.selected_scale = route.selected_scale;

  this->result_heap_.clear();
  if (route.outcome == RouteOutcome::kOutOfDomain) {
    stats.termination_reason = TerminationReason::kOutOfDomain;
    return LogicalAnnSearchResultView{
        ArrayView<SearchHit>(this->result_heap_.data(), 0U), stats};
  }
  if (route.outcome == RouteOutcome::kEmptyTheoreticalBlock) {
    stats.termination_reason = TerminationReason::kEmptyBlock;
    return LogicalAnnSearchResultView{
        ArrayView<SearchHit>(this->result_heap_.data(), 0U), stats};
  }
  if (route.block_id >= index.block_count()) {
    throw CorruptIndexError("routed materialized block is outside index");
  }

  // Bind builds one direct block_id -> route record table.  The lookup below
  // is the only route-table access in a query and never scans route members.
  if (route.block_id >= handle.state_->routes.size()) {
    throw CorruptIndexError("logical selected route is outside bound routes");
  }
  const LogicalQueryHandle::State::BoundRoute& bound =
      handle.state_->routes[route.block_id];
  if (bound.binding == nullptr || bound.level != route.level ||
      bound.index_members_size == 0U ||
      bound.anchor_local_row >= bound.index_members_size ||
      handle.state_->block_member_ids[
          bound.index_members_offset + bound.anchor_local_row] != bound.anchor ||
      index.block_anchor_id[route.block_id] != bound.anchor) {
    throw CorruptIndexError("logical selected route direct record changed");
  }
  const LogicalViewId view_id = bound.view_id;
  const ArrayView<std::int64_t> members(
      handle.state_->block_member_ids.data() + bound.index_members_offset,
      bound.index_members_size);
  stats.route_lookup_checks = 1U;
  stats.view_id = view_id;
  stats.route_validation_element_checks = 0U;
  if (members.empty() || members.size() > row_capacity_ ||
      members.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw LogicalArtifactError("routed logical view exceeds prepared workspace");
  }
  stats.block_object_count = members.size();
  stats.num_graphs_searched = 1U;

  ++this->current_epoch_;
  if (this->current_epoch_ == 0U) {
    std::fill(this->seen_epoch_.begin(), this->seen_epoch_.end(), 0U);
    this->current_epoch_ = 1U;
  }
  this->candidate_heap_.clear();
  this->entry_rank_.clear();
  this->predicate_order_.clear();

  auto candidate_less =
      [](const LogicalAnnWorkspace::CandidateRecord& lhs,
         const LogicalAnnWorkspace::CandidateRecord& rhs) {
        if (lhs.distance != rhs.distance) return lhs.distance > rhs.distance;
        return lhs.local_row > rhs.local_row;
      };
  auto entry_better =
      [](const LogicalAnnWorkspace::CandidateRecord& lhs,
         const LogicalAnnWorkspace::CandidateRecord& rhs) {
        if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
        return lhs.local_row < rhs.local_row;
      };
  auto result_heap_compare = [](const SearchHit& lhs, const SearchHit& rhs) {
    return hit_better(lhs, rhs);
  };

  auto distance_for = [&](std::int64_t object_id, bool planning) -> float {
    if (object_id < 0 || static_cast<std::uint64_t>(object_id) >=
                             index.object_count) {
      throw CorruptIndexError("logical query referenced an invalid object ID");
    }
    const std::size_t object = static_cast<std::size_t>(object_id);
    const std::size_t vector_offset = object * index.vector_dim;

    // The adaptive experiment is allowed to use a deliberately faster,
    // float32 squared-L2 ordering.  Keep this branch isolated from the
    // ordinary search path above/below: the latter retains its frozen
    // float32-to-float64 accumulation and per-candidate sqrt semantics.
    // Squared distances are monotone for ordering, and are converted back to
    // public L2 distances only for the final retained hits below.
    if (adaptive_early_stop) {
      float squared = 0.0F;
      std::size_t dimension = 0U;
#if defined(__AVX2__)
      __m256 squared_lanes = _mm256_setzero_ps();
      for (; dimension + 8U <= index.vector_dim; dimension += 8U) {
        const __m256 query_f32 =
            _mm256_loadu_ps(query_vector + dimension);
        const __m256 object_f32 = _mm256_loadu_ps(
            index.vectors.data() + vector_offset + dimension);
        const __m256 delta = _mm256_sub_ps(query_f32, object_f32);
        squared_lanes = _mm256_add_ps(
            squared_lanes, _mm256_mul_ps(delta, delta));
      }
      alignas(32) std::array<float, 8U> lane_values{};
      _mm256_storeu_ps(lane_values.data(), squared_lanes);
      for (const float value : lane_values) squared += value;
#endif
      for (; dimension < index.vector_dim; ++dimension) {
        const float delta = query_vector[dimension] -
                            index.vectors[vector_offset + dimension];
        squared += delta * delta;
      }
      if (!std::isfinite(squared)) {
        throw CorruptIndexError("squared L2 distance is non-finite");
      }
      if (planning) {
        ++stats.planning_distance_evaluations;
      } else {
        ++stats.traversal_distance_evaluations;
      }
      return squared;
    }

    double squared = 0.0;
    std::size_t dimension = 0U;
#if defined(__AVX2__)
    alignas(32) std::array<double, 4U> squared_terms{};
    for (; dimension + 4U <= index.vector_dim; dimension += 4U) {
      const __m128 query_f32 = _mm_loadu_ps(query_vector + dimension);
      const __m128 object_f32 =
          _mm_loadu_ps(index.vectors.data() + vector_offset + dimension);
      const __m256d delta = _mm256_sub_pd(_mm256_cvtps_pd(query_f32),
                                          _mm256_cvtps_pd(object_f32));
      const __m256d terms = _mm256_mul_pd(delta, delta);
      _mm256_storeu_pd(squared_terms.data(), terms);
      squared += squared_terms[0U];
      squared += squared_terms[1U];
      squared += squared_terms[2U];
      squared += squared_terms[3U];
    }
#endif
    for (; dimension < index.vector_dim; ++dimension) {
      const double delta = static_cast<double>(query_vector[dimension]) -
                           static_cast<double>(index.vectors[vector_offset +
                                                               dimension]);
      squared += delta * delta;
    }
    const float distance = static_cast<float>(std::sqrt(squared));
    if (!std::isfinite(distance)) {
      throw CorruptIndexError("L2 distance is non-finite");
    }
    if (planning) {
      ++stats.planning_distance_evaluations;
    } else {
      ++stats.traversal_distance_evaluations;
    }
    return distance;
  };

  auto admit_with_distance =
      [&](std::uint32_t local_row, float distance) -> bool {
    const std::size_t local = static_cast<std::size_t>(local_row);
    if (local >= members.size()) {
      throw CorruptIndexError("logical neighbor row is outside selected view");
    }
    if (this->seen_epoch_[local] == this->current_epoch_) return false;
    this->seen_epoch_[local] = this->current_epoch_;
    this->candidate_heap_.push_back({distance, local_row});
    std::push_heap(this->candidate_heap_.begin(),
                   this->candidate_heap_.end(), candidate_less);
    ++stats.candidate_pushes;
    return true;
  };

  const std::int64_t anchor_id = index.block_anchor_id[route.block_id];
  const std::uint32_t anchor_local_row = bound.anchor_local_row;
  (void)admit_with_distance(anchor_local_row,
                            distance_for(anchor_id, true));
  stats.entries_selected = 1U;

  const ArrayView<std::int64_t> extras = index.extra_entries(route.block_id);
  if (extras.size() != bound.index_entries_size ||
      bound.entry_local_rows_offset > handle.state_->entry_local_rows.size() ||
      bound.index_entries_size >
          handle.state_->entry_local_rows.size() -
              bound.entry_local_rows_offset) {
    throw CorruptIndexError("logical route entry row cache is invalid");
  }
  const ArrayView<std::uint32_t> entry_local_rows(
      handle.state_->entry_local_rows.data() + bound.entry_local_rows_offset,
      bound.index_entries_size);
  stats.entry_pool_considered = extras.size();
  if (extras.size() > this->entry_capacity_) {
    throw CorruptIndexError("logical route entry pool exceeds prepared capacity");
  }
  for (std::size_t entry_index = 0U; entry_index < extras.size();
       ++entry_index) {
    const std::int64_t object_id = extras[entry_index];
    if (object_id == anchor_id) continue;
    const std::uint32_t local_row = entry_local_rows[entry_index];
    if (local_row >= members.size()) {
      throw CorruptIndexError("logical route entry row cache is stale");
    }
    this->entry_rank_.push_back(
        {distance_for(object_id, true), local_row});
  }
  const std::size_t additional_entry_count = std::min<std::size_t>(
      this->entry_rank_.size(),
      static_cast<std::size_t>(index.entries_per_query - 1U));
  const auto selected_end =
      this->entry_rank_.begin() +
      static_cast<std::vector<LogicalAnnWorkspace::CandidateRecord>::
                     difference_type>(additional_entry_count);
  std::partial_sort(this->entry_rank_.begin(), selected_end,
                    this->entry_rank_.end(), entry_better);
  for (auto entry = this->entry_rank_.begin(); entry != selected_end;
       ++entry) {
    if (admit_with_distance(entry->local_row, entry->distance)) {
      ++stats.entries_selected;
    }
  }
  // Keep every planned entry distance available when traversal reaches an
  // entry that was not selected.  This is the same planned-distance reuse as
  // the materialized adapter, without adding a second epoch/cache domain.
  std::sort(this->entry_rank_.begin(), this->entry_rank_.end(),
            [](const LogicalAnnWorkspace::CandidateRecord& lhs,
               const LogicalAnnWorkspace::CandidateRecord& rhs) {
              return lhs.local_row < rhs.local_row;
            });

  this->predicate_order_.resize(window.dimensions.size());
  for (std::size_t dimension = 0U; dimension < window.dimensions.size();
       ++dimension) {
    this->predicate_order_[dimension] = dimension;
  }
  std::sort(this->predicate_order_.begin(),
            this->predicate_order_.end(), [&](std::size_t lhs,
                                                  std::size_t rhs) {
              const double left = this->normalized_upper_[lhs] -
                                  this->normalized_lower_[lhs];
              const double right = this->normalized_upper_[rhs] -
                                   this->normalized_lower_[rhs];
              if (left != right) return left < right;
              return lhs < rhs;
            });

  auto satisfies_original_window = [&](std::int64_t object_id) {
    ++stats.post_filter_checks;
    const std::size_t attr_offset =
        static_cast<std::size_t>(object_id) * index.attr_dim;
    for (const std::size_t dimension_index : this->predicate_order_) {
      ++stats.post_filter_dimension_checks;
      const std::size_t dimension = window.dimensions[dimension_index];
      const double value = index.attrs_raw[attr_offset + dimension];
      if (value < window.lower_closed[dimension_index] ||
          value > window.upper_closed[dimension_index]) {
        return false;
      }
    }
    return true;
  };

  auto retain_hit = [&](std::int64_t object_id, float distance) {
    const SearchHit hit{object_id, distance};
    if (this->result_heap_.size() < top_k) {
      this->result_heap_.push_back(hit);
      std::push_heap(this->result_heap_.begin(),
                     this->result_heap_.end(), result_heap_compare);
      ++stats.result_updates;
      return;
    }
    if (!hit_better(hit, this->result_heap_.front())) return;
    std::pop_heap(this->result_heap_.begin(),
                  this->result_heap_.end(), result_heap_compare);
    this->result_heap_.back() = hit;
    std::push_heap(this->result_heap_.begin(),
                   this->result_heap_.end(), result_heap_compare);
    ++stats.result_updates;
  };

  std::uint64_t expansions_without_result_improvement = 0U;
  bool stopped_adaptively = false;
  while (!this->candidate_heap_.empty() && stats.actual_budget < budget) {
    std::pop_heap(this->candidate_heap_.begin(),
                  this->candidate_heap_.end(), candidate_less);
    const LogicalAnnWorkspace::CandidateRecord candidate =
        this->candidate_heap_.back();
    this->candidate_heap_.pop_back();
    ++stats.candidate_pops;
    ++stats.actual_budget;
    const std::int64_t object_id = members[candidate.local_row];
    const std::uint64_t result_updates_before =
        adaptive_early_stop ? stats.result_updates : 0U;
    if (satisfies_original_window(object_id)) {
      ++stats.qualifying_expansions;
      retain_hit(object_id, candidate.distance);
    }
    if (adaptive_early_stop) {
      if (this->result_heap_.size() == top_k) {
        if (stats.result_updates != result_updates_before) {
          expansions_without_result_improvement = 0U;
        } else if (expansions_without_result_improvement <
                   std::numeric_limits<std::uint64_t>::max()) {
          ++expansions_without_result_improvement;
        }
      } else {
        expansions_without_result_improvement = 0U;
      }
    }

    const bool adaptive_stability_reached =
        adaptive_early_stop &&
        stats.actual_budget >= adaptive_minimum_expansions &&
        this->result_heap_.size() == top_k &&
        expansions_without_result_improvement >=
            adaptive_no_improvement_patience;
    // Stability-only mode does not inspect the next frontier. Once its stop
    // rule fires, resolving and scoring the current row's outgoing neighbors
    // cannot change the returned hits because none of them will be popped.
    if (adaptive_stability_reached && !adaptive_require_frontier_bound) {
      stopped_adaptively = true;
      break;
    }

    // Every logical neighbor is resolved through the concrete frozen view.
    const LogicalNeighbors neighbors = logical.logical_view().neighbors(
        view_id, candidate.local_row, this->logical_scratch_);
    // Prime the first few vectors before entering the distance loop.  Later
    // iterations keep the same fixed lookahead window, which is useful for
    // both flat leaf adjacency and prepared parent effective CSR.
    const std::size_t neighbor_count = neighbors.local_rows.size();
    const std::size_t initial_prefetch_count = std::min<std::size_t>(
        neighbor_count, kLogicalNeighborPrefetchLookahead);
    for (std::size_t prefetch_index = 0U;
         prefetch_index < initial_prefetch_count; ++prefetch_index) {
      const std::uint32_t target = neighbors.local_rows[prefetch_index];
      if (target >= members.size()) {
        throw CorruptIndexError("logical resolver returned an outside row");
      }
      prefetch_logical_neighbor(index, members[target], target,
                                this->seen_epoch_);
    }
    for (std::size_t neighbor_index = 0U; neighbor_index < neighbor_count;
         ++neighbor_index) {
      const std::size_t lookahead =
          neighbor_index + kLogicalNeighborPrefetchLookahead;
      if (lookahead < neighbor_count) {
        const std::uint32_t lookahead_target =
            neighbors.local_rows[lookahead];
        if (lookahead_target >= members.size()) {
          throw CorruptIndexError("logical resolver returned an outside row");
        }
        prefetch_logical_neighbor(index, members[lookahead_target],
                                  lookahead_target, this->seen_epoch_);
      }
      const std::uint32_t target = neighbors.local_rows[neighbor_index];
      if (target >= members.size()) {
        throw CorruptIndexError("logical resolver returned an outside row");
      }
      if (this->seen_epoch_[target] == this->current_epoch_) continue;
      const auto planned = std::lower_bound(
          this->entry_rank_.begin(), this->entry_rank_.end(), target,
          [](const LogicalAnnWorkspace::CandidateRecord& record,
             std::uint32_t local_row) { return record.local_row < local_row; });
      const float distance =
          planned != this->entry_rank_.end() &&
                  planned->local_row == target
              ? planned->distance
              : distance_for(members[target], false);
      (void)admit_with_distance(target, distance);
    }

    if (adaptive_stability_reached && adaptive_require_frontier_bound &&
        !this->candidate_heap_.empty() &&
        this->candidate_heap_.front().distance >=
            this->result_heap_.front().distance) {
      stopped_adaptively = true;
      break;
    }
  }

  stats.distance_evaluations = stats.planning_distance_evaluations +
                               stats.traversal_distance_evaluations;
  if (adaptive_early_stop) {
    for (SearchHit& hit : this->result_heap_) {
      if (hit.distance < 0.0F || !std::isfinite(hit.distance)) {
        throw CorruptIndexError("adaptive squared L2 result is non-finite");
      }
      hit.distance = static_cast<float>(std::sqrt(
          static_cast<double>(hit.distance)));
    }
  }
  stats.termination_reason =
      (stopped_adaptively ||
       (stats.actual_budget == budget && !this->candidate_heap_.empty()))
          ? TerminationReason::kBudgetExhausted
          : TerminationReason::kFrontierExhausted;
  std::sort(this->result_heap_.begin(), this->result_heap_.end(),
            hit_better);
  return LogicalAnnSearchResultView{
      ArrayView<SearchHit>(this->result_heap_.data(),
                           this->result_heap_.size()),
      stats};
}

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::uint64_t logical_search_member_comparisons_for_testing() noexcept {
  return g_member_comparisons.load(std::memory_order_relaxed);
}

void reset_logical_search_member_comparisons_for_testing() noexcept {
  g_member_comparisons.store(0U, std::memory_order_relaxed);
}
#endif

}  // namespace bounded_blowup

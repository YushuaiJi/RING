// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#include "bounded_blowup/leaf_build.hpp"

#include "bounded_blowup/numeric.hpp"

#include <algorithm>
#if defined(RFBB_LEAF_BUILD_TESTING)
#include <atomic>
#endif
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bounded_blowup {
namespace {

constexpr std::uint32_t kMaxDegree = 24U;
constexpr const char* kSamplerRevision = "affine_modular_per_source_v1";
constexpr std::size_t kProjectionCount = 8U;
constexpr std::size_t kProjectionTermLimit = 16U;

#if defined(RFBB_LEAF_BUILD_TESTING)
std::atomic<std::uint64_t> g_leaf_build_thread_spawn_count{0U};
#endif

struct Sha256 {
  static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
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

  static constexpr std::array<std::uint32_t, 8> kInitialState = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

  static std::uint32_t rotate_right(std::uint32_t value,
                                    std::uint32_t amount) noexcept {
    return (value >> amount) | (value << (32U - amount));
  }

  static std::array<std::uint8_t, 32> digest(const std::string& payload) {
    std::vector<std::uint8_t> padded;
    padded.reserve(payload.size() + 72U);
    for (const char value : payload) {
      padded.push_back(static_cast<std::uint8_t>(value));
    }
    padded.push_back(static_cast<std::uint8_t>(0x80U));
    while (padded.size() % 64U != 56U) {
      padded.push_back(static_cast<std::uint8_t>(0U));
    }
    const std::uint64_t bit_length =
        static_cast<std::uint64_t>(payload.size()) * 8U;
    for (int shift = 56; shift >= 0; shift -= 8) {
      padded.push_back(static_cast<std::uint8_t>(
          (bit_length >> static_cast<unsigned int>(shift)) & 0xffU));
    }

    std::array<std::uint32_t, 8> state = kInitialState;
    for (std::size_t block = 0U; block < padded.size(); block += 64U) {
      std::array<std::uint32_t, 64> words{};
      for (std::size_t word = 0U; word < 16U; ++word) {
        const std::size_t offset = block + word * 4U;
        words[word] =
            (static_cast<std::uint32_t>(padded[offset]) << 24U) |
            (static_cast<std::uint32_t>(padded[offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(padded[offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(padded[offset + 3U]);
      }
      for (std::size_t word = 16U; word < words.size(); ++word) {
        const std::uint32_t s0 =
            rotate_right(words[word - 15U], 7U) ^
            rotate_right(words[word - 15U], 18U) ^
            (words[word - 15U] >> 3U);
        const std::uint32_t s1 =
            rotate_right(words[word - 2U], 17U) ^
            rotate_right(words[word - 2U], 19U) ^
            (words[word - 2U] >> 10U);
        words[word] = words[word - 16U] + s0 + words[word - 7U] + s1;
      }

      std::uint32_t a = state[0];
      std::uint32_t b = state[1];
      std::uint32_t c = state[2];
      std::uint32_t d = state[3];
      std::uint32_t e = state[4];
      std::uint32_t f = state[5];
      std::uint32_t g = state[6];
      std::uint32_t h = state[7];
      for (std::size_t word = 0U; word < words.size(); ++word) {
        const std::uint32_t s1 =
            rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const std::uint32_t choose = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + choose +
                                    kRoundConstants[word] + words[word];
        const std::uint32_t s0 =
            rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
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

    std::array<std::uint8_t, 32> output{};
    for (std::size_t word = 0U; word < state.size(); ++word) {
      output[word * 4U] = static_cast<std::uint8_t>(state[word] >> 24U);
      output[word * 4U + 1U] =
          static_cast<std::uint8_t>(state[word] >> 16U);
      output[word * 4U + 2U] = static_cast<std::uint8_t>(state[word] >> 8U);
      output[word * 4U + 3U] = static_cast<std::uint8_t>(state[word]);
    }
    return output;
  }
};

constexpr std::array<std::uint32_t, 64> Sha256::kRoundConstants;
constexpr std::array<std::uint32_t, 8> Sha256::kInitialState;

std::uint64_t add_mod(std::uint64_t left, std::uint64_t right,
                      std::uint64_t modulus) noexcept {
  // Both operands are below the non-zero modulus, so this avoids a uint64
  // overflow while computing (left + right) % modulus.
  if (left >= modulus - right) return left - (modulus - right);
  return left + right;
}

std::uint64_t multiply_mod(std::uint64_t left, std::uint64_t right,
                           std::uint64_t modulus) noexcept {
  std::uint64_t result = 0U;
  while (right != 0U) {
    if ((right & 1U) != 0U) result = add_mod(result, left, modulus);
    right >>= 1U;
    if (right != 0U) left = add_mod(left, left, modulus);
  }
  return result;
}

std::uint64_t digest_u64_le(const std::array<std::uint8_t, 32>& digest,
                            std::size_t offset) noexcept {
  std::uint64_t value = 0U;
  for (std::size_t byte = 0U; byte < 8U; ++byte) {
    value |= static_cast<std::uint64_t>(digest[offset + byte])
             << static_cast<unsigned int>(byte * 8U);
  }
  return value;
}

std::vector<std::int64_t> affine_sample_member_ids(
    ArrayView<std::int64_t> members, std::int64_t source_id,
    std::uint32_t candidate_size, std::uint64_t seed) {
  const std::size_t member_count = members.size();
  const auto source_position_it =
      std::lower_bound(members.begin(), members.end(), source_id);
  const bool contains_source = source_position_it != members.end() &&
                               *source_position_it == source_id;
  const std::size_t source_position =
      static_cast<std::size_t>(source_position_it - members.begin());
  const std::uint64_t domain =
      static_cast<std::uint64_t>(member_count - (contains_source ? 1U : 0U));
  const std::uint64_t requested = static_cast<std::uint64_t>(candidate_size);
  const std::uint64_t sample_count = std::min(requested, domain);
  std::vector<std::int64_t> result;
  result.reserve(static_cast<std::size_t>(sample_count));
  if (sample_count == 0U) return result;
  if (sample_count == domain) {
    for (const std::int64_t member : members) {
      if (!contains_source || member != source_id) result.push_back(member);
    }
    return result;
  }

  const std::string payload = std::string(kSamplerRevision) + ":" +
                              std::to_string(seed) + ":" +
                              std::to_string(source_id);
  const std::array<std::uint8_t, 32> hash = Sha256::digest(payload);
  std::uint64_t multiplier = digest_u64_le(hash, 0U) % domain;
  if (multiplier == 0U) multiplier = 1U;
  while (std::gcd(multiplier, domain) != 1U) {
    ++multiplier;
    if (multiplier == domain) multiplier = 1U;
  }
  const std::uint64_t offset = digest_u64_le(hash, 8U) % domain;
  for (std::uint64_t step = 0U; step < sample_count; ++step) {
    const std::uint64_t reduced =
        add_mod(multiply_mod(multiplier, step, domain), offset, domain);
    std::size_t member_position = static_cast<std::size_t>(reduced);
    if (contains_source && reduced >= static_cast<std::uint64_t>(source_position)) {
      ++member_position;
    }
    result.push_back(members[member_position]);
  }
  return result;
}

std::uint64_t mix64(std::uint64_t value) noexcept {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

struct ProjectionNeighborhoodIndex {
  std::size_t row_count = 0U;
  std::vector<std::size_t> order;
  std::vector<std::size_t> position;
};

ProjectionNeighborhoodIndex build_projection_neighborhood_index(
    const CorpusView& corpus, ArrayView<std::int64_t> members,
    std::uint64_t seed) {
  ProjectionNeighborhoodIndex index;
  index.row_count = members.size();
  index.order.resize(kProjectionCount * members.size());
  index.position.resize(kProjectionCount * members.size());
  const std::size_t dimension = static_cast<std::size_t>(corpus.vector_dim);
  const std::size_t term_count = std::min(kProjectionTermLimit, dimension);

  struct ProjectedRow {
    double score = 0.0;
    std::size_t row = 0U;
  };
  std::vector<ProjectedRow> ranked(members.size());
  for (std::size_t projection = 0U; projection < kProjectionCount;
       ++projection) {
    std::vector<std::size_t> coordinates;
    coordinates.reserve(term_count);
    std::vector<double> signs;
    signs.reserve(term_count);
    for (std::size_t term = 0U; term < term_count; ++term) {
      std::uint64_t state = mix64(
          seed ^ (static_cast<std::uint64_t>(projection + 1U) << 32U) ^
          static_cast<std::uint64_t>(term + 1U));
      std::size_t coordinate = static_cast<std::size_t>(
          state % static_cast<std::uint64_t>(dimension));
      while (std::find(coordinates.begin(), coordinates.end(), coordinate) !=
             coordinates.end()) {
        state = mix64(state);
        coordinate = static_cast<std::size_t>(
            state % static_cast<std::uint64_t>(dimension));
      }
      coordinates.push_back(coordinate);
      signs.push_back((mix64(state) & UINT64_C(1)) == 0U ? -1.0 : 1.0);
    }

    for (std::size_t row = 0U; row < members.size(); ++row) {
      const std::uint64_t object_id =
          static_cast<std::uint64_t>(members[row]);
      const std::size_t base =
          static_cast<std::size_t>(object_id) * dimension;
      double score = 0.0;
      for (std::size_t term = 0U; term < term_count; ++term) {
        score += signs[term] *
                 static_cast<double>(corpus.vectors[base + coordinates[term]]);
      }
      ranked[row] = ProjectedRow{score, row};
    }
    std::sort(ranked.begin(), ranked.end(),
              [&](const ProjectedRow& left, const ProjectedRow& right) {
                if (left.score != right.score) return left.score < right.score;
                return members[left.row] < members[right.row];
              });
    const std::size_t base = projection * members.size();
    for (std::size_t rank = 0U; rank < ranked.size(); ++rank) {
      index.order[base + rank] = ranked[rank].row;
      index.position[base + ranked[rank].row] = rank;
    }
  }
  return index;
}

std::vector<std::int64_t> projection_neighborhood_member_ids(
    const ProjectionNeighborhoodIndex& index,
    ArrayView<std::int64_t> members, std::size_t source_row,
    std::uint32_t candidate_size) {
  const std::size_t limit = std::min(
      static_cast<std::size_t>(candidate_size), members.size() - 1U);
  std::vector<std::size_t> selected_rows;
  selected_rows.reserve(limit);
  const auto append = [&](std::size_t row,
                          std::vector<std::size_t>* selected) {
    if (row == source_row ||
        std::find(selected->begin(), selected->end(), row) != selected->end()) {
      return;
    }
    selected->push_back(row);
  };

  for (std::size_t offset = 1U;
       selected_rows.size() < limit && offset < index.row_count; ++offset) {
    for (std::size_t projection = 0U;
         projection < kProjectionCount && selected_rows.size() < limit;
         ++projection) {
      const std::size_t base = projection * index.row_count;
      const std::size_t rank = index.position[base + source_row];
      if (rank >= offset) {
        append(index.order[base + rank - offset], &selected_rows);
      }
      if (selected_rows.size() == limit) break;
      if (rank + offset < index.row_count) {
        append(index.order[base + rank + offset], &selected_rows);
      }
    }
  }

  std::vector<std::int64_t> result;
  result.reserve(selected_rows.size());
  for (const std::size_t row : selected_rows) result.push_back(members[row]);
  return result;
}

void validate_options(const LeafBuildOptions& options) {
  if (options.candidate_size != 32U && options.candidate_size != 64U &&
      options.candidate_size != 96U && options.candidate_size != 128U) {
    throw LeafBuildError("candidate_size must be one of 32, 64, 96, 128");
  }
  if (options.max_degree != kMaxDegree) {
    throw LeafBuildError("max_degree must be exactly 24");
  }
  if (options.candidate_policy != LeafCandidatePolicy::kAffineSampleV1 &&
      options.candidate_policy !=
          LeafCandidatePolicy::kProjectionSymmetricPropagationV1) {
    throw LeafBuildError("unknown leaf candidate policy");
  }
  if (!std::isfinite(options.alpha) || options.alpha <= 0.0) {
    throw LeafBuildError("alpha must be finite and positive");
  }
  if (options.thread_count == 0U || options.thread_count > 16U) {
    throw LeafBuildError("thread_count must be in [1, 16]");
  }
}

void validate_corpus(const CorpusView& corpus) {
  if (!corpus.finite_values_prevalidated) {
    throw LeafBuildError("corpus finite-values contract is not prevalidated");
  }
  if (corpus.vector_dim == 0U || corpus.object_count == 0U) {
    throw LeafBuildError("corpus shape must be non-empty");
  }
  if (corpus.object_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw LeafBuildError("corpus object IDs exceed signed ID range");
  }
  const std::uint64_t dimension = static_cast<std::uint64_t>(corpus.vector_dim);
  if (corpus.object_count > std::numeric_limits<std::uint64_t>::max() /
                                dimension) {
    throw LeafBuildError("corpus vector shape overflows");
  }
  const std::uint64_t expected = corpus.object_count * dimension;
  if (expected > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw LeafBuildError("corpus vector shape exceeds addressable memory");
  }
  if (corpus.vectors.size() != static_cast<std::size_t>(expected)) {
    throw LeafBuildError("corpus vector array has the wrong shape");
  }
  if (corpus.vectors.data() == nullptr && expected != 0U) {
    throw LeafBuildError("corpus vector array has a null data pointer");
  }
}

void validate_members(const CorpusView& corpus,
                      ArrayView<std::int64_t> members) {
  if (members.empty() || members.data() == nullptr) {
    throw LeafBuildError("leaf membership must be non-empty");
  }
  std::int64_t previous = -1;
  for (std::size_t position = 0U; position < members.size(); ++position) {
    const std::int64_t object_id = members[position];
    if (object_id < 0 ||
        static_cast<std::uint64_t>(object_id) >= corpus.object_count) {
      throw LeafBuildError("leaf membership contains an out-of-bounds ID");
    }
    if (position != 0U && object_id <= previous) {
      throw LeafBuildError("leaf membership must be strictly increasing");
    }
    previous = object_id;
  }
}

double squared_distance(const numeric_detail::NumericHotContext& numeric_context,
                        std::int64_t source_id,
                        std::int64_t target_id) {
  return numeric_detail::squared_l2_f32_f64_seq_hot(
      numeric_context, source_id, target_id);
}

struct RankedCandidate {
  std::int64_t object_id = -1;
  double squared_distance = 0.0;
};

struct RowWork {
  std::uint64_t candidate_visited = 0U;
  std::uint64_t candidate_distance_evaluations = 0U;
  std::uint64_t candidate_budget_exhausted_rows = 0U;
  std::uint64_t candidate_membership_exhausted_rows = 0U;
  std::uint64_t alpha_distance_evaluations = 0U;
  std::uint64_t alpha_rejected = 0U;
  std::uint64_t pruned_rows = 0U;
  std::uint64_t pruned_edges = 0U;
  std::vector<std::int64_t> candidates;
  std::vector<std::int64_t> targets;
};

void checked_add(std::uint64_t& destination, std::uint64_t amount,
                 const char* label) {
  if (destination > std::numeric_limits<std::uint64_t>::max() - amount) {
    throw LeafBuildError(std::string("leaf evidence overflow: ") + label);
  }
  destination += amount;
}

void append_csr_row(std::vector<std::uint64_t>& row_ptr,
                    std::vector<std::int64_t>& values,
                    const std::vector<std::int64_t>& row) {
  if (values.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) -
          row.size()) {
    throw LeafBuildError("leaf CSR exceeds addressable offset range");
  }
  values.insert(values.end(), row.begin(), row.end());
  row_ptr.push_back(static_cast<std::uint64_t>(values.size()));
}

}  // namespace

#if defined(RFBB_LEAF_BUILD_TESTING)
std::uint64_t leaf_build_thread_spawn_count_for_testing() noexcept {
  return g_leaf_build_thread_spawn_count.load(std::memory_order_relaxed);
}

void reset_leaf_build_thread_spawn_count_for_testing() noexcept {
  g_leaf_build_thread_spawn_count.store(0U, std::memory_order_relaxed);
}
#endif

LeafBuildResult build_leaf(const CorpusView& corpus,
                           ArrayView<std::int64_t> members,
                           const LeafBuildOptions& options) {
  validate_options(options);
  validate_corpus(corpus);
  validate_members(corpus, members);
  const numeric_detail::NumericHotContext numeric_context =
      numeric_detail::make_numeric_hot_context(
          corpus.vectors, corpus.vector_dim, corpus.object_count,
          corpus.finite_values_prevalidated);
  if (members.size() == std::numeric_limits<std::size_t>::max()) {
    throw LeafBuildError("leaf membership is too large for CSR row pointers");
  }

  const std::size_t row_count = members.size();
  const std::size_t requested_workers =
      static_cast<std::size_t>(options.thread_count);
  const std::size_t worker_count = std::min(requested_workers, row_count);
  std::vector<RowWork> rows(row_count);
  std::vector<std::exception_ptr> worker_errors(worker_count);

  const double alpha_squared = bounded_blowup::alpha_squared(options.alpha);
  ProjectionNeighborhoodIndex projection_index;
  if (options.candidate_policy ==
      LeafCandidatePolicy::kProjectionSymmetricPropagationV1) {
    projection_index =
        build_projection_neighborhood_index(corpus, members, options.seed);
  }
  const auto rank_and_prune = [&](std::int64_t source_id,
                                  const std::vector<std::int64_t>& candidates,
                                  RowWork* work) {
    std::vector<RankedCandidate> ranked;
    ranked.reserve(candidates.size());
    for (const std::int64_t target_id : candidates) {
      ranked.push_back(
          RankedCandidate{target_id, squared_distance(numeric_context, source_id,
                                                      target_id)});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const RankedCandidate& left, const RankedCandidate& right) {
                if (left.squared_distance != right.squared_distance) {
                  return left.squared_distance < right.squared_distance;
                }
                return left.object_id < right.object_id;
              });
    work->candidates.clear();
    work->candidates.reserve(ranked.size());
    for (const RankedCandidate& candidate : ranked) {
      work->candidates.push_back(candidate.object_id);
    }

    std::vector<RankedCandidate> selected;
    selected.reserve(std::min<std::size_t>(ranked.size(), kMaxDegree));
    for (const RankedCandidate& candidate : ranked) {
      bool rejected = false;
      for (const RankedCandidate& chosen : selected) {
        const double chosen_to_candidate =
            squared_distance(numeric_context, chosen.object_id,
                             candidate.object_id);
        ++work->alpha_distance_evaluations;
        if (numeric_detail::alpha_product_hot(alpha_squared,
                                              chosen_to_candidate) <=
            candidate.squared_distance) {
          rejected = true;
          break;
        }
      }
      if (rejected) {
        ++work->alpha_rejected;
        continue;
      }
      selected.push_back(candidate);
      if (selected.size() == static_cast<std::size_t>(kMaxDegree)) break;
    }
    work->targets.clear();
    work->targets.reserve(selected.size());
    for (const RankedCandidate& candidate : selected) {
      work->targets.push_back(candidate.object_id);
    }
    if (work->targets != work->candidates) ++work->pruned_rows;
    work->pruned_edges += static_cast<std::uint64_t>(
        work->candidates.size() - work->targets.size());
  };
  auto process_row = [&](std::size_t row_index) {
    RowWork& work = rows[row_index];
    const std::int64_t source_id = members[row_index];
    const std::vector<std::int64_t> sampled =
        options.candidate_policy == LeafCandidatePolicy::kAffineSampleV1
            ? affine_sample_member_ids(members, source_id,
                                       options.candidate_size, options.seed)
            : projection_neighborhood_member_ids(
                  projection_index, members, row_index,
                  options.candidate_size);
    work.candidate_visited = static_cast<std::uint64_t>(sampled.size());
    work.candidate_distance_evaluations = work.candidate_visited;

    const std::uint64_t available = static_cast<std::uint64_t>(members.size() - 1U);
    if (available > static_cast<std::uint64_t>(options.candidate_size)) {
      work.candidate_budget_exhausted_rows = 1U;
    } else {
      work.candidate_membership_exhausted_rows = 1U;
    }

    rank_and_prune(source_id, sampled, &work);
  };

  const auto run_pass = [&](const auto& pass) {
    std::fill(worker_errors.begin(), worker_errors.end(), nullptr);
    const auto run_worker = [&](std::size_t worker_id) {
      try {
        // Keep the expensive contract checks at the worker boundary.  The row
        // loop below is deliberately limited to fixed-order arithmetic.
        numeric_detail::validate_numeric_hot_context(numeric_context);
        for (std::size_t row_index = worker_id; row_index < row_count;
             row_index += worker_count) {
          pass(row_index);
        }
      } catch (...) {
        worker_errors[worker_id] = std::current_exception();
      }
    };

    if (requested_workers == 1U) {
      run_worker(0U);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(worker_count);
      try {
        for (std::size_t worker_id = 0U; worker_id < worker_count; ++worker_id) {
          workers.emplace_back([&, worker_id]() {
#if defined(RFBB_LEAF_BUILD_TESTING)
            g_leaf_build_thread_spawn_count.fetch_add(
                1U, std::memory_order_relaxed);
#endif
            run_worker(worker_id);
          });
        }
      } catch (...) {
        const std::exception_ptr creation_error = std::current_exception();
        for (std::thread& worker : workers) {
          if (worker.joinable()) worker.join();
        }
        std::rethrow_exception(creation_error);
      }
      for (std::thread& worker : workers) {
        if (worker.joinable()) worker.join();
      }
    }
    for (const std::exception_ptr& error : worker_errors) {
      if (error != nullptr) std::rethrow_exception(error);
    }
  };

  run_pass(process_row);
  if (options.candidate_policy ==
      LeafCandidatePolicy::kProjectionSymmetricPropagationV1) {
    std::vector<std::vector<std::int64_t>> first_targets;
    first_targets.reserve(row_count);
    for (const RowWork& work : rows) first_targets.push_back(work.targets);
    std::vector<std::vector<std::int64_t>> symmetric = first_targets;
    for (std::size_t source_row = 0U; source_row < row_count; ++source_row) {
      for (const std::int64_t target_id : first_targets[source_row]) {
        const auto target =
            std::lower_bound(members.begin(), members.end(), target_id);
        if (target == members.end() || *target != target_id) {
          throw LeafBuildError("projection target is outside membership");
        }
        const std::size_t target_row =
            static_cast<std::size_t>(target - members.begin());
        symmetric[target_row].push_back(members[source_row]);
      }
    }
    for (std::size_t row = 0U; row < row_count; ++row) {
      std::vector<std::int64_t>& neighbors = symmetric[row];
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(
          std::remove(neighbors.begin(), neighbors.end(), members[row]),
          neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                      neighbors.end());
    }
    const auto propagate_row = [&](std::size_t row_index) {
      RowWork& work = rows[row_index];
      std::vector<std::int64_t> propagated = symmetric[row_index];
      for (const std::int64_t neighbor_id : symmetric[row_index]) {
        const auto neighbor =
            std::lower_bound(members.begin(), members.end(), neighbor_id);
        if (neighbor == members.end() || *neighbor != neighbor_id) {
          throw LeafBuildError("projection neighbor is outside membership");
        }
        const std::size_t neighbor_row =
            static_cast<std::size_t>(neighbor - members.begin());
        propagated.insert(propagated.end(), symmetric[neighbor_row].begin(),
                          symmetric[neighbor_row].end());
      }
      const std::int64_t source_id = members[row_index];
      std::sort(propagated.begin(), propagated.end());
      propagated.erase(
          std::remove(propagated.begin(), propagated.end(), source_id),
          propagated.end());
      propagated.erase(std::unique(propagated.begin(), propagated.end()),
                       propagated.end());
      work.candidate_visited +=
          static_cast<std::uint64_t>(propagated.size());
      work.candidate_distance_evaluations +=
          static_cast<std::uint64_t>(propagated.size());
      rank_and_prune(source_id, propagated, &work);
    };
    run_pass(propagate_row);
  }

  LeafBuildResult result;
  result.row_object_ids.assign(members.begin(), members.end());
  result.candidate_row_ptr.reserve(row_count + 1U);
  result.row_ptr.reserve(row_count + 1U);
  result.candidate_row_ptr.push_back(0U);
  result.row_ptr.push_back(0U);
  result.evidence.rows = static_cast<std::uint64_t>(row_count);
  for (const RowWork& work : rows) {
    checked_add(result.evidence.candidate_visited, work.candidate_visited,
                "candidate_visited");
    checked_add(result.evidence.candidate_distance_evaluations,
                work.candidate_distance_evaluations,
                "candidate_distance_evaluations");
    checked_add(result.evidence.candidate_budget_exhausted_rows,
                work.candidate_budget_exhausted_rows,
                "candidate_budget_exhausted_rows");
    checked_add(result.evidence.candidate_membership_exhausted_rows,
                work.candidate_membership_exhausted_rows,
                "candidate_membership_exhausted_rows");
    checked_add(result.evidence.alpha_distance_evaluations,
                work.alpha_distance_evaluations, "alpha_distance_evaluations");
    checked_add(result.evidence.alpha_rejected, work.alpha_rejected,
                "alpha_rejected");
    checked_add(result.evidence.pruned_rows, work.pruned_rows, "pruned_rows");
    checked_add(result.evidence.pruned_edges, work.pruned_edges,
                "pruned_edges");
    append_csr_row(result.candidate_row_ptr, result.candidate_target_ids,
                   work.candidates);
    append_csr_row(result.row_ptr, result.target_ids, work.targets);
  }
  result.evidence.distance_evaluations =
      result.evidence.candidate_distance_evaluations;
  checked_add(result.evidence.distance_evaluations,
              result.evidence.alpha_distance_evaluations,
              "distance_evaluations");
  return result;
}

}  // namespace bounded_blowup

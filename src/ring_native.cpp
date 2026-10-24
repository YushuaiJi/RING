// Four-corpus searchopt r1: single and joint local-ID queues, batch4 and prefetch.
// Patience=0 preserves the previous search. This is an approximate heuristic,
// selected on learning queries, never a guarantee of recall.
// V3 engineering flags: 1=top-k selection,
// 2=vector prefetch, 4=compact all-legal multi-window queues/adjacency.
// SPDX-License-Identifier: MIT
// Fullscale V1: immutable density-adaptive membership lookup on Resource V15.
// Construction, distance kernels, worker semantics and FRNSG015 stay unchanged.
// V14's optional nearest2 multi-graph entry selection supports matched ablations;
// its default all retains the V15 scheduler, candidate and stopping semantics.
// Adds a predicate-free, bridge-free single-graph search path for workloads
// whose complete legal membership is exactly one stored graph. The caller
// must establish that equality; this path does NOT check range predicates.
// NOT a complete alpha-cover builder or an exact nearest-neighbor algorithm.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bounded_blowup/leaf_build.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define FRANS_V11_AVX2_DISPATCH 1
#else
#define FRANS_V11_AVX2_DISPATCH 0
#endif

namespace py = pybind11;
namespace fv11 {
using Id = std::int64_t;
using Ranked = std::pair<double, Id>;
constexpr std::size_t kProjections = 8;
class Graph;

// These counters cover this module's parallel_rows calls, not the unchanged
// bounded_blowup leaf builder. They count actual thread launches, not CPU time.
std::atomic<std::uint64_t> row_calls{0}, row_items{0}, row_thread_launches{0};
std::atomic<unsigned> row_max_workers{0};

// Versioned binary graph payload; vectors/attributes are persisted once by
// the adapter. Fixed-width values use host byte order with an endian marker.
template<class T> void write_binary(std::ostream& stream, const T& value) {
  static_assert(std::is_trivially_copyable<T>::value, "binary scalar required");
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!stream) throw std::runtime_error("graph write failed");
}
template<class T> T read_binary(std::istream& stream) {
  static_assert(std::is_trivially_copyable<T>::value, "binary scalar required");
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) throw std::runtime_error("truncated graph file");
  return value;
}
void write_ids(std::ostream& stream, const std::vector<Id>& values) {
  write_binary(stream, static_cast<std::uint64_t>(values.size()));
  if (!values.empty()) stream.write(reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(Id)));
  if (!stream) throw std::runtime_error("graph IDs write failed");
}
std::vector<Id> read_ids(std::istream& stream, std::size_t maximum) {
  auto count = read_binary<std::uint64_t>(stream);
  if (count > maximum || count > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamsize>::max()) / sizeof(Id))
    throw std::runtime_error("invalid graph ID count");
  std::vector<Id> values(static_cast<std::size_t>(count));
  if (count) stream.read(reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(count * sizeof(Id)));
  if (!stream) throw std::runtime_error("truncated graph IDs");
  return values;
}

bool supports_avx2_batch4() {
#if FRANS_V11_AVX2_DISPATCH
  static const bool supported = __builtin_cpu_supports("avx2");
  return supported;
#else
  return false;
#endif
}

#if FRANS_V11_AVX2_DISPATCH
// Each lane is a different object, NOT a partial sum over dimensions.
// Keep the scalar j=0..dim-1 addition order in every lane; no FMA/reduction.
// CMake's -ffp-contract=off prevents fusion. Disabling the ISA's FMA feature
// here conflicts with GCC 8 intrinsics compiled under -march=native.
__attribute__((target("avx2")))
void distance4_avx2(const float* query, const float* p0, const float* p1,
                    const float* p2, const float* p3, std::size_t dim,
                    double* output) {
  __m256d sums = _mm256_setzero_pd();
  for (std::size_t j = 0; j < dim; ++j) {
    const __m128 values = _mm_set_ps(p3[j], p2[j], p1[j], p0[j]);
    const __m256d difference = _mm256_sub_pd(
        _mm256_set1_pd(static_cast<double>(query[j])), _mm256_cvtps_pd(values));
    const __m256d squared = _mm256_mul_pd(difference, difference);
    sums = _mm256_add_pd(sums, squared);
  }
  _mm256_storeu_pd(output, sums);
}
#endif

std::uint64_t mix(std::uint64_t x) {
  x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

template<class Function>
void parallel_rows(std::size_t count, unsigned workers, Function fn) {
  if (!count) return;
  workers = static_cast<unsigned>(std::min<std::size_t>(workers, count));
  ++row_calls; row_items += count;
  unsigned previous_max = row_max_workers.load();
  while (previous_max < workers && !row_max_workers.compare_exchange_weak(previous_max, workers)) {}
  std::vector<std::exception_ptr> failures(workers);
  auto run = [&](unsigned worker) {
    try { for (std::size_t row = worker; row < count; row += workers) fn(row); }
    catch (...) { failures[worker] = std::current_exception(); }
  };
  if (workers == 1) run(0);
  else {
    std::vector<std::thread> threads;
    try { for (unsigned w = 0; w < workers; ++w) {
      threads.emplace_back(run, w); ++row_thread_launches;
    } }
    catch (...) { for (auto& t : threads) t.join(); throw; }
    for (auto& t : threads) t.join();
  }
  for (auto& failure : failures) if (failure) std::rethrow_exception(failure);
}

class Corpus : public std::enable_shared_from_this<Corpus> {
 public:
  std::size_t n = 0, dim = 0;
  std::vector<float> vectors;
  std::array<std::vector<double>, kProjections> projections;

  explicit Corpus(py::array_t<float, py::array::c_style> values) {
    auto b = values.request();
    if (b.ndim != 2 || b.shape[0] <= 0 || b.shape[1] <= 0)
      throw std::invalid_argument("vectors must be nonempty float32 [N,D]");
    n = static_cast<std::size_t>(b.shape[0]);
    dim = static_cast<std::size_t>(b.shape[1]);
    if (n > std::numeric_limits<std::size_t>::max() / dim ||
        dim > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("corpus shape is too large");
    const auto* begin = static_cast<const float*>(b.ptr);
    vectors.assign(begin, begin + n * dim);
    for (float v : vectors) if (!std::isfinite(v))
      throw std::invalid_argument("vectors must be finite");
    // Same inexpensive projection-neighborhood idea as the existing leaf
    // builder. Scores are shared once; each target membership only sorts IDs.
    for (std::size_t p = 0; p < kProjections; ++p) {
      projections[p].resize(n);
      std::vector<std::size_t> coordinates;
      std::vector<double> signs;
      for (std::size_t t = 0; t < std::min<std::size_t>(16, dim); ++t) {
        auto state = mix(((p + 1) << 32) ^ (t + 1));
        std::size_t c = state % dim;
        while (std::find(coordinates.begin(), coordinates.end(), c) != coordinates.end()) {
          state = mix(state); c = state % dim;
        }
        coordinates.push_back(c);
        signs.push_back((mix(state) & 1) ? 1.0 : -1.0);
      }
      for (std::size_t row = 0; row < n; ++row) {
        double score = 0;
        for (std::size_t t = 0; t < coordinates.size(); ++t)
          score += signs[t] * vectors[row * dim + coordinates[t]];
        projections[p][row] = score;
      }
    }
  }

  void check_id(Id id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= n)
      throw std::invalid_argument("object ID outside corpus");
  }
  std::vector<Id> canonical(std::vector<Id> ids) const {
    for (Id id : ids) check_id(id);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
  }
  double distance(const float* a, const float* b) const {
    double sum = 0;
    for (std::size_t j = 0; j < dim; ++j) {
      double d = static_cast<double>(a[j]) - static_cast<double>(b[j]);
      sum += d * d;
    }
    return sum;
  }
  double distance(Id a, Id b) const {
    return distance(vectors.data() + static_cast<std::size_t>(a) * dim,
                    vectors.data() + static_cast<std::size_t>(b) * dim);
  }
  void distance4(const float* query, const std::array<Id, 4>& ids,
                 std::array<double, 4>& output) const {
#if FRANS_V11_AVX2_DISPATCH
    distance4_avx2(query, vectors.data() + static_cast<std::size_t>(ids[0]) * dim,
                   vectors.data() + static_cast<std::size_t>(ids[1]) * dim,
                   vectors.data() + static_cast<std::size_t>(ids[2]) * dim,
                   vectors.data() + static_cast<std::size_t>(ids[3]) * dim,
                   dim, output.data());
#else
    for (std::size_t i = 0; i < ids.size(); ++i)
      output[i] = distance(query, vectors.data() + static_cast<std::size_t>(ids[i]) * dim);
#endif
  }
  static void check_alpha(double alpha) {
    if (!std::isfinite(alpha) || alpha < 1.0 || !std::isfinite(alpha * alpha))
      throw std::invalid_argument("alpha must be finite and >= 1");
  }
  // Existing outgoing edges are permanent witnesses; only candidates may be
  // rejected. No reverse edge or degree cap is introduced here.
  std::vector<Id> prune(Id source, std::vector<Id> candidates,
                        const std::vector<Id>& inherited, double alpha) const {
    candidates = canonical(std::move(candidates));
    std::vector<Ranked> ranked;
    for (Id id : candidates)
      if (id != source && std::find(inherited.begin(), inherited.end(), id) == inherited.end())
        ranked.emplace_back(distance(source, id), id);
    std::sort(ranked.begin(), ranked.end());
    std::vector<Id> selected = inherited;
    const double alpha2 = alpha * alpha;
    for (auto candidate : ranked) {
      bool covered = false;
      for (Id witness : selected)
        if (alpha2 * distance(witness, candidate.second) <= candidate.first) {
          covered = true; break;
        }
      if (!covered) selected.push_back(candidate.second);
    }
    return selected;
  }
  std::vector<Id> alpha_prune(Id source, std::vector<Id> candidates, double alpha) const {
    check_id(source); check_alpha(alpha);
    return prune(source, std::move(candidates), {}, alpha);
  }
  std::shared_ptr<Graph> build(std::vector<Id> members, unsigned candidate_size,
                               double alpha, unsigned workers);
};

class CandidateIndex {
  const Corpus& corpus_;
  const std::vector<Id>& members_;
  std::array<std::vector<Ranked>, kProjections> order_;
 public:
  CandidateIndex(const Corpus& corpus, const std::vector<Id>& members)
      : corpus_(corpus), members_(members) {
    for (std::size_t p = 0; p < kProjections; ++p) {
      auto& rows = order_[p]; rows.reserve(members.size());
      for (Id id : members) rows.emplace_back(corpus.projections[p][id], id);
      std::sort(rows.begin(), rows.end());
    }
  }
  std::vector<Id> candidates(Id source, std::size_t cap) const {
    if (members_.size() <= cap) {
      std::vector<Id> ids = members_;
      ids.erase(std::remove(ids.begin(), ids.end(), source), ids.end());
      return ids;
    }
    std::array<std::size_t, kProjections> ranks;
    for (std::size_t p = 0; p < kProjections; ++p)
      ranks[p] = static_cast<std::size_t>(std::lower_bound(
          order_[p].begin(), order_[p].end(), Ranked{corpus_.projections[p][source], source}) - order_[p].begin());
    std::vector<Id> ids; ids.reserve(cap);
    auto add = [&](Id id) {
      if (id != source && ids.size() < cap &&
          std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    };
    for (std::size_t offset = 0; ids.size() < cap && offset < members_.size(); ++offset)
      for (std::size_t p = 0; p < kProjections && ids.size() < cap; ++p) {
        if (ranks[p] + offset < members_.size()) add(order_[p][ranks[p] + offset].second);
        if (ranks[p] > offset) add(order_[p][ranks[p] - offset - 1].second);
      }
    return ids;
  }
};

struct SearchResult {
  std::vector<Id> ids, expanded;
  std::vector<double> distances;
  Id last = -1;
  std::size_t expansions = 0, evaluations = 0, pushes = 0;
  std::size_t distance_batches = 0;
  bool avx2_batch4 = false;
  bool frontier_exhausted = false;
};

// A graph's canonical members never change after construction. Sparse tables
// therefore need only row indices: members[row] is the corresponding key.
// Open addressing at <= 50% load provides expected O(1) hot-path membership
// lookup without allocating a corpus-sized array for every small graph. Dense
// storage is retained whenever it costs no more than the sparse table. The
// lookup is rebuilt from logical member IDs when loading FRNSG015 payloads.
class LocalLookup {
  const std::vector<Id>& members_;
  std::size_t corpus_size_;
  std::vector<Id> slots_;
  bool dense_ = false;
 public:
  LocalLookup(std::size_t corpus_size, const std::vector<Id>& members)
      : members_(members), corpus_size_(corpus_size) {
    if (members.empty()) return;
    std::size_t capacity = 1;
    if (members.size() > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("graph membership lookup is too large");
    const auto minimum = members.size() * 2;
    while (capacity < minimum && capacity < corpus_size_) {
      if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
        capacity = corpus_size_; break;
      }
      capacity *= 2;
    }
    dense_ = capacity >= corpus_size_;
    slots_.assign(dense_ ? corpus_size_ : capacity, -1);
    for (std::size_t row = 0; row < members.size(); ++row) {
      const Id id = members[row];
      std::size_t slot = dense_ ? static_cast<std::size_t>(id)
          : static_cast<std::size_t>(mix(static_cast<std::uint64_t>(id))) & (capacity - 1);
      if (!dense_) while (slots_[slot] >= 0) slot = (slot + 1) & (capacity - 1);
      slots_[slot] = static_cast<Id>(row);
    }
  }
  Id operator[](Id id) const noexcept {
    if (id < 0 || static_cast<std::size_t>(id) >= corpus_size_ || slots_.empty()) return -1;
    if (dense_) return slots_[static_cast<std::size_t>(id)];
    const auto mask = slots_.size() - 1;
    auto slot = static_cast<std::size_t>(mix(static_cast<std::uint64_t>(id))) & mask;
    while (slots_[slot] >= 0) {
      const Id row = slots_[slot];
      if (members_[static_cast<std::size_t>(row)] == id) return row;
      slot = (slot + 1) & mask;
    }
    return -1;
  }
  const char* mode() const noexcept { return dense_ ? "dense" : "sparse_open_addressed"; }
  std::size_t storage_bytes() const noexcept { return slots_.capacity() * sizeof(Id); }
};

class Graph {
 public:
  std::shared_ptr<Corpus> corpus;
  std::vector<Id> members, entries;
  std::vector<std::vector<Id>> rows;
  LocalLookup local;
  unsigned candidate_size = 64, workers = 16;
  double alpha = 1.2;
  std::string operation = "build";
  std::size_t dirty_rows = 0, inherited_edges = 0, added_edges = 0;
  std::size_t degree_cap = 0, edges_before_pruning = 0, removed_edges = 0;
 private:
  std::mutex search_mutex_;
  std::vector<std::uint32_t> seen_;
  std::uint32_t epoch_ = 0;
  std::vector<Ranked> frontier_, result_heap_;
 public:
  Graph(std::shared_ptr<Corpus> c, std::vector<Id> ids, unsigned cap, double a, unsigned w)
      : corpus(std::move(c)), members(std::move(ids)), rows(members.size()),
        local(corpus->n, members), candidate_size(cap), workers(w), alpha(a), seen_(members.size(), 0) {
    const auto count = std::min<std::size_t>(8, members.size());
    for (std::size_t i = 0; i < count; ++i) entries.push_back(members[i * members.size() / count]);
    frontier_.reserve(members.size());
  }
  std::size_t edge_count() const {
    std::size_t count = 0; for (const auto& row : rows) count += row.size(); return count;
  }
  void save(const std::string& path) const {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open graph for writing: " + path);
    stream.write("FRNSG015", 8);
    write_binary(stream, UINT32_C(0x01020304));
    write_binary(stream, static_cast<std::uint64_t>(corpus->n));
    write_binary(stream, static_cast<std::uint64_t>(corpus->dim));
    write_binary(stream, static_cast<std::uint32_t>(candidate_size));
    write_binary(stream, static_cast<std::uint32_t>(workers));
    write_binary(stream, alpha);
    write_binary(stream, static_cast<std::uint64_t>(operation.size()));
    stream.write(operation.data(), static_cast<std::streamsize>(operation.size()));
    for (auto value : {dirty_rows, inherited_edges, added_edges, degree_cap,
                       edges_before_pruning, removed_edges})
      write_binary(stream, static_cast<std::uint64_t>(value));
    write_ids(stream, members);
    write_ids(stream, entries);
    for (const auto& row : rows) write_ids(stream, row);
    stream.flush();
    if (!stream) throw std::runtime_error("graph flush failed: " + path);
  }
  static std::shared_ptr<Graph> load(std::shared_ptr<Corpus> corpus, const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open graph for reading: " + path);
    std::array<char, 8> magic{}; stream.read(magic.data(), magic.size());
    if (!stream || std::string(magic.data(), magic.size()) != "FRNSG015" ||
        read_binary<std::uint32_t>(stream) != UINT32_C(0x01020304))
      throw std::runtime_error("unsupported graph version or byte order");
    auto n = read_binary<std::uint64_t>(stream), dim = read_binary<std::uint64_t>(stream);
    if (n != corpus->n || dim != corpus->dim)
      throw std::runtime_error("saved graph corpus shape mismatch");
    auto cap = read_binary<std::uint32_t>(stream), workers = read_binary<std::uint32_t>(stream);
    double alpha = read_binary<double>(stream);
    Corpus::check_alpha(alpha);
    if (!cap || workers < 1 || workers > 16) throw std::runtime_error("invalid saved graph configuration");
    auto operation_size = read_binary<std::uint64_t>(stream);
    if (operation_size > 1024) throw std::runtime_error("invalid graph operation length");
    std::string operation(static_cast<std::size_t>(operation_size), '\0');
    stream.read(operation.data(), static_cast<std::streamsize>(operation_size));
    std::array<std::size_t, 6> counters{};
    for (auto& value : counters) value = read_binary<std::uint64_t>(stream);
    auto members = read_ids(stream, corpus->n);
    if (members != corpus->canonical(members)) throw std::runtime_error("saved graph members not canonical");
    auto graph = std::make_shared<Graph>(corpus, std::move(members), cap, alpha, workers);
    graph->operation = std::move(operation);
    graph->dirty_rows = counters[0]; graph->inherited_edges = counters[1];
    graph->added_edges = counters[2]; graph->degree_cap = counters[3];
    graph->edges_before_pruning = counters[4]; graph->removed_edges = counters[5];
    graph->entries = read_ids(stream, graph->members.size());
    auto check_member = [&](Id id) {
      corpus->check_id(id);
      if (graph->local[id] < 0) throw std::runtime_error("saved graph target outside membership");
    };
    for (Id entry : graph->entries) check_member(entry);
    for (auto& row : graph->rows) {
      row = read_ids(stream, graph->members.size());
      for (Id target : row) check_member(target);
    }
    if (stream.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("trailing bytes in saved graph");
    return graph;
  }
  // Explicit engineering variant: inherited edges are no longer permanent
  // witnesses in OVER-CAP rows. Under-cap rows remain byte-for-byte unchanged.
  std::shared_ptr<Graph> pruned(std::size_t max_degree) const {
    if (!max_degree) throw std::invalid_argument("max_degree must be positive");
    auto result = std::make_shared<Graph>(corpus, members, candidate_size, alpha, workers);
    result->operation = "pruned";
    result->entries = entries;
    result->degree_cap = max_degree;
    result->edges_before_pruning = edge_count();
    for (const auto& row : rows) if (row.size() > max_degree) ++result->dirty_rows;
    parallel_rows(rows.size(), workers, [&](std::size_t r) {
      if (rows[r].size() <= max_degree) result->rows[r] = rows[r];
      else {
        auto selected = corpus->prune(members[r], rows[r], {}, alpha);
        if (selected.size() > max_degree) selected.resize(max_degree);
        result->rows[r] = std::move(selected);
      }
    });
    result->inherited_edges = result->edge_count();
    result->removed_edges = result->edges_before_pruning - result->inherited_edges;
    return result;
  }
  std::shared_ptr<Graph> merge(const Graph& other) const {
    if (corpus.get() != other.corpus.get())
      throw std::invalid_argument("merge requires the same Corpus instance");
    if (alpha != other.alpha || candidate_size != other.candidate_size)
      throw std::invalid_argument("merge requires matching alpha and candidate_size");
    std::vector<Id> combined, left_only, right_only;
    std::set_union(members.begin(), members.end(), other.members.begin(), other.members.end(),
                   std::back_inserter(combined));
    std::set_difference(members.begin(), members.end(), other.members.begin(), other.members.end(),
                        std::back_inserter(left_only));
    std::set_difference(other.members.begin(), other.members.end(), members.begin(), members.end(),
                        std::back_inserter(right_only));
    auto result = std::make_shared<Graph>(corpus, std::move(combined), candidate_size,
                                          alpha, std::max(workers, other.workers));
    result->operation = "merge";
    parallel_rows(result->members.size(), result->workers, [&](std::size_t r) {
      Id id = result->members[r]; auto& inherited = result->rows[r];
      if (local[id] >= 0) inherited = rows[local[id]];
      if (other.local[id] >= 0) {
        const auto& second = other.rows[other.local[id]];
        inherited.insert(inherited.end(), second.begin(), second.end());
      }
      std::sort(inherited.begin(), inherited.end());
      inherited.erase(std::unique(inherited.begin(), inherited.end()), inherited.end());
    });
    result->inherited_edges = result->edge_count();
    // A source present in both children already inherits both internal rows.
    // Rechecking old pairs (including shared targets) is unnecessary.
    if (!left_only.empty() && !right_only.empty()) {
      CandidateIndex left_candidates(*corpus, left_only), right_candidates(*corpus, right_only);
      std::vector<Id> sources = left_only;
      sources.insert(sources.end(), right_only.begin(), right_only.end());
      result->dirty_rows = sources.size();
      parallel_rows(sources.size(), result->workers, [&](std::size_t i) {
        Id source = sources[i]; auto r = static_cast<std::size_t>(result->local[source]);
        const auto& targets = local[source] >= 0 ? right_candidates : left_candidates;
        result->rows[r] = corpus->prune(source, targets.candidates(source, candidate_size),
                                       result->rows[r], alpha);
      });
    }
    result->added_edges = result->edge_count() - result->inherited_edges;
    return result;
  }
  // Optional offline refinement. Reuse membership and existing adjacency; seek
  // bounded two-hop candidates, then apply the same alpha pruning and cap.
  // This is a measured engineering variant, not a connectivity guarantee.
  std::shared_ptr<Graph> refined_twohop(std::size_t candidate_cap,
                                       std::size_t max_degree) const {
    if (candidate_cap < max_degree || !max_degree)
      throw std::invalid_argument("candidate_cap must be >= positive max_degree");
    auto result = std::make_shared<Graph>(corpus, members, candidate_size, alpha, workers);
    result->operation = "refined_twohop";
    result->entries = entries;
    result->degree_cap = max_degree;
    result->edges_before_pruning = edge_count();
    result->dirty_rows = members.size();
    parallel_rows(rows.size(), workers, [&](std::size_t r) {
      const Id source = members[r];
      std::vector<Id> candidates = rows[r];
      std::size_t longest = 0;
      for (Id neighbor : rows[r])
        longest = std::max(longest, rows[static_cast<std::size_t>(local[neighbor])].size());
      // Round-robin across old neighbors avoids exhausting the budget in just
      // one direction. All candidate IDs are already legal graph members.
      for (std::size_t rank = 0; rank < longest && candidates.size() < candidate_cap; ++rank)
        for (Id neighbor : rows[r]) {
          const auto& second = rows[static_cast<std::size_t>(local[neighbor])];
          if (rank < second.size() && second[rank] != source &&
              std::find(candidates.begin(), candidates.end(), second[rank]) == candidates.end())
            candidates.push_back(second[rank]);
          if (candidates.size() >= candidate_cap) break;
        }
      auto selected = corpus->prune(source, std::move(candidates), {}, alpha);
      if (selected.size() > max_degree) selected.resize(max_degree);
      result->rows[r] = std::move(selected);
    });
    for (std::size_t r = 0; r < rows.size(); ++r)
      for (Id target : result->rows[r])
        if (std::find(rows[r].begin(), rows[r].end(), target) != rows[r].end())
          ++result->inherited_edges;
    result->added_edges = result->edge_count() - result->inherited_edges;
    result->removed_edges = result->edges_before_pruning - result->inherited_edges;
    return result;
  }
  std::shared_ptr<Graph> extract_repair(std::vector<Id> selected) const {
    selected = corpus->canonical(std::move(selected));
    for (Id id : selected) if (local[id] < 0)
      throw std::invalid_argument("extract members must be a subset");
    auto result = std::make_shared<Graph>(corpus, std::move(selected), candidate_size, alpha, workers);
    result->operation = "extract_repair";
    std::vector<std::size_t> dirty;
    std::vector<std::uint8_t> dirty_flags(result->members.size(), 0);
    parallel_rows(result->members.size(), workers, [&](std::size_t r) {
      const auto& old = rows[local[result->members[r]]];
      auto& retained = result->rows[r];
      for (Id target : old) if (result->local[target] >= 0) retained.push_back(target);
      dirty_flags[r] = retained.size() != old.size();
    });
    result->inherited_edges = result->edge_count();
    for (std::size_t r = 0; r < dirty_flags.size(); ++r)
      if (dirty_flags[r]) dirty.push_back(r);
    result->dirty_rows = dirty.size();
    if (!dirty.empty()) {
      CandidateIndex candidates(*corpus, result->members);
      parallel_rows(dirty.size(), workers, [&](std::size_t i) {
        auto row = dirty[i]; Id source = result->members[row];
        result->rows[row] = corpus->prune(source,
            candidates.candidates(source, candidate_size), result->rows[row], alpha);
      });
    }
    result->added_edges = result->edge_count() - result->inherited_edges;
    return result;
  }
  std::shared_ptr<Graph> expand(std::vector<Id> selected) const {
    selected = corpus->canonical(std::move(selected));
    if (!std::includes(selected.begin(), selected.end(), members.begin(), members.end()))
      throw std::invalid_argument("expand members must be a superset (full new membership)");
    auto result = std::make_shared<Graph>(corpus, std::move(selected), candidate_size, alpha, workers);
    result->operation = "expand";
    std::vector<Id> added;
    for (Id id : result->members) if (local[id] < 0) added.push_back(id);
    result->inherited_edges = edge_count();
    if (added.empty()) { result->rows = rows; return result; }
    CandidateIndex new_candidates(*corpus, added), all_candidates(*corpus, result->members);
    parallel_rows(result->members.size(), workers, [&](std::size_t r) {
      Id source = result->members[r];
      if (local[source] >= 0)
        result->rows[r] = corpus->prune(source, new_candidates.candidates(source, candidate_size),
                                        rows[local[source]], alpha);
      else
        result->rows[r] = corpus->prune(source, all_candidates.candidates(source, candidate_size), {}, alpha);
    });
    result->dirty_rows = result->members.size();
    result->added_edges = result->edge_count() - result->inherited_edges;
    return result;
  }
  SearchResult search(const float* query, std::size_t k, std::size_t budget,
                      const std::vector<Id>& requested_entries,
                      const std::uint8_t* allowed, bool trace, bool request_batch4) {
    if (!k || !budget) throw std::invalid_argument("k and budget must be positive");
    std::lock_guard<std::mutex> lock(search_mutex_);
    SearchResult out;
    out.avx2_batch4 = request_batch4 && supports_avx2_batch4();
    if (++epoch_ == 0) { std::fill(seen_.begin(), seen_.end(), 0); epoch_ = 1; }
    frontier_.clear(); result_heap_.clear(); result_heap_.reserve(k);
    const auto& starts = requested_entries.empty() ? entries : requested_entries;
    auto enqueue = [&](Id id, double d) {
      ++out.evaluations; ++out.pushes;
      Ranked hit{d, id}; frontier_.push_back(hit);
      std::push_heap(frontier_.begin(), frontier_.end(), std::greater<Ranked>());
      if (!allowed || allowed[id]) {
        if (result_heap_.size() < k) {
          result_heap_.push_back(hit); std::push_heap(result_heap_.begin(), result_heap_.end());
        } else if (hit < result_heap_.front()) {
          std::pop_heap(result_heap_.begin(), result_heap_.end()); result_heap_.back() = hit;
          std::push_heap(result_heap_.begin(), result_heap_.end());
        }
      }
    };
    auto visit = [&](Id id) {
      std::size_t row = static_cast<std::size_t>(local[id]);
      if (seen_[row] == epoch_) return;
      seen_[row] = epoch_;
      enqueue(id, corpus->distance(query,
          corpus->vectors.data() + static_cast<std::size_t>(id) * corpus->dim));
    };
    for (Id id : starts) {
      corpus->check_id(id);
      if (local[id] < 0) throw std::invalid_argument("entry ID is not in this graph");
      visit(id);
    }
    while (!frontier_.empty() && out.expansions < budget) {
      std::pop_heap(frontier_.begin(), frontier_.end(), std::greater<Ranked>());
      Id id = frontier_.back().second; frontier_.pop_back();
      out.last = id; ++out.expansions;
      if (trace) out.expanded.push_back(id);
      const auto& neighbors = rows[static_cast<std::size_t>(local[id])];
      if (!out.avx2_batch4) {
        for (Id neighbor : neighbors) visit(neighbor);
      } else {
        std::array<Id, 4> pending;
        std::array<double, 4> distances;
        std::size_t count = 0;
        for (Id neighbor : neighbors) {
          const auto row = static_cast<std::size_t>(local[neighbor]);
          if (seen_[row] == epoch_) continue;
          seen_[row] = epoch_;
          pending[count++] = neighbor;
          if (count == pending.size()) {
            corpus->distance4(query, pending, distances);
            ++out.distance_batches;
            for (std::size_t i = 0; i < count; ++i) enqueue(pending[i], distances[i]);
            count = 0;
          }
        }
        for (std::size_t i = 0; i < count; ++i)
          enqueue(pending[i], corpus->distance(query,
              corpus->vectors.data() + static_cast<std::size_t>(pending[i]) * corpus->dim));
      }
    }
    out.frontier_exhausted = frontier_.empty();
    std::sort(result_heap_.begin(), result_heap_.end());
    for (auto hit : result_heap_) { out.distances.push_back(hit.first); out.ids.push_back(hit.second); }
    return out;
  }
};

std::shared_ptr<Graph> Corpus::build(std::vector<Id> members, unsigned cap,
                                     double alpha, unsigned workers) {
  check_alpha(alpha);
  if (workers < 1 || workers > 16) throw std::invalid_argument("workers must be in [1,16]");
  if (cap != 32 && cap != 64 && cap != 96 && cap != 128)
    throw std::invalid_argument("candidate_size must be 32,64,96,128");
  auto graph = std::make_shared<Graph>(shared_from_this(), canonical(std::move(members)), cap, alpha, workers);
  if (graph->members.empty()) return graph;
  bounded_blowup::CorpusView view{bounded_blowup::ArrayView<float>(vectors.data(), vectors.size()),
      static_cast<std::uint32_t>(dim), static_cast<std::uint64_t>(n), true};
  bounded_blowup::LeafBuildOptions options;
  options.candidate_size = cap; options.alpha = alpha; options.thread_count = workers;
  options.candidate_policy = bounded_blowup::LeafCandidatePolicy::kProjectionSymmetricPropagationV1;
  auto leaf = bounded_blowup::build_leaf(view,
      bounded_blowup::ArrayView<Id>(graph->members.data(), graph->members.size()), options);
  for (std::size_t row = 0; row < graph->members.size(); ++row)
    graph->rows[row].assign(leaf.target_ids.begin() + leaf.row_ptr[row], leaf.target_ids.begin() + leaf.row_ptr[row + 1]);
  graph->added_edges = graph->edge_count();
  return graph;
}

using FloatRanked = std::pair<float, Id>;
using SearchState = std::tuple<float, std::size_t, Id, std::uint32_t>;  // distance, active graph slot, object

float distance_f32_scalar(const float* a, const float* b, std::size_t dim) {
  float sum = 0;
  for (std::size_t j = 0; j < dim; ++j) { const float d = a[j] - b[j]; sum += d * d; }
  return sum;
}
#if FRANS_V11_AVX2_DISPATCH
__attribute__((target("avx2")))
float distance_f32_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 sum = _mm256_setzero_ps();
  std::size_t j = 0;
  for (; j + 8 <= dim; j += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
    sum = _mm256_add_ps(sum, _mm256_mul_ps(d, d));
  }
  alignas(32) float parts[8]; _mm256_store_ps(parts, sum);
  float result = 0; for (float part : parts) result += part;
  for (; j < dim; ++j) { const float d = a[j] - b[j]; result += d * d; }
  return result;
}
// Four independent distances, each with exactly the previous eight-lane sum
// order. No FMA, compression, or changed candidate admission order.
__attribute__((target("avx2")))
void distances4_f32_avx2(const float* q, const std::array<const float*, 4>& p,
                         std::size_t dim, std::array<float, 4>& output) {
  __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
  std::size_t j = 0;
  for (; j + 8 <= dim; j += 8) {
    const auto query = _mm256_loadu_ps(q + j);
    const auto x = _mm256_sub_ps(query, _mm256_loadu_ps(p[0] + j));
    const auto y = _mm256_sub_ps(query, _mm256_loadu_ps(p[1] + j));
    const auto z = _mm256_sub_ps(query, _mm256_loadu_ps(p[2] + j));
    const auto w = _mm256_sub_ps(query, _mm256_loadu_ps(p[3] + j));
    a = _mm256_add_ps(a, _mm256_mul_ps(x, x));
    b = _mm256_add_ps(b, _mm256_mul_ps(y, y));
    c = _mm256_add_ps(c, _mm256_mul_ps(z, z));
    d = _mm256_add_ps(d, _mm256_mul_ps(w, w));
  }
  alignas(32) float parts[4][8];
  _mm256_store_ps(parts[0], a); _mm256_store_ps(parts[1], b);
  _mm256_store_ps(parts[2], c); _mm256_store_ps(parts[3], d);
  for (std::size_t i = 0; i < 4; ++i) {
    float sum = 0;
    for (float v : parts[i]) sum += v;
    for (std::size_t t = j; t < dim; ++t) { const float v = q[t] - p[i][t]; sum += v * v; }
    output[i] = sum;
  }
}
__attribute__((target("sse2")))
float distance_f32_sse(const float* a, const float* b, std::size_t dim) {
  __m128 sum = _mm_setzero_ps();
  std::size_t j = 0;
  for (; j + 4 <= dim; j += 4) {
    const __m128 d = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
    sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
  }
  alignas(16) float parts[4]; _mm_store_ps(parts, sum);
  float result = 0; for (float part : parts) result += part;
  for (; j < dim; ++j) { const float d = a[j] - b[j]; result += d * d; }
  return result;
}
#endif

struct JointResult {
  std::vector<Id> ids;
  std::vector<float> distances;
  std::vector<std::size_t> graph_order, per_graph_expansions;
  std::vector<Id> last_expanded;
  std::vector<std::vector<FloatRanked>> graph_hits;
  std::vector<std::pair<std::size_t, Id>> expanded_states;
  std::size_t expansions = 0, evaluations = 0, predicate_checks = 0;
  std::size_t graphs_searched = 0, bridge_starts = 0, candidate_pushes = 0;
  std::size_t frontier_peak = 0, effective_cap = 0, effective_beam = 0;
  std::size_t entry_candidates = 0, entry_starts = 0, entry_evaluations = 0;
  std::string stop_reason = "empty_request", distance_kernel;
};

struct SingleResult {
  std::vector<Id> ids;
  std::vector<float> distances;
  std::size_t expansions = 0, evaluations = 0, candidate_pushes = 0;
  std::size_t frontier_peak = 0, effective_beam = 0;
  std::size_t distance_batches = 0;
  std::size_t entry_candidates = 0, entry_starts = 0, entry_evaluations = 0;
  std::string stop_reason = "frontier_exhausted";
};

class SearchIndex {
  struct StoredGraph {
    std::shared_ptr<Graph> graph;
    std::vector<Id> entries;
    std::vector<std::uint32_t> discovered, expanded;
    std::vector<std::size_t> row_ptr;
    std::vector<std::uint32_t> local_neighbors, local_distance_epoch;
    std::vector<float> local_distances, packed_vectors;
  };
  using MinQueue = std::priority_queue<FloatRanked, std::vector<FloatRanked>, std::greater<FloatRanked>>;
  std::shared_ptr<Corpus> corpus_;
  std::size_t attributes_ = 0;
  std::vector<double> attrs_;
  std::vector<StoredGraph> graphs_;
  // Intentionally keyed by global source object, matching the learned bridge
  // model. Its target graph is explicit; inactive target graphs are ignored.
  std::vector<std::vector<std::pair<std::size_t, Id>>> bridges_;
  std::vector<float> distances_;
  std::vector<std::uint32_t> distance_epoch_, predicate_epoch_, answer_epoch_;
  std::vector<std::uint8_t> predicate_value_;
  std::uint32_t epoch_ = 0;
  std::mutex mutex_;
  // V11 single-graph scratch: one candidate min-heap and one bounded best
  // max-heap. Reused under mutex_; no per-graph frontier, bridge or predicate.
  std::vector<FloatRanked> single_frontier_, single_best_;
  using LocalHit = std::pair<float, std::uint32_t>;
  std::vector<LocalHit> local_frontier_, local_best_;
  unsigned single_optimization_ = 0, joint_optimization_ = 0;
  using DistanceFn = float (*)(const float*, const float*, std::size_t);
  DistanceFn distance_fn_ = distance_f32_scalar;
  std::string distance_kernel_ = "float32_scalar";
  unsigned engineering_mode_ = 0;

  void prefetch_vector(Id id) const {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(corpus_->vectors.data() + static_cast<std::size_t>(id) * corpus_->dim, 0, 1);
#else
    (void)id;
#endif
  }

  void next_epoch() {
    if (++epoch_) return;
    std::fill(distance_epoch_.begin(), distance_epoch_.end(), 0);
    std::fill(predicate_epoch_.begin(), predicate_epoch_.end(), 0);
    std::fill(answer_epoch_.begin(), answer_epoch_.end(), 0);
    for (auto& graph : graphs_) {
      std::fill(graph.discovered.begin(), graph.discovered.end(), 0);
      std::fill(graph.expanded.begin(), graph.expanded.end(), 0);
      std::fill(graph.local_distance_epoch.begin(), graph.local_distance_epoch.end(), 0);
    }
    epoch_ = 1;
  }
 public:
  SearchIndex(std::shared_ptr<Corpus> corpus, py::array_t<double, py::array::c_style> attrs)
      : corpus_(std::move(corpus)), bridges_(corpus_->n), distances_(corpus_->n),
        distance_epoch_(corpus_->n, 0), predicate_epoch_(corpus_->n, 0),
        answer_epoch_(corpus_->n, 0), predicate_value_(corpus_->n, 0) {
    auto b = attrs.request();
    if (b.ndim != 2 || static_cast<std::size_t>(b.shape[0]) != corpus_->n || b.shape[1] <= 0)
      throw std::invalid_argument("attrs must be float64 [Corpus.size,M], M>0");
    attributes_ = static_cast<std::size_t>(b.shape[1]);
    const auto* begin = static_cast<const double*>(b.ptr);
    attrs_.assign(begin, begin + corpus_->n * attributes_);
    for (double value : attrs_) if (!std::isfinite(value))
      throw std::invalid_argument("attributes must be finite");
#if FRANS_V11_AVX2_DISPATCH
    if (supports_avx2_batch4()) { distance_fn_ = distance_f32_avx2; distance_kernel_ = "float32_avx2_dim8"; }
    else if (__builtin_cpu_supports("sse2")) { distance_fn_ = distance_f32_sse; distance_kernel_ = "float32_sse_dim4"; }
#endif
  }
  std::size_t dimension() const { return corpus_->dim; }
  std::size_t attribute_count() const { return attributes_; }
  const std::string& distance_kernel() const { return distance_kernel_; }
  unsigned single_optimization() const { return single_optimization_; }
  void set_single_optimization(unsigned mode) {
    if (mode != 0 && mode != 1 && mode != 3 && mode != 7)
      throw std::invalid_argument("single optimization must be 0, 1, 3 or 7");
    std::lock_guard<std::mutex> lock(mutex_);
    single_optimization_ = mode;
  }
  void set_joint_optimization(unsigned mode) {
    if (mode != 0 && mode != 1 && mode != 3 && mode != 7)
      throw std::invalid_argument("joint optimization must be 0, 1, 3 or 7");
    std::lock_guard<std::mutex> lock(mutex_);
    joint_optimization_ = mode;
  }
  void set_entries(std::size_t graph_id, std::vector<Id> entries) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& stored = graphs_.at(graph_id);
    entries = corpus_->canonical(std::move(entries));
    if (entries.empty()) throw std::invalid_argument("entries required");
    for (Id id : entries) if (stored.graph->local[id] < 0)
      throw std::invalid_argument("entry outside graph");
    stored.entries = std::move(entries);
  }
  std::size_t add_graph(std::shared_ptr<Graph> graph, std::vector<Id> entries) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (graph->corpus.get() != corpus_.get()) throw std::invalid_argument("graph belongs to another Corpus");
    if (entries.empty()) entries = graph->entries;
    entries = corpus_->canonical(std::move(entries));
    for (Id id : entries) if (graph->local[id] < 0) throw std::invalid_argument("entry outside graph");
    if (!graph->members.empty() && entries.empty()) entries.push_back(graph->members.front());
    const auto count = graph->members.size();
    graphs_.push_back({std::move(graph), std::move(entries),
                       std::vector<std::uint32_t>(count, 0), std::vector<std::uint32_t>(count, 0)});
    return graphs_.size() - 1;
  }
  void set_engineering_mode(unsigned mode) {
    if (mode > 7) throw std::invalid_argument("engineering mode must be 0..7");
    std::lock_guard<std::mutex> lock(mutex_);
    engineering_mode_ = mode;
  }
  void pack_single(std::size_t graph_id, bool copy_vectors) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (graph_id >= graphs_.size()) throw std::invalid_argument("invalid graph ID");
    auto& stored = graphs_[graph_id];
    const auto& graph = *stored.graph;
    if (graph.members.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("packed graph exceeds local ID capacity");
    stored.row_ptr.assign(graph.members.size() + 1, 0);
    for (std::size_t r = 0; r < graph.rows.size(); ++r)
      stored.row_ptr[r + 1] = stored.row_ptr[r] + graph.rows[r].size();
    stored.local_neighbors.resize(stored.row_ptr.back());
    stored.local_distances.resize(graph.members.size());
    stored.local_distance_epoch.assign(graph.members.size(), 0);
    if (copy_vectors) stored.packed_vectors.resize(graph.members.size() * corpus_->dim);
    else stored.packed_vectors.clear();
    parallel_rows(graph.members.size(), graph.workers, [&](std::size_t r) {
      for (std::size_t j = 0; j < graph.rows[r].size(); ++j)
        stored.local_neighbors[stored.row_ptr[r] + j] =
            static_cast<std::uint32_t>(graph.local[graph.rows[r][j]]);
      if (copy_vectors) {
        const float* source = corpus_->vectors.data() +
            static_cast<std::size_t>(graph.members[r]) * corpus_->dim;
        std::copy(source, source + corpus_->dim, stored.packed_vectors.data() + r * corpus_->dim);
      }
    });
  }
  void set_bridges(const std::vector<Id>& sources, const std::vector<std::size_t>& gids,
                   const std::vector<Id>& targets) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sources.size() != gids.size() || sources.size() != targets.size())
      throw std::invalid_argument("bridge arrays must have the same length");
    std::vector<std::vector<std::pair<std::size_t, Id>>> replacement(corpus_->n);
    for (std::size_t i = 0; i < sources.size(); ++i) {
      corpus_->check_id(sources[i]); corpus_->check_id(targets[i]);
      if (gids[i] >= graphs_.size() || graphs_[gids[i]].graph->local[targets[i]] < 0)
        throw std::invalid_argument("bridge target is not a member of its destination graph");
      replacement[sources[i]].emplace_back(gids[i], targets[i]);
    }
    for (auto& row : replacement) {
      std::sort(row.begin(), row.end()); row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    bridges_ = std::move(replacement);
  }

  SingleResult search_single_local(const float* query, std::size_t graph_id,
                                  std::size_t k, std::size_t beam,
                                  const std::string& entry_mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& stored = graphs_.at(graph_id);
    const auto& graph = *stored.graph;
    if (!k || !beam || stored.row_ptr.empty())
      throw std::invalid_argument("local search needs k, beam and local adjacency");
    if (entry_mode != "nearest2" && entry_mode != "all")
      throw std::invalid_argument("unsupported entry mode");
    SingleResult out;
    out.effective_beam = std::max(k, beam);
    out.entry_candidates = stored.entries.size();
    next_epoch();
    local_frontier_.clear(); local_best_.clear();
    local_best_.reserve(std::min(out.effective_beam, graph.members.size()));
    // Global ID remains the tie-break even if local rows are not sorted.
    auto less = [&](const LocalHit& a, const LocalHit& b) {
      if (a.first != b.first) return a.first < b.first;
      return graph.members[a.second] < graph.members[b.second];
    };
    auto greater = [&](const LocalHit& a, const LocalHit& b) { return less(b, a); };
    auto vector = [&](std::uint32_t row) {
      return stored.packed_vectors.empty()
          ? corpus_->vectors.data() + static_cast<std::size_t>(graph.members[row]) * corpus_->dim
          : stored.packed_vectors.data() + static_cast<std::size_t>(row) * corpus_->dim;
    };
    auto distance = [&](std::uint32_t row) {
      if (stored.local_distance_epoch[row] != epoch_) {
        stored.local_distances[row] = distance_fn_(query, vector(row), corpus_->dim);
        stored.local_distance_epoch[row] = epoch_; ++out.evaluations;
      }
      return stored.local_distances[row];
    };
    auto admit = [&](std::uint32_t row) {
      const LocalHit hit{stored.local_distances[row], row};
      if (local_best_.size() == out.effective_beam && !less(hit, local_best_.front())) return;
      if (local_best_.size() == out.effective_beam) {
        std::pop_heap(local_best_.begin(), local_best_.end(), less);
        local_best_.back() = hit;
      } else local_best_.push_back(hit);
      std::push_heap(local_best_.begin(), local_best_.end(), less);
      local_frontier_.push_back(hit);
      std::push_heap(local_frontier_.begin(), local_frontier_.end(), greater);
      ++out.candidate_pushes;
      out.frontier_peak = std::max(out.frontier_peak, local_frontier_.size());
    };
    auto visit = [&](std::uint32_t row) {
      if (stored.discovered[row] == epoch_) return;
      stored.discovered[row] = epoch_; distance(row); admit(row);
    };
    if (entry_mode == "all") {
      for (Id id : stored.entries) visit(static_cast<std::uint32_t>(graph.local[id]));
      out.entry_starts = stored.entries.size(); out.entry_evaluations = out.evaluations;
    } else {
      std::array<LocalHit, 2> nearest;
      std::size_t count = 0;
      for (Id id : stored.entries) {
        const auto row = static_cast<std::uint32_t>(graph.local[id]);
        const LocalHit hit{distance(row), row};
        if (count < 2) {
          nearest[count++] = hit;
          if (count == 2 && less(nearest[1], nearest[0])) std::swap(nearest[0], nearest[1]);
        } else if (less(hit, nearest[1])) {
          nearest[1] = hit;
          if (less(nearest[1], nearest[0])) std::swap(nearest[0], nearest[1]);
        }
      }
      out.entry_starts = count; out.entry_evaluations = out.evaluations;
      for (std::size_t i = 0; i < count; ++i) visit(nearest[i].second);
    }
    while (!local_frontier_.empty()) {
      const auto current = local_frontier_.front();
      if (local_best_.size() == out.effective_beam && less(local_best_.front(), current)) {
        out.stop_reason = "beam_distance_bound"; break;
      }
      std::pop_heap(local_frontier_.begin(), local_frontier_.end(), greater);
      local_frontier_.pop_back(); ++out.expansions;
      const auto row = current.second;
      std::array<std::uint32_t, 4> pending;
      std::size_t count = 0;
      auto flush = [&]() {
        std::array<const float*, 4> pointers;
        std::array<std::uint32_t, 4> fresh;
        std::size_t fresh_count = 0;
        for (std::size_t i = 0; i < count; ++i) {
          if (stored.local_distance_epoch[pending[i]] != epoch_) {
            fresh[fresh_count] = pending[i]; pointers[fresh_count++] = vector(pending[i]);
          }
        }
#if FRANS_V11_AVX2_DISPATCH
        if (fresh_count == 4 && distance_fn_ == distance_f32_avx2) {
          std::array<float, 4> values;
          distances4_f32_avx2(query, pointers, corpus_->dim, values);
          for (std::size_t i = 0; i < 4; ++i) {
            stored.local_distances[fresh[i]] = values[i]; stored.local_distance_epoch[fresh[i]] = epoch_;
          }
          out.evaluations += 4; ++out.distance_batches;
        } else
#endif
        { for (std::size_t i = 0; i < fresh_count; ++i) distance(fresh[i]); }
        for (std::size_t i = 0; i < count; ++i) admit(pending[i]);
        count = 0;
      };
      for (std::size_t j = stored.row_ptr[row]; j < stored.row_ptr[row + 1]; ++j) {
        if ((single_optimization_ & 4) && j + 3 < stored.row_ptr[row + 1]) {
#if defined(__GNUC__) || defined(__clang__)
          const auto ahead = stored.local_neighbors[j + 3];
          if (stored.discovered[ahead] != epoch_) {
            __builtin_prefetch(vector(ahead), 0, 1);
            if (corpus_->dim > 16) __builtin_prefetch(vector(ahead) + 16, 0, 1);
          }
#endif
        }
        const auto next = stored.local_neighbors[j];
        if (!(single_optimization_ & 2)) { visit(next); continue; }
        if (stored.discovered[next] == epoch_) continue;
        stored.discovered[next] = epoch_;
        pending[count++] = next;
        if (count == 4) flush();
      }
      if (count) flush();
    }
    const auto count = std::min(k, local_best_.size());
    std::partial_sort(local_best_.begin(), local_best_.begin() + count, local_best_.end(), less);
    for (std::size_t i = 0; i < count; ++i) {
      out.ids.push_back(graph.members[local_best_[i].second]); out.distances.push_back(local_best_[i].first);
    }
    return out;
  }

  SingleResult search_single(const float* query, std::size_t graph_id,
                             std::size_t k, std::size_t beam,
                             const std::string& entry_mode,
                             std::size_t min_expansions = 128,
                             std::size_t stable_patience = 0) {
    if (single_optimization_) {
      if (stable_patience) throw std::invalid_argument("optimized path uses unchanged beam stopping only");
      return search_single_local(query, graph_id, k, beam, entry_mode);
    }
    if (!k || !beam) throw std::invalid_argument("k and beam must be positive");
    if (entry_mode != "all" && entry_mode != "nearest2")
      throw std::invalid_argument("entry_mode must be all or nearest2");
    std::lock_guard<std::mutex> lock(mutex_);
    if (graph_id >= graphs_.size()) throw std::invalid_argument("graph_id must be valid");
    auto& stored = graphs_[graph_id];
    const auto& graph = *stored.graph;
    SingleResult out;
    out.effective_beam = std::max(k, beam);
    out.entry_candidates = stored.entries.size();
    next_epoch();
    single_frontier_.clear(); single_best_.clear();
    std::vector<FloatRanked> stable_best;
    if (stable_patience) stable_best.reserve(k);
    std::size_t last_improvement = 0;
    single_best_.reserve(std::min(out.effective_beam, graph.members.size()));
    const bool packed = !stored.row_ptr.empty();
    auto distance = [&](std::size_t row) {
      const Id id = graph.members[row];
      if (packed) {
        if (stored.local_distance_epoch[row] != epoch_) {
          const float* vector = stored.packed_vectors.empty()
              ? corpus_->vectors.data() + static_cast<std::size_t>(id) * corpus_->dim
              : stored.packed_vectors.data() + row * corpus_->dim;
          stored.local_distances[row] = distance_fn_(query, vector, corpus_->dim);
          stored.local_distance_epoch[row] = epoch_;
          ++out.evaluations;
        }
        return stored.local_distances[row];
      }
      // nearest2 selection is inside this call and counted. Cache its eight
      // entry distances so reaching an unselected entry does not recompute it.
      if (distance_epoch_[id] != epoch_) {
        distances_[id] = distance_fn_(query,
            corpus_->vectors.data() + static_cast<std::size_t>(id) * corpus_->dim,
            corpus_->dim);
        distance_epoch_[id] = epoch_;
        ++out.evaluations;
      }
      return distances_[id];
    };
    auto visit = [&](std::size_t row) {
      const Id id = graph.members[row];
      if (stored.discovered[row] == epoch_) return;
      stored.discovered[row] = epoch_;
      const FloatRanked hit{distance(row), id};
      if (stable_patience && (stable_best.size() < k || hit < stable_best.front())) {
        if (stable_best.size() == k) {
          std::pop_heap(stable_best.begin(), stable_best.end());
          stable_best.back() = hit;
        } else stable_best.push_back(hit);
        std::push_heap(stable_best.begin(), stable_best.end());
        last_improvement = out.expansions;
      }
      // HNSW-style admission: an evaluated object is enqueued only if it
      // improves the full best beam (or the beam is not yet full).
      if (single_best_.size() == out.effective_beam && !(hit < single_best_.front())) return;
      if (single_best_.size() == out.effective_beam) {
        std::pop_heap(single_best_.begin(), single_best_.end());
        single_best_.back() = hit;
      } else single_best_.push_back(hit);
      std::push_heap(single_best_.begin(), single_best_.end());
      single_frontier_.push_back(hit);
      std::push_heap(single_frontier_.begin(), single_frontier_.end(), std::greater<FloatRanked>());
      ++out.candidate_pushes;
      out.frontier_peak = std::max(out.frontier_peak, single_frontier_.size());
    };
    if (entry_mode == "all") {
      for (Id id : stored.entries) visit(static_cast<std::size_t>(graph.local[id]));
      out.entry_starts = stored.entries.size();
      out.entry_evaluations = out.evaluations;
    } else {
      // A fixed two-element selection, not a second candidate frontier.
      // Existing offline entries are normally eight; no new entry index.
      std::array<FloatRanked, 2> nearest;
      std::size_t count = 0;
      for (Id id : stored.entries) {
        const FloatRanked hit{distance(static_cast<std::size_t>(graph.local[id])), id};
        if (count < nearest.size()) {
          nearest[count++] = hit;
          if (count == 2 && nearest[1] < nearest[0]) std::swap(nearest[0], nearest[1]);
        } else if (hit < nearest[1]) {
          nearest[1] = hit;
          if (nearest[1] < nearest[0]) std::swap(nearest[0], nearest[1]);
        }
      }
      out.entry_evaluations = out.evaluations;
      out.entry_starts = count;
      for (std::size_t i = 0; i < count; ++i)
        visit(static_cast<std::size_t>(graph.local[nearest[i].second]));
    }
    while (!single_frontier_.empty()) {
      const auto current = single_frontier_.front();
      if (stable_patience && out.expansions >= min_expansions &&
          out.expansions - last_improvement >= stable_patience &&
          stable_best.size() == k && current > stable_best.front()) {
        out.stop_reason = "topk_stable";
        break;
      }
      // This is an approximate graph-search stopping rule, not a proof that
      // an unvisited object cannot improve the result.
      if (single_best_.size() == out.effective_beam && current > single_best_.front()) {
        out.stop_reason = "beam_distance_bound";
        break;
      }
      std::pop_heap(single_frontier_.begin(), single_frontier_.end(), std::greater<FloatRanked>());
      single_frontier_.pop_back();
      ++out.expansions;
      const auto row = static_cast<std::size_t>(graph.local[current.second]);
      if (packed) {
        for (std::size_t j = stored.row_ptr[row]; j < stored.row_ptr[row + 1]; ++j) {
          if ((engineering_mode_ & 2) && j + 3 < stored.row_ptr[row + 1])
            prefetch_vector(graph.members[stored.local_neighbors[j + 3]]);
          visit(stored.local_neighbors[j]);
        }
      } else {
        for (Id id : graph.rows[row]) visit(static_cast<std::size_t>(graph.local[id]));
      }
    }
    const auto count = std::min(k, single_best_.size());
    if (engineering_mode_ & 1)
      std::partial_sort(single_best_.begin(), single_best_.begin() + count, single_best_.end());
    else std::sort(single_best_.begin(), single_best_.end());
    out.ids.reserve(count); out.distances.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      out.distances.push_back(single_best_[i].first);
      out.ids.push_back(single_best_[i].second);
    }
    return out;
  }

  JointResult search(const float* query, const std::vector<std::size_t>& graph_ids,
                     const double* lower, const double* upper, std::size_t boxes,
                     const std::vector<bool>& fully_inside, std::size_t k,
                     std::size_t beam, std::size_t max_expansions,
                     bool use_bridges, bool trace, const std::string& entry_mode) {
    if (!k || !beam) throw std::invalid_argument("k and beam must be positive");
    if (entry_mode != "all" && entry_mode != "nearest2")
      throw std::invalid_argument("entry_mode must be all or nearest2");
    if (graph_ids.size() != fully_inside.size())
      throw std::invalid_argument("fully_inside must align with graph_ids");
    if (beam > std::numeric_limits<std::size_t>::max() / 8)
      throw std::invalid_argument("beam too large");
    std::lock_guard<std::mutex> lock(mutex_);
    JointResult out; out.distance_kernel = distance_kernel_; out.effective_beam = std::max(k, beam);
    const bool compact_joint = (engineering_mode_ & 4) &&
        std::all_of(fully_inside.begin(), fully_inside.end(), [](bool v) { return v; });
    std::vector<Id> active_slot(graphs_.size(), -1);
    for (std::size_t slot = 0; slot < graph_ids.size(); ++slot) {
      auto gid = graph_ids[slot];
      if (gid >= graphs_.size() || active_slot[gid] >= 0)
        throw std::invalid_argument("graph_ids must be valid and unique");
      active_slot[gid] = static_cast<Id>(slot);
    }
    const auto count = graph_ids.size();
    out.effective_cap = max_expansions ? std::max(max_expansions, count * 4)
        : std::max({std::size_t(64), beam * 8, count * 4});
    out.per_graph_expansions.assign(count, 0); out.last_expanded.assign(count, -1);
    if (trace) out.graph_hits.resize(count);
    if (!boxes || !count) return out;
    next_epoch();
    std::priority_queue<SearchState, std::vector<SearchState>, std::greater<SearchState>> frontier;
    using LocalState = std::tuple<float, Id, std::uint32_t>;
    using LocalQueue = std::priority_queue<LocalState, std::vector<LocalState>, std::greater<LocalState>>;
    std::vector<LocalQueue> per_graph_frontier(count);
    std::vector<FloatRanked> legal_beam;
    legal_beam.reserve(std::min(out.effective_beam, corpus_->n));
    auto keep_best = [](std::vector<FloatRanked>& heap, FloatRanked hit, std::size_t limit) {
      if (heap.size() < limit) { heap.push_back(hit); std::push_heap(heap.begin(), heap.end()); }
      else if (hit < heap.front()) {
        std::pop_heap(heap.begin(), heap.end()); heap.back() = hit; std::push_heap(heap.begin(), heap.end());
      }
    };
    auto allowed = [&](Id id, std::size_t slot) {
      if (fully_inside[slot]) return true;
      if (predicate_epoch_[id] == epoch_) return predicate_value_[id] != 0;
      ++out.predicate_checks;
      bool valid = false;
      const double* values = attrs_.data() + static_cast<std::size_t>(id) * attributes_;
      for (std::size_t box = 0; box < boxes && !valid; ++box) {
        valid = true;
        for (std::size_t j = 0; j < attributes_; ++j)
          if (values[j] < lower[box * attributes_ + j] || values[j] > upper[box * attributes_ + j]) {
            valid = false; break;
          }
      }
      predicate_epoch_[id] = epoch_; predicate_value_[id] = valid;
      return valid;
    };
    auto vector_at = [&](const StoredGraph& stored, std::size_t row, Id id) {
      return joint_optimization_ && !stored.packed_vectors.empty()
          ? stored.packed_vectors.data() + row * corpus_->dim
          : corpus_->vectors.data() + static_cast<std::size_t>(id) * corpus_->dim;
    };
    auto visit = [&](std::size_t slot, Id id,
                     std::size_t known_row = std::numeric_limits<std::size_t>::max()) {
      auto& stored = graphs_[graph_ids[slot]];
      const auto row = known_row == std::numeric_limits<std::size_t>::max()
          ? static_cast<std::size_t>(stored.graph->local[id]) : known_row;
      if (stored.discovered[row] == epoch_) return false;
      stored.discovered[row] = epoch_;
      if (distance_epoch_[id] != epoch_) {
        distances_[id] = distance_fn_(query, vector_at(stored, row, id), corpus_->dim);
        distance_epoch_[id] = epoch_; ++out.evaluations;
      }
      const FloatRanked hit{distances_[id], id};
      // Keep the original first four expansions of every graph. Afterwards,
      // an all-legal candidate worse than the current full beam can never
      // pass the existing global stop bound, which only tightens over time.
      const bool mandatory = out.per_graph_expansions[slot] < 4;
      const bool competitive = legal_beam.size() < out.effective_beam || !(legal_beam.front() < hit);
      if (!compact_joint || mandatory || competitive) {
        frontier.emplace(hit.first, slot, id, static_cast<std::uint32_t>(row));
        ++out.candidate_pushes;
        out.frontier_peak = std::max(out.frontier_peak, frontier.size());
      }
      if (!compact_joint || mandatory) per_graph_frontier[slot].emplace(hit.first, id, static_cast<std::uint32_t>(row));
      if (allowed(id, slot)) {
        if (answer_epoch_[id] != epoch_) {
          answer_epoch_[id] = epoch_; keep_best(legal_beam, hit, out.effective_beam);
        }
        if (trace) keep_best(out.graph_hits[slot], hit, k);
      }
      return true;
    };
    auto expand = [&](std::size_t slot, Id id, std::uint32_t cached_row) {
      auto& stored = graphs_[graph_ids[slot]];
      const auto row = joint_optimization_ ? static_cast<std::size_t>(cached_row)
          : static_cast<std::size_t>(stored.graph->local[id]);
      stored.expanded[row] = epoch_;
      if (!out.per_graph_expansions[slot]++) {
        ++out.graphs_searched;
        if (trace) out.graph_order.push_back(slot);
      }
      ++out.expansions; out.last_expanded[slot] = id;
      if (trace) out.expanded_states.emplace_back(graph_ids[slot], id);
      if ((compact_joint || joint_optimization_) && !stored.row_ptr.empty()) {
        std::array<std::uint32_t, 4> pending;
        std::size_t pending_count = 0;
        auto flush = [&]() {
          std::array<const float*, 4> pointers;
          std::array<Id, 4> fresh;
          std::size_t fresh_count = 0;
          for (std::size_t i = 0; i < pending_count; ++i) {
            const auto local = pending[i]; const Id target = stored.graph->members[local];
            if (distance_epoch_[target] == epoch_) continue;
            bool duplicate = false;
            for (std::size_t j = 0; j < fresh_count; ++j) duplicate |= fresh[j] == target;
            if (duplicate) continue;
            fresh[fresh_count] = target;
            pointers[fresh_count++] = vector_at(stored, local, target);
          }
#if FRANS_V11_AVX2_DISPATCH
          if (fresh_count == 4 && distance_fn_ == distance_f32_avx2) {
            std::array<float, 4> values;
            distances4_f32_avx2(query, pointers, corpus_->dim, values);
            for (std::size_t i = 0; i < 4; ++i) {
              distances_[fresh[i]] = values[i]; distance_epoch_[fresh[i]] = epoch_;
            }
            out.evaluations += 4;
          } else
#endif
          {
            for (std::size_t i = 0; i < fresh_count; ++i) {
              distances_[fresh[i]] = distance_fn_(query, pointers[i], corpus_->dim);
              distance_epoch_[fresh[i]] = epoch_; ++out.evaluations;
            }
          }
          for (std::size_t i = 0; i < pending_count; ++i)
            visit(slot, stored.graph->members[pending[i]], pending[i]);
          pending_count = 0;
        };
        for (std::size_t j = stored.row_ptr[row]; j < stored.row_ptr[row + 1]; ++j) {
          if (j + 3 < stored.row_ptr[row + 1]) {
            const auto ahead = stored.local_neighbors[j + 3];
#if defined(__GNUC__) || defined(__clang__)
            if (joint_optimization_ & 4)
              __builtin_prefetch(vector_at(stored, ahead, stored.graph->members[ahead]), 0, 1);
            else
#endif
            if (engineering_mode_ & 2) prefetch_vector(stored.graph->members[ahead]);
          }
          const auto local = stored.local_neighbors[j];
          if (!(joint_optimization_ & 2)) { visit(slot, stored.graph->members[local], local); continue; }
          if (stored.discovered[local] == epoch_) continue;
          pending[pending_count++] = local;
          if (pending_count == 4) flush();
        }
        if (pending_count) flush();
      } else {
        for (Id target : stored.graph->rows[row]) visit(slot, target);
      }
      if (use_bridges) for (const auto& bridge : bridges_[id]) {
        const Id destination = active_slot[bridge.first];
        if (destination >= 0 && visit(static_cast<std::size_t>(destination), bridge.second)) ++out.bridge_starts;
      }
    };
    // Never rely on a learned bridge to enter a necessary graph.
    for (std::size_t slot = 0; slot < count; ++slot) {
      const auto& entries = graphs_[graph_ids[slot]].entries;
      out.entry_candidates += entries.size();
      if (entry_mode == "all") {
        for (Id id : entries) visit(slot, id);
        out.entry_starts += entries.size();
      } else {
        // Same fixed-size nearest-two selection as V14. Entry work stays
        // inside the timed query and the shared distance cache.
        std::array<FloatRanked, 2> nearest;
        std::size_t selected = 0;
        for (Id id : entries) {
          if (distance_epoch_[id] != epoch_) {
            distances_[id] = distance_fn_(query,
                corpus_->vectors.data() + static_cast<std::size_t>(id) * corpus_->dim,
                corpus_->dim);
            distance_epoch_[id] = epoch_; ++out.evaluations;
          }
          const FloatRanked hit{distances_[id], id};
          if (selected < nearest.size()) {
            nearest[selected++] = hit;
            if (selected == 2 && nearest[1] < nearest[0]) std::swap(nearest[0], nearest[1]);
          } else if (hit < nearest[1]) {
            nearest[1] = hit;
            if (nearest[1] < nearest[0]) std::swap(nearest[0], nearest[1]);
          }
        }
        out.entry_starts += selected;
        for (std::size_t i = 0; i < selected; ++i) visit(slot, nearest[i].second);
      }
    }
    out.entry_evaluations = out.evaluations;
    // Mandatory small allocation prevents the global queue starving a distant
    // disjoint box. All discoveries already update the same legal beam/cache.
    for (std::size_t slot = 0; slot < count; ++slot) {
      auto& local_queue = per_graph_frontier[slot];
      const auto& stored = graphs_[graph_ids[slot]];
      while (out.per_graph_expansions[slot] < 4 && !local_queue.empty()) {
        const auto item = local_queue.top(); local_queue.pop();
        const Id id = std::get<1>(item);
        const auto row = joint_optimization_ ? std::get<2>(item)
            : static_cast<std::uint32_t>(stored.graph->local[id]);
        if (stored.expanded[row] == epoch_) continue;
        expand(slot, id, row);
      }
    }
    out.stop_reason = "frontier_exhausted";
    while (!frontier.empty()) {
      const auto state = frontier.top();
      const auto slot = std::get<1>(state); const Id id = std::get<2>(state);
      const auto& stored = graphs_[graph_ids[slot]];
      const auto row = joint_optimization_ ? std::get<3>(state)
          : static_cast<std::uint32_t>(stored.graph->local[id]);
      if (stored.expanded[row] == epoch_) { frontier.pop(); continue; }
      if (out.expansions >= out.effective_cap) { out.stop_reason = "max_expansions"; break; }
      // Approximate graph-search stopping rule, NOT an exact-distance proof.
      // Unexpanded discovered candidates ARE eligible returned answers.
      if (legal_beam.size() == out.effective_beam &&
          FloatRanked{std::get<0>(state), id} > legal_beam.front()) {
        out.stop_reason = "beam_distance_bound"; break;
      }
      frontier.pop(); expand(slot, id, row);
    }
    const auto result_count = std::min(k, legal_beam.size());
    if (engineering_mode_ & 1)
      std::partial_sort(legal_beam.begin(), legal_beam.begin() + result_count, legal_beam.end());
    else std::sort(legal_beam.begin(), legal_beam.end());
    for (std::size_t i = 0; i < result_count; ++i) {
      out.distances.push_back(legal_beam[i].first); out.ids.push_back(legal_beam[i].second);
    }
    if (trace) for (auto& hits : out.graph_hits) std::sort(hits.begin(), hits.end());
    return out;
  }
};
}  // namespace fv11

PYBIND11_MODULE(_frans_v11, m) {
  using namespace fv11;
  m.doc() = "FRANS V11 independent native module with lightweight all-legal single-graph search";
  m.def("supports_avx2_batch4", &supports_avx2_batch4);
  m.attr("resource_persistence_version") = 15;
  m.attr("fullscale_revision") = "frans_fullscale_v1";
  m.attr("fullscale_lookup_version") = 1;
  m.def("reset_construction_parallel_stats", []() {
    row_calls = 0; row_items = 0; row_thread_launches = 0; row_max_workers = 0;
  });
  m.def("construction_parallel_stats", []() {
    py::dict d;
    d["scope"] = "native_parallel_rows_only_excludes_unchanged_leaf_builder";
    d["row_calls"] = row_calls.load(); d["row_items"] = row_items.load();
    d["thread_launches"] = row_thread_launches.load(); d["max_workers"] = row_max_workers.load();
    return d;
  });
  py::class_<Corpus, std::shared_ptr<Corpus>>(m, "Corpus")
    .def(py::init<py::array_t<float, py::array::c_style>>(), py::arg("vectors").noconvert())
    .def("build", &Corpus::build, py::arg("members"), py::arg("candidate_size") = 64,
         py::arg("alpha") = 1.2, py::arg("workers") = 16, py::call_guard<py::gil_scoped_release>())
    .def("alpha_prune", &Corpus::alpha_prune, py::arg("source"), py::arg("candidate_ids"),
         py::arg("alpha") = 1.2, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("size", [](const Corpus& c) { return c.n; })
    .def_property_readonly("dimension", [](const Corpus& c) { return c.dim; });
  py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
    .def("save", &Graph::save, py::arg("path"), py::call_guard<py::gil_scoped_release>())
    .def_static("load", &Graph::load, py::arg("corpus"), py::arg("path"),
                py::call_guard<py::gil_scoped_release>())
    .def("merge", &Graph::merge, py::arg("other"), py::call_guard<py::gil_scoped_release>())
    .def("extract_repair", &Graph::extract_repair, py::arg("new_members"), py::call_guard<py::gil_scoped_release>())
    .def("expand", &Graph::expand, py::arg("new_members"), py::call_guard<py::gil_scoped_release>())
    .def("pruned", &Graph::pruned, py::arg("max_degree"), py::call_guard<py::gil_scoped_release>())
    .def("refined_twohop", &Graph::refined_twohop, py::arg("candidate_cap") = 256,
         py::arg("max_degree") = 32, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("members", [](const Graph& g) { return g.members; })
    .def_property_readonly("entries", [](const Graph& g) { return g.entries; })
    .def_property_readonly("edge_count", &Graph::edge_count)
    .def("neighbors", [](const Graph& g, Id source) {
      g.corpus->check_id(source);
      if (g.local[source] < 0) throw std::invalid_argument("source outside graph");
      return g.rows[g.local[source]];
    }, py::arg("source"))
    .def_property_readonly("metadata", [](const Graph& g) {
      py::dict d;
      d["revision"] = "frans_v11_practical_v1"; d["operation"] = g.operation;
      d["complete_alpha_coverage_claim"] = false;
      d["candidate_policy"] = "base_projection_symmetric_propagation;merge_repair_expand_bounded_projection";
      d["numeric"] = "float32_vectors_float64_sequential_squared_l2_alpha_squared";
      d["search_distance_kernel_default"] = "scalar";
      d["avx2_batch4_available"] = supports_avx2_batch4();
      d["base_degree_cap"] = 24; d["merge_repair_expand_degree_cap"] = py::none();
      d["candidate_size"] = g.candidate_size; d["alpha"] = g.alpha; d["workers"] = g.workers;
      d["member_count"] = g.members.size(); d["edge_count"] = g.edge_count();
      d["local_lookup_mode"] = g.local.mode();
      d["local_lookup_storage_bytes"] = g.local.storage_bytes();
      d["dirty_rows"] = g.dirty_rows; d["inherited_edges"] = g.inherited_edges; d["added_edges"] = g.added_edges;
      d["degree_cap"] = g.degree_cap ? py::cast(g.degree_cap) : py::none();
      d["edges_before_pruning"] = g.edges_before_pruning;
      d["edges_after_pruning"] = g.edge_count(); d["removed_edges"] = g.removed_edges;
      d["pruning_preserves_connectivity_claim"] = false;
      std::size_t max_degree = 0; for (const auto& row : g.rows) max_degree = std::max(max_degree, row.size());
      d["max_degree"] = max_degree;
      return d;
    })
    .def("search", [](Graph& g, py::array_t<float, py::array::c_style> query,
                       std::size_t k, std::size_t budget, std::vector<Id> entries,
                       py::object allowed, bool trace, const std::string& distance_kernel) {
      if (distance_kernel != "scalar" && distance_kernel != "avx2_batch4")
        throw std::invalid_argument("distance_kernel must be scalar or avx2_batch4");
      auto q = query.request();
      if (q.ndim != 1 || static_cast<std::size_t>(q.shape[0]) != g.corpus->dim)
        throw std::invalid_argument("q must be float32 [D]");
      const auto* values = static_cast<const float*>(q.ptr);
      for (std::size_t j = 0; j < g.corpus->dim; ++j)
        if (!std::isfinite(values[j])) throw std::invalid_argument("q must be finite");
      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask;
      const std::uint8_t* mask_data = nullptr;
      if (!allowed.is_none()) {
        mask = allowed.cast<decltype(mask)>(); auto a = mask.request();
        if (a.ndim != 1 || static_cast<std::size_t>(a.shape[0]) != g.corpus->n)
          throw std::invalid_argument("allowed_mask must have one entry per Corpus object");
        mask_data = static_cast<const std::uint8_t*>(a.ptr);
      }
      SearchResult result;
      { py::gil_scoped_release release;
        result = g.search(values, k, budget, entries, mask_data, trace, distance_kernel == "avx2_batch4"); }
      py::dict d, counters;
      d["ids"] = result.ids; d["distances"] = result.distances;
      d["expanded_ids"] = result.expanded; d["last_expanded"] = result.last;
      counters["expansions"] = result.expansions; counters["distance_evaluations"] = result.evaluations;
      counters["candidate_pushes"] = result.pushes; counters["budget"] = budget;
      counters["distance_batches"] = result.distance_batches;
      counters["distance_kernel"] = result.avx2_batch4 ? "avx2_batch4" : "scalar";
      counters["requested_distance_kernel"] = distance_kernel;
      counters["frontier_exhausted"] = result.frontier_exhausted; d["counters"] = counters;
      return d;
    }, py::arg("q").noconvert(), py::arg("k") = 10, py::arg("budget") = 128,
       py::arg("entry_ids") = std::vector<Id>{}, py::arg("allowed_mask") = py::none(),
       py::arg("trace") = false, py::arg("distance_kernel") = "scalar");
  py::class_<SearchIndex>(m, "SearchIndex")
    .def(py::init<std::shared_ptr<Corpus>, py::array_t<double, py::array::c_style>>(),
         py::arg("corpus"), py::arg("attrs").noconvert())
    .def("add_graph", &SearchIndex::add_graph, py::arg("graph"), py::arg("entries") = std::vector<Id>{})
    .def("set_engineering_mode", &SearchIndex::set_engineering_mode, py::arg("mode"))
    .def("set_joint_optimization", &SearchIndex::set_joint_optimization, py::arg("mode"))
    .def("set_single_optimization", &SearchIndex::set_single_optimization, py::arg("mode"))
    .def("set_entries", &SearchIndex::set_entries, py::arg("graph_id"), py::arg("entries"))
    .def("pack_single", &SearchIndex::pack_single, py::arg("graph_id"), py::arg("copy_vectors") = true,
         py::call_guard<py::gil_scoped_release>())
    .def("set_bridges", &SearchIndex::set_bridges, py::arg("sources"), py::arg("dest_gids"), py::arg("targets"))
    .def_property_readonly("distance_kernel", &SearchIndex::distance_kernel)
    .def("search_single", [](SearchIndex& index, py::array_t<float, py::array::c_style> query,
                              std::size_t graph_id, std::size_t k, std::size_t beam,
                              const std::string& entry_mode, std::size_t min_expansions,
                              std::size_t stable_patience) {
      auto q = query.request();
      if (q.ndim != 1 || static_cast<std::size_t>(q.shape[0]) != index.dimension())
        throw std::invalid_argument("query must be contiguous float32 [D]");
      const auto* values = static_cast<const float*>(q.ptr);
      for (std::size_t j = 0; j < index.dimension(); ++j)
        if (!std::isfinite(values[j])) throw std::invalid_argument("query must be finite");
      SingleResult result;
      { py::gil_scoped_release release;
        result = index.search_single(values, graph_id, k, beam, entry_mode,
                                     min_expansions, stable_patience); }
      py::dict output, counters;
      output["ids"] = result.ids; output["distances"] = result.distances;
      counters["expansions"] = result.expansions;
      counters["distance_evaluations"] = result.evaluations;
      counters["distance_batches"] = result.distance_batches;
      counters["single_optimization"] = index.single_optimization();
      counters["predicate_checks"] = 0;
      counters["graphs_searched"] = 1;
      counters["bridge_starts"] = 0;
      counters["candidate_pushes"] = result.candidate_pushes;
      counters["frontier_peak"] = result.frontier_peak;
      counters["requested_beam"] = beam; counters["effective_beam"] = result.effective_beam;
      counters["per_graph_expansions"] = std::vector<std::size_t>{result.expansions};
      counters["stop_reason"] = result.stop_reason;
      counters["distance_kernel"] = index.distance_kernel();
      counters["entry_mode"] = entry_mode;
      counters["entry_candidates"] = result.entry_candidates;
      counters["entry_starts"] = result.entry_starts;
      counters["entry_distance_evaluations"] = result.entry_evaluations;
      counters["approximate_stopping"] = true;
      counters["search_path"] = "single_graph_all_legal";
      counters["min_expansions"] = min_expansions;
      counters["stable_patience"] = stable_patience;
      output["counters"] = counters;
      return output;
    }, py::arg("query").noconvert(), py::arg("graph_id"), py::arg("k") = 10,
       py::arg("beam") = 128, py::arg("entry_mode") = "nearest2",
       py::arg("min_expansions") = 128, py::arg("stable_patience") = 0)
    .def("search", [](SearchIndex& index, py::array_t<float, py::array::c_style> query,
                       std::vector<std::size_t> graph_ids,
                       py::array_t<double, py::array::c_style> lower,
                       py::array_t<double, py::array::c_style> upper,
                       std::vector<bool> fully_inside, std::size_t k, std::size_t beam,
                       std::size_t max_expansions, bool use_bridges, bool trace,
                       const std::string& entry_mode) {
      auto q = query.request(), lo = lower.request(), hi = upper.request();
      if (q.ndim != 1 || static_cast<std::size_t>(q.shape[0]) != index.dimension())
        throw std::invalid_argument("q must be contiguous float32 [D]");
      if (lo.ndim != 2 || hi.ndim != 2 || lo.shape != hi.shape ||
          static_cast<std::size_t>(lo.shape[1]) != index.attribute_count())
        throw std::invalid_argument("lower/upper must be contiguous float64 [B,M] with matching shape");
      const auto* qvalues = static_cast<const float*>(q.ptr);
      const auto* low = static_cast<const double*>(lo.ptr);
      const auto* high = static_cast<const double*>(hi.ptr);
      for (std::size_t j = 0; j < index.dimension(); ++j)
        if (!std::isfinite(qvalues[j])) throw std::invalid_argument("q must be finite");
      for (py::ssize_t i = 0; i < lo.size; ++i)
        if (std::isnan(low[i]) || std::isnan(high[i]) || low[i] > high[i])
          throw std::invalid_argument("range bounds must be non-NaN and lower<=upper");
      JointResult result;
      { py::gil_scoped_release release;
        result = index.search(qvalues, graph_ids, low, high, static_cast<std::size_t>(lo.shape[0]),
                              fully_inside, k, beam, max_expansions, use_bridges, trace, entry_mode); }
      py::dict output, counters;
      output["ids"] = result.ids; output["distances"] = result.distances;
      counters["expansions"] = result.expansions;
      counters["distance_evaluations"] = result.evaluations;
      counters["predicate_checks"] = result.predicate_checks;
      counters["graphs_searched"] = result.graphs_searched;
      counters["bridge_starts"] = result.bridge_starts;
      counters["candidate_pushes"] = result.candidate_pushes;
      counters["frontier_peak"] = result.frontier_peak;
      counters["requested_beam"] = beam; counters["effective_beam"] = result.effective_beam;
      counters["requested_max_expansions"] = max_expansions;
      counters["effective_max_expansions"] = result.effective_cap;
      counters["per_graph_expansions"] = result.per_graph_expansions;
      counters["stop_reason"] = result.stop_reason;
      counters["distance_kernel"] = result.distance_kernel;
      counters["entry_mode"] = entry_mode;
      counters["entry_candidates"] = result.entry_candidates;
      counters["entry_starts"] = result.entry_starts;
      counters["entry_distance_evaluations"] = result.entry_evaluations;
      counters["approximate_stopping"] = true;
      output["counters"] = counters;
      if (trace) {
        py::list records;
        for (auto slot : result.graph_order) {
          py::dict record; std::vector<Id> ids;
          for (const auto& hit : result.graph_hits[slot]) ids.push_back(hit.second);
          record["graph_id"] = graph_ids[slot]; record["last_expanded"] = result.last_expanded[slot];
          record["ids"] = ids; record["expansions"] = result.per_graph_expansions[slot];
          records.append(record);
        }
        output["graph_records"] = records; output["expanded_states"] = result.expanded_states;
      }
      return output;
    }, py::arg("q").noconvert(), py::arg("graph_ids"), py::arg("lower").noconvert(),
       py::arg("upper").noconvert(), py::arg("fully_inside"), py::arg("k") = 10,
       py::arg("beam") = 128, py::arg("max_expansions") = 0,
       py::arg("use_bridges") = true, py::arg("trace") = false,
       py::arg("entry_mode") = "all");
}

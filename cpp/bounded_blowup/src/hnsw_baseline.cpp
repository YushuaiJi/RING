// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.

#include "bounded_blowup/hnsw_baseline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

namespace bounded_blowup {
namespace {

constexpr std::uint64_t kRngSalt = 0x9e3779b97f4a7c15ULL;
constexpr char kMagic[8] = {'R', 'F', 'B', 'B', 'H', 'N', 'S', 'W'};
constexpr std::uint32_t kFormatVersion = 1U;

bool better(const StandardHnswIndex::Candidate& left,
            const StandardHnswIndex::Candidate& right) {
  if (left.distance != right.distance) return left.distance < right.distance;
  return left.id < right.id;
}

struct MinCandidate final {
  bool operator()(const StandardHnswIndex::Candidate& left,
                  const StandardHnswIndex::Candidate& right) const {
    return better(right, left);
  }
};

struct MaxCandidate final {
  bool operator()(const StandardHnswIndex::Candidate& left,
                  const StandardHnswIndex::Candidate& right) const {
    return better(left, right);
  }
};

// splitmix64 keeps level generation deterministic without depending on the
// implementation-defined state/layout of std::mt19937.
std::uint64_t splitmix64(std::uint64_t& state) noexcept {
  state += kRngSalt;
  std::uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

template <typename T>
void write_scalar(std::ofstream& output, T value, const char* label) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!output) throw std::runtime_error(std::string("failed to write HNSW ") + label);
}

template <typename T>
T read_scalar(std::ifstream& input, const char* label) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error(std::string("failed to read HNSW ") + label);
  return value;
}

}  // namespace

StandardHnswIndex::StandardHnswIndex(std::size_t count, std::size_t dimension,
                                     const HnswBuildOptions& options)
    : count_(count),
      dimension_(dimension),
      m_(options.m),
      ef_construction_(options.ef_construction),
      seed_(options.seed),
      rng_state_(options.seed ^ kRngSalt),
      vectors_(count * dimension),
      nodes_(count),
      visited_stamp_(count, 0U) {
  if (count_ == 0U) throw std::invalid_argument("HNSW corpus must not be empty");
  if (dimension_ == 0U) throw std::invalid_argument("HNSW dimension must be positive");
  if (m_ < 2U) throw std::invalid_argument("HNSW M must be at least 2");
  if (ef_construction_ < m_) {
    throw std::invalid_argument("HNSW ef_construction must be at least M");
  }
  if (count_ > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("HNSW object count exceeds uint32 id range");
  }
  if (count_ > std::numeric_limits<std::size_t>::max() / dimension_) {
    throw std::invalid_argument("HNSW vector storage size overflows size_t");
  }
}

std::shared_ptr<StandardHnswIndex> StandardHnswIndex::build(
    const float* vectors, std::size_t count, std::size_t dimension,
    const HnswBuildOptions& options) {
  if (vectors == nullptr) throw std::invalid_argument("HNSW vectors are null");
  auto index = std::shared_ptr<StandardHnswIndex>(
      new StandardHnswIndex(count, dimension, options));
  std::copy(vectors, vectors + index->vectors_.size(), index->vectors_.begin());
  for (const float value : index->vectors_) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("HNSW vectors must contain finite float32 values");
    }
  }
  for (std::uint32_t id = 0U; id < static_cast<std::uint32_t>(count); ++id) {
    index->insert(id);
  }
  return index;
}

float StandardHnswIndex::distance(const float* left,
                                  const float* right) const noexcept {
  float result = 0.0F;
  for (std::size_t dimension = 0U; dimension < dimension_; ++dimension) {
    const float difference = left[dimension] - right[dimension];
    result += difference * difference;
  }
  return result;
}

float StandardHnswIndex::distance_to_id(const float* query,
                                        std::uint32_t id) const noexcept {
  return distance(query, vectors_.data() +
                           static_cast<std::size_t>(id) * dimension_);
}

float StandardHnswIndex::distance_between(std::uint32_t left,
                                           std::uint32_t right) const noexcept {
  return distance(vectors_.data() + static_cast<std::size_t>(left) * dimension_,
                  vectors_.data() + static_cast<std::size_t>(right) * dimension_);
}

std::uint32_t StandardHnswIndex::random_level() {
  // HNSW samples an exponentially decaying level.  The M-dependent scale is
  // the usual 1/log(M) choice; cap only protects malformed/extreme seeds.
  const std::uint64_t bits = splitmix64(rng_state_);
  const double unit =
      (static_cast<double>(bits >> 11U) + 1.0) / 9007199254740992.0;
  const double level_value = -std::log(unit) / std::log(static_cast<double>(m_));
  const double clipped = std::min(level_value, 63.0);
  return static_cast<std::uint32_t>(std::floor(clipped));
}

std::vector<StandardHnswIndex::Candidate> StandardHnswIndex::search_layer(
    const float* query, std::uint32_t entry, std::size_t ef,
    std::uint32_t level, const std::uint8_t* allowed) const {
  if (count_ == 0U || entry >= count_ || ef == 0U) return {};
  if (level >= nodes_[entry].links.size()) {
    throw std::runtime_error("HNSW search entered an unavailable level");
  }
  ef = std::max<std::size_t>(1U, std::min(ef, count_));
  if (++search_stamp_ == 0U) {
    std::fill(visited_stamp_.begin(), visited_stamp_.end(), 0U);
    search_stamp_ = 1U;
  }
  const std::uint32_t stamp = search_stamp_;
  std::priority_queue<Candidate, std::vector<Candidate>, MinCandidate> candidates;
  std::priority_queue<Candidate, std::vector<Candidate>, MaxCandidate> results;
  const Candidate first{distance_to_id(query, entry), entry};
  visited_stamp_[entry] = stamp;
  candidates.push(first);
  if (allowed == nullptr || allowed[entry] != 0U) results.push(first);

  while (!candidates.empty()) {
    const Candidate current = candidates.top();
    candidates.pop();
    if (results.size() >= ef && better(results.top(), current)) break;
    for (const std::uint32_t neighbor : nodes_[current.id].links[level]) {
      if (neighbor >= count_ || visited_stamp_[neighbor] == stamp) continue;
      visited_stamp_[neighbor] = stamp;
      const Candidate candidate{distance_to_id(query, neighbor), neighbor};
      if (results.size() < ef || better(candidate, results.top())) {
        candidates.push(candidate);
        if (allowed == nullptr || allowed[neighbor] != 0U) {
          results.push(candidate);
          if (results.size() > ef) results.pop();
        }
      }
    }
  }

  std::vector<Candidate> output;
  output.reserve(results.size());
  while (!results.empty()) {
    output.push_back(results.top());
    results.pop();
  }
  std::sort(output.begin(), output.end(), better);
  return output;
}

std::vector<std::uint32_t> StandardHnswIndex::select_neighbors(
    std::uint32_t query_id, const std::vector<Candidate>& candidates,
    std::size_t maximum) const {
  if (maximum == 0U) return {};
  std::vector<Candidate> sorted = candidates;
  std::sort(sorted.begin(), sorted.end(), better);
  std::vector<std::uint32_t> selected;
  selected.reserve(std::min(maximum, sorted.size()));
  for (const Candidate candidate : sorted) {
    if (candidate.id == query_id ||
        std::find_if(selected.begin(), selected.end(), [&](std::uint32_t id) {
          return id == candidate.id;
        }) != selected.end()) {
      continue;
    }
    bool keep = true;
    for (const std::uint32_t selected_id : selected) {
      // HNSW's diversity heuristic rejects a candidate when it is closer to
      // an already selected neighbour than it is to the query object.
      if (distance_between(candidate.id, selected_id) < candidate.distance) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push_back(candidate.id);
      if (selected.size() == maximum) return selected;
    }
  }
  // If the diversity pass is sparse, fill the remaining slots by nearest
  // distance.  This is the standard HNSW fallback and prevents low-degree
  // graphs on clustered data.
  for (const Candidate candidate : sorted) {
    if (candidate.id == query_id ||
        std::find(selected.begin(), selected.end(), candidate.id) != selected.end()) {
      continue;
    }
    selected.push_back(candidate.id);
    if (selected.size() == maximum) break;
  }
  return selected;
}

void StandardHnswIndex::prune_neighbors(std::uint32_t owner,
                                        std::uint32_t level,
                                        std::size_t maximum) {
  auto& links = nodes_[owner].links[level];
  if (links.size() <= maximum) {
    std::sort(links.begin(), links.end());
    return;
  }
  std::vector<Candidate> candidates;
  candidates.reserve(links.size());
  for (const std::uint32_t id : links) {
    if (id != owner) candidates.push_back(Candidate{distance_between(owner, id), id});
  }
  links = select_neighbors(owner, candidates, maximum);
  std::sort(links.begin(), links.end());
}

void StandardHnswIndex::insert(std::uint32_t id) {
  const std::uint32_t level = random_level();
  nodes_[id].links.resize(static_cast<std::size_t>(level) + 1U);
  if (max_level_ < 0) {
    entrypoint_ = id;
    max_level_ = static_cast<std::int32_t>(level);
    return;
  }

  std::uint32_t current = entrypoint_;
  const float* query = vectors_.data() + static_cast<std::size_t>(id) * dimension_;
  for (std::int32_t current_level = max_level_; current_level >
       static_cast<std::int32_t>(level); --current_level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const float current_distance = distance_to_id(query, current);
      for (const std::uint32_t neighbor :
           nodes_[current].links[static_cast<std::size_t>(current_level)]) {
        const float neighbor_distance = distance_to_id(query, neighbor);
        const Candidate candidate{neighbor_distance, neighbor};
        const Candidate incumbent{current_distance, current};
        if (better(candidate, incumbent)) {
          current = neighbor;
          changed = true;
          break;
        }
      }
    }
  }

  const std::int32_t lower = std::min<std::int32_t>(
      static_cast<std::int32_t>(level), max_level_);
  for (std::int32_t current_level = lower; current_level >= 0; --current_level) {
    const std::uint32_t level_u = static_cast<std::uint32_t>(current_level);
    const std::vector<Candidate> candidates = search_layer(
        query, current, ef_construction_, level_u, nullptr);
    const std::size_t maximum = level_u == 0U ? static_cast<std::size_t>(2U) * m_ : m_;
    const std::vector<std::uint32_t> selected =
        select_neighbors(id, candidates, maximum);
    nodes_[id].links[level_u] = selected;
    std::sort(nodes_[id].links[level_u].begin(),
              nodes_[id].links[level_u].end());
    for (const std::uint32_t neighbor : selected) {
      if (neighbor >= nodes_.size() ||
          level_u >= nodes_[neighbor].links.size()) {
        throw std::runtime_error("HNSW insertion found an invalid neighbour level");
      }
      auto& reverse = nodes_[neighbor].links[level_u];
      if (std::find(reverse.begin(), reverse.end(), id) == reverse.end()) {
        reverse.push_back(id);
      }
      prune_neighbors(neighbor, level_u, maximum);
    }
    if (!candidates.empty()) current = candidates.front().id;
  }
  if (static_cast<std::int32_t>(level) > max_level_) {
    entrypoint_ = id;
    max_level_ = static_cast<std::int32_t>(level);
  }
}

std::vector<HnswSearchHit> StandardHnswIndex::search(
    const float* query, std::size_t k, std::size_t ef,
    const std::uint8_t* allowed) const {
  if (query == nullptr) throw std::invalid_argument("HNSW query is null");
  if (count_ == 0U || k == 0U) return {};
  for (std::size_t dimension = 0U; dimension < dimension_; ++dimension) {
    if (!std::isfinite(query[dimension])) {
      throw std::invalid_argument("HNSW query must contain finite float32 values");
    }
  }
  ef = std::max(k, std::min(ef, count_));
  std::uint32_t current = entrypoint_;
  for (std::int32_t level = max_level_; level > 0; --level) {
    bool changed = true;
    while (changed) {
      changed = false;
      const Candidate incumbent{distance_to_id(query, current), current};
      for (const std::uint32_t neighbor :
           nodes_[current].links[static_cast<std::size_t>(level)]) {
        const Candidate candidate{distance_to_id(query, neighbor), neighbor};
        if (better(candidate, incumbent)) {
          current = neighbor;
          changed = true;
          break;
        }
      }
    }
  }
  const auto candidates = search_layer(query, current, ef, 0U, allowed);
  const std::size_t take = std::min(k, candidates.size());
  std::vector<HnswSearchHit> output;
  output.reserve(take);
  for (std::size_t position = 0U; position < take; ++position) {
    output.push_back(HnswSearchHit{candidates[position].id,
                                   std::sqrt(std::max(0.0F, candidates[position].distance))});
  }
  return output;
}

void StandardHnswIndex::validate_loaded() const {
  if (count_ == 0U || dimension_ == 0U || m_ < 2U ||
      ef_construction_ < m_ || nodes_.size() != count_ ||
      vectors_.size() != count_ * dimension_ || max_level_ < 0 ||
      entrypoint_ >= count_) {
    throw std::runtime_error("invalid HNSW artifact header");
  }
  if (nodes_[entrypoint_].links.size() != static_cast<std::size_t>(max_level_) + 1U) {
    throw std::runtime_error("invalid HNSW entry point level");
  }
  for (std::size_t node = 0U; node < nodes_.size(); ++node) {
    if (nodes_[node].links.empty() ||
        nodes_[node].links.size() > 64U) {
      throw std::runtime_error("invalid HNSW node level");
    }
    for (std::size_t level = 0U; level < nodes_[node].links.size(); ++level) {
      const std::size_t maximum = level == 0U ? static_cast<std::size_t>(2U) * m_ : m_;
      const auto& links = nodes_[node].links[level];
      if (links.size() > maximum ||
          !std::is_sorted(links.begin(), links.end()) ||
          std::adjacent_find(links.begin(), links.end()) != links.end()) {
        throw std::runtime_error("invalid HNSW adjacency list");
      }
      for (const std::uint32_t neighbor : links) {
        if (neighbor >= count_ || neighbor == node) {
          throw std::runtime_error("HNSW adjacency id is out of range");
        }
      }
    }
  }
}

void StandardHnswIndex::save(const std::filesystem::path& path) const {
  validate_loaded();
  const auto temporary = path.string() + ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot open HNSW artifact for writing");
  output.write(kMagic, sizeof(kMagic));
  write_scalar(output, kFormatVersion, "version");
  write_scalar(output, static_cast<std::uint64_t>(count_), "count");
  write_scalar(output, static_cast<std::uint64_t>(dimension_), "dimension");
  write_scalar(output, m_, "M");
  write_scalar(output, ef_construction_, "ef_construction");
  write_scalar(output, seed_, "seed");
  write_scalar(output, entrypoint_, "entry point");
  write_scalar(output, static_cast<std::int32_t>(max_level_), "max level");
  output.write(reinterpret_cast<const char*>(vectors_.data()),
               static_cast<std::streamsize>(vectors_.size() * sizeof(float)));
  if (!output) throw std::runtime_error("failed to write HNSW vectors");
  for (const Node& node : nodes_) {
    write_scalar(output, static_cast<std::uint32_t>(node.links.size()), "node level");
    for (const auto& links : node.links) {
      write_scalar(output, static_cast<std::uint32_t>(links.size()), "degree");
      if (!links.empty()) {
        output.write(reinterpret_cast<const char*>(links.data()),
                     static_cast<std::streamsize>(links.size() * sizeof(std::uint32_t)));
        if (!output) throw std::runtime_error("failed to write HNSW adjacency");
      }
    }
  }
  output.close();
  if (!output) throw std::runtime_error("failed to close HNSW artifact");
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
  }
  if (error) throw std::runtime_error("cannot publish HNSW artifact: " + error.message());
}

std::shared_ptr<StandardHnswIndex> StandardHnswIndex::load(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open HNSW artifact for reading");
  char magic[sizeof(kMagic)]{};
  input.read(magic, sizeof(magic));
  if (!input || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("HNSW artifact magic/version mismatch");
  }
  if (read_scalar<std::uint32_t>(input, "version") != kFormatVersion) {
    throw std::runtime_error("unsupported HNSW artifact version");
  }
  const std::uint64_t count = read_scalar<std::uint64_t>(input, "count");
  const std::uint64_t dimension = read_scalar<std::uint64_t>(input, "dimension");
  const HnswBuildOptions options{
      read_scalar<std::uint32_t>(input, "M"),
      read_scalar<std::uint32_t>(input, "ef_construction"),
      read_scalar<std::uint64_t>(input, "seed")};
  if (count == 0U || count > std::numeric_limits<std::uint32_t>::max() ||
      dimension == 0U || dimension > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("HNSW artifact shape is out of range");
  }
  auto index = std::shared_ptr<StandardHnswIndex>(
      new StandardHnswIndex(static_cast<std::size_t>(count),
                            static_cast<std::size_t>(dimension), options));
  index->entrypoint_ = read_scalar<std::uint32_t>(input, "entry point");
  index->max_level_ = read_scalar<std::int32_t>(input, "max level");
  input.read(reinterpret_cast<char*>(index->vectors_.data()),
             static_cast<std::streamsize>(index->vectors_.size() * sizeof(float)));
  if (!input) throw std::runtime_error("failed to read HNSW vectors");
  for (Node& node : index->nodes_) {
    const std::uint32_t level_count = read_scalar<std::uint32_t>(input, "node level");
    if (level_count == 0U || level_count > 64U) {
      throw std::runtime_error("HNSW node level is out of range");
    }
    node.links.resize(level_count);
    for (std::size_t level = 0U; level < node.links.size(); ++level) {
      auto& links = node.links[level];
      const std::uint32_t degree = read_scalar<std::uint32_t>(input, "degree");
      const std::size_t maximum = level == 0U
                                       ? static_cast<std::size_t>(2U) * options.m
                                       : options.m;
      if (degree > maximum) throw std::runtime_error("HNSW degree is out of range");
      links.resize(degree);
      if (degree != 0U) {
        input.read(reinterpret_cast<char*>(links.data()),
                   static_cast<std::streamsize>(links.size() * sizeof(std::uint32_t)));
        if (!input) throw std::runtime_error("failed to read HNSW adjacency");
      }
    }
  }
  index->validate_loaded();
  return index;
}

}  // namespace bounded_blowup

// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
//
// A small, self-contained implementation of the standard HNSW hierarchy.  It
// deliberately lives outside the bounded-blowup graph/search code so that it
// can be used as an independent baseline.  The implementation uses float32
// vectors, squared L2 for ordering, and object-id tie breaking throughout.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace bounded_blowup {

struct HnswSearchHit final {
  std::uint32_t id = 0U;
  float distance = 0.0F;
};

struct HnswBuildOptions final {
  std::uint32_t m = 16U;
  std::uint32_t ef_construction = 200U;
  std::uint64_t seed = 0U;
};

// Standard HNSW (Malkov--Yashunin style) with deterministic insertion order.
// The class owns a float32 copy of the corpus so the saved graph and its data
// are one self-contained artifact.  Search does not apply an attribute
// predicate itself: callers can use the returned candidates for post-filtering
// or pass an allowed-id mask through the Python binding for HNSW-In.
class StandardHnswIndex final {
 public:
  StandardHnswIndex() = default;

  static std::shared_ptr<StandardHnswIndex> build(
      const float* vectors, std::size_t count, std::size_t dimension,
      const HnswBuildOptions& options = {});

  static std::shared_ptr<StandardHnswIndex> load(
      const std::filesystem::path& path);

  void save(const std::filesystem::path& path) const;

  // Return at most k hits ordered by (squared L2 distance, object id).  ef is
  // clamped to at least k.  If allowed is non-null, graph traversal remains
  // unrestricted while only allowed objects enter the result set; this is the
  // predicate-aware HNSW-In policy used by the experiment driver.
  std::vector<HnswSearchHit> search(const float* query, std::size_t k,
                                    std::size_t ef,
                                    const std::uint8_t* allowed = nullptr) const;

  std::size_t size() const noexcept { return count_; }
  std::size_t dimension() const noexcept { return dimension_; }
  std::uint32_t m() const noexcept { return m_; }
  std::uint32_t ef_construction() const noexcept { return ef_construction_; }
  std::uint64_t seed() const noexcept { return seed_; }

  // Public only so translation-unit comparators can remain small; callers
  // should treat this as an implementation detail.
  struct Candidate final {
    float distance = 0.0F;
    std::uint32_t id = 0U;
  };

 private:
  struct Node final {
    std::vector<std::vector<std::uint32_t>> links;
  };

  StandardHnswIndex(std::size_t count, std::size_t dimension,
                    const HnswBuildOptions& options);

  float distance(const float* left, const float* right) const noexcept;
  float distance_to_id(const float* query, std::uint32_t id) const noexcept;
  float distance_between(std::uint32_t left, std::uint32_t right) const noexcept;
  std::uint32_t random_level();
  std::vector<Candidate> search_layer(const float* query,
                                      std::uint32_t entry,
                                      std::size_t ef,
                                      std::uint32_t level,
                                      const std::uint8_t* allowed) const;
  std::vector<std::uint32_t> select_neighbors(
      std::uint32_t query_id, const std::vector<Candidate>& candidates,
      std::size_t maximum) const;
  void prune_neighbors(std::uint32_t owner, std::uint32_t level,
                       std::size_t maximum);
  void insert(std::uint32_t id);
  void validate_loaded() const;

  std::size_t count_ = 0U;
  std::size_t dimension_ = 0U;
  std::uint32_t m_ = 16U;
  std::uint32_t ef_construction_ = 200U;
  std::uint64_t seed_ = 0U;
  std::uint64_t rng_state_ = 0U;
  std::uint32_t entrypoint_ = 0U;
  std::int32_t max_level_ = -1;
  std::vector<float> vectors_;
  std::vector<Node> nodes_;
  mutable std::vector<std::uint32_t> visited_stamp_;
  mutable std::uint32_t search_stamp_ = 0U;
};

}  // namespace bounded_blowup

// SPDX-License-Identifier: MIT
#include "bounded_blowup/logical_artifact.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace bounded_blowup {

struct LogicalArtifactImage::State {
  LogicalArtifactMetadata metadata;
  std::array<std::vector<std::uint8_t>, 3U> sections;
  std::array<LogicalSectionDigest, 3U> section_digests{};
  LogicalSectionDigest joint_digest{};
};

namespace {

using Byte = std::uint8_t;

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::atomic<std::uint64_t> logical_artifact_sha256_calls{0U};
#endif

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::array<std::uint32_t, 8> kSha256Initial = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

std::uint32_t rotate_right(std::uint32_t value, unsigned amount) {
  return (value >> amount) | (value << (32U - amount));
}

class Sha256 {
 public:
  Sha256() : state_(kSha256Initial) {}

  void update(const Byte* data, std::size_t size) {
    if (size == 0U) return;
    total_bytes_ += size;
    while (size != 0U) {
      const std::size_t available = block_.size() - block_size_;
      const std::size_t count = std::min(available, size);
      std::memcpy(block_.data() + block_size_, data, count);
      data += count;
      size -= count;
      block_size_ += count;
      if (block_size_ == block_.size()) {
        transform(block_);
        block_size_ = 0U;
      }
    }
  }

  LogicalSectionDigest finish() {
    const std::uint64_t bit_length =
        static_cast<std::uint64_t>(total_bytes_) * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56U) {
      while (block_size_ < block_.size()) block_[block_size_++] = 0U;
      transform(block_);
      block_size_ = 0U;
    }
    while (block_size_ < 56U) block_[block_size_++] = 0U;
    for (unsigned index = 0U; index < 8U; ++index) {
      block_[56U + index] = static_cast<Byte>(
          bit_length >> (56U - 8U * static_cast<std::uint64_t>(index)));
    }
    transform(block_);

    LogicalSectionDigest digest{};
    for (unsigned word = 0U; word < state_.size(); ++word) {
      for (unsigned byte = 0U; byte < 4U; ++byte) {
        digest[word * 4U + byte] = static_cast<Byte>(
            state_[word] >> (24U - 8U * static_cast<std::uint32_t>(byte)));
      }
    }
    return digest;
  }

 private:
  void transform(const std::array<Byte, 64>& block) {
    std::array<std::uint32_t, 64> schedule{};
    for (unsigned word = 0U; word < 16U; ++word) {
      const unsigned offset = word * 4U;
      schedule[word] =
          (static_cast<std::uint32_t>(block[offset]) << 24U) |
          (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
          (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
          static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (unsigned word = 16U; word < schedule.size(); ++word) {
      const std::uint32_t x = schedule[word - 15U];
      const std::uint32_t y = schedule[word - 2U];
      const std::uint32_t sigma0 = rotate_right(x, 7U) ^ rotate_right(x, 18U) ^
                                   (x >> 3U);
      const std::uint32_t sigma1 = rotate_right(y, 17U) ^
                                   rotate_right(y, 19U) ^ (y >> 10U);
      schedule[word] = schedule[word - 16U] + sigma0 + schedule[word - 7U] +
                       sigma1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (unsigned round = 0U; round < 64U; ++round) {
      const std::uint32_t sigma1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                   rotate_right(e, 25U);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + sigma1 + choice +
                                  kSha256RoundConstants[round] +
                                  schedule[round];
      const std::uint32_t sigma0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                   rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<Byte, 64> block_{};
  std::size_t block_size_ = 0U;
  std::size_t total_bytes_ = 0U;
};

LogicalSectionDigest digest_bytes(const std::vector<Byte>& bytes) {
  Sha256 sha;
  sha.update(bytes.data(), bytes.size());
  return sha.finish();
}

void append_u32_le(std::vector<Byte>* bytes, std::uint32_t value) {
  for (unsigned index = 0U; index < 4U; ++index) {
    bytes->push_back(static_cast<Byte>(value >> (index * 8U)));
  }
}

void append_u64_le(std::vector<Byte>* bytes, std::uint64_t value) {
  for (unsigned index = 0U; index < 8U; ++index) {
    bytes->push_back(static_cast<Byte>(value >> (index * 8U)));
  }
}

std::uint32_t read_u32_le(const std::vector<Byte>& bytes, std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4U) {
    throw LogicalArtifactError("route section ends in a truncated uint32");
  }
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
         (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
         (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

std::uint64_t read_u64_le(const std::vector<Byte>& bytes, std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8U) {
    throw LogicalArtifactError("route section ends in a truncated uint64");
  }
  std::uint64_t value = 0U;
  for (unsigned index = 0U; index < 8U; ++index) {
    value |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8U);
  }
  return value;
}

void require_route_order(
    const std::vector<LogicalArtifactRouteBinding>& routes,
    const char* label);

std::vector<Byte> encode_route_records(
    const std::vector<LogicalArtifactRouteBinding>& routes) {
  std::vector<Byte> bytes;
  static constexpr std::array<Byte, 8U> kRouteMagic = {
      'R', 'F', 'B', 'B', 'R', 'T', 'V', '3'};
  bytes.insert(bytes.end(), kRouteMagic.begin(), kRouteMagic.end());
  append_u32_le(&bytes, 1U);
  if (routes.size() > std::numeric_limits<std::size_t>::max() / 12U) {
    throw LogicalArtifactError("route table size overflows bytes");
  }
  append_u64_le(&bytes, static_cast<std::uint64_t>(routes.size()));
  bytes.reserve(bytes.size() + routes.size() * 12U);
  for (const auto& route : routes) {
    append_u64_le(&bytes, route.block_id);
    append_u32_le(&bytes, route.view_id);
  }
  return bytes;
}

std::vector<LogicalArtifactRouteBinding> decode_route_records(
    const std::vector<Byte>& bytes) {
  static constexpr std::array<Byte, 8U> kRouteMagic = {
      'R', 'F', 'B', 'B', 'R', 'T', 'V', '3'};
  if (bytes.size() < kRouteMagic.size() + 4U + 8U) {
    throw LogicalArtifactError("V3 route section is truncated");
  }
  if (!std::equal(kRouteMagic.begin(), kRouteMagic.end(), bytes.begin())) {
    throw LogicalArtifactError("V3 route section magic is invalid");
  }
  const std::size_t header = kRouteMagic.size();
  const std::uint32_t version = read_u32_le(bytes, header);
  if (version != 1U) throw LogicalArtifactError("V3 route section version is unsupported");
  const std::uint64_t count64 = read_u64_le(bytes, header + 4U);
  if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      count64 > (std::numeric_limits<std::size_t>::max() - header - 12U) / 12U ||
      bytes.size() != header + 12U + static_cast<std::size_t>(count64) * 12U) {
    throw LogicalArtifactError("V3 route section count/length mismatch");
  }
  std::vector<LogicalArtifactRouteBinding> routes;
  routes.resize(static_cast<std::size_t>(count64));
  std::size_t offset = header + 12U;
  for (auto& route : routes) {
    route.block_id = read_u64_le(bytes, offset);
    route.view_id = read_u32_le(bytes, offset + 8U);
    offset += 12U;
  }
  require_route_order(routes, "V3 route section");
  for (std::size_t index = 0U; index < routes.size(); ++index) {
    if (routes[index].block_id != static_cast<std::uint64_t>(index)) {
      throw LogicalArtifactError("V3 route block IDs are not dense");
    }
  }
  return routes;
}

CorpusView corpus_view(const VerifiedCorpusSnapshot& snapshot) {
  if (snapshot.object_count() == 0U || snapshot.vector_dim() == 0U ||
      snapshot.vector_dim() >
          static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw LogicalArtifactError("verified corpus shape is outside graph limits");
  }
  const std::uint64_t object_count = snapshot.object_count();
  const std::uint64_t dimension = snapshot.vector_dim();
  if (object_count > std::numeric_limits<std::uint64_t>::max() / dimension) {
    throw LogicalArtifactError("verified corpus vector shape overflows");
  }
  const std::uint64_t expected_values = object_count * dimension;
  if (expected_values != static_cast<std::uint64_t>(snapshot.vectors().size())) {
    throw LogicalArtifactError("verified corpus vectors do not match shape");
  }
  return CorpusView{snapshot.vectors(), static_cast<std::uint32_t>(dimension),
                    object_count, true};
}

LogicalArtifactIdentity observed_identity(
    const VerifiedCorpusSnapshot& snapshot) {
  return LogicalArtifactIdentity{
      snapshot.identity().canonical_sha256().hex(),
      snapshot.identity().vectors().file_sha256().hex(),
      snapshot.identity().attrs_raw().file_sha256().hex()};
}

void require_identity_strings(const LogicalArtifactIdentity& identity,
                              const char* label) {
  try {
    (void)Digest256::parse(identity.corpus_sha256);
    (void)Digest256::parse(identity.corpus_vectors_sha256);
    (void)Digest256::parse(identity.corpus_attrs_sha256);
  } catch (const DigestError& error) {
    throw LogicalArtifactError(std::string(label) +
                               " corpus identity is not lowercase hex64: " +
                               error.what());
  }
}

std::uint64_t double_bits(const double value) noexcept {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void require_block_kind(const LogicalBlockKind kind) {
  switch (kind) {
    case LogicalBlockKind::kLeaf:
    case LogicalBlockKind::kParent:
      return;
  }
  throw LogicalArtifactError("logical artifact block kind is invalid");
}

void require_entries(const std::vector<std::int64_t>& members,
                     std::int64_t anchor,
                     const std::vector<std::int64_t>& entries) {
  for (std::size_t index = 0U; index < entries.size(); ++index) {
    const std::int64_t entry = entries[index];
    if (entry == anchor ||
        !std::binary_search(members.begin(), members.end(), entry) ||
        (index != 0U && entries[index - 1U] >= entry)) {
      throw LogicalArtifactError(
          "logical artifact entries must be strictly increasing members "
          "excluding the anchor");
    }
  }
}

void require_parent_options(const ParentComposeOptions& options) {
  if (options.max_degree != kLogicalMaxDegree || options.sample_count != 8U ||
      !std::isfinite(options.alpha) ||
      double_bits(options.alpha) != double_bits(1.2) ||
      options.thread_count == 0U || options.thread_count > 16U) {
    throw LogicalArtifactError(
        "logical parent options do not match the frozen repair contract");
  }
}

// The repaired leaf's object IDs are sorted, but its CSR targets are not
// required to be sorted.  Keep target validation linear in the number of
// stored edges with one flat, open-addressed object->row map.  In particular,
// do not create a tree/hash allocation for every row of a large leaf.
class FlatObjectRowMap final {
 public:
  explicit FlatObjectRowMap(const std::vector<std::int64_t>& object_ids) {
    if (object_ids.size() > std::numeric_limits<std::size_t>::max() / 2U) {
      throw LogicalArtifactError("repaired leaf row map is too large");
    }
    const std::size_t required = object_ids.size() * 2U;
    std::size_t capacity = 1U;
    while (capacity < required) {
      if (capacity > std::numeric_limits<std::size_t>::max() / 2U) {
        throw LogicalArtifactError("repaired leaf row map is too large");
      }
      capacity *= 2U;
    }
    keys_.assign(capacity, kEmptyKey);
    rows_.assign(capacity, 0U);
    mask_ = capacity - 1U;
    for (std::size_t row = 0U; row < object_ids.size(); ++row) {
      const std::int64_t object_id = object_ids[row];
      std::size_t slot = hash_slot(object_id);
      while (keys_[slot] != kEmptyKey) {
        if (keys_[slot] == object_id) {
          throw LogicalArtifactError("repaired leaf members are not unique");
        }
        slot = (slot + 1U) & mask_;
      }
      keys_[slot] = object_id;
      rows_[slot] = static_cast<std::uint32_t>(row);
    }
  }

  bool find(const std::int64_t object_id, std::uint32_t* row) const noexcept {
    std::size_t slot = hash_slot(object_id);
    while (keys_[slot] != kEmptyKey) {
      if (keys_[slot] == object_id) {
        *row = rows_[slot];
        return true;
      }
      slot = (slot + 1U) & mask_;
    }
    return false;
  }

 private:
  static constexpr std::int64_t kEmptyKey =
      std::numeric_limits<std::int64_t>::min();

  std::size_t hash_slot(const std::int64_t object_id) const noexcept {
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &object_id, sizeof(bits));
    bits ^= bits >> 30U;
    bits *= 0xbf58476d1ce4e5b9ULL;
    bits ^= bits >> 27U;
    bits *= 0x94d049bb133111ebULL;
    bits ^= bits >> 31U;
    return static_cast<std::size_t>(bits) & mask_;
  }

  std::vector<std::int64_t> keys_;
  std::vector<std::uint32_t> rows_;
  std::size_t mask_ = 0U;
};

LeafBuildResult validate_leaf_repair(const CorpusView& corpus,
                                     const LeafRepairResult& repaired,
                                     std::int64_t anchor) {
  const std::size_t row_count = repaired.row_object_ids.size();
  if (row_count == 0U ||
      row_count >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw LogicalArtifactError("repaired leaf row count is invalid");
  }
  if (repaired.row_ptr.size() != row_count + 1U ||
      repaired.row_ptr.empty() || repaired.row_ptr.front() != 0U ||
      repaired.row_ptr.back() !=
          static_cast<std::uint64_t>(repaired.target_ids.size())) {
    throw LogicalArtifactError("repaired leaf CSR shape is invalid");
  }
  if (repaired.witness_parent_row.size() != row_count) {
    throw LogicalArtifactError("repaired leaf witness shape is invalid");
  }
  if (repaired.evidence.rows != static_cast<std::uint64_t>(row_count)) {
    throw LogicalArtifactError("repaired leaf evidence row count is invalid");
  }
  if (repaired.identity.frozen_anchor != anchor ||
      repaired.identity.max_degree != kLeafRepairMaxDegree ||
      repaired.identity.sample_budget != kLeafRepairSampleBudget ||
      repaired.identity.algorithm_revision != kLeafRepairAlgorithmRevision ||
      repaired.identity.numeric_revision != kLeafRepairNumericRevision ||
      repaired.identity.tie_order != kLeafRepairTieOrder ||
      !repaired.identity.thread_independent ||
      repaired.identity.alpha_bits != double_bits(1.2) ||
      repaired.identity.alpha_squared_bits != double_bits(1.2 * 1.2)) {
    throw LogicalArtifactError(
        "repaired leaf identity does not match the frozen algorithm contract");
  }

  // Check the ordering precondition before calling lower_bound.  Besides
  // making duplicate/unsorted IDs fail closed, this keeps the binary search
  // below within its documented sorted-range contract.
  for (std::size_t row = 0U; row < row_count; ++row) {
    const std::int64_t object_id = repaired.row_object_ids[row];
    if (object_id < 0 ||
        static_cast<std::uint64_t>(object_id) >= corpus.object_count ||
        (row != 0U && repaired.row_object_ids[row - 1U] >= object_id)) {
      throw LogicalArtifactError(
          "repaired leaf members are not sorted, unique, and in corpus");
    }
  }

  const auto anchor_iterator = std::lower_bound(
      repaired.row_object_ids.begin(), repaired.row_object_ids.end(), anchor);
  if (anchor_iterator == repaired.row_object_ids.end() ||
      *anchor_iterator != anchor) {
    throw LogicalArtifactError("repaired leaf anchor is outside its members");
  }
  const std::size_t anchor_row = static_cast<std::size_t>(
      anchor_iterator - repaired.row_object_ids.begin());

  const FlatObjectRowMap object_to_row(repaired.row_object_ids);
  for (std::size_t row = 0U; row < row_count; ++row) {
    const std::int64_t object_id = repaired.row_object_ids[row];
    const std::uint64_t begin = repaired.row_ptr[row];
    const std::uint64_t end = repaired.row_ptr[row + 1U];
    if (begin > end ||
        end > static_cast<std::uint64_t>(repaired.target_ids.size()) ||
        end - begin > kLogicalMaxDegree) {
      throw LogicalArtifactError("repaired leaf row exceeds degree 24");
    }
    // A repaired row is bounded by the frozen degree. Keep this inner audit
    // allocation-free: a per-row std::set would turn a 50K-row validation
    // into 50K heap-backed trees despite the fixed 24-edge contract.
    std::array<std::uint32_t, kLogicalMaxDegree> seen_targets{};
    std::size_t seen_target_count = 0U;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      const std::int64_t target_id =
          repaired.target_ids[static_cast<std::size_t>(offset)];
      std::uint32_t target_row = 0U;
      if (!object_to_row.find(target_id, &target_row) ||
          target_id == object_id) {
        throw LogicalArtifactError("repaired leaf CSR target is invalid");
      }
      if (std::find(seen_targets.begin(),
                    seen_targets.begin() +
                        static_cast<std::ptrdiff_t>(seen_target_count),
                    target_row) !=
          seen_targets.begin() +
              static_cast<std::ptrdiff_t>(seen_target_count)) {
        throw LogicalArtifactError("repaired leaf CSR has duplicate targets");
      }
      seen_targets[seen_target_count++] = target_row;
    }
  }

  if (repaired.witness_parent_row[anchor_row] != kNoWitnessParent) {
    throw LogicalArtifactError("repaired leaf anchor has a witness parent");
  }
  for (std::size_t row = 0U; row < row_count; ++row) {
    if (row == anchor_row) continue;
    const std::uint64_t parent = repaired.witness_parent_row[row];
    if (parent >= static_cast<std::uint64_t>(row_count)) {
      throw LogicalArtifactError("repaired leaf witness parent is invalid");
    }
    // The receipt names the directed witness edge parent -> row. Checking the
    // reverse direction would accept a graph whose witness chain cannot
    // actually reach the anchor under the stored adjacency semantics.
    const std::uint64_t begin =
        repaired.row_ptr[static_cast<std::size_t>(parent)];
    const std::uint64_t end =
        repaired.row_ptr[static_cast<std::size_t>(parent) + 1U];
    const std::int64_t row_id = repaired.row_object_ids[row];
    bool has_witness_edge = false;
    for (std::uint64_t offset = begin; offset < end; ++offset) {
      if (repaired.target_ids[static_cast<std::size_t>(offset)] == row_id) {
        has_witness_edge = true;
        break;
      }
    }
    if (!has_witness_edge) {
      throw LogicalArtifactError("repaired leaf witness is not a CSR edge");
    }
  }

  // A witness is a compact reachability proof. Validate all parent pointers
  // in linear time so a 50K-row leaf cannot trigger a quadratic chain walk.
  std::vector<std::uint8_t> state(row_count, 0U);
  state[anchor_row] = 2U;
  std::vector<std::size_t> path;
  path.reserve(row_count);
  for (std::size_t start = 0U; start < row_count; ++start) {
    if (state[start] == 2U) continue;
    path.clear();
    std::size_t current = start;
    while (state[current] == 0U) {
      state[current] = 1U;
      path.push_back(current);
      current = static_cast<std::size_t>(repaired.witness_parent_row[current]);
    }
    if (state[current] != 2U) {
      throw LogicalArtifactError("repaired leaf witness contains a cycle");
    }
    for (const std::size_t row : path) state[row] = 2U;
  }

  LeafBuildResult builder_input;
  builder_input.row_object_ids = repaired.row_object_ids;
  builder_input.row_ptr = repaired.row_ptr;
  builder_input.target_ids = repaired.target_ids;
  return builder_input;
}

void require_route_order(
    const std::vector<LogicalArtifactRouteBinding>& routes,
    const char* label) {
  if (routes.empty()) throw LogicalArtifactError(std::string(label) + " is empty");
  std::set<std::uint64_t> block_ids;
  std::set<LogicalViewId> view_ids;
  for (std::size_t index = 0U; index < routes.size(); ++index) {
    if (!block_ids.insert(routes[index].block_id).second) {
      throw LogicalArtifactError(std::string(label) + " repeats a block ID");
    }
    if (!view_ids.insert(routes[index].view_id).second) {
      throw LogicalArtifactError(std::string(label) + " repeats a view ID");
    }
    if (index != 0U && routes[index - 1U].block_id >= routes[index].block_id) {
      throw LogicalArtifactError(std::string(label) +
                                 " must be sorted by explicit block ID");
    }
  }
}

const LogicalArtifactRouteBinding* find_route(
    const std::vector<LogicalArtifactRouteBinding>& routes,
    std::uint64_t block_id) {
  const auto iterator = std::lower_bound(
      routes.begin(), routes.end(), block_id,
      [](const LogicalArtifactRouteBinding& route, std::uint64_t value) {
        return route.block_id < value;
      });
  if (iterator == routes.end() || iterator->block_id != block_id) return nullptr;
  return &*iterator;
}

template <typename T>
ArrayView<T> section_slice(const std::vector<T>& values,
                           const std::vector<std::uint64_t>& pointers,
                           std::size_t section) {
  const std::uint64_t begin = pointers[section];
  const std::uint64_t end = pointers[section + 1U];
  if (begin == end) return ArrayView<T>();
  return ArrayView<T>(values.data() + static_cast<std::size_t>(begin),
                      static_cast<std::size_t>(end - begin));
}

bool equal_array_view(const std::vector<std::int64_t>& lhs,
                      ArrayView<std::int64_t> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool equal_array_view(const std::vector<LogicalViewId>& lhs,
                      ArrayView<LogicalViewId> rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

void verify_route_contract(const LogicalGraphSections& sections,
                           const std::vector<LogicalArtifactRouteBinding>&
                               route_bindings) {
  require_route_order(route_bindings, "route binding table");
  const LogicalGraphSectionData& data = sections.data();
  if (data.block_view_id.size() != route_bindings.size()) {
    throw LogicalArtifactError("route table and logical view count differ");
  }
  // Build one section-index map. A linear scan per route made opening a
  // 16K-route artifact quadratic in the number of logical blocks.
  std::map<LogicalViewId, std::size_t> view_index;
  for (std::size_t block = 0U; block < data.block_view_id.size(); ++block) {
    if (!view_index.emplace(data.block_view_id[block], block).second) {
      throw LogicalArtifactError("logical sections repeat a view ID");
    }
  }
  for (const auto& route : route_bindings) {
    const auto found = view_index.find(route.view_id);
    if (found == view_index.end()) {
      throw LogicalArtifactError("route points outside logical views");
    }
    const std::size_t block = found->second;
    const std::uint8_t expected_kind = static_cast<std::uint8_t>(
        route.level == 0U ? LogicalBlockKind::kLeaf
                          : LogicalBlockKind::kParent);
    if (data.block_depth[block] != route.level ||
        data.block_component_sort_key[block] != route.component_sort_key ||
        data.block_kind[block] != expected_kind) {
      throw LogicalArtifactError("route level/kind metadata mismatch");
    }
    const auto members = section_slice(data.member_ids, data.block_member_ptr,
                                       block);
    if (members.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      throw LogicalArtifactError("logical route view exceeds uint32 row space");
    }
    if (!equal_array_view(route.members, members)) {
      throw LogicalArtifactError("route members mismatch");
    }
    const auto entries = section_slice(data.entry_ids, data.block_entry_ptr,
                                       block);
    if (!equal_array_view(route.entries, entries)) {
      throw LogicalArtifactError("route entry binding mismatch");
    }
    if (route.anchor != members[data.block_anchor_local_row[block]]) {
      throw LogicalArtifactError("route anchor mismatch");
    }
    const auto binding = section_slice(data.binding_child_view_id,
                                       data.block_binding_ptr, block);
    if (!equal_array_view(route.children, binding)) {
      throw LogicalArtifactError("route child binding mismatch");
    }
  }
}

std::string section_kind_name(LogicalArtifactSectionKind kind) {
  switch (kind) {
    case LogicalArtifactSectionKind::kLogicalGraphCanonical:
      return "logical_graph_canonical";
    case LogicalArtifactSectionKind::kBlockLogicalViewId:
      return "block_logical_view_id";
    case LogicalArtifactSectionKind::kRoutingLayoutCanonical:
      return "routing_layout_canonical";
  }
  throw LogicalArtifactError("unknown V3 section kind");
}

std::vector<Byte> joint_image_bytes(
    const LogicalArtifactMetadata& metadata,
    const std::array<std::vector<Byte>, 3U>& sections,
    const std::array<LogicalSectionDigest, 3U>& digests) {
  std::vector<Byte> bytes;
  static constexpr std::array<Byte, 25U> kDomain = {
      'R', 'F', 'B', 'B', '-', 'A', 'R', 'T', 'I', 'F', 'A', 'C', 'T', '-',
      'V', '3', '-', 'J', 'O', 'I', 'N', 'T', '-', '1', 0U};
  bytes.insert(bytes.end(), kDomain.begin(), kDomain.end());
  append_u32_le(&bytes, kLogicalArtifactSchemaVersion);
  append_u32_le(&bytes, static_cast<std::uint32_t>(metadata.algorithm_revision.size()));
  bytes.insert(bytes.end(), metadata.algorithm_revision.begin(),
               metadata.algorithm_revision.end());
  append_u32_le(&bytes, static_cast<std::uint32_t>(metadata.numeric_revision.size()));
  bytes.insert(bytes.end(), metadata.numeric_revision.begin(),
               metadata.numeric_revision.end());
  const Digest256 corpus = Digest256::parse(metadata.identity.corpus_sha256);
  bytes.insert(bytes.end(), corpus.bytes().begin(), corpus.bytes().end());
  static constexpr std::array<LogicalArtifactSectionKind, 3U> kKinds = {
      LogicalArtifactSectionKind::kLogicalGraphCanonical,
      LogicalArtifactSectionKind::kBlockLogicalViewId,
      LogicalArtifactSectionKind::kRoutingLayoutCanonical};
  for (std::size_t index = 0U; index < kKinds.size(); ++index) {
    const std::string name = section_kind_name(kKinds[index]);
    append_u32_le(&bytes, static_cast<std::uint32_t>(kKinds[index]));
    append_u32_le(&bytes, static_cast<std::uint32_t>(name.size()));
    bytes.insert(bytes.end(), name.begin(), name.end());
    append_u64_le(&bytes, static_cast<std::uint64_t>(sections[index].size()));
    bytes.insert(bytes.end(), digests[index].begin(), digests[index].end());
  }
  return bytes;
}

void verify_metadata(const VerifiedCorpusSnapshot& snapshot,
                     const LogicalArtifactImage& image,
                     const LogicalGraphSections& sections,
                     const LogicalGraphView& graph,
                     const LogicalSectionDigest& routing_digest,
                     std::size_t routing_bytes) {
  const auto& metadata = image.metadata();
  if (metadata.schema_version != kLogicalArtifactSchemaVersion ||
      metadata.store_kind != kLogicalStoreKind ||
      metadata.algorithm_revision != kLogicalAlgorithmRevision ||
      metadata.numeric_revision != kLogicalNumericRevision) {
    throw LogicalArtifactError("V3 metadata identity is unsupported");
  }
  const LogicalArtifactIdentity snapshot_identity = observed_identity(snapshot);
  require_identity_strings(metadata.identity, "artifact");
  if (!(metadata.identity == snapshot_identity)) {
    throw LogicalArtifactError("artifact corpus identity mismatch");
  }
  if (metadata.corpus_object_count != snapshot.object_count() ||
      metadata.corpus_vector_dim != snapshot.vector_dim() ||
      snapshot.vectors().empty()) {
    throw LogicalArtifactError("artifact corpus shape/finite contract mismatch");
  }
  if (metadata.logical_sha256 != sections.sha256() ||
      metadata.routing_sha256 != routing_digest ||
      metadata.routing_section_bytes != routing_bytes ||
      metadata.logical_section_bytes != sections.canonical_bytes().size() ||
      !(metadata.logical_stats == graph.stats())) {
    throw LogicalArtifactError("V3 digest or storage statistics mismatch");
  }
}

}  // namespace

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::uint64_t logical_artifact_sha256_call_count_for_testing() noexcept {
  return logical_artifact_sha256_calls.load(std::memory_order_relaxed);
}

void reset_logical_artifact_sha256_call_count_for_testing() noexcept {
  logical_artifact_sha256_calls.store(0U, std::memory_order_relaxed);
}
#endif

LogicalSectionDigest logical_artifact_sha256(ArrayView<std::uint8_t> bytes) {
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  logical_artifact_sha256_calls.fetch_add(1U, std::memory_order_relaxed);
#endif
  Sha256 sha;
  sha.update(bytes.data(), bytes.size());
  return sha.finish();
}

void LogicalSearchWorkspace::prepare(std::size_t local_row_count) {
  if (local_row_count == 0U ||
      local_row_count > std::numeric_limits<std::uint32_t>::max()) {
    throw LogicalArtifactError("logical search workspace row count is invalid");
  }
  seen_epoch.assign(local_row_count, 0U);
  queue.reserve(local_row_count);
  output_object_ids.reserve(local_row_count);
  epoch = 0U;
}

LogicalArtifactImage::LogicalArtifactImage(
    std::shared_ptr<const State> state)
    : state_(std::move(state)) {
  if (!state_) throw LogicalArtifactError("logical artifact image is null");
}

LogicalArtifactImage::LogicalArtifactImage(
    const LogicalArtifactImage& other)
    : state_(std::make_shared<const State>(*other.state_)) {}

LogicalArtifactImage& LogicalArtifactImage::operator=(
    const LogicalArtifactImage& other) {
  if (this != &other) state_ = std::make_shared<const State>(*other.state_);
  return *this;
}

LogicalArtifactImage::LogicalArtifactImage(
    LogicalArtifactImage&& other) noexcept = default;
LogicalArtifactImage& LogicalArtifactImage::operator=(
    LogicalArtifactImage&& other) noexcept = default;
LogicalArtifactImage::~LogicalArtifactImage() = default;

std::uint32_t LogicalArtifactImage::schema_version() const noexcept {
  return state_->metadata.schema_version;
}

const LogicalArtifactMetadata& LogicalArtifactImage::metadata() const noexcept {
  return state_->metadata;
}

const std::vector<std::uint8_t>& LogicalArtifactImage::section(
    LogicalArtifactSectionKind kind) const noexcept {
  return state_->sections[static_cast<std::size_t>(kind)];
}

LogicalSectionDigest LogicalArtifactImage::section_digest(
    LogicalArtifactSectionKind kind) const noexcept {
  return state_->section_digests[static_cast<std::size_t>(kind)];
}

LogicalSectionDigest LogicalArtifactImage::joint_digest() const noexcept {
  return state_->joint_digest;
}

LogicalArtifactImage LogicalArtifactImage::build(
    VerifiedCorpusSnapshot corpus, RoutingLayoutSnapshot routing,
    std::vector<LogicalArtifactBlockInput> blocks,
    const LogicalArtifactBuildOptions& options) {
  auto corpus_snapshot =
      std::make_shared<const VerifiedCorpusSnapshot>(corpus);
  const CorpusView bound_corpus = corpus_view(*corpus_snapshot);
  if (blocks.empty()) throw LogicalArtifactError("cannot build empty artifact");
  if (options.thread_count == 0U || options.thread_count > 16U) {
    throw LogicalArtifactError("thread_count must be in the range [1, 16]");
  }
  if (!routing.state_) {
    throw LogicalArtifactError("V3 artifact build requires routing layout");
  }
  if (routing.block_count() != blocks.size()) {
    throw LogicalArtifactError("routing and graph block counts differ");
  }
  std::vector<std::uint8_t> seen_block_ids(blocks.size(), 0U);
  for (const auto& block : blocks) {
    require_block_kind(block.kind);
    if (block.level > kLogicalMaxDepth) {
      throw LogicalArtifactError("logical artifact block depth exceeds 16");
    }
    if (block.block_id >= blocks.size() ||
        seen_block_ids[static_cast<std::size_t>(block.block_id)] != 0U ||
        routing.block_level(static_cast<std::size_t>(block.block_id)) !=
            block.level) {
      throw LogicalArtifactError(
          "routing and graph block levels differ or block IDs are not dense");
    }
    seen_block_ids[static_cast<std::size_t>(block.block_id)] = 1U;
  }
  for (const std::uint8_t seen : seen_block_ids) {
    if (seen == 0U) throw LogicalArtifactError("graph block IDs are not dense");
  }
  std::sort(blocks.begin(), blocks.end(),
            [](const LogicalArtifactBlockInput& lhs,
               const LogicalArtifactBlockInput& rhs) {
              if (lhs.level != rhs.level) return lhs.level < rhs.level;
              return lhs.block_id < rhs.block_id;
            });
  std::set<LogicalViewId> views;
  for (const auto& block : blocks) {
    if (!views.insert(block.view_id).second) {
      throw LogicalArtifactError("bulk descriptor repeats a logical view ID");
    }
  }

  LogicalGraphBuilder builder(bound_corpus);
  std::set<LogicalViewId> added;
  for (auto& block : blocks) {
    require_entries(block.members, block.anchor, block.entries);
    switch (block.kind) {
      case LogicalBlockKind::kLeaf: {
        if (block.level != 0U || !block.children.empty()) {
          throw LogicalArtifactError(
              "leaf descriptor has invalid level/children");
        }
        const LeafBuildResult leaf =
            validate_leaf_repair(bound_corpus, block.leaf_repair, block.anchor);
        if (block.members != leaf.row_object_ids) {
          throw LogicalArtifactError(
              "leaf descriptor members disagree with repaired rows");
        }
        // The flat-map audit above is the Artifact seam's O(R+E) validation.
        // GraphBuilder::add_leaf is a separate canonical storage materializer;
        // its bounded per-row containers are not part of this audit path.
        builder.add_leaf(
            block.view_id, block.component_sort_key, block.anchor,
            ArrayView<std::int64_t>(block.entries.data(), block.entries.size()),
            leaf);
        break;
      }
      case LogicalBlockKind::kParent: {
        if (block.level == 0U || block.children.empty() || block.members.empty()) {
          throw LogicalArtifactError(
              "parent descriptor has invalid level/children/members");
        }
        require_parent_options(block.parent_options);
        for (const LogicalViewId child : block.children) {
          if (added.count(child) == 0U) {
            throw LogicalArtifactError(
                "parent child is not earlier in canonical (level,block_id) order");
          }
        }
        ParentComposeOptions parent_options = block.parent_options;
        parent_options.thread_count = options.thread_count;
        builder.add_parent(
            block.view_id, block.component_sort_key,
            ArrayView<std::int64_t>(block.members.data(), block.members.size()),
            block.anchor,
            ArrayView<std::int64_t>(block.entries.data(), block.entries.size()),
            ArrayView<LogicalViewId>(block.children.data(), block.children.size()),
            parent_options);
        break;
      }
      default:
        throw LogicalArtifactError("logical artifact block kind is invalid");
    }
    added.insert(block.view_id);
  }

  LogicalGraphSections logical_sections = builder.freeze_sections();
  LogicalGraphView logical_view =
      LogicalGraphView::open(logical_sections, bound_corpus);
  std::vector<LogicalArtifactRouteBinding> route_bindings;
  route_bindings.reserve(blocks.size());
  for (const auto& block : blocks) {
    LogicalArtifactRouteBinding route;
    route.block_id = block.block_id;
    route.view_id = block.view_id;
    route.level = block.level;
    route.component_sort_key = block.component_sort_key;
    route.anchor = block.anchor;
    route.members = block.members;
    switch (block.kind) {
      case LogicalBlockKind::kLeaf:
        route.members = block.leaf_repair.row_object_ids;
        break;
      case LogicalBlockKind::kParent:
        break;
      default:
        throw LogicalArtifactError("logical artifact block kind is invalid");
    }
    route.entries = block.entries;
    route.children = block.children;
    route_bindings.push_back(std::move(route));
  }
  std::sort(route_bindings.begin(), route_bindings.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.block_id < rhs.block_id;
            });
  require_route_order(route_bindings, "built route table");

  // Keep the canonical-byte owner alive while copying.  Calling
  // canonical_bytes() separately for begin/end would compare iterators from
  // two different temporaries and is undefined even when the bytes happen to
  // be equal.
  const auto logical_canonical_bytes =
      logical_sections.canonical_bytes();
  const std::vector<Byte> logical_bytes(logical_canonical_bytes.begin(),
                                       logical_canonical_bytes.end());
  const std::vector<Byte> route_bytes = encode_route_records(route_bindings);
  const std::vector<Byte> routing_bytes = routing.canonical_bytes();
  const LogicalSectionDigest logical_digest = logical_sections.sha256();
  const LogicalSectionDigest route_digest = digest_bytes(route_bytes);
  auto image_state = std::make_shared<State>();
  image_state->sections[0] = logical_bytes;
  image_state->sections[1] = route_bytes;
  image_state->sections[2] = routing_bytes;
  image_state->section_digests[0] = logical_digest;
  image_state->section_digests[1] = route_digest;
  image_state->section_digests[2] = routing.digest();
  image_state->metadata.identity = observed_identity(*corpus_snapshot);
  image_state->metadata.corpus_object_count = corpus_snapshot->object_count();
  image_state->metadata.corpus_vector_dim =
      static_cast<std::uint32_t>(corpus_snapshot->vector_dim());
  image_state->metadata.logical_sha256 = logical_digest;
  image_state->metadata.route_sha256 = route_digest;
  image_state->metadata.routing_sha256 = image_state->section_digests[2];
  image_state->metadata.logical_section_bytes = logical_bytes.size();
  image_state->metadata.route_section_bytes = route_bytes.size();
  image_state->metadata.routing_section_bytes = routing_bytes.size();
  image_state->metadata.logical_stats = logical_view.stats();
  const std::vector<Byte> joint = joint_image_bytes(
      image_state->metadata, image_state->sections,
      image_state->section_digests);
  image_state->joint_digest = digest_bytes(joint);
  image_state->metadata.joint_sha256 = image_state->joint_digest;
  if (image_state->metadata.schema_version != kLogicalArtifactSchemaVersion ||
      image_state->metadata.store_kind != kLogicalStoreKind ||
      image_state->metadata.algorithm_revision != kLogicalAlgorithmRevision ||
      image_state->metadata.numeric_revision != kLogicalNumericRevision) {
    throw LogicalArtifactError("V3 metadata identity is unsupported");
  }
  verify_route_contract(logical_sections, route_bindings);
  return LogicalArtifactImage(std::move(image_state));
}

LogicalArtifactIndex LogicalArtifactIndex::open(
    VerifiedCorpusSnapshot corpus, const LogicalArtifactImage& image) {
  if (!image.state_) throw LogicalArtifactError("logical artifact image is empty");
  auto corpus_snapshot =
      std::make_shared<const VerifiedCorpusSnapshot>(corpus);
  const CorpusView bound_corpus = corpus_view(*corpus_snapshot);
  if (image.metadata().schema_version != kLogicalArtifactSchemaVersion ||
      image.metadata().store_kind != kLogicalStoreKind ||
      image.metadata().algorithm_revision != kLogicalAlgorithmRevision ||
      image.metadata().numeric_revision != kLogicalNumericRevision) {
    throw LogicalArtifactError("V3 metadata identity is unsupported");
  }
  const LogicalArtifactIdentity observed = observed_identity(*corpus_snapshot);
  require_identity_strings(image.metadata().identity, "artifact");
  if (!(image.metadata().identity == observed) ||
      image.metadata().corpus_object_count != corpus_snapshot->object_count() ||
      image.metadata().corpus_vector_dim != corpus_snapshot->vector_dim()) {
    throw LogicalArtifactError("V3 artifact corpus identity mismatch");
  }
  std::array<LogicalSectionDigest, 3U> digests{};
  for (std::size_t index = 0U; index < digests.size(); ++index) {
    const auto& section = image.state_->sections[index];
    digests[index] = digest_bytes(section);
    if (digests[index] != image.state_->section_digests[index]) {
      throw LogicalArtifactError("V3 section descriptor digest mismatch");
    }
  }
  const LogicalSectionDigest joint = digest_bytes(joint_image_bytes(
      image.metadata(), image.state_->sections, digests));
  if (joint != image.state_->joint_digest ||
      joint != image.metadata().joint_sha256) {
    throw LogicalArtifactError("V3 joint identity mismatch");
  }
  if (image.metadata().logical_section_bytes != image.state_->sections[0].size() ||
      image.metadata().route_section_bytes != image.state_->sections[1].size() ||
      image.metadata().routing_section_bytes != image.state_->sections[2].size() ||
      image.metadata().logical_sha256 != digests[0] ||
      image.metadata().route_sha256 != digests[1] ||
      image.metadata().routing_sha256 != digests[2]) {
    throw LogicalArtifactError("V3 section metadata mismatch");
  }
  std::vector<LogicalArtifactRouteBinding> route_bindings =
      decode_route_records(image.state_->sections[1]);
  if (route_bindings.empty()) {
    throw LogicalArtifactError("V3 route section is empty");
  }
  RoutingLayoutSnapshot routing = RoutingLayoutSnapshot::from_canonical_bytes(
      image.state_->sections[2]);
  if (routing.object_count() != corpus_snapshot->object_count() ||
      routing.vector_dim() != corpus_snapshot->vector_dim() ||
      routing.attr_dim() != corpus_snapshot->attr_dim() ||
      routing.block_count() != route_bindings.size() ||
      routing.digest() != image.metadata().routing_sha256 ||
      image.metadata().routing_section_bytes != image.state_->sections[2].size()) {
    throw LogicalArtifactError(
        "V3 routing layout identity/shape/count mismatch");
  }
  LogicalGraphSections logical_sections;
  LogicalGraphView logical_view;
  try {
    logical_sections = LogicalGraphSections::from_canonical_bytes(
        image.state_->sections[0]);
    logical_view = LogicalGraphView::open(logical_sections, bound_corpus);
  } catch (const std::exception& error) {
    throw LogicalArtifactError(std::string("V3 logical section failed open: ") +
                               error.what());
  }
  LogicalArtifactIndex output(corpus_snapshot, image, std::move(routing));
  output.logical_sections_ = std::move(logical_sections);
  output.logical_view_ = std::move(logical_view);
  output.route_bindings_ = route_bindings;
  const auto& graph_data = output.logical_sections_.data();
  for (std::size_t route_index = 0U;
       route_index < output.route_bindings_.size(); ++route_index) {
    const auto& route = output.route_bindings_[route_index];
    if (route.block_id >= output.routing_.block_count()) {
      throw LogicalArtifactError("V3 route block is outside routing layout");
    }
    auto block_it = std::find(graph_data.block_view_id.begin(),
                              graph_data.block_view_id.end(),
                              route.view_id);
    if (block_it == graph_data.block_view_id.end()) {
      throw LogicalArtifactError("V3 route view is outside logical graph");
    }
    const std::size_t graph_block =
        static_cast<std::size_t>(block_it - graph_data.block_view_id.begin());
    auto& mutable_route = route_bindings[route_index];
    mutable_route.level = graph_data.block_depth[graph_block];
    mutable_route.component_sort_key =
        graph_data.block_component_sort_key[graph_block];
    mutable_route.members = std::vector<std::int64_t>(
        section_slice(graph_data.member_ids, graph_data.block_member_ptr,
                      graph_block)
            .begin(),
        section_slice(graph_data.member_ids, graph_data.block_member_ptr,
                      graph_block)
            .end());
    mutable_route.entries = std::vector<std::int64_t>(
        section_slice(graph_data.entry_ids, graph_data.block_entry_ptr,
                      graph_block)
            .begin(),
        section_slice(graph_data.entry_ids, graph_data.block_entry_ptr,
                      graph_block)
            .end());
    mutable_route.anchor = mutable_route.members[
        graph_data.block_anchor_local_row[graph_block]];
    const auto children = section_slice(graph_data.binding_child_view_id,
                                        graph_data.block_binding_ptr,
                                        graph_block);
    mutable_route.children.assign(children.begin(), children.end());
    if (output.routing_.block_level(route.block_id) !=
        graph_data.block_depth[graph_block]) {
      throw LogicalArtifactError("V3 route/layout depth mismatch");
    }
  }
  output.route_bindings_ = route_bindings;
  verify_metadata(*output.corpus_snapshot_, output.image_,
                  output.logical_sections_, output.logical_view_,
                  output.routing_.digest(),
                  output.routing_.canonical_bytes().size());
  verify_route_contract(output.logical_sections_, output.route_bindings_);
  return output;
}

LogicalViewId LogicalArtifactIndex::route(std::uint64_t block_id) const {
  const auto* route_binding = find_route(route_bindings_, block_id);
  if (route_binding == nullptr) {
    throw LogicalArtifactError("block ID is outside the explicit route map");
  }
  return route_binding->view_id;
}

LogicalNeighbors LogicalArtifactIndex::neighbors(std::uint64_t block_id,
                                                 std::uint32_t root_local_row,
                                                 LogicalScratch& scratch) const {
  return logical_view_.neighbors(route(block_id), root_local_row, scratch);
}

LogicalSearchResultView LogicalArtifactIndex::search_into(
    std::uint64_t block_id, std::uint32_t root_local_row,
    std::uint64_t budget, LogicalSearchWorkspace& workspace) const {
  const LogicalViewId view_id = route(block_id);
  const auto members = logical_view_.members(view_id);
  if (root_local_row >= members.size()) {
    throw LogicalArtifactError("logical search root is outside routed view");
  }
  if (workspace.seen_epoch.size() != members.size()) {
    throw LogicalArtifactError(
        "logical search workspace must be prepared for this routed view");
  }
  ++workspace.epoch;
  if (workspace.epoch == 0U) {
    std::fill(workspace.seen_epoch.begin(), workspace.seen_epoch.end(), 0U);
    workspace.epoch = 1U;
  }
  workspace.queue.clear();
  workspace.output_object_ids.clear();
  workspace.queue.push_back(root_local_row);
  workspace.seen_epoch[root_local_row] = workspace.epoch;
  std::size_t head = 0U;
  std::uint64_t actual_budget = 0U;
  while (head < workspace.queue.size() && actual_budget < budget) {
    const std::uint32_t row = workspace.queue[head++];
    workspace.output_object_ids.push_back(members[row]);
    ++actual_budget;
    const LogicalNeighbors resolved =
        logical_view_.neighbors(view_id, row, workspace.logical_scratch);
    for (const std::uint32_t target : resolved.local_rows) {
      if (workspace.seen_epoch[target] != workspace.epoch) {
        workspace.seen_epoch[target] = workspace.epoch;
        workspace.queue.push_back(target);
      }
    }
  }
  return LogicalSearchResultView{
      ArrayView<std::int64_t>(workspace.output_object_ids.data(),
                              workspace.output_object_ids.size()),
      budget, actual_budget, 1U, false};
}

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
LogicalArtifactImage logical_artifact_image_corrupt_copy_for_testing(
    const LogicalArtifactImage& image, LogicalArtifactSectionKind section,
    std::vector<std::uint8_t> bytes, bool recompute_section_digest) {
  if (!image.state_) throw LogicalArtifactError("logical artifact image is empty");
  auto state = std::make_shared<LogicalArtifactImage::State>(*image.state_);
  const std::size_t index = static_cast<std::size_t>(section);
  if (index >= state->sections.size()) {
    throw LogicalArtifactError("unknown V3 section kind");
  }
  state->sections[index] = std::move(bytes);
  if (recompute_section_digest) {
    state->section_digests[index] = digest_bytes(state->sections[index]);
  }
  return LogicalArtifactImage(std::move(state));
}

std::vector<std::uint8_t> logical_artifact_section_bytes_for_testing(
    const LogicalArtifactImage& image, LogicalArtifactSectionKind section) {
  return image.section(section);
}
#endif

}  // namespace bounded_blowup

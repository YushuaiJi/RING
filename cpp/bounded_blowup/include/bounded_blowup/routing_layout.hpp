// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
// Immutable canonical routing-layout image for the native query seam.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "bounded_blowup/index_view.hpp"

namespace bounded_blowup {

class LogicalArtifactImage;
class LogicalArtifactIndex;
class LogicalQueryHandle;
class LogicalAnnWorkspace;
class VerifiedCorpusSnapshot;
struct NativeBulkBlockDescriptor;
struct NativeBulkBuildOptions;
class NativeBulkBuildResult;

using RoutingLayoutDigest = std::array<std::uint8_t, 32>;

inline constexpr std::uint32_t kRoutingLayoutSchemaVersion = 1U;

// The layout is deliberately opaque.  IndexView is accepted at exactly one
// factory seam and all subsequent users borrow only immutable, owned columns.
// In particular, there is no default constructor and no alias factory that
// could make an unvalidated route image look like a sealed one.
class RoutingLayoutSnapshot final {
 public:
  struct State;

  RoutingLayoutSnapshot(const RoutingLayoutSnapshot& other);
  RoutingLayoutSnapshot& operator=(const RoutingLayoutSnapshot& other);
  RoutingLayoutSnapshot(RoutingLayoutSnapshot&& other) noexcept;
  RoutingLayoutSnapshot& operator=(RoutingLayoutSnapshot&& other) noexcept;
  ~RoutingLayoutSnapshot();

  static RoutingLayoutSnapshot from_index_view(const IndexView& index);

 private:
  std::shared_ptr<const State> state_;

  explicit RoutingLayoutSnapshot(std::shared_ptr<const State> state);

  static RoutingLayoutSnapshot from_canonical_bytes(
      std::vector<std::uint8_t> canonical_bytes);
  std::vector<std::uint8_t> canonical_bytes() const;
  RoutingLayoutDigest digest() const noexcept;
  std::size_t block_count() const noexcept;
  std::uint16_t block_level(std::size_t block_id) const;
  std::uint64_t object_count() const noexcept;
  std::uint32_t vector_dim() const noexcept;
  std::uint16_t attr_dim() const noexcept;
  std::uint32_t entries_per_query() const noexcept;
  IndexView materialize() const noexcept;

  friend class LogicalArtifactImage;
  friend class LogicalArtifactIndex;
  friend class LogicalQueryHandle;
  friend class LogicalAnnWorkspace;
  friend class NativeBulkBuildResult;
  friend NativeBulkBuildResult native_bulk_build(
      VerifiedCorpusSnapshot, NativeBulkBlockDescriptor,
      RoutingLayoutSnapshot, const NativeBulkBuildOptions&);
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  friend std::vector<std::uint8_t> routing_layout_canonical_bytes_for_testing(
      const RoutingLayoutSnapshot&);
  friend RoutingLayoutSnapshot routing_layout_from_canonical_bytes_for_testing(
      std::vector<std::uint8_t>);
  friend RoutingLayoutDigest routing_layout_sha256_for_testing(
      const RoutingLayoutSnapshot&) noexcept;
#endif
};

// These helpers are intentionally test-only.  Production code obtains the
// bytes and digest through the immutable Artifact V3 image, never by opening a
// second route sidecar.  Keeping the codec seam behind the fixture macro also
// prevents a mutable/partially decoded route image from entering production
// symbols.
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
std::vector<std::uint8_t> routing_layout_canonical_bytes_for_testing(
    const RoutingLayoutSnapshot& snapshot);
RoutingLayoutSnapshot routing_layout_from_canonical_bytes_for_testing(
    std::vector<std::uint8_t> canonical_bytes);
RoutingLayoutDigest routing_layout_sha256_for_testing(
    const RoutingLayoutSnapshot& snapshot) noexcept;
#endif

}  // namespace bounded_blowup

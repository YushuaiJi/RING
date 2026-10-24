// SPDX-License-Identifier: MIT
#pragma once

#include "bounded_blowup/digest.hpp"
#include "bounded_blowup/index_view.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace bounded_blowup {

using Rank2Shape = std::array<std::uint64_t, 2>;

enum class VerifiedCorpusErrorCode : std::uint8_t {
  kInvalidClaim = 0,
  kUnsafePath,
  kOpenFailure,
  kNotRegularFile,
  kUnsealedCorpus,
  kMapFailure,
  kMalformedNpy,
  kNpyDescriptorMismatch,
  kNpyShapeMismatch,
  kCorpusChangedDuringScan,
  kClaimDigestMismatch,
  kNonFiniteCorpus,
  kRowCountMismatch,
};

class VerifiedCorpusError : public std::runtime_error {
 public:
  VerifiedCorpusError(VerifiedCorpusErrorCode code,
                      const std::string& message)
      : std::runtime_error(message), code_(code) {}

  VerifiedCorpusErrorCode code() const noexcept { return code_; }

 private:
  VerifiedCorpusErrorCode code_;
};

class UnsealedCorpus final : public VerifiedCorpusError {
 public:
  explicit UnsealedCorpus(const std::string& message)
      : VerifiedCorpusError(VerifiedCorpusErrorCode::kUnsealedCorpus,
                            message) {}
};

class NonFiniteCorpus final : public VerifiedCorpusError {
 public:
  explicit NonFiniteCorpus(const std::string& message)
      : VerifiedCorpusError(VerifiedCorpusErrorCode::kNonFiniteCorpus,
                            message) {}
};

class MalformedNpy final : public VerifiedCorpusError {
 public:
  explicit MalformedNpy(const std::string& message)
      : VerifiedCorpusError(VerifiedCorpusErrorCode::kMalformedNpy,
                            message) {}
};

// The values are deliberately observable.  A non-reportable test fixture is
// never silently promoted to a formal/reportable snapshot.
enum class SealStatus : std::uint8_t {
  kReportable = 0,
  kNonreportableShadow,
  kNonreportableTestOnly,
  kUnavailable,
};

struct FileStatIdentity {
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t mode = 0U;
  std::uint64_t size = 0U;
  std::int64_t mtime_seconds = 0;
  std::int64_t mtime_nanoseconds = 0;
  std::int64_t ctime_seconds = 0;
  std::int64_t ctime_nanoseconds = 0;
  std::uint64_t inode_version = 0U;
  bool has_inode_version = false;

  bool operator==(const FileStatIdentity& other) const noexcept;
  bool operator!=(const FileStatIdentity& other) const noexcept {
    return !(*this == other);
  }
};

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
// This capability token is only declared when the test target explicitly
// opts into the internal fixture seam.  Production callers cannot construct
// a non-reportable snapshot or supply arbitrary seal evidence.
struct VerifiedCorpusTestOnlyTag {
  explicit constexpr VerifiedCorpusTestOnlyTag() = default;
};
inline constexpr VerifiedCorpusTestOnlyTag kVerifiedCorpusTestOnly{};
#endif

// Explicit capability for an engineering-only real-data shadow.  It keeps
// all native file/hash/shape/finite checks, but can never satisfy the formal
// reportability gate.
struct VerifiedCorpusShadowOnlyTag {
  explicit constexpr VerifiedCorpusShadowOnlyTag() = default;
};
inline constexpr VerifiedCorpusShadowOnlyTag kVerifiedCorpusShadowOnly{};

struct CorpusAssetClaim {
  std::filesystem::path path;
  Digest256 sha256;
  Rank2Shape shape;

  CorpusAssetClaim(std::filesystem::path asset_path, Digest256 file_sha256,
                   Rank2Shape expected_shape);
  CorpusAssetClaim(std::filesystem::path asset_path,
                   std::string_view file_sha256, Rank2Shape expected_shape);

  static CorpusAssetClaim from_hex(std::filesystem::path asset_path,
                                   std::string_view file_sha256,
                                   Rank2Shape expected_shape) {
    return CorpusAssetClaim(std::move(asset_path), file_sha256,
                            expected_shape);
  }

  const Digest256& expected_file_sha256() const noexcept { return sha256; }
  const Digest256& expected_sha256() const noexcept { return sha256; }
  const Rank2Shape& expected_shape() const noexcept { return shape; }
};

struct VerifiedCorpusOpenRequest {
  std::filesystem::path corpus_root;
  CorpusAssetClaim vectors;
  CorpusAssetClaim attrs_raw;

  VerifiedCorpusOpenRequest(std::filesystem::path root,
                            CorpusAssetClaim vector_claim,
                            CorpusAssetClaim attrs_claim);
};

struct VerifiedCorpusState;
class VerifiedCorpusSnapshot;

class VerifiedArrayIdentity final {
 public:
  VerifiedArrayIdentity(const VerifiedArrayIdentity&) = default;
  VerifiedArrayIdentity& operator=(const VerifiedArrayIdentity&) = default;

  const Digest256& file_sha256() const noexcept { return file_sha256_; }
  const Digest256& file_digest() const noexcept { return file_sha256_; }
  const Digest256& full_file_sha256() const noexcept { return file_sha256_; }
  const Digest256& payload_sha256() const noexcept { return payload_sha256_; }
  const Digest256& payload_digest() const noexcept { return payload_sha256_; }
  const Digest256& content_sha256() const noexcept {
    return canonical_content_sha256_;
  }
  const Digest256& canonical_content_sha256() const noexcept {
    return canonical_content_sha256_;
  }
  const Digest256& content_digest() const noexcept {
    return canonical_content_sha256_;
  }
  const Rank2Shape& shape() const noexcept { return shape_; }
  std::uint64_t rows() const noexcept { return shape_[0]; }
  std::uint64_t columns() const noexcept { return shape_[1]; }
  const std::string& dtype() const noexcept { return dtype_; }
  const std::string& byte_order() const noexcept { return byte_order_; }
  char order() const noexcept { return order_; }
  std::uint64_t payload_offset() const noexcept { return payload_offset_; }
  std::uint64_t file_bytes() const noexcept { return file_bytes_; }
  std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }
  SealStatus seal_status() const noexcept { return seal_status_; }
  bool reportable() const noexcept {
    return seal_status_ == SealStatus::kReportable;
  }
  bool is_reportable() const noexcept { return reportable(); }

 private:
  friend class VerifiedCorpusSnapshot;
  VerifiedArrayIdentity(Digest256 file_sha256, Digest256 payload_sha256,
                        Digest256 canonical_content_sha256, Rank2Shape shape,
                        std::string dtype, std::string byte_order, char order,
                        std::uint64_t payload_offset,
                        std::uint64_t file_bytes,
                        std::uint64_t payload_bytes, SealStatus seal_status);

  Digest256 file_sha256_;
  Digest256 payload_sha256_;
  Digest256 canonical_content_sha256_;
  Rank2Shape shape_{};
  std::string dtype_;
  std::string byte_order_;
  char order_ = 'C';
  std::uint64_t payload_offset_ = 0U;
  std::uint64_t file_bytes_ = 0U;
  std::uint64_t payload_bytes_ = 0U;
  SealStatus seal_status_ = SealStatus::kUnavailable;
};

class VerifiedCorpusIdentity final {
 public:
  VerifiedCorpusIdentity(const VerifiedCorpusIdentity&) = default;
  VerifiedCorpusIdentity& operator=(const VerifiedCorpusIdentity&) = default;

  const VerifiedArrayIdentity& vectors() const noexcept { return vectors_; }
  const VerifiedArrayIdentity& attrs_raw() const noexcept { return attrs_raw_; }
  const VerifiedArrayIdentity& vector_identity() const noexcept {
    return vectors_;
  }
  const VerifiedArrayIdentity& attrs_raw_identity() const noexcept {
    return attrs_raw_;
  }
  const Digest256& canonical_sha256() const noexcept {
    return canonical_sha256_;
  }
  const Digest256& canonical_sha() const noexcept { return canonical_sha256_; }
  const Digest256& sha256() const noexcept { return canonical_sha256_; }
  const Digest256& corpus_sha256() const noexcept { return canonical_sha256_; }
  std::uint64_t object_count() const noexcept { return object_count_; }
  std::uint64_t vector_dim() const noexcept { return vector_dim_; }
  std::uint64_t attr_dim() const noexcept { return attr_dim_; }
  SealStatus vectors_seal_status() const noexcept {
    return vectors_.seal_status();
  }
  SealStatus attrs_raw_seal_status() const noexcept {
    return attrs_raw_.seal_status();
  }
  bool reportable() const noexcept {
    return vectors_.reportable() && attrs_raw_.reportable();
  }
  bool is_reportable() const noexcept { return reportable(); }

 private:
  friend class VerifiedCorpusSnapshot;
  VerifiedCorpusIdentity(VerifiedArrayIdentity vectors,
                         VerifiedArrayIdentity attrs_raw,
                         Digest256 canonical_sha256,
                         std::uint64_t object_count,
                         std::uint64_t vector_dim, std::uint64_t attr_dim)
      : vectors_(std::move(vectors)),
        attrs_raw_(std::move(attrs_raw)),
        canonical_sha256_(std::move(canonical_sha256)),
        object_count_(object_count),
        vector_dim_(vector_dim),
        attr_dim_(attr_dim) {}

  VerifiedArrayIdentity vectors_;
  VerifiedArrayIdentity attrs_raw_;
  Digest256 canonical_sha256_;
  std::uint64_t object_count_ = 0U;
  std::uint64_t vector_dim_ = 0U;
  std::uint64_t attr_dim_ = 0U;
};

// The sole production constructor is open().  Copies retain the immutable
// native state and therefore keep both descriptors and mappings alive.
class VerifiedCorpusSnapshot final {
 public:
  static VerifiedCorpusSnapshot open(const VerifiedCorpusOpenRequest& request);
  static VerifiedCorpusSnapshot open_shadow(
      const VerifiedCorpusOpenRequest& request,
      VerifiedCorpusShadowOnlyTag shadow_only);
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  static VerifiedCorpusSnapshot open_for_testing(
      const VerifiedCorpusOpenRequest& request,
      VerifiedCorpusTestOnlyTag test_only);
#endif

  VerifiedCorpusSnapshot(const VerifiedCorpusSnapshot&) = default;
  VerifiedCorpusSnapshot& operator=(const VerifiedCorpusSnapshot&) = default;
  VerifiedCorpusSnapshot(VerifiedCorpusSnapshot&&) = delete;
  VerifiedCorpusSnapshot& operator=(VerifiedCorpusSnapshot&&) = delete;
  ~VerifiedCorpusSnapshot() = default;

  const VerifiedCorpusIdentity& identity() const noexcept;
  ArrayView<float> vectors() const noexcept;
  ArrayView<float> vector_view() const noexcept { return vectors(); }
  ArrayView<float> attrs_raw() const noexcept;
  ArrayView<float> attrs_raw_view() const noexcept { return attrs_raw(); }
  ArrayView<float> attrs() const noexcept { return attrs_raw(); }
  std::uint64_t object_count() const noexcept;
  std::uint64_t vector_dim() const noexcept;
  std::uint64_t attr_dim() const noexcept;
  bool reportable() const noexcept;
  bool is_reportable() const noexcept { return reportable(); }
  SealStatus vectors_seal_status() const noexcept;
  SealStatus attrs_raw_seal_status() const noexcept;

 private:
  static VerifiedCorpusSnapshot open_impl(
      const VerifiedCorpusOpenRequest& request);
  static VerifiedCorpusSnapshot open_impl_shadow(
      const VerifiedCorpusOpenRequest& request);
#if defined(RFBB_VERIFIED_CORPUS_TESTING)
  static VerifiedCorpusSnapshot open_impl_for_testing(
      const VerifiedCorpusOpenRequest& request);
#endif

  explicit VerifiedCorpusSnapshot(std::shared_ptr<const VerifiedCorpusState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<const VerifiedCorpusState> state_;
};

}  // namespace bounded_blowup

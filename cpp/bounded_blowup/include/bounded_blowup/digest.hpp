// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace bounded_blowup {

class DigestError : public std::invalid_argument {
 public:
  explicit DigestError(const std::string& message)
      : std::invalid_argument(message) {}
};

// A SHA-256 value whose textual representation is always exactly 64 lowercase
// hexadecimal characters.  The default constructor is deliberately absent so
// callers cannot accidentally use an uninitialised identity value.
class Digest256 final {
 public:
  static Digest256 parse(std::string_view lowercase_hex64);
  static Digest256 from_hex(std::string_view lowercase_hex64) {
    return parse(lowercase_hex64);
  }
  static Digest256 parse_hex(std::string_view lowercase_hex64) {
    return parse(lowercase_hex64);
  }
  static bool try_parse(std::string_view lowercase_hex64,
                        Digest256* output) noexcept;
  static Digest256 from_bytes(
      const std::array<std::uint8_t, 32>& binary_bytes);

  std::string hex() const;
  std::string to_hex() const { return hex(); }
  const std::array<std::uint8_t, 32>& bytes() const noexcept { return bytes_; }
  const std::array<std::uint8_t, 32>& binary() const noexcept { return bytes_; }
  std::string str() const { return hex(); }
  const std::uint8_t* data() const noexcept { return bytes_.data(); }

  bool operator==(const Digest256& other) const noexcept {
    return bytes_ == other.bytes_;
  }
  bool operator!=(const Digest256& other) const noexcept {
    return !(*this == other);
  }
  bool operator<(const Digest256& other) const noexcept {
    return bytes_ < other.bytes_;
  }

 private:
  explicit Digest256(const std::array<std::uint8_t, 32>& binary_bytes)
      : bytes_(binary_bytes) {}

  std::array<std::uint8_t, 32> bytes_{};
};

namespace digest_detail {

// Incremental SHA-256 is kept as a small native seam so corpus hashing can
// feed one mmap scan without allocating a second copy of the file.
class Sha256 {
 public:
  Sha256();

  void update(const std::uint8_t* data, std::size_t size);
  Digest256 finish();

 private:
  void transform(const std::array<std::uint8_t, 64>& block);

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0U;
  std::uint64_t total_bytes_ = 0U;
};

}  // namespace digest_detail

Digest256 sha256_bytes(const std::uint8_t* data, std::size_t size);
Digest256 sha256_bytes(std::string_view bytes);

}  // namespace bounded_blowup

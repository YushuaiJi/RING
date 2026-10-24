// SPDX-License-Identifier: LicenseRef-PVLDB2027-RFANN-Research
#include "bounded_blowup/digest.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace bounded_blowup {
namespace {

constexpr char kHex[] = "0123456789abcdef";

std::uint8_t hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  throw DigestError("Digest256 requires exactly lowercase hexadecimal");
}

// Keeping the normative table in one constexpr object makes the transform
// easy to audit and avoids a dependency on a platform crypto library.
constexpr std::array<std::uint32_t, 64> kSha256RoundConstants = {
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

constexpr std::array<std::uint32_t, 8> kSha256Initial = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};

std::uint32_t rotate_right(std::uint32_t value, unsigned amount) noexcept {
  return (value >> amount) | (value << (32U - amount));
}

}  // namespace

Digest256 Digest256::parse(std::string_view lowercase_hex64) {
  if (lowercase_hex64.size() != 64U) {
    throw DigestError("Digest256 requires exactly 64 lowercase hexadecimal characters");
  }
  std::array<std::uint8_t, 32> binary{};
  for (std::size_t index = 0U; index < binary.size(); ++index) {
    const std::uint8_t high = hex_nibble(lowercase_hex64[index * 2U]);
    const std::uint8_t low = hex_nibble(lowercase_hex64[index * 2U + 1U]);
    binary[index] = static_cast<std::uint8_t>((high << 4U) | low);
  }
  return Digest256(binary);
}

bool Digest256::try_parse(std::string_view lowercase_hex64,
                          Digest256* output) noexcept {
  if (output == nullptr) return false;
  try {
    *output = parse(lowercase_hex64);
    return true;
  } catch (...) {
    return false;
  }
}

Digest256 Digest256::from_bytes(
    const std::array<std::uint8_t, 32>& binary_bytes) {
  return Digest256(binary_bytes);
}

std::string Digest256::hex() const {
  std::string output;
  output.resize(64U);
  for (std::size_t index = 0U; index < bytes_.size(); ++index) {
    output[index * 2U] = kHex[bytes_[index] >> 4U];
    output[index * 2U + 1U] = kHex[bytes_[index] & 0x0fU];
  }
  return output;
}

namespace digest_detail {

Sha256::Sha256() : state_(kSha256Initial) {}

void Sha256::update(const std::uint8_t* data, std::size_t size) {
  if (size == 0U) return;
  if (data == nullptr) throw std::invalid_argument("SHA-256 data is null");
  if (size > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
    throw std::length_error("SHA-256 input is too large");
  }
  total_bytes_ += static_cast<std::uint64_t>(size);
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

Digest256 Sha256::finish() {
  const std::uint64_t bit_length = total_bytes_ * 8U;
  block_[block_size_++] = 0x80U;
  if (block_size_ > 56U) {
    while (block_size_ < block_.size()) block_[block_size_++] = 0U;
    transform(block_);
    block_size_ = 0U;
  }
  while (block_size_ < 56U) block_[block_size_++] = 0U;
  for (unsigned index = 0U; index < 8U; ++index) {
    block_[56U + index] = static_cast<std::uint8_t>(
        bit_length >> (56U - 8U * static_cast<std::uint64_t>(index)));
  }
  transform(block_);

  std::array<std::uint8_t, 32> output{};
  for (unsigned word = 0U; word < state_.size(); ++word) {
    for (unsigned byte = 0U; byte < 4U; ++byte) {
      output[word * 4U + byte] = static_cast<std::uint8_t>(
          state_[word] >> (24U - 8U * static_cast<std::uint32_t>(byte)));
    }
  }
  return Digest256::from_bytes(output);
}

void Sha256::transform(const std::array<std::uint8_t, 64>& block) {
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
                                kSha256RoundConstants[round] + schedule[round];
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

}  // namespace digest_detail

Digest256 sha256_bytes(const std::uint8_t* data, std::size_t size) {
  digest_detail::Sha256 sha;
  sha.update(data, size);
  return sha.finish();
}

Digest256 sha256_bytes(std::string_view bytes) {
  return sha256_bytes(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                      bytes.size());
}

}  // namespace bounded_blowup

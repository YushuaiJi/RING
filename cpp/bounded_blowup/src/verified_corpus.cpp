// SPDX-License-Identifier: MIT
#include "bounded_blowup/verified_corpus.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/fs.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#if __has_include(<linux/openat2.h>)
#include <linux/openat2.h>
#endif
#endif

namespace bounded_blowup {

namespace {

constexpr std::uint32_t kVerifiedCorpusSchemaVersion = 1U;
constexpr char kVerifiedCorpusIdentityPrefixBytes[] =
    "RFBB-VERIFIED-CORPUS-V1\0";
constexpr std::string_view kVerifiedCorpusIdentityPrefix(
    kVerifiedCorpusIdentityPrefixBytes,
    sizeof(kVerifiedCorpusIdentityPrefixBytes) - 1U);

std::string system_error_message(const char* operation,
                                 const std::filesystem::path& path) {
  std::ostringstream message;
  message << operation << " '" << path.string() << "': " << std::strerror(errno);
  return message.str();
}

[[noreturn]] void throw_error(VerifiedCorpusErrorCode code,
                              const std::string& message) {
  throw VerifiedCorpusError(code, message);
}

[[noreturn]] void throw_malformed(const std::string& message) {
  throw MalformedNpy(message);
}

std::uint64_t checked_add_u64(std::uint64_t left, std::uint64_t right,
                              const char* label) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw_malformed(std::string(label) + " overflows");
  }
  return left + right;
}

std::size_t to_size(std::uint64_t value, const char* label) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw_malformed(std::string(label) + " is not representable on this host");
  }
  return static_cast<std::size_t>(value);
}

std::uint16_t read_u16_le(const std::uint8_t* bytes, std::size_t offset) {
  return static_cast<std::uint16_t>(bytes[offset]) |
         static_cast<std::uint16_t>(bytes[offset + 1U] << 8U);
}

std::uint32_t read_u32_le(const std::uint8_t* bytes, std::size_t offset) {
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
         (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
         (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

bool is_header_space(char value) noexcept {
  return value == ' ' || value == '\t';
}

struct ParsedNpy {
  Rank2Shape shape{};
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  std::string dtype;
  std::string byte_order;
  char order = 'C';
};

class NpyHeaderParser final {
 public:
  explicit NpyHeaderParser(std::string_view text) : text_(text) {}

  ParsedNpy parse() {
    skip_space();
    require('{', "header dictionary");
    skip_space();
    bool have_descr = false;
    bool have_fortran_order = false;
    bool have_shape = false;
    ParsedNpy result;
    if (peek() == '}') throw_malformed("NPY header dictionary is empty");
    while (true) {
      const std::string key = parse_quoted("header key");
      skip_space();
      require(':', "header key separator");
      skip_space();
      if (key == "descr") {
        if (have_descr) throw_malformed("NPY header repeats key descr");
        result.dtype = parse_quoted("descr");
        have_descr = true;
      } else if (key == "fortran_order") {
        if (have_fortran_order) {
          throw_malformed("NPY header repeats key fortran_order");
        }
        result.order = parse_bool();
        have_fortran_order = true;
      } else if (key == "shape") {
        if (have_shape) throw_malformed("NPY header repeats key shape");
        result.shape = parse_rank2_shape();
        have_shape = true;
      } else {
        throw_malformed("NPY header contains unknown key " + key);
      }
      skip_space();
      if (peek() == ',') {
        ++position_;
        skip_space();
        if (peek() == '}') {
          ++position_;
          break;
        }
        continue;
      }
      if (peek() == '}') {
        ++position_;
        break;
      }
      throw_malformed("NPY header requires a comma or closing brace");
    }
    skip_space();
    if (position_ != text_.size()) {
      throw_malformed("NPY header contains trailing tokens");
    }
    if (!have_descr || !have_fortran_order || !have_shape) {
      throw_malformed("NPY header is missing a required key");
    }
    result.byte_order = result.dtype.size() >= 1U
                            ? std::string(1U, result.dtype.front())
                            : std::string();
    return result;
  }

 private:
  char peek() const noexcept {
    return position_ < text_.size() ? text_[position_] : '\0';
  }

  void skip_space() noexcept {
    while (position_ < text_.size() && is_header_space(text_[position_])) {
      ++position_;
    }
  }

  void require(char expected, const char* context) {
    if (peek() != expected) {
      throw_malformed(std::string("NPY ") + context + " has invalid syntax");
    }
    ++position_;
  }

  std::string parse_quoted(const char* context) {
    if (peek() != '\'') {
      throw_malformed(std::string("NPY ") + context + " must use single quotes");
    }
    ++position_;
    const std::size_t begin = position_;
    while (position_ < text_.size() && text_[position_] != '\'') {
      if (text_[position_] == '\\') {
        throw_malformed("NPY header strings may not contain escapes");
      }
      ++position_;
    }
    if (position_ == text_.size()) {
      throw_malformed(std::string("NPY ") + context + " is unterminated");
    }
    const std::string value(text_.substr(begin, position_ - begin));
    ++position_;
    return value;
  }

  char parse_bool() {
    constexpr std::string_view kTrue = "True";
    constexpr std::string_view kFalse = "False";
    if (text_.substr(position_, kTrue.size()) == kTrue) {
      position_ += kTrue.size();
      return 'F';
    }
    if (text_.substr(position_, kFalse.size()) == kFalse) {
      position_ += kFalse.size();
      return 'C';
    }
    throw_malformed("NPY fortran_order must be True or False");
  }

  std::uint64_t parse_uint(const char* label) {
    const std::size_t begin = position_;
    while (position_ < text_.size() && text_[position_] >= '0' &&
           text_[position_] <= '9') {
      ++position_;
    }
    if (begin == position_) {
      throw_malformed(std::string("NPY shape ") + label + " is not an integer");
    }
    std::uint64_t value = 0U;
    for (std::size_t index = begin; index < position_; ++index) {
      const std::uint64_t digit =
          static_cast<std::uint64_t>(text_[index] - '0');
      if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
        throw_malformed("NPY shape integer overflows");
      }
      value = value * 10U + digit;
    }
    return value;
  }

  Rank2Shape parse_rank2_shape() {
    require('(', "shape tuple");
    skip_space();
    const std::uint64_t rows = parse_uint("rows");
    skip_space();
    require(',', "shape rank separator");
    skip_space();
    const std::uint64_t columns = parse_uint("columns");
    skip_space();
    if (peek() == ',') {
      ++position_;
      skip_space();
    }
    require(')', "shape tuple");
    return {rows, columns};
  }

  std::string_view text_;
  std::size_t position_ = 0U;
};

ParsedNpy parse_npy(const std::uint8_t* bytes, std::uint64_t file_bytes) {
  if (bytes == nullptr || file_bytes < 10U) {
    throw_malformed("NPY file is shorter than its fixed prefix");
  }
  if (bytes[0] != 0x93U || bytes[1] != 'N' || bytes[2] != 'U' ||
      bytes[3] != 'M' || bytes[4] != 'P' || bytes[5] != 'Y') {
    throw_malformed("NPY magic is invalid");
  }
  const std::uint8_t major = bytes[6];
  const std::uint8_t minor = bytes[7];
  if (minor != 0U) throw_malformed("unsupported NPY minor version");
  std::uint64_t header_offset = 0U;
  std::uint64_t header_bytes = 0U;
  if (major == 1U) {
    header_offset = 10U;
    header_bytes = read_u16_le(bytes, 8U);
  } else if (major == 2U || major == 3U) {
    if (file_bytes < 12U) throw_malformed("NPY v2 prefix is truncated");
    header_offset = 12U;
    header_bytes = read_u32_le(bytes, 8U);
  } else {
    throw_malformed("unsupported NPY version");
  }
  if (header_offset > file_bytes || header_bytes > file_bytes - header_offset) {
    throw_malformed("NPY header is truncated");
  }
  if (header_bytes == 0U) throw_malformed("NPY header is empty");
  const std::size_t header_offset_size = to_size(header_offset, "NPY header offset");
  const std::size_t header_bytes_size = to_size(header_bytes, "NPY header size");
  const std::string_view header(
      reinterpret_cast<const char*>(bytes + header_offset_size),
      header_bytes_size);
  if (header.back() != '\n') throw_malformed("NPY header lacks a newline");
  ParsedNpy result =
      NpyHeaderParser(header.substr(0U, header.size() - 1U)).parse();
  if (result.dtype != "<f4") {
    throw_error(VerifiedCorpusErrorCode::kNpyDescriptorMismatch,
                "verified corpus requires dtype '<f4'");
  }
  if (result.order != 'C') {
    throw_error(VerifiedCorpusErrorCode::kNpyDescriptorMismatch,
                "verified corpus requires C-order NPY arrays");
  }
  result.byte_order = "<";
  if (result.shape[0] == 0U || result.shape[1] == 0U) {
    throw_error(VerifiedCorpusErrorCode::kNpyShapeMismatch,
                "verified corpus arrays must be non-empty rank-2 arrays");
  }
  if (result.shape[0] > std::numeric_limits<std::uint64_t>::max() /
                           result.shape[1]) {
    throw_malformed("NPY shape product overflows");
  }
  const std::uint64_t element_count = result.shape[0] * result.shape[1];
  if (element_count > std::numeric_limits<std::uint64_t>::max() / 4U) {
    throw_malformed("NPY payload size overflows");
  }
  result.payload_offset = checked_add_u64(header_offset, header_bytes,
                                          "NPY payload offset");
  result.payload_bytes = element_count * 4U;
  if (result.payload_offset % 16U != 0U ||
      result.payload_offset % alignof(float) != 0U) {
    throw_malformed("NPY payload offset is not aligned");
  }
  const std::uint64_t expected_file_bytes =
      checked_add_u64(result.payload_offset, result.payload_bytes,
                      "NPY file size");
  if (expected_file_bytes != file_bytes) {
    throw_malformed("NPY file has truncation or trailing bytes");
  }
  return result;
}

FileStatIdentity stat_identity_from_fd(int fd,
                                       const std::filesystem::path& path) {
  struct stat value {};
  if (::fstat(fd, &value) != 0) {
    throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                system_error_message("fstat", path));
  }
  if (!S_ISREG(value.st_mode)) {
    throw_error(VerifiedCorpusErrorCode::kNotRegularFile,
                "verified corpus asset is not a regular file: " +
                    path.string());
  }
  if (value.st_size <= 0) {
    throw_error(VerifiedCorpusErrorCode::kMalformedNpy,
                "verified corpus asset is empty: " + path.string());
  }

  FileStatIdentity result;
  result.device = static_cast<std::uint64_t>(value.st_dev);
  result.inode = static_cast<std::uint64_t>(value.st_ino);
  result.mode = static_cast<std::uint64_t>(value.st_mode & S_IFMT);
  result.size = static_cast<std::uint64_t>(value.st_size);
#if defined(__APPLE__)
  result.mtime_seconds = static_cast<std::int64_t>(value.st_mtimespec.tv_sec);
  result.mtime_nanoseconds =
      static_cast<std::int64_t>(value.st_mtimespec.tv_nsec);
  result.ctime_seconds = static_cast<std::int64_t>(value.st_ctimespec.tv_sec);
  result.ctime_nanoseconds =
      static_cast<std::int64_t>(value.st_ctimespec.tv_nsec);
#else
  result.mtime_seconds = static_cast<std::int64_t>(value.st_mtim.tv_sec);
  result.mtime_nanoseconds = static_cast<std::int64_t>(value.st_mtim.tv_nsec);
  result.ctime_seconds = static_cast<std::int64_t>(value.st_ctim.tv_sec);
  result.ctime_nanoseconds = static_cast<std::int64_t>(value.st_ctim.tv_nsec);
#endif
#if defined(__linux__) && defined(FS_IOC_GETVERSION)
  int version = 0;
  if (::ioctl(fd, FS_IOC_GETVERSION, &version) == 0 && version >= 0) {
    result.inode_version = static_cast<std::uint64_t>(version);
    result.has_inode_version = true;
  }
#endif
  return result;
}

struct UniqueFd final {
  int value = -1;

  UniqueFd() = default;
  explicit UniqueFd(int descriptor) : value(descriptor) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : value(other.value) {
    other.value = -1;
  }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      value = other.value;
      other.value = -1;
    }
    return *this;
  }
  ~UniqueFd() { reset(); }

  void reset(int descriptor = -1) noexcept {
    if (value >= 0) (void)::close(value);
    value = descriptor;
  }
  int release() noexcept {
    const int descriptor = value;
    value = -1;
    return descriptor;
  }
};

int directory_open_flags() noexcept {
  int flags = O_RDONLY | O_DIRECTORY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  return flags;
}

int asset_open_flags() noexcept {
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
#ifdef O_NONBLOCK
  flags |= O_NONBLOCK;
#endif
  return flags;
}

std::vector<std::string> safe_components(const std::filesystem::path& path,
                                         bool require_relative,
                                         const char* kind,
                                         bool require_nonempty = true) {
  if (path.empty()) {
    throw_error(VerifiedCorpusErrorCode::kInvalidClaim,
                std::string("verified corpus ") + kind + " path is empty");
  }
  const std::string text = path.string();
  if (text.find('\0') != std::string::npos) {
    throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                std::string("verified corpus ") + kind +
                    " path contains an embedded NUL");
  }
  if (require_relative && path.is_absolute()) {
    throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                std::string("verified corpus asset path must be relative: ") +
                    text);
  }
  std::vector<std::string> components;
  for (const auto& component : path) {
    if (component == path.root_name() || component == path.root_directory() ||
        component == ".") {
      continue;
    }
    if (component == ".." || component.empty()) {
      throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                  std::string("verified corpus rejects unsafe ") + kind +
                      " path: " + text);
    }
    components.push_back(component.string());
  }
  if (components.empty() && require_nonempty) {
    throw_error(VerifiedCorpusErrorCode::kInvalidClaim,
                std::string("verified corpus ") + kind +
                    " path names no file");
  }
  return components;
}

int open_corpus_root(const std::filesystem::path& root) {
  if (root.empty()) {
    throw_error(VerifiedCorpusErrorCode::kInvalidClaim,
                "verified corpus root path is empty");
  }
  const std::string root_text = root.string();
  if (root_text.find('\0') != std::string::npos) {
    throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                "verified corpus root path contains an embedded NUL");
  }
  UniqueFd current(::open(root.is_absolute() ? "/" : ".",
                          directory_open_flags()));
  if (current.value < 0) {
    throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                system_error_message("open corpus root", root));
  }
  const std::vector<std::string> components =
      safe_components(root, false, "root", false);
  for (std::size_t index = 0U; index < components.size(); ++index) {
    const std::string& component = components[index];
    int flags = directory_open_flags();
#if defined(__APPLE__) && defined(O_NOFOLLOW)
    // macOS exposes the system temporary directory through /var (and /tmp)
    // symlink aliases.  Permit only this fixed OS alias at the first root
    // component; all caller-controlled components remain O_NOFOLLOW.
    if (root.is_absolute() && index == 0U &&
        (component == "var" || component == "tmp")) {
      flags &= ~O_NOFOLLOW;
    }
#endif
    UniqueFd next(::openat(current.value, component.c_str(),
                           flags));
    if (next.value < 0) {
      if (errno == ELOOP || errno == ENOTDIR) {
        throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                    "verified corpus root traverses a symlink or non-directory: " +
                        root.string());
      }
      throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                  system_error_message("openat corpus root", root));
    }
    current = std::move(next);
  }
  return current.release();
}

int open_asset_fallback(int root_fd, const std::vector<std::string>& components,
                        const std::filesystem::path& path) {
  UniqueFd current(::dup(root_fd));
  if (current.value < 0) {
    throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                system_error_message("dup corpus root", path));
  }
  for (std::size_t index = 0U; index + 1U < components.size(); ++index) {
    UniqueFd next(::openat(current.value, components[index].c_str(),
                           directory_open_flags()));
    if (next.value < 0) {
      if (errno == ELOOP || errno == ENOTDIR) {
        throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                    "verified corpus asset traverses a symlink: " +
                        path.string());
      }
      throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                  system_error_message("openat corpus asset parent", path));
    }
    current = std::move(next);
  }
  const std::string& leaf = components.back();
  const int descriptor = ::openat(current.value, leaf.c_str(),
                                  asset_open_flags());
  if (descriptor < 0) {
    if (errno == ELOOP) {
      throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                  "verified corpus asset is a symlink: " + path.string());
    }
    throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                system_error_message("openat corpus asset", path));
  }
  return descriptor;
}

int open_asset_fd(int root_fd, const std::filesystem::path& path) {
  const std::vector<std::string> components = safe_components(path, true, "asset");
#if defined(__linux__) && defined(SYS_openat2) && defined(RESOLVE_BENEATH) && \
    defined(RESOLVE_NO_SYMLINKS) && defined(RESOLVE_NO_MAGICLINKS)
  struct open_how how {};
  how.flags = static_cast<std::uint64_t>(asset_open_flags());
  how.resolve = static_cast<std::uint64_t>(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS |
                                           RESOLVE_NO_MAGICLINKS);
  const int descriptor = static_cast<int>(::syscall(
      SYS_openat2, root_fd, path.c_str(), &how, sizeof(how)));
  if (descriptor >= 0) return descriptor;
  if (errno != ENOSYS && errno != EINVAL && errno != ENOTSUP) {
    if (errno == ELOOP || errno == EXDEV || errno == ENOTDIR) {
      throw_error(VerifiedCorpusErrorCode::kUnsafePath,
                  "verified corpus asset violates openat2 path constraints: " +
                      path.string());
    }
    throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                system_error_message("openat2 corpus asset", path));
  }
#endif
  return open_asset_fallback(root_fd, components, path);
}

struct NativeSealCheck final {
  SealStatus status = SealStatus::kUnavailable;
  std::string mechanism;
};

NativeSealCheck native_seal_check(int fd, const std::filesystem::path& path) {
  struct statvfs filesystem_info {};
  if (::fstatvfs(fd, &filesystem_info) != 0) {
    return {SealStatus::kUnavailable,
            "fstatvfs(fd) failed for formal seal check"};
  }
  (void)filesystem_info;
  // A read-only mount or immutable inode is useful observation, but this
  // stage has no independently accepted machine seal protocol.  Keep the
  // production result explicitly unavailable until external acceptance.
  return {SealStatus::kUnavailable,
          "formal machine seal unavailable for fd-bound asset " +
              path.string()};
}

struct HashResult {
  Digest256 full_file;
  Digest256 payload;
  Digest256 canonical_content;
  bool finite = true;
};

struct MappedAsset {
  int fd = -1;
  void* mapping = MAP_FAILED;
  std::size_t mapped_bytes = 0U;
  SealStatus seal_status = SealStatus::kUnavailable;
  ParsedNpy npy;
  Digest256 full_file;
  Digest256 payload;
  Digest256 canonical_content;
  ArrayView<float> view;

  MappedAsset(int descriptor, void* mapped, std::size_t bytes,
              SealStatus status, ParsedNpy parsed, HashResult hashes,
              ArrayView<float> values)
      : fd(descriptor),
        mapping(mapped),
        mapped_bytes(bytes),
        seal_status(status),
        npy(std::move(parsed)),
        full_file(std::move(hashes.full_file)),
        payload(std::move(hashes.payload)),
        canonical_content(std::move(hashes.canonical_content)),
        view(values) {}

  MappedAsset(const MappedAsset&) = delete;
  MappedAsset& operator=(const MappedAsset&) = delete;
  MappedAsset(MappedAsset&& other) noexcept
      : fd(other.fd),
        mapping(other.mapping),
        mapped_bytes(other.mapped_bytes),
        seal_status(other.seal_status),
        npy(std::move(other.npy)),
        full_file(std::move(other.full_file)),
        payload(std::move(other.payload)),
        canonical_content(std::move(other.canonical_content)),
        view(other.view) {
    other.fd = -1;
    other.mapping = MAP_FAILED;
    other.mapped_bytes = 0U;
    other.view = ArrayView<float>();
  }
  MappedAsset& operator=(MappedAsset&& other) noexcept {
    if (this != &other) {
      release();
      fd = other.fd;
      mapping = other.mapping;
      mapped_bytes = other.mapped_bytes;
      seal_status = other.seal_status;
      npy = std::move(other.npy);
      full_file = std::move(other.full_file);
      payload = std::move(other.payload);
      canonical_content = std::move(other.canonical_content);
      view = other.view;
      other.fd = -1;
      other.mapping = MAP_FAILED;
      other.mapped_bytes = 0U;
      other.view = ArrayView<float>();
    }
    return *this;
  }
  ~MappedAsset() { release(); }

  void release() noexcept {
    if (mapping != MAP_FAILED) {
      (void)::munmap(mapping, mapped_bytes);
      mapping = MAP_FAILED;
    }
    if (fd >= 0) {
      (void)::close(fd);
      fd = -1;
    }
  }
};

bool host_is_little_endian() noexcept {
  const std::uint16_t value = 1U;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1U;
}

float decode_little_endian_float(const std::array<std::uint8_t, 4>& bytes) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24U);
  std::uint32_t native_bits = bits;
  if (!host_is_little_endian()) {
    native_bits = ((bits & 0x000000ffU) << 24U) |
                  ((bits & 0x0000ff00U) << 8U) |
                  ((bits & 0x00ff0000U) >> 8U) |
                  ((bits & 0xff000000U) >> 24U);
  }
  float value = 0.0F;
  std::memcpy(&value, &native_bits, sizeof(value));
  return value;
}

std::string canonical_content_header(std::string_view role,
                                     const ParsedNpy& npy) {
  std::string schema;
  if (role == "vectors") {
    schema = "rfann-industrial-vectors-content-v1";
  } else if (role == "attrs_raw") {
    // This spelling is the existing industrial.py attrs-content schema;
    // attrs_raw is the native role name for that same canonical payload.
    schema = "rfann-industrial-attrs-content-v1";
  } else {
    throw_error(VerifiedCorpusErrorCode::kInvalidClaim,
                "unknown verified corpus canonical-content role");
  }
  return "{\"dtype\":\"<f4\",\"order\":\"C\",\"schema_version\":\"" +
         schema + "\",\"shape\":[" + std::to_string(npy.shape[0]) +
         "," + std::to_string(npy.shape[1]) + "]}\n";
}

HashResult hash_mapped_asset(const std::uint8_t* bytes, std::size_t file_bytes,
                             const ParsedNpy& npy, std::string_view role,
                             const FileStatIdentity& before, int fd,
                             const std::filesystem::path& path) {
  digest_detail::Sha256 full_hash;
  digest_detail::Sha256 payload_hash;
  digest_detail::Sha256 canonical_hash;
  const std::string canonical_header = canonical_content_header(role, npy);
  canonical_hash.update(
      reinterpret_cast<const std::uint8_t*>(canonical_header.data()),
      canonical_header.size());
  const std::size_t payload_begin = to_size(npy.payload_offset, "payload offset");
  const std::size_t payload_size = to_size(npy.payload_bytes, "payload size");
  const std::size_t payload_end = payload_begin + payload_size;
  std::array<std::uint8_t, 4> pending{};
  std::size_t pending_count = 0U;
  bool finite = true;
  constexpr std::size_t kChunkBytes = 1U << 20U;
  for (std::size_t offset = 0U; offset < file_bytes;) {
    const std::size_t count = std::min(kChunkBytes, file_bytes - offset);
    full_hash.update(bytes + offset, count);
    const std::size_t overlap_begin = std::max(offset, payload_begin);
    const std::size_t overlap_end = std::min(offset + count, payload_end);
    if (overlap_begin < overlap_end) {
      payload_hash.update(bytes + overlap_begin, overlap_end - overlap_begin);
      canonical_hash.update(bytes + overlap_begin, overlap_end - overlap_begin);
      for (std::size_t position = overlap_begin; position < overlap_end;
           ++position) {
        pending[pending_count++] = bytes[position];
        if (pending_count == pending.size()) {
          if (!std::isfinite(decode_little_endian_float(pending))) {
            finite = false;
          }
          pending_count = 0U;
        }
      }
    }
    offset += count;
  }
  if (pending_count != 0U) {
    throw_malformed("NPY payload does not contain complete float32 values");
  }
  const FileStatIdentity after = stat_identity_from_fd(fd, path);
  if (before != after) {
    throw_error(VerifiedCorpusErrorCode::kCorpusChangedDuringScan,
                "corpus asset changed while it was being scanned: " +
                    path.string());
  }
  HashResult result{full_hash.finish(), payload_hash.finish(),
                    canonical_hash.finish(), finite};
  return result;
}

MappedAsset map_and_verify_asset(int fd, const CorpusAssetClaim& claim,
                                 std::string_view role,
                                 SealStatus seal_status) {
  const FileStatIdentity before = stat_identity_from_fd(fd, claim.path);
  if (before.size >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw_error(VerifiedCorpusErrorCode::kOpenFailure,
                "verified corpus asset is too large to map: " +
                    claim.path.string());
  }
  const std::size_t mapped_bytes = static_cast<std::size_t>(before.size);
  void* mapping = ::mmap(nullptr, mapped_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    throw_error(VerifiedCorpusErrorCode::kMapFailure,
                system_error_message("mmap", claim.path));
  }
  try {
    const auto* mapped_bytes_ptr = static_cast<const std::uint8_t*>(mapping);
    const ParsedNpy npy = parse_npy(mapped_bytes_ptr, before.size);
    if (npy.shape != claim.shape) {
      throw_error(VerifiedCorpusErrorCode::kNpyShapeMismatch,
                  "NPY shape differs from the expected claim for " +
                      claim.path.string());
    }
    const HashResult hashes = hash_mapped_asset(
        mapped_bytes_ptr, mapped_bytes, npy, role, before, fd, claim.path);
    if (hashes.full_file != claim.sha256) {
      throw_error(VerifiedCorpusErrorCode::kClaimDigestMismatch,
                  "observed file SHA-256 differs from the claim for " +
                      claim.path.string());
    }
    if (!hashes.finite) {
      throw NonFiniteCorpus("verified corpus payload contains NaN or Inf: " +
                           claim.path.string());
    }
    const auto* payload = mapped_bytes_ptr +
                          to_size(npy.payload_offset, "payload offset");
    if (!host_is_little_endian()) {
      throw_error(VerifiedCorpusErrorCode::kNpyDescriptorMismatch,
                  "little-endian float32 views require a little-endian host");
    }
    const std::uint64_t elements = npy.shape[0] * npy.shape[1];
    const ArrayView<float> view(
        reinterpret_cast<const float*>(payload),
        to_size(elements, "element count"));
    return MappedAsset(fd, mapping, mapped_bytes, seal_status, npy,
                       std::move(hashes), view);
  } catch (...) {
    (void)::munmap(mapping, mapped_bytes);
    throw;
  }
}

MappedAsset open_and_verify_asset_formal(int root_fd,
                                         const CorpusAssetClaim& claim,
                                         std::string_view role) {
  const int fd = open_asset_fd(root_fd, claim.path);
  try {
    const NativeSealCheck seal = native_seal_check(fd, claim.path);
    if (seal.status != SealStatus::kReportable) {
      throw UnsealedCorpus(
          "verified corpus asset lacks an accepted formal read-only seal: " +
          claim.path.string() + " (" + seal.mechanism + ")");
    }
    return map_and_verify_asset(fd, claim, role, SealStatus::kReportable);
  } catch (...) {
    (void)::close(fd);
    throw;
  }
}

MappedAsset open_and_verify_asset_shadow(int root_fd,
                                         const CorpusAssetClaim& claim,
                                         std::string_view role) {
  const int fd = open_asset_fd(root_fd, claim.path);
  try {
    return map_and_verify_asset(fd, claim, role,
                                SealStatus::kNonreportableShadow);
  } catch (...) {
    (void)::close(fd);
    throw;
  }
}

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
MappedAsset open_and_verify_asset_test_only(int root_fd,
                                            const CorpusAssetClaim& claim,
                                            std::string_view role) {
  const int fd = open_asset_fd(root_fd, claim.path);
  try {
    return map_and_verify_asset(fd, claim, role,
                                SealStatus::kNonreportableTestOnly);
  } catch (...) {
    (void)::close(fd);
    throw;
  }
}
#endif

void append_u32_le(std::vector<std::uint8_t>* bytes, std::uint32_t value) {
  for (unsigned index = 0U; index < 4U; ++index) {
    bytes->push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

void append_u64_le(std::vector<std::uint8_t>* bytes, std::uint64_t value) {
  for (unsigned index = 0U; index < 8U; ++index) {
    bytes->push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

void append_array_record(std::vector<std::uint8_t>* bytes,
                         const VerifiedArrayIdentity& identity) {
  bytes->insert(bytes->end(), identity.file_sha256().bytes().begin(),
                identity.file_sha256().bytes().end());
  bytes->insert(bytes->end(), identity.payload_sha256().bytes().begin(),
                identity.payload_sha256().bytes().end());
  bytes->insert(bytes->end(), identity.canonical_content_sha256().bytes().begin(),
                identity.canonical_content_sha256().bytes().end());
  append_u64_le(bytes, identity.shape()[0]);
  append_u64_le(bytes, identity.shape()[1]);
  const std::string& dtype = identity.dtype();
  bytes->insert(bytes->end(), dtype.begin(), dtype.end());
  bytes->push_back(0U);
  bytes->push_back(static_cast<std::uint8_t>(identity.order()));
  bytes->push_back(identity.byte_order().empty()
                       ? 0U
                       : static_cast<std::uint8_t>(identity.byte_order()[0]));
  append_u64_le(bytes, identity.payload_offset());
  append_u64_le(bytes, identity.file_bytes());
  append_u64_le(bytes, identity.payload_bytes());
}

Digest256 canonical_corpus_sha256(const VerifiedArrayIdentity& vectors,
                                  const VerifiedArrayIdentity& attrs_raw,
                                  std::uint64_t object_count,
                                  std::uint64_t vector_dim,
                                  std::uint64_t attr_dim) {
  std::vector<std::uint8_t> encoded;
  encoded.insert(encoded.end(), kVerifiedCorpusIdentityPrefix.begin(),
                 kVerifiedCorpusIdentityPrefix.end());
  append_u32_le(&encoded, kVerifiedCorpusSchemaVersion);
  append_array_record(&encoded, vectors);
  append_array_record(&encoded, attrs_raw);
  append_u64_le(&encoded, object_count);
  append_u64_le(&encoded, vector_dim);
  append_u64_le(&encoded, attr_dim);
  return sha256_bytes(encoded.data(), encoded.size());
}

}  // namespace

bool FileStatIdentity::operator==(const FileStatIdentity& other) const noexcept {
  return device == other.device && inode == other.inode && mode == other.mode &&
         size == other.size && mtime_seconds == other.mtime_seconds &&
         mtime_nanoseconds == other.mtime_nanoseconds &&
         ctime_seconds == other.ctime_seconds &&
         ctime_nanoseconds == other.ctime_nanoseconds &&
         has_inode_version == other.has_inode_version &&
         (!has_inode_version || inode_version == other.inode_version);
}

CorpusAssetClaim::CorpusAssetClaim(std::filesystem::path asset_path,
                                   Digest256 file_sha256,
                                   Rank2Shape expected_shape)
    : path(std::move(asset_path)),
      sha256(std::move(file_sha256)),
      shape(expected_shape) {}

CorpusAssetClaim::CorpusAssetClaim(std::filesystem::path asset_path,
                                   std::string_view file_sha256,
                                   Rank2Shape expected_shape)
    : CorpusAssetClaim(std::move(asset_path), Digest256::parse(file_sha256),
                       expected_shape) {}

VerifiedCorpusOpenRequest::VerifiedCorpusOpenRequest(
    std::filesystem::path root, CorpusAssetClaim vector_claim,
    CorpusAssetClaim attrs_claim)
    : corpus_root(std::move(root)),
      vectors(std::move(vector_claim)),
      attrs_raw(std::move(attrs_claim)) {}

VerifiedArrayIdentity::VerifiedArrayIdentity(
    Digest256 file_sha256, Digest256 payload_sha256,
    Digest256 canonical_content_sha256, Rank2Shape shape, std::string dtype,
    std::string byte_order, char order, std::uint64_t payload_offset,
    std::uint64_t file_bytes, std::uint64_t payload_bytes,
    SealStatus seal_status)
    : file_sha256_(std::move(file_sha256)),
      payload_sha256_(std::move(payload_sha256)),
      canonical_content_sha256_(std::move(canonical_content_sha256)),
      shape_(shape),
      dtype_(std::move(dtype)),
      byte_order_(std::move(byte_order)),
      order_(order),
      payload_offset_(payload_offset),
      file_bytes_(file_bytes),
      payload_bytes_(payload_bytes),
      seal_status_(seal_status) {}

struct VerifiedCorpusState {
  UniqueFd corpus_root;
  MappedAsset vectors;
  MappedAsset attrs_raw;
  VerifiedCorpusIdentity identity;

  VerifiedCorpusState(UniqueFd root_descriptor, MappedAsset vector_asset,
                      MappedAsset attrs_asset,
                      VerifiedCorpusIdentity corpus_identity)
      : corpus_root(std::move(root_descriptor)),
        vectors(std::move(vector_asset)),
        attrs_raw(std::move(attrs_asset)),
        identity(std::move(corpus_identity)) {}
};

VerifiedCorpusSnapshot VerifiedCorpusSnapshot::open(
    const VerifiedCorpusOpenRequest& request) {
  return open_impl(request);
}

VerifiedCorpusSnapshot VerifiedCorpusSnapshot::open_impl(
    const VerifiedCorpusOpenRequest& request) {
  UniqueFd corpus_root(open_corpus_root(request.corpus_root));
  MappedAsset vectors = open_and_verify_asset_formal(
      corpus_root.value, request.vectors, "vectors");
  MappedAsset attrs_raw = open_and_verify_asset_formal(
      corpus_root.value, request.attrs_raw, "attrs_raw");
  if (vectors.npy.shape[0] != attrs_raw.npy.shape[0]) {
    throw_error(VerifiedCorpusErrorCode::kRowCountMismatch,
                "vectors and attrs_raw have different row counts");
  }
  const std::uint64_t object_count = vectors.npy.shape[0];
  const std::uint64_t vector_dim = vectors.npy.shape[1];
  const std::uint64_t attr_dim = attrs_raw.npy.shape[1];
  VerifiedArrayIdentity vectors_identity(
      vectors.full_file, vectors.payload, vectors.canonical_content,
      vectors.npy.shape, vectors.npy.dtype, vectors.npy.byte_order,
      vectors.npy.order, vectors.npy.payload_offset,
      static_cast<std::uint64_t>(vectors.mapped_bytes),
      vectors.npy.payload_bytes, vectors.seal_status);
  VerifiedArrayIdentity attrs_identity(
      attrs_raw.full_file, attrs_raw.payload, attrs_raw.canonical_content,
      attrs_raw.npy.shape, attrs_raw.npy.dtype, attrs_raw.npy.byte_order,
      attrs_raw.npy.order, attrs_raw.npy.payload_offset,
      static_cast<std::uint64_t>(attrs_raw.mapped_bytes),
      attrs_raw.npy.payload_bytes, attrs_raw.seal_status);
  const Digest256 corpus_sha = canonical_corpus_sha256(
      vectors_identity, attrs_identity, object_count, vector_dim, attr_dim);
  VerifiedCorpusIdentity identity(
      std::move(vectors_identity), std::move(attrs_identity), corpus_sha,
      object_count, vector_dim, attr_dim);
  std::shared_ptr<const VerifiedCorpusState> state =
      std::make_shared<const VerifiedCorpusState>(
          std::move(corpus_root), std::move(vectors), std::move(attrs_raw),
          std::move(identity));
  return VerifiedCorpusSnapshot(std::move(state));
}

VerifiedCorpusSnapshot VerifiedCorpusSnapshot::open_shadow(
    const VerifiedCorpusOpenRequest& request,
    VerifiedCorpusShadowOnlyTag /*shadow_only*/) {
  return open_impl_shadow(request);
}

VerifiedCorpusSnapshot VerifiedCorpusSnapshot::open_impl_shadow(
    const VerifiedCorpusOpenRequest& request) {
  UniqueFd corpus_root(open_corpus_root(request.corpus_root));
  MappedAsset vectors = open_and_verify_asset_shadow(
      corpus_root.value, request.vectors, "vectors");
  MappedAsset attrs_raw = open_and_verify_asset_shadow(
      corpus_root.value, request.attrs_raw, "attrs_raw");
  if (vectors.npy.shape[0] != attrs_raw.npy.shape[0]) {
    throw_error(VerifiedCorpusErrorCode::kRowCountMismatch,
                "vectors and attrs_raw have different row counts");
  }
  const std::uint64_t object_count = vectors.npy.shape[0];
  const std::uint64_t vector_dim = vectors.npy.shape[1];
  const std::uint64_t attr_dim = attrs_raw.npy.shape[1];
  VerifiedArrayIdentity vectors_identity(
      vectors.full_file, vectors.payload, vectors.canonical_content,
      vectors.npy.shape, vectors.npy.dtype, vectors.npy.byte_order,
      vectors.npy.order, vectors.npy.payload_offset,
      static_cast<std::uint64_t>(vectors.mapped_bytes),
      vectors.npy.payload_bytes, vectors.seal_status);
  VerifiedArrayIdentity attrs_identity(
      attrs_raw.full_file, attrs_raw.payload, attrs_raw.canonical_content,
      attrs_raw.npy.shape, attrs_raw.npy.dtype, attrs_raw.npy.byte_order,
      attrs_raw.npy.order, attrs_raw.npy.payload_offset,
      static_cast<std::uint64_t>(attrs_raw.mapped_bytes),
      attrs_raw.npy.payload_bytes, attrs_raw.seal_status);
  const Digest256 corpus_sha = canonical_corpus_sha256(
      vectors_identity, attrs_identity, object_count, vector_dim, attr_dim);
  VerifiedCorpusIdentity identity(
      std::move(vectors_identity), std::move(attrs_identity), corpus_sha,
      object_count, vector_dim, attr_dim);
  std::shared_ptr<const VerifiedCorpusState> state =
      std::make_shared<const VerifiedCorpusState>(
          std::move(corpus_root), std::move(vectors), std::move(attrs_raw),
          std::move(identity));
  return VerifiedCorpusSnapshot(std::move(state));
}

#if defined(RFBB_VERIFIED_CORPUS_TESTING)
VerifiedCorpusSnapshot VerifiedCorpusSnapshot::open_impl_for_testing(
    const VerifiedCorpusOpenRequest& request) {
  UniqueFd corpus_root(open_corpus_root(request.corpus_root));
  MappedAsset vectors = open_and_verify_asset_test_only(
      corpus_root.value, request.vectors, "vectors");
  MappedAsset attrs_raw = open_and_verify_asset_test_only(
      corpus_root.value, request.attrs_raw, "attrs_raw");
  if (vectors.npy.shape[0] != attrs_raw.npy.shape[0]) {
    throw_error(VerifiedCorpusErrorCode::kRowCountMismatch,
                "vectors and attrs_raw have different row counts");
  }
  const std::uint64_t object_count = vectors.npy.shape[0];
  const std::uint64_t vector_dim = vectors.npy.shape[1];
  const std::uint64_t attr_dim = attrs_raw.npy.shape[1];
  VerifiedArrayIdentity vectors_identity(
      vectors.full_file, vectors.payload, vectors.canonical_content,
      vectors.npy.shape, vectors.npy.dtype, vectors.npy.byte_order,
      vectors.npy.order, vectors.npy.payload_offset,
      static_cast<std::uint64_t>(vectors.mapped_bytes),
      vectors.npy.payload_bytes, vectors.seal_status);
  VerifiedArrayIdentity attrs_identity(
      attrs_raw.full_file, attrs_raw.payload, attrs_raw.canonical_content,
      attrs_raw.npy.shape, attrs_raw.npy.dtype, attrs_raw.npy.byte_order,
      attrs_raw.npy.order, attrs_raw.npy.payload_offset,
      static_cast<std::uint64_t>(attrs_raw.mapped_bytes),
      attrs_raw.npy.payload_bytes, attrs_raw.seal_status);
  const Digest256 corpus_sha = canonical_corpus_sha256(
      vectors_identity, attrs_identity, object_count, vector_dim, attr_dim);
  VerifiedCorpusIdentity identity(
      std::move(vectors_identity), std::move(attrs_identity), corpus_sha,
      object_count, vector_dim, attr_dim);
  std::shared_ptr<const VerifiedCorpusState> state =
      std::make_shared<const VerifiedCorpusState>(
          std::move(corpus_root), std::move(vectors), std::move(attrs_raw),
          std::move(identity));
  return VerifiedCorpusSnapshot(std::move(state));
}

VerifiedCorpusSnapshot VerifiedCorpusSnapshot::open_for_testing(
    const VerifiedCorpusOpenRequest& request,
    VerifiedCorpusTestOnlyTag /*test_only*/) {
  return open_impl_for_testing(request);
}
#endif

const VerifiedCorpusIdentity& VerifiedCorpusSnapshot::identity() const noexcept {
  return state_->identity;
}

ArrayView<float> VerifiedCorpusSnapshot::vectors() const noexcept {
  return state_->vectors.view;
}

ArrayView<float> VerifiedCorpusSnapshot::attrs_raw() const noexcept {
  return state_->attrs_raw.view;
}

std::uint64_t VerifiedCorpusSnapshot::object_count() const noexcept {
  return state_->identity.object_count();
}

std::uint64_t VerifiedCorpusSnapshot::vector_dim() const noexcept {
  return state_->identity.vector_dim();
}

std::uint64_t VerifiedCorpusSnapshot::attr_dim() const noexcept {
  return state_->identity.attr_dim();
}

bool VerifiedCorpusSnapshot::reportable() const noexcept {
  return state_->identity.reportable();
}

SealStatus VerifiedCorpusSnapshot::vectors_seal_status() const noexcept {
  return state_->identity.vectors().seal_status();
}

SealStatus VerifiedCorpusSnapshot::attrs_raw_seal_status() const noexcept {
  return state_->identity.attrs_raw().seal_status();
}

}  // namespace bounded_blowup

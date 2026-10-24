// SPDX-License-Identifier: MIT
// Original implementation for PVLDB2027-RFANN. No baseline source is copied.
#pragma once

#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bounded_blowup/index_view.hpp"

namespace bounded_blowup {

class NumericError : public std::runtime_error {
 public:
  explicit NumericError(const std::string& message)
      : std::runtime_error(message) {}
};

namespace numeric_detail {

inline void require_round_to_nearest() {
  if (std::fegetround() != FE_TONEAREST) {
    throw NumericError("numeric contract requires FE_TONEAREST");
  }
}

inline void validate_distance_shape(ArrayView<float> vectors,
                                    std::uint32_t vector_dim,
                                    std::uint64_t object_count,
                                    std::int64_t lhs,
                                    std::int64_t rhs) {
  if (vector_dim == 0U || object_count == 0U) {
    throw NumericError("numeric corpus shape must be non-empty");
  }
  if (lhs < 0 || rhs < 0 ||
      static_cast<std::uint64_t>(lhs) >= object_count ||
      static_cast<std::uint64_t>(rhs) >= object_count) {
    throw NumericError("numeric distance ID is outside corpus bounds");
  }
  const std::uint64_t dimension = static_cast<std::uint64_t>(vector_dim);
  if (object_count > std::numeric_limits<std::uint64_t>::max() / dimension) {
    throw NumericError("numeric corpus shape overflows");
  }
  const std::uint64_t expected = object_count * dimension;
  if (expected >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      vectors.size() != static_cast<std::size_t>(expected)) {
    throw NumericError("numeric corpus vector shape is invalid");
  }
  if (vectors.data() == nullptr && expected != 0U) {
    throw NumericError("numeric corpus vector data is null");
  }
}

}  // namespace numeric_detail

// The hot seam is compiled only into the native implementation and its
// focused tests.  Keeping it out of the public header surface prevents an
// external caller from fabricating an undersized view or bypassing the worker
// FE_TONEAREST boundary.  The public API below remains the checked wrapper.
#if defined(RFBB_NUMERIC_INTERNAL)
namespace numeric_detail {

// A hot-path context is created only after the caller has independently
// checked the corpus values.  The receipt is part of the context so a build
// cannot accidentally opt into the unchecked primitive.  Shape, rounding,
// and the receipt are checked once at context creation and once at each
// worker boundary; the distance loop itself does not repeat those checks.
class NumericHotContext {
 public:
  NumericHotContext(const NumericHotContext&) = default;
  NumericHotContext(NumericHotContext&&) noexcept = default;
  NumericHotContext& operator=(const NumericHotContext&) = default;
  NumericHotContext& operator=(NumericHotContext&&) noexcept = default;

 private:
  NumericHotContext(ArrayView<float> vectors, std::uint32_t vector_dim,
                    std::uint64_t object_count,
                    bool finite_values_prevalidated)
      : vectors_(vectors),
        vector_dim_(vector_dim),
        object_count_(object_count),
        finite_values_prevalidated_(finite_values_prevalidated) {}

  friend NumericHotContext make_numeric_hot_context(
      ArrayView<float> vectors, std::uint32_t vector_dim,
      std::uint64_t object_count, bool finite_values_prevalidated);
  friend void validate_numeric_hot_context(const NumericHotContext& context);
  friend double squared_l2_f32_f64_seq_hot(const NumericHotContext& context,
                                            std::int64_t lhs,
                                            std::int64_t rhs);

  ArrayView<float> vectors_;
  std::uint32_t vector_dim_ = 0U;
  std::uint64_t object_count_ = 0U;
  bool finite_values_prevalidated_ = false;
};

inline void validate_numeric_hot_context(const NumericHotContext& context) {
  require_round_to_nearest();
  if (!context.finite_values_prevalidated_) {
    throw NumericError(
        "hot numeric context requires a finite-values prevalidation receipt");
  }
  validate_distance_shape(context.vectors_, context.vector_dim_,
                          context.object_count_, 0, 0);
}

inline NumericHotContext make_numeric_hot_context(
    ArrayView<float> vectors, std::uint32_t vector_dim,
    std::uint64_t object_count, bool finite_values_prevalidated) {
  NumericHotContext context(vectors, vector_dim, object_count,
                            finite_values_prevalidated);
  validate_numeric_hot_context(context);
  return context;
}

// The caller has validated IDs while validating the member list.  Keep a
// cheap bounds guard here as a fail-closed protection for native callers;
// unlike the public wrapper it does not recheck shape, rounding, or input
// finiteness on every distance pair.
#pragma STDC FP_CONTRACT OFF
inline __attribute__((always_inline)) double squared_l2_f32_f64_seq_hot(
    const NumericHotContext& context, std::int64_t lhs, std::int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      static_cast<std::uint64_t>(lhs) >= context.object_count_ ||
      static_cast<std::uint64_t>(rhs) >= context.object_count_) {
    throw NumericError("hot numeric distance ID is outside corpus bounds");
  }
  const std::size_t dimension =
      static_cast<std::size_t>(context.vector_dim_);
  const std::size_t left_offset = static_cast<std::size_t>(lhs) * dimension;
  const std::size_t right_offset = static_cast<std::size_t>(rhs) * dimension;
  double distance = 0.0;
  for (std::size_t coordinate = 0U; coordinate < dimension; ++coordinate) {
    // Do not add per-pair input-finite checks here.  A NaN/Inf still fails
    // closed through the accumulator check below, while valid hot-path
    // pairs avoid two redundant isfinite calls.
    const double left = static_cast<double>(
        context.vectors_[left_offset + coordinate]);
    const double right = static_cast<double>(
        context.vectors_[right_offset + coordinate]);
    const double delta = left - right;
    const double term = delta * delta;
    const double next_distance = distance + term;
    // The recurrence is fixed-order.  A nonfinite term necessarily makes the
    // running sum nonfinite (including NaN propagation), so checking the
    // final accumulator is a fail-closed check for both term and sum while
    // keeping the valid hot loop branch-free.
    distance = next_distance;
  }
  if (!std::isfinite(distance)) {
    throw NumericError(
        "hot numeric squared-distance term or accumulation is nonfinite");
  }
  if (distance == 0.0) distance = 0.0;
  return distance;
}

// Called from the native build after alpha_squared() has checked
// FE_TONEAREST once.  Input and product finiteness remain checked because
// this is the fail-closed boundary for a prevalidated distance value.
inline double alpha_product_hot(double alpha_squared_value,
                                double chosen_to_candidate_squared) {
  if (!std::isfinite(alpha_squared_value) || alpha_squared_value < 0.0) {
    throw NumericError("hot alpha squared is invalid");
  }
  if (!std::isfinite(chosen_to_candidate_squared) ||
      chosen_to_candidate_squared < 0.0) {
    throw NumericError("hot chosen-to-candidate squared distance is invalid");
  }
  const double lhs = alpha_squared_value * chosen_to_candidate_squared;
  if (!std::isfinite(lhs)) {
    throw NumericError("hot alpha product is nonfinite");
  }
  return lhs == 0.0 ? 0.0 : lhs;
}

}  // namespace numeric_detail
#endif

// Keep the scalar operations separate and sequenced.  The build also sets
// -ffp-contract=off; this pragma protects direct users of the public helper
// when they compile a small standalone test without the project flags.
#pragma STDC FP_CONTRACT OFF
inline double squared_l2_f32_f64_seq(ArrayView<float> vectors,
                                     std::uint32_t vector_dim,
                                     std::uint64_t object_count,
                                     std::int64_t lhs,
                                     std::int64_t rhs) {
  numeric_detail::require_round_to_nearest();
  numeric_detail::validate_distance_shape(vectors, vector_dim, object_count,
                                           lhs, rhs);
  const std::size_t dimension = static_cast<std::size_t>(vector_dim);
  const std::size_t left_offset = static_cast<std::size_t>(lhs) * dimension;
  const std::size_t right_offset = static_cast<std::size_t>(rhs) * dimension;
  double distance = 0.0;
  for (std::size_t coordinate = 0U; coordinate < dimension; ++coordinate) {
    const double left =
        static_cast<double>(vectors[left_offset + coordinate]);
    const double right =
        static_cast<double>(vectors[right_offset + coordinate]);
    if (!std::isfinite(left) || !std::isfinite(right)) {
      throw NumericError("numeric distance input is nonfinite");
    }
    const double delta = left - right;
    const double term = delta * delta;
    if (!std::isfinite(term)) {
      throw NumericError("numeric squared-distance term is nonfinite");
    }
    const double next_distance = distance + term;
    if (!std::isfinite(next_distance)) {
      throw NumericError("numeric squared-distance accumulation is nonfinite");
    }
    distance = next_distance;
  }
  if (distance == 0.0) distance = 0.0;
  return distance;
}

inline double alpha_squared(double alpha) {
  numeric_detail::require_round_to_nearest();
  if (!std::isfinite(alpha) || alpha <= 0.0) {
    throw NumericError("alpha must be finite and positive");
  }
  const double squared = alpha * alpha;
  if (!std::isfinite(squared)) {
    throw NumericError("alpha squared is nonfinite");
  }
  return squared == 0.0 ? 0.0 : squared;
}

inline double alpha_product(double alpha_squared_value,
                            double chosen_to_candidate_squared) {
  numeric_detail::require_round_to_nearest();
  if (!std::isfinite(alpha_squared_value) || alpha_squared_value < 0.0) {
    throw NumericError("alpha squared is invalid");
  }
  if (!std::isfinite(chosen_to_candidate_squared) ||
      chosen_to_candidate_squared < 0.0) {
    throw NumericError("chosen-to-candidate squared distance is invalid");
  }
  const double lhs = alpha_squared_value * chosen_to_candidate_squared;
  if (!std::isfinite(lhs)) {
    throw NumericError("alpha product is nonfinite");
  }
  return lhs == 0.0 ? 0.0 : lhs;
}

}  // namespace bounded_blowup

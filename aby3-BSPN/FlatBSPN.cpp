#include "FlatBSPN.h"
#include "aby3-Basic/BuildingBlocks.h"
#include "aby3-GORAM-Core/Basics.h"
#include <aby3/sh3/Sh3Piecewise.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace aby3 {

namespace {

using json = nlohmann::json;

#pragma pack(push, 1)
struct PackedRawNodeRecord {
    std::uint32_t node_id;
    std::uint8_t node_type;
    std::uint8_t reserved0;
    std::uint16_t reserved1;
    std::uint64_t cardinality;
    std::uint32_t child_begin;
    std::uint32_t child_count;
    std::uint32_t scope_bitmap_begin;
    std::uint32_t scope_bitmap_len;
    std::uint32_t bucket_begin;
    std::uint32_t bucket_count;
    std::uint32_t weight_begin;
    std::uint32_t weight_count;
    std::int32_t leaf_column_id;
};

struct PackedRawBucketRecord {
    std::uint32_t bucket_id;
    std::uint64_t bitmap_begin;
    std::uint32_t bitmap_len;
    std::uint32_t value_index;
    std::uint32_t lower_bound_index;
    std::uint32_t upper_bound_index;
};
#pragma pack(pop)

static_assert(sizeof(PackedRawNodeRecord) == 52, "PackedRawNodeRecord size mismatch");
static_assert(sizeof(PackedRawBucketRecord) == 28, "PackedRawBucketRecord size mismatch");

template <typename T>
std::vector<T> read_binary_records(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open binary file: " + path);
    }

    in.seekg(0, std::ios::end);
    const std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("Malformed binary record file: " + path);
    }

    std::vector<T> out(static_cast<std::size_t>(bytes / sizeof(T)));
    if (!out.empty()) {
        in.read(reinterpret_cast<char*>(out.data()), bytes);
    }
    return out;
}

std::vector<std::uint8_t> read_binary_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open binary file: " + path);
    }

    in.seekg(0, std::ios::end);
    const std::streamsize bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> out(static_cast<std::size_t>(std::max<std::streamsize>(bytes, 0)));
    if (!out.empty()) {
        in.read(reinterpret_cast<char*>(out.data()), bytes);
    }
    return out;
}

std::vector<double> read_binary_doubles(const std::string& path) {
    return read_binary_records<double>(path);
}

std::string join_path(const std::string& base_dir, const std::string& file_name) {
    if (base_dir.empty()) {
        return file_name;
    }
    if (base_dir.back() == '/') {
        return base_dir + file_name;
    }
    return base_dir + "/" + file_name;
}

std::string dirname_from_path(const std::string& path) {
    const std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

template <Decimal D>
f64Matrix<D> doubles_to_fixed_column(const std::vector<double>& values) {
    f64Matrix<D> out(values.size(), 1);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        out(static_cast<u64>(idx), 0) = values[idx];
    }
    return out;
}

template <Decimal D>
f64Matrix<D> doubles_to_fixed_matrix(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols) {
    f64Matrix<D> out(rows, cols);
    for (std::size_t idx = 0; idx < values.size() && idx < rows * cols; ++idx) {
        out(static_cast<u64>(idx / cols), static_cast<u64>(idx % cols)) = values[idx];
    }
    return out;
}

i64Matrix u8_to_i64_matrix(
    const std::vector<std::uint8_t>& values,
    std::size_t rows,
    std::size_t cols) {
    i64Matrix out(rows, cols);
    out.setZero();
    for (std::size_t idx = 0; idx < values.size() && idx < rows * cols; ++idx) {
        out(static_cast<u64>(idx / cols), static_cast<u64>(idx % cols)) = static_cast<i64>(values[idx]);
    }
    return out;
}

i64Matrix i64_column_from_u64(const std::vector<std::uint64_t>& values) {
    i64Matrix out(values.size(), 1);
    out.setZero();
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        out(static_cast<u64>(idx), 0) = static_cast<i64>(values[idx]);
    }
    return out;
}

template <Decimal D>
void share_fixed_matrix(
    const f64Matrix<D>& plain,
    sf64Matrix<D>& shared,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    shared.resize(plain.rows(), plain.cols());
    if (context.role == owner_party) {
        context.enc->localFixedMatrix(*(context.runtime), plain, shared).get();
    } else {
        context.enc->remoteFixedMatrix(*(context.runtime), shared).get();
    }
}

void share_int_matrix(
    const i64Matrix& plain,
    si64Matrix& shared,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    shared.resize(plain.rows(), plain.cols());
    if (context.role == owner_party) {
        context.enc->localIntMatrix(*(context.runtime), plain, shared).get();
    } else {
        context.enc->remoteIntMatrix(*(context.runtime), shared).get();
    }
}

void share_bool_matrix(
    const i64Matrix& plain,
    sbMatrix& shared,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    shared.resize(plain.rows(), plain.cols() * 64);
    if (context.role == owner_party) {
        context.enc->localBinMatrix(*(context.runtime), plain, shared).get();
    } else {
        context.enc->remoteBinMatrix(*(context.runtime), shared).get();
    }
}

i64Matrix unpack_bitmap_to_dense_rows(
    const std::vector<std::uint8_t>& bytes,
    std::uint64_t total_rows) {
    i64Matrix out(total_rows, 1);
    out.setZero();
    for (std::uint64_t row = 0; row < total_rows; ++row) {
        const std::size_t byte_idx = static_cast<std::size_t>(row / 8);
        const std::size_t bit_idx = static_cast<std::size_t>(row % 8);
        if (byte_idx < bytes.size()) {
            out(static_cast<u64>(row), 0) =
                static_cast<i64>((bytes[byte_idx] >> bit_idx) & std::uint8_t(1));
        }
    }
    return out;
}

template <Decimal D>
sf64Matrix<D> share_fixed_scalar(
    double value,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    f64Matrix<D> plain(1, 1);
    plain(0, 0) = value;
    sf64Matrix<D> shared(1, 1);
    share_fixed_matrix(plain, shared, owner_party, context);
    return shared;
}

si64Matrix share_int_scalar(
    i64 value,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    i64Matrix plain(1, 1);
    plain(0, 0) = value;
    si64Matrix shared(1, 1);
    share_int_matrix(plain, shared, owner_party, context);
    return shared;
}

double reveal_fixed_scalar(
    const sf64Matrix<kFlatBSPNDecimal>& shared,
    const FlatBSPNSecureContext& context) {
    f64Matrix<kFlatBSPNDecimal> plain(shared.rows(), shared.cols());
    context.enc->revealAll(context.runtime->noDependencies(), shared, plain).get();
    return static_cast<double>(plain(0, 0));
}

struct SecureRationalShare {
    sf64Matrix<kFlatBSPNDecimal> numerator;
    sf64Matrix<kFlatBSPNDecimal> denominator;
    double numerator_scale = 1.0;
    double denominator_scale = 1.0;
    bool denominator_is_one = false;
};

double reveal_scaled_numerator(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context) {
    return reveal_fixed_scalar(value.numerator, context) * value.numerator_scale;
}

double reveal_scaled_denominator(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context) {
    return reveal_fixed_scalar(value.denominator, context) * value.denominator_scale;
}

struct SecureFactorBinding {
    int slot_index = -1;
    std::string operator_kind;
};

struct SecureBoundFactor {
    std::string model_id;
    std::string manifest_path;
    std::string secure_leaf_eval_mode = "reciprocal_fallback";
    FlatFactorSpec factor;
    std::vector<SecureFactorBinding> column_bindings;
    int secret_factor_binding_index = -1;
};

struct SecureBundleExecutionResult {
    SecureRationalShare result_rational;
    bool has_result = false;
    json debug_output;
};

struct SecureIndicatorEvalStats {
    std::uint64_t reciprocal_calls = 0;
    std::uint64_t phase1_batch_dot_calls = 0;
    std::uint64_t phase3_batch_b2a_calls = 0;
    std::uint64_t leaf_product_nodes = 0;
    double sum_node_ms = 0.0;
    double product_sum_ms = 0.0;
    double phase1_match_ms = 0.0;
    double phase1_local_ids_ms = 0.0;
    double phase2_intersection_ms = 0.0;
    double phase2_count_ms = 0.0;
    double phase3_numerator_ms = 0.0;
    double final_combine_ms = 0.0;
};

using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = SteadyClock::time_point;

double elapsed_ms_since(const SteadyTimePoint& start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

sf64Matrix<kFlatBSPNDecimal> si64_to_sf64(
    const si64Matrix& shared,
    std::uint64_t left_shift = static_cast<std::uint64_t>(kFlatBSPNDecimal)) {
    sf64Matrix<kFlatBSPNDecimal> out(shared.rows(), shared.cols());
    out[0] = shared.mShares[0];
    out[1] = shared.mShares[1];
    if (left_shift != 0) {
        const i64 scale = i64(1) << static_cast<i64>(left_shift);
        out[0] *= scale;
        out[1] *= scale;
    }
    return out;
}

sf64Matrix<kFlatBSPNDecimal> secure_mul_fixed(
    const sf64Matrix<kFlatBSPNDecimal>& lhs,
    const sf64Matrix<kFlatBSPNDecimal>& rhs,
    const FlatBSPNSecureContext& context);

sf64Matrix<kFlatBSPNDecimal> secure_mul_public_fixed(
    const sf64Matrix<kFlatBSPNDecimal>& value,
    double public_factor,
    const FlatBSPNSecureContext& context);

SecureRationalShare align_secure_rational_scales(
    const SecureRationalShare& value,
    double target_numerator_scale,
    double target_denominator_scale,
    const FlatBSPNSecureContext& context);

SecureRationalShare normalize_secure_rational_scales(SecureRationalShare value);

sf64Matrix<kFlatBSPNDecimal> fixed_row_slice(
    const sf64Matrix<kFlatBSPNDecimal>& src,
    std::uint32_t row_begin,
    std::uint32_t row_count) {
    sf64Matrix<kFlatBSPNDecimal> out(row_count, src.cols());
    out[0] = src[0].block(row_begin, 0, row_count, src.cols());
    out[1] = src[1].block(row_begin, 0, row_count, src.cols());
    return out;
}

sf64Matrix<kFlatBSPNDecimal> fixed_cell(
    const sf64Matrix<kFlatBSPNDecimal>& src,
    std::uint32_t row,
    std::uint32_t col) {
    sf64Matrix<kFlatBSPNDecimal> out(1, 1);
    out[0](0, 0) = src[0](row, col);
    out[1](0, 0) = src[1](row, col);
    return out;
}

si64Matrix int_row_slice(
    const si64Matrix& src,
    std::uint32_t row_begin,
    std::uint32_t row_count) {
    si64Matrix out(row_count, src.cols());
    out.mShares[0] = src.mShares[0].block(row_begin, 0, row_count, src.cols());
    out.mShares[1] = src.mShares[1].block(row_begin, 0, row_count, src.cols());
    return out;
}

si64Matrix int_cell(
    const si64Matrix& src,
    std::uint32_t row,
    std::uint32_t col) {
    si64Matrix out(1, 1);
    out.mShares[0](0, 0) = src.mShares[0](row, col);
    out.mShares[1](0, 0) = src.mShares[1](row, col);
    return out;
}

si64Matrix shared_zero_int_scalar() {
    si64Matrix out(1, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    return out;
}

si64Matrix shared_zero_int_matrix(u64 rows, u64 cols) {
    si64Matrix out(rows, cols);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    return out;
}

sbMatrix shared_zero_bool_scalar(const FlatBSPNSecureContext& context) {
    sbMatrix out(1, 1);
    bool_init_false(context.role, out);
    return out;
}

sbMatrix shared_true_bool_scalar(const FlatBSPNSecureContext& context) {
    sbMatrix out(1, 1);
    bool_init_true(context.role, out);
    return out;
}

sbMatrix shared_int_positive_flag(si64Matrix value, const FlatBSPNSecureContext& context) {
    si64Matrix zero = shared_zero_int_scalar();
    sbMatrix out;
    cipher_gt(context.role, value, zero, out, *(context.eval), *(context.runtime));
    return out;
}

sbMatrix shared_int_nonzero_flag(si64Matrix value, const FlatBSPNSecureContext& context) {
    return shared_int_positive_flag(std::move(value), context);
}

sbMatrix repeat_bool_scalar_rows(const sbMatrix& scalar, std::uint32_t rows) {
    sbMatrix out(rows, scalar.bitCount());
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (u64 col = 0; col < scalar.mShares[0].cols(); ++col) {
            out.mShares[0](row, col) = scalar.mShares[0](0, col);
            out.mShares[1](row, col) = scalar.mShares[1](0, col);
        }
    }
    return out;
}

sf64Matrix<kFlatBSPNDecimal> bool_scalar_to_fixed(
    const sbMatrix& scalar,
    const FlatBSPNSecureContext& context) {
    si64Matrix as_int(1, 1);
    bool2arith(context.role, const_cast<sbMatrix&>(scalar), as_int, *(context.enc), *(context.eval), *(context.runtime));
    return si64_to_sf64(as_int);
}

sf64Matrix<kFlatBSPNDecimal> select_fixed_by_bool(
    const sf64Matrix<kFlatBSPNDecimal>& true_value,
    const sf64Matrix<kFlatBSPNDecimal>& false_value,
    const sbMatrix& flag,
    const FlatBSPNSecureContext& context) {
    auto flag_fixed = bool_scalar_to_fixed(flag, context);
    auto delta = true_value - false_value;
    auto selected_delta = secure_mul_fixed(delta, flag_fixed, context);
    return false_value + selected_delta;
}

SecureRationalShare select_rational_by_bool(
    const SecureRationalShare& true_value,
    const SecureRationalShare& false_value,
    const sbMatrix& flag,
    const FlatBSPNSecureContext& context) {
    const double target_numerator_scale = std::max(
        true_value.numerator_scale,
        false_value.numerator_scale);
    const double target_denominator_scale = std::max(
        true_value.denominator_scale,
        false_value.denominator_scale);
    const auto aligned_true = align_secure_rational_scales(
        true_value,
        target_numerator_scale,
        target_denominator_scale,
        context);
    const auto aligned_false = align_secure_rational_scales(
        false_value,
        target_numerator_scale,
        target_denominator_scale,
        context);
    return normalize_secure_rational_scales({
        select_fixed_by_bool(aligned_true.numerator, aligned_false.numerator, flag, context),
        select_fixed_by_bool(aligned_true.denominator, aligned_false.denominator, flag, context),
        target_numerator_scale,
        target_denominator_scale,
        true_value.denominator_is_one && false_value.denominator_is_one,
    });
}

sbMatrix secure_scope_intersects(
    const std::vector<std::uint8_t>& public_scope_mask,
    const si64Matrix& secret_scope_row,
    const FlatBSPNSecureContext& context) {
    si64Matrix total = shared_zero_int_scalar();
    const std::size_t limit = std::min<std::size_t>(
        public_scope_mask.size(),
        static_cast<std::size_t>(secret_scope_row.cols()));
    for (std::size_t idx = 0; idx < limit; ++idx) {
        if (public_scope_mask[idx] == 0) {
            continue;
        }
        total = total + int_cell(secret_scope_row, 0, static_cast<std::uint32_t>(idx));
    }
    return shared_int_nonzero_flag(std::move(total), context);
}

sbMatrix share_secret_bool_column(
    const i64Matrix& plain,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    si64Matrix shared;
    share_int_matrix(plain, shared, owner_party, context);
    si64Matrix zero = shared_zero_int_matrix(shared.rows(), shared.cols());
    sbMatrix out;
    cipher_gt(context.role, shared, zero, out, *(context.eval), *(context.runtime));
    return out;
}

sbMatrix share_secret_bool_scalar(
    bool value,
    int owner_party,
    const FlatBSPNSecureContext& context) {
    i64Matrix plain(1, 1);
    plain(0, 0) = value ? 1 : 0;
    return share_secret_bool_column(plain, owner_party, context);
}

boolShare row0_bool_share(const sbMatrix& mat) {
    boolShare out;
    out.bshares[0] = static_cast<bool>(mat.mShares[0](0, 0));
    out.bshares[1] = static_cast<bool>(mat.mShares[1](0, 0));
    return out;
}

SecureRationalShare make_secure_rational(
    double numerator,
    double denominator,
    const FlatBSPNSecureContext& context) {
    SecureRationalShare out;
    out.numerator = share_fixed_scalar<kFlatBSPNDecimal>(numerator, 0, context);
    out.denominator = share_fixed_scalar<kFlatBSPNDecimal>(denominator, 0, context);
    out.denominator_is_one = std::abs(denominator - 1.0) <= 1e-12;
    return out;
}

SecureRationalShare normalize_secure_rational_scales(SecureRationalShare value) {
    const double common_scale = std::max(
        std::max(std::abs(value.numerator_scale), std::abs(value.denominator_scale)),
        1.0);
    if (std::isfinite(common_scale) && common_scale > 0.0) {
        value.numerator_scale /= common_scale;
        value.denominator_scale /= common_scale;
    }
    return value;
}

sf64Matrix<kFlatBSPNDecimal> secure_mul_fixed(
    const sf64Matrix<kFlatBSPNDecimal>& lhs,
    const sf64Matrix<kFlatBSPNDecimal>& rhs,
    const FlatBSPNSecureContext& context) {
    auto lhs_copy = lhs;
    auto rhs_copy = rhs;
    sf64Matrix<kFlatBSPNDecimal> out(1, 1);
    cipher_mul(
        context.role,
        lhs_copy,
        rhs_copy,
        out,
        *(context.eval),
        *(context.enc),
        *(context.runtime));
    return out;
}

sf64Matrix<kFlatBSPNDecimal> secure_mul_public_fixed(
    const sf64Matrix<kFlatBSPNDecimal>& value,
    double public_factor,
    const FlatBSPNSecureContext& context) {
    auto factor = share_fixed_scalar<kFlatBSPNDecimal>(public_factor, 0, context);
    return secure_mul_fixed(value, factor, context);
}

sf64Matrix<kFlatBSPNDecimal> secure_count_reciprocal_piecewise(
    const sf64Matrix<kFlatBSPNDecimal>& count,
    std::uint64_t max_count,
    const FlatBSPNSecureContext& context) {
    if (!context.has_runtime()) {
        throw std::runtime_error("Secure runtime is required for reciprocal approximation.");
    }

    // `effective_cnt` is an integer-valued secret count. A piecewise-constant
    // lookup over half-integer thresholds gives an exact reciprocal for every
    // count in [1, max_count] while keeping the entire path secret-shared.
    Sh3Piecewise reciprocal_lookup;
    reciprocal_lookup.mThresholds.reserve(static_cast<std::size_t>(max_count));
    reciprocal_lookup.mCoefficients.resize(static_cast<std::size_t>(max_count) + 1);

    reciprocal_lookup.mCoefficients[0].resize(1);
    reciprocal_lookup.mCoefficients[0][0] = 1.0;
    for (std::uint64_t value = 1; value <= max_count; ++value) {
        reciprocal_lookup.mThresholds.emplace_back(static_cast<double>(value) - 0.5);
        reciprocal_lookup.mCoefficients[static_cast<std::size_t>(value)].resize(1);
        reciprocal_lookup.mCoefficients[static_cast<std::size_t>(value)][0] =
            1.0 / static_cast<double>(value);
    }

    sf64Matrix<kFlatBSPNDecimal> reciprocal(count.rows(), count.cols());
    reciprocal_lookup.eval<kFlatBSPNDecimal>(
        context.runtime->noDependencies(),
        count,
        reciprocal,
        *(context.eval));
    return reciprocal;
}

SecureRationalShare multiply_secure_rational(
    const SecureRationalShare& lhs,
    const SecureRationalShare& rhs,
    const FlatBSPNSecureContext& context) {
    SecureRationalShare out;
    out.numerator = secure_mul_fixed(lhs.numerator, rhs.numerator, context);
    out.numerator_scale = lhs.numerator_scale * rhs.numerator_scale;
    if (lhs.denominator_is_one && rhs.denominator_is_one) {
        out.denominator = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
        out.denominator_scale = 1.0;
        out.denominator_is_one = true;
    } else {
        out.denominator = secure_mul_fixed(lhs.denominator, rhs.denominator, context);
        out.denominator_scale = lhs.denominator_scale * rhs.denominator_scale;
    }
    return normalize_secure_rational_scales(std::move(out));
}

SecureRationalShare scale_secure_rational(
    const SecureRationalShare& value,
    const sf64Matrix<kFlatBSPNDecimal>& public_scale,
    const FlatBSPNSecureContext& context) {
    return normalize_secure_rational_scales({
        secure_mul_fixed(value.numerator, public_scale, context),
        secure_mul_fixed(value.denominator, public_scale, context),
        value.numerator_scale,
        value.denominator_scale,
    });
}

SecureRationalShare scale_secure_rational_public(
    const SecureRationalShare& value,
    double public_factor,
    const FlatBSPNSecureContext& context) {
    return normalize_secure_rational_scales({
        secure_mul_public_fixed(value.numerator, public_factor, context),
        secure_mul_public_fixed(value.denominator, public_factor, context),
        value.numerator_scale / public_factor,
        value.denominator_scale / public_factor,
    });
}

SecureRationalShare align_secure_rational_scales(
    const SecureRationalShare& value,
    double target_numerator_scale,
    double target_denominator_scale,
    const FlatBSPNSecureContext& context) {
    SecureRationalShare out = value;
    const double numerator_factor = value.numerator_scale / target_numerator_scale;
    const double denominator_factor = value.denominator_scale / target_denominator_scale;
    out.numerator = secure_mul_public_fixed(value.numerator, numerator_factor, context);
    out.denominator = secure_mul_public_fixed(value.denominator, denominator_factor, context);
    out.numerator_scale = target_numerator_scale;
    out.denominator_scale = target_denominator_scale;
    return out;
}

SecureRationalShare invert_secure_rational(const SecureRationalShare& value) {
    return normalize_secure_rational_scales(
        {value.denominator, value.numerator, value.denominator_scale, value.numerator_scale});
}

SecureRationalShare maybe_invert_secure_rational(
    const SecureRationalShare& value,
    bool inverse) {
    return inverse ? invert_secure_rational(value) : value;
}

SecureRationalShare add_secure_rational(
    const SecureRationalShare& lhs,
    const SecureRationalShare& rhs,
    const FlatBSPNSecureContext& context) {
    SecureRationalShare out;
    if (lhs.denominator_is_one && rhs.denominator_is_one) {
        const double numerator_scale = std::max(lhs.numerator_scale, rhs.numerator_scale);
        auto lhs_num = secure_mul_public_fixed(lhs.numerator, lhs.numerator_scale / numerator_scale, context);
        auto rhs_num = secure_mul_public_fixed(rhs.numerator, rhs.numerator_scale / numerator_scale, context);
        out.numerator = lhs_num + rhs_num;
        out.denominator = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
        out.numerator_scale = numerator_scale;
        out.denominator_scale = 1.0;
        out.denominator_is_one = true;
        return normalize_secure_rational_scales(std::move(out));
    }
    sf64Matrix<kFlatBSPNDecimal> ad = secure_mul_fixed(lhs.numerator, rhs.denominator, context);
    sf64Matrix<kFlatBSPNDecimal> bc = secure_mul_fixed(rhs.numerator, lhs.denominator, context);
    const double ad_scale = lhs.numerator_scale * rhs.denominator_scale;
    const double bc_scale = rhs.numerator_scale * lhs.denominator_scale;
    const double numerator_scale = std::max(ad_scale, bc_scale);
    ad = secure_mul_public_fixed(ad, ad_scale / numerator_scale, context);
    bc = secure_mul_public_fixed(bc, bc_scale / numerator_scale, context);
    out.numerator = ad + bc;
    out.denominator = secure_mul_fixed(lhs.denominator, rhs.denominator, context);
    out.numerator_scale = numerator_scale;
    out.denominator_scale = lhs.denominator_scale * rhs.denominator_scale;
    return normalize_secure_rational_scales(std::move(out));
}

SecureRationalShare subtract_secure_rational(
    const SecureRationalShare& lhs,
    const SecureRationalShare& rhs,
    const FlatBSPNSecureContext& context) {
    SecureRationalShare out;
    if (lhs.denominator_is_one && rhs.denominator_is_one) {
        const double numerator_scale = std::max(lhs.numerator_scale, rhs.numerator_scale);
        auto lhs_num = secure_mul_public_fixed(lhs.numerator, lhs.numerator_scale / numerator_scale, context);
        auto rhs_num = secure_mul_public_fixed(rhs.numerator, rhs.numerator_scale / numerator_scale, context);
        out.numerator = lhs_num - rhs_num;
        out.denominator = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
        out.numerator_scale = numerator_scale;
        out.denominator_scale = 1.0;
        out.denominator_is_one = true;
        return normalize_secure_rational_scales(std::move(out));
    }
    sf64Matrix<kFlatBSPNDecimal> ad = secure_mul_fixed(lhs.numerator, rhs.denominator, context);
    sf64Matrix<kFlatBSPNDecimal> bc = secure_mul_fixed(rhs.numerator, lhs.denominator, context);
    const double ad_scale = lhs.numerator_scale * rhs.denominator_scale;
    const double bc_scale = rhs.numerator_scale * lhs.denominator_scale;
    const double numerator_scale = std::max(ad_scale, bc_scale);
    ad = secure_mul_public_fixed(ad, ad_scale / numerator_scale, context);
    bc = secure_mul_public_fixed(bc, bc_scale / numerator_scale, context);
    out.numerator = ad - bc;
    out.denominator = secure_mul_fixed(lhs.denominator, rhs.denominator, context);
    out.numerator_scale = numerator_scale;
    out.denominator_scale = lhs.denominator_scale * rhs.denominator_scale;
    return normalize_secure_rational_scales(std::move(out));
}

std::vector<std::uint8_t> unpack_scope_bits(
    const std::vector<std::uint8_t>& packed,
    std::size_t total_columns) {
    std::vector<std::uint8_t> out(total_columns, 0);
    for (std::size_t idx = 0; idx < total_columns; ++idx) {
        const std::size_t byte_idx = idx / 8;
        const std::size_t bit_idx = idx % 8;
        if (byte_idx < packed.size()) {
            out[idx] = static_cast<std::uint8_t>((packed[byte_idx] >> bit_idx) & std::uint8_t(1));
        }
    }
    return out;
}

bool scope_intersects(const std::vector<std::uint8_t>& lhs, const std::vector<int>& rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    for (std::size_t idx = 0; idx < limit; ++idx) {
        if (lhs[idx] != 0 && rhs[idx] != 0) {
            return true;
        }
    }
    return false;
}

bool scope_bit_at(const std::vector<int>& scope, std::size_t idx) {
    return idx < scope.size() && scope[idx] != 0;
}

double safe_inverse(double value) {
    return std::abs(value) > 1e-12 ? (1.0 / value) : 1.0;
}

std::vector<std::uint8_t> packed_ones(std::uint64_t total_rows) {
    const std::size_t num_bytes = static_cast<std::size_t>((total_rows + 7) / 8);
    std::vector<std::uint8_t> out(num_bytes, 0xFF);
    if (num_bytes != 0 && total_rows % 8 != 0) {
        const std::uint8_t low_mask = static_cast<std::uint8_t>((1u << (total_rows % 8)) - 1u);
        out.back() = low_mask;
    }
    return out;
}

std::size_t packed_popcount(const std::vector<std::uint8_t>& packed) {
    std::size_t total = 0;
    for (std::uint8_t byte : packed) {
        total += static_cast<std::size_t>(__builtin_popcount(static_cast<unsigned int>(byte)));
    }
    return total;
}

std::vector<std::uint8_t> packed_and(
    const std::vector<std::uint8_t>& lhs,
    const std::vector<std::uint8_t>& rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    std::vector<std::uint8_t> out(limit, 0);
    for (std::size_t idx = 0; idx < limit; ++idx) {
        out[idx] = static_cast<std::uint8_t>(lhs[idx] & rhs[idx]);
    }
    return out;
}

void packed_and_in_place(std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    for (std::size_t idx = 0; idx < limit; ++idx) {
        lhs[idx] = static_cast<std::uint8_t>(lhs[idx] & rhs[idx]);
    }
}

void packed_or_in_place(std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    for (std::size_t idx = 0; idx < limit; ++idx) {
        lhs[idx] = static_cast<std::uint8_t>(lhs[idx] | rhs[idx]);
    }
}

void packed_xor_in_place(std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    for (std::size_t idx = 0; idx < limit; ++idx) {
        lhs[idx] = static_cast<std::uint8_t>(lhs[idx] ^ rhs[idx]);
    }
}

std::vector<int> packed_to_dense_bits(const std::vector<std::uint8_t>& packed, std::uint64_t total_rows) {
    std::vector<int> out(static_cast<std::size_t>(total_rows), 0);
    for (std::uint64_t idx = 0; idx < total_rows; ++idx) {
        const std::size_t byte_idx = static_cast<std::size_t>(idx / 8);
        const std::size_t bit_idx = static_cast<std::size_t>(idx % 8);
        if (byte_idx < packed.size()) {
            out[static_cast<std::size_t>(idx)] = static_cast<int>((packed[byte_idx] >> bit_idx) & std::uint8_t(1));
        }
    }
    return out;
}

FlatRationalValue normalize_rational(FlatRationalValue value) {
    if (std::abs(value.numerator) <= 1e-12) {
        value.numerator = 0.0;
        value.denominator = 1.0;
        return value;
    }
    if (std::abs(value.denominator) <= 1e-12) {
        return value;
    }
    if (value.denominator < 0.0) {
        value.numerator = -value.numerator;
        value.denominator = -value.denominator;
    }
    return value;
}

FlatRationalValue make_rational(double numerator, double denominator) {
    return normalize_rational({numerator, denominator});
}

double materialize_rational(const FlatRationalValue& value) {
    if (std::abs(value.numerator) <= 1e-12) {
        return 0.0;
    }
    if (std::abs(value.denominator) <= 1e-12) {
        return 0.0;
    }
    return value.numerator / value.denominator;
}

FlatRationalValue multiply_rational(const FlatRationalValue& lhs, const FlatRationalValue& rhs) {
    return make_rational(lhs.numerator * rhs.numerator, lhs.denominator * rhs.denominator);
}

FlatRationalValue add_rational(const FlatRationalValue& lhs, const FlatRationalValue& rhs) {
    return make_rational(
        lhs.numerator * rhs.denominator + rhs.numerator * lhs.denominator,
        lhs.denominator * rhs.denominator);
}

FlatRationalValue subtract_rational(const FlatRationalValue& lhs, const FlatRationalValue& rhs) {
    return make_rational(
        lhs.numerator * rhs.denominator - rhs.numerator * lhs.denominator,
        lhs.denominator * rhs.denominator);
}

FlatRationalValue invert_rational(const FlatRationalValue& value) {
    return make_rational(value.denominator, value.numerator);
}

FlatRationalValue maybe_invert_rational(const FlatRationalValue& value, bool inverse) {
    return inverse ? invert_rational(value) : value;
}

FlatRationalValue weighted_sum_rational(
    const std::vector<FlatRationalValue>& values,
    const std::vector<double>& weights) {
    if (values.empty()) {
        return make_rational(0.0, 1.0);
    }

    long double denominator = 1.0L;
    for (const auto& value : values) {
        denominator *= static_cast<long double>(std::abs(value.denominator) <= 1e-12 ? 1.0 : value.denominator);
    }

    long double numerator = 0.0L;
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        long double scaled = static_cast<long double>(weights[idx]) * static_cast<long double>(values[idx].numerator);
        for (std::size_t other = 0; other < values.size(); ++other) {
            if (other == idx) {
                continue;
            }
            const double denom = std::abs(values[other].denominator) <= 1e-12 ? 1.0 : values[other].denominator;
            scaled *= static_cast<long double>(denom);
        }
        numerator += scaled;
    }

    return make_rational(static_cast<double>(numerator), static_cast<double>(denominator));
}

bool bucket_matches_evidence(
    double lower,
    double upper,
    const std::vector<FlatEvidenceInterval>& intervals) {
    if (intervals.empty()) {
        return true;
    }
    for (const auto& interval : intervals) {
        const bool lower_ok = !interval.has_lower || (interval.open_lower ? (lower > interval.lower) : (lower >= interval.lower));
        const bool upper_ok = !interval.has_upper || (interval.open_upper ? (upper < interval.upper) : (upper <= interval.upper));
        if (lower_ok && upper_ok) {
            return true;
        }
    }
    return false;
}

bool is_leaf_like(FlatBSPNNodeType type) {
    return type == FlatBSPNNodeType::LEAF || type == FlatBSPNNodeType::DUMMY;
}

std::string normalize_model_root(const std::string& value) {
    if (!value.empty()) {
        return value;
    }
    return ".";
}

std::string default_manifest_path_for_model(const std::string& model_root, const std::string& model_id) {
    return join_path(join_path(model_root, model_id), "manifest.json");
}

std::map<std::string, std::string> load_model_manifest_map(
    const json& doc,
    const std::string& model_root) {
    std::map<std::string, std::string> out;
    if (!doc.is_object()) {
        return out;
    }

    for (auto it = doc.begin(); it != doc.end(); ++it) {
        if (!it.value().is_string()) {
            continue;
        }
        out[it.key()] = it.value().get<std::string>();
    }

    if (out.empty() && doc.contains("model_manifests") && doc["model_manifests"].is_object()) {
        for (auto it = doc["model_manifests"].begin(); it != doc["model_manifests"].end(); ++it) {
            if (!it.value().is_string()) {
                continue;
            }
            out[it.key()] = it.value().get<std::string>();
        }
    }

    if (out.empty() && doc.contains("models") && doc["models"].is_array()) {
        for (const auto& item : doc["models"]) {
            if (!item.is_object()) {
                continue;
            }
            const std::string model_id = item.value("model_id", std::string());
            const std::string manifest_path = item.value("manifest_path", std::string());
            if (!model_id.empty() && !manifest_path.empty()) {
                out[model_id] = manifest_path;
            }
        }
    }

    for (auto& kv : out) {
        if (!kv.second.empty() && kv.second.front() != '/') {
            kv.second = join_path(model_root, kv.second);
        }
    }
    return out;
}

FlatDensePredicateBinding dense_binding_from_legacy_json(const json& binding_doc) {
    FlatDensePredicateBinding binding;
    binding.slot_id = binding_doc.value("slot_id", std::string());
    binding.source_kind = binding_doc.value("source_kind", std::string());
    binding.table_id = binding_doc.value("table_id", std::string());
    binding.column_id = binding_doc.value("column_id", std::string());
    binding.operator_kind = binding_doc.value("operator_kind", std::string());

    const bool open_lower = binding_doc.value("open_lower", false);
    const bool open_upper = binding_doc.value("open_upper", false);
    const auto& intervals = binding_doc.value("intervals", json::array());
    binding.interval_count = static_cast<std::uint64_t>(intervals.size());
    binding.has_evidence = binding.interval_count != 0;
    for (const auto& interval_item : intervals) {
        const bool has_lower = interval_item.is_array() && interval_item.size() == 2 && !interval_item[0].is_null();
        const bool has_upper = interval_item.is_array() && interval_item.size() == 2 && !interval_item[1].is_null();
        binding.lower_bounds.push_back(has_lower ? interval_item[0].get<double>() : 0.0);
        binding.upper_bounds.push_back(has_upper ? interval_item[1].get<double>() : 0.0);
        binding.has_lower.push_back(static_cast<std::uint8_t>(has_lower ? 1 : 0));
        binding.has_upper.push_back(static_cast<std::uint8_t>(has_upper ? 1 : 0));
        binding.open_lower.push_back(static_cast<std::uint8_t>((open_lower && has_lower) ? 1 : 0));
        binding.open_upper.push_back(static_cast<std::uint8_t>((open_upper && has_upper) ? 1 : 0));
    }
    return binding;
}

FlatDensePredicateBinding dense_binding_from_json(const json& binding_doc, std::size_t max_interval_count) {
    FlatDensePredicateBinding binding;
    binding.slot_id = binding_doc.value("slot_id", std::string());
    binding.source_kind = binding_doc.value("source_kind", std::string());
    binding.table_id = binding_doc.value("table_id", std::string());
    binding.column_id = binding_doc.value("column_id", std::string());
    binding.operator_kind = binding_doc.value("operator_kind", std::string());
    binding.interval_count = binding_doc.value("interval_count", std::uint64_t(0));
    binding.has_evidence = binding_doc.value("has_evidence", 0) != 0;

    auto read_u8_vector = [&](const char* key) {
        std::vector<std::uint8_t> out;
        for (const auto& item : binding_doc.value(key, json::array())) {
            out.push_back(static_cast<std::uint8_t>(item.get<int>()));
        }
        out.resize(max_interval_count, 0);
        return out;
    };
    auto read_double_vector = [&](const char* key) {
        std::vector<double> out;
        for (const auto& item : binding_doc.value(key, json::array())) {
            out.push_back(item.get<double>());
        }
        out.resize(max_interval_count, 0.0);
        return out;
    };

    binding.lower_bounds = read_double_vector("lower_bounds");
    binding.upper_bounds = read_double_vector("upper_bounds");
    binding.has_lower = read_u8_vector("has_lower");
    binding.has_upper = read_u8_vector("has_upper");
    binding.open_lower = read_u8_vector("open_lower");
    binding.open_upper = read_u8_vector("open_upper");
    return binding;
}

FlatDenseSecretFactorBinding dense_secret_factor_from_json(const json& binding_doc, std::size_t max_column_count) {
    FlatDenseSecretFactorBinding binding;
    binding.secret_factor_id = binding_doc.value("secret_factor_id", std::string());
    binding.factor_index = binding_doc.value("factor_index", -1);
    binding.column_count = binding_doc.value("column_count", std::uint64_t(0));

    auto read_u8_vector = [&](const char* key) {
        std::vector<std::uint8_t> out;
        for (const auto& item : binding_doc.value(key, json::array())) {
            out.push_back(static_cast<std::uint8_t>(item.get<int>()));
        }
        out.resize(max_column_count, 0);
        return out;
    };

    binding.feature_scope = read_u8_vector("feature_scope");
    binding.relevant_scope = read_u8_vector("relevant_scope");
    binding.feature_inverted_scope = read_u8_vector("feature_inverted_scope");
    return binding;
}

FlatSecureQueryPayload parse_secure_query_payload_doc(const json& doc) {
    FlatSecureQueryPayload payload;
    payload.payload_version = doc.value("payload_version", std::string());
    payload.query_skeleton_id = doc.value("query_skeleton_id", std::string());
    payload.binding_layout_kind = doc.value("binding_layout_kind", std::string());

    if (doc.contains("slot_payload_shape") && doc["slot_payload_shape"].is_object()) {
        payload.slot_count = doc["slot_payload_shape"].value("slot_count", std::uint64_t(0));
        payload.max_interval_count = doc["slot_payload_shape"].value("max_interval_count", std::uint64_t(0));
    }
    if (doc.contains("factor_payload_shape") && doc["factor_payload_shape"].is_object()) {
        payload.factor_count = doc["factor_payload_shape"].value("factor_count", std::uint64_t(0));
        payload.max_factor_column_count = doc["factor_payload_shape"].value("max_column_count", std::uint64_t(0));
    }

    if (doc.contains("predicate_slot_bindings_dense") && doc["predicate_slot_bindings_dense"].is_array()) {
        if (payload.max_interval_count == 0) {
            for (const auto& binding_doc : doc["predicate_slot_bindings_dense"]) {
                payload.max_interval_count = std::max<std::uint64_t>(
                    payload.max_interval_count,
                    static_cast<std::uint64_t>(binding_doc.value("interval_count", 0)));
            }
        }
        for (const auto& binding_doc : doc["predicate_slot_bindings_dense"]) {
            payload.predicate_slot_bindings.push_back(
                dense_binding_from_json(binding_doc, static_cast<std::size_t>(payload.max_interval_count)));
        }
    } else if (doc.contains("predicate_slot_bindings") && doc["predicate_slot_bindings"].is_array()) {
        for (const auto& binding_doc : doc["predicate_slot_bindings"]) {
            payload.predicate_slot_bindings.push_back(dense_binding_from_legacy_json(binding_doc));
        }
        for (const auto& binding : payload.predicate_slot_bindings) {
            payload.max_interval_count = std::max<std::uint64_t>(
                payload.max_interval_count,
                static_cast<std::uint64_t>(binding.lower_bounds.size()));
        }
        for (auto& binding : payload.predicate_slot_bindings) {
            binding.lower_bounds.resize(payload.max_interval_count, 0.0);
            binding.upper_bounds.resize(payload.max_interval_count, 0.0);
            binding.has_lower.resize(payload.max_interval_count, 0);
            binding.has_upper.resize(payload.max_interval_count, 0);
            binding.open_lower.resize(payload.max_interval_count, 0);
            binding.open_upper.resize(payload.max_interval_count, 0);
        }
    }

    if (payload.slot_count == 0) {
        payload.slot_count = static_cast<std::uint64_t>(payload.predicate_slot_bindings.size());
    }
    if (doc.contains("secret_factor_bindings_dense") && doc["secret_factor_bindings_dense"].is_array()) {
        if (payload.max_factor_column_count == 0) {
            for (const auto& binding_doc : doc["secret_factor_bindings_dense"]) {
                payload.max_factor_column_count = std::max<std::uint64_t>(
                    payload.max_factor_column_count,
                    static_cast<std::uint64_t>(binding_doc.value("column_count", 0)));
            }
        }
        for (const auto& binding_doc : doc["secret_factor_bindings_dense"]) {
            payload.secret_factor_bindings.push_back(
                dense_secret_factor_from_json(binding_doc, static_cast<std::size_t>(payload.max_factor_column_count)));
        }
    }
    if (payload.factor_count == 0) {
        payload.factor_count = static_cast<std::uint64_t>(payload.secret_factor_bindings.size());
    }
    return payload;
}

sf64Matrix<kFlatBSPNDecimal> repeat_fixed_scalar_rows(
    const sf64Matrix<kFlatBSPNDecimal>& scalar,
    std::uint32_t rows) {
    sf64Matrix<kFlatBSPNDecimal> out(rows, 1);
    for (std::uint32_t row = 0; row < rows; ++row) {
        out[0](row, 0) = scalar[0](0, 0);
        out[1](row, 0) = scalar[1](0, 0);
    }
    return out;
}

sf64Matrix<kFlatBSPNDecimal> sum_boolean_mask_to_fixed(
    const sbMatrix& mask,
    const FlatBSPNSecureContext& context) {
    si64Matrix as_int(mask.rows(), 1);
    bool2arith(context.role, const_cast<sbMatrix&>(mask), as_int, *(context.enc), *(context.eval), *(context.runtime));
    si64Matrix total(1, 1);
    arith_aggregation(context.role, as_int, total, *(context.enc), *(context.eval), *(context.runtime), "ADD");
    return si64_to_sf64(total);
}

si64Matrix sum_boolean_mask_to_int(
    const sbMatrix& mask,
    const FlatBSPNSecureContext& context) {
    si64Matrix as_int(mask.rows(), 1);
    bool2arith(context.role, const_cast<sbMatrix&>(mask), as_int, *(context.enc), *(context.eval), *(context.runtime));
    si64Matrix total(1, 1);
    arith_aggregation(context.role, as_int, total, *(context.enc), *(context.eval), *(context.runtime), "ADD");
    return total;
}

std::vector<sf64Matrix<kFlatBSPNDecimal>> compute_leaf_target_numerator_sums_batched(
    const FlatBSPNModel& model,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const sbMatrix& final_ids,
    const FlatBSPNSecureContext& context,
    std::uint64_t* phase3_batch_counter = nullptr) {
    std::vector<sf64Matrix<kFlatBSPNDecimal>> numerator_sums;
    numerator_sums.reserve(leaf_children.size());
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        (void)child_idx;
        numerator_sums.push_back(share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context));
    }

    struct BucketRef {
        std::size_t child_idx = 0;
        std::uint32_t bucket_index = 0;
    };
    std::vector<BucketRef> bucket_refs;
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        for (std::uint32_t bucket_offset = 0; bucket_offset < child.bucket_count; ++bucket_offset) {
            bucket_refs.push_back({child_idx, child.bucket_begin + bucket_offset});
        }
    }
    if (bucket_refs.empty()) {
        return numerator_sums;
    }

    const u64 mask_rows = final_ids.rows();
    const u64 bit_count = final_ids.bitCount();
    const u64 share_cols = final_ids.mShares[0].cols();
    const u64 stacked_rows = static_cast<u64>(bucket_refs.size()) * mask_rows;
    sbMatrix stacked_bitmaps(stacked_rows, bit_count);
    sbMatrix stacked_final_ids(stacked_rows, bit_count);

    for (std::size_t bucket_idx = 0; bucket_idx < bucket_refs.size(); ++bucket_idx) {
        const auto& bucket_bitmap = model.secret_shared_payload().dense_bucket_bitmaps[bucket_refs[bucket_idx].bucket_index];
        if (bucket_bitmap.rows() != mask_rows || bucket_bitmap.bitCount() != bit_count) {
            throw std::runtime_error("Bucket bitmap shape does not match final id mask shape.");
        }
        const u64 row_begin = static_cast<u64>(bucket_idx) * mask_rows;
        for (u64 row = 0; row < mask_rows; ++row) {
            for (u64 col = 0; col < share_cols; ++col) {
                stacked_bitmaps.mShares[0](row_begin + row, col) = bucket_bitmap.mShares[0](row, col);
                stacked_bitmaps.mShares[1](row_begin + row, col) = bucket_bitmap.mShares[1](row, col);
                stacked_final_ids.mShares[0](row_begin + row, col) = final_ids.mShares[0](row, col);
                stacked_final_ids.mShares[1](row_begin + row, col) = final_ids.mShares[1](row, col);
            }
        }
    }

    sbMatrix stacked_overlap(stacked_rows, bit_count);
    bool_cipher_and(
        context.role,
        stacked_bitmaps,
        stacked_final_ids,
        stacked_overlap,
        *(context.enc),
        *(context.eval),
        *(context.runtime));

    si64Matrix overlap_int(stacked_overlap.rows(), 1);
    bool2arith(context.role, stacked_overlap, overlap_int, *(context.enc), *(context.eval), *(context.runtime));
    if (phase3_batch_counter != nullptr) {
        ++(*phase3_batch_counter);
    }

    for (std::size_t bucket_idx = 0; bucket_idx < bucket_refs.size(); ++bucket_idx) {
        const u64 row_begin = static_cast<u64>(bucket_idx) * mask_rows;
        si64Matrix overlap_count(1, 1);
        overlap_count.mShares[0](0, 0) = overlap_int.mShares[0].block(row_begin, 0, mask_rows, 1).sum();
        overlap_count.mShares[1](0, 0) = overlap_int.mShares[1].block(row_begin, 0, mask_rows, 1).sum();
        const auto overlap_cnt = si64_to_sf64(overlap_count);
        auto bucket_value = fixed_row_slice(
            model.secret_shared_payload().bucket_values,
            bucket_refs[bucket_idx].bucket_index,
            1);
        numerator_sums[bucket_refs[bucket_idx].child_idx] += secure_mul_fixed(bucket_value, overlap_cnt, context);
    }

    return numerator_sums;
}

std::vector<sbMatrix> compute_leaf_local_ids_batched(
    const FlatBSPNModel& model,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const std::vector<sbMatrix>& match_masks,
    const FlatBSPNSecureContext& context,
    std::uint64_t* phase1_batch_counter = nullptr) {
    if (leaf_children.size() != match_masks.size()) {
        throw std::runtime_error("Leaf child and match mask counts do not match.");
    }
    std::vector<sbMatrix> local_ids;
    local_ids.reserve(leaf_children.size());
    if (leaf_children.empty()) {
        return local_ids;
    }

    const sbMatrix* shape_bitmap = nullptr;
    std::size_t total_buckets = 0;
    std::vector<std::size_t> child_bucket_begin;
    child_bucket_begin.reserve(leaf_children.size());
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        if (match_masks[child_idx].rows() != child.bucket_count) {
            throw std::runtime_error("Match mask row count does not match leaf bucket count.");
        }
        child_bucket_begin.push_back(total_buckets);
        total_buckets += child.bucket_count;
        if (shape_bitmap == nullptr && child.bucket_count != 0) {
            shape_bitmap = &model.secret_shared_payload().dense_bucket_bitmaps[child.bucket_begin];
        }
    }

    if (shape_bitmap == nullptr || total_buckets == 0) {
        for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
            sbMatrix child_local_ids(0, 64);
            bool_init_false(context.role, child_local_ids);
            local_ids.push_back(std::move(child_local_ids));
        }
        return local_ids;
    }

    const u64 block_len = shape_bitmap->rows();
    const u64 bit_count = shape_bitmap->bitCount();
    const u64 share_cols = shape_bitmap->mShares[0].cols();
    const u64 stacked_rows = static_cast<u64>(total_buckets) * block_len;
    sbMatrix stacked_bitmaps(stacked_rows, bit_count);
    sbMatrix stacked_match(stacked_rows, bit_count);

    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        const auto& match_mask = match_masks[child_idx];
        const bool expand_one_bit_mask = match_mask.bitCount() == 1;
        for (std::uint32_t bucket_offset = 0; bucket_offset < child.bucket_count; ++bucket_offset) {
            const auto& bucket_bitmap =
                model.secret_shared_payload().dense_bucket_bitmaps[child.bucket_begin + bucket_offset];
            if (bucket_bitmap.rows() != block_len || bucket_bitmap.bitCount() != bit_count) {
                throw std::runtime_error("Bucket bitmap shape mismatch in batched local id computation.");
            }
            const u64 row_begin =
                static_cast<u64>(child_bucket_begin[child_idx] + bucket_offset) * block_len;
            i64 match_share0 = 0;
            i64 match_share1 = 0;
            if (expand_one_bit_mask) {
                match_share0 = (match_mask.mShares[0](bucket_offset, 0) == 1) ? -1 : 0;
                match_share1 = (match_mask.mShares[1](bucket_offset, 0) == 1) ? -1 : 0;
            }
            for (u64 row = 0; row < block_len; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    stacked_bitmaps.mShares[0](row_begin + row, col) = bucket_bitmap.mShares[0](row, col);
                    stacked_bitmaps.mShares[1](row_begin + row, col) = bucket_bitmap.mShares[1](row, col);
                    if (expand_one_bit_mask) {
                        stacked_match.mShares[0](row_begin + row, col) = match_share0;
                        stacked_match.mShares[1](row_begin + row, col) = match_share1;
                    } else {
                        stacked_match.mShares[0](row_begin + row, col) = match_mask.mShares[0](bucket_offset, col);
                        stacked_match.mShares[1](row_begin + row, col) = match_mask.mShares[1](bucket_offset, col);
                    }
                }
            }
        }
    }

    sbMatrix stacked_products(stacked_rows, bit_count);
    bool_cipher_and(
        context.role,
        stacked_bitmaps,
        stacked_match,
        stacked_products,
        *(context.enc),
        *(context.eval),
        *(context.runtime));
    if (phase1_batch_counter != nullptr) {
        ++(*phase1_batch_counter);
    }

    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        sbMatrix child_local_ids(block_len, bit_count);
        if (child.bucket_count == 0) {
            bool_init_false(context.role, child_local_ids);
            local_ids.push_back(std::move(child_local_ids));
            continue;
        }
        const u64 first_row_begin = static_cast<u64>(child_bucket_begin[child_idx]) * block_len;
        for (u64 row = 0; row < block_len; ++row) {
            for (u64 col = 0; col < share_cols; ++col) {
                child_local_ids.mShares[0](row, col) = stacked_products.mShares[0](first_row_begin + row, col);
                child_local_ids.mShares[1](row, col) = stacked_products.mShares[1](first_row_begin + row, col);
            }
        }
        for (std::uint32_t bucket_offset = 1; bucket_offset < child.bucket_count; ++bucket_offset) {
            const u64 row_begin =
                static_cast<u64>(child_bucket_begin[child_idx] + bucket_offset) * block_len;
            for (u64 row = 0; row < block_len; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    child_local_ids.mShares[0](row, col) ^= stacked_products.mShares[0](row_begin + row, col);
                    child_local_ids.mShares[1](row, col) ^= stacked_products.mShares[1](row_begin + row, col);
                }
            }
        }
        local_ids.push_back(std::move(child_local_ids));
    }

    return local_ids;
}

SecureBoundFactor bind_secure_factor_from_secure_bundle(
    const json& factor_doc,
    const json& public_slot_by_id,
    const std::map<std::string, std::size_t>& slot_index_by_id,
    const std::map<std::string, std::size_t>& secret_factor_index_by_id,
    const std::map<std::string, std::string>& manifest_map,
    const std::string& model_root,
    std::map<std::string, FlatBSPNModel>& model_cache) {
    SecureBoundFactor bound;
    bound.factor.factor_index = factor_doc.value("factor_index", -1);
    bound.factor.factor_kind = factor_doc.value("factor_kind", std::string());
    bound.factor.inverse = factor_doc.value("inverse", false);
    bound.factor.public_constant_value = factor_doc.value("public_constant_value", 0.0);
    bound.secure_leaf_eval_mode = factor_doc.value(
        "secure_leaf_eval_mode",
        std::string("reciprocal_fallback"));

    if (bound.factor.factor_kind == "CONSTANT") {
        return bound;
    }

    bound.model_id = factor_doc.value("spn_model_id", std::string());
    if (bound.model_id.empty()) {
        throw std::runtime_error("Secure bundle factor is missing spn_model_id.");
    }
    auto manifest_it = manifest_map.find(bound.model_id);
    bound.manifest_path =
        manifest_it != manifest_map.end()
            ? manifest_it->second
            : default_manifest_path_for_model(model_root, bound.model_id);

    auto model_it = model_cache.find(bound.manifest_path);
    if (model_it == model_cache.end()) {
        FlatBSPNModel model;
        model.load_public_manifest(bound.manifest_path);
        model_it = model_cache.emplace(bound.manifest_path, std::move(model)).first;
    }
    const auto& model = model_it->second;
    const std::size_t total_columns = model.manifest().column_names.size();
    bound.factor.feature_scope.assign(total_columns, 0);
    bound.factor.relevant_scope.assign(total_columns, 0);
    bound.factor.feature_inverted_scope.assign(total_columns, 0);
    bound.factor.total_rows = factor_doc.value("total_rows", model.manifest().total_rows);
    bound.column_bindings.assign(total_columns, {});
    for (auto& binding : bound.column_bindings) {
        binding.slot_index = -1;
    }

    const std::string secret_factor_id = factor_doc.value("secret_factor_id", std::string());
    if (!secret_factor_id.empty()) {
        auto secret_factor_it = secret_factor_index_by_id.find(secret_factor_id);
        if (secret_factor_it == secret_factor_index_by_id.end()) {
            throw std::runtime_error("Missing secret factor binding index for factor: " + secret_factor_id);
        }
        bound.secret_factor_binding_index = static_cast<int>(secret_factor_it->second);
    }

    const auto predicate_slots = factor_doc.value("predicate_slot_ids", std::vector<std::string>());
    for (const auto& slot_id : predicate_slots) {
        if (!public_slot_by_id.contains(slot_id)) {
            throw std::runtime_error("Missing public predicate slot: " + slot_id);
        }
        const auto& slot = public_slot_by_id.at(slot_id);
        const std::string operator_kind = slot.value("operator_kind", std::string());
        if (operator_kind == "IS_NOT_NULL") {
            continue;
        }
        const std::string column_id = slot.value("column_id", std::string());
        const auto col_it = std::find(
            model.manifest().column_names.begin(),
            model.manifest().column_names.end(),
            column_id);
        if (col_it == model.manifest().column_names.end()) {
            throw std::runtime_error("Column not found in model manifest: " + column_id);
        }
        const std::size_t col_idx = static_cast<std::size_t>(std::distance(model.manifest().column_names.begin(), col_it));
        auto slot_it = slot_index_by_id.find(slot_id);
        if (slot_it == slot_index_by_id.end()) {
            throw std::runtime_error("Missing secure slot binding index for slot: " + slot_id);
        }
        bound.column_bindings[col_idx].slot_index = static_cast<int>(slot_it->second);
        bound.column_bindings[col_idx].operator_kind = operator_kind;
    }

    return bound;
}

SecureRationalShare weighted_sum_secure_rational(
    const std::vector<SecureRationalShare>& values,
    const std::vector<sf64Matrix<kFlatBSPNDecimal>>& weights,
    const FlatBSPNSecureContext& context) {
    if (values.empty()) {
        return make_secure_rational(0.0, 1.0, context);
    }
    const bool all_unit_denominators = std::all_of(
        values.begin(),
        values.end(),
        [](const SecureRationalShare& value) { return value.denominator_is_one; });
    if (all_unit_denominators) {
        double numerator_scale = 1.0;
        std::vector<sf64Matrix<kFlatBSPNDecimal>> terms(values.size());
        std::vector<double> term_scales(values.size(), 1.0);
        for (std::size_t idx = 0; idx < values.size(); ++idx) {
            terms[idx] = secure_mul_fixed(values[idx].numerator, weights[idx], context);
            term_scales[idx] = values[idx].numerator_scale;
            numerator_scale = std::max(numerator_scale, term_scales[idx]);
        }
        auto numerator_sum = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
        for (std::size_t idx = 0; idx < terms.size(); ++idx) {
            numerator_sum += secure_mul_public_fixed(
                terms[idx],
                term_scales[idx] / numerator_scale,
                context);
        }
        return normalize_secure_rational_scales({
            numerator_sum,
            share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
            numerator_scale,
            1.0,
            true,
        });
    }
    constexpr double kWeightedSumStorageUnit = 1.0;
    auto total = make_secure_rational(0.0, 1.0, context);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        SecureRationalShare term{
            secure_mul_fixed(values[idx].numerator, weights[idx], context),
            values[idx].denominator,
            values[idx].numerator_scale,
            values[idx].denominator_scale,
        };
        total = add_secure_rational(total, term, context);
        total = scale_secure_rational_public(total, kWeightedSumStorageUnit, context);
    }
    return total;
}

sbMatrix secure_interval_match_mask(
    const sf64Matrix<kFlatBSPNDecimal>& bucket_lowers,
    const sf64Matrix<kFlatBSPNDecimal>& bucket_uppers,
    const sf64Matrix<kFlatBSPNDecimal>& interval_lower,
    const sf64Matrix<kFlatBSPNDecimal>& interval_upper,
    const sbMatrix& has_lower_flag,
    const sbMatrix& has_upper_flag,
    const sbMatrix& open_lower_flag,
    const sbMatrix& open_upper_flag,
    const FlatBSPNSecureContext& context) {
    const auto rows = static_cast<std::uint32_t>(bucket_lowers.rows());
    auto lower_repeat = repeat_fixed_scalar_rows(interval_lower, rows);
    auto upper_repeat = repeat_fixed_scalar_rows(interval_upper, rows);

    auto bucket_lowers_copy = bucket_lowers;
    auto bucket_uppers_copy = bucket_uppers;
    sbMatrix ge_lower;
    cipher_ge(context.role, bucket_lowers_copy.i64Cast(), lower_repeat.i64Cast(), ge_lower, *(context.eval), *(context.enc), *(context.runtime));
    bucket_lowers_copy = bucket_lowers;
    sbMatrix gt_lower;
    cipher_gt(context.role, bucket_lowers_copy, lower_repeat, gt_lower, *(context.eval), *(context.runtime));

    sbMatrix lower_match(rows, 1);
    lower_match.mShares[0].setZero();
    lower_match.mShares[1].setZero();
    auto open_lower_repeat = repeat_bool_scalar_rows(open_lower_flag, rows);
    {
        sbMatrix not_open_lower(rows, 1);
        bool_cipher_not(context.role, open_lower_repeat, not_open_lower);
        sbMatrix ge_case(rows, 1);
        sbMatrix gt_case(rows, 1);
        bool_cipher_and(context.role, ge_lower, not_open_lower, ge_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, gt_lower, open_lower_repeat, gt_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, ge_case, gt_case, lower_match, *(context.enc), *(context.eval), *(context.runtime));
    }

    sbMatrix ge_upper;
    cipher_ge(context.role, upper_repeat.i64Cast(), bucket_uppers_copy.i64Cast(), ge_upper, *(context.eval), *(context.enc), *(context.runtime));
    bucket_uppers_copy = bucket_uppers;
    sbMatrix gt_upper;
    cipher_gt(context.role, upper_repeat, bucket_uppers_copy, gt_upper, *(context.eval), *(context.runtime));

    sbMatrix upper_match(rows, 1);
    upper_match.mShares[0].setZero();
    upper_match.mShares[1].setZero();
    {
        auto open_upper_repeat = repeat_bool_scalar_rows(open_upper_flag, rows);
        sbMatrix not_open_upper(rows, 1);
        bool_cipher_not(context.role, open_upper_repeat, not_open_upper);
        sbMatrix ge_case(rows, 1);
        sbMatrix gt_case(rows, 1);
        bool_cipher_and(context.role, ge_upper, not_open_upper, ge_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, gt_upper, open_upper_repeat, gt_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, ge_case, gt_case, upper_match, *(context.enc), *(context.eval), *(context.runtime));
    }

    auto has_lower_repeat = repeat_bool_scalar_rows(has_lower_flag, rows);
    auto has_upper_repeat = repeat_bool_scalar_rows(has_upper_flag, rows);
    auto true_rows = repeat_bool_scalar_rows(shared_true_bool_scalar(context), rows);
    sbMatrix lower_ok(rows, 1);
    sbMatrix upper_ok(rows, 1);
    {
        sbMatrix no_lower(rows, 1);
        bool_cipher_not(context.role, has_lower_repeat, no_lower);
        sbMatrix with_lower(rows, 1);
        sbMatrix without_lower(rows, 1);
        bool_cipher_and(context.role, has_lower_repeat, lower_match, with_lower, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, no_lower, true_rows, without_lower, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, with_lower, without_lower, lower_ok, *(context.enc), *(context.eval), *(context.runtime));
    }
    {
        sbMatrix no_upper(rows, 1);
        bool_cipher_not(context.role, has_upper_repeat, no_upper);
        sbMatrix with_upper(rows, 1);
        sbMatrix without_upper(rows, 1);
        bool_cipher_and(context.role, has_upper_repeat, upper_match, with_upper, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, no_upper, true_rows, without_upper, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, with_upper, without_upper, upper_ok, *(context.enc), *(context.eval), *(context.runtime));
    }

    sbMatrix out(rows, 1);
    bool_cipher_and(context.role, lower_ok, upper_ok, out, *(context.enc), *(context.eval), *(context.runtime));
    return out;
}

SecureRationalShare evaluate_indicator_oblivious_secure(
    const FlatBSPNModel& model,
    const SecureBoundFactor& factor,
    const FlatSecureQueryPayload& secure_payload,
    const FlatSecureQueryTensorPayload& shared_query_payload,
    const FlatBSPNSecureContext& context,
    SecureIndicatorEvalStats* eval_stats = nullptr) {
    (void)secure_payload;
    const auto& manifest = model.manifest();
    const auto& secret_payload = model.secret_shared_payload().dense_bucket_bitmaps_loaded
        ? model.secret_shared_payload()
        : throw std::runtime_error("secure shared payload not loaded");
    (void)secret_payload;
    const auto& node_records = model.nodes();
    const auto& child_ids = model.children();

    std::vector<SecureRationalShare> node_values(model.manifest().node_count);
    const auto zero = make_secure_rational(0.0, 1.0, context);
    const auto one = make_secure_rational(1.0, 1.0, context);
    std::fill(node_values.begin(), node_values.end(), zero);

    if (!shared_query_payload.shared_loaded) {
        throw std::runtime_error("shared query payload not loaded");
    }
    if (factor.secret_factor_binding_index < 0 ||
        static_cast<std::size_t>(factor.secret_factor_binding_index) >=
            static_cast<std::size_t>(shared_query_payload.feature_scope_shared.rows()) ||
        static_cast<std::size_t>(factor.secret_factor_binding_index) >=
            static_cast<std::size_t>(shared_query_payload.relevant_scope_shared.rows())) {
        throw std::runtime_error("secure factor scope binding not loaded");
    }
    const auto secret_factor_row = static_cast<std::uint32_t>(factor.secret_factor_binding_index);
    const auto secret_feature_scope = int_row_slice(shared_query_payload.feature_scope_shared, secret_factor_row, 1);
    const auto secret_relevant_scope = int_row_slice(shared_query_payload.relevant_scope_shared, secret_factor_row, 1);
    // Store count/count rational pairs in a public coarser unit. Scaling the
    // numerator and denominator by the same public value preserves the value
    // while reducing fixed-point overflow risk when rational denominators are
    // multiplied higher in the SPN.
    i64Matrix global_rows_plain(factor.factor.total_rows, 1);
    global_rows_plain.setOnes();
    sbMatrix global_rows_shared;
    share_bool_matrix(global_rows_plain, global_rows_shared, 0, context);

    for (std::size_t node_idx = 0; node_idx < model.manifest().node_count; ++node_idx) {
        const auto& node = node_records[node_idx];
        if (node.node_type == FlatBSPNNodeType::SUM) {
            const auto phase_start = SteadyClock::now();
            std::vector<SecureRationalShare> child_values;
            std::vector<sf64Matrix<kFlatBSPNDecimal>> child_weights;
            child_values.reserve(node.child_count);
            child_weights.reserve(node.child_count);
            for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
                const std::uint32_t child_id = child_ids[node.child_begin + offset];
                child_values.push_back(node_values[child_id]);
                child_weights.push_back(fixed_row_slice(model.secret_shared_payload().weights, node.weight_begin + offset, 1));
            }
            node_values[node_idx] = weighted_sum_secure_rational(child_values, child_weights, context);
            if (eval_stats != nullptr) {
                eval_stats->sum_node_ms += elapsed_ms_since(phase_start);
            }
            continue;
        }

        if (node.node_type == FlatBSPNNodeType::LEAF) {
            node_values[node_idx] = one;
            continue;
        }

        if (node.node_type != FlatBSPNNodeType::PRODUCT) {
            throw std::runtime_error("Unsupported node type in secure evaluation.");
        }

        if (node.child_count == 0) {
            node_values[node_idx] = one;
            continue;
        }

        const auto& first_child = node_records[child_ids[node.child_begin]];
        if (first_child.node_type == FlatBSPNNodeType::SUM) {
            const auto phase_start = SteadyClock::now();
            SecureRationalShare product = one;
            for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
                const auto& child = node_records[child_ids[node.child_begin + offset]];
                auto relevant_flag = secure_scope_intersects(child.scope_mask, secret_relevant_scope, context);
                const auto selected_child = select_rational_by_bool(node_values[child.node_id], one, relevant_flag, context);
                product = multiply_secure_rational(product, selected_child, context);
            }
            node_values[node_idx] = product;
            if (eval_stats != nullptr) {
                eval_stats->product_sum_ms += elapsed_ms_since(phase_start);
            }
            continue;
        }

        if (first_child.node_type != FlatBSPNNodeType::LEAF) {
            throw std::runtime_error("Unsupported product child shape in secure evaluation.");
        }

        std::vector<sbMatrix> local_ids;
        local_ids.reserve(node.child_count);
        std::vector<const FlatBSPNNodeRecord*> leaf_children;
        leaf_children.reserve(node.child_count);
        std::vector<sbMatrix> match_masks;
        match_masks.reserve(node.child_count);
        std::vector<sf64Matrix<kFlatBSPNDecimal>> target_numerator_sums;
        std::vector<sbMatrix> target_flags;
        sbMatrix has_target = shared_zero_bool_scalar(context);
        if (eval_stats != nullptr) {
            ++eval_stats->leaf_product_nodes;
        }
        auto phase_start = SteadyClock::now();
        for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
            const auto& child = node_records[child_ids[node.child_begin + offset]];
            leaf_children.push_back(&child);
            const int col_idx = child.leaf_column_id;
            sbMatrix is_target = secure_scope_intersects(child.scope_mask, secret_feature_scope, context);
            bool_cipher_or(context.role, has_target, is_target, has_target, *(context.enc), *(context.eval), *(context.runtime));

            sbMatrix match_mask(child.bucket_count, 1);
            bool has_predicate_binding = false;
            if (col_idx >= 0 &&
                static_cast<std::size_t>(col_idx) < factor.column_bindings.size() &&
                factor.column_bindings[static_cast<std::size_t>(col_idx)].slot_index >= 0) {
                const std::size_t slot_index = static_cast<std::size_t>(factor.column_bindings[static_cast<std::size_t>(col_idx)].slot_index);
                if (slot_index < static_cast<std::size_t>(shared_query_payload.lower_bounds_shared.rows())) {
                    has_predicate_binding = true;
                    const auto bucket_lowers = fixed_row_slice(
                        model.secret_shared_payload().bucket_lowers,
                        child.bucket_begin,
                        child.bucket_count);
                    const auto bucket_uppers = fixed_row_slice(
                        model.secret_shared_payload().bucket_uppers,
                        child.bucket_begin,
                        child.bucket_count);
                    bool_init_false(context.role, match_mask);

                    const auto interval_count = int_cell(
                        shared_query_payload.interval_counts_shared,
                        static_cast<std::uint32_t>(slot_index),
                        0);
                    const std::size_t max_interval_count =
                        static_cast<std::size_t>(shared_query_payload.lower_bounds_shared.cols());
                    for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
                        auto lower = fixed_cell(
                            shared_query_payload.lower_bounds_shared,
                            static_cast<std::uint32_t>(slot_index),
                            static_cast<std::uint32_t>(interval_idx));
                        auto upper = fixed_cell(
                            shared_query_payload.upper_bounds_shared,
                            static_cast<std::uint32_t>(slot_index),
                            static_cast<std::uint32_t>(interval_idx));
                        auto has_lower = shared_int_nonzero_flag(int_cell(
                            shared_query_payload.has_lower_shared,
                            static_cast<std::uint32_t>(slot_index),
                            static_cast<std::uint32_t>(interval_idx)), context);
                        auto has_upper = shared_int_nonzero_flag(int_cell(
                            shared_query_payload.has_upper_shared,
                            static_cast<std::uint32_t>(slot_index),
                            static_cast<std::uint32_t>(interval_idx)), context);
                        auto open_lower = shared_int_nonzero_flag(int_cell(
                            shared_query_payload.open_lower_shared,
                            static_cast<std::uint32_t>(slot_index),
                            static_cast<std::uint32_t>(interval_idx)), context);
                        auto open_upper = shared_int_nonzero_flag(int_cell(
                            shared_query_payload.open_upper_shared,
                            static_cast<std::uint32_t>(slot_index),
                            static_cast<std::uint32_t>(interval_idx)), context);

                        auto interval_match = secure_interval_match_mask(
                            bucket_lowers,
                            bucket_uppers,
                            lower,
                            upper,
                            has_lower,
                            has_upper,
                            open_lower,
                            open_upper,
                            context);
                        auto interval_idx_shared = share_int_scalar(
                            static_cast<i64>(interval_idx),
                            0,
                            context);
                        sbMatrix interval_active;
                        auto interval_count_copy = interval_count;
                        cipher_gt(context.role, interval_count_copy, interval_idx_shared, interval_active, *(context.eval), *(context.runtime));
                        auto interval_active_rows = repeat_bool_scalar_rows(interval_active, child.bucket_count);
                        sbMatrix active_match(child.bucket_count, 1);
                        bool_cipher_and(context.role, interval_match, interval_active_rows, active_match, *(context.enc), *(context.eval), *(context.runtime));
                        bool_cipher_or(context.role, match_mask, active_match, match_mask, *(context.enc), *(context.eval), *(context.runtime));
                    }

                    auto has_evidence = shared_int_nonzero_flag(int_cell(
                        shared_query_payload.has_evidence_shared,
                        static_cast<std::uint32_t>(slot_index),
                        0), context);
                    sbMatrix no_evidence(has_evidence.rows(), has_evidence.bitCount());
                    bool_cipher_not(context.role, has_evidence, no_evidence);
                    auto no_evidence_rows = repeat_bool_scalar_rows(no_evidence, child.bucket_count);
                    bool_cipher_or(context.role, match_mask, no_evidence_rows, match_mask, *(context.enc), *(context.eval), *(context.runtime));
                }
            }
            if (!has_predicate_binding) {
                bool_init_true(context.role, match_mask);
            }

            match_masks.push_back(std::move(match_mask));
            target_flags.push_back(std::move(is_target));
        }
        if (eval_stats != nullptr) {
            eval_stats->phase1_match_ms += elapsed_ms_since(phase_start);
        }
        phase_start = SteadyClock::now();
        local_ids = compute_leaf_local_ids_batched(
            model,
            leaf_children,
            match_masks,
            context,
            eval_stats != nullptr ? &eval_stats->phase1_batch_dot_calls : nullptr);
        if (eval_stats != nullptr) {
            eval_stats->phase1_local_ids_ms += elapsed_ms_since(phase_start);
        }

        phase_start = SteadyClock::now();
        sbMatrix final_ids = local_ids.empty() ? global_rows_shared : local_ids.front();
        for (std::size_t idx = 1; idx < local_ids.size(); ++idx) {
            sbMatrix next(final_ids.rows(), final_ids.bitCount());
            bool_cipher_and(context.role, final_ids, local_ids[idx], next, *(context.enc), *(context.eval), *(context.runtime));
            final_ids = std::move(next);
        }
        if (eval_stats != nullptr) {
            eval_stats->phase2_intersection_ms += elapsed_ms_since(phase_start);
        }

        phase_start = SteadyClock::now();
        auto final_cnt_int = sum_boolean_mask_to_int(final_ids, context);
        const auto final_cnt = si64_to_sf64(final_cnt_int);
        const auto node_cardinality = fixed_row_slice(
            model.secret_shared_payload().node_cardinalities,
            static_cast<std::uint32_t>(node_idx),
            1);
        auto total_rows_int = share_int_scalar(
            static_cast<i64>(factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows),
            0,
            context);
        sbMatrix is_full;
        cipher_eq(context.role, final_cnt_int, total_rows_int, is_full, *(context.eval), *(context.runtime));
        const auto effective_cnt = select_fixed_by_bool(node_cardinality, final_cnt, is_full, context);
        if (eval_stats != nullptr) {
            eval_stats->phase2_count_ms += elapsed_ms_since(phase_start);
        }
        phase_start = SteadyClock::now();
        target_numerator_sums = compute_leaf_target_numerator_sums_batched(
            model,
            leaf_children,
            final_ids,
            context,
            eval_stats != nullptr ? &eval_stats->phase3_batch_b2a_calls : nullptr);
        if (eval_stats != nullptr) {
            eval_stats->phase3_numerator_ms += elapsed_ms_since(phase_start);
        }
        phase_start = SteadyClock::now();
        auto zero_fixed = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
        const auto inv_cardinality = fixed_row_slice(
            model.secret_shared_payload().node_inv_cardinalities,
            static_cast<std::uint32_t>(node_idx),
            1);
        const auto selectivity_num = secure_mul_fixed(effective_cnt, inv_cardinality, context);

        if (factor.secure_leaf_eval_mode == "single_target_fast") {
            auto target_sum = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
            for (std::size_t child_idx = 0; child_idx < target_numerator_sums.size(); ++child_idx) {
                const auto selected_target_sum = select_fixed_by_bool(
                    target_numerator_sums[child_idx],
                    zero_fixed,
                    target_flags[child_idx],
                    context);
                target_sum += selected_target_sum;
            }
            const auto target_num = secure_mul_fixed(target_sum, inv_cardinality, context);
            const auto node_num = select_fixed_by_bool(target_num, selectivity_num, has_target, context);
            node_values[node_idx] = {
                node_num,
                share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
                1.0,
                1.0,
                true,
            };
            if (eval_stats != nullptr) {
                eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
            }
            continue;
        }
        if (factor.secure_leaf_eval_mode != "reciprocal_fallback") {
            throw std::runtime_error("Unknown secure_leaf_eval_mode: " + factor.secure_leaf_eval_mode);
        }

        auto node_cardinality_for_eq = node_cardinality;
        sbMatrix is_empty_node;
        cipher_eq(context.role, node_cardinality_for_eq, zero_fixed, is_empty_node, *(context.eval), *(context.runtime));
        auto denom_safe = effective_cnt + bool_scalar_to_fixed(is_empty_node, context);
        const SecureRationalShare selectivity_rational{
            selectivity_num,
            share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
            1.0,
            1.0,
            true,
        };

        SecureRationalShare exp_product = one;
        if (eval_stats != nullptr) {
            ++eval_stats->reciprocal_calls;
        }
        const auto inv_denom_safe = secure_count_reciprocal_piecewise(
            denom_safe,
            factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows,
            context);
        for (std::size_t child_idx = 0; child_idx < target_numerator_sums.size(); ++child_idx) {
            const SecureRationalShare component{
                secure_mul_fixed(target_numerator_sums[child_idx], inv_denom_safe, context),
                share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
                1.0,
                1.0,
                true,
            };
            auto base_value = select_rational_by_bool(component, one, target_flags[child_idx], context);
            auto factor_value = select_rational_by_bool(one, base_value, is_empty_node, context);
            exp_product = multiply_secure_rational(exp_product, factor_value, context);
        }
        const auto expectation_rational = multiply_secure_rational(exp_product, selectivity_rational, context);
        node_values[node_idx] = select_rational_by_bool(expectation_rational, selectivity_rational, has_target, context);
        if (eval_stats != nullptr) {
            eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
        }
    }

    const std::size_t root_idx = static_cast<std::size_t>(manifest.root_node_id);
    if (root_idx >= node_values.size()) {
        throw std::runtime_error("Root node id is out of bounds for secure evaluation.");
    }
    return node_values[root_idx];
}

SecureBundleExecutionResult evaluate_secure_bundle_impl_secure(
    const json& public_plan_doc,
    const FlatSecureQueryPayload& secure_payload,
    const FlatSecureQueryTensorPayload& shared_query_payload,
    const std::map<std::string, std::string>& manifest_map,
    const std::string& model_root,
    const FlatBSPNSecureContext& context,
    std::map<std::string, FlatBSPNModel>& model_cache) {
    json public_slot_by_id = json::object();
    for (const auto& slot : public_plan_doc.value("predicate_slots", json::array())) {
        public_slot_by_id[slot.at("slot_id").get<std::string>()] = slot;
    }

    std::map<std::string, std::size_t> slot_index_by_id;
    for (std::size_t idx = 0; idx < public_plan_doc.value("predicate_slots", json::array()).size(); ++idx) {
        const auto& slot = public_plan_doc["predicate_slots"][idx];
        slot_index_by_id[slot.at("slot_id").get<std::string>()] = idx;
    }

    std::map<std::string, std::size_t> secret_factor_index_by_id;
    std::size_t next_secret_factor_index = 0;
    auto register_factor_array = [&](const json& factors) {
        for (const auto& factor_doc : factors) {
            const std::string secret_factor_id = factor_doc.value("secret_factor_id", std::string());
            if (secret_factor_id.empty()) {
                continue;
            }
            if (secret_factor_index_by_id.find(secret_factor_id) == secret_factor_index_by_id.end()) {
                secret_factor_index_by_id.emplace(secret_factor_id, next_secret_factor_index++);
            }
        }
    };
    if (public_plan_doc.contains("cardinality_plan") && public_plan_doc["cardinality_plan"].is_object()) {
        register_factor_array(public_plan_doc["cardinality_plan"].value("factors", json::array()));
    }
    for (const auto& term_doc : public_plan_doc.value("aggregate_terms", json::array())) {
        if (term_doc.contains("expectation_plan")) {
            register_factor_array(term_doc["expectation_plan"].value("factors", json::array()));
        }
        if (term_doc.contains("numerator_plan")) {
            register_factor_array(term_doc["numerator_plan"].value("factors", json::array()));
        }
        if (term_doc.contains("denominator_plan")) {
            register_factor_array(term_doc["denominator_plan"].value("factors", json::array()));
        }
    }

    std::map<int, std::string> term_operation_by_index;
    std::string pending_operation;
    for (const auto& op_doc : public_plan_doc.value("aggregation_operations", json::array())) {
        const std::string op_type = op_doc.value("operation_type", std::string());
        if (op_type == "PLUS" || op_type == "MINUS") {
            pending_operation = op_type;
            continue;
        }
        if (op_type == "AGGREGATION") {
            term_operation_by_index[op_doc.value("term_index", -1)] = pending_operation;
            pending_operation.clear();
        }
    }

    auto eval_factor_product_secure = [&](const json& factors_doc, json* factor_debug) {
        SecureRationalShare product = make_secure_rational(1.0, 1.0, context);
        for (const auto& factor_doc : factors_doc) {
            auto bound = bind_secure_factor_from_secure_bundle(
                factor_doc,
                public_slot_by_id,
                slot_index_by_id,
                secret_factor_index_by_id,
                manifest_map,
                model_root,
                model_cache);
            SecureRationalShare factor_value = make_secure_rational(1.0, 1.0, context);
            SecureIndicatorEvalStats indicator_stats;
            if (bound.factor.factor_kind == "CONSTANT") {
                factor_value = make_secure_rational(bound.factor.public_constant_value, 1.0, context);
            } else if (bound.factor.factor_kind == "INDICATOR_EXPECTATION") {
                auto model_it = model_cache.find(bound.manifest_path);
                if (model_it == model_cache.end()) {
                    throw std::runtime_error("Missing preloaded model for secure execution: " + bound.manifest_path);
                }
                factor_value = evaluate_indicator_oblivious_secure(
                    model_it->second,
                    bound,
                    secure_payload,
                    shared_query_payload,
                    context,
                    &indicator_stats);
            } else {
                throw std::runtime_error(
                    "Secure production path currently supports CONSTANT and INDICATOR_EXPECTATION factors.");
            }
            factor_value = maybe_invert_secure_rational(factor_value, bound.factor.inverse);
            if (factor_debug != nullptr && context.debug_reveal) {
                const double numerator = reveal_scaled_numerator(factor_value, context);
                const double denominator = reveal_scaled_denominator(factor_value, context);
                factor_debug->push_back({
                    {"factor_index", bound.factor.factor_index},
                    {"factor_kind", bound.factor.factor_kind},
                    {"inverse", bound.factor.inverse},
                    {"model_id", bound.model_id},
                    {"secure_leaf_eval_mode", bound.secure_leaf_eval_mode},
                    {"reciprocal_calls", indicator_stats.reciprocal_calls},
                    {"phase1_batch_dot_calls", indicator_stats.phase1_batch_dot_calls},
                    {"phase3_batch_b2a_calls", indicator_stats.phase3_batch_b2a_calls},
                    {"leaf_product_nodes", indicator_stats.leaf_product_nodes},
                    {"timing_ms", {
                        {"sum_node", indicator_stats.sum_node_ms},
                        {"product_sum", indicator_stats.product_sum_ms},
                        {"phase1_match", indicator_stats.phase1_match_ms},
                        {"phase1_local_ids", indicator_stats.phase1_local_ids_ms},
                        {"phase2_intersection", indicator_stats.phase2_intersection_ms},
                        {"phase2_count", indicator_stats.phase2_count_ms},
                        {"phase3_numerator", indicator_stats.phase3_numerator_ms},
                        {"final_combine", indicator_stats.final_combine_ms},
                    }},
                    {"numerator", numerator},
                    {"denominator", denominator},
                    {"value", std::abs(denominator) <= 1e-12 ? 0.0 : numerator / denominator},
                });
            }
            product = multiply_secure_rational(product, factor_value, context);
        }
        return product;
    };

    SecureBundleExecutionResult out;
    SecureRationalShare cardinality_rational = make_secure_rational(0.0, 1.0, context);
    bool has_cardinality = false;
    json cardinality_factor_debug = json::array();
    if (public_plan_doc.contains("cardinality_plan") && public_plan_doc["cardinality_plan"].is_object()) {
        cardinality_rational = eval_factor_product_secure(
            public_plan_doc["cardinality_plan"].value("factors", json::array()),
            &cardinality_factor_debug);
        has_cardinality = true;
    }

    SecureRationalShare expectation_value = make_secure_rational(0.0, 1.0, context);
    SecureRationalShare aggregate_result = make_secure_rational(0.0, 1.0, context);
    bool has_expectation = false;
    bool has_aggregate_result = false;
    json term_debug = json::array();

    for (const auto& term_doc : public_plan_doc.value("aggregate_terms", json::array())) {
        const int term_index = term_doc.value("term_index", -1);
        const std::string aggregation_type = term_doc.value("aggregation_type", std::string());
        const std::string evaluation_mode = term_doc.value("evaluation_mode", std::string());
        SecureRationalShare current_value = make_secure_rational(0.0, 1.0, context);
        SecureRationalShare current_result_value = make_secure_rational(0.0, 1.0, context);

        if (evaluation_mode == "single_spn") {
            current_value = eval_factor_product_secure(
                term_doc.at("expectation_plan").value("factors", json::array()),
                nullptr);
            if (aggregation_type == "SUM") {
                if (!has_cardinality) {
                    throw std::runtime_error("Secure SUM single_spn requires cardinality.");
                }
                current_result_value = multiply_secure_rational(current_value, cardinality_rational, context);
            } else {
                current_result_value = current_value;
            }
        } else {
                auto numerator_rational = eval_factor_product_secure(
                    term_doc.at("numerator_plan").value("factors", json::array()),
                    nullptr);
                if (aggregation_type == "AVG") {
                    auto denominator_rational = eval_factor_product_secure(
                    term_doc.at("denominator_plan").value("factors", json::array()),
                    nullptr);
                current_value = multiply_secure_rational(
                    numerator_rational,
                    invert_secure_rational(denominator_rational),
                    context);
            } else {
                current_value = numerator_rational;
            }
            current_result_value = current_value;
        }

        if (!has_expectation) {
            expectation_value = current_value;
            aggregate_result = current_result_value;
            has_expectation = true;
            has_aggregate_result = true;
        } else {
            const std::string op =
                term_operation_by_index.count(term_index) != 0
                    ? term_operation_by_index.at(term_index)
                    : std::string("PLUS");
            if (op == "MINUS") {
                expectation_value = subtract_secure_rational(expectation_value, current_value, context);
                aggregate_result = subtract_secure_rational(aggregate_result, current_result_value, context);
            } else {
                expectation_value = add_secure_rational(expectation_value, current_value, context);
                aggregate_result = add_secure_rational(aggregate_result, current_result_value, context);
            }
        }

        term_debug.push_back({
            {"term_index", term_index},
            {"aggregation_type", aggregation_type},
            {"evaluation_mode", evaluation_mode},
        });
    }

    const std::string query_kind = public_plan_doc.value("query_kind", std::string());
    if (query_kind == "CARDINALITY") {
        out.result_rational = cardinality_rational;
        out.has_result = has_cardinality;
    } else if (has_aggregate_result) {
        out.result_rational = aggregate_result;
        out.has_result = true;
    } else if (has_cardinality) {
        out.result_rational = cardinality_rational;
        out.has_result = true;
    }

    out.debug_output = {
        {"query_skeleton_id", public_plan_doc.value("query_skeleton_id", std::string())},
        {"query_kind", query_kind},
        {"cardinality_factors", cardinality_factor_debug},
        {"aggregate_terms", term_debug},
    };
    return out;
}

void require_secure_bundle_cli_contract(const oc::CLP& cmd) {
    if (!cmd.isSet("role")) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --role in {0,1,2}.");
    }
    if (!cmd.isSet("public_plan_json")) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --public_plan_json.");
    }
    if (!cmd.isSet("secret_payload_json")) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --secret_payload_json.");
    }
    if (!cmd.isSet("bspn_model_root")) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --bspn_model_root.");
    }
}

void init_secure_context_from_cmd(
    const oc::CLP& cmd,
    FlatBSPNSecureContext& context,
    oc::IOService& ios,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime) {
    require_secure_bundle_cli_contract(cmd);
    const int role = cmd.getMany<int>("role")[0];
    if (role < 0 || role > 2) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --role in {0,1,2}.");
    }

    context.role = role;
    context.model_owner_party = cmd.isSet("model_owner_party") ? cmd.getMany<int>("model_owner_party")[0] : 0;
    context.query_owner_party = cmd.isSet("query_owner_party") ? cmd.getMany<int>("query_owner_party")[0] : 0;
    context.debug_reveal = cmd.isSet("debug_reveal");

    basic_setup(static_cast<u64>(role), ios, enc, eval, runtime);
    context.io_service = &ios;
    context.enc = &enc;
    context.eval = &eval;
    context.runtime = &runtime;
}

void collect_model_ids_from_factor_array(const json& factors, std::vector<std::string>& model_ids) {
    if (!factors.is_array()) {
        return;
    }
    for (const auto& factor_doc : factors) {
        if (!factor_doc.is_object()) {
            continue;
        }
        const std::string model_id = factor_doc.value("spn_model_id", std::string());
        if (!model_id.empty()) {
            model_ids.push_back(model_id);
        }
    }
}

std::vector<std::string> collect_secure_bundle_model_ids(const json& public_plan_doc) {
    std::vector<std::string> model_ids;
    if (public_plan_doc.contains("cardinality_plan") && public_plan_doc["cardinality_plan"].is_object()) {
        collect_model_ids_from_factor_array(public_plan_doc["cardinality_plan"].value("factors", json::array()), model_ids);
    }

    for (const auto& term_doc : public_plan_doc.value("aggregate_terms", json::array())) {
        if (!term_doc.is_object()) {
            continue;
        }
        if (term_doc.contains("expectation_plan") && term_doc["expectation_plan"].is_object()) {
            collect_model_ids_from_factor_array(term_doc["expectation_plan"].value("factors", json::array()), model_ids);
        }
        if (term_doc.contains("numerator_plan") && term_doc["numerator_plan"].is_object()) {
            collect_model_ids_from_factor_array(term_doc["numerator_plan"].value("factors", json::array()), model_ids);
        }
        if (term_doc.contains("denominator_plan") && term_doc["denominator_plan"].is_object()) {
            collect_model_ids_from_factor_array(term_doc["denominator_plan"].value("factors", json::array()), model_ids);
        }
    }

    std::sort(model_ids.begin(), model_ids.end());
    model_ids.erase(std::unique(model_ids.begin(), model_ids.end()), model_ids.end());
    return model_ids;
}

}  // namespace

// 读取模型
void FlatBSPNModel::load_public_manifest(const std::string& manifest_path) {
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open manifest: " + manifest_path);
    }

    manifest_path_ = manifest_path;
    base_dir_ = dirname_from_path(manifest_path);

    json manifest_doc;
    in >> manifest_doc;

    manifest_.model_id = manifest_doc.value("model_id", std::string());
    manifest_.table_name = manifest_doc.value("table_name", std::string());
    manifest_.column_names = manifest_doc.value("column_names", std::vector<std::string>());
    manifest_.column_types = manifest_doc.value("column_types", std::vector<std::string>());
    manifest_.node_count = manifest_doc.value("node_count", std::uint64_t(0));
    manifest_.root_node_id = manifest_doc.value("root_node_id", std::uint64_t(0));
    manifest_.total_rows = manifest_doc.value("total_rows", std::uint64_t(0));
    manifest_.scope_bitmap_bytes = manifest_doc.value("scope_bitmap_bytes", std::uint64_t(0));
    manifest_.children_count = manifest_doc.value("children_count", std::uint64_t(0));
    manifest_.bucket_count = manifest_doc.value("bucket_count", std::uint64_t(0));
    manifest_.weights_count = manifest_doc.value("weights_count", std::uint64_t(0));
    manifest_.leaf_bitmap_bytes = manifest_doc.value("leaf_bitmap_bytes", std::uint64_t(0));
    manifest_.leaf_bucket_width = manifest_doc.value("leaf_bucket_width", std::uint64_t(0));
    manifest_.leaf_node_count = manifest_doc.value("leaf_node_count", std::uint64_t(0));
    manifest_.node_cardinality_count = manifest_doc.value("node_cardinality_count", std::uint64_t(0));
    manifest_.node_inv_cardinality_count = manifest_doc.value("node_inv_cardinality_count", std::uint64_t(0));
    manifest_.secret_payload_dir = manifest_doc.value("secret_payload_dir", std::string("secret"));

    const auto raw_nodes = read_binary_records<PackedRawNodeRecord>(join_path(base_dir_, "nodes.bin"));
    const auto raw_buckets = read_binary_records<PackedRawBucketRecord>(join_path(base_dir_, "bucket_index.bin"));
    children_ = read_binary_records<std::uint32_t>(join_path(base_dir_, "children.bin"));
    const auto scope_blob = read_binary_bytes(join_path(base_dir_, "scope_bitmaps.bin"));

    nodes_.clear();
    nodes_.reserve(raw_nodes.size());
    for (const auto& raw : raw_nodes) {
        FlatBSPNNodeRecord node;
        node.node_id = raw.node_id;
        node.node_type = static_cast<FlatBSPNNodeType>(raw.node_type);
        node.cardinality = raw.cardinality;
        node.child_begin = raw.child_begin;
        node.child_count = raw.child_count;
        node.scope_bitmap_begin = raw.scope_bitmap_begin;
        node.scope_bitmap_len = raw.scope_bitmap_len;
        node.bucket_begin = raw.bucket_begin;
        node.bucket_count = raw.bucket_count;
        node.weight_begin = raw.weight_begin;
        node.weight_count = raw.weight_count;
        node.leaf_column_id = raw.leaf_column_id;

        if (node.scope_bitmap_begin + node.scope_bitmap_len > scope_blob.size()) {
            throw std::runtime_error("Node scope bitmap slice is out of bounds.");
        }
        const std::vector<std::uint8_t> packed_scope(
            scope_blob.begin() + static_cast<std::ptrdiff_t>(node.scope_bitmap_begin),
            scope_blob.begin() + static_cast<std::ptrdiff_t>(node.scope_bitmap_begin + node.scope_bitmap_len));
        node.scope_mask = unpack_scope_bits(packed_scope, manifest_.column_names.size());

        nodes_.push_back(node);
    }

    buckets_.clear();
    buckets_.reserve(raw_buckets.size());
    for (const auto& raw : raw_buckets) {
        FlatBSPNBucketRecord bucket;
        bucket.bucket_id = raw.bucket_id;
        bucket.bitmap_begin = raw.bitmap_begin;
        bucket.bitmap_len = raw.bitmap_len;
        bucket.value_index = raw.value_index;
        bucket.lower_bound_index = raw.lower_bound_index;
        bucket.upper_bound_index = raw.upper_bound_index;
        buckets_.push_back(bucket);
    }
}

void FlatBSPNModel::load_secret_payload() {
    if (base_dir_.empty()) {
        throw std::runtime_error("load_public_manifest must be called before load_secret_payload.");
    }

    const std::string secret_dir = join_path(base_dir_, manifest_.secret_payload_dir.empty() ? "secret" : manifest_.secret_payload_dir);
    weights_ = read_binary_doubles(join_path(secret_dir, "weights.bin"));
    bucket_values_ = read_binary_doubles(join_path(secret_dir, "bucket_values.bin"));
    leaf_bitmaps_ = read_binary_bytes(join_path(secret_dir, "leaf_bitmaps.bin"));
    node_cardinalities_ = read_binary_doubles(join_path(secret_dir, "node_cardinalities.bin"));
    node_inv_cardinalities_ = read_binary_doubles(join_path(secret_dir, "node_inv_cardinalities.bin"));

    if (manifest_.node_cardinality_count != 0 &&
        manifest_.node_cardinality_count != static_cast<std::uint64_t>(node_cardinalities_.size())) {
        throw std::runtime_error("node_cardinalities.bin count mismatch with manifest.");
    }
    if (manifest_.node_inv_cardinality_count != 0 &&
        manifest_.node_inv_cardinality_count != static_cast<std::uint64_t>(node_inv_cardinalities_.size())) {
        throw std::runtime_error("node_inv_cardinalities.bin count mismatch with manifest.");
    }

    std::vector<double> bucket_representatives;
    std::vector<double> bucket_lowers;
    std::vector<double> bucket_uppers;
    bucket_representatives.reserve(buckets_.size());
    bucket_lowers.reserve(buckets_.size());
    bucket_uppers.reserve(buckets_.size());
    for (const auto& bucket : buckets_) {
        if (bucket.value_index >= bucket_values_.size() ||
            bucket.lower_bound_index >= bucket_values_.size() ||
            bucket.upper_bound_index >= bucket_values_.size()) {
            throw std::runtime_error("Bucket value indices are out of bounds.");
        }
        bucket_representatives.push_back(bucket_values_[bucket.value_index]);
        bucket_lowers.push_back(bucket_values_[bucket.lower_bound_index]);
        bucket_uppers.push_back(bucket_values_[bucket.upper_bound_index]);
    }

    secret_host_payload_.node_cardinalities = doubles_to_fixed_column<kFlatBSPNDecimal>(node_cardinalities_);
    secret_host_payload_.node_inv_cardinalities = doubles_to_fixed_column<kFlatBSPNDecimal>(node_inv_cardinalities_);
    secret_host_payload_.weights = doubles_to_fixed_column<kFlatBSPNDecimal>(weights_);
    secret_host_payload_.bucket_values = doubles_to_fixed_column<kFlatBSPNDecimal>(bucket_representatives);
    secret_host_payload_.bucket_lowers = doubles_to_fixed_column<kFlatBSPNDecimal>(bucket_lowers);
    secret_host_payload_.bucket_uppers = doubles_to_fixed_column<kFlatBSPNDecimal>(bucket_uppers);
    secret_host_payload_.leaf_bitmaps = u8_to_i64_matrix(
        leaf_bitmaps_,
        manifest_.bucket_count,
        manifest_.leaf_bitmap_bytes);

    secret_shared_payload_ = FlatBSPNSecretSharedPayload{};
}

void FlatBSPNModel::load_secret_payload(const FlatBSPNSecureContext& context) {
    if (!context.has_runtime()) {
        throw std::runtime_error("FlatBSPNSecureContext runtime is not initialized.");
    }
    if (context.role == context.model_owner_party &&
        secret_host_payload_.node_cardinalities.size() == 0 &&
        !base_dir_.empty()) {
        load_secret_payload();
    }

    if (context.role != context.model_owner_party && secret_host_payload_.node_cardinalities.size() == 0) {
        secret_host_payload_.node_cardinalities = f64Matrix<kFlatBSPNDecimal>(manifest_.node_count, 1);
        secret_host_payload_.node_inv_cardinalities = f64Matrix<kFlatBSPNDecimal>(manifest_.node_count, 1);
        secret_host_payload_.weights = f64Matrix<kFlatBSPNDecimal>(manifest_.weights_count, 1);
        secret_host_payload_.bucket_values = f64Matrix<kFlatBSPNDecimal>(manifest_.bucket_count, 1);
        secret_host_payload_.bucket_lowers = f64Matrix<kFlatBSPNDecimal>(manifest_.bucket_count, 1);
        secret_host_payload_.bucket_uppers = f64Matrix<kFlatBSPNDecimal>(manifest_.bucket_count, 1);
        secret_host_payload_.leaf_bitmaps = i64Matrix(manifest_.bucket_count, manifest_.leaf_bitmap_bytes);
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.node_cardinalities.size()); ++idx) secret_host_payload_.node_cardinalities(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.node_inv_cardinalities.size()); ++idx) secret_host_payload_.node_inv_cardinalities(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.weights.size()); ++idx) secret_host_payload_.weights(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_values.size()); ++idx) secret_host_payload_.bucket_values(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_lowers.size()); ++idx) secret_host_payload_.bucket_lowers(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_uppers.size()); ++idx) secret_host_payload_.bucket_uppers(idx) = 0;
        secret_host_payload_.leaf_bitmaps.setZero();
    }

    share_fixed_matrix(secret_host_payload_.node_cardinalities, secret_shared_payload_.node_cardinalities, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.node_inv_cardinalities, secret_shared_payload_.node_inv_cardinalities, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.weights, secret_shared_payload_.weights, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_values, secret_shared_payload_.bucket_values, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_lowers, secret_shared_payload_.bucket_lowers, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_uppers, secret_shared_payload_.bucket_uppers, context.model_owner_party, context);
    share_int_matrix(secret_host_payload_.leaf_bitmaps, secret_shared_payload_.leaf_bitmaps, context.model_owner_party, context);

    secret_shared_payload_.dense_bucket_bitmaps.clear();
    secret_shared_payload_.dense_bucket_bitmaps.reserve(buckets_.size());
    for (const auto& bucket : buckets_) {
        i64Matrix dense_rows(manifest_.total_rows, 1);
        dense_rows.setZero();
        if (context.role == context.model_owner_party) {
            if (bucket.bitmap_begin + bucket.bitmap_len > leaf_bitmaps_.size()) {
                throw std::runtime_error("Leaf bitmap slice is out of bounds.");
            }
            const std::vector<std::uint8_t> bitmap_bytes(
                leaf_bitmaps_.begin() + static_cast<std::ptrdiff_t>(bucket.bitmap_begin),
                leaf_bitmaps_.begin() + static_cast<std::ptrdiff_t>(bucket.bitmap_begin + bucket.bitmap_len));
            dense_rows = unpack_bitmap_to_dense_rows(bitmap_bytes, manifest_.total_rows);
        }
        sbMatrix shared_bitmap;
        share_bool_matrix(dense_rows, shared_bitmap, context.model_owner_party, context);
        secret_shared_payload_.dense_bucket_bitmaps.push_back(std::move(shared_bitmap));
    }
    secret_shared_payload_.dense_bucket_bitmaps_loaded = true;
    secret_shared_payload_.loaded = true;
}

FlatSecureQueryPayload load_secure_query_payload_json(const std::string& payload_json_path) {
    std::ifstream in(payload_json_path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open secure query payload json: " + payload_json_path);
    }
    json doc;
    in >> doc;
    return parse_secure_query_payload_doc(doc);
}

FlatSecureQueryTensorPayload build_secure_query_tensor_payload(const FlatSecureQueryPayload& payload) {
    FlatSecureQueryTensorPayload tensors;
    const std::size_t slot_count = static_cast<std::size_t>(payload.slot_count);
    const std::size_t max_interval_count = static_cast<std::size_t>(payload.max_interval_count);
    const std::size_t factor_count = static_cast<std::size_t>(payload.factor_count);
    const std::size_t max_factor_columns = static_cast<std::size_t>(payload.max_factor_column_count);

    tensors.lower_bounds = f64Matrix<kFlatBSPNDecimal>(slot_count, max_interval_count);
    tensors.upper_bounds = f64Matrix<kFlatBSPNDecimal>(slot_count, max_interval_count);
    tensors.has_lower = i64Matrix(slot_count, max_interval_count);
    tensors.has_upper = i64Matrix(slot_count, max_interval_count);
    tensors.open_lower = i64Matrix(slot_count, max_interval_count);
    tensors.open_upper = i64Matrix(slot_count, max_interval_count);
    tensors.has_evidence = i64Matrix(slot_count, 1);
    tensors.interval_counts = i64Matrix(slot_count, 1);
    tensors.feature_scope = i64Matrix(factor_count, max_factor_columns);
    tensors.relevant_scope = i64Matrix(factor_count, max_factor_columns);
    tensors.feature_inverted_scope = i64Matrix(factor_count, max_factor_columns);
    tensors.factor_column_counts = i64Matrix(factor_count, 1);

    tensors.has_lower.setZero();
    tensors.has_upper.setZero();
    tensors.open_lower.setZero();
    tensors.open_upper.setZero();
    tensors.has_evidence.setZero();
    tensors.interval_counts.setZero();
    tensors.feature_scope.setZero();
    tensors.relevant_scope.setZero();
    tensors.feature_inverted_scope.setZero();
    tensors.factor_column_counts.setZero();

    for (std::size_t slot_idx = 0; slot_idx < payload.predicate_slot_bindings.size() && slot_idx < slot_count; ++slot_idx) {
        const auto& binding = payload.predicate_slot_bindings[slot_idx];
        tensors.has_evidence(static_cast<u64>(slot_idx), 0) = binding.has_evidence ? 1 : 0;
        tensors.interval_counts(static_cast<u64>(slot_idx), 0) = static_cast<i64>(binding.interval_count);

        for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
            const double lower = interval_idx < binding.lower_bounds.size() ? binding.lower_bounds[interval_idx] : 0.0;
            const double upper = interval_idx < binding.upper_bounds.size() ? binding.upper_bounds[interval_idx] : 0.0;
            tensors.lower_bounds(static_cast<u64>(slot_idx), static_cast<u64>(interval_idx)) = lower;
            tensors.upper_bounds(static_cast<u64>(slot_idx), static_cast<u64>(interval_idx)) = upper;
            tensors.has_lower(static_cast<u64>(slot_idx), static_cast<u64>(interval_idx)) =
                interval_idx < binding.has_lower.size() ? static_cast<i64>(binding.has_lower[interval_idx]) : 0;
            tensors.has_upper(static_cast<u64>(slot_idx), static_cast<u64>(interval_idx)) =
                interval_idx < binding.has_upper.size() ? static_cast<i64>(binding.has_upper[interval_idx]) : 0;
            tensors.open_lower(static_cast<u64>(slot_idx), static_cast<u64>(interval_idx)) =
                interval_idx < binding.open_lower.size() ? static_cast<i64>(binding.open_lower[interval_idx]) : 0;
            tensors.open_upper(static_cast<u64>(slot_idx), static_cast<u64>(interval_idx)) =
                interval_idx < binding.open_upper.size() ? static_cast<i64>(binding.open_upper[interval_idx]) : 0;
        }
    }

    for (std::size_t factor_idx = 0; factor_idx < payload.secret_factor_bindings.size() && factor_idx < factor_count; ++factor_idx) {
        const auto& binding = payload.secret_factor_bindings[factor_idx];
        tensors.factor_column_counts(static_cast<u64>(factor_idx), 0) = static_cast<i64>(binding.column_count);
        for (std::size_t col_idx = 0; col_idx < max_factor_columns; ++col_idx) {
            tensors.feature_scope(static_cast<u64>(factor_idx), static_cast<u64>(col_idx)) =
                col_idx < binding.feature_scope.size() ? static_cast<i64>(binding.feature_scope[col_idx]) : 0;
            tensors.relevant_scope(static_cast<u64>(factor_idx), static_cast<u64>(col_idx)) =
                col_idx < binding.relevant_scope.size() ? static_cast<i64>(binding.relevant_scope[col_idx]) : 0;
            tensors.feature_inverted_scope(static_cast<u64>(factor_idx), static_cast<u64>(col_idx)) =
                col_idx < binding.feature_inverted_scope.size() ? static_cast<i64>(binding.feature_inverted_scope[col_idx]) : 0;
        }
    }

    return tensors;
}

FlatSecureQueryPayload empty_secure_query_payload_from_public_doc(const json& public_doc) {
    FlatSecureQueryPayload payload;
    payload.query_skeleton_id = public_doc.value("query_skeleton_id", std::string());
    if (public_doc.contains("secret_tensor_shape") && public_doc["secret_tensor_shape"].is_object()) {
        const auto& shape_doc = public_doc["secret_tensor_shape"];
        if (shape_doc.contains("slot_payload_shape") && shape_doc["slot_payload_shape"].is_object()) {
            payload.slot_count = shape_doc["slot_payload_shape"].value("slot_count", std::uint64_t(0));
            payload.max_interval_count = shape_doc["slot_payload_shape"].value("max_interval_count", std::uint64_t(0));
        }
        if (shape_doc.contains("factor_payload_shape") && shape_doc["factor_payload_shape"].is_object()) {
            payload.factor_count = shape_doc["factor_payload_shape"].value("factor_count", std::uint64_t(0));
            payload.max_factor_column_count = shape_doc["factor_payload_shape"].value("max_column_count", std::uint64_t(0));
        }
    }
    return payload;
}

FlatSecureQueryTensorPayload share_secure_query_tensor_payload(
    const FlatSecureQueryPayload& payload,
    const FlatBSPNSecureContext& context) {
    if (!context.has_runtime()) {
        throw std::runtime_error("FlatBSPNSecureContext runtime is not initialized.");
    }

    FlatSecureQueryTensorPayload tensors = build_secure_query_tensor_payload(payload);
    share_fixed_matrix(tensors.lower_bounds, tensors.lower_bounds_shared, context.query_owner_party, context);
    share_fixed_matrix(tensors.upper_bounds, tensors.upper_bounds_shared, context.query_owner_party, context);
    share_int_matrix(tensors.has_lower, tensors.has_lower_shared, context.query_owner_party, context);
    share_int_matrix(tensors.has_upper, tensors.has_upper_shared, context.query_owner_party, context);
    share_int_matrix(tensors.open_lower, tensors.open_lower_shared, context.query_owner_party, context);
    share_int_matrix(tensors.open_upper, tensors.open_upper_shared, context.query_owner_party, context);
    share_int_matrix(tensors.has_evidence, tensors.has_evidence_shared, context.query_owner_party, context);
    share_int_matrix(tensors.interval_counts, tensors.interval_counts_shared, context.query_owner_party, context);
    share_int_matrix(tensors.feature_scope, tensors.feature_scope_shared, context.query_owner_party, context);
    share_int_matrix(tensors.relevant_scope, tensors.relevant_scope_shared, context.query_owner_party, context);
    share_int_matrix(tensors.feature_inverted_scope, tensors.feature_inverted_scope_shared, context.query_owner_party, context);
    share_int_matrix(tensors.factor_column_counts, tensors.factor_column_counts_shared, context.query_owner_party, context);
    tensors.shared_loaded = true;
    return tensors;
}

void BSPN_flat_eval(const oc::CLP& cmd) {
    const std::string public_plan_path = cmd.getOr<std::string>("public_plan_json", "");
    const std::string secret_payload_path = cmd.getOr<std::string>("secret_payload_json", "");
    if (public_plan_path.empty() || secret_payload_path.empty()) {
        throw std::runtime_error(
            "bspn_flat_eval now supports only the secure bundle contract: "
            "--role, --public_plan_json, --secret_payload_json, and --bspn_model_root.");
    }
    BSPN_secure_bundle_eval(cmd);
}

void BSPN_secure_bundle_eval(const oc::CLP& cmd) {
    const std::string public_plan_path = cmd.getOr<std::string>("public_plan_json", "");
    const std::string secret_payload_path = cmd.getOr<std::string>("secret_payload_json", "");
    if (public_plan_path.empty()) {
        throw std::runtime_error("public_plan_json not set");
    }
    if (secret_payload_path.empty()) {
        throw std::runtime_error("secret_payload_json not set");
    }

    std::ifstream public_in(public_plan_path);
    if (!public_in.is_open()) {
        throw std::runtime_error("Could not open public plan json: " + public_plan_path);
    }
    json public_doc;
    public_in >> public_doc;

    oc::IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    FlatBSPNSecureContext secure_context;
    init_secure_context_from_cmd(cmd, secure_context, ios, enc, eval, runtime);

    const std::string model_root = normalize_model_root(cmd.getOr<std::string>("bspn_model_root", ""));
    std::map<std::string, std::string> manifest_map;
    const std::string manifest_map_path = cmd.getOr<std::string>("model_manifest_map_json", "");
    if (!manifest_map_path.empty()) {
        std::ifstream map_in(manifest_map_path);
        if (!map_in.is_open()) {
            throw std::runtime_error("Could not open model manifest map json: " + manifest_map_path);
        }
        json map_doc;
        map_in >> map_doc;
        manifest_map = load_model_manifest_map(map_doc, dirname_from_path(manifest_map_path));
    }

    json secret_doc = json::object();
    FlatSecureQueryPayload secure_payload = empty_secure_query_payload_from_public_doc(public_doc);
    if (secure_context.role == secure_context.query_owner_party) {
        std::ifstream secret_in(secret_payload_path);
        if (!secret_in.is_open()) {
            throw std::runtime_error("Could not open secret payload json: " + secret_payload_path);
        }
        secret_in >> secret_doc;
        secure_payload = parse_secure_query_payload_doc(secret_doc);
    }
    if (cmd.isSet("debug_bundle_rational") || cmd.isSet("debug_factor_rational") || cmd.isSet("debug_oblivious_trace")) {
        throw std::runtime_error("Plaintext/debug FlatBSPN evaluators have been removed from the production frontend.");
    }

    auto shared_query_payload = share_secure_query_tensor_payload(secure_payload, secure_context);

    std::map<std::string, FlatBSPNModel> preloaded_model_cache;
    for (const auto& model_id : collect_secure_bundle_model_ids(public_doc)) {
        const auto manifest_it = manifest_map.find(model_id);
        const std::string manifest_path =
            manifest_it != manifest_map.end()
                ? manifest_it->second
                : default_manifest_path_for_model(model_root, model_id);
        FlatBSPNModel model;
        model.load_public_manifest(manifest_path);
        model.load_secret_payload(secure_context);
        preloaded_model_cache.emplace(manifest_path, std::move(model));
    }

    const auto secure_eval_start = std::chrono::steady_clock::now();
    const auto secure_eval = evaluate_secure_bundle_impl_secure(
        public_doc,
        secure_payload,
        shared_query_payload,
        manifest_map,
        model_root,
        secure_context,
        preloaded_model_cache);
    const auto secure_eval_end = std::chrono::steady_clock::now();
    const double secure_eval_wall_time_ms =
        std::chrono::duration<double, std::milli>(secure_eval_end - secure_eval_start).count();

    json out = {
        {"query_skeleton_id", public_doc.value("query_skeleton_id", std::string())},
        {"query_kind", public_doc.value("query_kind", std::string())},
        {"secure_evaluator_wall_time_ms", secure_eval_wall_time_ms},
        {"result", nullptr},
    };

    if (secure_context.debug_reveal && secure_eval.has_result) {
        const double numerator = reveal_scaled_numerator(secure_eval.result_rational, secure_context);
        const double denominator = reveal_scaled_denominator(secure_eval.result_rational, secure_context);
        out["result"] = std::abs(denominator) > 1e-12 ? (numerator / denominator) : 0.0;
        out["debug"] = secure_eval.debug_output;
    }
    std::cout << out.dump(2) << std::endl;
}

}  // namespace aby3

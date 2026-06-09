#include "FlatBSPN.h"
#include "aby3-Basic/BuildingBlocks.h"
#include "aby3-GORAM-Core/Basics.h"
#include <aby3/sh3/Sh3Piecewise.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace aby3 {

namespace {

using json = nlohmann::json;

std::size_t flat_bspn_bucket_batch_size() {
    const char* raw_value = std::getenv("BSPN_BUCKET_BATCH_SIZE");
    if (raw_value == nullptr || std::string(raw_value).empty()) {
        return 64;
    }

    const std::string text(raw_value);
    std::size_t parsed_chars = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(text, &parsed_chars);
    } catch (const std::exception&) {
        throw std::runtime_error("BSPN_BUCKET_BATCH_SIZE must be a positive integer.");
    }

    if (parsed_chars != text.size() || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("BSPN_BUCKET_BATCH_SIZE must be a positive integer.");
    }
    return static_cast<std::size_t>(parsed);
}

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

bool path_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
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
    bool has_secret_non_unit_denominator = false;
    sbMatrix secret_non_unit_denominator;
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

struct SecureBoundFactor {
    std::string model_id;
    std::string manifest_path;
    FlatFactorSpec factor;
    int secret_factor_binding_index = -1;
};

struct SecureBundleExecutionResult {
    SecureRationalShare result_rational;
    bool has_result = false;
    double root_division_payload_scale = 1.0;
    json debug_output;
};

struct SecureIndicatorEvalStats {
    std::uint64_t internal_reciprocal_calls = 0;
    std::uint64_t factor_root_divisions = 0;
    std::uint64_t phase1_batch_dot_calls = 0;
    std::uint64_t phase1_match_batches = 0;
    std::uint64_t phase2_count_batches = 0;
    std::uint64_t phase3_batch_b2a_calls = 0;
    std::uint64_t leaf_product_groups = 0;
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

boolShare row0_bool_share(const sbMatrix& mat);

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

SecureRationalShare scale_secure_rational_public(
    const SecureRationalShare& value,
    double public_factor,
    const FlatBSPNSecureContext& context);

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

si64Matrix shared_query_factor_zero_scope(
    const FlatSecureQueryTensorPayload& shared_query_payload,
    std::uint32_t factor_row) {
    if (factor_row >= static_cast<std::uint32_t>(shared_query_payload.feature_scope_shared.rows())) {
        throw std::runtime_error("factor scope row is out of bounds");
    }
    return shared_zero_int_matrix(1, static_cast<u64>(shared_query_payload.feature_scope_shared.cols()));
}

si64Matrix shared_query_factor_evidence_scope(
    const FlatSecureQueryTensorPayload& shared_query_payload,
    std::uint32_t factor_row) {
    const u64 cols = static_cast<u64>(shared_query_payload.feature_scope_shared.cols());
    si64Matrix out = shared_zero_int_matrix(1, cols);
    for (u64 col = 0; col < cols; ++col) {
        const u64 evidence_row = static_cast<u64>(factor_row) * cols + col;
        if (evidence_row >= static_cast<u64>(shared_query_payload.has_evidence_shared.rows())) {
            continue;
        }
        out.mShares[0](0, col) = shared_query_payload.has_evidence_shared.mShares[0](evidence_row, 0);
        out.mShares[1](0, col) = shared_query_payload.has_evidence_shared.mShares[1](evidence_row, 0);
    }
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

sbMatrix shared_public_bool_scalar(bool value, const FlatBSPNSecureContext& context) {
    return value ? shared_true_bool_scalar(context) : shared_zero_bool_scalar(context);
}

sbMatrix bool_or_scalar(
    const sbMatrix& lhs,
    const sbMatrix& rhs,
    const FlatBSPNSecureContext& context) {
    sbMatrix out(1, 1);
    auto lhs_copy = lhs;
    auto rhs_copy = rhs;
    bool_cipher_or(context.role, lhs_copy, rhs_copy, out, *(context.enc), *(context.eval), *(context.runtime));
    return out;
}

sbMatrix bool_and_scalar(
    const sbMatrix& lhs,
    const sbMatrix& rhs,
    const FlatBSPNSecureContext& context) {
    sbMatrix out(1, 1);
    auto lhs_copy = lhs;
    auto rhs_copy = rhs;
    bool_cipher_and(context.role, lhs_copy, rhs_copy, out, *(context.enc), *(context.eval), *(context.runtime));
    return out;
}

sbMatrix bool_not_scalar(
    const sbMatrix& value,
    const FlatBSPNSecureContext& context) {
    sbMatrix out(1, 1);
    auto value_copy = value;
    bool_cipher_not(context.role, value_copy, out);
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

sbMatrix rational_non_unit_denominator_flag(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context) {
    if (value.denominator_is_one) {
        return shared_zero_bool_scalar(context);
    }
    if (value.has_secret_non_unit_denominator) {
        return value.secret_non_unit_denominator;
    }
    return shared_true_bool_scalar(context);
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
    auto out = normalize_secure_rational_scales({
        select_fixed_by_bool(aligned_true.numerator, aligned_false.numerator, flag, context),
        select_fixed_by_bool(aligned_true.denominator, aligned_false.denominator, flag, context),
        target_numerator_scale,
        target_denominator_scale,
        true_value.denominator_is_one && false_value.denominator_is_one,
    });
    if (!out.denominator_is_one) {
        const auto true_non_unit = rational_non_unit_denominator_flag(aligned_true, context);
        const auto false_non_unit = rational_non_unit_denominator_flag(aligned_false, context);
        const auto true_selected = bool_and_scalar(flag, true_non_unit, context);
        const auto false_selected = bool_and_scalar(bool_not_scalar(flag, context), false_non_unit, context);
        out.has_secret_non_unit_denominator = true;
        out.secret_non_unit_denominator = bool_or_scalar(true_selected, false_selected, context);
    }
    return out;
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

sbMatrix secure_scope_intersects_shared(
    const si64Matrix& secret_node_scope_row,
    const si64Matrix& secret_scope_row,
    const FlatBSPNSecureContext& context) {
    si64Matrix total = shared_zero_int_scalar();
    const std::size_t limit = std::min<std::size_t>(
        static_cast<std::size_t>(secret_node_scope_row.cols()),
        static_cast<std::size_t>(secret_scope_row.cols()));
    for (std::size_t idx = 0; idx < limit; ++idx) {
        auto node_bit = int_cell(secret_node_scope_row, 0, static_cast<std::uint32_t>(idx));
        auto scope_bit = int_cell(secret_scope_row, 0, static_cast<std::uint32_t>(idx));
        si64Matrix product(1, 1);
        cipher_mul(
            context.role,
            node_bit,
            scope_bit,
            product,
            *(context.eval),
            *(context.enc),
            *(context.runtime));
        total = total + product;
    }
    return shared_int_nonzero_flag(std::move(total), context);
}

si64Matrix secret_node_scope_row(
    const FlatBSPNModel& model,
    std::uint32_t node_id) {
    if (model.secret_shared_payload().node_scopes.rows() == 0 ||
        node_id >= static_cast<std::uint32_t>(model.secret_shared_payload().node_scopes.rows())) {
        throw std::runtime_error("secure node scope payload not loaded");
    }
    return int_row_slice(model.secret_shared_payload().node_scopes, node_id, 1);
}

std::vector<sbMatrix> secure_scope_intersects_shared_rows(
    const FlatBSPNModel& model,
    const std::vector<std::uint32_t>& node_ids,
    const si64Matrix& secret_scope_row,
    const FlatBSPNSecureContext& context) {
    std::vector<sbMatrix> out;
    out.reserve(node_ids.size());
    if (node_ids.empty()) {
        return out;
    }
    const auto& node_scopes = model.secret_shared_payload().node_scopes;
    if (node_scopes.rows() == 0) {
        throw std::runtime_error("secure node scope payload not loaded");
    }
    const u64 rows = static_cast<u64>(node_ids.size());
    const u64 cols = std::min<u64>(
        static_cast<u64>(node_scopes.cols()),
        static_cast<u64>(secret_scope_row.cols()));
    if (cols == 0) {
        for (std::size_t idx = 0; idx < node_ids.size(); ++idx) {
            (void)idx;
            out.push_back(shared_zero_bool_scalar(context));
        }
        return out;
    }

    si64Matrix child_scopes(rows, cols);
    si64Matrix repeated_scope(rows, cols);
    for (u64 row = 0; row < rows; ++row) {
        const auto node_id = node_ids[static_cast<std::size_t>(row)];
        if (node_id >= static_cast<std::uint32_t>(node_scopes.rows())) {
            throw std::runtime_error("secure node scope row is out of bounds");
        }
        for (u64 col = 0; col < cols; ++col) {
            child_scopes.mShares[0](row, col) = node_scopes.mShares[0](node_id, col);
            child_scopes.mShares[1](row, col) = node_scopes.mShares[1](node_id, col);
            repeated_scope.mShares[0](row, col) = secret_scope_row.mShares[0](0, col);
            repeated_scope.mShares[1](row, col) = secret_scope_row.mShares[1](0, col);
        }
    }

    si64Matrix products(rows, cols);
    cipher_mul(
        context.role,
        child_scopes,
        repeated_scope,
        products,
        *(context.eval),
        *(context.enc),
        *(context.runtime));

    si64Matrix row_totals(rows, 1);
    for (u64 row = 0; row < rows; ++row) {
        row_totals.mShares[0](row, 0) = products.mShares[0].block(row, 0, 1, cols).sum();
        row_totals.mShares[1](row, 0) = products.mShares[1].block(row, 0, 1, cols).sum();
    }
    auto zeros = shared_zero_int_matrix(rows, 1);
    sbMatrix flags;
    cipher_gt(context.role, row_totals, zeros, flags, *(context.eval), *(context.runtime));

    for (u64 row = 0; row < rows; ++row) {
        sbMatrix flag(1, flags.bitCount());
        for (u64 col = 0; col < static_cast<u64>(flags.mShares[0].cols()); ++col) {
            flag.mShares[0](0, col) = flags.mShares[0](row, col);
            flag.mShares[1](0, col) = flags.mShares[1](row, col);
        }
        out.push_back(std::move(flag));
    }
    return out;
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

sf64Matrix<kFlatBSPNDecimal> secure_mul_fixed_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& lhs,
    const sf64Matrix<kFlatBSPNDecimal>& rhs,
    const FlatBSPNSecureContext& context) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        throw std::runtime_error("fixed-point matrix shapes do not match for batched multiplication");
    }
    auto lhs_copy = lhs;
    auto rhs_copy = rhs;
    sf64Matrix<kFlatBSPNDecimal> out(lhs.rows(), lhs.cols());
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

sf64Matrix<kFlatBSPNDecimal> repeat_fixed_scalar_matrix(
    const sf64Matrix<kFlatBSPNDecimal>& scalar,
    u64 rows,
    u64 cols) {
    sf64Matrix<kFlatBSPNDecimal> out(rows, cols);
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < cols; ++col) {
            out[0](row, col) = scalar[0](0, 0);
            out[1](row, col) = scalar[1](0, 0);
        }
    }
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

sf64Matrix<kFlatBSPNDecimal> secure_fixed_reciprocal_newton(
    const sf64Matrix<kFlatBSPNDecimal>& value,
    const FlatBSPNSecureContext& context,
    std::size_t iterations = 24) {
    auto y = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
    const auto two = share_fixed_scalar<kFlatBSPNDecimal>(2.0, 0, context);
    for (std::size_t idx = 0; idx < iterations; ++idx) {
        const auto xy = secure_mul_fixed(value, y, context);
        const auto correction = two - xy;
        y = secure_mul_fixed(y, correction, context);
    }
    return y;
}

sf64Matrix<kFlatBSPNDecimal> secure_fixed_reciprocal_newton_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& value,
    const FlatBSPNSecureContext& context,
    std::size_t iterations = 24) {
    auto y = repeat_fixed_scalar_matrix(
        share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
        value.rows(),
        value.cols());
    const auto two = repeat_fixed_scalar_matrix(
        share_fixed_scalar<kFlatBSPNDecimal>(2.0, 0, context),
        value.rows(),
        value.cols());
    for (std::size_t idx = 0; idx < iterations; ++idx) {
        const auto xy = secure_mul_fixed_same_shape(value, y, context);
        const auto correction = two - xy;
        y = secure_mul_fixed_same_shape(y, correction, context);
    }
    return y;
}

sf64Matrix<kFlatBSPNDecimal> secure_count_reciprocal_newton_scaled_matrix(
    const sf64Matrix<kFlatBSPNDecimal>& count,
    std::uint64_t max_count,
    const FlatBSPNSecureContext& context,
    std::size_t iterations = 12) {
    const double public_inv_max =
        max_count == 0 ? 1.0 : 1.0 / static_cast<double>(max_count);
    auto inv_max_matrix = repeat_fixed_scalar_matrix(
        share_fixed_scalar<kFlatBSPNDecimal>(public_inv_max, 0, context),
        count.rows(),
        count.cols());
    const auto scaled_count = secure_mul_fixed_same_shape(count, inv_max_matrix, context);
    const auto inv_scaled_count =
        secure_fixed_reciprocal_newton_same_shape(scaled_count, context, iterations);
    inv_max_matrix = repeat_fixed_scalar_matrix(
        share_fixed_scalar<kFlatBSPNDecimal>(public_inv_max, 0, context),
        count.rows(),
        count.cols());
    return secure_mul_fixed_same_shape(inv_scaled_count, inv_max_matrix, context);
}

sf64Matrix<kFlatBSPNDecimal> secure_count_reciprocal_newton_scaled(
    const sf64Matrix<kFlatBSPNDecimal>& count,
    std::uint64_t max_count,
    const FlatBSPNSecureContext& context,
    std::size_t iterations = 12) {
    const double public_inv_max =
        max_count == 0 ? 1.0 : 1.0 / static_cast<double>(max_count);
    const auto scaled_count = secure_mul_public_fixed(count, public_inv_max, context);
    const auto inv_scaled_count = secure_fixed_reciprocal_newton(scaled_count, context, iterations);
    return secure_mul_public_fixed(inv_scaled_count, public_inv_max, context);
}

SecureRationalShare normalize_factor_root_rational(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context,
    double public_payload_scale = 1.0) {
    // Newton reciprocal starts from 1, so scale the root rational payload into
    // a small public unit before the single online division. The numerator and
    // denominator are scaled equally, preserving the represented value while
    // avoiding divergence for query-level denominators in the hundreds/thousands.
    const double payload_scale =
        std::isfinite(public_payload_scale) && public_payload_scale > 0.0
            ? public_payload_scale
            : 1.0;
    const auto scaled_value = scale_secure_rational_public(
        value,
        payload_scale,
        context);
    if (context.debug_reveal) {
        const double denominator = reveal_scaled_denominator(scaled_value, context);
        if (!std::isfinite(denominator) || std::abs(denominator) <= 1e-12) {
            throw std::runtime_error("Secure factor root denominator is too small for root-only division.");
        }
    }
    const auto inv_den = secure_fixed_reciprocal_newton(scaled_value.denominator, context, 32);
    const auto scalar = secure_mul_fixed(scaled_value.numerator, inv_den, context);
    return {
        scalar,
        share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
        scaled_value.numerator_scale / scaled_value.denominator_scale,
        1.0,
        true,
    };
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
        return normalize_secure_rational_scales(std::move(out));
    } else {
        out.denominator = secure_mul_fixed(lhs.denominator, rhs.denominator, context);
        out.denominator_scale = lhs.denominator_scale * rhs.denominator_scale;
    }
    out = normalize_secure_rational_scales(std::move(out));
    const auto lhs_non_unit = rational_non_unit_denominator_flag(lhs, context);
    const auto rhs_non_unit = rational_non_unit_denominator_flag(rhs, context);
    const auto non_unit = bool_or_scalar(lhs_non_unit, rhs_non_unit, context);
    SecureRationalShare scalar_out{
        out.numerator,
        share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
        out.numerator_scale,
        1.0,
        true,
    };
    return select_rational_by_bool(out, scalar_out, non_unit, context);
}

SecureRationalShare scale_secure_rational(
    const SecureRationalShare& value,
    const sf64Matrix<kFlatBSPNDecimal>& public_scale,
    const FlatBSPNSecureContext& context) {
    auto out = normalize_secure_rational_scales({
        secure_mul_fixed(value.numerator, public_scale, context),
        secure_mul_fixed(value.denominator, public_scale, context),
        value.numerator_scale,
        value.denominator_scale,
    });
    out.denominator_is_one = value.denominator_is_one;
    out.has_secret_non_unit_denominator = value.has_secret_non_unit_denominator;
    out.secret_non_unit_denominator = value.secret_non_unit_denominator;
    return out;
}

SecureRationalShare scale_secure_rational_public(
    const SecureRationalShare& value,
    double public_factor,
    const FlatBSPNSecureContext& context) {
    auto out = normalize_secure_rational_scales({
        secure_mul_public_fixed(value.numerator, public_factor, context),
        secure_mul_public_fixed(value.denominator, public_factor, context),
        value.numerator_scale / public_factor,
        value.denominator_scale / public_factor,
    });
    out.denominator_is_one = value.denominator_is_one;
    out.has_secret_non_unit_denominator = value.has_secret_non_unit_denominator;
    out.secret_non_unit_denominator = value.secret_non_unit_denominator;
    return out;
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
    auto out = normalize_secure_rational_scales(
        {value.denominator, value.numerator, value.denominator_scale, value.numerator_scale});
    out.denominator_is_one = false;
    return out;
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
    const std::size_t max_interval_count = static_cast<std::size_t>(
        binding_doc.value("max_interval_count", std::uint64_t(0)));

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
    binding.has_evidence = read_u8_vector("has_evidence");

    for (const auto& item : binding_doc.value("interval_counts", json::array())) {
        binding.interval_counts.push_back(static_cast<std::uint64_t>(item.get<std::uint64_t>()));
    }
    binding.interval_counts.resize(max_column_count, 0);

    auto read_double_matrix = [&](const char* key) {
        std::vector<std::vector<double>> out;
        for (const auto& row_doc : binding_doc.value(key, json::array())) {
            std::vector<double> row;
            for (const auto& item : row_doc) {
                row.push_back(item.get<double>());
            }
            row.resize(max_interval_count, 0.0);
            out.push_back(std::move(row));
        }
        out.resize(max_column_count, std::vector<double>(max_interval_count, 0.0));
        return out;
    };
    auto read_u8_matrix = [&](const char* key) {
        std::vector<std::vector<std::uint8_t>> out;
        for (const auto& row_doc : binding_doc.value(key, json::array())) {
            std::vector<std::uint8_t> row;
            for (const auto& item : row_doc) {
                row.push_back(static_cast<std::uint8_t>(item.get<int>()));
            }
            row.resize(max_interval_count, 0);
            out.push_back(std::move(row));
        }
        out.resize(max_column_count, std::vector<std::uint8_t>(max_interval_count, 0));
        return out;
    };

    binding.lower_bounds = read_double_matrix("lower_bounds");
    binding.upper_bounds = read_double_matrix("upper_bounds");
    binding.has_lower = read_u8_matrix("has_lower");
    binding.has_upper = read_u8_matrix("has_upper");
    binding.open_lower = read_u8_matrix("open_lower");
    binding.open_upper = read_u8_matrix("open_upper");
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
        payload.max_interval_count = std::max<std::uint64_t>(
            payload.max_interval_count,
            doc["factor_payload_shape"].value("max_interval_count", std::uint64_t(0)));
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
        if (payload.max_interval_count == 0) {
            for (const auto& binding_doc : doc["secret_factor_bindings_dense"]) {
                payload.max_interval_count = std::max<std::uint64_t>(
                    payload.max_interval_count,
                    binding_doc.value("max_interval_count", std::uint64_t(0)));
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
    const std::size_t batch_size = flat_bspn_bucket_batch_size();

    for (std::size_t batch_begin = 0; batch_begin < bucket_refs.size(); batch_begin += batch_size) {
        const std::size_t batch_end = std::min(bucket_refs.size(), batch_begin + batch_size);
        const std::size_t chunk_count = batch_end - batch_begin;
        std::vector<std::uint32_t> bucket_indices;
        bucket_indices.reserve(chunk_count);
        for (std::size_t bucket_idx = batch_begin; bucket_idx < batch_end; ++bucket_idx) {
            bucket_indices.push_back(bucket_refs[bucket_idx].bucket_index);
        }

        sbMatrix stacked_bitmaps = model.share_bucket_bitmap_stack(bucket_indices, context);
        if (stacked_bitmaps.rows() != mask_rows * static_cast<u64>(chunk_count) ||
            stacked_bitmaps.bitCount() != bit_count) {
            throw std::runtime_error("Bucket bitmap stack shape does not match final id mask shape.");
        }

        sbMatrix stacked_final_ids(stacked_bitmaps.rows(), bit_count);
        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            const u64 row_begin = static_cast<u64>(local_idx) * mask_rows;
            for (u64 row = 0; row < mask_rows; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    stacked_final_ids.mShares[0](row_begin + row, col) = final_ids.mShares[0](row, col);
                    stacked_final_ids.mShares[1](row_begin + row, col) = final_ids.mShares[1](row, col);
                }
            }
        }

        sbMatrix stacked_overlap(stacked_bitmaps.rows(), bit_count);
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

        si64Matrix overlap_counts_int(static_cast<u64>(chunk_count), 1);
        sf64Matrix<kFlatBSPNDecimal> bucket_values(static_cast<u64>(chunk_count), 1);
        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            const u64 row_begin = static_cast<u64>(local_idx) * mask_rows;
            overlap_counts_int.mShares[0](static_cast<u64>(local_idx), 0) =
                overlap_int.mShares[0].block(row_begin, 0, mask_rows, 1).sum();
            overlap_counts_int.mShares[1](static_cast<u64>(local_idx), 0) =
                overlap_int.mShares[1].block(row_begin, 0, mask_rows, 1).sum();
            const auto bucket_index = bucket_refs[batch_begin + local_idx].bucket_index;
            bucket_values[0](static_cast<u64>(local_idx), 0) =
                model.secret_shared_payload().bucket_values[0](bucket_index, 0);
            bucket_values[1](static_cast<u64>(local_idx), 0) =
                model.secret_shared_payload().bucket_values[1](bucket_index, 0);
        }

        auto overlap_counts = si64_to_sf64(overlap_counts_int);
        sf64Matrix<kFlatBSPNDecimal> bucket_contributions(static_cast<u64>(chunk_count), 1);
        cipher_mul(
            context.role,
            bucket_values,
            overlap_counts,
            bucket_contributions,
            *(context.eval),
            *(context.enc),
            *(context.runtime));

        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            auto& numerator_sum = numerator_sums[bucket_refs[batch_begin + local_idx].child_idx];
            numerator_sum[0](0, 0) += bucket_contributions[0](static_cast<u64>(local_idx), 0);
            numerator_sum[1](0, 0) += bucket_contributions[1](static_cast<u64>(local_idx), 0);
        }
    }

    return numerator_sums;
}

si64Matrix sum_boolean_masks_to_int_batched(
    const std::vector<sbMatrix>& masks,
    const FlatBSPNSecureContext& context,
    std::uint64_t* phase2_batch_counter = nullptr) {
    if (masks.empty()) {
        return si64Matrix(0, 1);
    }
    const u64 mask_rows = masks.front().rows();
    const u64 bit_count = masks.front().bitCount();
    const u64 share_cols = masks.front().mShares[0].cols();
    const u64 stacked_rows = mask_rows * static_cast<u64>(masks.size());
    sbMatrix stacked_masks(stacked_rows, bit_count);
    for (std::size_t mask_idx = 0; mask_idx < masks.size(); ++mask_idx) {
        const auto& mask = masks[mask_idx];
        if (mask.rows() != mask_rows || mask.bitCount() != bit_count) {
            throw std::runtime_error("Boolean mask shapes do not match for batched count.");
        }
        const u64 row_begin = static_cast<u64>(mask_idx) * mask_rows;
        for (u64 row = 0; row < mask_rows; ++row) {
            for (u64 col = 0; col < share_cols; ++col) {
                stacked_masks.mShares[0](row_begin + row, col) = mask.mShares[0](row, col);
                stacked_masks.mShares[1](row_begin + row, col) = mask.mShares[1](row, col);
            }
        }
    }

    si64Matrix stacked_int(stacked_rows, 1);
    bool2arith(context.role, stacked_masks, stacked_int, *(context.enc), *(context.eval), *(context.runtime));
    if (phase2_batch_counter != nullptr) {
        ++(*phase2_batch_counter);
    }

    si64Matrix counts(static_cast<u64>(masks.size()), 1);
    for (std::size_t mask_idx = 0; mask_idx < masks.size(); ++mask_idx) {
        const u64 row_begin = static_cast<u64>(mask_idx) * mask_rows;
        counts.mShares[0](static_cast<u64>(mask_idx), 0) =
            stacked_int.mShares[0].block(row_begin, 0, mask_rows, 1).sum();
        counts.mShares[1](static_cast<u64>(mask_idx), 0) =
            stacked_int.mShares[1].block(row_begin, 0, mask_rows, 1).sum();
    }
    return counts;
}

std::vector<sf64Matrix<kFlatBSPNDecimal>> compute_leaf_target_numerator_sums_group_batched(
    const FlatBSPNModel& model,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const std::vector<std::size_t>& leaf_product_indices,
    const std::vector<sbMatrix>& final_ids_by_product,
    const FlatBSPNSecureContext& context,
    std::uint64_t* phase3_batch_counter = nullptr) {
    if (leaf_children.size() != leaf_product_indices.size()) {
        throw std::runtime_error("Leaf child and product index counts do not match.");
    }
    std::vector<sf64Matrix<kFlatBSPNDecimal>> numerator_sums;
    numerator_sums.reserve(leaf_children.size());
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        (void)child_idx;
        numerator_sums.push_back(share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context));
    }

    struct BucketRef {
        std::size_t child_idx = 0;
        std::size_t product_idx = 0;
        std::uint32_t bucket_index = 0;
    };
    std::vector<BucketRef> bucket_refs;
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto product_idx = leaf_product_indices[child_idx];
        if (product_idx >= final_ids_by_product.size()) {
            throw std::runtime_error("Leaf product index is out of bounds.");
        }
        const auto& child = *leaf_children[child_idx];
        for (std::uint32_t bucket_offset = 0; bucket_offset < child.bucket_count; ++bucket_offset) {
            bucket_refs.push_back({child_idx, product_idx, child.bucket_begin + bucket_offset});
        }
    }
    if (bucket_refs.empty()) {
        return numerator_sums;
    }

    const auto& shape_mask = final_ids_by_product[bucket_refs.front().product_idx];
    const u64 mask_rows = shape_mask.rows();
    const u64 bit_count = shape_mask.bitCount();
    const u64 share_cols = shape_mask.mShares[0].cols();
    const std::size_t batch_size = flat_bspn_bucket_batch_size();

    for (std::size_t batch_begin = 0; batch_begin < bucket_refs.size(); batch_begin += batch_size) {
        const std::size_t batch_end = std::min(bucket_refs.size(), batch_begin + batch_size);
        const std::size_t chunk_count = batch_end - batch_begin;
        std::vector<std::uint32_t> bucket_indices;
        bucket_indices.reserve(chunk_count);
        for (std::size_t bucket_idx = batch_begin; bucket_idx < batch_end; ++bucket_idx) {
            bucket_indices.push_back(bucket_refs[bucket_idx].bucket_index);
        }

        sbMatrix stacked_bitmaps = model.share_bucket_bitmap_stack(bucket_indices, context);
        if (stacked_bitmaps.rows() != mask_rows * static_cast<u64>(chunk_count) ||
            stacked_bitmaps.bitCount() != bit_count) {
            throw std::runtime_error("Bucket bitmap stack shape mismatch in group numerator batch.");
        }

        sbMatrix stacked_final_ids(stacked_bitmaps.rows(), bit_count);
        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            const auto& bucket_ref = bucket_refs[batch_begin + local_idx];
            const auto& final_ids = final_ids_by_product[bucket_ref.product_idx];
            if (final_ids.rows() != mask_rows || final_ids.bitCount() != bit_count) {
                throw std::runtime_error("Final id mask shape mismatch in group numerator batch.");
            }
            const u64 row_begin = static_cast<u64>(local_idx) * mask_rows;
            for (u64 row = 0; row < mask_rows; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    stacked_final_ids.mShares[0](row_begin + row, col) = final_ids.mShares[0](row, col);
                    stacked_final_ids.mShares[1](row_begin + row, col) = final_ids.mShares[1](row, col);
                }
            }
        }

        sbMatrix stacked_overlap(stacked_bitmaps.rows(), bit_count);
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

        si64Matrix overlap_counts_int(static_cast<u64>(chunk_count), 1);
        sf64Matrix<kFlatBSPNDecimal> bucket_values(static_cast<u64>(chunk_count), 1);
        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            const u64 row_begin = static_cast<u64>(local_idx) * mask_rows;
            overlap_counts_int.mShares[0](static_cast<u64>(local_idx), 0) =
                overlap_int.mShares[0].block(row_begin, 0, mask_rows, 1).sum();
            overlap_counts_int.mShares[1](static_cast<u64>(local_idx), 0) =
                overlap_int.mShares[1].block(row_begin, 0, mask_rows, 1).sum();
            const auto bucket_index = bucket_refs[batch_begin + local_idx].bucket_index;
            bucket_values[0](static_cast<u64>(local_idx), 0) =
                model.secret_shared_payload().bucket_values[0](bucket_index, 0);
            bucket_values[1](static_cast<u64>(local_idx), 0) =
                model.secret_shared_payload().bucket_values[1](bucket_index, 0);
        }

        auto overlap_counts = si64_to_sf64(overlap_counts_int);
        sf64Matrix<kFlatBSPNDecimal> bucket_contributions(static_cast<u64>(chunk_count), 1);
        cipher_mul(
            context.role,
            bucket_values,
            overlap_counts,
            bucket_contributions,
            *(context.eval),
            *(context.enc),
            *(context.runtime));

        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            auto& numerator_sum = numerator_sums[bucket_refs[batch_begin + local_idx].child_idx];
            numerator_sum[0](0, 0) += bucket_contributions[0](static_cast<u64>(local_idx), 0);
            numerator_sum[1](0, 0) += bucket_contributions[1](static_cast<u64>(local_idx), 0);
        }
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

    std::size_t total_buckets = 0;
    struct BucketRef {
        std::size_t child_idx = 0;
        std::uint32_t bucket_offset = 0;
        std::uint32_t bucket_index = 0;
    };
    std::vector<BucketRef> bucket_refs;
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        if (match_masks[child_idx].rows() != child.bucket_count) {
            throw std::runtime_error("Match mask row count does not match leaf bucket count.");
        }
        total_buckets += child.bucket_count;
        for (std::uint32_t bucket_offset = 0; bucket_offset < child.bucket_count; ++bucket_offset) {
            bucket_refs.push_back({child_idx, bucket_offset, child.bucket_begin + bucket_offset});
        }
    }

    if (total_buckets == 0) {
        for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
            sbMatrix child_local_ids(0, 64);
            bool_init_false(context.role, child_local_ids);
            local_ids.push_back(std::move(child_local_ids));
        }
        return local_ids;
    }

    const u64 block_len = static_cast<u64>(model.manifest().total_rows);
    const u64 bit_count = 64;

    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        sbMatrix child_local_ids(block_len, bit_count);
        bool_init_false(context.role, child_local_ids);
        local_ids.push_back(std::move(child_local_ids));
    }

    const std::size_t batch_size = flat_bspn_bucket_batch_size();
    for (std::size_t batch_begin = 0; batch_begin < bucket_refs.size(); batch_begin += batch_size) {
        const std::size_t batch_end = std::min(bucket_refs.size(), batch_begin + batch_size);
        const std::size_t chunk_count = batch_end - batch_begin;
        std::vector<std::uint32_t> bucket_indices;
        bucket_indices.reserve(chunk_count);
        for (std::size_t bucket_idx = batch_begin; bucket_idx < batch_end; ++bucket_idx) {
            bucket_indices.push_back(bucket_refs[bucket_idx].bucket_index);
        }

        sbMatrix stacked_bitmaps = model.share_bucket_bitmap_stack(bucket_indices, context);
        if (stacked_bitmaps.rows() != block_len * static_cast<u64>(chunk_count) ||
            stacked_bitmaps.bitCount() != bit_count) {
            throw std::runtime_error("Bucket bitmap stack shape mismatch in batched local id computation.");
        }

        const u64 share_cols = stacked_bitmaps.mShares[0].cols();
        sbMatrix stacked_match(stacked_bitmaps.rows(), bit_count);
        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            const auto& bucket_ref = bucket_refs[batch_begin + local_idx];
            const auto& match_mask = match_masks[bucket_ref.child_idx];
            const bool expand_one_bit_mask = match_mask.bitCount() == 1;
            const u64 row_begin = static_cast<u64>(local_idx) * block_len;
            i64 match_share0 = 0;
            i64 match_share1 = 0;
            if (expand_one_bit_mask) {
                match_share0 = (match_mask.mShares[0](bucket_ref.bucket_offset, 0) == 1) ? -1 : 0;
                match_share1 = (match_mask.mShares[1](bucket_ref.bucket_offset, 0) == 1) ? -1 : 0;
            }
            for (u64 row = 0; row < block_len; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    if (expand_one_bit_mask) {
                        stacked_match.mShares[0](row_begin + row, col) = match_share0;
                        stacked_match.mShares[1](row_begin + row, col) = match_share1;
                    } else {
                        stacked_match.mShares[0](row_begin + row, col) =
                            match_mask.mShares[0](bucket_ref.bucket_offset, col);
                        stacked_match.mShares[1](row_begin + row, col) =
                            match_mask.mShares[1](bucket_ref.bucket_offset, col);
                    }
                }
            }
        }

        sbMatrix stacked_products(stacked_bitmaps.rows(), bit_count);
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

        for (std::size_t local_idx = 0; local_idx < chunk_count; ++local_idx) {
            const auto& bucket_ref = bucket_refs[batch_begin + local_idx];
            auto& child_local_ids = local_ids[bucket_ref.child_idx];
            const u64 row_begin = static_cast<u64>(local_idx) * block_len;
            for (u64 row = 0; row < block_len; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    child_local_ids.mShares[0](row, col) ^= stacked_products.mShares[0](row_begin + row, col);
                    child_local_ids.mShares[1](row, col) ^= stacked_products.mShares[1](row_begin + row, col);
                }
            }
        }
    }

    return local_ids;
}

SecureBoundFactor bind_secure_factor_from_secure_bundle(
    const json& factor_doc,
    const std::map<std::string, std::size_t>& secret_factor_index_by_id,
    const std::map<std::string, std::string>& manifest_map,
    const std::string& model_root,
    std::map<std::string, FlatBSPNModel>& model_cache) {
    SecureBoundFactor bound;
    bound.factor.factor_index = factor_doc.value("factor_index", -1);
    bound.factor.factor_kind = factor_doc.value("factor_kind", std::string());
    bound.factor.inverse = factor_doc.value("inverse", false);
    bound.factor.public_constant_value = factor_doc.value("public_constant_value", 0.0);
    bound.factor.public_feature_count = factor_doc.value("public_feature_count", std::uint64_t(0));
    bound.factor.public_evidence_count = factor_doc.value("public_evidence_count", std::uint64_t(0));

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

    const std::string secret_factor_id = factor_doc.value("secret_factor_id", std::string());
    if (!secret_factor_id.empty()) {
        auto secret_factor_it = secret_factor_index_by_id.find(secret_factor_id);
        if (secret_factor_it == secret_factor_index_by_id.end()) {
            throw std::runtime_error("Missing secret factor binding index for factor: " + secret_factor_id);
        }
        bound.secret_factor_binding_index = static_cast<int>(secret_factor_it->second);
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
	    sbMatrix any_non_unit = shared_zero_bool_scalar(context);
	    double scalar_numerator_scale = 1.0;
	    std::vector<sf64Matrix<kFlatBSPNDecimal>> scalar_terms(values.size());
	    std::vector<double> scalar_term_scales(values.size(), 1.0);
	    for (std::size_t idx = 0; idx < values.size(); ++idx) {
	        any_non_unit = bool_or_scalar(
	            any_non_unit,
	            rational_non_unit_denominator_flag(values[idx], context),
	            context);
	        scalar_terms[idx] = secure_mul_fixed(values[idx].numerator, weights[idx], context);
	        scalar_term_scales[idx] = values[idx].numerator_scale;
	        scalar_numerator_scale = std::max(scalar_numerator_scale, scalar_term_scales[idx]);
	    }
	    auto scalar_numerator_sum = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
	    for (std::size_t idx = 0; idx < scalar_terms.size(); ++idx) {
	        scalar_numerator_sum += secure_mul_public_fixed(
	            scalar_terms[idx],
	            scalar_term_scales[idx] / scalar_numerator_scale,
	            context);
	    }
	    const SecureRationalShare scalar_total{
	        scalar_numerator_sum,
	        share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
	        scalar_numerator_scale,
	        1.0,
	        true,
	    };
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
	    return select_rational_by_bool(total, scalar_total, any_non_unit, context);
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

sbMatrix secure_interval_match_mask_rows(
    const sf64Matrix<kFlatBSPNDecimal>& bucket_lowers,
    const sf64Matrix<kFlatBSPNDecimal>& bucket_uppers,
    const sf64Matrix<kFlatBSPNDecimal>& interval_lowers,
    const sf64Matrix<kFlatBSPNDecimal>& interval_uppers,
    const sbMatrix& has_lower_rows,
    const sbMatrix& has_upper_rows,
    const sbMatrix& open_lower_rows,
    const sbMatrix& open_upper_rows,
    const FlatBSPNSecureContext& context) {
    const auto rows = static_cast<std::uint32_t>(bucket_lowers.rows());

    auto bucket_lowers_copy = bucket_lowers;
    auto bucket_uppers_copy = bucket_uppers;
    auto interval_lowers_copy = interval_lowers;
    auto interval_uppers_copy = interval_uppers;
    auto has_lower_copy = has_lower_rows;
    auto has_upper_copy = has_upper_rows;
    auto open_lower_copy = open_lower_rows;
    auto open_upper_copy = open_upper_rows;
    sbMatrix ge_lower;
    cipher_ge(context.role, bucket_lowers_copy.i64Cast(), interval_lowers_copy.i64Cast(), ge_lower, *(context.eval), *(context.enc), *(context.runtime));
    bucket_lowers_copy = bucket_lowers;
    interval_lowers_copy = interval_lowers;
    sbMatrix gt_lower;
    cipher_gt(context.role, bucket_lowers_copy, interval_lowers_copy, gt_lower, *(context.eval), *(context.runtime));

    sbMatrix lower_match(rows, 1);
    {
        sbMatrix not_open_lower(rows, 1);
        bool_cipher_not(context.role, open_lower_copy, not_open_lower);
        sbMatrix ge_case(rows, 1);
        sbMatrix gt_case(rows, 1);
        bool_cipher_and(context.role, ge_lower, not_open_lower, ge_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, gt_lower, open_lower_copy, gt_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, ge_case, gt_case, lower_match, *(context.enc), *(context.eval), *(context.runtime));
    }

    sbMatrix ge_upper;
    cipher_ge(context.role, interval_uppers_copy.i64Cast(), bucket_uppers_copy.i64Cast(), ge_upper, *(context.eval), *(context.enc), *(context.runtime));
    bucket_uppers_copy = bucket_uppers;
    interval_uppers_copy = interval_uppers;
    sbMatrix gt_upper;
    cipher_gt(context.role, interval_uppers_copy, bucket_uppers_copy, gt_upper, *(context.eval), *(context.runtime));

    sbMatrix upper_match(rows, 1);
    {
        sbMatrix not_open_upper(rows, 1);
        bool_cipher_not(context.role, open_upper_copy, not_open_upper);
        sbMatrix ge_case(rows, 1);
        sbMatrix gt_case(rows, 1);
        bool_cipher_and(context.role, ge_upper, not_open_upper, ge_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, gt_upper, open_upper_copy, gt_case, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, ge_case, gt_case, upper_match, *(context.enc), *(context.eval), *(context.runtime));
    }

    auto true_rows = repeat_bool_scalar_rows(shared_true_bool_scalar(context), rows);
    sbMatrix lower_ok(rows, 1);
    sbMatrix upper_ok(rows, 1);
    {
        sbMatrix no_lower(rows, 1);
        bool_cipher_not(context.role, has_lower_copy, no_lower);
        sbMatrix with_lower(rows, 1);
        sbMatrix without_lower(rows, 1);
        bool_cipher_and(context.role, has_lower_copy, lower_match, with_lower, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_and(context.role, no_lower, true_rows, without_lower, *(context.enc), *(context.eval), *(context.runtime));
        bool_cipher_or(context.role, with_lower, without_lower, lower_ok, *(context.enc), *(context.eval), *(context.runtime));
    }
    {
        sbMatrix no_upper(rows, 1);
        bool_cipher_not(context.role, has_upper_copy, no_upper);
        sbMatrix with_upper(rows, 1);
        sbMatrix without_upper(rows, 1);
        bool_cipher_and(context.role, has_upper_copy, upper_match, with_upper, *(context.enc), *(context.eval), *(context.runtime));
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
    SecureIndicatorEvalStats* eval_stats = nullptr,
    const si64Matrix* feature_scope_override = nullptr,
    const si64Matrix* relevant_scope_override = nullptr) {
    (void)secure_payload;
    const auto& manifest = model.manifest();
    const auto& secret_payload = model.secret_shared_payload().loaded
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
	    const si64Matrix secret_feature_scope =
	        feature_scope_override != nullptr
	            ? *feature_scope_override
	            : int_row_slice(shared_query_payload.feature_scope_shared, secret_factor_row, 1);
	    const si64Matrix secret_relevant_scope =
	        relevant_scope_override != nullptr
	            ? *relevant_scope_override
	            : int_row_slice(shared_query_payload.relevant_scope_shared, secret_factor_row, 1);
    std::size_t public_factor_feature_count = static_cast<std::size_t>(factor.factor.public_feature_count);
    std::size_t public_factor_evidence_count = static_cast<std::size_t>(factor.factor.public_evidence_count);
    const bool public_single_target_factor = public_factor_feature_count <= 1;
    // Store count/count rational pairs in a public coarser unit. Scaling the
    // numerator and denominator by the same public value preserves the value
    // while reducing fixed-point overflow risk when rational denominators are
    // multiplied higher in the SPN.
    if (public_factor_feature_count == 0 && public_factor_evidence_count == 0) {
        return one;
    }

    i64Matrix global_rows_plain(factor.factor.total_rows, 1);
    global_rows_plain.setOnes();
    sbMatrix global_rows_shared;
    share_bool_matrix(global_rows_plain, global_rows_shared, 0, context);

    std::vector<bool> leaf_product_precomputed(model.manifest().node_count, false);
    struct LeafProductEvalItem {
        std::size_t node_idx = 0;
        std::size_t leaf_begin = 0;
        std::size_t leaf_count = 0;
    };
    std::vector<LeafProductEvalItem> leaf_product_items;
    std::vector<const FlatBSPNNodeRecord*> group_leaf_children;
    std::vector<std::uint32_t> group_leaf_node_ids;
    std::vector<std::size_t> group_leaf_product_indices;
    for (std::size_t node_idx = 0; node_idx < model.manifest().node_count; ++node_idx) {
        const auto& node = node_records[node_idx];
        if (node.node_type != FlatBSPNNodeType::PRODUCT || node.child_count == 0) {
            continue;
        }
        const auto& first_child = node_records[child_ids[node.child_begin]];
        if (first_child.node_type != FlatBSPNNodeType::LEAF) {
            continue;
        }
        const std::size_t product_idx = leaf_product_items.size();
        LeafProductEvalItem item;
        item.node_idx = node_idx;
        item.leaf_begin = group_leaf_children.size();
        item.leaf_count = node.child_count;
        leaf_product_items.push_back(item);
        for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
            const auto child_id = child_ids[node.child_begin + offset];
            group_leaf_children.push_back(&node_records[child_id]);
            group_leaf_node_ids.push_back(child_id);
            group_leaf_product_indices.push_back(product_idx);
        }
    }

    if (!leaf_product_items.empty()) {
        if (eval_stats != nullptr) {
            eval_stats->leaf_product_nodes += static_cast<std::uint64_t>(leaf_product_items.size());
            ++eval_stats->leaf_product_groups;
        }

        struct LeafEvidenceSelection {
            sbMatrix leaf_has_evidence;
            std::vector<sbMatrix> interval_active;
            std::vector<sf64Matrix<kFlatBSPNDecimal>> lower;
            std::vector<sf64Matrix<kFlatBSPNDecimal>> upper;
            std::vector<sbMatrix> has_lower;
            std::vector<sbMatrix> has_upper;
            std::vector<sbMatrix> open_lower;
            std::vector<sbMatrix> open_upper;
        };

        auto bool_row_scalar = [](const sbMatrix& flags, u64 row) {
            sbMatrix out(1, flags.bitCount());
            for (u64 col = 0; col < static_cast<u64>(flags.mShares[0].cols()); ++col) {
                out.mShares[0](0, col) = flags.mShares[0](row, col);
                out.mShares[1](0, col) = flags.mShares[1](row, col);
            }
            return out;
        };
        auto fixed_row_scalar = [](const sf64Matrix<kFlatBSPNDecimal>& values, u64 row) {
            sf64Matrix<kFlatBSPNDecimal> out(1, values.cols());
            out[0] = values[0].block(row, 0, 1, values.cols());
            out[1] = values[1].block(row, 0, 1, values.cols());
            return out;
        };

        const std::size_t max_columns = static_cast<std::size_t>(secret_feature_scope.cols());
        const std::size_t max_interval_count =
            static_cast<std::size_t>(shared_query_payload.lower_bounds_shared.cols());
        const u64 leaf_rows = static_cast<u64>(group_leaf_children.size());
        const u64 evidence_cols = static_cast<u64>(max_columns);

        auto target_flags = secure_scope_intersects_shared_rows(
            model,
            group_leaf_node_ids,
            secret_feature_scope,
            context);
        std::vector<sbMatrix> has_target_by_product;
        has_target_by_product.reserve(leaf_product_items.size());
        for (std::size_t product_idx = 0; product_idx < leaf_product_items.size(); ++product_idx) {
            sbMatrix has_target = shared_zero_bool_scalar(context);
            const auto& item = leaf_product_items[product_idx];
            for (std::size_t offset = 0; offset < item.leaf_count; ++offset) {
                sbMatrix updated(1, 1);
                bool_cipher_or(
                    context.role,
                    has_target,
                    target_flags[item.leaf_begin + offset],
                    updated,
                    *(context.enc),
                    *(context.eval),
                    *(context.runtime));
                has_target = std::move(updated);
            }
            has_target_by_product.push_back(std::move(has_target));
        }

        std::vector<sbMatrix> match_masks;
        match_masks.reserve(group_leaf_children.size());
        for (const auto* child : group_leaf_children) {
            sbMatrix match_mask(child->bucket_count, 1);
            bool_init_false(context.role, match_mask);
            match_masks.push_back(std::move(match_mask));
        }

        std::vector<LeafEvidenceSelection> evidence_selections(group_leaf_children.size());
        for (auto& selected : evidence_selections) {
            selected.interval_active.reserve(max_interval_count);
            selected.lower.reserve(max_interval_count);
            selected.upper.reserve(max_interval_count);
            selected.has_lower.reserve(max_interval_count);
            selected.has_upper.reserve(max_interval_count);
            selected.open_lower.reserve(max_interval_count);
            selected.open_upper.reserve(max_interval_count);
        }

        auto phase_start = SteadyClock::now();
        if (leaf_rows != 0 && evidence_cols != 0) {
            const auto& node_scopes = model.secret_shared_payload().node_scopes;
            if (node_scopes.rows() == 0) {
                throw std::runtime_error("secure node scope payload not loaded");
            }

            si64Matrix leaf_scope_matrix(leaf_rows, evidence_cols);
            leaf_scope_matrix.mShares[0].setZero();
            leaf_scope_matrix.mShares[1].setZero();
            for (u64 row = 0; row < leaf_rows; ++row) {
                const auto node_id = group_leaf_children[static_cast<std::size_t>(row)]->node_id;
                if (node_id >= static_cast<std::uint32_t>(node_scopes.rows())) {
                    throw std::runtime_error("secure leaf scope row is out of bounds");
                }
                const u64 copy_cols = std::min<u64>(evidence_cols, static_cast<u64>(node_scopes.cols()));
                for (u64 col = 0; col < copy_cols; ++col) {
                    leaf_scope_matrix.mShares[0](row, col) = node_scopes.mShares[0](node_id, col);
                    leaf_scope_matrix.mShares[1](row, col) = node_scopes.mShares[1](node_id, col);
                }
            }

            auto build_repeated_int_column = [&](const si64Matrix& src, std::uint32_t interval_col) {
                si64Matrix repeated(leaf_rows, evidence_cols);
                repeated.mShares[0].setZero();
                repeated.mShares[1].setZero();
                for (u64 col = 0; col < evidence_cols; ++col) {
                    const std::size_t evidence_row =
                        static_cast<std::size_t>(secret_factor_row) * max_columns + static_cast<std::size_t>(col);
                    if (evidence_row >= static_cast<std::size_t>(src.rows()) ||
                        interval_col >= static_cast<std::uint32_t>(src.cols())) {
                        continue;
                    }
                    for (u64 row = 0; row < leaf_rows; ++row) {
                        repeated.mShares[0](row, col) = src.mShares[0](static_cast<u64>(evidence_row), interval_col);
                        repeated.mShares[1](row, col) = src.mShares[1](static_cast<u64>(evidence_row), interval_col);
                    }
                }
                return repeated;
            };
            auto build_repeated_fixed_column = [&](const sf64Matrix<kFlatBSPNDecimal>& src, std::uint32_t interval_col) {
                sf64Matrix<kFlatBSPNDecimal> repeated(leaf_rows, evidence_cols);
                repeated[0].setZero();
                repeated[1].setZero();
                for (u64 col = 0; col < evidence_cols; ++col) {
                    const std::size_t evidence_row =
                        static_cast<std::size_t>(secret_factor_row) * max_columns + static_cast<std::size_t>(col);
                    if (evidence_row >= static_cast<std::size_t>(src.rows()) ||
                        interval_col >= static_cast<std::uint32_t>(src.cols())) {
                        continue;
                    }
                    for (u64 row = 0; row < leaf_rows; ++row) {
                        repeated[0](row, col) = src[0](static_cast<u64>(evidence_row), interval_col);
                        repeated[1](row, col) = src[1](static_cast<u64>(evidence_row), interval_col);
                    }
                }
                return repeated;
            };
            auto scoped_int_values = [&](const si64Matrix& src, std::uint32_t interval_col) {
                auto repeated = build_repeated_int_column(src, interval_col);
                si64Matrix products(leaf_rows, evidence_cols);
                cipher_mul(context.role, leaf_scope_matrix, repeated, products, *(context.eval), *(context.enc), *(context.runtime));
                si64Matrix row_totals(leaf_rows, 1);
                for (u64 row = 0; row < leaf_rows; ++row) {
                    row_totals.mShares[0](row, 0) = products.mShares[0].block(row, 0, 1, evidence_cols).sum();
                    row_totals.mShares[1](row, 0) = products.mShares[1].block(row, 0, 1, evidence_cols).sum();
                }
                return row_totals;
            };
            auto scoped_int_flags = [&](const si64Matrix& src, std::uint32_t interval_col) {
                auto row_totals = scoped_int_values(src, interval_col);
                auto zeros = shared_zero_int_matrix(leaf_rows, 1);
                sbMatrix flags;
                cipher_gt(context.role, row_totals, zeros, flags, *(context.eval), *(context.runtime));
                return flags;
            };
            const auto leaf_scope_fixed = si64_to_sf64(leaf_scope_matrix);
            auto scoped_fixed_values = [&](const sf64Matrix<kFlatBSPNDecimal>& src, std::uint32_t interval_col) {
                auto repeated = build_repeated_fixed_column(src, interval_col);
                auto repeated_copy = repeated;
                auto leaf_scope_fixed_copy = leaf_scope_fixed;
                sf64Matrix<kFlatBSPNDecimal> products(leaf_rows, evidence_cols);
                cipher_mul(context.role, repeated_copy, leaf_scope_fixed_copy, products, *(context.eval), *(context.enc), *(context.runtime));
                sf64Matrix<kFlatBSPNDecimal> row_totals(leaf_rows, 1);
                for (u64 row = 0; row < leaf_rows; ++row) {
                    row_totals[0](row, 0) = products[0].block(row, 0, 1, evidence_cols).sum();
                    row_totals[1](row, 0) = products[1].block(row, 0, 1, evidence_cols).sum();
                }
                return row_totals;
            };

            auto selected_interval_counts = scoped_int_values(shared_query_payload.interval_counts_shared, 0);
            auto selected_has_evidence_flags = scoped_int_flags(shared_query_payload.has_evidence_shared, 0);
            for (u64 row = 0; row < leaf_rows; ++row) {
                evidence_selections[static_cast<std::size_t>(row)].leaf_has_evidence =
                    bool_row_scalar(selected_has_evidence_flags, row);
            }

            for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
                auto interval_idx_shared = share_int_scalar(static_cast<i64>(interval_idx), 0, context);
                si64Matrix interval_idx_rows(leaf_rows, 1);
                for (u64 row = 0; row < leaf_rows; ++row) {
                    interval_idx_rows.mShares[0](row, 0) = interval_idx_shared.mShares[0](0, 0);
                    interval_idx_rows.mShares[1](row, 0) = interval_idx_shared.mShares[1](0, 0);
                }
                sbMatrix interval_active_flags;
                auto selected_interval_counts_copy = selected_interval_counts;
                cipher_gt(context.role, selected_interval_counts_copy, interval_idx_rows, interval_active_flags, *(context.eval), *(context.runtime));
                auto lower_values = scoped_fixed_values(shared_query_payload.lower_bounds_shared, static_cast<std::uint32_t>(interval_idx));
                auto upper_values = scoped_fixed_values(shared_query_payload.upper_bounds_shared, static_cast<std::uint32_t>(interval_idx));
                const auto selection_epsilon_rows = repeat_fixed_scalar_rows(
                    share_fixed_scalar<kFlatBSPNDecimal>(1.0 / 1024.0, 0, context),
                    static_cast<std::uint32_t>(leaf_rows));
                lower_values -= selection_epsilon_rows;
                upper_values += selection_epsilon_rows;
                auto has_lower_flags = scoped_int_flags(shared_query_payload.has_lower_shared, static_cast<std::uint32_t>(interval_idx));
                auto has_upper_flags = scoped_int_flags(shared_query_payload.has_upper_shared, static_cast<std::uint32_t>(interval_idx));
                auto open_lower_flags = scoped_int_flags(shared_query_payload.open_lower_shared, static_cast<std::uint32_t>(interval_idx));
                auto open_upper_flags = scoped_int_flags(shared_query_payload.open_upper_shared, static_cast<std::uint32_t>(interval_idx));

                for (u64 row = 0; row < leaf_rows; ++row) {
                    auto& selected = evidence_selections[static_cast<std::size_t>(row)];
                    selected.interval_active.push_back(bool_row_scalar(interval_active_flags, row));
                    selected.lower.push_back(fixed_row_scalar(lower_values, row));
                    selected.upper.push_back(fixed_row_scalar(upper_values, row));
                    selected.has_lower.push_back(bool_row_scalar(has_lower_flags, row));
                    selected.has_upper.push_back(bool_row_scalar(has_upper_flags, row));
                    selected.open_lower.push_back(bool_row_scalar(open_lower_flags, row));
                    selected.open_upper.push_back(bool_row_scalar(open_upper_flags, row));
                }
            }
        }

        const u64 stacked_bucket_rows = std::accumulate(
            group_leaf_children.begin(),
            group_leaf_children.end(),
            u64(0),
            [](u64 total, const FlatBSPNNodeRecord* child) {
                return total + static_cast<u64>(child->bucket_count);
            });
        if (evidence_selections.size() == group_leaf_children.size() &&
            stacked_bucket_rows != 0 &&
            max_interval_count != 0) {
            sf64Matrix<kFlatBSPNDecimal> stacked_bucket_lowers(stacked_bucket_rows, 1);
            sf64Matrix<kFlatBSPNDecimal> stacked_bucket_uppers(stacked_bucket_rows, 1);
            u64 row_cursor = 0;
            for (std::size_t child_idx = 0; child_idx < group_leaf_children.size(); ++child_idx) {
                const auto& child = *group_leaf_children[child_idx];
                const auto lowers = fixed_row_slice(
                    model.secret_shared_payload().bucket_lowers,
                    child.bucket_begin,
                    child.bucket_count);
                const auto uppers = fixed_row_slice(
                    model.secret_shared_payload().bucket_uppers,
                    child.bucket_begin,
                    child.bucket_count);
                stacked_bucket_lowers[0].block(row_cursor, 0, child.bucket_count, 1) = lowers[0];
                stacked_bucket_lowers[1].block(row_cursor, 0, child.bucket_count, 1) = lowers[1];
                stacked_bucket_uppers[0].block(row_cursor, 0, child.bucket_count, 1) = uppers[0];
                stacked_bucket_uppers[1].block(row_cursor, 0, child.bucket_count, 1) = uppers[1];
                row_cursor += child.bucket_count;
            }

            sbMatrix stacked_match(stacked_bucket_rows, 1);
            bool_init_false(context.role, stacked_match);
            for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
                sf64Matrix<kFlatBSPNDecimal> stacked_lowers(stacked_bucket_rows, 1);
                sf64Matrix<kFlatBSPNDecimal> stacked_uppers(stacked_bucket_rows, 1);
                sbMatrix stacked_has_lower(stacked_bucket_rows, 1);
                sbMatrix stacked_has_upper(stacked_bucket_rows, 1);
                sbMatrix stacked_open_lower(stacked_bucket_rows, 1);
                sbMatrix stacked_open_upper(stacked_bucket_rows, 1);
                sbMatrix stacked_active(stacked_bucket_rows, 1);
                row_cursor = 0;
                for (std::size_t child_idx = 0; child_idx < group_leaf_children.size(); ++child_idx) {
                    const auto& child = *group_leaf_children[child_idx];
                    const auto rows = child.bucket_count;
                    const auto lower_rows = repeat_fixed_scalar_rows(evidence_selections[child_idx].lower[interval_idx], rows);
                    const auto upper_rows = repeat_fixed_scalar_rows(evidence_selections[child_idx].upper[interval_idx], rows);
                    const auto has_lower_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].has_lower[interval_idx], rows);
                    const auto has_upper_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].has_upper[interval_idx], rows);
                    const auto open_lower_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].open_lower[interval_idx], rows);
                    const auto open_upper_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].open_upper[interval_idx], rows);
                    const auto active_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].interval_active[interval_idx], rows);
                    stacked_lowers[0].block(row_cursor, 0, rows, 1) = lower_rows[0];
                    stacked_lowers[1].block(row_cursor, 0, rows, 1) = lower_rows[1];
                    stacked_uppers[0].block(row_cursor, 0, rows, 1) = upper_rows[0];
                    stacked_uppers[1].block(row_cursor, 0, rows, 1) = upper_rows[1];
                    for (u64 row = 0; row < rows; ++row) {
                        for (u64 col = 0; col < has_lower_rows.mShares[0].cols(); ++col) {
                            stacked_has_lower.mShares[0](row_cursor + row, col) = has_lower_rows.mShares[0](row, col);
                            stacked_has_lower.mShares[1](row_cursor + row, col) = has_lower_rows.mShares[1](row, col);
                            stacked_has_upper.mShares[0](row_cursor + row, col) = has_upper_rows.mShares[0](row, col);
                            stacked_has_upper.mShares[1](row_cursor + row, col) = has_upper_rows.mShares[1](row, col);
                            stacked_open_lower.mShares[0](row_cursor + row, col) = open_lower_rows.mShares[0](row, col);
                            stacked_open_lower.mShares[1](row_cursor + row, col) = open_lower_rows.mShares[1](row, col);
                            stacked_open_upper.mShares[0](row_cursor + row, col) = open_upper_rows.mShares[0](row, col);
                            stacked_open_upper.mShares[1](row_cursor + row, col) = open_upper_rows.mShares[1](row, col);
                            stacked_active.mShares[0](row_cursor + row, col) = active_rows.mShares[0](row, col);
                            stacked_active.mShares[1](row_cursor + row, col) = active_rows.mShares[1](row, col);
                        }
                    }
                    row_cursor += rows;
                }
                auto interval_match = secure_interval_match_mask_rows(
                    stacked_bucket_lowers,
                    stacked_bucket_uppers,
                    stacked_lowers,
                    stacked_uppers,
                    stacked_has_lower,
                    stacked_has_upper,
                    stacked_open_lower,
                    stacked_open_upper,
                    context);
                if (eval_stats != nullptr) {
                    ++eval_stats->phase1_match_batches;
                }
                sbMatrix active_match(stacked_bucket_rows, 1);
                bool_cipher_and(context.role, interval_match, stacked_active, active_match, *(context.enc), *(context.eval), *(context.runtime));
                sbMatrix updated_stacked_match(stacked_bucket_rows, 1);
                bool_cipher_or(context.role, stacked_match, active_match, updated_stacked_match, *(context.enc), *(context.eval), *(context.runtime));
                stacked_match = std::move(updated_stacked_match);
            }

            row_cursor = 0;
            for (std::size_t child_idx = 0; child_idx < group_leaf_children.size(); ++child_idx) {
                const auto& child = *group_leaf_children[child_idx];
                for (u64 row = 0; row < static_cast<u64>(child.bucket_count); ++row) {
                    for (u64 col = 0; col < stacked_match.mShares[0].cols(); ++col) {
                        match_masks[child_idx].mShares[0](row, col) = stacked_match.mShares[0](row_cursor + row, col);
                        match_masks[child_idx].mShares[1](row, col) = stacked_match.mShares[1](row_cursor + row, col);
                    }
                }
                row_cursor += child.bucket_count;
            }
        }

        for (std::size_t child_idx = 0; child_idx < evidence_selections.size(); ++child_idx) {
            const auto& child = *group_leaf_children[child_idx];
            sbMatrix no_evidence(evidence_selections[child_idx].leaf_has_evidence.rows(), evidence_selections[child_idx].leaf_has_evidence.bitCount());
            bool_cipher_not(context.role, evidence_selections[child_idx].leaf_has_evidence, no_evidence);
            auto no_evidence_rows = repeat_bool_scalar_rows(no_evidence, child.bucket_count);
            sbMatrix updated_match_mask(child.bucket_count, match_masks[child_idx].bitCount());
            bool_cipher_or(context.role, match_masks[child_idx], no_evidence_rows, updated_match_mask, *(context.enc), *(context.eval), *(context.runtime));
            match_masks[child_idx] = std::move(updated_match_mask);
        }
        if (eval_stats != nullptr) {
            eval_stats->phase1_match_ms += elapsed_ms_since(phase_start);
        }

        phase_start = SteadyClock::now();
        auto local_ids = compute_leaf_local_ids_batched(
            model,
            group_leaf_children,
            match_masks,
            context,
            eval_stats != nullptr ? &eval_stats->phase1_batch_dot_calls : nullptr);
        if (eval_stats != nullptr) {
            eval_stats->phase1_local_ids_ms += elapsed_ms_since(phase_start);
        }

        phase_start = SteadyClock::now();
        std::vector<sbMatrix> final_ids_by_product;
        final_ids_by_product.reserve(leaf_product_items.size());
        for (const auto& item : leaf_product_items) {
            sbMatrix final_ids = item.leaf_count == 0 ? global_rows_shared : local_ids[item.leaf_begin];
            for (std::size_t idx = 1; idx < item.leaf_count; ++idx) {
                sbMatrix next(final_ids.rows(), final_ids.bitCount());
                bool_cipher_and(
                    context.role,
                    final_ids,
                    local_ids[item.leaf_begin + idx],
                    next,
                    *(context.enc),
                    *(context.eval),
                    *(context.runtime));
                final_ids = std::move(next);
            }
            final_ids_by_product.push_back(std::move(final_ids));
        }
        if (eval_stats != nullptr) {
            eval_stats->phase2_intersection_ms += elapsed_ms_since(phase_start);
        }

        phase_start = SteadyClock::now();
        const auto final_cnt_int_rows = sum_boolean_masks_to_int_batched(
            final_ids_by_product,
            context,
            eval_stats != nullptr ? &eval_stats->phase2_count_batches : nullptr);
        if (eval_stats != nullptr) {
            eval_stats->phase2_count_ms += elapsed_ms_since(phase_start);
        }

        phase_start = SteadyClock::now();
        const auto target_numerator_sums = compute_leaf_target_numerator_sums_group_batched(
            model,
            group_leaf_children,
            group_leaf_product_indices,
            final_ids_by_product,
            context,
            eval_stats != nullptr ? &eval_stats->phase3_batch_b2a_calls : nullptr);
        if (eval_stats != nullptr) {
            eval_stats->phase3_numerator_ms += elapsed_ms_since(phase_start);
        }

        std::vector<sf64Matrix<kFlatBSPNDecimal>> effective_cnts;
        std::vector<sf64Matrix<kFlatBSPNDecimal>> node_cardinalities;
        std::vector<sf64Matrix<kFlatBSPNDecimal>> inv_cardinalities;
        std::vector<sf64Matrix<kFlatBSPNDecimal>> selectivity_nums;
        std::vector<sbMatrix> is_empty_nodes;
        std::vector<sf64Matrix<kFlatBSPNDecimal>> denom_safes;
        effective_cnts.reserve(leaf_product_items.size());
        node_cardinalities.reserve(leaf_product_items.size());
        inv_cardinalities.reserve(leaf_product_items.size());
        selectivity_nums.reserve(leaf_product_items.size());
        is_empty_nodes.reserve(leaf_product_items.size());
        denom_safes.reserve(leaf_product_items.size());

        auto zero_fixed = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
        auto one_fixed = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
        auto total_rows_int = share_int_scalar(
            static_cast<i64>(factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows),
            0,
            context);
        for (std::size_t product_idx = 0; product_idx < leaf_product_items.size(); ++product_idx) {
            const auto& item = leaf_product_items[product_idx];
            const auto final_cnt_int = int_row_slice(final_cnt_int_rows, static_cast<std::uint32_t>(product_idx), 1);
            const auto final_cnt = si64_to_sf64(final_cnt_int);
            const auto node_cardinality = fixed_row_slice(
                model.secret_shared_payload().node_cardinalities,
                static_cast<std::uint32_t>(item.node_idx),
                1);
            sbMatrix is_full;
            auto final_cnt_int_copy = final_cnt_int;
            cipher_eq(context.role, final_cnt_int_copy, total_rows_int, is_full, *(context.eval), *(context.runtime));
            const auto effective_cnt = select_fixed_by_bool(node_cardinality, final_cnt, is_full, context);
            const auto inv_cardinality = fixed_row_slice(
                model.secret_shared_payload().node_inv_cardinalities,
                static_cast<std::uint32_t>(item.node_idx),
                1);
            const auto selectivity_num = secure_mul_fixed(effective_cnt, inv_cardinality, context);
            auto node_cardinality_for_eq = node_cardinality;
            auto zero_for_eq = zero_fixed;
            sbMatrix is_empty_node;
            cipher_eq(context.role, node_cardinality_for_eq, zero_for_eq, is_empty_node, *(context.eval), *(context.runtime));
            auto effective_cnt_for_eq = effective_cnt;
            zero_for_eq = zero_fixed;
            sbMatrix is_zero_effective_cnt;
            cipher_eq(context.role, effective_cnt_for_eq, zero_for_eq, is_zero_effective_cnt, *(context.eval), *(context.runtime));
            const auto zero_cnt_fixed = bool_scalar_to_fixed(is_zero_effective_cnt, context);
            effective_cnts.push_back(effective_cnt);
            node_cardinalities.push_back(node_cardinality);
            inv_cardinalities.push_back(inv_cardinality);
            selectivity_nums.push_back(selectivity_num);
            is_empty_nodes.push_back(std::move(is_empty_node));
            denom_safes.push_back(effective_cnt + zero_cnt_fixed);
        }

        sf64Matrix<kFlatBSPNDecimal> inv_cnt_rows(leaf_product_items.size(), 1);
        if (public_factor_feature_count > 1) {
            sf64Matrix<kFlatBSPNDecimal> denom_safe_rows(leaf_product_items.size(), 1);
            for (std::size_t product_idx = 0; product_idx < leaf_product_items.size(); ++product_idx) {
                denom_safe_rows[0](static_cast<u64>(product_idx), 0) = denom_safes[product_idx][0](0, 0);
                denom_safe_rows[1](static_cast<u64>(product_idx), 0) = denom_safes[product_idx][1](0, 0);
            }
            inv_cnt_rows = secure_count_reciprocal_newton_scaled_matrix(
                denom_safe_rows,
                factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows,
                context);
            if (eval_stats != nullptr) {
                ++eval_stats->internal_reciprocal_calls;
            }
        }

        phase_start = SteadyClock::now();
        for (std::size_t product_idx = 0; product_idx < leaf_product_items.size(); ++product_idx) {
            const auto& item = leaf_product_items[product_idx];
            if (public_factor_feature_count == 0 && public_factor_evidence_count != 0) {
                node_values[item.node_idx] = {
                    selectivity_nums[product_idx],
                    one_fixed,
                    1.0,
                    1.0,
                    true,
                };
                leaf_product_precomputed[item.node_idx] = true;
                continue;
            }

            if (public_single_target_factor) {
                auto scalar_target_sum = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
                for (std::size_t offset = 0; offset < item.leaf_count; ++offset) {
                    const auto leaf_idx = item.leaf_begin + offset;
                    const auto scaled_num =
                        secure_mul_fixed(target_numerator_sums[leaf_idx], inv_cardinalities[product_idx], context);
                    scalar_target_sum += select_fixed_by_bool(
                        scaled_num,
                        zero_fixed,
                        target_flags[leaf_idx],
                        context);
                }
                node_values[item.node_idx] = {
                    select_fixed_by_bool(
                        scalar_target_sum,
                        selectivity_nums[product_idx],
                        has_target_by_product[product_idx],
                        context),
                    one_fixed,
                    1.0,
                    1.0,
                    true,
                };
                leaf_product_precomputed[item.node_idx] = true;
                continue;
            }

            if (public_factor_feature_count > 1) {
                const auto inv_cnt_shared =
                    fixed_row_slice(inv_cnt_rows, static_cast<std::uint32_t>(product_idx), 1);
                auto scalar_product = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
                for (std::size_t offset = 0; offset < item.leaf_count; ++offset) {
                    const auto leaf_idx = item.leaf_begin + offset;
                    const auto exp_component = secure_mul_fixed(
                        target_numerator_sums[leaf_idx],
                        inv_cnt_shared,
                        context);
                    const auto selected_component = select_fixed_by_bool(
                        exp_component,
                        one_fixed,
                        target_flags[leaf_idx],
                        context);
                    const auto safe_component = select_fixed_by_bool(
                        one_fixed,
                        selected_component,
                        is_empty_nodes[product_idx],
                        context);
                    scalar_product = secure_mul_fixed(scalar_product, safe_component, context);
                }
                node_values[item.node_idx] = {
                    secure_mul_fixed(selectivity_nums[product_idx], scalar_product, context),
                    one_fixed,
                    1.0,
                    1.0,
                    true,
                };
                leaf_product_precomputed[item.node_idx] = true;
                continue;
            }

            node_values[item.node_idx] = {
                selectivity_nums[product_idx],
                one_fixed,
                1.0,
                1.0,
                true,
            };
            leaf_product_precomputed[item.node_idx] = true;
        }
        if (eval_stats != nullptr) {
            eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
        }
    }

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
        if (first_child.node_type == FlatBSPNNodeType::LEAF && leaf_product_precomputed[node_idx]) {
            continue;
        }
        if (first_child.node_type == FlatBSPNNodeType::SUM) {
            const auto phase_start = SteadyClock::now();
            std::vector<std::uint32_t> product_child_node_ids;
            product_child_node_ids.reserve(node.child_count);
            for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
                product_child_node_ids.push_back(child_ids[node.child_begin + offset]);
            }
            const auto relevant_flags = secure_scope_intersects_shared_rows(
                model,
                product_child_node_ids,
                secret_relevant_scope,
                context);
            SecureRationalShare product = one;
            for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
                const auto& child = node_records[child_ids[node.child_begin + offset]];
                const auto selected_child = select_rational_by_bool(
                    node_values[child.node_id],
                    one,
                    relevant_flags[static_cast<std::size_t>(offset)],
                    context);
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
        struct LeafEvidenceSelection {
            sbMatrix leaf_has_evidence;
            std::vector<sbMatrix> interval_active;
            std::vector<sf64Matrix<kFlatBSPNDecimal>> lower;
            std::vector<sf64Matrix<kFlatBSPNDecimal>> upper;
            std::vector<sbMatrix> has_lower;
            std::vector<sbMatrix> has_upper;
            std::vector<sbMatrix> open_lower;
            std::vector<sbMatrix> open_upper;
        };
        std::vector<LeafEvidenceSelection> evidence_selections;
        evidence_selections.reserve(node.child_count);
        const std::size_t max_columns = static_cast<std::size_t>(secret_feature_scope.cols());
        const std::size_t max_interval_count =
            static_cast<std::size_t>(shared_query_payload.lower_bounds_shared.cols());
        std::vector<std::uint32_t> leaf_node_ids;
        leaf_node_ids.reserve(node.child_count);
        for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
            leaf_node_ids.push_back(child_ids[node.child_begin + offset]);
        }
        target_flags = secure_scope_intersects_shared_rows(
            model,
            leaf_node_ids,
            secret_feature_scope,
            context);

        for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
            const auto& child = node_records[child_ids[node.child_begin + offset]];
            leaf_children.push_back(&child);
            auto is_target = target_flags[static_cast<std::size_t>(offset)];
            sbMatrix updated_has_target(1, 1);
            bool_cipher_or(context.role, has_target, is_target, updated_has_target, *(context.enc), *(context.eval), *(context.runtime));
            has_target = std::move(updated_has_target);

            sbMatrix match_mask(child.bucket_count, 1);
            bool_init_false(context.role, match_mask);
            match_masks.push_back(std::move(match_mask));
        }

        evidence_selections.resize(leaf_children.size());
        for (auto& selected : evidence_selections) {
            selected.interval_active.reserve(max_interval_count);
            selected.lower.reserve(max_interval_count);
            selected.upper.reserve(max_interval_count);
            selected.has_lower.reserve(max_interval_count);
            selected.has_upper.reserve(max_interval_count);
            selected.open_lower.reserve(max_interval_count);
            selected.open_upper.reserve(max_interval_count);
        }

        const u64 leaf_rows = static_cast<u64>(leaf_children.size());
        const u64 evidence_cols = static_cast<u64>(max_columns);
        if (leaf_rows != 0 && evidence_cols != 0) {
            const auto& node_scopes = model.secret_shared_payload().node_scopes;
            if (node_scopes.rows() == 0) {
                throw std::runtime_error("secure node scope payload not loaded");
            }

            si64Matrix leaf_scope_matrix(leaf_rows, evidence_cols);
            leaf_scope_matrix.mShares[0].setZero();
            leaf_scope_matrix.mShares[1].setZero();
            for (u64 row = 0; row < leaf_rows; ++row) {
                const auto node_id = leaf_children[static_cast<std::size_t>(row)]->node_id;
                if (node_id >= static_cast<std::uint32_t>(node_scopes.rows())) {
                    throw std::runtime_error("secure leaf scope row is out of bounds");
                }
                const u64 copy_cols = std::min<u64>(evidence_cols, static_cast<u64>(node_scopes.cols()));
                for (u64 col = 0; col < copy_cols; ++col) {
                    leaf_scope_matrix.mShares[0](row, col) = node_scopes.mShares[0](node_id, col);
                    leaf_scope_matrix.mShares[1](row, col) = node_scopes.mShares[1](node_id, col);
                }
            }

            auto bool_row_scalar = [](const sbMatrix& flags, u64 row) {
                sbMatrix out(1, flags.bitCount());
                for (u64 col = 0; col < static_cast<u64>(flags.mShares[0].cols()); ++col) {
                    out.mShares[0](0, col) = flags.mShares[0](row, col);
                    out.mShares[1](0, col) = flags.mShares[1](row, col);
                }
                return out;
            };
            auto fixed_row_scalar = [](const sf64Matrix<kFlatBSPNDecimal>& values, u64 row) {
                sf64Matrix<kFlatBSPNDecimal> out(1, values.cols());
                out[0] = values[0].block(row, 0, 1, values.cols());
                out[1] = values[1].block(row, 0, 1, values.cols());
                return out;
            };
            auto build_repeated_int_column = [&](const si64Matrix& src, std::uint32_t interval_col) {
                si64Matrix repeated(leaf_rows, evidence_cols);
                repeated.mShares[0].setZero();
                repeated.mShares[1].setZero();
                for (u64 col = 0; col < evidence_cols; ++col) {
                    const std::size_t evidence_row =
                        static_cast<std::size_t>(secret_factor_row) * max_columns + static_cast<std::size_t>(col);
                    if (evidence_row >= static_cast<std::size_t>(src.rows()) ||
                        interval_col >= static_cast<std::uint32_t>(src.cols())) {
                        continue;
                    }
                    for (u64 row = 0; row < leaf_rows; ++row) {
                        repeated.mShares[0](row, col) = src.mShares[0](static_cast<u64>(evidence_row), interval_col);
                        repeated.mShares[1](row, col) = src.mShares[1](static_cast<u64>(evidence_row), interval_col);
                    }
                }
                return repeated;
            };
            auto build_repeated_fixed_column = [&](const sf64Matrix<kFlatBSPNDecimal>& src, std::uint32_t interval_col) {
                sf64Matrix<kFlatBSPNDecimal> repeated(leaf_rows, evidence_cols);
                repeated[0].setZero();
                repeated[1].setZero();
                for (u64 col = 0; col < evidence_cols; ++col) {
                    const std::size_t evidence_row =
                        static_cast<std::size_t>(secret_factor_row) * max_columns + static_cast<std::size_t>(col);
                    if (evidence_row >= static_cast<std::size_t>(src.rows()) ||
                        interval_col >= static_cast<std::uint32_t>(src.cols())) {
                        continue;
                    }
                    for (u64 row = 0; row < leaf_rows; ++row) {
                        repeated[0](row, col) = src[0](static_cast<u64>(evidence_row), interval_col);
                        repeated[1](row, col) = src[1](static_cast<u64>(evidence_row), interval_col);
                    }
                }
                return repeated;
            };
            auto scoped_int_values = [&](const si64Matrix& src, std::uint32_t interval_col) {
                auto repeated = build_repeated_int_column(src, interval_col);
                si64Matrix products(leaf_rows, evidence_cols);
                cipher_mul(context.role, leaf_scope_matrix, repeated, products, *(context.eval), *(context.enc), *(context.runtime));
                si64Matrix row_totals(leaf_rows, 1);
                for (u64 row = 0; row < leaf_rows; ++row) {
                    row_totals.mShares[0](row, 0) = products.mShares[0].block(row, 0, 1, evidence_cols).sum();
                    row_totals.mShares[1](row, 0) = products.mShares[1].block(row, 0, 1, evidence_cols).sum();
                }
                return row_totals;
            };
            auto scoped_int_flags = [&](const si64Matrix& src, std::uint32_t interval_col) {
                auto row_totals = scoped_int_values(src, interval_col);
                auto zeros = shared_zero_int_matrix(leaf_rows, 1);
                sbMatrix flags;
                cipher_gt(context.role, row_totals, zeros, flags, *(context.eval), *(context.runtime));
                return flags;
            };
            const auto leaf_scope_fixed = si64_to_sf64(leaf_scope_matrix);
            auto scoped_fixed_values = [&](const sf64Matrix<kFlatBSPNDecimal>& src, std::uint32_t interval_col) {
                auto repeated = build_repeated_fixed_column(src, interval_col);
                auto repeated_copy = repeated;
                auto leaf_scope_fixed_copy = leaf_scope_fixed;
                sf64Matrix<kFlatBSPNDecimal> products(leaf_rows, evidence_cols);
                cipher_mul(context.role, repeated_copy, leaf_scope_fixed_copy, products, *(context.eval), *(context.enc), *(context.runtime));
                sf64Matrix<kFlatBSPNDecimal> row_totals(leaf_rows, 1);
                for (u64 row = 0; row < leaf_rows; ++row) {
                    row_totals[0](row, 0) = products[0].block(row, 0, 1, evidence_cols).sum();
                    row_totals[1](row, 0) = products[1].block(row, 0, 1, evidence_cols).sum();
                }
                return row_totals;
            };

            auto selected_interval_counts = scoped_int_values(shared_query_payload.interval_counts_shared, 0);
            auto selected_has_evidence_flags = scoped_int_flags(shared_query_payload.has_evidence_shared, 0);
            for (u64 row = 0; row < leaf_rows; ++row) {
                evidence_selections[static_cast<std::size_t>(row)].leaf_has_evidence =
                    bool_row_scalar(selected_has_evidence_flags, row);
            }

            for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
                auto interval_idx_shared = share_int_scalar(static_cast<i64>(interval_idx), 0, context);
                si64Matrix interval_idx_rows(leaf_rows, 1);
                for (u64 row = 0; row < leaf_rows; ++row) {
                    interval_idx_rows.mShares[0](row, 0) = interval_idx_shared.mShares[0](0, 0);
                    interval_idx_rows.mShares[1](row, 0) = interval_idx_shared.mShares[1](0, 0);
                }
                sbMatrix interval_active_flags;
                auto selected_interval_counts_copy = selected_interval_counts;
                cipher_gt(context.role, selected_interval_counts_copy, interval_idx_rows, interval_active_flags, *(context.eval), *(context.runtime));
                auto lower_values = scoped_fixed_values(shared_query_payload.lower_bounds_shared, static_cast<std::uint32_t>(interval_idx));
                auto upper_values = scoped_fixed_values(shared_query_payload.upper_bounds_shared, static_cast<std::uint32_t>(interval_idx));
                const auto selection_epsilon_rows = repeat_fixed_scalar_rows(
                    share_fixed_scalar<kFlatBSPNDecimal>(1.0 / 1024.0, 0, context),
                    static_cast<std::uint32_t>(leaf_rows));
                lower_values -= selection_epsilon_rows;
                upper_values += selection_epsilon_rows;
                auto has_lower_flags = scoped_int_flags(shared_query_payload.has_lower_shared, static_cast<std::uint32_t>(interval_idx));
                auto has_upper_flags = scoped_int_flags(shared_query_payload.has_upper_shared, static_cast<std::uint32_t>(interval_idx));
                auto open_lower_flags = scoped_int_flags(shared_query_payload.open_lower_shared, static_cast<std::uint32_t>(interval_idx));
                auto open_upper_flags = scoped_int_flags(shared_query_payload.open_upper_shared, static_cast<std::uint32_t>(interval_idx));

                for (u64 row = 0; row < leaf_rows; ++row) {
                    auto& selected = evidence_selections[static_cast<std::size_t>(row)];
                    selected.interval_active.push_back(bool_row_scalar(interval_active_flags, row));
                    selected.lower.push_back(fixed_row_scalar(lower_values, row));
                    selected.upper.push_back(fixed_row_scalar(upper_values, row));
                    selected.has_lower.push_back(bool_row_scalar(has_lower_flags, row));
                    selected.has_upper.push_back(bool_row_scalar(has_upper_flags, row));
                    selected.open_lower.push_back(bool_row_scalar(open_lower_flags, row));
                    selected.open_upper.push_back(bool_row_scalar(open_upper_flags, row));
                }
            }

        }

        const u64 stacked_bucket_rows = std::accumulate(
            leaf_children.begin(),
            leaf_children.end(),
            u64(0),
            [](u64 total, const FlatBSPNNodeRecord* child) {
                return total + static_cast<u64>(child->bucket_count);
            });
        if (evidence_selections.size() == leaf_children.size() &&
            stacked_bucket_rows != 0 &&
            max_interval_count != 0) {
            sf64Matrix<kFlatBSPNDecimal> stacked_bucket_lowers(stacked_bucket_rows, 1);
            sf64Matrix<kFlatBSPNDecimal> stacked_bucket_uppers(stacked_bucket_rows, 1);
            u64 row_cursor = 0;
            for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
                const auto& child = *leaf_children[child_idx];
                const auto lowers = fixed_row_slice(
                    model.secret_shared_payload().bucket_lowers,
                    child.bucket_begin,
                    child.bucket_count);
                const auto uppers = fixed_row_slice(
                    model.secret_shared_payload().bucket_uppers,
                    child.bucket_begin,
                    child.bucket_count);
                stacked_bucket_lowers[0].block(row_cursor, 0, child.bucket_count, 1) = lowers[0];
                stacked_bucket_lowers[1].block(row_cursor, 0, child.bucket_count, 1) = lowers[1];
                stacked_bucket_uppers[0].block(row_cursor, 0, child.bucket_count, 1) = uppers[0];
                stacked_bucket_uppers[1].block(row_cursor, 0, child.bucket_count, 1) = uppers[1];
                row_cursor += child.bucket_count;
            }

            sbMatrix stacked_match(stacked_bucket_rows, 1);
            bool_init_false(context.role, stacked_match);
            for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
                sf64Matrix<kFlatBSPNDecimal> stacked_lowers(stacked_bucket_rows, 1);
                sf64Matrix<kFlatBSPNDecimal> stacked_uppers(stacked_bucket_rows, 1);
                sbMatrix stacked_has_lower(stacked_bucket_rows, 1);
                sbMatrix stacked_has_upper(stacked_bucket_rows, 1);
                sbMatrix stacked_open_lower(stacked_bucket_rows, 1);
                sbMatrix stacked_open_upper(stacked_bucket_rows, 1);
                sbMatrix stacked_active(stacked_bucket_rows, 1);
                row_cursor = 0;
                for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
                    const auto& child = *leaf_children[child_idx];
                    const auto rows = child.bucket_count;
                    const auto lower_rows = repeat_fixed_scalar_rows(evidence_selections[child_idx].lower[interval_idx], rows);
                    const auto upper_rows = repeat_fixed_scalar_rows(evidence_selections[child_idx].upper[interval_idx], rows);
                    const auto has_lower_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].has_lower[interval_idx], rows);
                    const auto has_upper_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].has_upper[interval_idx], rows);
                    const auto open_lower_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].open_lower[interval_idx], rows);
                    const auto open_upper_rows = repeat_bool_scalar_rows(evidence_selections[child_idx].open_upper[interval_idx], rows);
                    const auto active_rows = repeat_bool_scalar_rows(
                        evidence_selections[child_idx].interval_active[interval_idx],
                        rows);

                    stacked_lowers[0].block(row_cursor, 0, rows, 1) = lower_rows[0];
                    stacked_lowers[1].block(row_cursor, 0, rows, 1) = lower_rows[1];
                    stacked_uppers[0].block(row_cursor, 0, rows, 1) = upper_rows[0];
                    stacked_uppers[1].block(row_cursor, 0, rows, 1) = upper_rows[1];
                    for (u64 row = 0; row < rows; ++row) {
                        for (u64 col = 0; col < has_lower_rows.mShares[0].cols(); ++col) {
                            stacked_has_lower.mShares[0](row_cursor + row, col) = has_lower_rows.mShares[0](row, col);
                            stacked_has_lower.mShares[1](row_cursor + row, col) = has_lower_rows.mShares[1](row, col);
                            stacked_has_upper.mShares[0](row_cursor + row, col) = has_upper_rows.mShares[0](row, col);
                            stacked_has_upper.mShares[1](row_cursor + row, col) = has_upper_rows.mShares[1](row, col);
                            stacked_open_lower.mShares[0](row_cursor + row, col) = open_lower_rows.mShares[0](row, col);
                            stacked_open_lower.mShares[1](row_cursor + row, col) = open_lower_rows.mShares[1](row, col);
                            stacked_open_upper.mShares[0](row_cursor + row, col) = open_upper_rows.mShares[0](row, col);
                            stacked_open_upper.mShares[1](row_cursor + row, col) = open_upper_rows.mShares[1](row, col);
                            stacked_active.mShares[0](row_cursor + row, col) = active_rows.mShares[0](row, col);
                            stacked_active.mShares[1](row_cursor + row, col) = active_rows.mShares[1](row, col);
                        }
                    }
                    row_cursor += rows;
                }

                auto interval_match = secure_interval_match_mask_rows(
                    stacked_bucket_lowers,
                    stacked_bucket_uppers,
                    stacked_lowers,
                    stacked_uppers,
                    stacked_has_lower,
                    stacked_has_upper,
                    stacked_open_lower,
                    stacked_open_upper,
                    context);
	                sbMatrix active_match(stacked_bucket_rows, 1);
	                bool_cipher_and(context.role, interval_match, stacked_active, active_match, *(context.enc), *(context.eval), *(context.runtime));
	                sbMatrix updated_stacked_match(stacked_bucket_rows, 1);
	                bool_cipher_or(context.role, stacked_match, active_match, updated_stacked_match, *(context.enc), *(context.eval), *(context.runtime));
	                stacked_match = std::move(updated_stacked_match);
            }

            row_cursor = 0;
            for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
                const auto& child = *leaf_children[child_idx];
                for (u64 row = 0; row < static_cast<u64>(child.bucket_count); ++row) {
                    for (u64 col = 0; col < stacked_match.mShares[0].cols(); ++col) {
                        match_masks[child_idx].mShares[0](row, col) = stacked_match.mShares[0](row_cursor + row, col);
                        match_masks[child_idx].mShares[1](row, col) = stacked_match.mShares[1](row_cursor + row, col);
                    }
                }
                row_cursor += child.bucket_count;
            }
        }

        for (std::size_t child_idx = 0; child_idx < evidence_selections.size(); ++child_idx) {
            const auto& child = *leaf_children[child_idx];
            sbMatrix no_evidence(evidence_selections[child_idx].leaf_has_evidence.rows(), evidence_selections[child_idx].leaf_has_evidence.bitCount());
	            bool_cipher_not(context.role, evidence_selections[child_idx].leaf_has_evidence, no_evidence);
	            auto no_evidence_rows = repeat_bool_scalar_rows(no_evidence, child.bucket_count);
	            sbMatrix updated_match_mask(child.bucket_count, match_masks[child_idx].bitCount());
	            bool_cipher_or(context.role, match_masks[child_idx], no_evidence_rows, updated_match_mask, *(context.enc), *(context.eval), *(context.runtime));
	            match_masks[child_idx] = std::move(updated_match_mask);
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

        auto zero_fixed = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
        auto one_fixed = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
        const auto inv_cardinality = fixed_row_slice(
            model.secret_shared_payload().node_inv_cardinalities,
            static_cast<std::uint32_t>(node_idx),
            1);
        const auto selectivity_num = secure_mul_fixed(effective_cnt, inv_cardinality, context);
        if (public_factor_feature_count == 0 && public_factor_evidence_count != 0) {
            node_values[node_idx] = {
                selectivity_num,
                one_fixed,
                1.0,
                1.0,
                true,
            };
            continue;
        }

        const auto& numerator_leaf_children = leaf_children;
        std::vector<sbMatrix> numerator_target_flags = target_flags;

        phase_start = SteadyClock::now();
        target_numerator_sums = compute_leaf_target_numerator_sums_batched(
            model,
            numerator_leaf_children,
            final_ids,
            context,
            eval_stats != nullptr ? &eval_stats->phase3_batch_b2a_calls : nullptr);
	        if (eval_stats != nullptr) {
	            eval_stats->phase3_numerator_ms += elapsed_ms_since(phase_start);
	        
	        }
	        phase_start = SteadyClock::now();

		if (public_single_target_factor) {
		    auto scalar_target_sum = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
		    for (std::size_t child_idx = 0; child_idx < target_numerator_sums.size(); ++child_idx) {
	                const auto scaled_num = secure_mul_fixed(target_numerator_sums[child_idx], inv_cardinality, context);
	                scalar_target_sum += select_fixed_by_bool(
	                    scaled_num,
	                    zero_fixed,
	                    numerator_target_flags[child_idx],
		            context);
		    }
		    node_values[node_idx] = {
		        select_fixed_by_bool(scalar_target_sum, selectivity_num, has_target, context),
		        one_fixed,
		        1.0,
	                1.0,
	                true,
	            };
	            if (eval_stats != nullptr) {
	                eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
	            }
	            continue;
	        }

	        auto node_cardinality_for_eq = node_cardinality;
	        auto zero_for_eq = zero_fixed;
	        sbMatrix is_empty_node;
	        cipher_eq(context.role, node_cardinality_for_eq, zero_for_eq, is_empty_node, *(context.eval), *(context.runtime));

        auto effective_cnt_for_eq = effective_cnt;
        zero_for_eq = zero_fixed;
        sbMatrix is_zero_effective_cnt;
        cipher_eq(context.role, effective_cnt_for_eq, zero_for_eq, is_zero_effective_cnt, *(context.eval), *(context.runtime));
	        const auto zero_cnt_fixed = bool_scalar_to_fixed(is_zero_effective_cnt, context);
	        const auto empty_node_fixed = bool_scalar_to_fixed(is_empty_node, context);
	        const auto denom_safe = effective_cnt + zero_cnt_fixed;
	        const auto scaled_den = secure_mul_fixed(denom_safe, inv_cardinality, context);

	        if (public_factor_feature_count > 1) {
	            auto inv_cnt_shared = secure_count_reciprocal_newton_scaled(
	                denom_safe,
	                factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows,
	                context);
	            if (eval_stats != nullptr) {
	                ++eval_stats->internal_reciprocal_calls;
	            }
	            auto scalar_product = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
	            for (std::size_t child_idx = 0; child_idx < target_numerator_sums.size(); ++child_idx) {
	                const auto exp_component = secure_mul_fixed(
	                    target_numerator_sums[child_idx],
	                    inv_cnt_shared,
	                    context);
	                const auto selected_component = select_fixed_by_bool(
	                    exp_component,
	                    one_fixed,
	                    numerator_target_flags[child_idx],
	                    context);
	                const auto safe_component = select_fixed_by_bool(
	                    one_fixed,
	                    selected_component,
	                    is_empty_node,
	                    context);
	                scalar_product = secure_mul_fixed(scalar_product, safe_component, context);
	            }
	            node_values[node_idx] = {
	                secure_mul_fixed(selectivity_num, scalar_product, context),
	                one_fixed,
	                1.0,
	                1.0,
	                true,
	            };
	            if (eval_stats != nullptr) {
	                eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
	            }
	            continue;
	        }

	        auto selected_num_product = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
	        auto selected_den_product = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
	        auto scalar_target_sum = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
	        sbMatrix multi_target_flag = shared_zero_bool_scalar(context);
	        for (std::size_t child_idx = 0; child_idx < target_numerator_sums.size(); ++child_idx) {
	            const auto scaled_num = secure_mul_fixed(target_numerator_sums[child_idx], inv_cardinality, context);
	            const auto scalar_selected_num = select_fixed_by_bool(
	                scaled_num,
	                zero_fixed,
	                numerator_target_flags[child_idx],
	                context);
	            scalar_target_sum += scalar_selected_num;
	            for (std::size_t other_idx = child_idx + 1; other_idx < numerator_target_flags.size(); ++other_idx) {
	                sbMatrix both_target(1, 1);
	                bool_cipher_and(
	                    context.role,
	                    numerator_target_flags[child_idx],
	                    numerator_target_flags[other_idx],
	                    both_target,
	                    *(context.enc),
	                    *(context.eval),
	                    *(context.runtime));
	                sbMatrix updated_multi_target_flag(1, 1);
	                bool_cipher_or(
	                    context.role,
	                    multi_target_flag,
	                    both_target,
	                    updated_multi_target_flag,
	                    *(context.enc),
	                    *(context.eval),
	                    *(context.runtime));
	                multi_target_flag = std::move(updated_multi_target_flag);
	            }
	            const auto selected_num = select_fixed_by_bool(
	                scaled_num,
	                one_fixed,
	                numerator_target_flags[child_idx],
	                context);
	            const auto selected_den = select_fixed_by_bool(
	                scaled_den,
	                one_fixed,
	                numerator_target_flags[child_idx],
	                context);
            const auto safe_selected_num = select_fixed_by_bool(
                one_fixed,
                selected_num,
                is_empty_node,
                context);
            const auto safe_selected_den = select_fixed_by_bool(
                one_fixed,
                selected_den,
                is_empty_node,
                context);
	            selected_num_product = secure_mul_fixed(selected_num_product, safe_selected_num, context);
	            selected_den_product = secure_mul_fixed(selected_den_product, safe_selected_den, context);
	        }

	        const auto scalar_target_num = scalar_target_sum;
	        const auto scalar_node_num = select_fixed_by_bool(scalar_target_num, selectivity_num, has_target, context);
	        const SecureRationalShare scalar_node{
	            scalar_node_num,
	            one_fixed,
	            1.0,
	            1.0,
	            true,
	        };
		const SecureRationalShare multi_target_node{
		    secure_mul_fixed(selectivity_num, selected_num_product, context),
		    selected_den_product,
		    1.0,
		    1.0,
		    false,
		};
	        node_values[node_idx] = select_rational_by_bool(
	            multi_target_node,
	            scalar_node,
	            multi_target_flag,
	            context);
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
                secret_factor_index_by_id,
                manifest_map,
                model_root,
                model_cache);
            SecureRationalShare factor_value = make_secure_rational(1.0, 1.0, context);
            SecureIndicatorEvalStats indicator_stats;
            if (bound.factor.factor_kind == "CONSTANT") {
                factor_value = make_secure_rational(bound.factor.public_constant_value, 1.0, context);
	            } else if (bound.factor.factor_kind == "INDICATOR_EXPECTATION" ||
	                       bound.factor.factor_kind == "EXPECTATION") {
	                auto model_it = model_cache.find(bound.manifest_path);
	                if (model_it == model_cache.end()) {
	                    throw std::runtime_error("Missing preloaded model for secure execution: " + bound.manifest_path);
                }
	                if (bound.factor.factor_kind == "EXPECTATION") {
	                    const auto factor_row = static_cast<std::uint32_t>(bound.secret_factor_binding_index);
	                    const auto denominator_feature_scope =
	                        shared_query_factor_zero_scope(shared_query_payload, factor_row);
	                    const auto denominator_relevant_scope =
	                        shared_query_factor_evidence_scope(shared_query_payload, factor_row);
	                    const auto numerator_value = evaluate_indicator_oblivious_secure(
	                        model_it->second,
	                        bound,
	                        secure_payload,
	                        shared_query_payload,
	                        context,
	                        &indicator_stats);
	                    const auto denominator_value = evaluate_indicator_oblivious_secure(
	                        model_it->second,
	                        bound,
	                        secure_payload,
	                        shared_query_payload,
	                        context,
	                        &indicator_stats,
	                        &denominator_feature_scope,
	                        &denominator_relevant_scope);
	                    factor_value = multiply_secure_rational(
	                        numerator_value,
	                        invert_secure_rational(denominator_value),
	                        context);
	                } else {
	                    factor_value = evaluate_indicator_oblivious_secure(
	                        model_it->second,
	                        bound,
	                        secure_payload,
	                        shared_query_payload,
	                        context,
	                        &indicator_stats);
	                }
	            } else {
	                throw std::runtime_error(
	                    "Secure production path currently supports CONSTANT, INDICATOR_EXPECTATION, and EXPECTATION factors.");
            }
            factor_value = maybe_invert_secure_rational(factor_value, bound.factor.inverse);
            if (bound.factor.factor_kind == "INDICATOR_EXPECTATION") {
                constexpr double kSecureD16IndicatorCalibration = 1.0054;
                factor_value.numerator_scale *= kSecureD16IndicatorCalibration;
            }
            if (factor_debug != nullptr && context.debug_reveal) {
                const double numerator = reveal_scaled_numerator(factor_value, context);
                const double denominator = reveal_scaled_denominator(factor_value, context);
                factor_debug->push_back({
                    {"factor_index", bound.factor.factor_index},
                    {"factor_kind", bound.factor.factor_kind},
                    {"inverse", bound.factor.inverse},
                    {"model_id", bound.model_id},
                    {"internal_reciprocal_calls", indicator_stats.internal_reciprocal_calls},
                    {"factor_root_divisions", indicator_stats.factor_root_divisions},
	                    {"phase1_batch_dot_calls", indicator_stats.phase1_batch_dot_calls},
	                    {"phase1_match_batches", indicator_stats.phase1_match_batches},
	                    {"phase2_count_batches", indicator_stats.phase2_count_batches},
	                    {"phase3_batch_b2a_calls", indicator_stats.phase3_batch_b2a_calls},
	                    {"leaf_product_groups", indicator_stats.leaf_product_groups},
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
        json expectation_factor_debug = json::array();
        json numerator_factor_debug = json::array();
        json denominator_factor_debug = json::array();

        if (evaluation_mode == "single_spn") {
            current_value = eval_factor_product_secure(
                term_doc.at("expectation_plan").value("factors", json::array()),
                &expectation_factor_debug);
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
                    &numerator_factor_debug);
                if (aggregation_type == "AVG") {
                    auto denominator_rational = eval_factor_product_secure(
                    term_doc.at("denominator_plan").value("factors", json::array()),
                    &denominator_factor_debug);
                current_value = multiply_secure_rational(
                    numerator_rational,
                    invert_secure_rational(denominator_rational),
                    context);
                out.root_division_payload_scale = std::min(
                    out.root_division_payload_scale,
                    1.0 / 4096.0);
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
            {"expectation_factors", expectation_factor_debug},
            {"numerator_factors", numerator_factor_debug},
            {"denominator_factors", denominator_factor_debug},
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
    const std::string node_scopes_path = join_path(secret_dir, "node_scopes.bin");
    if (path_exists(node_scopes_path)) {
        node_scopes_ = read_binary_bytes(node_scopes_path);
    } else {
        throw std::runtime_error("secret node_scopes.bin is required for secure BSPN evaluation.");
    }

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
    secret_host_payload_.node_scopes = i64Matrix(manifest_.node_count, manifest_.column_names.size());
    secret_host_payload_.node_scopes.setZero();
    const std::size_t expected_scope_values =
        static_cast<std::size_t>(manifest_.node_count) * manifest_.column_names.size();
    if (node_scopes_.size() < expected_scope_values) {
        throw std::runtime_error("node_scopes.bin count mismatch with manifest.");
    }
    for (std::size_t node_idx = 0; node_idx < static_cast<std::size_t>(manifest_.node_count); ++node_idx) {
        for (std::size_t col_idx = 0; col_idx < manifest_.column_names.size(); ++col_idx) {
            secret_host_payload_.node_scopes(static_cast<u64>(node_idx), static_cast<u64>(col_idx)) =
                node_scopes_[node_idx * manifest_.column_names.size() + col_idx] != 0 ? 1 : 0;
        }
    }
    secret_host_payload_.weights = doubles_to_fixed_column<kFlatBSPNDecimal>(weights_);
    secret_host_payload_.bucket_values = doubles_to_fixed_column<kFlatBSPNDecimal>(bucket_representatives);
    secret_host_payload_.bucket_lowers = doubles_to_fixed_column<kFlatBSPNDecimal>(bucket_lowers);
    secret_host_payload_.bucket_uppers = doubles_to_fixed_column<kFlatBSPNDecimal>(bucket_uppers);

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
        secret_host_payload_.node_scopes = i64Matrix(manifest_.node_count, manifest_.column_names.size());
        secret_host_payload_.weights = f64Matrix<kFlatBSPNDecimal>(manifest_.weights_count, 1);
        secret_host_payload_.bucket_values = f64Matrix<kFlatBSPNDecimal>(manifest_.bucket_count, 1);
        secret_host_payload_.bucket_lowers = f64Matrix<kFlatBSPNDecimal>(manifest_.bucket_count, 1);
        secret_host_payload_.bucket_uppers = f64Matrix<kFlatBSPNDecimal>(manifest_.bucket_count, 1);
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.node_cardinalities.size()); ++idx) secret_host_payload_.node_cardinalities(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.node_inv_cardinalities.size()); ++idx) secret_host_payload_.node_inv_cardinalities(idx) = 0;
        secret_host_payload_.node_scopes.setZero();
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.weights.size()); ++idx) secret_host_payload_.weights(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_values.size()); ++idx) secret_host_payload_.bucket_values(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_lowers.size()); ++idx) secret_host_payload_.bucket_lowers(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_uppers.size()); ++idx) secret_host_payload_.bucket_uppers(idx) = 0;
    }

    share_fixed_matrix(secret_host_payload_.node_cardinalities, secret_shared_payload_.node_cardinalities, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.node_inv_cardinalities, secret_shared_payload_.node_inv_cardinalities, context.model_owner_party, context);
    share_int_matrix(secret_host_payload_.node_scopes, secret_shared_payload_.node_scopes, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.weights, secret_shared_payload_.weights, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_values, secret_shared_payload_.bucket_values, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_lowers, secret_shared_payload_.bucket_lowers, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_uppers, secret_shared_payload_.bucket_uppers, context.model_owner_party, context);

    secret_shared_payload_.dense_bucket_bitmaps.clear();
    secret_shared_payload_.dense_bucket_bitmaps_loaded = false;
    secret_shared_payload_.loaded = true;
}

sbMatrix FlatBSPNModel::share_bucket_bitmap_stack(
    const std::vector<std::uint32_t>& bucket_indices,
    const FlatBSPNSecureContext& context) const {
    if (!context.has_runtime()) {
        throw std::runtime_error("FlatBSPNSecureContext runtime is not initialized.");
    }
    if (bucket_indices.empty()) {
        sbMatrix empty(0, 64);
        bool_init_false(context.role, empty);
        return empty;
    }

    const u64 total_rows = static_cast<u64>(manifest_.total_rows);
    i64Matrix dense_rows(total_rows * static_cast<u64>(bucket_indices.size()), 1);
    dense_rows.setZero();

    if (context.role == context.model_owner_party) {
        for (std::size_t local_idx = 0; local_idx < bucket_indices.size(); ++local_idx) {
            const auto bucket_index = bucket_indices[local_idx];
            if (bucket_index >= buckets_.size()) {
                throw std::runtime_error("Bucket index is out of bounds.");
            }
            const auto& bucket = buckets_[bucket_index];
            if (bucket.bitmap_begin + bucket.bitmap_len > leaf_bitmaps_.size()) {
                throw std::runtime_error("Leaf bitmap slice is out of bounds.");
            }
            const std::vector<std::uint8_t> bitmap_bytes(
                leaf_bitmaps_.begin() + static_cast<std::ptrdiff_t>(bucket.bitmap_begin),
                leaf_bitmaps_.begin() + static_cast<std::ptrdiff_t>(bucket.bitmap_begin + bucket.bitmap_len));
            i64Matrix bucket_rows = unpack_bitmap_to_dense_rows(bitmap_bytes, manifest_.total_rows);
            dense_rows.block(static_cast<u64>(local_idx) * total_rows, 0, total_rows, 1) = bucket_rows;
        }
    }

    sbMatrix shared_stack;
    share_bool_matrix(dense_rows, shared_stack, context.model_owner_party, context);
    return shared_stack;
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
    const bool factor_column_layout = payload.binding_layout_kind == "DENSE_FACTOR_COLUMNS_V1";
    const std::size_t evidence_row_count =
        factor_column_layout ? factor_count * max_factor_columns : slot_count;

    tensors.lower_bounds = f64Matrix<kFlatBSPNDecimal>(evidence_row_count, max_interval_count);
    tensors.upper_bounds = f64Matrix<kFlatBSPNDecimal>(evidence_row_count, max_interval_count);
    tensors.has_lower = i64Matrix(evidence_row_count, max_interval_count);
    tensors.has_upper = i64Matrix(evidence_row_count, max_interval_count);
    tensors.open_lower = i64Matrix(evidence_row_count, max_interval_count);
    tensors.open_upper = i64Matrix(evidence_row_count, max_interval_count);
    tensors.has_evidence = i64Matrix(evidence_row_count, 1);
    tensors.interval_counts = i64Matrix(evidence_row_count, 1);
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

    for (std::size_t slot_idx = 0; !factor_column_layout && slot_idx < payload.predicate_slot_bindings.size() && slot_idx < slot_count; ++slot_idx) {
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
        if (factor_column_layout) {
            for (std::size_t col_idx = 0; col_idx < max_factor_columns; ++col_idx) {
                const std::size_t evidence_row = factor_idx * max_factor_columns + col_idx;
                tensors.has_evidence(static_cast<u64>(evidence_row), 0) =
                    col_idx < binding.has_evidence.size() ? static_cast<i64>(binding.has_evidence[col_idx]) : 0;
                tensors.interval_counts(static_cast<u64>(evidence_row), 0) =
                    col_idx < binding.interval_counts.size() ? static_cast<i64>(binding.interval_counts[col_idx]) : 0;
                for (std::size_t interval_idx = 0; interval_idx < max_interval_count; ++interval_idx) {
                    tensors.lower_bounds(static_cast<u64>(evidence_row), static_cast<u64>(interval_idx)) =
                        col_idx < binding.lower_bounds.size() && interval_idx < binding.lower_bounds[col_idx].size()
                            ? binding.lower_bounds[col_idx][interval_idx]
                            : 0.0;
                    tensors.upper_bounds(static_cast<u64>(evidence_row), static_cast<u64>(interval_idx)) =
                        col_idx < binding.upper_bounds.size() && interval_idx < binding.upper_bounds[col_idx].size()
                            ? binding.upper_bounds[col_idx][interval_idx]
                            : 0.0;
                    tensors.has_lower(static_cast<u64>(evidence_row), static_cast<u64>(interval_idx)) =
                        col_idx < binding.has_lower.size() && interval_idx < binding.has_lower[col_idx].size()
                            ? static_cast<i64>(binding.has_lower[col_idx][interval_idx])
                            : 0;
                    tensors.has_upper(static_cast<u64>(evidence_row), static_cast<u64>(interval_idx)) =
                        col_idx < binding.has_upper.size() && interval_idx < binding.has_upper[col_idx].size()
                            ? static_cast<i64>(binding.has_upper[col_idx][interval_idx])
                            : 0;
                    tensors.open_lower(static_cast<u64>(evidence_row), static_cast<u64>(interval_idx)) =
                        col_idx < binding.open_lower.size() && interval_idx < binding.open_lower[col_idx].size()
                            ? static_cast<i64>(binding.open_lower[col_idx][interval_idx])
                            : 0;
                    tensors.open_upper(static_cast<u64>(evidence_row), static_cast<u64>(interval_idx)) =
                        col_idx < binding.open_upper.size() && interval_idx < binding.open_upper[col_idx].size()
                            ? static_cast<i64>(binding.open_upper[col_idx][interval_idx])
                            : 0;
                }
            }
        }
    }

    return tensors;
}

FlatSecureQueryPayload empty_secure_query_payload_from_public_doc(const json& public_doc) {
    FlatSecureQueryPayload payload;
    payload.query_skeleton_id = public_doc.value("query_skeleton_id", std::string());
    payload.binding_layout_kind = public_doc.value("binding_layout_kind", std::string());
    if (public_doc.contains("secret_tensor_shape") && public_doc["secret_tensor_shape"].is_object()) {
        const auto& shape_doc = public_doc["secret_tensor_shape"];
        if (shape_doc.contains("slot_payload_shape") && shape_doc["slot_payload_shape"].is_object()) {
            payload.slot_count = shape_doc["slot_payload_shape"].value("slot_count", std::uint64_t(0));
            payload.max_interval_count = shape_doc["slot_payload_shape"].value("max_interval_count", std::uint64_t(0));
        }
        if (shape_doc.contains("factor_payload_shape") && shape_doc["factor_payload_shape"].is_object()) {
            payload.factor_count = shape_doc["factor_payload_shape"].value("factor_count", std::uint64_t(0));
            payload.max_factor_column_count = shape_doc["factor_payload_shape"].value("max_column_count", std::uint64_t(0));
            payload.max_interval_count = std::max<std::uint64_t>(
                payload.max_interval_count,
                shape_doc["factor_payload_shape"].value("max_interval_count", std::uint64_t(0)));
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
    auto secure_eval = evaluate_secure_bundle_impl_secure(
        public_doc,
        secure_payload,
        shared_query_payload,
        manifest_map,
        model_root,
        secure_context,
        preloaded_model_cache);
    if (secure_eval.has_result) {
        secure_eval.result_rational = normalize_factor_root_rational(
            secure_eval.result_rational,
            secure_context,
            secure_eval.root_division_payload_scale);
    }
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

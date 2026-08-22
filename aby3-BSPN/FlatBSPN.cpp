#include "FlatBSPN.h"
#include "aby3-Basic/BuildingBlocks.h"
#include "aby3-GORAM-Core/Basics.h"
#include <aby3/sh3/Sh3Piecewise.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace aby3 {

namespace {

using json = nlohmann::json;

struct BSPNNetworkConfig {
    std::array<std::string, 3> hosts;
    int basic_base_port;
};

std::array<std::string, 3> parse_bspn_hosts_env(const char* value) {
    std::array<std::string, 3> hosts = { "127.0.0.1", "127.0.0.1", "127.0.0.1" };
    if (value == nullptr || *value == '\0') {
        return hosts;
    }
    std::stringstream stream(value);
    std::string item;
    for (std::size_t idx = 0; idx < hosts.size() && std::getline(stream, item, ','); ++idx) {
        if (!item.empty()) {
            hosts[idx] = item;
        }
    }
    return hosts;
}

int parse_bspn_port_env(const char* value, int fallback) {
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 65533) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

std::uint64_t parse_bspn_u64_env(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        return fallback;
    }
    return static_cast<std::uint64_t>(parsed);
}

std::vector<std::uint32_t> parse_bspn_u32_list_env(const char* name) {
    std::vector<std::uint32_t> out;
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return out;
    }
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(item.c_str(), &end, 10);
        if (end != item.c_str() && *end == '\0') {
            out.push_back(static_cast<std::uint32_t>(parsed));
        }
    }
    return out;
}

std::uint64_t bspn_max_stacked_bitmap_rows() {
    return parse_bspn_u64_env("BSPN_MAX_STACKED_BITMAP_ROWS", std::uint64_t(1) << 20);
}

std::uint64_t bspn_bitmap_share_rows_per_chunk() {
    return parse_bspn_u64_env("BSPN_BITMAP_SHARE_ROWS_PER_CHUNK", std::uint64_t(1) << 20);
}

bool bspn_env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool remote_share_only_required() {
    return bspn_env_flag_enabled("SECURE_USE_REMOTE_ABY3") ||
        bspn_env_flag_enabled("SECURE_REMOTE_ABY3");
}

i64 add_i64_mod(i64 left, i64 right) {
    return static_cast<i64>(
        static_cast<std::uint64_t>(left) + static_cast<std::uint64_t>(right));
}

bool bspn_use_row_value_eval() {
    const char* value = std::getenv("BSPN_USE_ROW_VALUE_EVAL");
    if (value == nullptr || *value == '\0') {
        return true;
    }
    const std::string normalized(value);
    return normalized != "0" && normalized != "false" && normalized != "FALSE" && normalized != "off";
}

BSPNNetworkConfig bspn_network_config_from_env() {
    BSPNNetworkConfig config;
    config.hosts = parse_bspn_hosts_env(std::getenv("ABY3_PARTY_HOSTS"));
    config.basic_base_port = parse_bspn_port_env(std::getenv("ABY3_BASIC_BASE_PORT"), 1213);
    return config;
}

std::string bspn_endpoint(const std::string& host, int port) {
    return host + ":" + std::to_string(port);
}

std::string bspn_server_endpoint(int port) {
    return bspn_endpoint("0.0.0.0", port);
}

void bspn_basic_setup(
    u64 party_idx,
    oc::IOService& ios,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime) {
    const auto net = bspn_network_config_from_env();
    const int base_port = net.basic_base_port;
    CommPkg comm;
    switch (party_idx) {
        case 0:
            comm.mNext = oc::Session(
                ios,
                bspn_server_endpoint(base_port),
                oc::SessionMode::Server,
                "01").addChannel();
            comm.mPrev = oc::Session(
                ios,
                bspn_server_endpoint(base_port + 1),
                oc::SessionMode::Server,
                "02").addChannel();
            break;
        case 1:
            comm.mNext = oc::Session(
                ios,
                bspn_server_endpoint(base_port + 2),
                oc::SessionMode::Server,
                "12").addChannel();
            comm.mPrev = oc::Session(
                ios,
                bspn_endpoint(net.hosts[0], base_port),
                oc::SessionMode::Client,
                "01").addChannel();
            break;
        default:
            comm.mNext = oc::Session(
                ios,
                bspn_endpoint(net.hosts[0], base_port + 1),
                oc::SessionMode::Client,
                "02").addChannel();
            comm.mPrev = oc::Session(
                ios,
                bspn_endpoint(net.hosts[1], base_port + 2),
                oc::SessionMode::Client,
                "12").addChannel();
            break;
    }
    enc.init(party_idx, comm, oc::sysRandomSeed());
    eval.init(party_idx, comm, oc::sysRandomSeed());
    runtime.init(party_idx, comm);
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

std::vector<i64> read_i64_share_pair_records(const std::string& path) {
    return read_binary_records<i64>(path);
}

std::string join_path(const std::string& base_dir, const std::string& file_name) {
    if (!file_name.empty() && file_name.front() == '/') {
        return file_name;
    }
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

template <Decimal D>
sf64Matrix<D> read_fixed_share_pair_matrix(
    const std::string& path,
    std::size_t expected_rows,
    std::size_t expected_cols) {
    const auto raw = read_i64_share_pair_records(path);
    if (raw.size() != expected_rows * expected_cols * 2) {
        throw std::runtime_error("Fixed share file shape mismatch: " + path);
    }
    sf64Matrix<D> out(expected_rows, expected_cols);
    for (std::size_t row = 0; row < expected_rows; ++row) {
        for (std::size_t col = 0; col < expected_cols; ++col) {
            const std::size_t idx = (row * expected_cols + col) * 2;
            out[0](static_cast<u64>(row), static_cast<u64>(col)) = raw[idx];
            out[1](static_cast<u64>(row), static_cast<u64>(col)) = raw[idx + 1];
        }
    }
    return out;
}

template <Decimal D>
sf64Matrix<D> read_fixed_share_pair_column(
    const std::string& path,
    std::size_t expected_rows) {
    return read_fixed_share_pair_matrix<D>(path, expected_rows, 1);
}

si64Matrix read_int_share_pair_matrix(
    const std::string& path,
    std::size_t expected_rows,
    std::size_t expected_cols) {
    const auto raw = read_i64_share_pair_records(path);
    if (raw.size() != expected_rows * expected_cols * 2) {
        throw std::runtime_error("Int share file shape mismatch: " + path);
    }
    si64Matrix out(expected_rows, expected_cols);
    for (std::size_t row = 0; row < expected_rows; ++row) {
        for (std::size_t col = 0; col < expected_cols; ++col) {
            const std::size_t idx = (row * expected_cols + col) * 2;
            out[0](static_cast<u64>(row), static_cast<u64>(col)) = raw[idx];
            out[1](static_cast<u64>(row), static_cast<u64>(col)) = raw[idx + 1];
        }
    }
    return out;
}

sbMatrix read_bool_share_pair_column(
    const std::string& path,
    std::size_t expected_rows) {
    const auto raw = read_i64_share_pair_records(path);
    if (raw.size() != expected_rows * 2) {
        throw std::runtime_error("Boolean share file row count mismatch: " + path);
    }
    sbMatrix out(static_cast<u64>(expected_rows), 64);
    for (std::size_t row = 0; row < expected_rows; ++row) {
        out.mShares[0](static_cast<u64>(row), 0) = raw[row * 2];
        out.mShares[1](static_cast<u64>(row), 0) = raw[row * 2 + 1];
    }
    for (u64 row = 0; row < out.rows(); ++row) {
        for (u64 col = 1; col < static_cast<u64>(out.mShares[0].cols()); ++col) {
            out.mShares[0](row, col) = 0;
            out.mShares[1](row, col) = 0;
        }
    }
    return out;
}

sbMatrix read_bool_share_pair_column_bitpacked(
    const std::string& path,
    std::size_t expected_rows) {
    const auto raw = read_binary_bytes(path);
    const std::size_t bytes_per_share = (expected_rows + 7) / 8;
    if (raw.size() != bytes_per_share * 2) {
        throw std::runtime_error("Bit-packed boolean share file row count mismatch: " + path);
    }
    sbMatrix out(static_cast<u64>(expected_rows), 64);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    for (std::size_t row = 0; row < expected_rows; ++row) {
        const std::size_t byte_idx = row / 8;
        const std::uint8_t mask = static_cast<std::uint8_t>(1u << (row % 8));
        out.mShares[0](static_cast<u64>(row), 0) = (raw[byte_idx] & mask) ? 1 : 0;
        out.mShares[1](static_cast<u64>(row), 0) = (raw[bytes_per_share + byte_idx] & mask) ? 1 : 0;
    }
    return out;
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

void unpack_bitmap_into_dense_rows(
    const std::vector<std::uint8_t>& bytes,
    std::uint64_t total_rows,
    i64Matrix& out,
    std::uint64_t row_offset) {
    for (std::uint64_t row = 0; row < total_rows; ++row) {
        const std::size_t byte_idx = static_cast<std::size_t>(row / 8);
        const std::size_t bit_idx = static_cast<std::size_t>(row % 8);
        if (byte_idx < bytes.size()) {
            out(static_cast<u64>(row_offset + row), 0) =
                static_cast<i64>((bytes[byte_idx] >> bit_idx) & std::uint8_t(1));
        }
    }
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

si64Matrix repeat_int_scalar_matrix(const si64Matrix& scalar, u64 rows, u64 cols) {
    si64Matrix out(rows, cols);
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < cols; ++col) {
            out.mShares[0](row, col) = scalar.mShares[0](0, 0);
            out.mShares[1](row, col) = scalar.mShares[1](0, 0);
        }
    }
    return out;
}

double reveal_fixed_scalar(
    const sf64Matrix<kFlatBSPNDecimal>& shared,
    const FlatBSPNSecureContext& context) {
    f64Matrix<kFlatBSPNDecimal> plain(shared.rows(), shared.cols());
    context.enc->revealAll(context.runtime->noDependencies(), shared, plain).get();
    return static_cast<double>(plain(0, 0));
}

i64 reveal_int_cell(
    const si64Matrix& shared,
    u64 row,
    u64 col,
    const FlatBSPNSecureContext& context) {
    i64Matrix plain(shared.rows(), shared.cols());
    context.enc->revealAll(context.runtime->noDependencies(), shared, plain).get();
    return plain(row, col);
}

double reveal_fixed_cell(
    const sf64Matrix<kFlatBSPNDecimal>& shared,
    u64 row,
    u64 col,
    const FlatBSPNSecureContext& context) {
    f64Matrix<kFlatBSPNDecimal> plain(shared.rows(), shared.cols());
    context.enc->revealAll(context.runtime->noDependencies(), shared, plain).get();
    return static_cast<double>(plain(row, col));
}

struct SecureRationalShare {
    sf64Matrix<kFlatBSPNDecimal> numerator;
    sf64Matrix<kFlatBSPNDecimal> denominator;
    double numerator_scale = 1.0;
    double denominator_scale = 1.0;
    bool denominator_is_one = false;
    bool has_secret_non_unit_denominator = false;
    sbMatrix secret_non_unit_denominator;
    bool has_secret_zero_numerator = false;
    sbMatrix secret_zero_numerator;
};

struct SecureFixedScalarShare {
    sf64Matrix<kFlatBSPNDecimal> value;
    double scale = 1.0;
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

double reveal_scaled_fixed_scalar(
    const SecureFixedScalarShare& value,
    const FlatBSPNSecureContext& context) {
    return reveal_fixed_scalar(value.value, context) * value.scale;
}

void synchronize_secure_parties(const FlatBSPNSecureContext& context) {
    if (!context.has_runtime()) {
        throw std::runtime_error("FlatBSPNSecureContext runtime is not initialized.");
    }
    std::uint64_t token = 0;
    std::uint64_t from_prev = 0;
    std::uint64_t from_next = 0;
    auto send_next = context.runtime->mComm.mNext.asyncSendFuture(&token, 1);
    auto send_prev = context.runtime->mComm.mPrev.asyncSendFuture(&token, 1);
    auto recv_prev = context.runtime->mComm.mPrev.asyncRecv(&from_prev, 1);
    auto recv_next = context.runtime->mComm.mNext.asyncRecv(&from_next, 1);
    recv_prev.get();
    recv_next.get();
    send_next.get();
    send_prev.get();
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
    bool root_division_scale_denominator_payload = false;
    json debug_output;
    json factor_trace_shares = json::array();
    json timing_profile;
};

struct SecureIndicatorEvalStats {
    std::uint64_t internal_reciprocal_calls = 0;
    std::uint64_t factor_root_divisions = 0;
    std::uint64_t phase1_batch_dot_calls = 0;
    std::uint64_t phase1_match_batches = 0;
    std::uint64_t phase2_count_batches = 0;
    std::uint64_t phase3_batch_b2a_calls = 0;
    std::uint64_t row_value_eval_used = 0;
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

json secure_indicator_stats_json(const SecureIndicatorEvalStats& stats) {
    return {
        {"internal_reciprocal_calls", stats.internal_reciprocal_calls},
        {"factor_root_divisions", stats.factor_root_divisions},
        {"phase1_batch_dot_calls", stats.phase1_batch_dot_calls},
        {"phase1_match_batches", stats.phase1_match_batches},
        {"phase2_count_batches", stats.phase2_count_batches},
        {"phase3_batch_b2a_calls", stats.phase3_batch_b2a_calls},
        {"row_value_eval_used", stats.row_value_eval_used},
        {"leaf_product_groups", stats.leaf_product_groups},
        {"leaf_product_nodes", stats.leaf_product_nodes},
        {"max_stacked_bitmap_rows", bspn_max_stacked_bitmap_rows()},
        {"bitmap_share_rows_per_chunk", bspn_bitmap_share_rows_per_chunk()},
        {"timing_ms", {
            {"sum_node", stats.sum_node_ms},
            {"product_sum", stats.product_sum_ms},
            {"phase1_match", stats.phase1_match_ms},
            {"phase1_local_ids", stats.phase1_local_ids_ms},
            {"phase2_intersection", stats.phase2_intersection_ms},
            {"phase2_count", stats.phase2_count_ms},
            {"phase3_numerator", stats.phase3_numerator_ms},
            {"final_combine", stats.final_combine_ms},
        }},
    };
}

using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = SteadyClock::time_point;

double elapsed_ms_since(const SteadyTimePoint& start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
}

int bspn_openmp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
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

sf64Matrix<kFlatBSPNDecimal> secure_mul_fixed_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& lhs,
    const sf64Matrix<kFlatBSPNDecimal>& rhs,
    const FlatBSPNSecureContext& context);

sf64Matrix<kFlatBSPNDecimal> fixed_mul_bool_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& values,
    sbMatrix flags,
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

sf64Matrix<kFlatBSPNDecimal> gather_fixed_rows(
    const sf64Matrix<kFlatBSPNDecimal>& src,
    const std::vector<std::uint32_t>& rows) {
    sf64Matrix<kFlatBSPNDecimal> out(static_cast<u64>(rows.size()), src.cols());
    #pragma omp parallel for schedule(static)
    for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(rows.size()); ++idx) {
        const auto src_row = rows[static_cast<std::size_t>(idx)];
        for (u64 col = 0; col < static_cast<u64>(src.cols()); ++col) {
            out[0](static_cast<u64>(idx), col) = src[0](src_row, col);
            out[1](static_cast<u64>(idx), col) = src[1](src_row, col);
        }
    }
    return out;
}

sf64Matrix<kFlatBSPNDecimal> stack_fixed_scalars(
    const std::vector<sf64Matrix<kFlatBSPNDecimal>>& values) {
    sf64Matrix<kFlatBSPNDecimal> out(static_cast<u64>(values.size()), 1);
    #pragma omp parallel for schedule(static)
    for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(values.size()); ++idx) {
        out[0](static_cast<u64>(idx), 0) = values[static_cast<std::size_t>(idx)][0](0, 0);
        out[1](static_cast<u64>(idx), 0) = values[static_cast<std::size_t>(idx)][1](0, 0);
    }
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

sbMatrix bool_row_slice(
    const sbMatrix& src,
    std::uint32_t row_begin,
    std::uint32_t row_count) {
    sbMatrix out(row_count, src.bitCount());
    for (std::uint32_t row = 0; row < row_count; ++row) {
        for (u64 col = 0; col < static_cast<u64>(src.mShares[0].cols()); ++col) {
            out.mShares[0](row, col) = src.mShares[0](row_begin + row, col);
            out.mShares[1](row, col) = src.mShares[1](row_begin + row, col);
        }
    }
    return out;
}

sbMatrix rational_zero_numerator_flag(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context) {
    if (value.has_secret_zero_numerator) {
        return value.secret_zero_numerator;
    }
    return shared_zero_bool_scalar(context);
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

sf64Matrix<kFlatBSPNDecimal> bool_matrix_to_fixed_same_shape(
    const sbMatrix& flags,
    const FlatBSPNSecureContext& context) {
    sbMatrix flags_copy = flags;
    si64Matrix as_int(flags.rows(), 1);
    bool2arith(context.role, flags_copy, as_int, *(context.enc), *(context.eval), *(context.runtime));
    return si64_to_sf64(as_int);
}

sf64Matrix<kFlatBSPNDecimal> select_fixed_by_bool(
    const sf64Matrix<kFlatBSPNDecimal>& true_value,
    const sf64Matrix<kFlatBSPNDecimal>& false_value,
    const sbMatrix& flag,
    const FlatBSPNSecureContext& context) {
    auto delta = true_value - false_value;
    auto selected_delta = fixed_mul_bool_same_shape(delta, flag, context);
    return false_value + selected_delta;
}

sf64Matrix<kFlatBSPNDecimal> select_fixed_by_bool_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& true_value,
    const sf64Matrix<kFlatBSPNDecimal>& false_value,
    const sbMatrix& flags,
    const FlatBSPNSecureContext& context) {
    if (true_value.rows() != false_value.rows() || true_value.cols() != false_value.cols() ||
        true_value.rows() != flags.rows()) {
        throw std::runtime_error("select_fixed_by_bool_same_shape shape mismatch.");
    }
    auto delta = true_value - false_value;
    auto selected_delta = fixed_mul_bool_same_shape(delta, flags, context);
    return false_value + selected_delta;
}

sf64Matrix<kFlatBSPNDecimal> secure_abs_fixed_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& values,
    const FlatBSPNSecureContext& context) {
    sf64Matrix<kFlatBSPNDecimal> zero(values.rows(), values.cols());
    zero[0].setZero();
    zero[1].setZero();
    auto zero_for_cmp = zero;
    auto values_for_cmp = values;
    sbMatrix is_negative;
    cipher_gt(context.role, zero_for_cmp, values_for_cmp, is_negative, *(context.eval), *(context.runtime));
    return select_fixed_by_bool_same_shape(zero - values, values, is_negative, context);
}

sf64Matrix<kFlatBSPNDecimal> secure_nonnegative_fixed_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& values,
    const FlatBSPNSecureContext& context) {
    sf64Matrix<kFlatBSPNDecimal> zero(values.rows(), values.cols());
    zero[0].setZero();
    zero[1].setZero();
    auto values_for_cmp = values;
    auto zero_for_cmp = zero;
    sbMatrix is_positive;
    cipher_gt(context.role, values_for_cmp, zero_for_cmp, is_positive, *(context.eval), *(context.runtime));
    return fixed_mul_bool_same_shape(values, is_positive, context);
}

sbMatrix stack_bool_scalars(const std::vector<sbMatrix>& values) {
    if (values.empty()) {
        sbMatrix out(0, 1);
        bool_init_false(0, out);
        return out;
    }
    const u64 bit_count = values.front().bitCount();
    const u64 share_cols = values.front().mShares[0].cols();
    for (const auto& value : values) {
        if (value.rows() != 1 || value.bitCount() != bit_count) {
            throw std::runtime_error("Boolean scalar shapes do not match for stacking.");
        }
    }
    sbMatrix out(values.size(), bit_count);
    #pragma omp parallel for schedule(static)
    for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(values.size()); ++idx) {
        const auto& value = values[static_cast<std::size_t>(idx)];
        for (u64 col = 0; col < share_cols; ++col) {
            out.mShares[0](static_cast<u64>(idx), col) = value.mShares[0](0, col);
            out.mShares[1](static_cast<u64>(idx), col) = value.mShares[1](0, col);
        }
    }
    return out;
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
    SecureRationalShare out{
        select_fixed_by_bool(aligned_true.numerator, aligned_false.numerator, flag, context),
        select_fixed_by_bool(aligned_true.denominator, aligned_false.denominator, flag, context),
        target_numerator_scale,
        target_denominator_scale,
        true_value.denominator_is_one && false_value.denominator_is_one,
    };
    if (!out.denominator_is_one) {
        const auto true_non_unit = rational_non_unit_denominator_flag(aligned_true, context);
        const auto false_non_unit = rational_non_unit_denominator_flag(aligned_false, context);
        const auto true_selected = bool_and_scalar(flag, true_non_unit, context);
        const auto false_selected = bool_and_scalar(bool_not_scalar(flag, context), false_non_unit, context);
        out.has_secret_non_unit_denominator = true;
        out.secret_non_unit_denominator = bool_or_scalar(true_selected, false_selected, context);
    }
    const auto true_zero = rational_zero_numerator_flag(aligned_true, context);
    const auto false_zero = rational_zero_numerator_flag(aligned_false, context);
    const auto true_zero_selected = bool_and_scalar(flag, true_zero, context);
    const auto false_zero_selected = bool_and_scalar(bool_not_scalar(flag, context), false_zero, context);
    out.has_secret_zero_numerator = true;
    out.secret_zero_numerator = bool_or_scalar(true_zero_selected, false_zero_selected, context);
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

std::vector<std::uint8_t> public_scope_complement(
    const std::vector<std::uint8_t>& public_scope_mask,
    std::size_t total_columns) {
    std::vector<std::uint8_t> out(total_columns, 1);
    const std::size_t limit = std::min<std::size_t>(public_scope_mask.size(), total_columns);
    for (std::size_t idx = 0; idx < limit; ++idx) {
        out[idx] = public_scope_mask[idx] == 0 ? 1 : 0;
    }
    return out;
}

sbMatrix secure_scope_missing_from_public_scope(
    const std::vector<std::uint8_t>& public_scope_mask,
    const si64Matrix& secret_scope_row,
    const FlatBSPNSecureContext& context) {
    return secure_scope_intersects(
        public_scope_complement(
            public_scope_mask,
            static_cast<std::size_t>(secret_scope_row.cols())),
        secret_scope_row,
        context);
}

sbMatrix bool_row_slice(
    const sbMatrix& src,
    u64 row) {
    sbMatrix out(1, src.bitCount());
    for (u64 col = 0; col < static_cast<u64>(src.mShares[0].cols()); ++col) {
        out.mShares[0](0, col) = src.mShares[0](row, col);
        out.mShares[1](0, col) = src.mShares[1](row, col);
    }
    return out;
}

sbMatrix secure_scope_intersects_shared_matrix(
    const si64Matrix& node_scope_rows,
    const si64Matrix& secret_scope_row,
    const FlatBSPNSecureContext& context) {
    const u64 rows = static_cast<u64>(node_scope_rows.rows());
    const u64 cols = std::min<u64>(
        static_cast<u64>(node_scope_rows.cols()),
        static_cast<u64>(secret_scope_row.cols()));
    if (rows == 0 || cols == 0) {
        sbMatrix out(rows, 1);
        bool_init_false(context.role, out);
        return out;
    }

    si64Matrix scoped_rows(rows, cols);
    si64Matrix repeated_scope(rows, cols);
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < cols; ++col) {
            scoped_rows.mShares[0](row, col) = node_scope_rows.mShares[0](row, col);
            scoped_rows.mShares[1](row, col) = node_scope_rows.mShares[1](row, col);
            repeated_scope.mShares[0](row, col) = secret_scope_row.mShares[0](0, col);
            repeated_scope.mShares[1](row, col) = secret_scope_row.mShares[1](0, col);
        }
    }

    si64Matrix products(rows, cols);
    cipher_mul(
        context.role,
        scoped_rows,
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
    return flags;
}

sbMatrix secure_scope_intersects_shared(
    const si64Matrix& secret_node_scope_row,
    const si64Matrix& secret_scope_row,
    const FlatBSPNSecureContext& context) {
    return bool_row_slice(
        secure_scope_intersects_shared_matrix(secret_node_scope_row, secret_scope_row, context),
        0);
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
    for (u64 row = 0; row < rows; ++row) {
        const auto node_id = node_ids[static_cast<std::size_t>(row)];
        if (node_id >= static_cast<std::uint32_t>(node_scopes.rows())) {
            throw std::runtime_error("secure node scope row is out of bounds");
        }
        for (u64 col = 0; col < cols; ++col) {
            child_scopes.mShares[0](row, col) = node_scopes.mShares[0](node_id, col);
            child_scopes.mShares[1](row, col) = node_scopes.mShares[1](node_id, col);
        }
    }

    const auto flags = secure_scope_intersects_shared_matrix(child_scopes, secret_scope_row, context);

    for (u64 row = 0; row < rows; ++row) {
        out.push_back(bool_row_slice(flags, row));
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
    if (std::abs(numerator) <= 1e-12) {
        out.has_secret_zero_numerator = true;
        out.secret_zero_numerator = shared_true_bool_scalar(context);
    }
    return out;
}

SecureRationalShare zero_rational_when_scope_missing(
    const SecureRationalShare& value,
    const std::vector<std::uint8_t>& public_scope_mask,
    const si64Matrix& secret_scope_row,
    const sbMatrix& domain_miss,
    const FlatBSPNSecureContext& context) {
    auto scope_missing = secure_scope_missing_from_public_scope(
        public_scope_mask,
        secret_scope_row,
        context);
    scope_missing = bool_and_scalar(scope_missing, domain_miss, context);
    return select_rational_by_bool(
        make_secure_rational(0.0, 1.0, context),
        value,
        scope_missing,
        context);
}

SecureRationalShare make_secure_public_scaled_constant(
    double value,
    const FlatBSPNSecureContext& context) {
    if (std::abs(value) <= 1e-12) {
        return make_secure_rational(0.0, 1.0, context);
    }
    SecureRationalShare out = make_secure_rational(value < 0.0 ? -1.0 : 1.0, 1.0, context);
    out.numerator_scale = std::abs(value);
    return out;
}

json secure_rational_debug_json(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context) {
    const double numerator_raw = reveal_fixed_scalar(value.numerator, context);
    const double denominator_raw = reveal_fixed_scalar(value.denominator, context);
    const double numerator = numerator_raw * value.numerator_scale;
    const double denominator = denominator_raw * value.denominator_scale;
    return {
        {"numerator", numerator},
        {"denominator", denominator},
        {"numerator_raw", numerator_raw},
        {"denominator_raw", denominator_raw},
        {"numerator_scale", value.numerator_scale},
        {"denominator_scale", value.denominator_scale},
        {"value", std::abs(denominator) <= 1e-12 ? 0.0 : numerator / denominator},
    };
}

json fixed_scalar_share_json(
    const sf64Matrix<kFlatBSPNDecimal>& value) {
    auto cell = const_cast<sf64Matrix<kFlatBSPNDecimal>&>(value)(0, 0);
    return {
        {"share0", static_cast<long long>(cell[0])},
        {"share1", static_cast<long long>(cell[1])},
        {"rows", static_cast<std::uint64_t>(value.rows())},
        {"cols", static_cast<std::uint64_t>(value.cols())},
    };
}

json bool_scalar_share_json(
    const sbMatrix& value) {
    return {
        {"share0", static_cast<int>(value.mShares[0](0, 0))},
        {"share1", static_cast<int>(value.mShares[1](0, 0))},
        {"rows", static_cast<std::uint64_t>(value.rows())},
        {"cols", static_cast<std::uint64_t>(value.mShares[0].cols())},
    };
}

json secure_rational_share_json(
    const SecureRationalShare& value) {
    json out = {
        {"numerator_share", fixed_scalar_share_json(value.numerator)},
        {"denominator_share", fixed_scalar_share_json(value.denominator)},
        {"numerator_scale", value.numerator_scale},
        {"denominator_scale", value.denominator_scale},
        {"denominator_is_one", value.denominator_is_one},
        {"has_secret_non_unit_denominator", value.has_secret_non_unit_denominator},
        {"has_secret_zero_numerator", value.has_secret_zero_numerator},
        {"fixed_decimal_bits", static_cast<int>(kFlatBSPNDecimal)},
    };
    if (value.has_secret_non_unit_denominator) {
        out["secret_non_unit_denominator_share"] =
            bool_scalar_share_json(value.secret_non_unit_denominator);
    }
    if (value.has_secret_zero_numerator) {
        out["secret_zero_numerator_share"] =
            bool_scalar_share_json(value.secret_zero_numerator);
    }
    return out;
}

json secure_fixed_scalar_share_json(
    const SecureFixedScalarShare& value) {
    return {
        {"value_share", fixed_scalar_share_json(value.value)},
        {"scale", value.scale},
        {"fixed_decimal_bits", static_cast<int>(kFlatBSPNDecimal)},
    };
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

sf64Matrix<kFlatBSPNDecimal> fixed_mul_bool_same_shape(
    const sf64Matrix<kFlatBSPNDecimal>& values,
    sbMatrix flags,
    const FlatBSPNSecureContext& context) {
    if (values.rows() != flags.rows() || values.cols() != 1) {
        throw std::runtime_error("fixed_mul_bool_same_shape expects fixed column values and one flag per row.");
    }
    si64Matrix value_int(values.rows(), values.cols());
    value_int.mShares[0] = values[0];
    value_int.mShares[1] = values[1];

    si64Matrix flag_int(flags.rows(), 1);
    flag_int.mShares[0].setZero();
    flag_int.mShares[1].setZero();
    bool2arith(context.role, flags, flag_int, *(context.enc), *(context.eval), *(context.runtime));

    si64Matrix selected_int(values.rows(), values.cols());
    selected_int.mShares[0].setZero();
    selected_int.mShares[1].setZero();
    cipher_mul(
        context.role,
        value_int,
        flag_int,
        selected_int,
        *(context.eval),
        *(context.enc),
        *(context.runtime));

    sf64Matrix<kFlatBSPNDecimal> out(values.rows(), values.cols());
    out[0] = selected_int.mShares[0];
    out[1] = selected_int.mShares[1];
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

SecureFixedScalarShare secure_divide_rational_to_fixed_scalar(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context,
    double public_payload_scale = 1.0,
    bool scale_denominator_payload = false,
    std::size_t iterations = 16) {
    auto zero_fixed_scalar_when_needed = [&](
        const sf64Matrix<kFlatBSPNDecimal>& scalar_value) {
        const auto zero_numerator = rational_zero_numerator_flag(value, context);
        return fixed_mul_bool_same_shape(
            scalar_value,
            bool_not_scalar(zero_numerator, context),
            context);
    };
    if (value.denominator_is_one) {
        return {
            zero_fixed_scalar_when_needed(value.numerator),
            value.numerator_scale,
        };
    }
    // The final scalar conversion multiplies the numerator payload by the
    // reciprocal of the denominator payload. Keep the denominator close to the
    // original probability-like unit for Newton convergence, and shrink only
    // the numerator payload when public scaling has made it too large.
    const double payload_scale =
        std::isfinite(public_payload_scale) && public_payload_scale > 0.0
            ? public_payload_scale
            : 1.0;
    SecureRationalShare scaled_value = value;
    if (payload_scale != 1.0) {
        const double numerator_payload_scale =
            scale_denominator_payload ? 1.0 : payload_scale;
        scaled_value.numerator = secure_mul_public_fixed(
            value.numerator,
            numerator_payload_scale,
            context);
        scaled_value.numerator_scale = value.numerator_scale / numerator_payload_scale;
        if (scale_denominator_payload) {
            const double denominator_payload_scale = std::max(payload_scale, 1.0 / 4096.0);
            scaled_value.denominator = secure_mul_public_fixed(
                value.denominator,
                denominator_payload_scale,
                context);
            scaled_value.denominator_scale = value.denominator_scale / denominator_payload_scale;
        }
    }
    if (context.debug_internal_reveal) {
        const double denominator = reveal_scaled_denominator(scaled_value, context);
        if (!std::isfinite(denominator) || std::abs(denominator) <= 1e-12) {
            throw std::runtime_error("Secure factor root denominator is too small for root-only division.");
        }
    }
    const auto inv_den = secure_fixed_reciprocal_newton(scaled_value.denominator, context, iterations);
    const auto scalar = zero_fixed_scalar_when_needed(
        secure_mul_fixed(scaled_value.numerator, inv_den, context));
    return {
        scalar,
        scaled_value.numerator_scale / scaled_value.denominator_scale,
    };
}

SecureRationalShare normalize_factor_root_rational(
    const SecureRationalShare& value,
    const FlatBSPNSecureContext& context,
    double public_payload_scale = 1.0,
    bool scale_denominator_payload = false) {
    const auto scalar = secure_divide_rational_to_fixed_scalar(
        value,
        context,
        public_payload_scale,
        scale_denominator_payload);
    return {
        scalar.value,
        share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
        scalar.scale,
        1.0,
        true,
    };
}

bool has_large_public_rational_scale(const SecureRationalShare& value) {
    return std::abs(value.numerator_scale) > 1000000.0 ||
           std::abs(value.denominator_scale) > 1000000.0;
}

SecureRationalShare multiply_secure_rational(
    const SecureRationalShare& lhs,
    const SecureRationalShare& rhs,
    const FlatBSPNSecureContext& context) {
    SecureRationalShare out;
    out.numerator = secure_mul_fixed(lhs.numerator, rhs.numerator, context);
    out.numerator_scale = lhs.numerator_scale * rhs.numerator_scale;
    const auto lhs_zero = rational_zero_numerator_flag(lhs, context);
    const auto rhs_zero = rational_zero_numerator_flag(rhs, context);
    const auto product_zero = bool_or_scalar(lhs_zero, rhs_zero, context);
    out.numerator = fixed_mul_bool_same_shape(
        out.numerator,
        bool_not_scalar(product_zero, context),
        context);
    out.has_secret_zero_numerator = true;
    out.secret_zero_numerator = product_zero;
    if (lhs.denominator_is_one && rhs.denominator_is_one) {
        out.denominator = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
        out.denominator_scale = 1.0;
        out.denominator_is_one = true;
        return out;
    } else {
        out.denominator = secure_mul_fixed(lhs.denominator, rhs.denominator, context);
        out.denominator_scale = lhs.denominator_scale * rhs.denominator_scale;
    }
    out.denominator_is_one = false;
    if (has_large_public_rational_scale(lhs) || has_large_public_rational_scale(rhs)) {
        return out;
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
    if (payload.binding_layout_kind != "DENSE_FACTOR_COLUMNS_V1") {
        throw std::runtime_error("Secure query payload must use DENSE_FACTOR_COLUMNS_V1.");
    }
    if (doc.contains("factor_payload_shape") && doc["factor_payload_shape"].is_object()) {
        payload.factor_count = doc["factor_payload_shape"].value("factor_count", std::uint64_t(0));
        payload.max_factor_column_count = doc["factor_payload_shape"].value("max_column_count", std::uint64_t(0));
        payload.max_interval_count = doc["factor_payload_shape"].value("max_interval_count", std::uint64_t(0));
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
    }
    #pragma omp parallel for schedule(static)
    for (std::int64_t mask_idx_signed = 0; mask_idx_signed < static_cast<std::int64_t>(masks.size()); ++mask_idx_signed) {
        const auto mask_idx = static_cast<std::size_t>(mask_idx_signed);
        const auto& mask = masks[mask_idx];
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
    #pragma omp parallel for schedule(static)
    for (std::int64_t mask_idx_signed = 0; mask_idx_signed < static_cast<std::int64_t>(masks.size()); ++mask_idx_signed) {
        const auto mask_idx = static_cast<std::size_t>(mask_idx_signed);
        const u64 row_begin = static_cast<u64>(mask_idx) * mask_rows;
        counts.mShares[0](static_cast<u64>(mask_idx), 0) =
            stacked_int.mShares[0].block(row_begin, 0, mask_rows, 1).sum();
        counts.mShares[1](static_cast<u64>(mask_idx), 0) =
            stacked_int.mShares[1].block(row_begin, 0, mask_rows, 1).sum();
    }
    return counts;
}

sf64Matrix<kFlatBSPNDecimal> sum_boolean_masks_weighted_to_fixed_batched(
    const std::vector<sbMatrix>& masks,
    const sf64Matrix<kFlatBSPNDecimal>& row_weights,
    const FlatBSPNSecureContext& context,
    std::uint64_t* phase2_batch_counter = nullptr) {
    if (masks.empty()) {
        return sf64Matrix<kFlatBSPNDecimal>(0, 1);
    }
    const u64 mask_rows = masks.front().rows();
    const u64 bit_count = masks.front().bitCount();
    const u64 share_cols = masks.front().mShares[0].cols();
    if (row_weights.rows() != mask_rows || row_weights.cols() != 1) {
        throw std::runtime_error("Row weight shape does not match boolean masks.");
    }
    const u64 stacked_rows = mask_rows * static_cast<u64>(masks.size());
    sbMatrix stacked_masks(stacked_rows, bit_count);
    sf64Matrix<kFlatBSPNDecimal> stacked_weights(stacked_rows, 1);
    for (std::size_t mask_idx = 0; mask_idx < masks.size(); ++mask_idx) {
        const auto& mask = masks[mask_idx];
        if (mask.rows() != mask_rows || mask.bitCount() != bit_count) {
            throw std::runtime_error("Boolean mask shapes do not match for weighted batched count.");
        }
    }
    #pragma omp parallel for schedule(static)
    for (std::int64_t mask_idx_signed = 0; mask_idx_signed < static_cast<std::int64_t>(masks.size()); ++mask_idx_signed) {
        const auto mask_idx = static_cast<std::size_t>(mask_idx_signed);
        const auto& mask = masks[mask_idx];
        const u64 row_begin = static_cast<u64>(mask_idx) * mask_rows;
        for (u64 row = 0; row < mask_rows; ++row) {
            for (u64 col = 0; col < share_cols; ++col) {
                stacked_masks.mShares[0](row_begin + row, col) = mask.mShares[0](row, col);
                stacked_masks.mShares[1](row_begin + row, col) = mask.mShares[1](row, col);
            }
            stacked_weights[0](row_begin + row, 0) = row_weights[0](row, 0);
            stacked_weights[1](row_begin + row, 0) = row_weights[1](row, 0);
        }
    }

    const auto selected_weights = fixed_mul_bool_same_shape(stacked_weights, stacked_masks, context);
    if (phase2_batch_counter != nullptr) {
        ++(*phase2_batch_counter);
    }

    sf64Matrix<kFlatBSPNDecimal> counts(static_cast<u64>(masks.size()), 1);
    #pragma omp parallel for schedule(static)
    for (std::int64_t mask_idx_signed = 0; mask_idx_signed < static_cast<std::int64_t>(masks.size()); ++mask_idx_signed) {
        const auto mask_idx = static_cast<std::size_t>(mask_idx_signed);
        const u64 row_begin = static_cast<u64>(mask_idx) * mask_rows;
        counts[0](static_cast<u64>(mask_idx), 0) =
            selected_weights[0].block(row_begin, 0, mask_rows, 1).sum();
        counts[1](static_cast<u64>(mask_idx), 0) =
            selected_weights[1].block(row_begin, 0, mask_rows, 1).sum();
    }
    return counts;
}

struct LeafProductBatchItem {
    std::size_t node_idx = 0;
    std::size_t leaf_begin = 0;
    std::size_t leaf_count = 0;
};

sf64Matrix<kFlatBSPNDecimal> sum_single_leaf_match_bucket_weights_batched(
    const FlatBSPNModel& model,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const std::vector<LeafProductBatchItem>& product_items,
    const std::vector<sbMatrix>& match_masks,
    const FlatBSPNSecureContext& context,
    std::uint64_t* batch_counter = nullptr) {
    const auto& payload = model.secret_shared_payload();
    if (!payload.leaf_bucket_weight_sums_loaded ||
        payload.leaf_bucket_weight_sums.rows() != model.manifest().bucket_count ||
        payload.leaf_bucket_weight_sums.cols() != 1) {
        throw std::runtime_error("Leaf bucket weight sums are not loaded or have the wrong shape.");
    }

    struct BucketRef {
        std::size_t product_idx = 0;
        std::size_t child_idx = 0;
        std::uint32_t bucket_index = 0;
    };
    std::vector<BucketRef> bucket_refs;
    for (std::size_t product_idx = 0; product_idx < product_items.size(); ++product_idx) {
        const auto& item = product_items[product_idx];
        if (item.leaf_count != 1) {
            throw std::runtime_error("single-leaf bucket weight count requested for a multi-leaf product.");
        }
        const auto child_idx = item.leaf_begin;
        if (child_idx >= leaf_children.size() || child_idx >= match_masks.size()) {
            throw std::runtime_error("Leaf child index out of bounds for bucket weight count.");
        }
        const auto& child = *leaf_children[child_idx];
        for (std::uint32_t bucket_offset = 0; bucket_offset < child.bucket_count; ++bucket_offset) {
            bucket_refs.push_back({product_idx, child_idx, child.bucket_begin + bucket_offset});
        }
    }

    sf64Matrix<kFlatBSPNDecimal> counts(static_cast<u64>(product_items.size()), 1);
    counts[0].setZero();
    counts[1].setZero();
    if (bucket_refs.empty()) {
        return counts;
    }

    const u64 kMaxStackedBucketRows = bspn_max_stacked_bitmap_rows();
    const std::size_t buckets_per_chunk = static_cast<std::size_t>(std::max<u64>(1, kMaxStackedBucketRows));
    for (std::size_t chunk_begin = 0; chunk_begin < bucket_refs.size(); chunk_begin += buckets_per_chunk) {
        const std::size_t chunk_end = std::min<std::size_t>(bucket_refs.size(), chunk_begin + buckets_per_chunk);
        const std::size_t chunk_bucket_count = chunk_end - chunk_begin;
        sbMatrix stacked_matches(static_cast<u64>(chunk_bucket_count), 1);
        sf64Matrix<kFlatBSPNDecimal> stacked_weight_sums(static_cast<u64>(chunk_bucket_count), 1);

        for (std::size_t local_idx = 0; local_idx < chunk_bucket_count; ++local_idx) {
            const auto& ref = bucket_refs[chunk_begin + local_idx];
            const auto& child = *leaf_children[ref.child_idx];
            const auto local_bucket_idx = static_cast<u64>(ref.bucket_index - child.bucket_begin);
            const auto& match_mask = match_masks[ref.child_idx];
            if (match_mask.rows() != child.bucket_count) {
                throw std::runtime_error("Match mask shape does not match child bucket count.");
            }
            for (u64 col = 0; col < static_cast<u64>(stacked_matches.mShares[0].cols()); ++col) {
                stacked_matches.mShares[0](static_cast<u64>(local_idx), col) =
                    match_mask.mShares[0](local_bucket_idx, col);
                stacked_matches.mShares[1](static_cast<u64>(local_idx), col) =
                    match_mask.mShares[1](local_bucket_idx, col);
            }
            stacked_weight_sums[0](static_cast<u64>(local_idx), 0) =
                payload.leaf_bucket_weight_sums[0](ref.bucket_index, 0);
            stacked_weight_sums[1](static_cast<u64>(local_idx), 0) =
                payload.leaf_bucket_weight_sums[1](ref.bucket_index, 0);
        }

        const auto selected_weight_sums =
            fixed_mul_bool_same_shape(stacked_weight_sums, stacked_matches, context);
        if (batch_counter != nullptr) {
            ++(*batch_counter);
        }

        std::vector<i64> chunk_sum0(product_items.size(), 0);
        std::vector<i64> chunk_sum1(product_items.size(), 0);
        for (std::size_t local_idx = 0; local_idx < chunk_bucket_count; ++local_idx) {
            const auto product_idx = bucket_refs[chunk_begin + local_idx].product_idx;
            chunk_sum0[product_idx] = add_i64_mod(
                chunk_sum0[product_idx],
                selected_weight_sums[0](static_cast<u64>(local_idx), 0));
            chunk_sum1[product_idx] = add_i64_mod(
                chunk_sum1[product_idx],
                selected_weight_sums[1](static_cast<u64>(local_idx), 0));
        }
        for (std::size_t product_idx = 0; product_idx < product_items.size(); ++product_idx) {
            counts[0](static_cast<u64>(product_idx), 0) = add_i64_mod(
                counts[0](static_cast<u64>(product_idx), 0),
                chunk_sum0[product_idx]);
            counts[1](static_cast<u64>(product_idx), 0) = add_i64_mod(
                counts[1](static_cast<u64>(product_idx), 0),
                chunk_sum1[product_idx]);
        }
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
    const u64 kMaxStackedBitmapRows = bspn_max_stacked_bitmap_rows();
    const std::size_t buckets_per_chunk = static_cast<std::size_t>(
        std::max<u64>(1, kMaxStackedBitmapRows / std::max<u64>(1, mask_rows)));

    for (std::size_t chunk_begin = 0; chunk_begin < bucket_refs.size(); chunk_begin += buckets_per_chunk) {
        const std::size_t chunk_end = std::min<std::size_t>(bucket_refs.size(), chunk_begin + buckets_per_chunk);
        const std::size_t chunk_bucket_count = chunk_end - chunk_begin;
        const u64 stacked_rows = static_cast<u64>(chunk_bucket_count) * mask_rows;
        sbMatrix stacked_bitmaps(stacked_rows, bit_count);
        sbMatrix stacked_final_ids(stacked_rows, bit_count);

        for (std::size_t local_bucket_idx = 0; local_bucket_idx < chunk_bucket_count; ++local_bucket_idx) {
            const auto& bucket_ref = bucket_refs[chunk_begin + local_bucket_idx];
            const auto& bucket_bitmap =
                model.secret_shared_payload().dense_bucket_bitmaps[bucket_ref.bucket_index];
            const auto& final_ids = final_ids_by_product[bucket_ref.product_idx];
            if (bucket_bitmap.rows() != mask_rows || bucket_bitmap.bitCount() != bit_count ||
                final_ids.rows() != mask_rows || final_ids.bitCount() != bit_count) {
                throw std::runtime_error("Bucket/final id mask shape mismatch in group numerator batch.");
            }
        }

        #pragma omp parallel for schedule(static)
        for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
            const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
            const auto& bucket_ref = bucket_refs[chunk_begin + local_bucket_idx];
            const auto& bucket_bitmap =
                model.secret_shared_payload().dense_bucket_bitmaps[bucket_ref.bucket_index];
            const auto& final_ids = final_ids_by_product[bucket_ref.product_idx];
            const u64 row_begin = static_cast<u64>(local_bucket_idx) * mask_rows;
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

        sf64Matrix<kFlatBSPNDecimal> overlap_counts(static_cast<u64>(chunk_bucket_count), 1);
        sf64Matrix<kFlatBSPNDecimal> bucket_values(static_cast<u64>(chunk_bucket_count), 1);

        const auto& payload = model.secret_shared_payload();
        if (payload.row_weights_loaded) {
            if (payload.row_weights.rows() != mask_rows || payload.row_weights.cols() != 1) {
                throw std::runtime_error("Row weight shape does not match group numerator masks.");
            }
            sf64Matrix<kFlatBSPNDecimal> stacked_weights(stacked_rows, 1);
            #pragma omp parallel for schedule(static)
            for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
                const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
                const u64 row_begin = static_cast<u64>(local_bucket_idx) * mask_rows;
                for (u64 row = 0; row < mask_rows; ++row) {
                    stacked_weights[0](row_begin + row, 0) = payload.row_weights[0](row, 0);
                    stacked_weights[1](row_begin + row, 0) = payload.row_weights[1](row, 0);
                }
            }
            const auto selected_weights = fixed_mul_bool_same_shape(stacked_weights, stacked_overlap, context);
            if (phase3_batch_counter != nullptr) {
                ++(*phase3_batch_counter);
            }
            #pragma omp parallel for schedule(static)
            for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
                const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
                const u64 row_begin = static_cast<u64>(local_bucket_idx) * mask_rows;
                overlap_counts[0](static_cast<u64>(local_bucket_idx), 0) =
                    selected_weights[0].block(row_begin, 0, mask_rows, 1).sum();
                overlap_counts[1](static_cast<u64>(local_bucket_idx), 0) =
                    selected_weights[1].block(row_begin, 0, mask_rows, 1).sum();
            }
        } else {
            si64Matrix overlap_int(stacked_overlap.rows(), 1);
            bool2arith(context.role, stacked_overlap, overlap_int, *(context.enc), *(context.eval), *(context.runtime));
            if (phase3_batch_counter != nullptr) {
                ++(*phase3_batch_counter);
            }

            si64Matrix overlap_counts_int(static_cast<u64>(chunk_bucket_count), 1);
            #pragma omp parallel for schedule(static)
            for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
                const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
                const u64 row_begin = static_cast<u64>(local_bucket_idx) * mask_rows;
                overlap_counts_int.mShares[0](static_cast<u64>(local_bucket_idx), 0) =
                    overlap_int.mShares[0].block(row_begin, 0, mask_rows, 1).sum();
                overlap_counts_int.mShares[1](static_cast<u64>(local_bucket_idx), 0) =
                    overlap_int.mShares[1].block(row_begin, 0, mask_rows, 1).sum();
            }
            overlap_counts = si64_to_sf64(overlap_counts_int);
        }

        #pragma omp parallel for schedule(static)
        for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
            const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
            const auto bucket_index = bucket_refs[chunk_begin + local_bucket_idx].bucket_index;
            bucket_values[0](static_cast<u64>(local_bucket_idx), 0) =
                model.secret_shared_payload().bucket_values[0](bucket_index, 0);
            bucket_values[1](static_cast<u64>(local_bucket_idx), 0) =
                model.secret_shared_payload().bucket_values[1](bucket_index, 0);
        }

        sf64Matrix<kFlatBSPNDecimal> bucket_contributions(static_cast<u64>(chunk_bucket_count), 1);
        cipher_mul(
            context.role,
            bucket_values,
            overlap_counts,
            bucket_contributions,
            *(context.eval),
            *(context.enc),
            *(context.runtime));

        std::vector<i64> chunk_sum0(leaf_children.size(), 0);
        std::vector<i64> chunk_sum1(leaf_children.size(), 0);
        for (std::size_t local_bucket_idx = 0; local_bucket_idx < chunk_bucket_count; ++local_bucket_idx) {
            const auto child_idx = bucket_refs[chunk_begin + local_bucket_idx].child_idx;
            chunk_sum0[child_idx] = add_i64_mod(
                chunk_sum0[child_idx],
                bucket_contributions[0](static_cast<u64>(local_bucket_idx), 0));
            chunk_sum1[child_idx] = add_i64_mod(
                chunk_sum1[child_idx],
                bucket_contributions[1](static_cast<u64>(local_bucket_idx), 0));
        }
        for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
            if (chunk_sum0[child_idx] == 0 && chunk_sum1[child_idx] == 0) {
                continue;
            }
            numerator_sums[child_idx][0](0, 0) = add_i64_mod(
                numerator_sums[child_idx][0](0, 0),
                chunk_sum0[child_idx]);
            numerator_sums[child_idx][1](0, 0) = add_i64_mod(
                numerator_sums[child_idx][1](0, 0),
                chunk_sum1[child_idx]);
        }
    }
    return numerator_sums;
}

std::vector<sf64Matrix<kFlatBSPNDecimal>> compute_leaf_target_numerator_sums_from_row_values(
    const FlatBSPNModel& model,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const std::vector<std::size_t>& leaf_product_indices,
    const std::vector<sbMatrix>& final_ids_by_product,
    const FlatBSPNSecureContext& context,
    std::uint64_t* phase3_batch_counter = nullptr) {
    if (leaf_children.size() != leaf_product_indices.size()) {
        throw std::runtime_error("Leaf child and product index counts do not match for row-value numerator.");
    }
    std::vector<sf64Matrix<kFlatBSPNDecimal>> numerator_sums;
    numerator_sums.reserve(leaf_children.size());
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        (void)child_idx;
        numerator_sums.push_back(share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context));
    }
    if (leaf_children.empty()) {
        return numerator_sums;
    }
    const auto& payload = model.secret_shared_payload();
    if (!payload.leaf_row_values_loaded || payload.leaf_row_values.rows() == 0) {
        throw std::runtime_error("row-value numerator requested without loaded leaf row values.");
    }
    const u64 total_rows = static_cast<u64>(model.manifest().total_rows);
    if (total_rows == 0) {
        return numerator_sums;
    }
    const auto& shape_mask = final_ids_by_product[leaf_product_indices.front()];
    const u64 bit_count = shape_mask.bitCount();
    const u64 share_cols = shape_mask.mShares[0].cols();
    const u64 leaf_count = static_cast<u64>(leaf_children.size());
    const u64 stacked_rows = leaf_count * total_rows;
    sbMatrix stacked_final_ids(stacked_rows, bit_count);
    sf64Matrix<kFlatBSPNDecimal> stacked_values(stacked_rows, 1);
    const bool use_row_weights = payload.row_weights_loaded;
    if (use_row_weights &&
        (payload.row_weights.rows() != total_rows || payload.row_weights.cols() != 1)) {
        throw std::runtime_error("Row weight shape does not match leaf row values.");
    }

    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto product_idx = leaf_product_indices[child_idx];
        if (product_idx >= final_ids_by_product.size()) {
            throw std::runtime_error("Leaf product index out of bounds in row-value numerator.");
        }
        const auto& final_ids = final_ids_by_product[product_idx];
        if (final_ids.rows() != total_rows || final_ids.bitCount() != bit_count) {
            throw std::runtime_error("final_ids shape mismatch in row-value numerator.");
        }
        const auto node_id = leaf_children[child_idx]->node_id;
        if (node_id >= payload.leaf_row_value_offset_by_node.size() ||
            payload.leaf_row_value_offset_by_node[node_id] < 0) {
            throw std::runtime_error("Missing row-value payload for leaf node.");
        }
    }

    #pragma omp parallel for schedule(static)
    for (std::int64_t child_idx_signed = 0; child_idx_signed < static_cast<std::int64_t>(leaf_children.size()); ++child_idx_signed) {
        const auto child_idx = static_cast<std::size_t>(child_idx_signed);
        const auto product_idx = leaf_product_indices[child_idx];
        const auto& final_ids = final_ids_by_product[product_idx];
        const auto node_id = leaf_children[child_idx]->node_id;
        const u64 value_offset = static_cast<u64>(payload.leaf_row_value_offset_by_node[node_id]);
        const u64 row_begin = static_cast<u64>(child_idx) * total_rows;
        for (u64 row = 0; row < total_rows; ++row) {
            for (u64 col = 0; col < share_cols; ++col) {
                stacked_final_ids.mShares[0](row_begin + row, col) = final_ids.mShares[0](row, col);
                stacked_final_ids.mShares[1](row_begin + row, col) = final_ids.mShares[1](row, col);
            }
            stacked_values[0](row_begin + row, 0) = payload.leaf_row_values[0](value_offset + row, 0);
            stacked_values[1](row_begin + row, 0) = payload.leaf_row_values[1](value_offset + row, 0);
        }
    }

    if (use_row_weights) {
        sf64Matrix<kFlatBSPNDecimal> stacked_weights(stacked_rows, 1);
        #pragma omp parallel for schedule(static)
        for (std::int64_t child_idx_signed = 0; child_idx_signed < static_cast<std::int64_t>(leaf_children.size()); ++child_idx_signed) {
            const auto child_idx = static_cast<std::size_t>(child_idx_signed);
            const u64 row_begin = static_cast<u64>(child_idx) * total_rows;
            for (u64 row = 0; row < total_rows; ++row) {
                stacked_weights[0](row_begin + row, 0) = payload.row_weights[0](row, 0);
                stacked_weights[1](row_begin + row, 0) = payload.row_weights[1](row, 0);
            }
        }
        stacked_values = secure_mul_fixed_same_shape(stacked_values, stacked_weights, context);
    }

    auto contributions = fixed_mul_bool_same_shape(stacked_values, stacked_final_ids, context);
    if (phase3_batch_counter != nullptr) {
        ++(*phase3_batch_counter);
    }

    #pragma omp parallel for schedule(static)
    for (std::int64_t child_idx_signed = 0; child_idx_signed < static_cast<std::int64_t>(leaf_children.size()); ++child_idx_signed) {
        const auto child_idx = static_cast<std::size_t>(child_idx_signed);
        const u64 row_begin = static_cast<u64>(child_idx) * total_rows;
        numerator_sums[child_idx][0](0, 0) =
            contributions[0].block(row_begin, 0, total_rows, 1).sum();
        numerator_sums[child_idx][1](0, 0) =
            contributions[1].block(row_begin, 0, total_rows, 1).sum();
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

    struct BucketRef {
        std::size_t child_idx = 0;
        std::uint32_t bucket_offset = 0;
        std::uint32_t bucket_index = 0;
    };
    const sbMatrix* shape_bitmap = nullptr;
    std::vector<BucketRef> bucket_refs;
    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        if (match_masks[child_idx].rows() != child.bucket_count) {
            throw std::runtime_error("Match mask row count does not match leaf bucket count.");
        }
        for (std::uint32_t bucket_offset = 0; bucket_offset < child.bucket_count; ++bucket_offset) {
            bucket_refs.push_back({child_idx, bucket_offset, child.bucket_begin + bucket_offset});
        }
        if (shape_bitmap == nullptr && child.bucket_count != 0) {
            shape_bitmap = &model.secret_shared_payload().dense_bucket_bitmaps[child.bucket_begin];
        }
    }

    if (shape_bitmap == nullptr || bucket_refs.empty()) {
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

    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        sbMatrix child_local_ids(block_len, bit_count);
        bool_init_false(context.role, child_local_ids);
        local_ids.push_back(std::move(child_local_ids));
    }

    const u64 kMaxStackedBitmapRows = bspn_max_stacked_bitmap_rows();
    const std::size_t buckets_per_chunk = static_cast<std::size_t>(
        std::max<u64>(1, kMaxStackedBitmapRows / std::max<u64>(1, block_len)));

    for (std::size_t chunk_begin = 0; chunk_begin < bucket_refs.size(); chunk_begin += buckets_per_chunk) {
        const std::size_t chunk_end = std::min<std::size_t>(bucket_refs.size(), chunk_begin + buckets_per_chunk);
        const std::size_t chunk_bucket_count = chunk_end - chunk_begin;
        const u64 stacked_rows = static_cast<u64>(chunk_bucket_count) * block_len;
        sbMatrix stacked_bitmaps(stacked_rows, bit_count);
        sbMatrix stacked_match(stacked_rows, bit_count);

        for (std::size_t ref_idx = 0; ref_idx < chunk_bucket_count; ++ref_idx) {
            const auto& bucket_ref = bucket_refs[chunk_begin + ref_idx];
            const auto& bucket_bitmap = model.secret_shared_payload().dense_bucket_bitmaps[bucket_ref.bucket_index];
            const auto& match_mask = match_masks[bucket_ref.child_idx];
            if (bucket_bitmap.rows() != block_len || bucket_bitmap.bitCount() != bit_count) {
                throw std::runtime_error("Bucket bitmap shape mismatch in batched local id computation.");
            }
            if (match_mask.bitCount() != 1 && match_mask.bitCount() != bit_count) {
                throw std::runtime_error("Match mask bit width mismatch in batched local id computation.");
            }
        }

        #pragma omp parallel for schedule(static)
        for (std::int64_t ref_idx_signed = 0; ref_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++ref_idx_signed) {
            const auto ref_idx = static_cast<std::size_t>(ref_idx_signed);
            const auto& bucket_ref = bucket_refs[chunk_begin + ref_idx];
            const auto& bucket_bitmap = model.secret_shared_payload().dense_bucket_bitmaps[bucket_ref.bucket_index];
            const auto& match_mask = match_masks[bucket_ref.child_idx];
            const bool expand_one_bit_mask = match_mask.bitCount() == 1;
            const u64 row_begin = static_cast<u64>(ref_idx) * block_len;
            i64 match_share0 = 0;
            i64 match_share1 = 0;
            if (expand_one_bit_mask) {
                match_share0 = (match_mask.mShares[0](bucket_ref.bucket_offset, 0) & 1) ? -1 : 0;
                match_share1 = (match_mask.mShares[1](bucket_ref.bucket_offset, 0) & 1) ? -1 : 0;
            }
            for (u64 row = 0; row < block_len; ++row) {
                for (u64 col = 0; col < share_cols; ++col) {
                    stacked_bitmaps.mShares[0](row_begin + row, col) = bucket_bitmap.mShares[0](row, col);
                    stacked_bitmaps.mShares[1](row_begin + row, col) = bucket_bitmap.mShares[1](row, col);
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

        for (std::size_t ref_idx = 0; ref_idx < chunk_bucket_count; ++ref_idx) {
            const auto& bucket_ref = bucket_refs[chunk_begin + ref_idx];
            auto& child_local_ids = local_ids[bucket_ref.child_idx];
            const u64 row_begin = static_cast<u64>(ref_idx) * block_len;
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
    bound.factor.weighted_count_direct = factor_doc.value("weighted_count_direct", false);
    bound.factor.requires_model_eval = factor_doc.value("requires_model_eval", false);

    if (bound.factor.factor_kind == "CONSTANT") {
        return bound;
    }
    if ((bound.factor.factor_kind == "INDICATOR_EXPECTATION" ||
         bound.factor.factor_kind == "EXPECTATION") &&
        bound.factor.public_feature_count == 0 &&
        bound.factor.public_evidence_count == 0 &&
        !bound.factor.requires_model_eval) {
        bound.model_id = factor_doc.value("spn_model_id", std::string());
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
    if (bound.factor.total_rows != model.manifest().total_rows) {
        throw std::runtime_error(
            "Secure bundle factor total_rows does not match model sample_total_rows for model " +
            bound.model_id);
    }

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

bool factor_array_has_large_public_scale(const json& factors_doc) {
    double max_public_constant = 0.0;
    double max_eval_rows = 0.0;
    for (const auto& factor_doc : factors_doc) {
        const std::string factor_kind = factor_doc.value("factor_kind", std::string());
        if (factor_kind == "CONSTANT") {
            max_public_constant = std::max(
                max_public_constant,
                std::abs(factor_doc.value("public_constant_value", 0.0)));
        } else {
            max_eval_rows = std::max(
                max_eval_rows,
                static_cast<double>(factor_doc.value("total_rows", std::uint64_t(0))));
        }
    }
    return max_eval_rows > 0.0 && max_public_constant > max_eval_rows * 16.0;
}

bool aggregate_term_has_large_public_scale(const json& term_doc) {
    if (term_doc.contains("expectation_plan") &&
        factor_array_has_large_public_scale(term_doc["expectation_plan"].value("factors", json::array()))) {
        return true;
    }
    if (term_doc.contains("numerator_plan") &&
        factor_array_has_large_public_scale(term_doc["numerator_plan"].value("factors", json::array()))) {
        return true;
    }
    if (term_doc.contains("denominator_plan") &&
        factor_array_has_large_public_scale(term_doc["denominator_plan"].value("factors", json::array()))) {
        return true;
    }
    return false;
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
        constexpr double kWeightedSumTermUpscale = 256.0;
        double numerator_scale = 1.0;
        std::vector<sf64Matrix<kFlatBSPNDecimal>> terms(values.size());
        std::vector<double> term_scales(values.size(), 1.0);
        sbMatrix all_zero = shared_true_bool_scalar(context);
        for (std::size_t idx = 0; idx < values.size(); ++idx) {
            auto scaled_numerator = secure_mul_public_fixed(
                values[idx].numerator,
                kWeightedSumTermUpscale,
                context);
            terms[idx] = secure_mul_fixed(scaled_numerator, weights[idx], context);
            const auto value_zero = rational_zero_numerator_flag(values[idx], context);
            terms[idx] = fixed_mul_bool_same_shape(
                terms[idx],
                bool_not_scalar(value_zero, context),
                context);
            all_zero = bool_and_scalar(all_zero, value_zero, context);
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
        numerator_sum = secure_mul_public_fixed(
            numerator_sum,
            1.0 / kWeightedSumTermUpscale,
            context);
        auto out = normalize_secure_rational_scales({
            numerator_sum,
            share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context),
            numerator_scale,
            1.0,
            true,
        });
        out.has_secret_zero_numerator = true;
        out.secret_zero_numerator = all_zero;
        return out;
	    }
	    constexpr double kWeightedSumStorageUnit = 1.0;
	    sbMatrix any_non_unit = shared_zero_bool_scalar(context);
	    sbMatrix all_zero = shared_true_bool_scalar(context);
	    double scalar_numerator_scale = 1.0;
	    std::vector<sf64Matrix<kFlatBSPNDecimal>> scalar_terms(values.size());
	    std::vector<double> scalar_term_scales(values.size(), 1.0);
	    for (std::size_t idx = 0; idx < values.size(); ++idx) {
	        any_non_unit = bool_or_scalar(
	            any_non_unit,
	            rational_non_unit_denominator_flag(values[idx], context),
	            context);
	        scalar_terms[idx] = secure_mul_fixed(values[idx].numerator, weights[idx], context);
	        const auto value_zero = rational_zero_numerator_flag(values[idx], context);
	        scalar_terms[idx] = fixed_mul_bool_same_shape(
	            scalar_terms[idx],
	            bool_not_scalar(value_zero, context),
	            context);
	        all_zero = bool_and_scalar(all_zero, value_zero, context);
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
	        const auto value_zero = rational_zero_numerator_flag(values[idx], context);
	        term.numerator = fixed_mul_bool_same_shape(
	            term.numerator,
	            bool_not_scalar(value_zero, context),
	            context);
	        term.has_secret_zero_numerator = true;
	        term.secret_zero_numerator = value_zero;
	        total = add_secure_rational(total, term, context);
	        total = scale_secure_rational_public(total, kWeightedSumStorageUnit, context);
	    }
	    total.has_secret_zero_numerator = true;
	    total.secret_zero_numerator = all_zero;
	    auto scalar_total_with_zero = scalar_total;
	    scalar_total_with_zero.has_secret_zero_numerator = true;
	    scalar_total_with_zero.secret_zero_numerator = all_zero;
	    return select_rational_by_bool(total, scalar_total_with_zero, any_non_unit, context);
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
    const FlatBSPNSecureContext& context);

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
    return secure_interval_match_mask_rows(
        bucket_lowers,
        bucket_uppers,
        repeat_fixed_scalar_rows(interval_lower, rows),
        repeat_fixed_scalar_rows(interval_upper, rows),
        repeat_bool_scalar_rows(has_lower_flag, rows),
        repeat_bool_scalar_rows(has_upper_flag, rows),
        repeat_bool_scalar_rows(open_lower_flag, rows),
        repeat_bool_scalar_rows(open_upper_flag, rows),
        context);
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
    const auto epsilon_rows = repeat_fixed_scalar_rows(
        share_fixed_scalar<kFlatBSPNDecimal>(1.0 / 1024.0, 0, context),
        rows);

    sbMatrix gt_or_eq_lower;
    sbMatrix ge_lower;
    interval_lowers_copy -= epsilon_rows;
    cipher_gt(context.role, bucket_lowers_copy, interval_lowers_copy, gt_or_eq_lower, *(context.eval), *(context.runtime));
    bucket_lowers_copy = bucket_lowers;
    interval_lowers_copy = interval_lowers;
    sbMatrix eq_lower;
    cipher_eq(context.role, bucket_lowers_copy, interval_lowers_copy, eq_lower, *(context.eval), *(context.runtime));
    bool_cipher_or(context.role, gt_or_eq_lower, eq_lower, ge_lower, *(context.enc), *(context.eval), *(context.runtime));
    bucket_lowers_copy = bucket_lowers;
    interval_lowers_copy = interval_lowers;
    sbMatrix gt_lower;
    interval_lowers_copy += epsilon_rows;
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

    sbMatrix gt_or_eq_upper;
    sbMatrix ge_upper;
    interval_uppers_copy += epsilon_rows;
    cipher_gt(context.role, interval_uppers_copy, bucket_uppers_copy, gt_or_eq_upper, *(context.eval), *(context.runtime));
    bucket_uppers_copy = bucket_uppers;
    interval_uppers_copy = interval_uppers;
    sbMatrix eq_upper;
    cipher_eq(context.role, interval_uppers_copy, bucket_uppers_copy, eq_upper, *(context.eval), *(context.runtime));
    bool_cipher_or(context.role, gt_or_eq_upper, eq_upper, ge_upper, *(context.enc), *(context.eval), *(context.runtime));
    bucket_uppers_copy = bucket_uppers;
    interval_uppers_copy = interval_uppers;
    sbMatrix gt_upper;
    interval_uppers_copy -= epsilon_rows;
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

std::vector<sbMatrix> compute_leaf_match_masks_for_evidence(
    const FlatBSPNModel& model,
    const FlatSecureQueryTensorPayload& shared_query_payload,
    std::uint32_t secret_factor_row,
    std::size_t max_columns,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const FlatBSPNSecureContext& context,
    SecureIndicatorEvalStats* eval_stats,
    sbMatrix* evidence_domain_miss_out = nullptr) {
    std::vector<sbMatrix> match_masks;
    match_masks.reserve(leaf_children.size());
    for (const auto* child : leaf_children) {
        sbMatrix match_mask(child->bucket_count, 1);
        bool_init_false(context.role, match_mask);
        match_masks.push_back(std::move(match_mask));
    }

    const u64 leaf_rows = static_cast<u64>(leaf_children.size());
    const u64 evidence_cols = std::min<u64>(
        static_cast<u64>(max_columns),
        static_cast<u64>(model.manifest().column_names.size()));
    const std::size_t max_interval_count =
        static_cast<std::size_t>(shared_query_payload.lower_bounds_shared.cols());
    if (leaf_rows == 0 || evidence_cols == 0) {
        return match_masks;
    }

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
    }
    #pragma omp parallel for schedule(static)
    for (std::int64_t row_signed = 0; row_signed < static_cast<std::int64_t>(leaf_rows); ++row_signed) {
        const auto row = static_cast<u64>(row_signed);
        const auto node_id = leaf_children[static_cast<std::size_t>(row)]->node_id;
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
        #pragma omp parallel for schedule(static)
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
        #pragma omp parallel for schedule(static)
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
        #pragma omp parallel for schedule(static)
        for (std::int64_t row_signed = 0; row_signed < static_cast<std::int64_t>(leaf_rows); ++row_signed) {
            const auto row = static_cast<u64>(row_signed);
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
        #pragma omp parallel for schedule(static)
        for (std::int64_t row_signed = 0; row_signed < static_cast<std::int64_t>(leaf_rows); ++row_signed) {
            const auto row = static_cast<u64>(row_signed);
            row_totals[0](row, 0) = products[0].block(row, 0, 1, evidence_cols).sum();
            row_totals[1](row, 0) = products[1].block(row, 0, 1, evidence_cols).sum();
        }
        return row_totals;
    };

    const auto selected_interval_counts = scoped_int_values(shared_query_payload.interval_counts_shared, 0);
    const auto selected_has_evidence_flags = scoped_int_flags(shared_query_payload.has_evidence_shared, 0);
    const bool debug_leaf_evidence_flags =
        context.debug_internal_reveal && std::getenv("BSPN_DEBUG_LEAF_EVIDENCE_FLAGS") != nullptr;
    if (debug_leaf_evidence_flags) {
        i64Matrix interval_counts_plain(
            selected_interval_counts.rows(),
            selected_interval_counts.cols());
        context.enc->revealAll(
            context.runtime->noDependencies(),
            selected_interval_counts,
            interval_counts_plain).get();
        auto has_evidence_flags_copy = selected_has_evidence_flags;
        si64Matrix has_evidence_int(selected_has_evidence_flags.rows(), 1);
        bool2arith(
            context.role,
            has_evidence_flags_copy,
            has_evidence_int,
            *(context.enc),
            *(context.eval),
            *(context.runtime));
        i64Matrix has_evidence_plain(
            has_evidence_int.rows(),
            has_evidence_int.cols());
        context.enc->revealAll(
            context.runtime->noDependencies(),
            has_evidence_int,
            has_evidence_plain).get();
        if (context.role == 0) {
            std::cerr << "bspn_debug_leaf_evidence_flags:"
                      << " factor_row=" << secret_factor_row
                      << " leaf_rows=" << leaf_rows
                      << " evidence_cols=" << evidence_cols
                      << " interval_counts=[";
            for (u64 row = 0; row < static_cast<u64>(interval_counts_plain.rows()); ++row) {
                if (row != 0) {
                    std::cerr << ",";
                }
                std::cerr << interval_counts_plain(row, 0);
            }
            std::cerr << "] has_evidence=[";
            for (u64 row = 0; row < static_cast<u64>(has_evidence_plain.rows()); ++row) {
                if (row != 0) {
                    std::cerr << ",";
                }
                std::cerr << has_evidence_plain(row, 0);
            }
            std::cerr << "]\n";
        }
    }
    const u64 stacked_bucket_rows = std::accumulate(
        leaf_children.begin(),
        leaf_children.end(),
        u64(0),
        [](u64 total, const FlatBSPNNodeRecord* child) {
            return total + static_cast<u64>(child->bucket_count);
        });
    if (stacked_bucket_rows != 0 && max_interval_count != 0) {
        std::vector<u64> leaf_bucket_offsets(leaf_children.size() + 1, 0);
        for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
            leaf_bucket_offsets[child_idx + 1] =
                leaf_bucket_offsets[child_idx] + static_cast<u64>(leaf_children[child_idx]->bucket_count);
        }
        sf64Matrix<kFlatBSPNDecimal> stacked_bucket_lowers(stacked_bucket_rows, 1);
        sf64Matrix<kFlatBSPNDecimal> stacked_bucket_uppers(stacked_bucket_rows, 1);
        #pragma omp parallel for schedule(static)
        for (std::int64_t child_idx_signed = 0; child_idx_signed < static_cast<std::int64_t>(leaf_children.size()); ++child_idx_signed) {
            const auto child_idx = static_cast<std::size_t>(child_idx_signed);
            const auto& child = *leaf_children[child_idx];
            const auto lowers = fixed_row_slice(
                model.secret_shared_payload().bucket_lowers,
                child.bucket_begin,
                child.bucket_count);
            const auto uppers = fixed_row_slice(
                model.secret_shared_payload().bucket_uppers,
                child.bucket_begin,
                child.bucket_count);
            const u64 row_cursor = leaf_bucket_offsets[child_idx];
            stacked_bucket_lowers[0].block(row_cursor, 0, child.bucket_count, 1) = lowers[0];
            stacked_bucket_lowers[1].block(row_cursor, 0, child.bucket_count, 1) = lowers[1];
            stacked_bucket_uppers[0].block(row_cursor, 0, child.bucket_count, 1) = uppers[0];
            stacked_bucket_uppers[1].block(row_cursor, 0, child.bucket_count, 1) = uppers[1];
        }

        sbMatrix stacked_match(stacked_bucket_rows, 1);
        bool_init_false(context.role, stacked_match);
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
            const auto has_lower_flags = scoped_int_flags(shared_query_payload.has_lower_shared, static_cast<std::uint32_t>(interval_idx));
            const auto has_upper_flags = scoped_int_flags(shared_query_payload.has_upper_shared, static_cast<std::uint32_t>(interval_idx));
            const auto open_lower_flags = scoped_int_flags(shared_query_payload.open_lower_shared, static_cast<std::uint32_t>(interval_idx));
            const auto open_upper_flags = scoped_int_flags(shared_query_payload.open_upper_shared, static_cast<std::uint32_t>(interval_idx));

            sf64Matrix<kFlatBSPNDecimal> stacked_lowers(stacked_bucket_rows, 1);
            sf64Matrix<kFlatBSPNDecimal> stacked_uppers(stacked_bucket_rows, 1);
            sbMatrix stacked_has_lower(stacked_bucket_rows, 1);
            sbMatrix stacked_has_upper(stacked_bucket_rows, 1);
            sbMatrix stacked_open_lower(stacked_bucket_rows, 1);
            sbMatrix stacked_open_upper(stacked_bucket_rows, 1);
            sbMatrix stacked_active(stacked_bucket_rows, 1);
            #pragma omp parallel for schedule(static)
            for (std::int64_t child_idx_signed = 0; child_idx_signed < static_cast<std::int64_t>(leaf_children.size()); ++child_idx_signed) {
                const auto child_idx = static_cast<std::size_t>(child_idx_signed);
                const auto& child = *leaf_children[child_idx];
                const u64 row_cursor = leaf_bucket_offsets[child_idx];
                for (u64 bucket_row = 0; bucket_row < static_cast<u64>(child.bucket_count); ++bucket_row) {
                    const u64 dst_row = row_cursor + bucket_row;
                    stacked_lowers[0](dst_row, 0) = lower_values[0](static_cast<u64>(child_idx), 0);
                    stacked_lowers[1](dst_row, 0) = lower_values[1](static_cast<u64>(child_idx), 0);
                    stacked_uppers[0](dst_row, 0) = upper_values[0](static_cast<u64>(child_idx), 0);
                    stacked_uppers[1](dst_row, 0) = upper_values[1](static_cast<u64>(child_idx), 0);
                    for (u64 col = 0; col < static_cast<u64>(stacked_has_lower.mShares[0].cols()); ++col) {
                        stacked_has_lower.mShares[0](dst_row, col) = has_lower_flags.mShares[0](static_cast<u64>(child_idx), col);
                        stacked_has_lower.mShares[1](dst_row, col) = has_lower_flags.mShares[1](static_cast<u64>(child_idx), col);
                        stacked_has_upper.mShares[0](dst_row, col) = has_upper_flags.mShares[0](static_cast<u64>(child_idx), col);
                        stacked_has_upper.mShares[1](dst_row, col) = has_upper_flags.mShares[1](static_cast<u64>(child_idx), col);
                        stacked_open_lower.mShares[0](dst_row, col) = open_lower_flags.mShares[0](static_cast<u64>(child_idx), col);
                        stacked_open_lower.mShares[1](dst_row, col) = open_lower_flags.mShares[1](static_cast<u64>(child_idx), col);
                        stacked_open_upper.mShares[0](dst_row, col) = open_upper_flags.mShares[0](static_cast<u64>(child_idx), col);
                        stacked_open_upper.mShares[1](dst_row, col) = open_upper_flags.mShares[1](static_cast<u64>(child_idx), col);
                        stacked_active.mShares[0](dst_row, col) = interval_active_flags.mShares[0](static_cast<u64>(child_idx), col);
                        stacked_active.mShares[1](dst_row, col) = interval_active_flags.mShares[1](static_cast<u64>(child_idx), col);
                    }
                }
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

        #pragma omp parallel for schedule(static)
        for (std::int64_t child_idx_signed = 0; child_idx_signed < static_cast<std::int64_t>(leaf_children.size()); ++child_idx_signed) {
            const auto child_idx = static_cast<std::size_t>(child_idx_signed);
            const auto& child = *leaf_children[child_idx];
            const u64 row_cursor = leaf_bucket_offsets[child_idx];
            for (u64 row = 0; row < static_cast<u64>(child.bucket_count); ++row) {
                for (u64 col = 0; col < static_cast<u64>(stacked_match.mShares[0].cols()); ++col) {
                    match_masks[child_idx].mShares[0](row, col) = stacked_match.mShares[0](row_cursor + row, col);
                    match_masks[child_idx].mShares[1](row, col) = stacked_match.mShares[1](row_cursor + row, col);
                }
            }
        }
    }

    if (evidence_domain_miss_out != nullptr) {
        auto domain_miss = shared_zero_bool_scalar(context);
        std::vector<sbMatrix> leaf_has_match;
        leaf_has_match.reserve(match_masks.size());
        for (const auto& match_mask : match_masks) {
            auto match_count = sum_boolean_mask_to_int(match_mask, context);
            auto zero_count = shared_zero_int_matrix(match_count.rows(), match_count.cols());
            sbMatrix has_match;
            cipher_gt(
                context.role,
                match_count,
                zero_count,
                has_match,
                *(context.eval),
                *(context.runtime));
            leaf_has_match.push_back(std::move(has_match));
        }
        const si64Matrix factor_evidence_scope =
            shared_query_factor_evidence_scope(shared_query_payload, secret_factor_row);
        for (u64 col = 0; col < evidence_cols; ++col) {
            sbMatrix column_has_match = shared_zero_bool_scalar(context);
            for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
                if (leaf_children[child_idx]->leaf_column_id != static_cast<std::int32_t>(col)) {
                    continue;
                }
                column_has_match = bool_or_scalar(
                    column_has_match,
                    leaf_has_match[child_idx],
                    context);
            }
            const auto column_has_evidence = shared_int_nonzero_flag(
                int_cell(factor_evidence_scope, 0, static_cast<std::uint32_t>(col)),
                context);
            const auto column_miss = bool_and_scalar(
                column_has_evidence,
                bool_not_scalar(column_has_match, context),
                context);
            domain_miss = bool_or_scalar(domain_miss, column_miss, context);
        }
        *evidence_domain_miss_out = std::move(domain_miss);
    }

    for (std::size_t child_idx = 0; child_idx < leaf_children.size(); ++child_idx) {
        const auto& child = *leaf_children[child_idx];
        const auto leaf_has_evidence = bool_row_slice(selected_has_evidence_flags, static_cast<u64>(child_idx));
        sbMatrix no_evidence(leaf_has_evidence.rows(), leaf_has_evidence.bitCount());
        auto leaf_has_evidence_copy = leaf_has_evidence;
        bool_cipher_not(context.role, leaf_has_evidence_copy, no_evidence);
        auto no_evidence_rows = repeat_bool_scalar_rows(no_evidence, child.bucket_count);
        sbMatrix updated_match_mask(child.bucket_count, match_masks[child_idx].bitCount());
        bool_cipher_or(context.role, match_masks[child_idx], no_evidence_rows, updated_match_mask, *(context.enc), *(context.eval), *(context.runtime));
        match_masks[child_idx] = std::move(updated_match_mask);
    }
    if (debug_leaf_evidence_flags) {
        std::vector<sbMatrix> match_masks_copy = match_masks;
        const auto match_counts = sum_boolean_masks_to_int_batched(
            match_masks_copy,
            context,
            nullptr);
        i64Matrix match_counts_plain(match_counts.rows(), match_counts.cols());
        context.enc->revealAll(
            context.runtime->noDependencies(),
            match_counts,
            match_counts_plain).get();
        if (context.role == 0) {
            std::cerr << "bspn_debug_leaf_match_counts:"
                      << " factor_row=" << secret_factor_row
                      << " counts=[";
            for (u64 row = 0; row < static_cast<u64>(match_counts_plain.rows()); ++row) {
                if (row != 0) {
                    std::cerr << ",";
                }
                std::cerr << match_counts_plain(row, 0);
            }
            std::cerr << "]\n";
        }
    }

    return match_masks;
}

std::vector<SecureRationalShare> evaluate_leaf_product_batch_values(
    const FlatBSPNModel& model,
    const SecureBoundFactor& factor,
    const FlatSecureQueryTensorPayload& shared_query_payload,
    std::uint32_t secret_factor_row,
    const si64Matrix& secret_feature_scope,
    const std::vector<const FlatBSPNNodeRecord*>& leaf_children,
    const std::vector<std::uint32_t>& leaf_node_ids,
    const std::vector<LeafProductBatchItem>& product_items,
    const std::vector<std::size_t>& leaf_product_indices,
    const sbMatrix& global_rows_shared,
    const FlatBSPNSecureContext& context,
    SecureIndicatorEvalStats* eval_stats,
    sbMatrix* evidence_domain_miss_out = nullptr) {
    const std::size_t public_factor_feature_count =
        static_cast<std::size_t>(factor.factor.public_feature_count);
    const std::size_t public_factor_evidence_count =
        static_cast<std::size_t>(factor.factor.public_evidence_count);
    const bool needs_evidence_filter = public_factor_evidence_count != 0;
    const bool needs_target_numerator = public_factor_feature_count != 0;
    const bool needs_leaf_domain_filter =
        needs_evidence_filter ||
        (factor.factor.requires_model_eval && !needs_target_numerator);
    const bool public_single_target_factor = public_factor_feature_count <= 1;
    const std::size_t max_columns = static_cast<std::size_t>(secret_feature_scope.cols());

    std::vector<sbMatrix> target_flags;
    std::vector<sbMatrix> has_target_by_product;
    if (needs_target_numerator) {
        target_flags = secure_scope_intersects_shared_rows(
            model,
            leaf_node_ids,
            secret_feature_scope,
            context);
        has_target_by_product.reserve(product_items.size());
        for (const auto& item : product_items) {
            sbMatrix has_target = shared_zero_bool_scalar(context);
            for (std::size_t offset = 0; offset < item.leaf_count; ++offset) {
                has_target = bool_or_scalar(
                    has_target,
                    target_flags[item.leaf_begin + offset],
                    context);
            }
            has_target_by_product.push_back(std::move(has_target));
        }
    }

    std::vector<sbMatrix> final_ids_by_product;
    final_ids_by_product.reserve(product_items.size());
    si64Matrix final_cnt_int_rows(static_cast<u64>(product_items.size()), 1);
    sf64Matrix<kFlatBSPNDecimal> weighted_final_cnt_rows(static_cast<u64>(product_items.size()), 1);
    final_cnt_int_rows.mShares[0].setZero();
    final_cnt_int_rows.mShares[1].setZero();
    weighted_final_cnt_rows[0].setZero();
    weighted_final_cnt_rows[1].setZero();
    const bool use_row_weights_for_rows = model.secret_shared_payload().row_weights_loaded;
    if (needs_leaf_domain_filter) {
        auto phase_start = SteadyClock::now();
        std::vector<sbMatrix> match_masks;
        if (needs_evidence_filter) {
            match_masks = compute_leaf_match_masks_for_evidence(
                model,
                shared_query_payload,
                secret_factor_row,
                max_columns,
                leaf_children,
                context,
                eval_stats,
                evidence_domain_miss_out);
            if (eval_stats != nullptr) {
                eval_stats->phase1_match_ms += elapsed_ms_since(phase_start);
            }
        } else {
            match_masks.reserve(leaf_children.size());
            const auto all_buckets_match = shared_true_bool_scalar(context);
            for (const auto* child : leaf_children) {
                match_masks.push_back(repeat_bool_scalar_rows(all_buckets_match, child->bucket_count));
            }
        }

        bool can_use_bucket_weight_sums =
            use_row_weights_for_rows &&
            !needs_target_numerator &&
            factor.factor.weighted_count_direct &&
            model.secret_shared_payload().leaf_bucket_weight_sums_loaded;
        if (can_use_bucket_weight_sums) {
            for (const auto& item : product_items) {
                if (item.leaf_count != 1) {
                    can_use_bucket_weight_sums = false;
                    break;
                }
            }
        }

        if (can_use_bucket_weight_sums) {
            phase_start = SteadyClock::now();
            weighted_final_cnt_rows = sum_single_leaf_match_bucket_weights_batched(
                model,
                leaf_children,
                product_items,
                match_masks,
                context,
                eval_stats != nullptr ? &eval_stats->phase2_count_batches : nullptr);
            if (eval_stats != nullptr) {
                eval_stats->phase2_count_ms += elapsed_ms_since(phase_start);
            }
        } else {
            phase_start = SteadyClock::now();
            auto local_ids = compute_leaf_local_ids_batched(
                model,
                leaf_children,
                match_masks,
                context,
                eval_stats != nullptr ? &eval_stats->phase1_batch_dot_calls : nullptr);
            if (eval_stats != nullptr) {
                eval_stats->phase1_local_ids_ms += elapsed_ms_since(phase_start);
            }

            phase_start = SteadyClock::now();
            for (const auto& item : product_items) {
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
            final_cnt_int_rows = sum_boolean_masks_to_int_batched(
                final_ids_by_product,
                context,
                eval_stats != nullptr ? &eval_stats->phase2_count_batches : nullptr);
            if (use_row_weights_for_rows) {
                weighted_final_cnt_rows = sum_boolean_masks_weighted_to_fixed_batched(
                    final_ids_by_product,
                    model.secret_shared_payload().row_weights,
                    context,
                    eval_stats != nullptr ? &eval_stats->phase2_count_batches : nullptr);
            }
            if (eval_stats != nullptr) {
                eval_stats->phase2_count_ms += elapsed_ms_since(phase_start);
            }
        }
    } else {
        for (std::size_t product_idx = 0; product_idx < product_items.size(); ++product_idx) {
            (void)product_idx;
            final_ids_by_product.push_back(global_rows_shared);
        }
        if (use_row_weights_for_rows) {
            const auto phase_start = SteadyClock::now();
            weighted_final_cnt_rows = sum_boolean_masks_weighted_to_fixed_batched(
                final_ids_by_product,
                model.secret_shared_payload().row_weights,
                context,
                eval_stats != nullptr ? &eval_stats->phase2_count_batches : nullptr);
            if (eval_stats != nullptr) {
                eval_stats->phase2_count_ms += elapsed_ms_since(phase_start);
            }
        }
    }

    std::vector<sf64Matrix<kFlatBSPNDecimal>> target_numerator_sums;
    if (needs_target_numerator) {
        const auto phase_start = SteadyClock::now();
        const bool use_row_values =
            !use_row_weights_for_rows &&
            bspn_use_row_value_eval() &&
            model.secret_shared_payload().leaf_row_values_loaded;
        if (use_row_values) {
            target_numerator_sums = compute_leaf_target_numerator_sums_from_row_values(
                model,
                leaf_children,
                leaf_product_indices,
                final_ids_by_product,
                context,
                eval_stats != nullptr ? &eval_stats->phase3_batch_b2a_calls : nullptr);
            if (eval_stats != nullptr) {
                ++eval_stats->row_value_eval_used;
            }
        } else {
            target_numerator_sums = compute_leaf_target_numerator_sums_group_batched(
                model,
                leaf_children,
                leaf_product_indices,
                final_ids_by_product,
                context,
                eval_stats != nullptr ? &eval_stats->phase3_batch_b2a_calls : nullptr);
        }
        if (eval_stats != nullptr) {
            eval_stats->phase3_numerator_ms += elapsed_ms_since(phase_start);
        }
    }

    std::vector<SecureRationalShare> out;
    out.reserve(product_items.size());
    const auto phase_start = SteadyClock::now();
    if (product_items.empty()) {
        if (eval_stats != nullptr) {
            eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
        }
        return out;
    }
    auto zero_fixed = share_fixed_scalar<kFlatBSPNDecimal>(0.0, 0, context);
    auto one_fixed = share_fixed_scalar<kFlatBSPNDecimal>(1.0, 0, context);
    auto total_rows_int = share_int_scalar(
        static_cast<i64>(factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows),
        0,
        context);
    const u64 product_count = static_cast<u64>(product_items.size());
    std::vector<std::uint32_t> product_node_rows(product_items.size());
    for (std::size_t product_idx = 0; product_idx < product_items.size(); ++product_idx) {
        product_node_rows[product_idx] = static_cast<std::uint32_t>(product_items[product_idx].node_idx);
    }

    const auto node_cardinality_rows = gather_fixed_rows(
        model.secret_shared_payload().node_cardinalities,
        product_node_rows);
    const auto inv_cardinality_rows = gather_fixed_rows(
        model.secret_shared_payload().node_inv_cardinalities,
        product_node_rows);
    const auto zero_product_rows = repeat_fixed_scalar_matrix(zero_fixed, product_count, 1);
    auto node_cardinality_for_empty = node_cardinality_rows;
    auto zero_for_empty = zero_product_rows;
    sbMatrix is_empty_rows;
    cipher_eq(
        context.role,
        node_cardinality_for_empty,
        zero_for_empty,
        is_empty_rows,
        *(context.eval),
        *(context.runtime));
    sf64Matrix<kFlatBSPNDecimal> effective_cnt_rows = node_cardinality_rows;
    if (needs_leaf_domain_filter) {
        auto final_cnt_for_full_match = final_cnt_int_rows;
        si64Matrix total_rows_int_rows(static_cast<u64>(product_items.size()), 1);
        for (u64 row = 0; row < static_cast<u64>(product_items.size()); ++row) {
            total_rows_int_rows.mShares[0](row, 0) = total_rows_int.mShares[0](0, 0);
            total_rows_int_rows.mShares[1](row, 0) = total_rows_int.mShares[1](0, 0);
        }
        sbMatrix full_match_rows;
        cipher_eq(
            context.role,
            final_cnt_for_full_match,
            total_rows_int_rows,
            full_match_rows,
            *(context.eval),
            *(context.runtime));
        const auto final_cnt_fixed_rows =
            use_row_weights_for_rows ? weighted_final_cnt_rows : si64_to_sf64(final_cnt_int_rows);
        effective_cnt_rows = select_fixed_by_bool_same_shape(
            node_cardinality_rows,
            final_cnt_fixed_rows,
            full_match_rows,
            context);
    } else if (use_row_weights_for_rows) {
        effective_cnt_rows = weighted_final_cnt_rows;
    }
    if (context.debug_internal_reveal && std::getenv("BSPN_DEBUG_LEAF_PRODUCT_COUNTS") != nullptr) {
        i64Matrix final_cnt_plain(final_cnt_int_rows.rows(), final_cnt_int_rows.cols());
        context.enc->revealAll(context.runtime->noDependencies(), final_cnt_int_rows, final_cnt_plain).get();
        f64Matrix<kFlatBSPNDecimal> effective_cnt_plain(effective_cnt_rows.rows(), effective_cnt_rows.cols());
        context.enc->revealAll(context.runtime->noDependencies(), effective_cnt_rows, effective_cnt_plain).get();
        f64Matrix<kFlatBSPNDecimal> node_cardinality_plain(node_cardinality_rows.rows(), node_cardinality_rows.cols());
        context.enc->revealAll(context.runtime->noDependencies(), node_cardinality_rows, node_cardinality_plain).get();
        if (context.role == 0) {
            std::cerr << "bspn_debug_leaf_product_counts:"
                      << " factor_index=" << factor.factor.factor_index
                      << " model_total_rows=" << (factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows)
                      << " product_count=" << product_count
                      << "\n";
            for (u64 product_idx = 0; product_idx < product_count; ++product_idx) {
                const auto node_id = product_items[static_cast<std::size_t>(product_idx)].node_idx;
                const auto final_cnt = final_cnt_plain(product_idx, 0);
                const double effective_cnt = static_cast<double>(effective_cnt_plain(product_idx, 0));
                const double node_card = static_cast<double>(node_cardinality_plain(product_idx, 0));
                if (final_cnt != 0 || std::abs(effective_cnt) > 1e-12) {
                    std::cerr << "bspn_debug_leaf_product_count:"
                              << " factor_index=" << factor.factor.factor_index
                              << " product_idx=" << product_idx
                              << " node_id=" << node_id
                              << " final_cnt=" << final_cnt
                              << " effective_cnt=" << effective_cnt
                              << " node_cardinality=" << node_card
                              << "\n";
                }
            }
        }
    }
    const auto selectivity_num_rows = secure_mul_fixed_same_shape(
        effective_cnt_rows,
        inv_cardinality_rows,
        context);
    sf64Matrix<kFlatBSPNDecimal> selectivity_num_nonzero_rows = selectivity_num_rows;
    sbMatrix product_zero_rows(product_count, 1);
    bool_init_false(context.role, product_zero_rows);
    if (needs_leaf_domain_filter) {
        sbMatrix has_positive_count_rows;
        if (use_row_weights_for_rows) {
            auto final_cnt_for_positive = weighted_final_cnt_rows;
            auto zero_count_rows = repeat_fixed_scalar_matrix(
                zero_fixed,
                final_cnt_for_positive.rows(),
                final_cnt_for_positive.cols());
            cipher_gt(
                context.role,
                final_cnt_for_positive,
                zero_count_rows,
                has_positive_count_rows,
                *(context.eval),
                *(context.runtime));
        } else {
            auto final_cnt_for_positive = final_cnt_int_rows;
            si64Matrix zero_count_rows(final_cnt_int_rows.rows(), final_cnt_int_rows.cols());
            zero_count_rows.mShares[0].setZero();
            zero_count_rows.mShares[1].setZero();
            cipher_gt(
                context.role,
                final_cnt_for_positive,
                zero_count_rows,
                has_positive_count_rows,
                *(context.eval),
                *(context.runtime));
        }
        selectivity_num_nonzero_rows = fixed_mul_bool_same_shape(
            selectivity_num_rows,
            has_positive_count_rows,
            context);
        sbMatrix no_positive_count_rows(has_positive_count_rows.rows(), has_positive_count_rows.bitCount());
        auto has_positive_count_rows_copy = has_positive_count_rows;
        bool_cipher_not(context.role, has_positive_count_rows_copy, no_positive_count_rows);
        product_zero_rows = no_positive_count_rows;
    }
    sbMatrix has_nonempty_node_rows(is_empty_rows.rows(), is_empty_rows.bitCount());
    auto is_empty_rows_copy = is_empty_rows;
    bool_cipher_not(context.role, is_empty_rows_copy, has_nonempty_node_rows);
    selectivity_num_nonzero_rows = fixed_mul_bool_same_shape(
        selectivity_num_nonzero_rows,
        has_nonempty_node_rows,
        context);
    if (needs_leaf_domain_filter) {
        sbMatrix updated_product_zero_rows(product_zero_rows.rows(), product_zero_rows.bitCount());
        bool_cipher_or(
            context.role,
            product_zero_rows,
            is_empty_rows,
            updated_product_zero_rows,
            *(context.enc),
            *(context.eval),
            *(context.runtime));
        product_zero_rows = std::move(updated_product_zero_rows);
    }

    sf64Matrix<kFlatBSPNDecimal> inv_cnt_rows(product_count, 1);
    if (public_factor_feature_count > 1) {
        auto effective_cnt_for_eq = effective_cnt_rows;
        auto zero_for_eq = zero_product_rows;
        sbMatrix is_zero_effective_rows;
        cipher_eq(
            context.role,
            effective_cnt_for_eq,
            zero_for_eq,
            is_zero_effective_rows,
            *(context.eval),
            *(context.runtime));
        const auto zero_cnt_rows = bool_matrix_to_fixed_same_shape(is_zero_effective_rows, context);
        const auto denom_safe_rows = effective_cnt_rows + zero_cnt_rows;
        inv_cnt_rows = secure_count_reciprocal_newton_scaled_matrix(
            denom_safe_rows,
            factor.factor.total_rows != 0 ? factor.factor.total_rows : model.manifest().total_rows,
            context);
        if (eval_stats != nullptr) {
            ++eval_stats->internal_reciprocal_calls;
        }
    }

    auto push_fixed_rows_as_rationals = [&](const sf64Matrix<kFlatBSPNDecimal>& rows) {
        for (u64 product_idx = 0; product_idx < product_count; ++product_idx) {
            SecureRationalShare value{
                fixed_row_slice(rows, static_cast<std::uint32_t>(product_idx), 1),
                one_fixed,
                1.0,
                1.0,
                true,
            };
            if (needs_leaf_domain_filter) {
                value.has_secret_zero_numerator = true;
                value.secret_zero_numerator = bool_row_slice(
                    product_zero_rows,
                    static_cast<std::uint32_t>(product_idx),
                    1);
                value.numerator = fixed_mul_bool_same_shape(
                    value.numerator,
                    bool_not_scalar(value.secret_zero_numerator, context),
                    context);
            }
            out.push_back(std::move(value));
        }
    };
    auto push_count_selectivity_rows_as_rationals = [&]() {
        if (use_row_weights_for_rows && factor.factor.weighted_count_direct) {
            for (u64 product_idx = 0; product_idx < product_count; ++product_idx) {
                SecureRationalShare value{
                    fixed_row_slice(effective_cnt_rows, static_cast<std::uint32_t>(product_idx), 1),
                    one_fixed,
                    1.0,
                    1.0,
                    true,
                };
                value.has_secret_zero_numerator = true;
                value.secret_zero_numerator = bool_row_slice(
                    product_zero_rows,
                    static_cast<std::uint32_t>(product_idx),
                    1);
                value.numerator = fixed_mul_bool_same_shape(
                    value.numerator,
                    bool_not_scalar(value.secret_zero_numerator, context),
                    context);
                out.push_back(std::move(value));
            }
            return;
        }
        if (product_count > 1) {
            push_fixed_rows_as_rationals(selectivity_num_nonzero_rows);
            return;
        }
        const auto empty_fixed_rows = bool_matrix_to_fixed_same_shape(is_empty_rows, context);
        const auto denom_safe_rows = node_cardinality_rows + empty_fixed_rows;
        const double denominator_public_scale =
            static_cast<double>(std::uint64_t(1) << static_cast<unsigned>(kFlatBSPNDecimal));
        const auto denom_payload_rows = secure_mul_public_fixed(
            denom_safe_rows,
            1.0 / denominator_public_scale,
            context);
        for (u64 product_idx = 0; product_idx < product_count; ++product_idx) {
            SecureRationalShare value{
                fixed_row_slice(effective_cnt_rows, static_cast<std::uint32_t>(product_idx), 1),
                fixed_row_slice(denom_payload_rows, static_cast<std::uint32_t>(product_idx), 1),
                1.0,
                denominator_public_scale,
                false,
            };
            value.has_secret_zero_numerator = true;
            value.secret_zero_numerator = bool_row_slice(
                product_zero_rows,
                static_cast<std::uint32_t>(product_idx),
                1);
            value.numerator = fixed_mul_bool_same_shape(
                value.numerator,
                bool_not_scalar(value.secret_zero_numerator, context),
                context);
            out.push_back(std::move(value));
        }
    };

    if (public_factor_feature_count == 0 &&
        (public_factor_evidence_count != 0 || factor.factor.requires_model_eval)) {
        push_count_selectivity_rows_as_rationals();
    } else if (public_single_target_factor) {
        const auto target_sum_rows = stack_fixed_scalars(target_numerator_sums);
        std::vector<std::uint32_t> leaf_to_product_rows(leaf_product_indices.size());
        for (std::size_t leaf_idx = 0; leaf_idx < leaf_product_indices.size(); ++leaf_idx) {
            leaf_to_product_rows[leaf_idx] = static_cast<std::uint32_t>(leaf_product_indices[leaf_idx]);
        }
        const auto inv_cardinality_by_leaf_rows = gather_fixed_rows(
            inv_cardinality_rows,
            leaf_to_product_rows);
        const auto scaled_num_rows = secure_mul_fixed_same_shape(
            target_sum_rows,
            inv_cardinality_by_leaf_rows,
            context);
        const auto target_flags_rows = stack_bool_scalars(target_flags);
        const auto zero_leaf_rows = repeat_fixed_scalar_matrix(
            zero_fixed,
            static_cast<u64>(leaf_children.size()),
            1);
        const auto selected_scaled_rows = select_fixed_by_bool_same_shape(
            scaled_num_rows,
            zero_leaf_rows,
            target_flags_rows,
            context);
        sf64Matrix<kFlatBSPNDecimal> scalar_target_sum_rows(product_count, 1);
        scalar_target_sum_rows[0].setZero();
        scalar_target_sum_rows[1].setZero();
        for (std::size_t product_idx = 0; product_idx < product_items.size(); ++product_idx) {
            const auto& item = product_items[product_idx];
            for (std::size_t offset = 0; offset < item.leaf_count; ++offset) {
                const u64 leaf_idx = static_cast<u64>(item.leaf_begin + offset);
                scalar_target_sum_rows[0](static_cast<u64>(product_idx), 0) = add_i64_mod(
                    scalar_target_sum_rows[0](static_cast<u64>(product_idx), 0),
                    selected_scaled_rows[0](leaf_idx, 0));
                scalar_target_sum_rows[1](static_cast<u64>(product_idx), 0) = add_i64_mod(
                    scalar_target_sum_rows[1](static_cast<u64>(product_idx), 0),
                    selected_scaled_rows[1](leaf_idx, 0));
            }
        }
        const auto has_target_rows = stack_bool_scalars(has_target_by_product);
        auto final_rows = select_fixed_by_bool_same_shape(
            scalar_target_sum_rows,
            selectivity_num_nonzero_rows,
            has_target_rows,
            context);
        final_rows = select_fixed_by_bool_same_shape(
            zero_product_rows,
            final_rows,
            is_empty_rows,
            context);
        push_fixed_rows_as_rationals(final_rows);
    } else {
        const auto target_sum_rows = stack_fixed_scalars(target_numerator_sums);
        std::vector<std::uint32_t> leaf_to_product_rows(leaf_product_indices.size());
        for (std::size_t leaf_idx = 0; leaf_idx < leaf_product_indices.size(); ++leaf_idx) {
            leaf_to_product_rows[leaf_idx] = static_cast<std::uint32_t>(leaf_product_indices[leaf_idx]);
        }
        const auto inv_cnt_by_leaf_rows = gather_fixed_rows(inv_cnt_rows, leaf_to_product_rows);
        const auto exp_component_rows = secure_mul_fixed_same_shape(
            target_sum_rows,
            inv_cnt_by_leaf_rows,
            context);
        const auto target_flags_rows = stack_bool_scalars(target_flags);
        const auto one_leaf_rows = repeat_fixed_scalar_matrix(
            one_fixed,
            static_cast<u64>(leaf_children.size()),
            1);
        const auto selected_component_rows = select_fixed_by_bool_same_shape(
            exp_component_rows,
            one_leaf_rows,
            target_flags_rows,
            context);
        sbMatrix is_empty_by_leaf_rows(static_cast<u64>(leaf_children.size()), is_empty_rows.bitCount());
        const u64 is_empty_cols = static_cast<u64>(is_empty_rows.mShares[0].cols());
        #pragma omp parallel for schedule(static)
        for (std::int64_t leaf_idx_signed = 0; leaf_idx_signed < static_cast<std::int64_t>(leaf_product_indices.size()); ++leaf_idx_signed) {
            const auto leaf_idx = static_cast<std::size_t>(leaf_idx_signed);
            const auto product_idx = static_cast<u64>(leaf_product_indices[leaf_idx]);
            for (u64 col = 0; col < is_empty_cols; ++col) {
                is_empty_by_leaf_rows.mShares[0](static_cast<u64>(leaf_idx), col) =
                    is_empty_rows.mShares[0](product_idx, col);
                is_empty_by_leaf_rows.mShares[1](static_cast<u64>(leaf_idx), col) =
                    is_empty_rows.mShares[1](product_idx, col);
            }
        }
        const auto safe_component_rows = select_fixed_by_bool_same_shape(
            one_leaf_rows,
            selected_component_rows,
            is_empty_by_leaf_rows,
            context);

        sf64Matrix<kFlatBSPNDecimal> scalar_product_rows =
            repeat_fixed_scalar_matrix(one_fixed, product_count, 1);
        std::size_t max_leaf_count = 0;
        for (const auto& item : product_items) {
            max_leaf_count = std::max<std::size_t>(max_leaf_count, item.leaf_count);
        }
        for (std::size_t offset = 0; offset < max_leaf_count; ++offset) {
            auto component_rows = repeat_fixed_scalar_matrix(one_fixed, product_count, 1);
            #pragma omp parallel for schedule(static)
            for (std::int64_t product_idx_signed = 0; product_idx_signed < static_cast<std::int64_t>(product_items.size()); ++product_idx_signed) {
                const auto product_idx = static_cast<std::size_t>(product_idx_signed);
                const auto& item = product_items[product_idx];
                if (offset >= item.leaf_count) {
                    continue;
                }
                const auto leaf_idx = static_cast<u64>(item.leaf_begin + offset);
                component_rows[0](static_cast<u64>(product_idx), 0) = safe_component_rows[0](leaf_idx, 0);
                component_rows[1](static_cast<u64>(product_idx), 0) = safe_component_rows[1](leaf_idx, 0);
            }
            scalar_product_rows = secure_mul_fixed_same_shape(scalar_product_rows, component_rows, context);
        }
        const auto final_rows = secure_mul_fixed_same_shape(selectivity_num_nonzero_rows, scalar_product_rows, context);
        push_fixed_rows_as_rationals(final_rows);
    }

    if (eval_stats != nullptr) {
        eval_stats->final_combine_ms += elapsed_ms_since(phase_start);
    }
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
    const auto& secret_payload = model.secret_shared_payload().dense_bucket_bitmaps_loaded
        ? model.secret_shared_payload()
        : throw std::runtime_error("secure shared payload not loaded");
    (void)secret_payload;
    const auto& node_records = model.nodes();
    const auto& child_ids = model.children();
    auto product_child_scope_mask = [&](const FlatBSPNNodeRecord& node) {
        std::vector<std::uint8_t> scope(manifest.column_names.size(), 0);
        for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
            const auto child_id = child_ids[node.child_begin + offset];
            const auto& child_scope = node_records[child_id].scope_mask;
            const std::size_t limit = std::min<std::size_t>(scope.size(), child_scope.size());
            for (std::size_t idx = 0; idx < limit; ++idx) {
                scope[idx] = static_cast<std::uint8_t>(scope[idx] | child_scope[idx]);
            }
        }
        return scope;
    };

    std::vector<SecureRationalShare> node_values(model.manifest().node_count);
    const auto zero = make_secure_rational(0.0, 1.0, context);
    const auto one = make_secure_rational(1.0, 1.0, context);
    std::fill(node_values.begin(), node_values.end(), zero);
    sbMatrix evidence_domain_miss = shared_zero_bool_scalar(context);

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
    const si64Matrix secret_evidence_scope =
        shared_query_factor_evidence_scope(shared_query_payload, secret_factor_row);
    std::size_t public_factor_feature_count = static_cast<std::size_t>(factor.factor.public_feature_count);
    std::size_t public_factor_evidence_count = static_cast<std::size_t>(factor.factor.public_evidence_count);
    if (feature_scope_override != nullptr) {
        public_factor_feature_count = 0;
    }
    std::vector<std::uint8_t> company_id_scope_mask(manifest.column_names.size(), 0);
    for (std::size_t col_idx = 0; col_idx < manifest.column_names.size(); ++col_idx) {
        const auto& column_name = manifest.column_names[col_idx];
        if (column_name.size() >= std::string(".company_id").size() &&
            column_name.compare(
                column_name.size() - std::string(".company_id").size(),
                std::string(".company_id").size(),
                ".company_id") == 0) {
            company_id_scope_mask[col_idx] = 1;
        }
    }
    const auto secret_company_id_evidence = secure_scope_intersects(
        company_id_scope_mask,
        secret_evidence_scope,
        context);
    // Store count/count rational pairs in a public coarser unit. Scaling the
    // numerator and denominator by the same public value preserves the value
    // while reducing fixed-point overflow risk when rational denominators are
    // multiplied higher in the SPN.
    if (public_factor_feature_count == 0 &&
        public_factor_evidence_count == 0 &&
        !factor.factor.requires_model_eval) {
        return one;
    }

    i64Matrix global_rows_plain(factor.factor.total_rows, 1);
    global_rows_plain.setOnes();
    sbMatrix global_rows_shared;
    share_bool_matrix(global_rows_plain, global_rows_shared, 0, context);

    std::vector<bool> leaf_product_precomputed(model.manifest().node_count, false);
    std::vector<LeafProductBatchItem> leaf_product_items;
    std::vector<const FlatBSPNNodeRecord*> group_leaf_children;
    std::vector<std::uint32_t> group_leaf_node_ids;
    std::vector<std::size_t> group_leaf_product_indices;
    auto leaf_product_item_scope_mask = [&](const LeafProductBatchItem& item) {
        std::vector<std::uint8_t> scope(manifest.column_names.size(), 0);
        for (std::size_t offset = 0; offset < item.leaf_count; ++offset) {
            const auto& child_scope = group_leaf_children[item.leaf_begin + offset]->scope_mask;
            const std::size_t limit = std::min<std::size_t>(scope.size(), child_scope.size());
            for (std::size_t idx = 0; idx < limit; ++idx) {
                scope[idx] = static_cast<std::uint8_t>(scope[idx] | child_scope[idx]);
            }
        }
        return scope;
    };
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
        LeafProductBatchItem item;
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
        sbMatrix leaf_product_domain_miss = shared_zero_bool_scalar(context);
        const auto product_values = evaluate_leaf_product_batch_values(
            model,
            factor,
            shared_query_payload,
            secret_factor_row,
            secret_feature_scope,
            group_leaf_children,
            group_leaf_node_ids,
            leaf_product_items,
            group_leaf_product_indices,
            global_rows_shared,
            context,
            eval_stats,
            &leaf_product_domain_miss);
        evidence_domain_miss = bool_or_scalar(
            evidence_domain_miss,
            leaf_product_domain_miss,
            context);
        const auto company_id_domain_miss = bool_and_scalar(
            evidence_domain_miss,
            secret_company_id_evidence,
            context);
        for (std::size_t product_idx = 0; product_idx < leaf_product_items.size(); ++product_idx) {
            const auto node_idx = leaf_product_items[product_idx].node_idx;
            node_values[node_idx] =
                (public_factor_feature_count == 0 &&
                 public_factor_evidence_count != 0 &&
                 manifest.column_names.size() > 1)
                    ? zero_rational_when_scope_missing(
                        product_values[product_idx],
                        leaf_product_item_scope_mask(leaf_product_items[product_idx]),
                        secret_evidence_scope,
                        company_id_domain_miss,
                        context)
                    : product_values[product_idx];
            leaf_product_precomputed[node_idx] = true;
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
            sbMatrix has_relevant_child = shared_zero_bool_scalar(context);
            for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
                const auto child_id = child_ids[node.child_begin + offset];
                const auto& child_value = node_values[child_id];
                const auto& relevant_flag = relevant_flags[static_cast<std::size_t>(offset)];
                const auto multiplied = multiply_secure_rational(product, child_value, context);
                const auto first_relevant_child = bool_and_scalar(
                    relevant_flag,
                    bool_not_scalar(has_relevant_child, context),
                    context);
                const auto relevant_product = select_rational_by_bool(
                    child_value,
                    multiplied,
                    first_relevant_child,
                    context);
                product = select_rational_by_bool(
                    relevant_product,
                    product,
                    relevant_flag,
                    context);
                has_relevant_child = bool_or_scalar(has_relevant_child, relevant_flag, context);
            }
            node_values[node_idx] =
                (public_factor_feature_count == 0 &&
                 public_factor_evidence_count != 0 &&
                 manifest.column_names.size() > 1)
                    ? zero_rational_when_scope_missing(
                        product,
                        product_child_scope_mask(node),
                        secret_evidence_scope,
                        bool_and_scalar(evidence_domain_miss, secret_company_id_evidence, context),
                        context)
                    : product;
            if (eval_stats != nullptr) {
                eval_stats->product_sum_ms += elapsed_ms_since(phase_start);
            }
            continue;
        }

        if (first_child.node_type != FlatBSPNNodeType::LEAF) {
            throw std::runtime_error("Unsupported product child shape in secure evaluation.");
        }

        std::vector<const FlatBSPNNodeRecord*> leaf_children;
        leaf_children.reserve(node.child_count);
        std::vector<std::uint32_t> leaf_node_ids;
        leaf_node_ids.reserve(node.child_count);
        std::vector<std::size_t> leaf_product_indices;
        leaf_product_indices.reserve(node.child_count);
        for (std::uint32_t offset = 0; offset < node.child_count; ++offset) {
            const auto child_id = child_ids[node.child_begin + offset];
            leaf_children.push_back(&node_records[child_id]);
            leaf_node_ids.push_back(child_id);
            leaf_product_indices.push_back(0);
        }
        if (eval_stats != nullptr) {
            ++eval_stats->leaf_product_nodes;
        }
        const std::vector<LeafProductBatchItem> single_product_item{{
            node_idx,
            0,
            static_cast<std::size_t>(node.child_count),
        }};
        sbMatrix single_domain_miss = shared_zero_bool_scalar(context);
        const auto product_values = evaluate_leaf_product_batch_values(
            model,
            factor,
            shared_query_payload,
            secret_factor_row,
            secret_feature_scope,
            leaf_children,
            leaf_node_ids,
            single_product_item,
            leaf_product_indices,
            global_rows_shared,
            context,
            eval_stats,
            &single_domain_miss);
        evidence_domain_miss = bool_or_scalar(
            evidence_domain_miss,
            single_domain_miss,
            context);
        const auto single_company_id_domain_miss = bool_and_scalar(
            single_domain_miss,
            secret_company_id_evidence,
            context);
        node_values[node_idx] =
            (public_factor_feature_count == 0 &&
             public_factor_evidence_count != 0 &&
             manifest.column_names.size() > 1)
                ? zero_rational_when_scope_missing(
                    product_values.front(),
                    product_child_scope_mask(node),
                    secret_evidence_scope,
                    single_company_id_domain_miss,
                    context)
                : product_values.front();
    }

    if (context.debug_internal_reveal && std::getenv("BSPN_DEBUG_NODE_VALUE_IDS") != nullptr) {
        const int factor_filter = std::getenv("BSPN_DEBUG_NODE_VALUE_FACTOR") != nullptr
            ? std::atoi(std::getenv("BSPN_DEBUG_NODE_VALUE_FACTOR"))
            : factor.factor.factor_index;
        if (factor_filter == factor.factor.factor_index) {
            const auto debug_node_ids = parse_bspn_u32_list_env("BSPN_DEBUG_NODE_VALUE_IDS");
            for (const auto node_id : debug_node_ids) {
                if (node_id >= node_values.size()) {
                    continue;
                }
                const double numerator = reveal_scaled_numerator(node_values[node_id], context);
                const double denominator = reveal_scaled_denominator(node_values[node_id], context);
                const double value = std::abs(denominator) <= 1e-12 ? 0.0 : numerator / denominator;
                if (context.role == 0) {
                    std::cerr << "bspn_debug_node_value:"
                              << " factor_index=" << factor.factor.factor_index
                              << " node_id=" << node_id
                              << " numerator=" << numerator
                              << " denominator=" << denominator
                              << " value=" << value
                              << "\n";
                }
            }
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

    json factor_timing_profile = json::array();
    json factor_trace_shares = json::array();
    std::map<std::string, SecureRationalShare> factor_value_cache;
    std::uint64_t factor_cache_hits = 0;
    std::uint64_t factor_cache_misses = 0;
    auto eval_factor_product_secure = [&](
        const json& factors_doc,
        json* factor_debug,
        const std::string& profile_section) {
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
            const std::string factor_cache_key =
                bound.manifest_path + "|" +
                std::to_string(bound.secret_factor_binding_index) + "|" +
                factor_doc.dump();
            const auto cache_it = factor_value_cache.find(factor_cache_key);
            if (cache_it != factor_value_cache.end()) {
                factor_value = cache_it->second;
                ++factor_cache_hits;
                json factor_timing_doc = secure_indicator_stats_json(indicator_stats);
                factor_timing_doc["profile_section"] = profile_section;
                factor_timing_doc["factor_index"] = bound.factor.factor_index;
                factor_timing_doc["factor_kind"] = bound.factor.factor_kind;
                factor_timing_doc["inverse"] = bound.factor.inverse;
                factor_timing_doc["model_id"] = bound.model_id;
                factor_timing_doc["public_feature_count"] = bound.factor.public_feature_count;
                factor_timing_doc["public_evidence_count"] = bound.factor.public_evidence_count;
                factor_timing_doc["cache_hit"] = true;
                if (!bound.manifest_path.empty()) {
                    const auto timing_model_it = model_cache.find(bound.manifest_path);
                    if (timing_model_it != model_cache.end()) {
                        const auto& manifest = timing_model_it->second.manifest();
                        factor_timing_doc["bucket_count"] = manifest.bucket_count;
                        if (manifest.has_real_bucket_count) {
                            factor_timing_doc["real_bucket_count"] = manifest.real_bucket_count;
                        }
                        factor_timing_doc["padding_bucket_count"] = manifest.padding_bucket_count;
                        factor_timing_doc["bucket_padding_scope"] = manifest.bucket_padding_scope;
                        factor_timing_doc["sample_total_rows"] = manifest.sample_total_rows;
                    }
                }
                factor_timing_profile.push_back(std::move(factor_timing_doc));
                if (factor_debug != nullptr && context.debug_internal_reveal) {
                    const double numerator = reveal_scaled_numerator(factor_value, context);
                    const double denominator = reveal_scaled_denominator(factor_value, context);
                    json factor_debug_doc = {
                        {"factor_index", bound.factor.factor_index},
                        {"factor_kind", bound.factor.factor_kind},
                        {"inverse", bound.factor.inverse},
                        {"model_id", bound.model_id},
                        {"cache_hit", true},
                        {"numerator", numerator},
                        {"denominator", denominator},
                        {"value", std::abs(denominator) <= 1e-12 ? 0.0 : numerator / denominator},
                    };
                    factor_debug->push_back(factor_debug_doc);
                }
                if (context.factor_trace_shares) {
                    factor_trace_shares.push_back({
                        {"stage", "factor"},
                        {"profile_section", profile_section},
                        {"factor_index", bound.factor.factor_index},
                        {"factor_kind", bound.factor.factor_kind},
                        {"inverse", bound.factor.inverse},
                        {"model_id", bound.model_id},
                        {"cache_hit", true},
                        {"public_feature_count", bound.factor.public_feature_count},
                        {"public_evidence_count", bound.factor.public_evidence_count},
                        {"weighted_count_direct", bound.factor.weighted_count_direct},
                        {"rational_share", secure_rational_share_json(factor_value)},
                    });
                }
                product = multiply_secure_rational(product, factor_value, context);
                if (context.factor_trace_shares) {
                    factor_trace_shares.push_back({
                        {"stage", "prefix"},
                        {"profile_section", profile_section},
                        {"factor_index", bound.factor.factor_index},
                        {"factor_kind", "FACTOR_PRODUCT_PREFIX"},
                        {"model_id", bound.model_id},
                        {"rational_share", secure_rational_share_json(product)},
                    });
                }
                continue;
            }
            ++factor_cache_misses;
            bool exact_unit_factor = false;
            if (bound.factor.factor_kind == "CONSTANT") {
                if (std::abs(bound.factor.public_constant_value) > 1000000.0) {
                    factor_value = make_secure_public_scaled_constant(bound.factor.public_constant_value, context);
                } else {
                    factor_value = make_secure_rational(bound.factor.public_constant_value, 1.0, context);
                }
	            } else if (bound.factor.factor_kind == "INDICATOR_EXPECTATION" ||
	                       bound.factor.factor_kind == "EXPECTATION") {
                if (bound.factor.public_feature_count == 0 &&
                    bound.factor.public_evidence_count == 0 &&
                    !bound.factor.requires_model_eval) {
                    factor_value = make_secure_rational(1.0, 1.0, context);
                    exact_unit_factor = true;
                } else {
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
                }
	            } else {
	                throw std::runtime_error(
	                    "Secure production path currently supports CONSTANT, INDICATOR_EXPECTATION, and EXPECTATION factors.");
            }
            if (!exact_unit_factor &&
                bound.factor.factor_kind == "INDICATOR_EXPECTATION" &&
                !bound.factor.inverse &&
                !bound.factor.weighted_count_direct &&
                !(bound.factor.public_feature_count == 0 &&
                  bound.factor.public_evidence_count > 0)) {
                constexpr double kSecureD16IndicatorPayloadUpscale = 4096.0;
                factor_value.numerator = secure_nonnegative_fixed_same_shape(
                    factor_value.numerator,
                    context);
                if (bound.factor.public_feature_count == 0 &&
                    bound.factor.public_evidence_count > 0) {
                    factor_value.numerator = secure_mul_public_fixed(
                        factor_value.numerator,
                        kSecureD16IndicatorPayloadUpscale,
                        context);
                    factor_value.numerator_scale /= kSecureD16IndicatorPayloadUpscale;
                    factor_value.numerator = secure_nonnegative_fixed_same_shape(
                        factor_value.numerator,
                        context);
                }
                auto factor_numerator_for_cmp = factor_value.numerator;
                sf64Matrix<kFlatBSPNDecimal> zero_numerator_for_cmp(
                    factor_numerator_for_cmp.rows(),
                    factor_numerator_for_cmp.cols());
                zero_numerator_for_cmp[0].setZero();
                zero_numerator_for_cmp[1].setZero();
                sbMatrix factor_numerator_positive;
                cipher_gt(
                    context.role,
                    factor_numerator_for_cmp,
                    zero_numerator_for_cmp,
                    factor_numerator_positive,
                    *(context.eval),
                    *(context.runtime));
                factor_value.has_secret_zero_numerator = true;
                factor_value.secret_zero_numerator = bool_not_scalar(
                    factor_numerator_positive,
                    context);
            }
            if (!exact_unit_factor) {
                factor_value = maybe_invert_secure_rational(factor_value, bound.factor.inverse);
            }
            factor_value_cache[factor_cache_key] = factor_value;
            json factor_timing_doc = secure_indicator_stats_json(indicator_stats);
            factor_timing_doc["profile_section"] = profile_section;
            factor_timing_doc["factor_index"] = bound.factor.factor_index;
            factor_timing_doc["factor_kind"] = bound.factor.factor_kind;
            factor_timing_doc["inverse"] = bound.factor.inverse;
            factor_timing_doc["model_id"] = bound.model_id;
            factor_timing_doc["public_feature_count"] = bound.factor.public_feature_count;
            factor_timing_doc["public_evidence_count"] = bound.factor.public_evidence_count;
            factor_timing_doc["cache_hit"] = false;
            if (!bound.manifest_path.empty()) {
                const auto timing_model_it = model_cache.find(bound.manifest_path);
                if (timing_model_it != model_cache.end()) {
                    const auto& manifest = timing_model_it->second.manifest();
                    factor_timing_doc["bucket_count"] = manifest.bucket_count;
                    if (manifest.has_real_bucket_count) {
                        factor_timing_doc["real_bucket_count"] = manifest.real_bucket_count;
                    }
                    factor_timing_doc["padding_bucket_count"] = manifest.padding_bucket_count;
                    factor_timing_doc["bucket_padding_scope"] = manifest.bucket_padding_scope;
                    factor_timing_doc["sample_total_rows"] = manifest.sample_total_rows;
                }
            }
            factor_timing_profile.push_back(std::move(factor_timing_doc));
            if (factor_debug != nullptr && context.debug_internal_reveal) {
                const double numerator = reveal_scaled_numerator(factor_value, context);
                const double denominator = reveal_scaled_denominator(factor_value, context);
                const double numerator_raw = reveal_fixed_scalar(factor_value.numerator, context);
                const double denominator_raw = reveal_fixed_scalar(factor_value.denominator, context);
                json factor_debug_doc = {
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
                    {"numerator_raw", numerator_raw},
                    {"denominator_raw", denominator_raw},
                    {"numerator_scale", factor_value.numerator_scale},
                    {"denominator_scale", factor_value.denominator_scale},
                    {"value", std::abs(denominator) <= 1e-12 ? 0.0 : numerator / denominator},
                };
                if (!bound.manifest_path.empty()) {
                    const auto debug_model_it = model_cache.find(bound.manifest_path);
                    if (debug_model_it != model_cache.end()) {
                        const auto& manifest = debug_model_it->second.manifest();
                        factor_debug_doc["factor_total_rows"] = bound.factor.total_rows;
                        factor_debug_doc["sample_total_rows"] = manifest.sample_total_rows;
                        factor_debug_doc["actual_total_rows"] = manifest.actual_total_rows;
                        factor_debug_doc["sample_scale"] = manifest.sample_scale;
                    }
                }
                factor_debug->push_back(factor_debug_doc);
            }
            if (context.factor_trace_shares) {
                factor_trace_shares.push_back({
                    {"stage", "factor"},
                    {"profile_section", profile_section},
                    {"factor_index", bound.factor.factor_index},
                    {"factor_kind", bound.factor.factor_kind},
                    {"inverse", bound.factor.inverse},
                    {"model_id", bound.model_id},
                    {"cache_hit", false},
                    {"public_feature_count", bound.factor.public_feature_count},
                    {"public_evidence_count", bound.factor.public_evidence_count},
                    {"weighted_count_direct", bound.factor.weighted_count_direct},
                    {"rational_share", secure_rational_share_json(factor_value)},
                });
            }
            product = multiply_secure_rational(product, factor_value, context);
            if (context.factor_trace_shares) {
                factor_trace_shares.push_back({
                    {"stage", "prefix"},
                    {"profile_section", profile_section},
                    {"factor_index", bound.factor.factor_index},
                    {"factor_kind", "FACTOR_PRODUCT_PREFIX"},
                    {"model_id", bound.model_id},
                    {"rational_share", secure_rational_share_json(product)},
                });
            }
        }
        if (factor_debug != nullptr && context.debug_internal_reveal) {
            json product_debug_doc = secure_rational_debug_json(product, context);
            product_debug_doc["factor_index"] = -1;
            product_debug_doc["factor_kind"] = "FACTOR_PRODUCT";
            factor_debug->push_back(product_debug_doc);
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
            &cardinality_factor_debug,
            "cardinality");
        if (factor_array_has_large_public_scale(public_plan_doc["cardinality_plan"].value("factors", json::array()))) {
            out.root_division_payload_scale = std::min(
                out.root_division_payload_scale,
                1.0 / 4096.0);
            out.root_division_scale_denominator_payload = true;
        }
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
                &expectation_factor_debug,
                "expectation");
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
                    &numerator_factor_debug,
                    "numerator");
                if (aggregation_type == "AVG") {
                    auto denominator_rational = eval_factor_product_secure(
                    term_doc.at("denominator_plan").value("factors", json::array()),
                    &denominator_factor_debug,
                    "denominator");
                current_value = multiply_secure_rational(
                    numerator_rational,
                    invert_secure_rational(denominator_rational),
                    context);
                out.root_division_payload_scale = std::min(
                    out.root_division_payload_scale,
                    1.0 / 4096.0);
                out.root_division_scale_denominator_payload = true;
            } else {
                current_value = numerator_rational;
                out.root_division_payload_scale = std::min(
                    out.root_division_payload_scale,
                    1.0 / 4096.0);
                if (aggregate_term_has_large_public_scale(term_doc)) {
                    out.root_division_scale_denominator_payload = true;
                }
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

        json term_debug_doc = {
            {"term_index", term_index},
            {"aggregation_type", aggregation_type},
            {"evaluation_mode", evaluation_mode},
            {"expectation_factors", expectation_factor_debug},
            {"numerator_factors", numerator_factor_debug},
            {"denominator_factors", denominator_factor_debug},
        };
        if (context.debug_internal_reveal) {
            term_debug_doc["current_value"] = secure_rational_debug_json(current_value, context);
            term_debug_doc["current_result_value"] = secure_rational_debug_json(current_result_value, context);
        }
        term_debug.push_back(term_debug_doc);
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
    out.factor_trace_shares = factor_trace_shares;
    out.timing_profile = {
        {"query_skeleton_id", public_plan_doc.value("query_skeleton_id", std::string())},
        {"query_kind", query_kind},
        {"cache_hits", factor_cache_hits},
        {"cache_misses", factor_cache_misses},
        {"factors", factor_timing_profile},
    };
    return out;
}

void require_secure_bundle_cli_contract(const oc::CLP& cmd) {
    if (!cmd.isSet("role")) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --role in {0,1,2}.");
    }
    const bool batch_mode = cmd.isSet("batch_bundle_json");
    const bool has_query_share_dir = cmd.isSet("query_share_payload_dir");
    if (!batch_mode && !cmd.isSet("public_plan_json")) {
        throw std::runtime_error("bspn_flat_eval secure mode requires --public_plan_json.");
    }
    if (!batch_mode && !has_query_share_dir && !cmd.isSet("secret_payload_json")) {
        throw std::runtime_error(
            "bspn_flat_eval secure mode requires --secret_payload_json or --query_share_payload_dir.");
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
    context.debug_internal_reveal =
        context.debug_reveal && !cmd.isSet("debug_reveal_final_only");
    context.factor_trace_shares =
        cmd.isSet("factor_trace_shares") ||
        std::getenv("BSPN_FACTOR_TRACE_SHARES") != nullptr;

    bspn_basic_setup(static_cast<u64>(role), ios, enc, eval, runtime);
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
        if (factor_doc.value("public_feature_count", std::uint64_t(0)) == 0 &&
            factor_doc.value("public_evidence_count", std::uint64_t(0)) == 0 &&
            !factor_doc.value("requires_model_eval", false)) {
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
    manifest_.sample_total_rows = manifest_doc.value("sample_total_rows", manifest_.total_rows);
    manifest_.actual_total_rows = manifest_doc.value("actual_total_rows", manifest_.sample_total_rows);
    manifest_.sample_scale = manifest_doc.value(
        "sample_scale",
        manifest_.sample_total_rows == 0
            ? 1.0
            : static_cast<double>(manifest_.actual_total_rows) / static_cast<double>(manifest_.sample_total_rows));
    manifest_.scope_bitmap_bytes = manifest_doc.value("scope_bitmap_bytes", std::uint64_t(0));
    manifest_.children_count = manifest_doc.value("children_count", std::uint64_t(0));
    manifest_.bucket_count = manifest_doc.value("bucket_count", std::uint64_t(0));
    manifest_.weights_count = manifest_doc.value("weights_count", std::uint64_t(0));
    manifest_.leaf_bitmap_bytes = manifest_doc.value("leaf_bitmap_bytes", std::uint64_t(0));
    manifest_.leaf_bucket_width = manifest_doc.value("leaf_bucket_width", std::uint64_t(0));
    manifest_.max_leaf_bucket_width = manifest_doc.value(
        "max_leaf_bucket_width",
        manifest_.leaf_bucket_width);
    manifest_.leaf_node_count = manifest_doc.value("leaf_node_count", std::uint64_t(0));
    manifest_.has_real_bucket_count = manifest_doc.contains("real_bucket_count");
    manifest_.real_bucket_count = manifest_doc.value("real_bucket_count", manifest_.bucket_count);
    manifest_.padding_bucket_count = manifest_doc.value("padding_bucket_count", std::uint64_t(0));
    manifest_.node_cardinality_count = manifest_doc.value("node_cardinality_count", std::uint64_t(0));
    manifest_.node_inv_cardinality_count = manifest_doc.value("node_inv_cardinality_count", std::uint64_t(0));
    manifest_.payload_layout_version = manifest_doc.value("payload_layout_version", std::uint64_t(0));
    manifest_.bucket_padding_scope = manifest_doc.value(
        "bucket_padding_scope",
        manifest_.payload_layout_version >= 5 ? std::string("leaf_product") : std::string("global"));
    if (manifest_.bucket_padding_scope != "none" &&
        manifest_.bucket_padding_scope != "leaf_product" &&
        manifest_.bucket_padding_scope != "global_leaf" &&
        !(manifest_.payload_layout_version < 5 && manifest_.bucket_padding_scope == "global")) {
        throw std::runtime_error(
            "Unsupported bucket_padding_scope in FlatBSPN manifest: " + manifest_.bucket_padding_scope);
    }
    manifest_.secret_payload_dir = manifest_doc.value("secret_payload_dir", std::string("secret"));
    manifest_.secret_payload_encoding = manifest_doc.value("secret_payload_encoding", std::string());
    manifest_.model_secret_share_payload_dir =
        manifest_doc.value("model_secret_share_payload_dir", std::string("secret_shares/model"));
    manifest_.model_secret_share_payload_layout =
        manifest_doc.value("model_secret_share_payload_layout", std::string("pair_matrix_v1"));
    manifest_.secure_multiplier_materialized = manifest_doc.value("secure_multiplier_materialized", false);
    manifest_.secure_share_payload_dir = manifest_doc.value("secure_share_payload_dir", std::string("secret_shares"));
    manifest_.secure_share_payload_layout = manifest_doc.value("secure_share_payload_layout", std::string("consolidated_v1"));
    manifest_.secure_share_bool_encoding = manifest_doc.value(
        "secure_share_bool_encoding",
        std::string("bitpacked_pair_lsb_v1"));
    manifest_.secure_multiplier_bucket_indices.clear();
    if (manifest_doc.contains("secure_multiplier_bucket_indices") &&
        manifest_doc["secure_multiplier_bucket_indices"].is_array()) {
        for (const auto& idx_doc : manifest_doc["secure_multiplier_bucket_indices"]) {
            manifest_.secure_multiplier_bucket_indices.push_back(idx_doc.get<std::uint32_t>());
        }
    }
    manifest_.secure_multiplier_bucket_mapping.clear();
    if (manifest_doc.contains("secure_multiplier_bucket_mapping") &&
        manifest_doc["secure_multiplier_bucket_mapping"].is_array()) {
        for (const auto& item : manifest_doc["secure_multiplier_bucket_mapping"]) {
            FlatSecureBucketMapping mapping;
            mapping.secure_local_idx = item.value("secure_local_idx", std::uint64_t(0));
            mapping.global_bucket_id = item.value("global_bucket_id", std::uint32_t(0));
            mapping.leaf_node_id = item.value("leaf_node_id", std::uint32_t(0));
            mapping.leaf_local_bucket_idx = item.value("leaf_local_bucket_idx", std::uint64_t(0));
            manifest_.secure_multiplier_bucket_mapping.push_back(mapping);
        }
    }
    manifest_.has_leaf_row_values = manifest_doc.value("has_leaf_row_values", false);
    manifest_.leaf_row_value_total_rows = manifest_doc.value("leaf_row_value_total_rows", std::uint64_t(0));
    manifest_.leaf_row_value_node_ids.clear();
    manifest_.leaf_row_value_offsets.clear();
    if (manifest_doc.contains("leaf_row_value_node_ids") &&
        manifest_doc["leaf_row_value_node_ids"].is_array()) {
        for (const auto& item : manifest_doc["leaf_row_value_node_ids"]) {
            manifest_.leaf_row_value_node_ids.push_back(item.get<std::uint32_t>());
        }
    }
    if (manifest_doc.contains("leaf_row_value_offsets") &&
        manifest_doc["leaf_row_value_offsets"].is_array()) {
        for (const auto& item : manifest_doc["leaf_row_value_offsets"]) {
            manifest_.leaf_row_value_offsets.push_back(item.get<std::uint64_t>());
        }
    }
    manifest_.secure_multiplier_leaf_row_value_total_rows =
        manifest_doc.value("secure_multiplier_leaf_row_value_total_rows", std::uint64_t(0));
    manifest_.secure_multiplier_leaf_row_value_node_ids.clear();
    manifest_.secure_multiplier_leaf_row_value_offsets.clear();
    if (manifest_doc.contains("secure_multiplier_leaf_row_value_node_ids") &&
        manifest_doc["secure_multiplier_leaf_row_value_node_ids"].is_array()) {
        for (const auto& item : manifest_doc["secure_multiplier_leaf_row_value_node_ids"]) {
            manifest_.secure_multiplier_leaf_row_value_node_ids.push_back(item.get<std::uint32_t>());
        }
    }
    if (manifest_doc.contains("secure_multiplier_leaf_row_value_offsets") &&
        manifest_doc["secure_multiplier_leaf_row_value_offsets"].is_array()) {
        for (const auto& item : manifest_doc["secure_multiplier_leaf_row_value_offsets"]) {
            manifest_.secure_multiplier_leaf_row_value_offsets.push_back(item.get<std::uint64_t>());
        }
    }
    if (manifest_.has_leaf_row_values &&
        manifest_.leaf_row_value_node_ids.size() != manifest_.leaf_row_value_offsets.size()) {
        throw std::runtime_error("leaf_row_value_node_ids/offsets length mismatch in manifest.");
    }
    manifest_.has_row_weights = manifest_doc.value("has_row_weights", false);
    manifest_.row_weight_count = manifest_doc.value("row_weight_count", std::uint64_t(0));
    manifest_.row_weight_encoding = manifest_doc.value("row_weight_encoding", std::string());
    manifest_.has_leaf_bucket_weight_sums = manifest_doc.value("has_leaf_bucket_weight_sums", false);
    manifest_.leaf_bucket_weight_sum_count = manifest_doc.value("leaf_bucket_weight_sum_count", std::uint64_t(0));
    manifest_.leaf_bucket_weight_sum_encoding = manifest_doc.value("leaf_bucket_weight_sum_encoding", std::string());
    if (manifest_.has_row_weights) {
        if (manifest_.row_weight_count != manifest_.total_rows) {
            throw std::runtime_error("row_weight_count must equal total_rows when has_row_weights is true.");
        }
        if (!manifest_.row_weight_encoding.empty() && manifest_.row_weight_encoding != "fixed_f64_v1") {
            throw std::runtime_error("Unsupported row_weight_encoding in manifest: " + manifest_.row_weight_encoding);
        }
    }
    if (manifest_.has_leaf_bucket_weight_sums) {
        if (manifest_.leaf_bucket_weight_sum_count != manifest_.bucket_count) {
            throw std::runtime_error("leaf_bucket_weight_sum_count must equal bucket_count when enabled.");
        }
        if (!manifest_.leaf_bucket_weight_sum_encoding.empty() &&
            manifest_.leaf_bucket_weight_sum_encoding != "fixed_f64_v1") {
            throw std::runtime_error(
                "Unsupported leaf_bucket_weight_sum_encoding in manifest: " +
                manifest_.leaf_bucket_weight_sum_encoding);
        }
    }

    if (manifest_.payload_layout_version < 4) {
        throw std::runtime_error("FlatBSPN payload_layout_version must be >= 4 for padded secure evaluation.");
    }
    if (manifest_.secure_multiplier_materialized &&
        manifest_.secure_multiplier_bucket_indices.empty()) {
        throw std::runtime_error("secure_multiplier_materialized requires secure_multiplier_bucket_indices.");
    }

    const auto raw_nodes = read_binary_records<PackedRawNodeRecord>(join_path(base_dir_, "nodes.bin"));
    const auto raw_buckets = read_binary_records<PackedRawBucketRecord>(join_path(base_dir_, "bucket_index.bin"));
    children_ = read_binary_records<std::uint32_t>(join_path(base_dir_, "children.bin"));
    const auto scope_blob = read_binary_bytes(join_path(base_dir_, "scope_bitmaps.bin"));

    if (raw_nodes.size() != static_cast<std::size_t>(manifest_.node_count)) {
        throw std::runtime_error("nodes.bin count mismatch with manifest.");
    }
    if (raw_buckets.size() != static_cast<std::size_t>(manifest_.bucket_count)) {
        throw std::runtime_error("bucket_index.bin count mismatch with manifest.");
    }
    if (children_.size() != static_cast<std::size_t>(manifest_.children_count)) {
        throw std::runtime_error("children.bin count mismatch with manifest.");
    }

    nodes_.clear();
    nodes_.reserve(raw_nodes.size());
    std::uint64_t observed_leaf_count = 0;
    std::uint64_t observed_leaf_bucket_count = 0;
    for (const auto& raw : raw_nodes) {
        if (raw.node_type > static_cast<std::uint8_t>(FlatBSPNNodeType::DUMMY)) {
            throw std::runtime_error("Invalid FlatBSPN node type.");
        }
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

        if (node.node_id >= manifest_.node_count) {
            throw std::runtime_error("Node id is out of bounds.");
        }
        if (node.child_begin + node.child_count > children_.size()) {
            throw std::runtime_error("Node child slice is out of bounds.");
        }
        if (node.node_type == FlatBSPNNodeType::DUMMY) {
            throw std::runtime_error("Legacy DUMMY nodes are not valid in padded FlatBSPN v4 payloads.");
        }
        if (node.node_type == FlatBSPNNodeType::LEAF) {
            if (node.bucket_count == 0) {
                throw std::runtime_error("Leaf bucket width must be nonzero when leaves are present.");
            }
            if (manifest_.payload_layout_version < 5 && node.bucket_count != manifest_.leaf_bucket_width) {
                throw std::runtime_error("Leaf bucket_count does not match manifest leaf_bucket_width.");
            }
            if (manifest_.max_leaf_bucket_width != 0 && node.bucket_count > manifest_.max_leaf_bucket_width) {
                throw std::runtime_error("Leaf bucket_count exceeds manifest max_leaf_bucket_width.");
            }
            if (node.bucket_begin + node.bucket_count > raw_buckets.size()) {
                throw std::runtime_error("Leaf bucket slice is out of bounds.");
            }
            ++observed_leaf_count;
            observed_leaf_bucket_count += node.bucket_count;
        } else if (node.bucket_count != 0) {
            throw std::runtime_error("Non-leaf nodes must not expose buckets in padded FlatBSPN v4 payloads.");
        }
        if (node.scope_bitmap_begin + node.scope_bitmap_len > scope_blob.size()) {
            throw std::runtime_error("Node scope bitmap slice is out of bounds.");
        }
        const std::vector<std::uint8_t> packed_scope(
            scope_blob.begin() + static_cast<std::ptrdiff_t>(node.scope_bitmap_begin),
            scope_blob.begin() + static_cast<std::ptrdiff_t>(node.scope_bitmap_begin + node.scope_bitmap_len));
        node.scope_mask = unpack_scope_bits(packed_scope, manifest_.column_names.size());

        nodes_.push_back(node);
    }

    if (observed_leaf_count != manifest_.leaf_node_count) {
        throw std::runtime_error("Leaf node count mismatch with manifest.");
    }
    if (observed_leaf_bucket_count != manifest_.bucket_count) {
        throw std::runtime_error("Padded bucket count mismatch with manifest leaf dimensions.");
    }
    if (manifest_.payload_layout_version < 5 &&
        manifest_.bucket_count != manifest_.leaf_node_count * manifest_.leaf_bucket_width) {
        throw std::runtime_error("Padded bucket count mismatch with manifest leaf dimensions.");
    }
    if (manifest_.payload_layout_version >= 5 &&
        manifest_.has_real_bucket_count &&
        manifest_.padding_bucket_count != 0 &&
        manifest_.real_bucket_count + manifest_.padding_bucket_count != manifest_.bucket_count) {
        throw std::runtime_error("Padded bucket count mismatch with manifest bucket padding metadata.");
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
        if (bucket.bucket_id >= manifest_.bucket_count) {
            throw std::runtime_error("Bucket id is out of bounds.");
        }
        if (bucket.bitmap_len != manifest_.leaf_bitmap_bytes) {
            throw std::runtime_error("Bucket bitmap length does not match manifest leaf_bitmap_bytes.");
        }
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
    row_weights_.clear();
    if (manifest_.has_row_weights) {
        const std::string row_weights_path = join_path(secret_dir, "row_weights.bin");
        if (!path_exists(row_weights_path)) {
            throw std::runtime_error("manifest declares row weights but secret/row_weights.bin is missing.");
        }
        row_weights_ = read_binary_doubles(row_weights_path);
        if (row_weights_.size() != static_cast<std::size_t>(manifest_.row_weight_count)) {
            throw std::runtime_error("row_weights.bin count mismatch with manifest.");
        }
    }
    leaf_bucket_weight_sums_.clear();
    if (manifest_.has_leaf_bucket_weight_sums) {
        const std::string bucket_weight_sums_path = join_path(secret_dir, "leaf_bucket_weight_sums.bin");
        if (!path_exists(bucket_weight_sums_path)) {
            throw std::runtime_error(
                "manifest declares leaf bucket weight sums but secret/leaf_bucket_weight_sums.bin is missing.");
        }
        leaf_bucket_weight_sums_ = read_binary_doubles(bucket_weight_sums_path);
        if (leaf_bucket_weight_sums_.size() !=
            static_cast<std::size_t>(manifest_.leaf_bucket_weight_sum_count)) {
            throw std::runtime_error("leaf_bucket_weight_sums.bin count mismatch with manifest.");
        }
    }
    std::vector<double> leaf_row_values;
    if (manifest_.has_leaf_row_values) {
        const std::string leaf_row_values_path = join_path(secret_dir, "leaf_row_values.bin");
        if (!path_exists(leaf_row_values_path)) {
            throw std::runtime_error("manifest declares leaf row values but secret/leaf_row_values.bin is missing.");
        }
        leaf_row_values = read_binary_doubles(leaf_row_values_path);
        const std::uint64_t expected_rows =
            static_cast<std::uint64_t>(manifest_.leaf_row_value_node_ids.size()) *
            std::max<std::uint64_t>(1, manifest_.leaf_row_value_total_rows);
        if (expected_rows != 0 && leaf_row_values.size() < expected_rows) {
            throw std::runtime_error("leaf_row_values.bin count mismatch with manifest.");
        }
    }
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
        if (bucket.bitmap_begin + bucket.bitmap_len > leaf_bitmaps_.size()) {
            throw std::runtime_error("Bucket bitmap slice is out of bounds.");
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
    secret_host_payload_.leaf_row_values = doubles_to_fixed_column<kFlatBSPNDecimal>(leaf_row_values);
    secret_host_payload_.row_weights = doubles_to_fixed_column<kFlatBSPNDecimal>(row_weights_);
    secret_host_payload_.leaf_bucket_weight_sums =
        doubles_to_fixed_column<kFlatBSPNDecimal>(leaf_bucket_weight_sums_);
    secret_host_payload_.leaf_bitmaps = i64Matrix(0, 0);

    secret_shared_payload_ = FlatBSPNSecretSharedPayload{};
}

void FlatBSPNModel::load_secret_payload(const FlatBSPNSecureContext& context) {
    if (!context.has_runtime()) {
        throw std::runtime_error("FlatBSPNSecureContext runtime is not initialized.");
    }
    const std::string model_share_root = join_path(
        base_dir_,
        manifest_.model_secret_share_payload_dir.empty()
            ? std::string("secret_shares/model")
            : manifest_.model_secret_share_payload_dir);
    const std::string model_share_role_dir = join_path(model_share_root, "role_" + std::to_string(context.role));
    bool loaded_from_model_share_dir = false;
    if (manifest_.model_secret_share_payload_layout == "pair_matrix_v1" &&
        !base_dir_.empty() &&
        (path_exists(join_path(model_share_role_dir, "node_cardinalities.shares.bin")) ||
         path_exists(join_path(model_share_role_dir, "manifest.json")))) {
        const std::size_t node_count = static_cast<std::size_t>(manifest_.node_count);
        const std::size_t column_count = manifest_.column_names.size();
        const std::size_t bucket_count = static_cast<std::size_t>(manifest_.bucket_count);
        const std::size_t total_rows = static_cast<std::size_t>(manifest_.total_rows);
        const std::size_t leaf_row_value_rows =
            manifest_.has_leaf_row_values
                ? static_cast<std::size_t>(manifest_.leaf_row_value_node_ids.size()) *
                    static_cast<std::size_t>(std::max<std::uint64_t>(1, manifest_.leaf_row_value_total_rows))
                : 0;
        auto load_fixed = [&](const std::string& name, std::size_t rows, std::size_t cols) {
            return read_fixed_share_pair_matrix<kFlatBSPNDecimal>(join_path(model_share_role_dir, name), rows, cols);
        };
        auto load_int = [&](const std::string& name, std::size_t rows, std::size_t cols) {
            return read_int_share_pair_matrix(join_path(model_share_role_dir, name), rows, cols);
        };
        secret_shared_payload_.node_cardinalities = load_fixed("node_cardinalities.shares.bin", node_count, 1);
        secret_shared_payload_.node_inv_cardinalities = load_fixed("node_inv_cardinalities.shares.bin", node_count, 1);
        secret_shared_payload_.node_scopes = load_int("node_scopes.shares.bin", node_count, column_count);
        secret_shared_payload_.weights = load_fixed("weights.shares.bin", static_cast<std::size_t>(manifest_.weights_count), 1);
        secret_shared_payload_.bucket_values = load_fixed("bucket_values.shares.bin", bucket_count, 1);
        secret_shared_payload_.bucket_lowers = load_fixed("bucket_lowers.shares.bin", bucket_count, 1);
        secret_shared_payload_.bucket_uppers = load_fixed("bucket_uppers.shares.bin", bucket_count, 1);
        if (manifest_.has_row_weights) {
            secret_shared_payload_.row_weights = load_fixed(
                "row_weights.shares.bin",
                static_cast<std::size_t>(manifest_.row_weight_count),
                1);
            secret_shared_payload_.row_weights_loaded = true;
        } else {
            secret_shared_payload_.row_weights = sf64Matrix<kFlatBSPNDecimal>(0, 1);
            secret_shared_payload_.row_weights_loaded = false;
        }
        if (manifest_.has_leaf_bucket_weight_sums) {
            secret_shared_payload_.leaf_bucket_weight_sums = load_fixed(
                "leaf_bucket_weight_sums.shares.bin",
                static_cast<std::size_t>(manifest_.leaf_bucket_weight_sum_count),
                1);
            secret_shared_payload_.leaf_bucket_weight_sums_loaded = true;
        } else {
            secret_shared_payload_.leaf_bucket_weight_sums = sf64Matrix<kFlatBSPNDecimal>(0, 1);
            secret_shared_payload_.leaf_bucket_weight_sums_loaded = false;
        }
        if (manifest_.has_leaf_row_values) {
            secret_shared_payload_.leaf_row_values = load_fixed("leaf_row_values.shares.bin", leaf_row_value_rows, 1);
            secret_shared_payload_.leaf_row_value_offset_by_node.assign(node_count, -1);
            for (std::size_t idx = 0; idx < manifest_.leaf_row_value_node_ids.size(); ++idx) {
                const auto node_id = manifest_.leaf_row_value_node_ids[idx];
                if (node_id < secret_shared_payload_.leaf_row_value_offset_by_node.size()) {
                    secret_shared_payload_.leaf_row_value_offset_by_node[node_id] =
                        static_cast<std::int64_t>(manifest_.leaf_row_value_offsets[idx]);
                }
            }
            secret_shared_payload_.leaf_row_values_loaded = true;
        } else {
            secret_shared_payload_.leaf_row_values = sf64Matrix<kFlatBSPNDecimal>(0, 1);
            secret_shared_payload_.leaf_row_value_offset_by_node.clear();
            secret_shared_payload_.leaf_row_values_loaded = false;
        }
        secret_shared_payload_.leaf_bitmaps = si64Matrix(0, 0);
        secret_shared_payload_.dense_bucket_bitmaps.clear();
        secret_shared_payload_.dense_bucket_bitmaps.reserve(buckets_.size());
        const u64 kMaxBitmapShareRowsPerChunk = bspn_bitmap_share_rows_per_chunk();
        const u64 total_rows_u64 = static_cast<u64>(manifest_.total_rows);
        const u64 buckets_per_chunk =
            total_rows_u64 == 0
                ? 1
                : std::max<u64>(1, kMaxBitmapShareRowsPerChunk / std::max<u64>(1, total_rows_u64));
        if (bucket_count > 0 && total_rows_u64 > 0) {
            const std::string dense_bitmap_path = join_path(model_share_role_dir, "leaf_bitmaps.shares.bin");
            const auto shared_flat_bitmaps = read_bool_share_pair_column_bitpacked(
                dense_bitmap_path,
                bucket_count * static_cast<std::size_t>(total_rows_u64));
            for (std::size_t bucket_begin = 0; bucket_begin < buckets_.size(); bucket_begin += buckets_per_chunk) {
                const std::size_t bucket_end = std::min<std::size_t>(
                    buckets_.size(),
                    bucket_begin + static_cast<std::size_t>(buckets_per_chunk));
                const std::size_t chunk_bucket_count = bucket_end - bucket_begin;
                sbMatrix shared_chunk(static_cast<u64>(chunk_bucket_count) * total_rows_u64, shared_flat_bitmaps.bitCount());
                for (u64 row = 0; row < static_cast<u64>(chunk_bucket_count) * total_rows_u64; ++row) {
                    const u64 source_row = static_cast<u64>(bucket_begin) * total_rows_u64 + row;
                    for (u64 col = 0; col < static_cast<u64>(shared_flat_bitmaps.mShares[0].cols()); ++col) {
                        shared_chunk.mShares[0](row, col) = shared_flat_bitmaps.mShares[0](source_row, col);
                        shared_chunk.mShares[1](row, col) = shared_flat_bitmaps.mShares[1](source_row, col);
                    }
                }
                std::vector<sbMatrix> chunk_bitmaps;
                chunk_bitmaps.reserve(chunk_bucket_count);
                for (std::size_t local_bucket_idx = 0; local_bucket_idx < chunk_bucket_count; ++local_bucket_idx) {
                    chunk_bitmaps.emplace_back(total_rows_u64, shared_chunk.bitCount());
                }
                #pragma omp parallel for schedule(static)
                for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
                    const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
                    auto& shared_bitmap = chunk_bitmaps[local_bucket_idx];
                    const u64 row_begin = static_cast<u64>(local_bucket_idx) * total_rows_u64;
                    for (u64 row = 0; row < total_rows_u64; ++row) {
                        for (u64 col = 0; col < static_cast<u64>(shared_chunk.mShares[0].cols()); ++col) {
                            shared_bitmap.mShares[0](row, col) = shared_chunk.mShares[0](row_begin + row, col);
                            shared_bitmap.mShares[1](row, col) = shared_chunk.mShares[1](row_begin + row, col);
                        }
                    }
                }
                for (auto& shared_bitmap : chunk_bitmaps) {
                    secret_shared_payload_.dense_bucket_bitmaps.push_back(std::move(shared_bitmap));
                }
            }
        }
        secret_shared_payload_.dense_bucket_bitmaps_loaded = true;
        loaded_from_model_share_dir = true;
    }

    if (!loaded_from_model_share_dir) {
        if (remote_share_only_required()) {
            throw std::runtime_error(
                "Remote ABY3 share-only mode requires model secret role shares; "
                "plaintext model secret payload fallback is not allowed.");
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
        secret_host_payload_.row_weights = f64Matrix<kFlatBSPNDecimal>(
            manifest_.has_row_weights ? static_cast<u64>(manifest_.row_weight_count) : 0,
            1);
        secret_host_payload_.leaf_bucket_weight_sums = f64Matrix<kFlatBSPNDecimal>(
            manifest_.has_leaf_bucket_weight_sums ? static_cast<u64>(manifest_.leaf_bucket_weight_sum_count) : 0,
            1);
        const u64 leaf_row_value_rows =
            manifest_.has_leaf_row_values
                ? static_cast<u64>(manifest_.leaf_row_value_node_ids.size()) *
                    std::max<u64>(1, static_cast<u64>(manifest_.leaf_row_value_total_rows))
                : 0;
        secret_host_payload_.leaf_row_values = f64Matrix<kFlatBSPNDecimal>(leaf_row_value_rows, 1);
        secret_host_payload_.leaf_bitmaps = i64Matrix(0, 0);
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.node_cardinalities.size()); ++idx) secret_host_payload_.node_cardinalities(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.node_inv_cardinalities.size()); ++idx) secret_host_payload_.node_inv_cardinalities(idx) = 0;
        secret_host_payload_.node_scopes.setZero();
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.weights.size()); ++idx) secret_host_payload_.weights(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_values.size()); ++idx) secret_host_payload_.bucket_values(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_lowers.size()); ++idx) secret_host_payload_.bucket_lowers(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.bucket_uppers.size()); ++idx) secret_host_payload_.bucket_uppers(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.row_weights.size()); ++idx) secret_host_payload_.row_weights(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.leaf_bucket_weight_sums.size()); ++idx) secret_host_payload_.leaf_bucket_weight_sums(idx) = 0;
        for (u64 idx = 0; idx < static_cast<u64>(secret_host_payload_.leaf_row_values.size()); ++idx) secret_host_payload_.leaf_row_values(idx) = 0;
    }

    share_fixed_matrix(secret_host_payload_.node_cardinalities, secret_shared_payload_.node_cardinalities, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.node_inv_cardinalities, secret_shared_payload_.node_inv_cardinalities, context.model_owner_party, context);
    share_int_matrix(secret_host_payload_.node_scopes, secret_shared_payload_.node_scopes, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.weights, secret_shared_payload_.weights, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_values, secret_shared_payload_.bucket_values, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_lowers, secret_shared_payload_.bucket_lowers, context.model_owner_party, context);
    share_fixed_matrix(secret_host_payload_.bucket_uppers, secret_shared_payload_.bucket_uppers, context.model_owner_party, context);
    if (manifest_.has_row_weights) {
        share_fixed_matrix(secret_host_payload_.row_weights, secret_shared_payload_.row_weights, context.model_owner_party, context);
        secret_shared_payload_.row_weights_loaded = true;
    } else {
        secret_shared_payload_.row_weights = sf64Matrix<kFlatBSPNDecimal>(0, 1);
        secret_shared_payload_.row_weights_loaded = false;
    }
    if (manifest_.has_leaf_bucket_weight_sums) {
        share_fixed_matrix(
            secret_host_payload_.leaf_bucket_weight_sums,
            secret_shared_payload_.leaf_bucket_weight_sums,
            context.model_owner_party,
            context);
        secret_shared_payload_.leaf_bucket_weight_sums_loaded = true;
    } else {
        secret_shared_payload_.leaf_bucket_weight_sums = sf64Matrix<kFlatBSPNDecimal>(0, 1);
        secret_shared_payload_.leaf_bucket_weight_sums_loaded = false;
    }
    if (manifest_.has_leaf_row_values) {
        share_fixed_matrix(secret_host_payload_.leaf_row_values, secret_shared_payload_.leaf_row_values, context.model_owner_party, context);
        secret_shared_payload_.leaf_row_value_offset_by_node.assign(
            static_cast<std::size_t>(manifest_.node_count),
            -1);
        for (std::size_t idx = 0; idx < manifest_.leaf_row_value_node_ids.size(); ++idx) {
            const auto node_id = manifest_.leaf_row_value_node_ids[idx];
            if (node_id < secret_shared_payload_.leaf_row_value_offset_by_node.size()) {
                secret_shared_payload_.leaf_row_value_offset_by_node[node_id] =
                    static_cast<std::int64_t>(manifest_.leaf_row_value_offsets[idx]);
            }
        }
        secret_shared_payload_.leaf_row_values_loaded = true;
    } else {
        secret_shared_payload_.leaf_row_values = sf64Matrix<kFlatBSPNDecimal>(0, 1);
        secret_shared_payload_.leaf_row_value_offset_by_node.clear();
        secret_shared_payload_.leaf_row_values_loaded = false;
    }
    secret_shared_payload_.leaf_bitmaps = si64Matrix(0, 0);

    secret_shared_payload_.dense_bucket_bitmaps.clear();
    secret_shared_payload_.dense_bucket_bitmaps.reserve(buckets_.size());
    const u64 kMaxBitmapShareRowsPerChunk = bspn_bitmap_share_rows_per_chunk();
    const u64 total_rows = static_cast<u64>(manifest_.total_rows);
    const u64 buckets_per_chunk =
        total_rows == 0
            ? 1
            : std::max<u64>(1, kMaxBitmapShareRowsPerChunk / std::max<u64>(1, total_rows));
    for (std::size_t bucket_begin = 0; bucket_begin < buckets_.size(); bucket_begin += buckets_per_chunk) {
        const std::size_t bucket_end = std::min<std::size_t>(
            buckets_.size(),
            bucket_begin + static_cast<std::size_t>(buckets_per_chunk));
        const std::size_t chunk_bucket_count = bucket_end - bucket_begin;
        i64Matrix dense_rows(static_cast<u64>(chunk_bucket_count) * total_rows, 1);
        dense_rows.setZero();
        if (context.role == context.model_owner_party) {
            for (std::size_t local_bucket_idx = 0; local_bucket_idx < chunk_bucket_count; ++local_bucket_idx) {
                const auto& bucket = buckets_[bucket_begin + local_bucket_idx];
                if (bucket.bitmap_begin + bucket.bitmap_len > leaf_bitmaps_.size()) {
                    throw std::runtime_error("Leaf bitmap slice is out of bounds.");
                }
            }
            #pragma omp parallel for schedule(static)
            for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
                const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
                const auto& bucket = buckets_[bucket_begin + local_bucket_idx];
                const std::vector<std::uint8_t> bitmap_bytes(
                    leaf_bitmaps_.begin() + static_cast<std::ptrdiff_t>(bucket.bitmap_begin),
                    leaf_bitmaps_.begin() + static_cast<std::ptrdiff_t>(bucket.bitmap_begin + bucket.bitmap_len));
                unpack_bitmap_into_dense_rows(
                    bitmap_bytes,
                    manifest_.total_rows,
                    dense_rows,
                    static_cast<std::uint64_t>(local_bucket_idx) * manifest_.total_rows);
            }
        }

        sbMatrix shared_chunk;
        share_bool_matrix(dense_rows, shared_chunk, context.model_owner_party, context);
        std::vector<sbMatrix> chunk_bitmaps;
        chunk_bitmaps.reserve(chunk_bucket_count);
        for (std::size_t local_bucket_idx = 0; local_bucket_idx < chunk_bucket_count; ++local_bucket_idx) {
            chunk_bitmaps.emplace_back(total_rows, shared_chunk.bitCount());
        }
        #pragma omp parallel for schedule(static)
        for (std::int64_t local_bucket_idx_signed = 0; local_bucket_idx_signed < static_cast<std::int64_t>(chunk_bucket_count); ++local_bucket_idx_signed) {
            const auto local_bucket_idx = static_cast<std::size_t>(local_bucket_idx_signed);
            auto& shared_bitmap = chunk_bitmaps[local_bucket_idx];
            const u64 row_begin = static_cast<u64>(local_bucket_idx) * total_rows;
            for (u64 row = 0; row < total_rows; ++row) {
                for (u64 col = 0; col < static_cast<u64>(shared_chunk.mShares[0].cols()); ++col) {
                    shared_bitmap.mShares[0](row, col) = shared_chunk.mShares[0](row_begin + row, col);
                    shared_bitmap.mShares[1](row, col) = shared_chunk.mShares[1](row_begin + row, col);
                }
            }
        }
    for (auto& shared_bitmap : chunk_bitmaps) {
        secret_shared_payload_.dense_bucket_bitmaps.push_back(std::move(shared_bitmap));
    }
}
secret_shared_payload_.dense_bucket_bitmaps_loaded = true;
    }

    if (manifest_.secure_multiplier_materialized ||
        !manifest_.secure_multiplier_bucket_indices.empty()) {
        if (context.role < 0 || context.role > 2) {
            throw std::runtime_error("Secure multiplier share overlay requires role in {0,1,2}.");
        }
        const std::size_t secure_bucket_count = manifest_.secure_multiplier_bucket_indices.size();
        if (secure_bucket_count == 0) {
            throw std::runtime_error("Secure multiplier materialized payload has no bucket indices.");
        }
        const std::string share_root = join_path(base_dir_, manifest_.secure_share_payload_dir.empty()
            ? std::string("secret_shares")
            : manifest_.secure_share_payload_dir);
        if (manifest_.secure_share_payload_layout == "per_leaf_materialized_v1") {
            if (manifest_.secure_multiplier_bucket_mapping.size() != secure_bucket_count) {
                throw std::runtime_error("per_leaf_materialized_v1 requires secure_multiplier_bucket_mapping for every bucket.");
            }
            const std::string role_dir = join_path(share_root, "role_" + std::to_string(context.role));
            const std::size_t leaf_bucket_count = static_cast<std::size_t>(std::max<std::uint64_t>(
                1,
                manifest_.max_leaf_bucket_width == 0 ? manifest_.leaf_bucket_width : manifest_.max_leaf_bucket_width));
            const std::string bool_share_encoding = manifest_.secure_share_bool_encoding.empty()
                ? std::string("bitpacked_pair_lsb_v1")
                : manifest_.secure_share_bool_encoding;

            std::unordered_map<std::uint32_t, std::shared_ptr<sf64Matrix<kFlatBSPNDecimal>>> value_cache;
            std::unordered_map<std::uint32_t, std::shared_ptr<sf64Matrix<kFlatBSPNDecimal>>> lower_cache;
            std::unordered_map<std::uint32_t, std::shared_ptr<sf64Matrix<kFlatBSPNDecimal>>> upper_cache;
            std::unordered_map<std::uint32_t, std::shared_ptr<sbMatrix>> bitmap_cache;

            auto load_fixed_leaf = [&](std::unordered_map<std::uint32_t, std::shared_ptr<sf64Matrix<kFlatBSPNDecimal>>>& cache,
                                       std::uint32_t leaf_node_id,
                                       const std::string& suffix) -> sf64Matrix<kFlatBSPNDecimal>& {
                auto found = cache.find(leaf_node_id);
                if (found == cache.end()) {
                    const std::string path = join_path(
                        role_dir,
                        "leaf_" + std::to_string(leaf_node_id) + "." + suffix + ".shares.bin");
                    auto matrix = std::make_shared<sf64Matrix<kFlatBSPNDecimal>>(
                        read_fixed_share_pair_column<kFlatBSPNDecimal>(path, leaf_bucket_count));
                    found = cache.emplace(leaf_node_id, std::move(matrix)).first;
                }
                return *found->second;
            };

            auto load_bitmap_leaf = [&](std::uint32_t leaf_node_id) -> sbMatrix& {
                auto found = bitmap_cache.find(leaf_node_id);
                if (found == bitmap_cache.end()) {
                    const std::string path = join_path(
                        role_dir,
                        "leaf_" + std::to_string(leaf_node_id) + ".leaf_bitmaps.shares.bin");
                    std::shared_ptr<sbMatrix> matrix;
                    if (bool_share_encoding == "bitpacked_pair_lsb_v1") {
                        matrix = std::make_shared<sbMatrix>(read_bool_share_pair_column_bitpacked(
                            path,
                            leaf_bucket_count * static_cast<std::size_t>(manifest_.total_rows)));
                    } else if (bool_share_encoding == "i64_pair_v1" || bool_share_encoding.empty()) {
                        matrix = std::make_shared<sbMatrix>(read_bool_share_pair_column(
                            path,
                            leaf_bucket_count * static_cast<std::size_t>(manifest_.total_rows)));
                    } else {
                        throw std::runtime_error("Unsupported secure_share_bool_encoding in manifest: " + bool_share_encoding);
                    }
                    found = bitmap_cache.emplace(leaf_node_id, std::move(matrix)).first;
                }
                return *found->second;
            };

            for (std::size_t local_idx = 0; local_idx < secure_bucket_count; ++local_idx) {
                const auto& mapping = manifest_.secure_multiplier_bucket_mapping[local_idx];
                const std::uint32_t bucket_index = manifest_.secure_multiplier_bucket_indices[local_idx];
                if (mapping.global_bucket_id != bucket_index) {
                    throw std::runtime_error("Secure multiplier per-leaf mapping bucket id mismatch.");
                }
                if (bucket_index >= secret_shared_payload_.dense_bucket_bitmaps.size() ||
                    bucket_index >= static_cast<std::uint32_t>(secret_shared_payload_.bucket_values.rows())) {
                    throw std::runtime_error("Secure multiplier bucket index is out of bounds.");
                }
                if (mapping.leaf_local_bucket_idx >= leaf_bucket_count) {
                    throw std::runtime_error("Secure multiplier per-leaf bucket index is out of bounds.");
                }

                auto& shared_values = load_fixed_leaf(value_cache, mapping.leaf_node_id, "bucket_values");
                auto& shared_lowers = load_fixed_leaf(lower_cache, mapping.leaf_node_id, "bucket_lowers");
                auto& shared_uppers = load_fixed_leaf(upper_cache, mapping.leaf_node_id, "bucket_uppers");
                secret_shared_payload_.bucket_values[0](bucket_index, 0) =
                    shared_values[0](static_cast<u64>(mapping.leaf_local_bucket_idx), 0);
                secret_shared_payload_.bucket_values[1](bucket_index, 0) =
                    shared_values[1](static_cast<u64>(mapping.leaf_local_bucket_idx), 0);
                secret_shared_payload_.bucket_lowers[0](bucket_index, 0) =
                    shared_lowers[0](static_cast<u64>(mapping.leaf_local_bucket_idx), 0);
                secret_shared_payload_.bucket_lowers[1](bucket_index, 0) =
                    shared_lowers[1](static_cast<u64>(mapping.leaf_local_bucket_idx), 0);
                secret_shared_payload_.bucket_uppers[0](bucket_index, 0) =
                    shared_uppers[0](static_cast<u64>(mapping.leaf_local_bucket_idx), 0);
                secret_shared_payload_.bucket_uppers[1](bucket_index, 0) =
                    shared_uppers[1](static_cast<u64>(mapping.leaf_local_bucket_idx), 0);

                auto& leaf_bitmaps = load_bitmap_leaf(mapping.leaf_node_id);
                sbMatrix bitmap(static_cast<u64>(manifest_.total_rows), leaf_bitmaps.bitCount());
                const u64 source_begin =
                    static_cast<u64>(mapping.leaf_local_bucket_idx) * static_cast<u64>(manifest_.total_rows);
                for (u64 row = 0; row < static_cast<u64>(manifest_.total_rows); ++row) {
                    for (u64 col = 0; col < static_cast<u64>(leaf_bitmaps.mShares[0].cols()); ++col) {
                        bitmap.mShares[0](row, col) = leaf_bitmaps.mShares[0](source_begin + row, col);
                        bitmap.mShares[1](row, col) = leaf_bitmaps.mShares[1](source_begin + row, col);
                    }
                }
                secret_shared_payload_.dense_bucket_bitmaps[bucket_index] = std::move(bitmap);
            }

            if (secret_shared_payload_.leaf_row_values_loaded &&
                !manifest_.secure_multiplier_leaf_row_value_node_ids.empty()) {
                if (manifest_.secure_multiplier_leaf_row_value_node_ids.size() !=
                    manifest_.secure_multiplier_leaf_row_value_offsets.size()) {
                    throw std::runtime_error("Secure multiplier row-value node/offset manifest mismatch.");
                }
                for (std::size_t local_idx = 0;
                     local_idx < manifest_.secure_multiplier_leaf_row_value_node_ids.size();
                     ++local_idx) {
                    const auto node_id = manifest_.secure_multiplier_leaf_row_value_node_ids[local_idx];
                    if (node_id >= secret_shared_payload_.leaf_row_value_offset_by_node.size() ||
                        secret_shared_payload_.leaf_row_value_offset_by_node[node_id] < 0) {
                        throw std::runtime_error("Secure multiplier row-value node is missing in leaf row value payload.");
                    }
                    const std::string row_value_path = join_path(
                        role_dir,
                        "leaf_" + std::to_string(node_id) + ".row_values.shares.bin");
                    if (!path_exists(row_value_path)) {
                        throw std::runtime_error("Secure multiplier leaf row value shares are missing: " + row_value_path);
                    }
                    const auto shared_row_values = read_fixed_share_pair_column<kFlatBSPNDecimal>(
                        row_value_path,
                        static_cast<std::size_t>(manifest_.secure_multiplier_leaf_row_value_total_rows));
                    const u64 dst_begin =
                        static_cast<u64>(secret_shared_payload_.leaf_row_value_offset_by_node[node_id]);
                    for (u64 row = 0; row < static_cast<u64>(manifest_.secure_multiplier_leaf_row_value_total_rows); ++row) {
                        secret_shared_payload_.leaf_row_values[0](dst_begin + row, 0) =
                            shared_row_values[0](row, 0);
                        secret_shared_payload_.leaf_row_values[1](dst_begin + row, 0) =
                            shared_row_values[1](row, 0);
                    }
                }
            }
        } else {
        std::string bool_share_encoding = "i64_pair_v1";
        const std::string share_manifest_path = join_path(share_root, "manifest.json");
        if (path_exists(share_manifest_path)) {
            std::ifstream share_manifest_in(share_manifest_path);
            json share_manifest_doc;
            share_manifest_in >> share_manifest_doc;
            bool_share_encoding = share_manifest_doc.value("bool_share_encoding", std::string("i64_pair_v1"));
        }
        const std::string role_dir = join_path(share_root, "role_" + std::to_string(context.role));
        const auto shared_values = read_fixed_share_pair_column<kFlatBSPNDecimal>(
            join_path(role_dir, "bucket_values.shares.bin"),
            secure_bucket_count);
        const auto shared_lowers = read_fixed_share_pair_column<kFlatBSPNDecimal>(
            join_path(role_dir, "bucket_lowers.shares.bin"),
            secure_bucket_count);
        const auto shared_uppers = read_fixed_share_pair_column<kFlatBSPNDecimal>(
            join_path(role_dir, "bucket_uppers.shares.bin"),
            secure_bucket_count);
        sbMatrix shared_bitmaps;
        const std::string bitmap_share_path = join_path(role_dir, "leaf_bitmaps.shares.bin");
        if (bool_share_encoding == "bitpacked_pair_lsb_v1") {
            shared_bitmaps = read_bool_share_pair_column_bitpacked(
                bitmap_share_path,
                secure_bucket_count * static_cast<std::size_t>(manifest_.total_rows));
        } else if (bool_share_encoding == "i64_pair_v1" || bool_share_encoding.empty()) {
            shared_bitmaps = read_bool_share_pair_column(
                bitmap_share_path,
                secure_bucket_count * static_cast<std::size_t>(manifest_.total_rows));
        } else {
            throw std::runtime_error("Unsupported bool_share_encoding in secure share manifest: " + bool_share_encoding);
        }

        for (std::size_t local_idx = 0; local_idx < secure_bucket_count; ++local_idx) {
            const std::uint32_t bucket_index = manifest_.secure_multiplier_bucket_indices[local_idx];
            if (bucket_index >= secret_shared_payload_.dense_bucket_bitmaps.size() ||
                bucket_index >= static_cast<std::uint32_t>(secret_shared_payload_.bucket_values.rows())) {
                throw std::runtime_error("Secure multiplier bucket index is out of bounds.");
            }
            secret_shared_payload_.bucket_values[0](bucket_index, 0) = shared_values[0](static_cast<u64>(local_idx), 0);
            secret_shared_payload_.bucket_values[1](bucket_index, 0) = shared_values[1](static_cast<u64>(local_idx), 0);
            secret_shared_payload_.bucket_lowers[0](bucket_index, 0) = shared_lowers[0](static_cast<u64>(local_idx), 0);
            secret_shared_payload_.bucket_lowers[1](bucket_index, 0) = shared_lowers[1](static_cast<u64>(local_idx), 0);
            secret_shared_payload_.bucket_uppers[0](bucket_index, 0) = shared_uppers[0](static_cast<u64>(local_idx), 0);
            secret_shared_payload_.bucket_uppers[1](bucket_index, 0) = shared_uppers[1](static_cast<u64>(local_idx), 0);

            sbMatrix bitmap(static_cast<u64>(manifest_.total_rows), shared_bitmaps.bitCount());
            const u64 source_begin = static_cast<u64>(local_idx) * static_cast<u64>(manifest_.total_rows);
            for (u64 row = 0; row < static_cast<u64>(manifest_.total_rows); ++row) {
                for (u64 col = 0; col < static_cast<u64>(shared_bitmaps.mShares[0].cols()); ++col) {
                    bitmap.mShares[0](row, col) = shared_bitmaps.mShares[0](source_begin + row, col);
                    bitmap.mShares[1](row, col) = shared_bitmaps.mShares[1](source_begin + row, col);
                }
            }
            secret_shared_payload_.dense_bucket_bitmaps[bucket_index] = std::move(bitmap);
        }
        if (secret_shared_payload_.leaf_row_values_loaded &&
            !manifest_.secure_multiplier_leaf_row_value_node_ids.empty()) {
            if (manifest_.secure_multiplier_leaf_row_value_node_ids.size() !=
                manifest_.secure_multiplier_leaf_row_value_offsets.size()) {
                throw std::runtime_error("Secure multiplier row-value node/offset manifest mismatch.");
            }
            const u64 row_value_rows = static_cast<u64>(
                manifest_.secure_multiplier_leaf_row_value_node_ids.size()) *
                static_cast<u64>(manifest_.secure_multiplier_leaf_row_value_total_rows);
            const std::string row_value_path = join_path(role_dir, "leaf_row_values.shares.bin");
            if (!path_exists(row_value_path)) {
                throw std::runtime_error("Secure multiplier leaf row value shares are missing: " + row_value_path);
            }
            const auto shared_row_values = read_fixed_share_pair_column<kFlatBSPNDecimal>(
                row_value_path,
                row_value_rows);
            for (std::size_t local_idx = 0;
                 local_idx < manifest_.secure_multiplier_leaf_row_value_node_ids.size();
                 ++local_idx) {
                const auto node_id = manifest_.secure_multiplier_leaf_row_value_node_ids[local_idx];
                if (node_id >= secret_shared_payload_.leaf_row_value_offset_by_node.size() ||
                    secret_shared_payload_.leaf_row_value_offset_by_node[node_id] < 0) {
                    throw std::runtime_error("Secure multiplier row-value node is missing in leaf row value payload.");
                }
                const u64 dst_begin =
                    static_cast<u64>(secret_shared_payload_.leaf_row_value_offset_by_node[node_id]);
                const u64 src_begin = manifest_.secure_multiplier_leaf_row_value_offsets[local_idx];
                for (u64 row = 0; row < static_cast<u64>(manifest_.secure_multiplier_leaf_row_value_total_rows); ++row) {
                    secret_shared_payload_.leaf_row_values[0](dst_begin + row, 0) =
                        shared_row_values[0](src_begin + row, 0);
                    secret_shared_payload_.leaf_row_values[1](dst_begin + row, 0) =
                        shared_row_values[1](src_begin + row, 0);
                }
            }
        }
        }
        secure_share_payload_loaded_ = true;
    }
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
    const std::size_t max_interval_count = static_cast<std::size_t>(payload.max_interval_count);
    const std::size_t factor_count = static_cast<std::size_t>(payload.factor_count);
    const std::size_t max_factor_columns = static_cast<std::size_t>(payload.max_factor_column_count);
    if (payload.binding_layout_kind != "DENSE_FACTOR_COLUMNS_V1") {
        throw std::runtime_error("Secure query payload tensors require DENSE_FACTOR_COLUMNS_V1.");
    }
    const std::size_t evidence_row_count = factor_count * max_factor_columns;

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

    return tensors;
}

FlatSecureQueryPayload empty_secure_query_payload_from_public_doc(const json& public_doc) {
    FlatSecureQueryPayload payload;
    payload.query_skeleton_id = public_doc.value("query_skeleton_id", std::string());
    payload.binding_layout_kind = public_doc.value("binding_layout_kind", std::string());
    if (payload.binding_layout_kind != "DENSE_FACTOR_COLUMNS_V1") {
        throw std::runtime_error("Public secure plan must use DENSE_FACTOR_COLUMNS_V1.");
    }
    if (public_doc.contains("secret_tensor_shape") && public_doc["secret_tensor_shape"].is_object()) {
        const auto& shape_doc = public_doc["secret_tensor_shape"];
        if (shape_doc.contains("factor_payload_shape") && shape_doc["factor_payload_shape"].is_object()) {
            payload.factor_count = shape_doc["factor_payload_shape"].value("factor_count", std::uint64_t(0));
            payload.max_factor_column_count = shape_doc["factor_payload_shape"].value("max_column_count", std::uint64_t(0));
            payload.max_interval_count = shape_doc["factor_payload_shape"].value("max_interval_count", std::uint64_t(0));
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

FlatSecureQueryTensorPayload load_secure_query_tensor_payload_shares(
    const json& public_doc,
    const std::string& query_share_payload_dir,
    const FlatBSPNSecureContext& context) {
    if (!context.has_runtime()) {
        throw std::runtime_error("FlatBSPNSecureContext runtime is not initialized.");
    }
    if (query_share_payload_dir.empty()) {
        throw std::runtime_error("query_share_payload_dir not set");
    }

    const FlatSecureQueryPayload shape_payload = empty_secure_query_payload_from_public_doc(public_doc);
    FlatSecureQueryTensorPayload tensors = build_secure_query_tensor_payload(shape_payload);
    const std::string role_dir = join_path(query_share_payload_dir, "role_" + std::to_string(context.role));
    if (!path_exists(role_dir)) {
        throw std::runtime_error("Could not open query share payload dir: " + role_dir);
    }

    const std::size_t factor_rows =
        static_cast<std::size_t>(shape_payload.factor_count) *
        static_cast<std::size_t>(shape_payload.max_factor_column_count);
    tensors.lower_bounds_shared = read_fixed_share_pair_matrix<kFlatBSPNDecimal>(
        join_path(role_dir, "lower_bounds.shares.bin"),
        factor_rows,
        static_cast<std::size_t>(shape_payload.max_interval_count));
    tensors.upper_bounds_shared = read_fixed_share_pair_matrix<kFlatBSPNDecimal>(
        join_path(role_dir, "upper_bounds.shares.bin"),
        factor_rows,
        static_cast<std::size_t>(shape_payload.max_interval_count));
    tensors.has_lower_shared = read_int_share_pair_matrix(
        join_path(role_dir, "has_lower.shares.bin"),
        factor_rows,
        static_cast<std::size_t>(shape_payload.max_interval_count));
    tensors.has_upper_shared = read_int_share_pair_matrix(
        join_path(role_dir, "has_upper.shares.bin"),
        factor_rows,
        static_cast<std::size_t>(shape_payload.max_interval_count));
    tensors.open_lower_shared = read_int_share_pair_matrix(
        join_path(role_dir, "open_lower.shares.bin"),
        factor_rows,
        static_cast<std::size_t>(shape_payload.max_interval_count));
    tensors.open_upper_shared = read_int_share_pair_matrix(
        join_path(role_dir, "open_upper.shares.bin"),
        factor_rows,
        static_cast<std::size_t>(shape_payload.max_interval_count));
    tensors.has_evidence_shared = read_int_share_pair_matrix(
        join_path(role_dir, "has_evidence.shares.bin"),
        factor_rows,
        1);
    tensors.interval_counts_shared = read_int_share_pair_matrix(
        join_path(role_dir, "interval_counts.shares.bin"),
        factor_rows,
        1);
    tensors.feature_scope_shared = read_int_share_pair_matrix(
        join_path(role_dir, "feature_scope.shares.bin"),
        static_cast<std::size_t>(shape_payload.factor_count),
        static_cast<std::size_t>(shape_payload.max_factor_column_count));
    tensors.relevant_scope_shared = read_int_share_pair_matrix(
        join_path(role_dir, "relevant_scope.shares.bin"),
        static_cast<std::size_t>(shape_payload.factor_count),
        static_cast<std::size_t>(shape_payload.max_factor_column_count));
    tensors.feature_inverted_scope_shared = read_int_share_pair_matrix(
        join_path(role_dir, "feature_inverted_scope.shares.bin"),
        static_cast<std::size_t>(shape_payload.factor_count),
        static_cast<std::size_t>(shape_payload.max_factor_column_count));
    tensors.factor_column_counts_shared = read_int_share_pair_matrix(
        join_path(role_dir, "factor_column_counts.shares.bin"),
        static_cast<std::size_t>(shape_payload.factor_count),
        1);
    tensors.shared_loaded = true;
    return tensors;
}

json evaluate_secure_bundle_paths(
    const std::string& public_plan_path,
    const std::string& secret_payload_path,
    const std::string& query_share_payload_dir,
    const std::string& model_root,
    const std::map<std::string, std::string>& manifest_map,
    FlatBSPNSecureContext& secure_context,
    std::map<std::string, FlatBSPNModel>& preloaded_model_cache) {
    if (public_plan_path.empty()) {
        throw std::runtime_error("public_plan_json not set");
    }
    if (secret_payload_path.empty() && query_share_payload_dir.empty()) {
        throw std::runtime_error("secret_payload_json not set");
    }
    if (remote_share_only_required() && query_share_payload_dir.empty()) {
        throw std::runtime_error(
            "Remote ABY3 share-only mode requires query_share_payload_dir; "
            "plaintext secret_payload_json is not allowed.");
    }
    if (remote_share_only_required() && !secret_payload_path.empty()) {
        throw std::runtime_error(
            "Remote ABY3 share-only mode forbids plaintext secret_payload_json.");
    }

    const auto path_total_start = std::chrono::steady_clock::now();
    json stage_timing_ms = json::object();

    auto phase_start = std::chrono::steady_clock::now();
    std::ifstream public_in(public_plan_path);
    if (!public_in.is_open()) {
        throw std::runtime_error("Could not open public plan json: " + public_plan_path);
    }
    json public_doc;
    public_in >> public_doc;
    stage_timing_ms["public_plan_read"] = elapsed_ms_since(phase_start);

    FlatSecureQueryPayload secure_payload = empty_secure_query_payload_from_public_doc(public_doc);
    FlatSecureQueryTensorPayload shared_query_payload;
    if (!query_share_payload_dir.empty()) {
        phase_start = std::chrono::steady_clock::now();
        shared_query_payload = load_secure_query_tensor_payload_shares(
            public_doc,
            query_share_payload_dir,
            secure_context);
        stage_timing_ms["secret_payload_read"] = 0.0;
        stage_timing_ms["query_payload_share"] = elapsed_ms_since(phase_start);
    } else {
        json secret_doc = json::object();
        if (secure_context.role == secure_context.query_owner_party) {
            phase_start = std::chrono::steady_clock::now();
            std::ifstream secret_in(secret_payload_path);
            if (!secret_in.is_open()) {
                throw std::runtime_error("Could not open secret payload json: " + secret_payload_path);
            }
            secret_in >> secret_doc;
            secure_payload = parse_secure_query_payload_doc(secret_doc);
            stage_timing_ms["secret_payload_read"] = elapsed_ms_since(phase_start);
        } else {
            stage_timing_ms["secret_payload_read"] = 0.0;
        }

        phase_start = std::chrono::steady_clock::now();
        shared_query_payload = share_secure_query_tensor_payload(secure_payload, secure_context);
        stage_timing_ms["query_payload_share"] = elapsed_ms_since(phase_start);
    }

    phase_start = std::chrono::steady_clock::now();
    std::uint64_t model_cache_hits = 0;
    std::uint64_t model_cache_misses = 0;
    for (const auto& model_id : collect_secure_bundle_model_ids(public_doc)) {
        const auto manifest_it = manifest_map.find(model_id);
        const std::string manifest_path =
            manifest_it != manifest_map.end()
                ? manifest_it->second
                : default_manifest_path_for_model(model_root, model_id);
        if (preloaded_model_cache.find(manifest_path) != preloaded_model_cache.end()) {
            ++model_cache_hits;
            continue;
        }
        ++model_cache_misses;
        FlatBSPNModel model;
        model.load_public_manifest(manifest_path);
        model.load_secret_payload(secure_context);
        preloaded_model_cache.emplace(manifest_path, std::move(model));
    }
    stage_timing_ms["model_load"] = elapsed_ms_since(phase_start);

    phase_start = std::chrono::steady_clock::now();
    synchronize_secure_parties(secure_context);
    stage_timing_ms["pre_eval_synchronize"] = elapsed_ms_since(phase_start);
    const auto secure_total_start = std::chrono::steady_clock::now();
    const auto secure_core_start = std::chrono::steady_clock::now();
    auto secure_eval = evaluate_secure_bundle_impl_secure(
        public_doc,
        secure_payload,
        shared_query_payload,
        manifest_map,
        model_root,
        secure_context,
        preloaded_model_cache);
    SecureFixedScalarShare secure_result_scalar;
    if (secure_context.factor_trace_shares && secure_eval.has_result) {
        secure_eval.factor_trace_shares.push_back({
            {"stage", "root_input"},
            {"profile_section", "root"},
            {"factor_index", -1},
            {"factor_kind", "ROOT_RATIONAL"},
            {"rational_share", secure_rational_share_json(secure_eval.result_rational)},
        });
    }
    if (secure_eval.has_result) {
        phase_start = std::chrono::steady_clock::now();
        secure_result_scalar = secure_divide_rational_to_fixed_scalar(
            secure_eval.result_rational,
            secure_context,
            secure_eval.root_division_payload_scale,
            secure_eval.root_division_scale_denominator_payload);
        stage_timing_ms["root_division"] = elapsed_ms_since(phase_start);
        if (secure_context.factor_trace_shares) {
            secure_eval.factor_trace_shares.push_back({
                {"stage", "final_scalar"},
                {"profile_section", "root"},
                {"factor_index", -1},
                {"factor_kind", "FINAL_SCALAR"},
                {"scalar_share", secure_fixed_scalar_share_json(secure_result_scalar)},
                {"root_division_payload_scale", secure_eval.root_division_payload_scale},
                {"root_division_scale_denominator_payload", secure_eval.root_division_scale_denominator_payload},
            });
        }
    } else {
        stage_timing_ms["root_division"] = 0.0;
    }
    const auto secure_core_end = std::chrono::steady_clock::now();
    bool result_revealed = false;
    double revealed_result = 0.0;
    double final_reveal_wall_time_ms = 0.0;
    if (secure_context.debug_reveal && secure_eval.has_result) {
        const auto reveal_start = std::chrono::steady_clock::now();
        revealed_result = reveal_scaled_fixed_scalar(secure_result_scalar, secure_context);
        result_revealed = true;
        const auto reveal_end = std::chrono::steady_clock::now();
        final_reveal_wall_time_ms =
            std::chrono::duration<double, std::milli>(reveal_end - reveal_start).count();
    }
    phase_start = std::chrono::steady_clock::now();
    synchronize_secure_parties(secure_context);
    stage_timing_ms["post_eval_synchronize"] = elapsed_ms_since(phase_start);
    const auto secure_total_end = std::chrono::steady_clock::now();
    const double secure_core_wall_time_ms =
        std::chrono::duration<double, std::milli>(secure_core_end - secure_core_start).count();
    const double secure_total_wall_time_ms =
        std::chrono::duration<double, std::milli>(secure_total_end - secure_total_start).count();

    json out = {
        {"query_skeleton_id", public_doc.value("query_skeleton_id", std::string())},
        {"query_kind", public_doc.value("query_kind", std::string())},
        {"secure_evaluator_core_wall_time_ms", secure_core_wall_time_ms},
        {"secure_evaluator_final_reveal_wall_time_ms", final_reveal_wall_time_ms},
        {"secure_evaluator_total_wall_time_ms", secure_total_wall_time_ms},
        {"secure_evaluator_wall_time_ms", secure_core_wall_time_ms},
        {"secure_evaluator_synchronized_wall_time_ms", secure_total_wall_time_ms},
        {"debug_internal_reveal", secure_context.debug_internal_reveal},
        {"result", nullptr},
    };

    if (result_revealed) {
        out["result"] = revealed_result;
    }
    if (secure_context.debug_internal_reveal) {
        out["debug"] = secure_eval.debug_output;
    }
    if (secure_context.factor_trace_shares) {
        out["factor_trace_shares"] = secure_eval.factor_trace_shares;
    }
    stage_timing_ms["secure_core"] = secure_core_wall_time_ms;
    stage_timing_ms["final_reveal"] = final_reveal_wall_time_ms;
    stage_timing_ms["secure_total_synchronized"] = secure_total_wall_time_ms;
    stage_timing_ms["path_total"] = elapsed_ms_since(path_total_start);
    secure_eval.timing_profile["stage_timing_ms"] = stage_timing_ms;
    secure_eval.timing_profile["model_cache_hits"] = model_cache_hits;
    secure_eval.timing_profile["model_cache_misses"] = model_cache_misses;
    out["timing_profile"] = secure_eval.timing_profile;
    out["runtime_params"] = {
        {"BSPN_MAX_STACKED_BITMAP_ROWS", bspn_max_stacked_bitmap_rows()},
        {"BSPN_BITMAP_SHARE_ROWS_PER_CHUNK", bspn_bitmap_share_rows_per_chunk()},
        {"BSPN_USE_ROW_VALUE_EVAL", bspn_use_row_value_eval()},
        {"OMP_MAX_THREADS", bspn_openmp_max_threads()},
        {"IO_SERVICE_THREAD_DEFAULT", std::thread::hardware_concurrency()},
    };
    out["model_cache_size"] = preloaded_model_cache.size();
    return out;
}

std::map<std::string, std::string> load_manifest_map_from_cmd(const oc::CLP& cmd) {
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
    return manifest_map;
}

std::string resolve_bundle_path(const std::string& base_dir, const json& doc, const std::string& key) {
    const std::string path = doc.value(key, std::string());
    if (path.empty()) {
        throw std::runtime_error("batch bundle entry missing " + key);
    }
    return join_path(base_dir, path);
}

void BSPN_flat_eval(const oc::CLP& cmd) {
    const std::string public_plan_path = cmd.getOr<std::string>("public_plan_json", "");
    const std::string secret_payload_path = cmd.getOr<std::string>("secret_payload_json", "");
    const std::string query_share_payload_dir = cmd.getOr<std::string>("query_share_payload_dir", "");
    const std::string batch_bundle_path = cmd.getOr<std::string>("batch_bundle_json", "");
    if (batch_bundle_path.empty() &&
        (public_plan_path.empty() || (secret_payload_path.empty() && query_share_payload_dir.empty()))) {
        throw std::runtime_error(
            "bspn_flat_eval now supports the secure bundle contract: "
            "--role, --public_plan_json, --query_share_payload_dir or --secret_payload_json, and --bspn_model_root; "
            "or --role, --batch_bundle_json, and --bspn_model_root.");
    }
    BSPN_secure_bundle_eval(cmd);
}

void BSPN_secure_bundle_eval(const oc::CLP& cmd) {
    if (cmd.isSet("debug_bundle_rational") || cmd.isSet("debug_factor_rational") || cmd.isSet("debug_oblivious_trace")) {
        throw std::runtime_error("Plaintext/debug FlatBSPN evaluators have been removed from the production frontend.");
    }

    oc::IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    FlatBSPNSecureContext secure_context;
    init_secure_context_from_cmd(cmd, secure_context, ios, enc, eval, runtime);

    const std::string model_root = normalize_model_root(cmd.getOr<std::string>("bspn_model_root", ""));
    const std::map<std::string, std::string> manifest_map = load_manifest_map_from_cmd(cmd);
    std::map<std::string, FlatBSPNModel> preloaded_model_cache;

    const std::string batch_bundle_path = cmd.getOr<std::string>("batch_bundle_json", "");
    if (!batch_bundle_path.empty()) {
        const auto batch_total_start = std::chrono::steady_clock::now();
        const auto batch_read_start = std::chrono::steady_clock::now();
        std::ifstream batch_in(batch_bundle_path);
        if (!batch_in.is_open()) {
            throw std::runtime_error("Could not open batch bundle json: " + batch_bundle_path);
        }
        json batch_doc;
        batch_in >> batch_doc;
        const double batch_read_ms = elapsed_ms_since(batch_read_start);
        const json entries = batch_doc.is_array() ? batch_doc : batch_doc.value("queries", json::array());
        if (!entries.is_array()) {
            throw std::runtime_error("batch_bundle_json must be an array or an object with a queries array.");
        }
        const std::string batch_base_dir = dirname_from_path(batch_bundle_path);
        json batch_results = json::array();
        std::size_t batch_index = 0;
        for (const auto& entry : entries) {
            if (remote_share_only_required() && entry.contains("secret_payload_json")) {
                throw std::runtime_error(
                    "Remote ABY3 share-only mode forbids batch secret_payload_json entries.");
            }
            const std::string entry_public_plan = resolve_bundle_path(batch_base_dir, entry, "public_plan_json");
            const std::string entry_secret_payload = entry.contains("secret_payload_json")
                ? resolve_bundle_path(batch_base_dir, entry, "secret_payload_json")
                : std::string();
            const std::string entry_query_share_payload_dir = entry.value("query_share_payload_dir", std::string());
            if (remote_share_only_required() && entry_query_share_payload_dir.empty()) {
                throw std::runtime_error(
                    "Remote ABY3 share-only mode requires batch query_share_payload_dir entries.");
            }
            const std::string entry_query_share_payload_dir_resolved = entry_query_share_payload_dir.empty()
                ? std::string()
                : join_path(batch_base_dir, entry_query_share_payload_dir);
            auto result = evaluate_secure_bundle_paths(
                entry_public_plan,
                entry_secret_payload,
                entry_query_share_payload_dir_resolved,
                model_root,
                manifest_map,
                secure_context,
                preloaded_model_cache);
            result["batch_index"] = entry.value("batch_index", static_cast<std::uint64_t>(batch_index));
            result["query_label"] = entry.value("query_label", std::string());
            batch_results.push_back(std::move(result));
            ++batch_index;
        }
        json out = {
            {"batch_results", batch_results},
            {"batch_query_count", batch_results.size()},
            {"persistent_frontend_eval", true},
            {"model_cache_size", preloaded_model_cache.size()},
            {"batch_bundle_read_ms", batch_read_ms},
            {"batch_total_wall_time_ms", elapsed_ms_since(batch_total_start)},
            {"runtime_params", {
                {"OMP_MAX_THREADS", bspn_openmp_max_threads()},
                {"IO_SERVICE_THREAD_DEFAULT", std::thread::hardware_concurrency()},
            }},
        };
        std::cout << out.dump(2) << std::endl;
        return;
    }

    const std::string public_plan_path = cmd.getOr<std::string>("public_plan_json", "");
    const std::string secret_payload_path = cmd.getOr<std::string>("secret_payload_json", "");
    const std::string query_share_payload_dir = cmd.getOr<std::string>("query_share_payload_dir", "");
    auto out = evaluate_secure_bundle_paths(
        public_plan_path,
        secret_payload_path,
        query_share_payload_dir,
        model_root,
        manifest_map,
        secure_context,
        preloaded_model_cache);
    out["persistent_frontend_eval"] = false;
    std::cout << out.dump(2) << std::endl;
}

}  // namespace aby3

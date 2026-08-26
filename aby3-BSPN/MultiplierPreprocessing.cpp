#include "MultiplierPreprocessing.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../aby3-Basic/BuildingBlocks.h"
#include "../aby3-GORAM-Core/Basics.h"
#include "../aby3-GORAM-Core/SqrtOram.h"
#include "../aby3-GORAM-Core/Sort.h"
#include <nlohmann/json.hpp>

using namespace aby3;
using namespace oc;

namespace {

using json = nlohmann::json;

constexpr int kMultiplierFixedDecimalBits = 16;
constexpr i64 kMultiplierFixedOne = static_cast<i64>(1) << kMultiplierFixedDecimalBits;
constexpr int kSecureLeafMaterializerVersion = 2;

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

enum class SecureLeafProfileLevel {
    None = 0,
    Group = 1,
    Batch = 2,
    Leaf = 3,
};

struct NormalizedKey {
    i64 value = 0;
    bool is_null = true;
};

struct MultiplierPreprocessConfig {
    std::string pk_csv_path;
    std::string fk_csv_path;
    std::string secure_leaf_plan_path;
    u64 pk_key_column = 0;
    u64 fk_key_column = 0;
    double fk_sample_rate = 1.0;
    std::string output_prefix;
    std::string relationship_id;
    bool pk_has_header = false;
    bool fk_has_header = false;
    std::string mode = "secure_shared_values";
    int role = -1;
    int input_party = 0;
    int pk_input_party = 0;
    int fk_input_party = 1;
    u64 secure_sort_min_size = 32;
};

bool env_flag_enabled(const char* name, bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    return true;
}

bool secure_multiplier_fast_preprocess_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_FAST_PREPROCESS", true);
}

bool secure_multiplier_provider_separated_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_PROVIDER_SEPARATED", false);
}

bool secure_leaf_sorted_group_enabled()
{
    const char* value = std::getenv("SECURE_LEAF_MATERIALIZE_STRATEGY");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string normalized(value);
    return normalized == "sorted_group" || normalized == "sort_group" || normalized == "grouped_sort";
}

std::string secure_leaf_materialize_strategy_name()
{
    return secure_leaf_sorted_group_enabled() ? "sorted_group" : "fixed_cap";
}

SecureLeafProfileLevel secure_leaf_profile_level()
{
    const char* value = std::getenv("SECURE_LEAF_MATERIALIZE_PROFILE_LEVEL");
    if (value == nullptr || *value == '\0') {
        return SecureLeafProfileLevel::Group;
    }
    const std::string normalized(value);
    if (normalized == "none" || normalized == "off" || normalized == "0") {
        return SecureLeafProfileLevel::None;
    }
    if (normalized == "group") {
        return SecureLeafProfileLevel::Group;
    }
    if (normalized == "batch") {
        return SecureLeafProfileLevel::Batch;
    }
    if (normalized == "leaf") {
        return SecureLeafProfileLevel::Leaf;
    }
    throw std::runtime_error("Invalid SECURE_LEAF_MATERIALIZE_PROFILE_LEVEL.");
}

bool secure_leaf_profile_enabled(SecureLeafProfileLevel current, SecureLeafProfileLevel required)
{
    return static_cast<int>(current) >= static_cast<int>(required);
}

double elapsed_seconds_since(const TimePoint& started)
{
    return std::chrono::duration<double>(SteadyClock::now() - started).count();
}

void secure_leaf_profile_log(
    int role,
    SecureLeafProfileLevel current,
    SecureLeafProfileLevel required,
    json fields)
{
    if (role != 0 || !secure_leaf_profile_enabled(current, required)) {
        return;
    }
    fields["materializer_version"] = kSecureLeafMaterializerVersion;
    std::cout << "[SECURE_LEAF_PROFILE] " << fields.dump() << std::endl;
}

std::size_t secure_leaf_sort_batch_size()
{
    const char* value = std::getenv("SECURE_LEAF_MATERIALIZE_SORT_BATCH_SIZE");
    if (value == nullptr || *value == '\0') {
        return 8;
    }
    try {
        return std::max<std::size_t>(1, static_cast<std::size_t>(std::stoull(value)));
    } catch (...) {
        throw std::runtime_error("Invalid SECURE_LEAF_MATERIALIZE_SORT_BATCH_SIZE.");
    }
}

void secure_share_i64_column(
    int role,
    int input_party,
    const i64Matrix& plain,
    si64Matrix& shared,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime);

std::string json_escape(const std::string& input);
void write_text_file(const std::string& path, const std::string& content);
sbMatrix bool_not_matrix(sbMatrix value, int role);
sbMatrix bool_and_matrix(sbMatrix lhs, sbMatrix rhs, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime);

void sync_value_from_party(int role, int owner_party, Sh3Runtime& runtime, u64& value)
{
    if (role == owner_party) {
        runtime.mComm.mNext.asyncSendCopy(&value, 1);
        runtime.mComm.mPrev.asyncSendCopy(&value, 1);
        return;
    }

    if (role == ((owner_party + 1) % 3)) {
        runtime.mComm.mPrev.recv(&value, 1);
        return;
    }

    if (role == ((owner_party + 2) % 3)) {
        runtime.mComm.mNext.recv(&value, 1);
        return;
    }

    throw std::runtime_error("Invalid ABY3 role.");
}

si64Matrix int_row_slice(const si64Matrix& src, u64 row_begin, u64 row_count)
{
    si64Matrix out(row_count, src.cols());
    out.mShares[0] = src.mShares[0].block(row_begin, 0, row_count, src.cols());
    out.mShares[1] = src.mShares[1].block(row_begin, 0, row_count, src.cols());
    return out;
}

si64Matrix repeat_int_scalar_rows(const si64Matrix& scalar, u64 rows)
{
    si64Matrix out(rows, scalar.cols());
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < scalar.mShares[0].cols(); ++col) {
            out.mShares[0](row, col) = scalar.mShares[0](0, col);
            out.mShares[1](row, col) = scalar.mShares[1](0, col);
        }
    }
    return out;
}

si64Matrix shared_zero_int_matrix(u64 rows, u64 cols)
{
    si64Matrix out(rows, cols);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    return out;
}

si64Matrix shared_zero_int_scalar()
{
    return shared_zero_int_matrix(1, 1);
}

si64Matrix share_int_scalar(
    i64 value,
    int owner_party,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime,
    int role)
{
    i64Matrix plain(1, 1);
    plain(0, 0) = value;
    si64Matrix shared(1, 1);
    secure_share_i64_column(role, owner_party, plain, shared, enc, runtime);
    return shared;
}

si64Matrix bool_to_si64(
    sbMatrix& bool_mat,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    si64Matrix out(bool_mat.rows(), 1);
    bool2arith(role, bool_mat, out, enc, eval, runtime);
    return out;
}

si64Matrix select_si64_by_bool(
    const si64Matrix& true_value,
    const si64Matrix& false_value,
    sbMatrix& flag,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto delta = true_value - false_value;
    auto flag_int = bool_to_si64(flag, role, enc, eval, runtime);
    si64Matrix selected_delta(true_value.rows(), true_value.cols());
    cipher_mul(role, delta, flag_int, selected_delta, eval, enc, runtime);
    return false_value + selected_delta;
}

si64Matrix concat_shared_row_blocks(const std::vector<si64Matrix>& blocks)
{
    if (blocks.empty()) {
        return si64Matrix(0, 0);
    }

    const auto cols = blocks.front().cols();
    u64 total_rows = 0;
    for (const auto& block : blocks) {
        if (block.cols() != cols) {
            throw std::runtime_error("Shared row blocks must have identical column counts.");
        }
        total_rows += block.rows();
    }

    si64Matrix out(total_rows, cols);
    u64 row_offset = 0;
    for (const auto& block : blocks) {
        if (block.rows() == 0) {
            continue;
        }
        out.mShares[0].block(row_offset, 0, block.rows(), cols) = block.mShares[0];
        out.mShares[1].block(row_offset, 0, block.rows(), cols) = block.mShares[1];
        row_offset += block.rows();
    }
    return out;
}

std::string trim_copy(const std::string& input)
{
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(start, end - start);
}

std::vector<std::string> parse_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }

        if (ch == ',' && !in_quotes) {
            fields.push_back(current);
            current.clear();
            continue;
        }

        current.push_back(ch);
    }

    fields.push_back(current);
    return fields;
}

bool is_null_token(const std::string& token)
{
    std::string s = trim_copy(token);
    if (s.empty()) {
        return true;
    }

    std::string lower;
    lower.reserve(s.size());
    for (char ch : s) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    return lower == "null" || lower == "nan" || lower == "none";
}

NormalizedKey normalize_join_key_token(const std::string& token)
{
    if (is_null_token(token)) {
        return {};
    }

    std::string s = trim_copy(token);
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        s = s.substr(1, s.size() - 2);
    }

    std::size_t consumed = 0;
    long long int_val = 0;
    try {
        int_val = std::stoll(s, &consumed);
    } catch (const std::exception&) {
        consumed = 0;
    }

    if (consumed == s.size()) {
        NormalizedKey key;
        key.value = static_cast<i64>(int_val);
        key.is_null = false;
        return key;
    }

    std::size_t float_consumed = 0;
    double float_val = 0.0;
    try {
        float_val = std::stod(s, &float_consumed);
    } catch (const std::exception&) {
        float_consumed = 0;
    }

    if (float_consumed != s.size() || !std::isfinite(float_val) || std::floor(float_val) != float_val) {
        throw std::runtime_error("Unsupported join-key token for multiplier preprocessing: " + token);
    }

    if (float_val < static_cast<double>(std::numeric_limits<i64>::min()) ||
        float_val > static_cast<double>(std::numeric_limits<i64>::max())) {
        throw std::runtime_error("Join-key token out of int64 range: " + token);
    }

    NormalizedKey key;
    key.value = static_cast<i64>(float_val);
    key.is_null = false;
    return key;
}

std::vector<NormalizedKey> load_key_column_csv(
    const std::string& csv_path,
    u64 key_column,
    bool has_header)
{
    std::ifstream input(csv_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_path);
    }

    std::vector<NormalizedKey> keys;
    std::string line;
    bool is_first_line = true;
    while (std::getline(input, line)) {
        if (is_first_line && has_header) {
            is_first_line = false;
            continue;
        }
        is_first_line = false;
        if (line.empty()) {
            keys.push_back({});
            continue;
        }

        auto fields = parse_csv_line(line);
        if (key_column >= fields.size()) {
            throw std::runtime_error(
                "Requested key column out of range for CSV file: " + csv_path + " column=" + std::to_string(key_column)
            );
        }
        keys.push_back(normalize_join_key_token(fields[key_column]));
    }
    return keys;
}

void ensure_dir(const std::string& path)
{
    if (path.empty()) {
        return;
    }
    if (::mkdir(path.c_str(), 0775) != 0 && errno != EEXIST) {
        throw std::runtime_error("Failed to create directory " + path + ": " + std::strerror(errno));
    }
}

void write_share_pair_matrix(const std::string& path, const si64Matrix& values)
{
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open share output file: " + path);
    }
    for (u64 row = 0; row < values.rows(); ++row) {
        const i64 lhs = values.mShares[0](row, 0);
        const i64 rhs = values.mShares[1](row, 0);
        output.write(reinterpret_cast<const char*>(&lhs), sizeof(i64));
        output.write(reinterpret_cast<const char*>(&rhs), sizeof(i64));
    }
}

std::vector<i64> read_i64_records(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open share input file: " + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamsize bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(i64)) != 0) {
        throw std::runtime_error("Malformed int64 share file: " + path);
    }
    std::vector<i64> out(static_cast<std::size_t>(bytes / sizeof(i64)));
    if (!out.empty()) {
        input.read(reinterpret_cast<char*>(out.data()), bytes);
    }
    return out;
}

si64Matrix read_share_pair_matrix(const std::string& path, u64 rows)
{
    const auto raw = read_i64_records(path);
    if (raw.size() != static_cast<std::size_t>(rows) * 2) {
        throw std::runtime_error("Share pair file row count mismatch: " + path);
    }
    si64Matrix out(rows, 1);
    for (u64 row = 0; row < rows; ++row) {
        out.mShares[0](row, 0) = raw[static_cast<std::size_t>(row) * 2];
        out.mShares[1](row, 0) = raw[static_cast<std::size_t>(row) * 2 + 1];
    }
    return out;
}

si64Matrix read_share_pair_matrix_auto_rows(const std::string& path)
{
    const auto raw = read_i64_records(path);
    if (raw.size() % 2 != 0) {
        throw std::runtime_error("Share pair file has an odd number of int64 records: " + path);
    }
    const u64 rows = static_cast<u64>(raw.size() / 2);
    si64Matrix out(rows, 1);
    for (u64 row = 0; row < rows; ++row) {
        out.mShares[0](row, 0) = raw[static_cast<std::size_t>(row) * 2];
        out.mShares[1](row, 0) = raw[static_cast<std::size_t>(row) * 2 + 1];
    }
    return out;
}

sbMatrix read_bool_share_pair_matrix(const std::string& path, u64 rows, u64 bit_count = 1)
{
    const auto raw = read_i64_records(path);
    if (raw.size() != static_cast<std::size_t>(rows) * 2) {
        throw std::runtime_error("Boolean share pair file row count mismatch: " + path);
    }
    sbMatrix out(rows, bit_count);
    for (u64 row = 0; row < rows; ++row) {
        out.mShares[0](row, 0) = raw[static_cast<std::size_t>(row) * 2];
        out.mShares[1](row, 0) = raw[static_cast<std::size_t>(row) * 2 + 1];
    }
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 1; col < static_cast<u64>(out.mShares[0].cols()); ++col) {
            out.mShares[0](row, col) = 0;
            out.mShares[1](row, col) = 0;
        }
    }
    return out;
}

struct ReplicatedShares {
    std::vector<i64> a0;
    std::vector<i64> a1;
    std::vector<i64> a2;
};

std::uint64_t next_share_word(std::mt19937_64& rng)
{
    return rng() & ((1ULL << 63) - 1ULL);
}

ReplicatedShares split_arithmetic_replicated_fast(const std::vector<i64>& values, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    ReplicatedShares shares;
    shares.a0.resize(values.size());
    shares.a1.resize(values.size());
    shares.a2.resize(values.size());
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        const std::uint64_t value = static_cast<std::uint64_t>(values[idx]);
        const std::uint64_t a0 = next_share_word(rng);
        const std::uint64_t a1 = next_share_word(rng);
        const std::uint64_t a2 = value - a0 - a1;
        shares.a0[idx] = static_cast<i64>(a0);
        shares.a1[idx] = static_cast<i64>(a1);
        shares.a2[idx] = static_cast<i64>(a2);
    }
    return shares;
}

ReplicatedShares split_boolean_replicated_fast(const std::vector<i64>& values, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    ReplicatedShares shares;
    shares.a0.resize(values.size());
    shares.a1.resize(values.size());
    shares.a2.resize(values.size());
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        const std::uint64_t value = static_cast<std::uint64_t>(values[idx]) & 1ULL;
        const std::uint64_t a0 = next_share_word(rng);
        const std::uint64_t a1 = next_share_word(rng);
        const std::uint64_t a2 = value ^ a0 ^ a1;
        shares.a0[idx] = static_cast<i64>(a0);
        shares.a1[idx] = static_cast<i64>(a1);
        shares.a2[idx] = static_cast<i64>(a2);
    }
    return shares;
}

void write_role_share_pairs(const std::string& path, const ReplicatedShares& shares, int role)
{
    if (!(shares.a0.size() == shares.a1.size() && shares.a1.size() == shares.a2.size())) {
        throw std::runtime_error("Replicated share vector size mismatch.");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open share output file: " + path);
    }
    for (std::size_t idx = 0; idx < shares.a0.size(); ++idx) {
        i64 lhs = 0;
        i64 rhs = 0;
        if (role == 0) {
            lhs = shares.a0[idx];
            rhs = shares.a2[idx];
        } else if (role == 1) {
            lhs = shares.a1[idx];
            rhs = shares.a0[idx];
        } else if (role == 2) {
            lhs = shares.a2[idx];
            rhs = shares.a1[idx];
        } else {
            throw std::runtime_error("Invalid ABY3 role.");
        }
        output.write(reinterpret_cast<const char*>(&lhs), sizeof(i64));
        output.write(reinterpret_cast<const char*>(&rhs), sizeof(i64));
    }
}

void write_bool_share_pair_matrix(const std::string& path, const sbMatrix& values)
{
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open boolean share output file: " + path);
    }
    for (u64 row = 0; row < values.rows(); ++row) {
        const i64 lhs = values.mShares[0](row, 0);
        const i64 rhs = values.mShares[1](row, 0);
        output.write(reinterpret_cast<const char*>(&lhs), sizeof(i64));
        output.write(reinterpret_cast<const char*>(&rhs), sizeof(i64));
    }
}

void write_bool_share_pair_matrix_bitpacked(const std::string& path, const sbMatrix& values)
{
    const u64 rows = values.rows();
    const std::size_t bytes_per_share = static_cast<std::size_t>((rows + 7) / 8);
    std::vector<std::uint8_t> packed(bytes_per_share * 2, 0);
    for (u64 row = 0; row < rows; ++row) {
        const std::size_t byte_idx = static_cast<std::size_t>(row / 8);
        const std::uint8_t mask = static_cast<std::uint8_t>(1u << (row % 8));
        if ((static_cast<std::uint64_t>(values.mShares[0](row, 0)) & 1ULL) != 0) {
            packed[byte_idx] |= mask;
        }
        if ((static_cast<std::uint64_t>(values.mShares[1](row, 0)) & 1ULL) != 0) {
            packed[bytes_per_share + byte_idx] |= mask;
        }
    }
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open bit-packed boolean share output file: " + path);
    }
    if (!packed.empty()) {
        output.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    }
}

si64Matrix public_i64_column(i64 value, u64 rows, int role)
{
    si64Matrix out(rows, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    if (role == 0) {
        out.mShares[0].setConstant(value);
    } else if (role == 1) {
        out.mShares[1].setConstant(value);
    } else if (role != 2) {
        throw std::runtime_error("Invalid ABY3 role.");
    }
    return out;
}

si64Matrix public_i64_row_ids(u64 rows, int role)
{
    si64Matrix out(rows, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    for (u64 row = 0; row < rows; ++row) {
        if (role == 0) {
            out.mShares[0](row, 0) = static_cast<i64>(row);
        } else if (role == 1) {
            out.mShares[1](row, 0) = static_cast<i64>(row);
        } else if (role != 2) {
            throw std::runtime_error("Invalid ABY3 role.");
        }
    }
    return out;
}

void set_public_i64_cell(si64Matrix& out, u64 row, i64 value, int role)
{
    out.mShares[0](row, 0) = 0;
    out.mShares[1](row, 0) = 0;
    if (role == 0) {
        out.mShares[0](row, 0) = value;
    } else if (role == 1) {
        out.mShares[1](row, 0) = value;
    } else if (role != 2) {
        throw std::runtime_error("Invalid ABY3 role.");
    }
}

si64Matrix public_i64_scalar(i64 value, int role)
{
    return public_i64_column(value, 1, role);
}

sbMatrix bool_row_slice(const sbMatrix& src, u64 row_begin, u64 row_count)
{
    sbMatrix out(row_count, src.bitCount());
    for (u64 row = 0; row < row_count; ++row) {
        for (u64 col = 0; col < static_cast<u64>(src.mShares[0].cols()); ++col) {
            out.mShares[0](row, col) = src.mShares[0](row_begin + row, col);
            out.mShares[1](row, col) = src.mShares[1](row_begin + row, col);
        }
    }
    return out;
}

sbMatrix bool_repeat_row(const sbMatrix& src, u64 source_row, u64 rows)
{
    sbMatrix out(rows, src.bitCount());
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < static_cast<u64>(src.mShares[0].cols()); ++col) {
            out.mShares[0](row, col) = src.mShares[0](source_row, col);
            out.mShares[1](row, col) = src.mShares[1](source_row, col);
        }
    }
    return out;
}

si64Matrix repeat_shared_row(const si64Matrix& src, u64 source_row, u64 rows)
{
    si64Matrix out(rows, src.cols());
    for (u64 row = 0; row < rows; ++row) {
        for (u64 col = 0; col < src.cols(); ++col) {
            out.mShares[0](row, col) = src.mShares[0](source_row, col);
            out.mShares[1](row, col) = src.mShares[1](source_row, col);
        }
    }
    return out;
}

void write_text_file(const std::string& path, const std::string& content)
{
    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output text file: " + path);
    }
    output << content;
}

std::string json_escape(const std::string& input)
{
    std::ostringstream out;
    for (char ch : input) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

void write_secure_shared_values_manifest(
    const MultiplierPreprocessConfig& config,
    size_t pk_row_count,
    size_t fk_row_count,
    const std::string& artifact_dir)
{
    const bool fast_preprocess = secure_multiplier_fast_preprocess_enabled();
    const bool provider_separated = secure_multiplier_provider_separated_enabled();
    const std::string join_key_input_mode = provider_separated
        ? "provider_separated_secret_shares"
        : "centralized_plaintext_input_party";
    const std::string secure_core_status = fast_preprocess
        ? "legacy_plaintext_count_then_secret_share"
        : (provider_separated
            ? "pairwise_secret_shared_key_equality_count_no_reveal"
            : "centralized_plaintext_input_party_sorted_segmented_scan_count_no_reveal");
    const std::string manifest_path = artifact_dir + "/manifest.json";
    std::ostringstream content;
    content << "{\n";
    content << "  \"format_name\": \"BSPN_MULTIPLIER_PAYLOAD\",\n";
    content << "  \"format_version\": 2,\n";
    content << "  \"mode\": \"secure_shared_values\",\n";
    content << "  \"relationship_id\": \"" << json_escape(config.relationship_id) << "\",\n";
    content << "  \"pk_csv_path\": \"" << json_escape(config.pk_csv_path) << "\",\n";
    content << "  \"fk_csv_path\": \"" << json_escape(config.fk_csv_path) << "\",\n";
    content << "  \"pk_key_column\": " << config.pk_key_column << ",\n";
    content << "  \"fk_key_column\": " << config.fk_key_column << ",\n";
    content << "  \"fk_sample_rate\": " << std::setprecision(17) << config.fk_sample_rate << ",\n";
    content << "  \"pk_row_count\": " << pk_row_count << ",\n";
    content << "  \"fk_row_count\": " << fk_row_count << ",\n";
    content << "  \"fixed_decimal_bits\": " << kMultiplierFixedDecimalBits << ",\n";
    content << "  \"pk_input_party\": " << config.pk_input_party << ",\n";
    content << "  \"fk_input_party\": " << config.fk_input_party << ",\n";
    content << "  \"share_kind\": \"ABY3_REPLICATED_PAIR_I64\",\n";
    content << "  \"construction_mode\": \"" << (provider_separated ? "privacy_aligned" : "legacy") << "\",\n";
    content << "  \"join_key_input_mode\": \"" << join_key_input_mode << "\",\n";
    content << "  \"provider_separated_key_inputs\": " << (provider_separated ? "true" : "false") << ",\n";
    content << "  \"plaintext_count_then_share\": " << (fast_preprocess ? "true" : "false") << ",\n";
    content << "  \"secure_core_status\": \"" << secure_core_status << "\",\n";
    content << "  \"role_paths\": {\n";
    for (int role = 0; role < 3; ++role) {
        content << "    \"" << role << "\": {";
        content << "\"mu_path\": \"" << json_escape(artifact_dir + "/role_" + std::to_string(role) + "/mu.shares.bin") << "\", ";
        content << "\"mu_nn_path\": \"" << json_escape(artifact_dir + "/role_" + std::to_string(role) + "/mu_nn.shares.bin") << "\"}";
        content << (role == 2 ? "\n" : ",\n");
    }
    content << "  }\n";
    content << "}\n";
    write_text_file(manifest_path, content.str());
}

MultiplierPreprocessConfig parse_config(const CLP& cmd)
{
    MultiplierPreprocessConfig config;

    auto read_required_string = [&](const std::string& key) {
        if (!cmd.isSet(key)) {
            throw std::runtime_error("Missing required argument: --" + key);
        }
        return cmd.getMany<std::string>(key)[0];
    };

    auto read_optional_u64 = [&](const std::string& key, u64 default_value) {
        if (!cmd.isSet(key)) {
            return default_value;
        }
        return static_cast<u64>(cmd.getMany<u64>(key)[0]);
    };

    if (cmd.isSet("multiplier_mode")) {
        config.mode = cmd.getMany<std::string>("multiplier_mode")[0];
    }

    if (config.mode == "secure_leaf_materialize") {
        config.secure_leaf_plan_path = read_required_string("secure_leaf_plan_json");
        config.output_prefix = read_required_string("output_prefix");
    } else {
        config.pk_csv_path = read_required_string("pk_csv");
        config.fk_csv_path = read_required_string("fk_csv");
        config.output_prefix = read_required_string("output_prefix");
    }
    config.relationship_id = cmd.isSet("relationship_id")
        ? cmd.getMany<std::string>("relationship_id")[0]
        : "unknown_relationship";

    config.pk_key_column = read_optional_u64("pk_key_column", 0);
    config.fk_key_column = read_optional_u64("fk_key_column", 0);

    if (cmd.isSet("fk_sample_rate")) {
        config.fk_sample_rate = cmd.getMany<double>("fk_sample_rate")[0];
    }

    if (cmd.isSet("role")) {
        config.role = cmd.getMany<int>("role")[0];
    }

    if (cmd.isSet("input_party")) {
        config.input_party = cmd.getMany<int>("input_party")[0];
    }
    if (cmd.isSet("pk_input_party")) {
        config.pk_input_party = cmd.getMany<int>("pk_input_party")[0];
    } else {
        config.pk_input_party = config.input_party;
    }
    if (cmd.isSet("fk_input_party")) {
        config.fk_input_party = cmd.getMany<int>("fk_input_party")[0];
    }

    config.pk_has_header = cmd.isSet("pk_has_header");
    config.fk_has_header = cmd.isSet("fk_has_header");
    config.secure_sort_min_size = read_optional_u64("secure_sort_min_size", 32);

    return config;
}

void secure_share_i64_column(
    int role,
    int input_party,
    const i64Matrix& plain,
    si64Matrix& shared,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime)
{
    shared.resize(plain.rows(), plain.cols());
    if (role == input_party) {
        enc.localIntMatrix(runtime, plain, shared).get();
    } else {
        enc.remoteIntMatrix(runtime, shared).get();
    }
}

void build_secure_sorted_multiplier_rows(
    const MultiplierPreprocessConfig& config,
    int role,
    const std::vector<NormalizedKey>& pk_keys,
    const std::vector<NormalizedKey>& fk_keys,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    u64 n_pk,
    u64 n_fk,
    u64 padded,
    si64Matrix& sec_key,
    std::vector<si64Matrix>& payloads_out,
    bool include_group_key = false)
{
    const u64 pad_rows = padded - n_pk - n_fk;
    const int sort_input_party = config.pk_input_party;

    i64Matrix plain_pk_key(n_pk, 1);
    i64Matrix plain_pk_table_id(n_pk, 1);
    i64Matrix plain_pk_row_id(n_pk, 1);
    i64Matrix plain_pk_is_null(n_pk, 1);
    i64Matrix plain_pk_valid(n_pk, 1);
    i64Matrix plain_pk_fk_contrib(n_pk, 1);
    i64Matrix plain_pk_group_key(n_pk, 1);
    plain_pk_key.setZero();
    plain_pk_table_id.setZero();
    plain_pk_row_id.setZero();
    plain_pk_is_null.setZero();
    plain_pk_valid.setZero();
    plain_pk_fk_contrib.setZero();
    plain_pk_group_key.setZero();

    i64Matrix plain_fk_key(n_fk, 1);
    i64Matrix plain_fk_table_id(n_fk, 1);
    i64Matrix plain_fk_row_id(n_fk, 1);
    i64Matrix plain_fk_is_null(n_fk, 1);
    i64Matrix plain_fk_valid(n_fk, 1);
    i64Matrix plain_fk_contrib(n_fk, 1);
    i64Matrix plain_fk_group_key(n_fk, 1);
    plain_fk_key.setZero();
    plain_fk_table_id.setZero();
    plain_fk_row_id.setZero();
    plain_fk_is_null.setZero();
    plain_fk_valid.setZero();
    plain_fk_contrib.setZero();
    plain_fk_group_key.setZero();

    i64Matrix plain_pad_key(pad_rows, 1);
    i64Matrix plain_pad_table_id(pad_rows, 1);
    i64Matrix plain_pad_row_id(pad_rows, 1);
    i64Matrix plain_pad_is_null(pad_rows, 1);
    i64Matrix plain_pad_valid(pad_rows, 1);
    i64Matrix plain_pad_fk_contrib(pad_rows, 1);
    i64Matrix plain_pad_group_key(pad_rows, 1);
    plain_pad_key.setZero();
    plain_pad_table_id.setConstant(2);
    plain_pad_row_id.setZero();
    plain_pad_is_null.setConstant(1);
    plain_pad_valid.setZero();
    plain_pad_fk_contrib.setZero();
    plain_pad_group_key.setZero();

    std::unordered_map<i64, i64> dense_key_rank;
    bool have_non_null_key = false;
    if (role == sort_input_party) {
        std::vector<i64> distinct_keys;
        distinct_keys.reserve(static_cast<std::size_t>(n_pk + n_fk));
        auto collect_keys = [&](const std::vector<NormalizedKey>& keys) {
            for (const auto& key : keys) {
                if (key.is_null) {
                    continue;
                }
                distinct_keys.push_back(key.value);
            }
        };
        collect_keys(pk_keys);
        collect_keys(fk_keys);
        std::sort(distinct_keys.begin(), distinct_keys.end());
        distinct_keys.erase(std::unique(distinct_keys.begin(), distinct_keys.end()), distinct_keys.end());
        have_non_null_key = !distinct_keys.empty();
        for (std::size_t idx = 0; idx < distinct_keys.size(); ++idx) {
            dense_key_rank.emplace(distinct_keys[idx], static_cast<i64>(idx + 1));
        }
    }

    const __int128 stride = static_cast<__int128>(std::max<u64>(1, padded));
    auto dense_rank_for_key = [&](const NormalizedKey& key) -> i64 {
        if (key.is_null || !have_non_null_key) {
            return 0;
        }
        const auto iter = dense_key_rank.find(key.value);
        if (iter == dense_key_rank.end()) {
            throw std::runtime_error("secure multiplier dense key rank missing.");
        }
        return iter->second;
    };
    auto encode_key = [&](const NormalizedKey& key, i64 tag) -> i64 {
        if (key.is_null || !have_non_null_key) {
            return tag;
        }

        const __int128 base = static_cast<__int128>(dense_rank_for_key(key));
        const __int128 composite = base * stride + static_cast<__int128>(tag);
        if (composite < static_cast<__int128>(std::numeric_limits<i64>::min()) ||
            composite > static_cast<__int128>(std::numeric_limits<i64>::max())) {
            throw std::runtime_error("secure multiplier sort key overflowed int64 range.");
        }
        return static_cast<i64>(composite);
    };
    auto encode_group_key = [&](const NormalizedKey& key) -> i64 {
        if (key.is_null || !have_non_null_key) {
            return 0;
        }
        const __int128 base = static_cast<__int128>(dense_rank_for_key(key));
        if (base < static_cast<__int128>(std::numeric_limits<i64>::min()) ||
            base > static_cast<__int128>(std::numeric_limits<i64>::max())) {
            throw std::runtime_error("secure multiplier group key overflowed int64 range.");
        }
        return static_cast<i64>(base);
    };

    if (role == sort_input_party) {
        for (u64 i = 0; i < n_pk; ++i) {
            plain_pk_key(i, 0) = encode_key(pk_keys[static_cast<size_t>(i)], static_cast<i64>(i));
            plain_pk_table_id(i, 0) = 0;
            plain_pk_row_id(i, 0) = static_cast<i64>(i);
            plain_pk_is_null(i, 0) = pk_keys[static_cast<size_t>(i)].is_null ? 1 : 0;
            plain_pk_valid(i, 0) = 1;
            plain_pk_group_key(i, 0) = encode_group_key(pk_keys[static_cast<size_t>(i)]);
        }
        for (u64 i = 0; i < n_fk; ++i) {
            plain_fk_key(i, 0) = encode_key(
                fk_keys[static_cast<size_t>(i)],
                static_cast<i64>(n_pk + i)
            );
            plain_fk_table_id(i, 0) = 1;
            plain_fk_row_id(i, 0) = static_cast<i64>(i);
            plain_fk_is_null(i, 0) = fk_keys[static_cast<size_t>(i)].is_null ? 1 : 0;
            plain_fk_valid(i, 0) = 1;
            plain_fk_contrib(i, 0) = fk_keys[static_cast<size_t>(i)].is_null ? 0 : 1;
            plain_fk_group_key(i, 0) = encode_group_key(fk_keys[static_cast<size_t>(i)]);
        }
        for (u64 i = 0; i < pad_rows; ++i) {
            plain_pad_key(i, 0) = static_cast<i64>(n_pk + n_fk + i);
            plain_pad_row_id(i, 0) = static_cast<i64>(i);
        }
    }

    si64Matrix sec_pk_key;
    si64Matrix sec_pk_table_id;
    si64Matrix sec_pk_row_id;
    si64Matrix sec_pk_is_null;
    si64Matrix sec_pk_valid;
    si64Matrix sec_pk_fk_contrib;
    si64Matrix sec_pk_group_key;
    secure_share_i64_column(role, sort_input_party, plain_pk_key, sec_pk_key, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pk_table_id, sec_pk_table_id, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pk_row_id, sec_pk_row_id, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pk_is_null, sec_pk_is_null, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pk_valid, sec_pk_valid, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pk_fk_contrib, sec_pk_fk_contrib, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pk_group_key, sec_pk_group_key, enc, runtime);

    si64Matrix sec_fk_key;
    si64Matrix sec_fk_table_id;
    si64Matrix sec_fk_row_id;
    si64Matrix sec_fk_is_null;
    si64Matrix sec_fk_valid;
    si64Matrix sec_fk_fk_contrib;
    si64Matrix sec_fk_group_key;
    secure_share_i64_column(role, sort_input_party, plain_fk_key, sec_fk_key, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_fk_table_id, sec_fk_table_id, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_fk_row_id, sec_fk_row_id, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_fk_is_null, sec_fk_is_null, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_fk_valid, sec_fk_valid, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_fk_contrib, sec_fk_fk_contrib, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_fk_group_key, sec_fk_group_key, enc, runtime);

    si64Matrix sec_pad_key;
    si64Matrix sec_pad_table_id;
    si64Matrix sec_pad_row_id;
    si64Matrix sec_pad_is_null;
    si64Matrix sec_pad_valid;
    si64Matrix sec_pad_fk_contrib;
    si64Matrix sec_pad_group_key;
    secure_share_i64_column(role, sort_input_party, plain_pad_key, sec_pad_key, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pad_table_id, sec_pad_table_id, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pad_row_id, sec_pad_row_id, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pad_is_null, sec_pad_is_null, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pad_valid, sec_pad_valid, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pad_fk_contrib, sec_pad_fk_contrib, enc, runtime);
    secure_share_i64_column(role, sort_input_party, plain_pad_group_key, sec_pad_group_key, enc, runtime);

    sec_key = concat_shared_row_blocks({sec_pk_key, sec_fk_key, sec_pad_key});
    si64Matrix sec_table_id = concat_shared_row_blocks({sec_pk_table_id, sec_fk_table_id, sec_pad_table_id});
    si64Matrix sec_row_id = concat_shared_row_blocks({sec_pk_row_id, sec_fk_row_id, sec_pad_row_id});
    si64Matrix sec_is_null = concat_shared_row_blocks({sec_pk_is_null, sec_fk_is_null, sec_pad_is_null});
    si64Matrix sec_valid = concat_shared_row_blocks({sec_pk_valid, sec_fk_valid, sec_pad_valid});
    si64Matrix sec_fk_contrib = concat_shared_row_blocks({sec_pk_fk_contrib, sec_fk_fk_contrib, sec_pad_fk_contrib});
    si64Matrix sec_group_key = concat_shared_row_blocks({sec_pk_group_key, sec_fk_group_key, sec_pad_group_key});

    const u64 payload_cols = include_group_key ? 6 : 5;
    payloads_out.resize(static_cast<size_t>(padded));
    for (u64 row = 0; row < padded; ++row) {
        si64Matrix payload(payload_cols, 1);
        payload.mShares[0](0, 0) = sec_table_id.mShares[0](row, 0);
        payload.mShares[1](0, 0) = sec_table_id.mShares[1](row, 0);
        payload.mShares[0](1, 0) = sec_row_id.mShares[0](row, 0);
        payload.mShares[1](1, 0) = sec_row_id.mShares[1](row, 0);
        payload.mShares[0](2, 0) = sec_is_null.mShares[0](row, 0);
        payload.mShares[1](2, 0) = sec_is_null.mShares[1](row, 0);
        payload.mShares[0](3, 0) = sec_valid.mShares[0](row, 0);
        payload.mShares[1](3, 0) = sec_valid.mShares[1](row, 0);
        payload.mShares[0](4, 0) = sec_fk_contrib.mShares[0](row, 0);
        payload.mShares[1](4, 0) = sec_fk_contrib.mShares[1](row, 0);
        if (include_group_key) {
            payload.mShares[0](5, 0) = sec_group_key.mShares[0](row, 0);
            payload.mShares[1](5, 0) = sec_group_key.mShares[1](row, 0);
        }
        payloads_out[static_cast<size_t>(row)] = std::move(payload);
    }
    quick_sort_with_other_elements(sec_key, payloads_out, role, enc, eval, runtime, config.secure_sort_min_size);
}

si64Matrix payload_column_to_matrix(const std::vector<si64Matrix>& payloads, u64 column_idx)
{
    si64Matrix out(static_cast<u64>(payloads.size()), 1);
    for (u64 row = 0; row < static_cast<u64>(payloads.size()); ++row) {
        if (column_idx >= payloads[static_cast<std::size_t>(row)].rows()) {
            throw std::runtime_error("Sorted multiplier payload column index is out of bounds.");
        }
        out.mShares[0](row, 0) = payloads[static_cast<std::size_t>(row)].mShares[0](column_idx, 0);
        out.mShares[1](row, 0) = payloads[static_cast<std::size_t>(row)].mShares[1](column_idx, 0);
    }
    return out;
}

std::vector<si64Matrix> column_to_payload_vector(const si64Matrix& column)
{
    std::vector<si64Matrix> out(static_cast<std::size_t>(column.rows()));
    for (u64 row = 0; row < column.rows(); ++row) {
        si64Matrix payload(1, 1);
        payload.mShares[0](0, 0) = column.mShares[0](row, 0);
        payload.mShares[1](0, 0) = column.mShares[1](row, 0);
        out[static_cast<std::size_t>(row)] = std::move(payload);
    }
    return out;
}

sbMatrix shared_false_bool_matrix(u64 rows, u64 bit_count, int role)
{
    sbMatrix out(rows, bit_count);
    bool_init_false(role, out);
    return out;
}

sbMatrix bool_and_matrix_columns(
    sbMatrix lhs,
    sbMatrix rhs,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    sbMatrix out(lhs.rows(), lhs.bitCount());
    bool_cipher_and(role, lhs, rhs, out, enc, eval, runtime);
    return out;
}

si64Matrix segmented_prefix_sum(
    const si64Matrix& values,
    const sbMatrix& same_previous,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    si64Matrix sums = values;
    sbMatrix links = same_previous;
    const u64 rows = values.rows();
    for (u64 offset = 1; offset < rows; offset <<= 1) {
        si64Matrix shifted_sums(rows, 1);
        shifted_sums.mShares[0].setZero();
        shifted_sums.mShares[1].setZero();
        sbMatrix shifted_links = shared_false_bool_matrix(rows, links.bitCount(), role);
        for (u64 row = offset; row < rows; ++row) {
            shifted_sums.mShares[0](row, 0) = sums.mShares[0](row - offset, 0);
            shifted_sums.mShares[1](row, 0) = sums.mShares[1](row - offset, 0);
            for (u64 col = 0; col < static_cast<u64>(links.mShares[0].cols()); ++col) {
                shifted_links.mShares[0](row, col) = links.mShares[0](row - offset, col);
                shifted_links.mShares[1](row, col) = links.mShares[1](row - offset, col);
            }
        }
        auto candidate = sums + shifted_sums;
        auto select_flag = links;
        sums = select_si64_by_bool(candidate, sums, select_flag, role, enc, eval, runtime);
        links = bool_and_matrix_columns(links, shifted_links, role, enc, eval, runtime);
    }
    return sums;
}

si64Matrix segmented_suffix_sum(
    const si64Matrix& values,
    const sbMatrix& same_next,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    si64Matrix sums = values;
    sbMatrix links = same_next;
    const u64 rows = values.rows();
    for (u64 offset = 1; offset < rows; offset <<= 1) {
        si64Matrix shifted_sums(rows, 1);
        shifted_sums.mShares[0].setZero();
        shifted_sums.mShares[1].setZero();
        sbMatrix shifted_links = shared_false_bool_matrix(rows, links.bitCount(), role);
        for (u64 row = 0; row + offset < rows; ++row) {
            shifted_sums.mShares[0](row, 0) = sums.mShares[0](row + offset, 0);
            shifted_sums.mShares[1](row, 0) = sums.mShares[1](row + offset, 0);
            for (u64 col = 0; col < static_cast<u64>(links.mShares[0].cols()); ++col) {
                shifted_links.mShares[0](row, col) = links.mShares[0](row + offset, col);
                shifted_links.mShares[1](row, col) = links.mShares[1](row + offset, col);
            }
        }
        auto candidate = sums + shifted_sums;
        auto select_flag = links;
        sums = select_si64_by_bool(candidate, sums, select_flag, role, enc, eval, runtime);
        links = bool_and_matrix_columns(links, shifted_links, role, enc, eval, runtime);
    }
    return sums;
}

sbMatrix adjacent_group_equal_flags(
    si64Matrix& group_key,
    bool compare_previous,
    int role,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const u64 rows = group_key.rows();
    si64Matrix lhs(rows, 1);
    si64Matrix rhs(rows, 1);
    lhs.mShares[0].setZero();
    lhs.mShares[1].setZero();
    rhs.mShares[0].setZero();
    rhs.mShares[1].setZero();
    for (u64 row = 0; row < rows; ++row) {
        lhs.mShares[0](row, 0) = group_key.mShares[0](row, 0);
        lhs.mShares[1](row, 0) = group_key.mShares[1](row, 0);
        if (compare_previous) {
            if (row == 0) {
                continue;
            }
            rhs.mShares[0](row, 0) = group_key.mShares[0](row - 1, 0);
            rhs.mShares[1](row, 0) = group_key.mShares[1](row - 1, 0);
        } else {
            if (row + 1 >= rows) {
                continue;
            }
            rhs.mShares[0](row, 0) = group_key.mShares[0](row + 1, 0);
            rhs.mShares[1](row, 0) = group_key.mShares[1](row + 1, 0);
        }
    }
    sbMatrix out;
    cipher_eq(role, lhs, rhs, out, eval, runtime);
    if (rows > 0) {
        const u64 boundary_row = compare_previous ? 0 : rows - 1;
        for (u64 col = 0; col < static_cast<u64>(out.mShares[0].cols()); ++col) {
            out.mShares[0](boundary_row, col) = 0;
            out.mShares[1](boundary_row, col) = 0;
        }
    }
    return out;
}

si64Matrix int_column_slice(const si64Matrix& src, u64 column_idx)
{
    if (column_idx >= src.cols()) {
        throw std::runtime_error("Integer matrix column index is out of bounds.");
    }
    si64Matrix out(src.rows(), 1);
    for (u64 row = 0; row < src.rows(); ++row) {
        out.mShares[0](row, 0) = src.mShares[0](row, column_idx);
        out.mShares[1](row, 0) = src.mShares[1](row, column_idx);
    }
    return out;
}

si64Matrix int_columns_to_matrix(const std::vector<si64Matrix>& columns)
{
    if (columns.empty()) {
        return si64Matrix(0, 0);
    }
    const u64 rows = columns.front().rows();
    si64Matrix out(rows, static_cast<u64>(columns.size()));
    for (u64 col = 0; col < static_cast<u64>(columns.size()); ++col) {
        if (columns[static_cast<std::size_t>(col)].rows() != rows ||
            columns[static_cast<std::size_t>(col)].cols() != 1) {
            throw std::runtime_error("Integer payload columns must all be rows x 1.");
        }
        for (u64 row = 0; row < rows; ++row) {
            out.mShares[0](row, col) = columns[static_cast<std::size_t>(col)].mShares[0](row, 0);
            out.mShares[1](row, col) = columns[static_cast<std::size_t>(col)].mShares[1](row, 0);
        }
    }
    return out;
}

si64Matrix prefix_sum(
    const si64Matrix& values,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto same_previous = bool_not_matrix(shared_false_bool_matrix(values.rows(), 1, role), role);
    if (values.rows() > 0) {
        same_previous.mShares[0](0, 0) = 0;
        same_previous.mShares[1](0, 0) = 0;
    }
    return segmented_prefix_sum(values, same_previous, role, enc, eval, runtime);
}

sbMatrix int_eq_public(
    const si64Matrix& values,
    i64 public_value,
    int role,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto lhs = values;
    auto rhs = public_i64_column(public_value, values.rows(), role);
    sbMatrix out;
    cipher_eq(role, lhs, rhs, out, eval, runtime);
    return out;
}

void run_secure_multiplier_shared_values_fast(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_shared_values mode requires --role in {0,1,2}.");
    }
    if (config.fk_sample_rate <= 0.0) {
        throw std::runtime_error("FK sample rate must be positive.");
    }

    const auto pk_keys = load_key_column_csv(config.pk_csv_path, config.pk_key_column, config.pk_has_header);
    const auto fk_keys = load_key_column_csv(config.fk_csv_path, config.fk_key_column, config.fk_has_header);

    std::unordered_map<i64, i64> fk_counts;
    for (const auto& fk : fk_keys) {
        if (!fk.is_null) {
            ++fk_counts[fk.value];
        }
    }

    const i64 fixed_scale = static_cast<i64>(std::llround(
        static_cast<double>(kMultiplierFixedOne) / config.fk_sample_rate));
    std::vector<i64> mu_fixed(pk_keys.size(), 0);
    std::vector<i64> mu_nn_fixed(pk_keys.size(), kMultiplierFixedOne);
    for (std::size_t idx = 0; idx < pk_keys.size(); ++idx) {
        i64 count = 0;
        if (!pk_keys[idx].is_null) {
            auto iter = fk_counts.find(pk_keys[idx].value);
            if (iter != fk_counts.end()) {
                count = iter->second;
            }
        }
        const i64 value = count * fixed_scale;
        mu_fixed[idx] = value;
        mu_nn_fixed[idx] = (count == 0) ? kMultiplierFixedOne : value;
    }

    ensure_dir(config.output_prefix);
    const std::string role_dir = config.output_prefix + "/role_" + std::to_string(config.role);
    ensure_dir(role_dir);
    const auto mu_shares = split_arithmetic_replicated_fast(mu_fixed, 13001);
    const auto mu_nn_shares = split_arithmetic_replicated_fast(mu_nn_fixed, 13003);
    write_role_share_pairs(role_dir + "/mu.shares.bin", mu_shares, config.role);
    write_role_share_pairs(role_dir + "/mu_nn.shares.bin", mu_nn_shares, config.role);
    if (config.role == 0) {
        write_secure_shared_values_manifest(
            config,
            pk_keys.size(),
            fk_keys.size(),
            config.output_prefix);
    }
}

void run_secure_multiplier_shared_values_provider_separated(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_shared_values mode requires --role in {0,1,2}.");
    }
    if (config.pk_input_party < 0 || config.pk_input_party > 2) {
        throw std::runtime_error("secure_shared_values mode requires --pk_input_party in {0,1,2}.");
    }
    if (config.fk_input_party < 0 || config.fk_input_party > 2) {
        throw std::runtime_error("secure_shared_values mode requires --fk_input_party in {0,1,2}.");
    }
    if (config.pk_input_party == config.fk_input_party) {
        throw std::runtime_error(
            "provider-separated secure multiplier preprocessing requires distinct PK and FK input parties.");
    }
    if (config.fk_sample_rate <= 0.0) {
        throw std::runtime_error("FK sample rate must be positive.");
    }

    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(config.role), ios, enc, eval, runtime);

    std::vector<NormalizedKey> pk_keys;
    std::vector<NormalizedKey> fk_keys;
    u64 n_pk = 0;
    u64 n_fk = 0;

    if (config.role == config.pk_input_party) {
        pk_keys = load_key_column_csv(config.pk_csv_path, config.pk_key_column, config.pk_has_header);
        n_pk = static_cast<u64>(pk_keys.size());
    }
    if (config.role == config.fk_input_party) {
        fk_keys = load_key_column_csv(config.fk_csv_path, config.fk_key_column, config.fk_has_header);
        n_fk = static_cast<u64>(fk_keys.size());
    }

    sync_value_from_party(config.role, config.pk_input_party, runtime, n_pk);
    sync_value_from_party(config.role, config.fk_input_party, runtime, n_fk);

    i64Matrix plain_pk_key(n_pk, 1);
    i64Matrix plain_pk_is_null(n_pk, 1);
    i64Matrix plain_fk_key(n_fk, 1);
    i64Matrix plain_fk_is_null(n_fk, 1);
    plain_pk_key.setZero();
    plain_pk_is_null.setConstant(1);
    plain_fk_key.setZero();
    plain_fk_is_null.setConstant(1);

    if (config.role == config.pk_input_party) {
        if (static_cast<u64>(pk_keys.size()) != n_pk) {
            throw std::runtime_error("PK owner has inconsistent key row count.");
        }
        for (u64 row = 0; row < n_pk; ++row) {
            plain_pk_key(row, 0) = pk_keys[static_cast<std::size_t>(row)].value;
            plain_pk_is_null(row, 0) = pk_keys[static_cast<std::size_t>(row)].is_null ? 1 : 0;
        }
    }
    if (config.role == config.fk_input_party) {
        if (static_cast<u64>(fk_keys.size()) != n_fk) {
            throw std::runtime_error("FK owner has inconsistent key row count.");
        }
        for (u64 row = 0; row < n_fk; ++row) {
            plain_fk_key(row, 0) = fk_keys[static_cast<std::size_t>(row)].value;
            plain_fk_is_null(row, 0) = fk_keys[static_cast<std::size_t>(row)].is_null ? 1 : 0;
        }
    }

    si64Matrix pk_key_shared;
    si64Matrix pk_null_shared;
    si64Matrix fk_key_shared;
    si64Matrix fk_null_shared;
    secure_share_i64_column(config.role, config.pk_input_party, plain_pk_key, pk_key_shared, enc, runtime);
    secure_share_i64_column(config.role, config.pk_input_party, plain_pk_is_null, pk_null_shared, enc, runtime);
    secure_share_i64_column(config.role, config.fk_input_party, plain_fk_key, fk_key_shared, enc, runtime);
    secure_share_i64_column(config.role, config.fk_input_party, plain_fk_is_null, fk_null_shared, enc, runtime);

    si64Matrix counts(n_pk, 1);
    counts.mShares[0].setZero();
    counts.mShares[1].setZero();

    if (n_fk > 0) {
        auto fk_not_null = int_eq_public(fk_null_shared, 0, config.role, eval, runtime);
        for (u64 pk_row = 0; pk_row < n_pk; ++pk_row) {
            auto pk_key_rows = repeat_int_scalar_rows(int_row_slice(pk_key_shared, pk_row, 1), n_fk);
            auto pk_null_rows = repeat_int_scalar_rows(int_row_slice(pk_null_shared, pk_row, 1), n_fk);
            sbMatrix key_equal;
            cipher_eq(config.role, pk_key_rows, fk_key_shared, key_equal, eval, runtime);
            auto pk_not_null = int_eq_public(pk_null_rows, 0, config.role, eval, runtime);
            auto valid = bool_and_matrix(key_equal, pk_not_null, config.role, enc, eval, runtime);
            valid = bool_and_matrix(valid, fk_not_null, config.role, enc, eval, runtime);
            auto valid_int = bool_to_si64(valid, config.role, enc, eval, runtime);
            for (u64 fk_row = 0; fk_row < n_fk; ++fk_row) {
                counts.mShares[0](pk_row, 0) += valid_int.mShares[0](fk_row, 0);
                counts.mShares[1](pk_row, 0) += valid_int.mShares[1](fk_row, 0);
            }
        }
    }

    const i64 fixed_scale = static_cast<i64>(std::llround(
        static_cast<double>(kMultiplierFixedOne) / config.fk_sample_rate));
    si64Matrix mu_fixed(n_pk, 1);
    mu_fixed.mShares[0].setZero();
    mu_fixed.mShares[1].setZero();
    for (u64 row = 0; row < n_pk; ++row) {
        mu_fixed.mShares[0](row, 0) = counts.mShares[0](row, 0) * fixed_scale;
        mu_fixed.mShares[1](row, 0) = counts.mShares[1](row, 0) * fixed_scale;
    }

    auto zero_counts = shared_zero_int_matrix(n_pk, 1);
    sbMatrix is_zero_count;
    cipher_eq(config.role, counts, zero_counts, is_zero_count, eval, runtime);
    const auto one_scalar = share_int_scalar(kMultiplierFixedOne, 0, enc, runtime, config.role);
    const auto one_rows = repeat_int_scalar_rows(one_scalar, n_pk);
    si64Matrix mu_nn_fixed = select_si64_by_bool(
        one_rows,
        mu_fixed,
        is_zero_count,
        config.role,
        enc,
        eval,
        runtime);

    ensure_dir(config.output_prefix);
    const std::string role_dir = config.output_prefix + "/role_" + std::to_string(config.role);
    ensure_dir(role_dir);
    write_share_pair_matrix(role_dir + "/mu.shares.bin", mu_fixed);
    write_share_pair_matrix(role_dir + "/mu_nn.shares.bin", mu_nn_fixed);
    if (config.role == 0) {
        write_secure_shared_values_manifest(
            config,
            static_cast<std::size_t>(n_pk),
            static_cast<std::size_t>(n_fk),
            config.output_prefix);
    }
}

void run_secure_multiplier_shared_values(const MultiplierPreprocessConfig& config)
{
    if (secure_multiplier_fast_preprocess_enabled()) {
        run_secure_multiplier_shared_values_fast(config);
        return;
    }
    if (secure_multiplier_provider_separated_enabled()) {
        run_secure_multiplier_shared_values_provider_separated(config);
        return;
    }

    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_shared_values mode requires --role in {0,1,2}.");
    }
    if (config.pk_input_party < 0 || config.pk_input_party > 2) {
        throw std::runtime_error("secure_shared_values mode requires --pk_input_party in {0,1,2}.");
    }
    if (config.fk_input_party < 0 || config.fk_input_party > 2) {
        throw std::runtime_error("secure_shared_values mode requires --fk_input_party in {0,1,2}.");
    }
    if (config.fk_sample_rate <= 0.0) {
        throw std::runtime_error("FK sample rate must be positive.");
    }

    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(config.role), ios, enc, eval, runtime);

    std::vector<NormalizedKey> pk_keys;
    std::vector<NormalizedKey> fk_keys;
    u64 n_pk = 0;
    u64 n_fk = 0;

    const int sort_input_party = config.pk_input_party;
    if (config.role == sort_input_party) {
        pk_keys = load_key_column_csv(config.pk_csv_path, config.pk_key_column, config.pk_has_header);
        fk_keys = load_key_column_csv(config.fk_csv_path, config.fk_key_column, config.fk_has_header);
        n_pk = static_cast<u64>(pk_keys.size());
        n_fk = static_cast<u64>(fk_keys.size());
    }

    sync_value_from_party(config.role, sort_input_party, runtime, n_pk);
    sync_value_from_party(config.role, sort_input_party, runtime, n_fk);

    const u64 padded = roundUpToPowerOfTwo(n_pk + n_fk);
    si64Matrix sec_key;
    std::vector<si64Matrix> sorted_payloads;
    build_secure_sorted_multiplier_rows(
        config,
        config.role,
        pk_keys,
        fk_keys,
        enc,
        eval,
        runtime,
        n_pk,
        n_fk,
        padded,
        sec_key,
        sorted_payloads,
        true);

    auto table_id = payload_column_to_matrix(sorted_payloads, 0);
    auto row_id = payload_column_to_matrix(sorted_payloads, 1);
    auto fk_contrib = payload_column_to_matrix(sorted_payloads, 4);
    auto group_key = payload_column_to_matrix(sorted_payloads, 5);

    auto same_previous = adjacent_group_equal_flags(group_key, true, config.role, eval, runtime);
    auto same_next = adjacent_group_equal_flags(group_key, false, config.role, eval, runtime);
    auto prefix_counts = segmented_prefix_sum(fk_contrib, same_previous, config.role, enc, eval, runtime);
    auto suffix_counts = segmented_suffix_sum(fk_contrib, same_next, config.role, enc, eval, runtime);
    auto sorted_group_counts = prefix_counts + suffix_counts - fk_contrib;

    auto zero_sorted = public_i64_column(0, padded, config.role);
    sbMatrix is_pk_row;
    cipher_eq(config.role, table_id, zero_sorted, is_pk_row, eval, runtime);

    si64Matrix non_pk_sort_key(padded, 1);
    non_pk_sort_key.mShares[0].setZero();
    non_pk_sort_key.mShares[1].setZero();
    for (u64 row = 0; row < padded; ++row) {
        const i64 public_key = static_cast<i64>(n_pk + row);
        if (config.role == 0) {
            non_pk_sort_key.mShares[0](row, 0) = public_key;
        } else if (config.role == 1) {
            non_pk_sort_key.mShares[1](row, 0) = public_key;
        }
    }
    auto sort_back_key = select_si64_by_bool(row_id, non_pk_sort_key, is_pk_row, config.role, enc, eval, runtime);
    auto sorted_count_payloads = column_to_payload_vector(sorted_group_counts);
    quick_sort_with_other_elements(
        sort_back_key,
        sorted_count_payloads,
        config.role,
        enc,
        eval,
        runtime,
        config.secure_sort_min_size);

    si64Matrix counts(n_pk, 1);
    for (u64 row = 0; row < n_pk; ++row) {
        counts.mShares[0](row, 0) = sorted_count_payloads[static_cast<std::size_t>(row)].mShares[0](0, 0);
        counts.mShares[1](row, 0) = sorted_count_payloads[static_cast<std::size_t>(row)].mShares[1](0, 0);
    }

    const i64 fixed_scale = static_cast<i64>(std::llround(
        static_cast<double>(kMultiplierFixedOne) / config.fk_sample_rate));
    si64Matrix mu_fixed(n_pk, 1);
    for (u64 row = 0; row < n_pk; ++row) {
        mu_fixed.mShares[0](row, 0) = counts.mShares[0](row, 0) * fixed_scale;
        mu_fixed.mShares[1](row, 0) = counts.mShares[1](row, 0) * fixed_scale;
    }

    auto zero_counts = shared_zero_int_matrix(n_pk, 1);
    sbMatrix is_zero_count;
    cipher_eq(config.role, counts, zero_counts, is_zero_count, eval, runtime);
    const auto one_scalar = share_int_scalar(kMultiplierFixedOne, 0, enc, runtime, config.role);
    const auto one_rows = repeat_int_scalar_rows(one_scalar, n_pk);
    si64Matrix mu_nn_fixed = select_si64_by_bool(
        one_rows,
        mu_fixed,
        is_zero_count,
        config.role,
        enc,
        eval,
        runtime);

    ensure_dir(config.output_prefix);
    const std::string role_dir = config.output_prefix + "/role_" + std::to_string(config.role);
    ensure_dir(role_dir);
    write_share_pair_matrix(role_dir + "/mu.shares.bin", mu_fixed);
    write_share_pair_matrix(role_dir + "/mu_nn.shares.bin", mu_nn_fixed);
    if (config.role == 0) {
        write_secure_shared_values_manifest(
            config,
            static_cast<std::size_t>(n_pk),
            static_cast<std::size_t>(n_fk),
            config.output_prefix);
    }
}

std::string dirname_from_path(const std::string& path)
{
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

std::string join_path(const std::string& base, const std::string& child)
{
    if (child.empty()) {
        return base;
    }
    if (!child.empty() && child.front() == '/') {
        return child;
    }
    if (base.empty() || base == ".") {
        return child;
    }
    if (base.back() == '/') {
        return base + child;
    }
    return base + "/" + child;
}

json read_json_file(const std::string& path)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open json file: " + path);
    }
    json doc;
    input >> doc;
    return doc;
}

sbMatrix bool_and_matrix(sbMatrix lhs, sbMatrix rhs, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime)
{
    sbMatrix out(lhs.rows(), lhs.bitCount());
    bool_cipher_and(role, lhs, rhs, out, enc, eval, runtime);
    return out;
}

sbMatrix bool_or_matrix(sbMatrix lhs, sbMatrix rhs, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime)
{
    sbMatrix out(lhs.rows(), lhs.bitCount());
    bool_cipher_or(role, lhs, rhs, out, enc, eval, runtime);
    return out;
}

sbMatrix bool_not_matrix(sbMatrix value, int role)
{
    sbMatrix out(value.rows(), value.bitCount());
    bool_cipher_not(role, value, out);
    return out;
}

sbMatrix bool_prefix_or_matrix(
    sbMatrix values,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    sbMatrix prefix = values;
    const u64 rows = values.rows();
    for (u64 offset = 1; offset < rows; offset <<= 1) {
        sbMatrix shifted = shared_false_bool_matrix(rows, values.bitCount(), role);
        for (u64 row = offset; row < rows; ++row) {
            for (u64 col = 0; col < static_cast<u64>(values.mShares[0].cols()); ++col) {
                shifted.mShares[0](row, col) = prefix.mShares[0](row - offset, col);
                shifted.mShares[1](row, col) = prefix.mShares[1](row - offset, col);
            }
        }
        prefix = bool_or_matrix(prefix, shifted, role, enc, eval, runtime);
    }
    return prefix;
}

sbMatrix bool_shift_down_one_false(sbMatrix values, int role)
{
    sbMatrix shifted = shared_false_bool_matrix(values.rows(), values.bitCount(), role);
    for (u64 row = 1; row < values.rows(); ++row) {
        for (u64 col = 0; col < static_cast<u64>(values.mShares[0].cols()); ++col) {
            shifted.mShares[0](row, col) = values.mShares[0](row - 1, col);
            shifted.mShares[1](row, col) = values.mShares[1](row - 1, col);
        }
    }
    return shifted;
}

sbMatrix bool_reduce_or_matrix(
    sbMatrix values,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    if (values.rows() == 0) {
        return shared_false_bool_matrix(1, values.bitCount(), role);
    }
    auto prefix = bool_prefix_or_matrix(values, role, enc, eval, runtime);
    return bool_row_slice(prefix, values.rows() - 1, 1);
}

si64Matrix bool_to_arith_matrix(sbMatrix value, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime)
{
    si64Matrix out(value.rows(), 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    bool2arith(role, value, out, enc, eval, runtime);
    return out;
}

si64Matrix arith_mul_bool(const si64Matrix& value, sbMatrix flag, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime)
{
    auto flag_arith = bool_to_arith_matrix(flag, role, enc, eval, runtime);
    si64Matrix out(value.rows(), value.cols());
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    cipher_mul(role, value, flag_arith, out, eval, enc, runtime);
    return out;
}

struct MaterializedLeafResult {
    std::uint64_t leaf_node_id = 0;
    std::uint64_t product_group_id = 0;
    std::uint64_t public_bucket_width = 0;
    std::uint64_t real_bucket_count = 0;
    std::string bucket_mode = "fixed_cap";
    bool include_row_values = true;
};

void write_secure_leaf_counts_manifest(
    const std::string& output_prefix,
    const std::vector<MaterializedLeafResult>& leaves,
    u64 total_rows);

std::string role_dir_from_template(
    const std::string& base_dir,
    const std::string& role_share_dir,
    int role)
{
    std::string rendered = role_share_dir;
    const std::string token = "{role}";
    const auto pos = rendered.find(token);
    if (pos != std::string::npos) {
        rendered.replace(pos, token.size(), std::to_string(role));
    }
    return join_path(base_dir, rendered);
}

bool is_absolute_path(const std::string& path)
{
    return !path.empty() && path[0] == '/';
}

std::string resolve_plan_relative_path(const std::string& plan_base_dir, const std::string& path)
{
    if (path.empty() || is_absolute_path(path)) {
        return path;
    }
    return join_path(plan_base_dir, path);
}

std::string secure_leaf_row_value_cache_key(
    const json& plan,
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    int role)
{
    const std::string sampled_row_values_file = leaf_doc.value("sampled_row_values_share_file", std::string());
    if (!sampled_row_values_file.empty()) {
        const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
        return "file:" + join_path(plan_role_dir, sampled_row_values_file);
    }

    const std::string artifact_dir = leaf_doc.value("multiplier_artifact_dir", std::string());
    if (artifact_dir.empty()) {
        throw std::runtime_error("Secure leaf plan is missing multiplier_artifact_dir.");
    }
    const std::string kind = leaf_doc.value("multiplier_kind", std::string("mu"));
    if (kind != "mu" && kind != "mu_nn") {
        throw std::runtime_error("Unsupported secure multiplier leaf multiplier_kind: " + kind);
    }
    const std::string resolved_artifact_dir = resolve_plan_relative_path(plan_base_dir, artifact_dir);
    (void)plan;
    return "artifact:" + resolved_artifact_dir + "|" + kind;
}

si64Matrix load_secure_leaf_row_values(
    const json& plan,
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    u64 total_rows,
    int role)
{
    const std::string sampled_row_values_file = leaf_doc.value("sampled_row_values_share_file", std::string());
    if (!sampled_row_values_file.empty()) {
        const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
        return read_share_pair_matrix(join_path(plan_role_dir, sampled_row_values_file), total_rows);
    }

    if (!plan.contains("sample_row_id_by_position") || !plan["sample_row_id_by_position"].is_array()) {
        throw std::runtime_error("Secure leaf plan is missing sample_row_id_by_position.");
    }
    const auto& sample_rows = plan["sample_row_id_by_position"];
    if (sample_rows.size() != static_cast<std::size_t>(total_rows)) {
        throw std::runtime_error("Secure leaf plan sample_row_id_by_position length mismatch.");
    }

    const std::string artifact_dir = leaf_doc.value("multiplier_artifact_dir", std::string());
    if (artifact_dir.empty()) {
        throw std::runtime_error("Secure leaf plan is missing multiplier_artifact_dir.");
    }
    const std::string kind = leaf_doc.value("multiplier_kind", std::string("mu"));
    if (kind != "mu" && kind != "mu_nn") {
        throw std::runtime_error("Unsupported secure multiplier leaf multiplier_kind: " + kind);
    }
    const std::string resolved_artifact_dir = resolve_plan_relative_path(plan_base_dir, artifact_dir);
    const std::string role_dir = join_path(resolved_artifact_dir, "role_" + std::to_string(role));
    const auto full_values = read_share_pair_matrix_auto_rows(join_path(role_dir, kind + ".shares.bin"));

    si64Matrix sampled_values(total_rows, 1);
    sampled_values.mShares[0].setZero();
    sampled_values.mShares[1].setZero();
    for (u64 pos = 0; pos < total_rows; ++pos) {
        const auto pk_row_id = sample_rows[static_cast<std::size_t>(pos)].get<std::uint64_t>();
        if (pk_row_id >= full_values.rows()) {
            throw std::runtime_error("Secure leaf sample row id is outside multiplier artifact row count.");
        }
        sampled_values.mShares[0](pos, 0) = full_values.mShares[0](pk_row_id, 0);
        sampled_values.mShares[1](pos, 0) = full_values.mShares[1](pk_row_id, 0);
    }
    return sampled_values;
}

MaterializedLeafResult materialize_one_secure_leaf_fixed_cap(
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    const std::string& output_prefix,
    const si64Matrix& values,
    u64 total_rows,
    u64 public_leaf_bucket_width,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    sbMatrix& model_overflow,
    SecureLeafProfileLevel profile_level,
    std::size_t leaf_index,
    std::size_t total_leaf_count)
{
    const auto leaf_started = SteadyClock::now();
    const std::uint64_t leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
    const bool include_row_values = leaf_doc.value("include_row_values", true);
    const auto product_group_signed = leaf_doc.value("product_group_id", std::int64_t(-1));
    const std::uint64_t product_group_id = static_cast<std::uint64_t>(product_group_signed);
    const std::string role_name = "role_" + std::to_string(role);
    const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
    const auto read_started = SteadyClock::now();
    const auto membership = read_bool_share_pair_matrix(
        join_path(plan_role_dir, leaf_doc.value("membership_share_file", std::string())),
        total_rows,
        1);
    const double read_membership_elapsed = elapsed_seconds_since(read_started);
    if (values.rows() != total_rows) {
        throw std::runtime_error("Secure leaf cached row value vector size mismatch.");
    }

    const auto mask_started = SteadyClock::now();
    si64Matrix bucket_values(public_leaf_bucket_width, 1);
    bucket_values.mShares[0].setZero();
    bucket_values.mShares[1].setZero();
    si64Matrix row_values;
    if (include_row_values) {
        row_values = arith_mul_bool(values, membership, role, enc, eval, runtime);
    }
    const double row_value_mask_elapsed = elapsed_seconds_since(mask_started);

    const auto bucket_loop_started = SteadyClock::now();
    sbMatrix bucket_bitmaps(public_leaf_bucket_width * total_rows, 1);
    bool_init_false(role, bucket_bitmaps);

    sbMatrix remaining = membership;
    for (u64 bucket_idx = 0; bucket_idx < public_leaf_bucket_width; ++bucket_idx) {
        auto prior_or_inclusive = bool_prefix_or_matrix(remaining, role, enc, eval, runtime);
        auto has_prior = bool_shift_down_one_false(prior_or_inclusive, role);
        auto no_prior = bool_not_matrix(has_prior, role);
        auto representative_rows = bool_and_matrix(remaining, no_prior, role, enc, eval, runtime);

        auto selected_values = arith_mul_bool(values, representative_rows, role, enc, eval, runtime);
        bucket_values.mShares[0](bucket_idx, 0) = selected_values.mShares[0].sum();
        bucket_values.mShares[1](bucket_idx, 0) = selected_values.mShares[1].sum();

        auto repeated_bucket_value = repeat_shared_row(bucket_values, bucket_idx, total_rows);
        sbMatrix value_matches;
        auto values_copy = values;
        cipher_eq(role, values_copy, repeated_bucket_value, value_matches, eval, runtime);
        auto bucket_rows = bool_and_matrix(value_matches, remaining, role, enc, eval, runtime);
        const u64 out_begin = bucket_idx * total_rows;
        for (u64 row = 0; row < total_rows; ++row) {
            bucket_bitmaps.mShares[0](out_begin + row, 0) = bucket_rows.mShares[0](row, 0);
            bucket_bitmaps.mShares[1](out_begin + row, 0) = bucket_rows.mShares[1](row, 0);
        }

        auto not_bucket_rows = bool_not_matrix(bucket_rows, role);
        remaining = bool_and_matrix(remaining, not_bucket_rows, role, enc, eval, runtime);
    }
    const double bucket_loop_elapsed = elapsed_seconds_since(bucket_loop_started);

    const auto overflow_started = SteadyClock::now();
    auto leaf_overflow = bool_reduce_or_matrix(remaining, role, enc, eval, runtime);
    model_overflow = bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);
    const double overflow_elapsed = elapsed_seconds_since(overflow_started);

    const auto write_started = SteadyClock::now();
    ensure_dir(output_prefix);
    const std::string role_dir = join_path(output_prefix, role_name);
    ensure_dir(role_dir);
    const std::string leaf_prefix = "leaf_" + std::to_string(leaf_node_id);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_values.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_lowers.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_uppers.shares.bin"), bucket_values);
    write_bool_share_pair_matrix_bitpacked(join_path(role_dir, leaf_prefix + ".leaf_bitmaps.shares.bin"), bucket_bitmaps);
    if (include_row_values) {
        write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".row_values.shares.bin"), row_values);
    }
    const double write_elapsed = elapsed_seconds_since(write_started);

    secure_leaf_profile_log(
        role,
        profile_level,
        SecureLeafProfileLevel::Leaf,
        {
            {"event", "fixed_cap_leaf"},
            {"leaf_index", static_cast<std::uint64_t>(leaf_index)},
            {"leaf_count", static_cast<std::uint64_t>(total_leaf_count)},
            {"leaf_node_id", leaf_node_id},
            {"relationship_id", leaf_doc.value("relationship_id", std::string())},
            {"multiplier_kind", leaf_doc.value("multiplier_kind", std::string("mu"))},
            {"rows", total_rows},
            {"cap", public_leaf_bucket_width},
            {"read_membership_seconds", read_membership_elapsed},
            {"row_value_mask_seconds", row_value_mask_elapsed},
            {"bucket_loop_seconds", bucket_loop_elapsed},
            {"overflow_seconds", overflow_elapsed},
            {"write_seconds", write_elapsed},
            {"elapsed_seconds", elapsed_seconds_since(leaf_started)},
        });

    MaterializedLeafResult result;
    result.leaf_node_id = leaf_node_id;
    result.product_group_id = product_group_id;
    result.public_bucket_width = public_leaf_bucket_width;
    result.real_bucket_count = 0;
    result.bucket_mode = "fixed_cap";
    result.include_row_values = include_row_values;
    return result;
}

MaterializedLeafResult materialize_one_secure_leaf_sorted_group(
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    const std::string& output_prefix,
    const si64Matrix& values,
    u64 total_rows,
    u64 public_leaf_bucket_width,
    u64 secure_sort_min_size,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    sbMatrix& model_overflow)
{
    const std::uint64_t leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
    const bool include_row_values = leaf_doc.value("include_row_values", true);
    const auto product_group_signed = leaf_doc.value("product_group_id", std::int64_t(-1));
    const std::uint64_t product_group_id = static_cast<std::uint64_t>(product_group_signed);
    const std::string role_name = "role_" + std::to_string(role);
    const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
    const auto membership = read_bool_share_pair_matrix(
        join_path(plan_role_dir, leaf_doc.value("membership_share_file", std::string())),
        total_rows,
        1);
    if (values.rows() != total_rows) {
        throw std::runtime_error("Secure leaf cached row value vector size mismatch.");
    }

    si64Matrix row_values;
    if (include_row_values) {
        row_values = arith_mul_bool(values, membership, role, enc, eval, runtime);
    }

    auto membership_int = bool_to_arith_matrix(membership, role, enc, eval, runtime);
    auto row_ids = public_i64_row_ids(total_rows, role);
    auto sort_key = values;
    const i64 row_id_stride = static_cast<i64>(total_rows + 1);
    for (u64 row = 0; row < total_rows; ++row) {
        sort_key.mShares[0](row, 0) *= row_id_stride;
        sort_key.mShares[1](row, 0) *= row_id_stride;
    }
    sort_key = sort_key + row_ids;
    auto sort_payload = int_columns_to_matrix({membership_int, row_ids, values});
    quick_sort_with_payload_matrix(
        sort_key,
        sort_payload,
        role,
        enc,
        eval,
        runtime,
        static_cast<std::size_t>(secure_sort_min_size));

    auto membership_sorted = int_column_slice(sort_payload, 0);
    auto row_ids_sorted = int_column_slice(sort_payload, 1);
    auto values_sorted = int_column_slice(sort_payload, 2);
    auto same_previous = adjacent_group_equal_flags(values_sorted, true, role, eval, runtime);
    auto same_next = adjacent_group_equal_flags(values_sorted, false, role, eval, runtime);
    auto prefix_counts = segmented_prefix_sum(membership_sorted, same_previous, role, enc, eval, runtime);
    auto suffix_counts = segmented_suffix_sum(membership_sorted, same_next, role, enc, eval, runtime);
    auto group_counts = prefix_counts + suffix_counts - membership_sorted;
    auto group_empty = int_eq_public(group_counts, 0, role, eval, runtime);
    auto group_has_member = bool_not_matrix(group_empty, role);
    auto group_start = bool_not_matrix(same_previous, role);
    auto active_group_start = bool_and_matrix(group_start, group_has_member, role, enc, eval, runtime);
    auto active_group_start_int = bool_to_arith_matrix(active_group_start, role, enc, eval, runtime);
    auto active_prefix = prefix_sum(active_group_start_int, role, enc, eval, runtime);
    auto zero_rows = shared_zero_int_matrix(total_rows, 1);
    auto group_start_ranks = select_si64_by_bool(
        active_prefix,
        zero_rows,
        active_group_start,
        role,
        enc,
        eval,
        runtime);
    auto group_ranks_sorted = segmented_prefix_sum(
        group_start_ranks,
        same_previous,
        role,
        enc,
        eval,
        runtime);

    si64Matrix bucket_values(public_leaf_bucket_width, 1);
    bucket_values.mShares[0].setZero();
    bucket_values.mShares[1].setZero();
    for (u64 bucket_idx = 0; bucket_idx < public_leaf_bucket_width; ++bucket_idx) {
        auto rank_match = int_eq_public(active_prefix, static_cast<i64>(bucket_idx + 1), role, eval, runtime);
        auto bucket_start = bool_and_matrix(rank_match, active_group_start, role, enc, eval, runtime);
        auto selected_values = arith_mul_bool(values_sorted, bucket_start, role, enc, eval, runtime);
        bucket_values.mShares[0](bucket_idx, 0) = selected_values.mShares[0].sum();
        bucket_values.mShares[1](bucket_idx, 0) = selected_values.mShares[1].sum();
    }

    auto overflow_rank_match = int_eq_public(
        active_prefix,
        static_cast<i64>(public_leaf_bucket_width + 1),
        role,
        eval,
        runtime);
    auto overflow_starts = bool_and_matrix(
        overflow_rank_match,
        active_group_start,
        role,
        enc,
        eval,
        runtime);
    auto leaf_overflow = bool_reduce_or_matrix(overflow_starts, role, enc, eval, runtime);
    model_overflow = bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);

    auto sort_back_key = row_ids_sorted;
    auto sort_back_payload = int_columns_to_matrix({membership_sorted, group_ranks_sorted});
    quick_sort_with_payload_matrix(
        sort_back_key,
        sort_back_payload,
        role,
        enc,
        eval,
        runtime,
        static_cast<std::size_t>(secure_sort_min_size));
    auto membership_by_row = int_column_slice(sort_back_payload, 0);
    auto group_rank_by_row = int_column_slice(sort_back_payload, 1);
    auto membership_by_row_bool = int_eq_public(membership_by_row, 1, role, eval, runtime);

    sbMatrix bucket_bitmaps(public_leaf_bucket_width * total_rows, 1);
    bool_init_false(role, bucket_bitmaps);
    for (u64 bucket_idx = 0; bucket_idx < public_leaf_bucket_width; ++bucket_idx) {
        auto rank_match = int_eq_public(group_rank_by_row, static_cast<i64>(bucket_idx + 1), role, eval, runtime);
        auto bucket_rows = bool_and_matrix(rank_match, membership_by_row_bool, role, enc, eval, runtime);
        const u64 out_begin = bucket_idx * total_rows;
        for (u64 row = 0; row < total_rows; ++row) {
            bucket_bitmaps.mShares[0](out_begin + row, 0) = bucket_rows.mShares[0](row, 0);
            bucket_bitmaps.mShares[1](out_begin + row, 0) = bucket_rows.mShares[1](row, 0);
        }
    }

    ensure_dir(output_prefix);
    const std::string role_dir = join_path(output_prefix, role_name);
    ensure_dir(role_dir);
    const std::string leaf_prefix = "leaf_" + std::to_string(leaf_node_id);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_values.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_lowers.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_uppers.shares.bin"), bucket_values);
    write_bool_share_pair_matrix_bitpacked(join_path(role_dir, leaf_prefix + ".leaf_bitmaps.shares.bin"), bucket_bitmaps);
    if (include_row_values) {
        write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".row_values.shares.bin"), row_values);
    }

    MaterializedLeafResult result;
    result.leaf_node_id = leaf_node_id;
    result.product_group_id = product_group_id;
    result.public_bucket_width = public_leaf_bucket_width;
    result.real_bucket_count = 0;
    result.bucket_mode = "fixed_cap";
    result.include_row_values = include_row_values;
    return result;
}

std::vector<MaterializedLeafResult> materialize_secure_leaf_sorted_group_batch(
    const std::vector<json>& leaf_docs,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    const std::string& output_prefix,
    const si64Matrix& values,
    u64 total_rows,
    u64 public_leaf_bucket_width,
    u64 secure_sort_min_size,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    sbMatrix& model_overflow,
    SecureLeafProfileLevel profile_level,
    std::size_t group_index,
    std::size_t batch_index,
    std::size_t group_batch_count)
{
    const auto batch_started = SteadyClock::now();
    if (leaf_docs.empty()) {
        return {};
    }
    if (values.rows() != total_rows) {
        throw std::runtime_error("Secure leaf cached row value vector size mismatch.");
    }

    const std::string role_name = "role_" + std::to_string(role);
    const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
    const std::size_t leaf_count = leaf_docs.size();
    std::vector<sbMatrix> memberships;
    std::vector<si64Matrix> membership_ints;
    std::vector<si64Matrix> row_values_by_leaf;
    std::vector<bool> include_row_values_by_leaf;
    memberships.reserve(leaf_count);
    membership_ints.reserve(leaf_count);
    row_values_by_leaf.reserve(leaf_count);
    include_row_values_by_leaf.reserve(leaf_count);

    double read_membership_elapsed = 0.0;
    double row_value_mask_elapsed = 0.0;
    for (const auto& leaf_doc : leaf_docs) {
        const auto read_started = SteadyClock::now();
        auto membership = read_bool_share_pair_matrix(
            join_path(plan_role_dir, leaf_doc.value("membership_share_file", std::string())),
            total_rows,
            1);
        read_membership_elapsed += elapsed_seconds_since(read_started);
        const auto mask_started = SteadyClock::now();
        const bool include_row_values = leaf_doc.value("include_row_values", true);
        si64Matrix row_values;
        if (include_row_values) {
            row_values = arith_mul_bool(values, membership, role, enc, eval, runtime);
        }
        auto membership_int = bool_to_arith_matrix(membership, role, enc, eval, runtime);
        row_value_mask_elapsed += elapsed_seconds_since(mask_started);
        memberships.push_back(std::move(membership));
        row_values_by_leaf.push_back(std::move(row_values));
        include_row_values_by_leaf.push_back(include_row_values);
        membership_ints.push_back(std::move(membership_int));
    }

    auto row_ids = public_i64_row_ids(total_rows, role);
    auto sort_key = values;
    const i64 row_id_stride = static_cast<i64>(total_rows + 1);
    for (u64 row = 0; row < total_rows; ++row) {
        sort_key.mShares[0](row, 0) *= row_id_stride;
        sort_key.mShares[1](row, 0) *= row_id_stride;
    }
    sort_key = sort_key + row_ids;

    std::vector<si64Matrix> sort_columns;
    sort_columns.reserve(2 + leaf_count);
    sort_columns.push_back(row_ids);
    sort_columns.push_back(values);
    for (const auto& membership_int : membership_ints) {
        sort_columns.push_back(membership_int);
    }
    auto sort_payload = int_columns_to_matrix(sort_columns);
    const auto sort_started = SteadyClock::now();
    quick_sort_with_payload_matrix(
        sort_key,
        sort_payload,
        role,
        enc,
        eval,
        runtime,
        static_cast<std::size_t>(secure_sort_min_size));
    const double sort_elapsed = elapsed_seconds_since(sort_started);

    auto row_ids_sorted = int_column_slice(sort_payload, 0);
    auto values_sorted = int_column_slice(sort_payload, 1);
    auto same_previous = adjacent_group_equal_flags(values_sorted, true, role, eval, runtime);
    auto same_next = adjacent_group_equal_flags(values_sorted, false, role, eval, runtime);
    auto zero_rows = shared_zero_int_matrix(total_rows, 1);

    std::vector<MaterializedLeafResult> results;
    std::vector<si64Matrix> sort_back_columns;
    std::vector<si64Matrix> bucket_values_by_leaf;
    results.reserve(leaf_count);
    sort_back_columns.reserve(leaf_count * 2);
    bucket_values_by_leaf.reserve(leaf_count);

    double per_leaf_scan_elapsed = 0.0;
    for (std::size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
        const auto leaf_scan_started = SteadyClock::now();
        const auto& leaf_doc = leaf_docs[leaf_idx];
        auto membership_sorted = int_column_slice(sort_payload, static_cast<u64>(2 + leaf_idx));
        auto prefix_counts = segmented_prefix_sum(membership_sorted, same_previous, role, enc, eval, runtime);
        auto suffix_counts = segmented_suffix_sum(membership_sorted, same_next, role, enc, eval, runtime);
        auto group_counts = prefix_counts + suffix_counts - membership_sorted;
        auto group_empty = int_eq_public(group_counts, 0, role, eval, runtime);
        auto group_has_member = bool_not_matrix(group_empty, role);
        auto group_start = bool_not_matrix(same_previous, role);
        auto active_group_start = bool_and_matrix(group_start, group_has_member, role, enc, eval, runtime);
        auto active_group_start_int = bool_to_arith_matrix(active_group_start, role, enc, eval, runtime);
        auto active_prefix = prefix_sum(active_group_start_int, role, enc, eval, runtime);
        auto group_start_ranks = select_si64_by_bool(
            active_prefix,
            zero_rows,
            active_group_start,
            role,
            enc,
            eval,
            runtime);
        auto group_ranks_sorted = segmented_prefix_sum(
            group_start_ranks,
            same_previous,
            role,
            enc,
            eval,
            runtime);

        si64Matrix bucket_values(public_leaf_bucket_width, 1);
        bucket_values.mShares[0].setZero();
        bucket_values.mShares[1].setZero();
        for (u64 bucket_idx = 0; bucket_idx < public_leaf_bucket_width; ++bucket_idx) {
            auto rank_match = int_eq_public(active_prefix, static_cast<i64>(bucket_idx + 1), role, eval, runtime);
            auto bucket_start = bool_and_matrix(rank_match, active_group_start, role, enc, eval, runtime);
            auto selected_values = arith_mul_bool(values_sorted, bucket_start, role, enc, eval, runtime);
            bucket_values.mShares[0](bucket_idx, 0) = selected_values.mShares[0].sum();
            bucket_values.mShares[1](bucket_idx, 0) = selected_values.mShares[1].sum();
        }

        auto overflow_rank_match = int_eq_public(
            active_prefix,
            static_cast<i64>(public_leaf_bucket_width + 1),
            role,
            eval,
            runtime);
        auto overflow_starts = bool_and_matrix(
            overflow_rank_match,
            active_group_start,
            role,
            enc,
            eval,
            runtime);
        auto leaf_overflow = bool_reduce_or_matrix(overflow_starts, role, enc, eval, runtime);
        model_overflow = bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);

        sort_back_columns.push_back(std::move(membership_sorted));
        sort_back_columns.push_back(std::move(group_ranks_sorted));
        bucket_values_by_leaf.push_back(std::move(bucket_values));

        MaterializedLeafResult result;
        result.leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
        const auto product_group_signed = leaf_doc.value("product_group_id", std::int64_t(-1));
        result.product_group_id = static_cast<std::uint64_t>(product_group_signed);
        result.public_bucket_width = public_leaf_bucket_width;
        result.real_bucket_count = 0;
        result.bucket_mode = "fixed_cap";
        result.include_row_values = include_row_values_by_leaf[leaf_idx];
        results.push_back(result);
        const double leaf_scan_elapsed = elapsed_seconds_since(leaf_scan_started);
        per_leaf_scan_elapsed += leaf_scan_elapsed;
        secure_leaf_profile_log(
            role,
            profile_level,
            SecureLeafProfileLevel::Leaf,
            {
                {"event", "sorted_group_leaf_scan"},
                {"group_index", static_cast<std::uint64_t>(group_index)},
                {"batch_index", static_cast<std::uint64_t>(batch_index)},
                {"leaf_index_in_batch", static_cast<std::uint64_t>(leaf_idx)},
                {"leaf_node_id", result.leaf_node_id},
                {"relationship_id", leaf_doc.value("relationship_id", std::string())},
                {"multiplier_kind", leaf_doc.value("multiplier_kind", std::string("mu"))},
                {"rows", total_rows},
                {"cap", public_leaf_bucket_width},
                {"elapsed_seconds", leaf_scan_elapsed},
            });
    }

    auto sort_back_key = row_ids_sorted;
    auto sort_back_payload = int_columns_to_matrix(sort_back_columns);
    const auto sort_back_started = SteadyClock::now();
    quick_sort_with_payload_matrix(
        sort_back_key,
        sort_back_payload,
        role,
        enc,
        eval,
        runtime,
        static_cast<std::size_t>(secure_sort_min_size));
    const double sort_back_elapsed = elapsed_seconds_since(sort_back_started);

    ensure_dir(output_prefix);
    const std::string role_dir = join_path(output_prefix, role_name);
    ensure_dir(role_dir);
    double bitmap_write_elapsed = 0.0;
    for (std::size_t leaf_idx = 0; leaf_idx < leaf_count; ++leaf_idx) {
        const auto bitmap_started = SteadyClock::now();
        auto membership_by_row = int_column_slice(sort_back_payload, static_cast<u64>(leaf_idx * 2));
        auto group_rank_by_row = int_column_slice(sort_back_payload, static_cast<u64>(leaf_idx * 2 + 1));
        auto membership_by_row_bool = int_eq_public(membership_by_row, 1, role, eval, runtime);

        sbMatrix bucket_bitmaps(public_leaf_bucket_width * total_rows, 1);
        bool_init_false(role, bucket_bitmaps);
        for (u64 bucket_idx = 0; bucket_idx < public_leaf_bucket_width; ++bucket_idx) {
            auto rank_match = int_eq_public(group_rank_by_row, static_cast<i64>(bucket_idx + 1), role, eval, runtime);
            auto bucket_rows = bool_and_matrix(rank_match, membership_by_row_bool, role, enc, eval, runtime);
            const u64 out_begin = bucket_idx * total_rows;
            for (u64 row = 0; row < total_rows; ++row) {
                bucket_bitmaps.mShares[0](out_begin + row, 0) = bucket_rows.mShares[0](row, 0);
                bucket_bitmaps.mShares[1](out_begin + row, 0) = bucket_rows.mShares[1](row, 0);
            }
        }

        const std::string leaf_prefix = "leaf_" + std::to_string(results[leaf_idx].leaf_node_id);
        write_share_pair_matrix(
            join_path(role_dir, leaf_prefix + ".bucket_values.shares.bin"),
            bucket_values_by_leaf[leaf_idx]);
        write_share_pair_matrix(
            join_path(role_dir, leaf_prefix + ".bucket_lowers.shares.bin"),
            bucket_values_by_leaf[leaf_idx]);
        write_share_pair_matrix(
            join_path(role_dir, leaf_prefix + ".bucket_uppers.shares.bin"),
            bucket_values_by_leaf[leaf_idx]);
        write_bool_share_pair_matrix_bitpacked(
            join_path(role_dir, leaf_prefix + ".leaf_bitmaps.shares.bin"),
            bucket_bitmaps);
        if (results[leaf_idx].include_row_values) {
            write_share_pair_matrix(
                join_path(role_dir, leaf_prefix + ".row_values.shares.bin"),
                row_values_by_leaf[leaf_idx]);
        }
        const double leaf_bitmap_elapsed = elapsed_seconds_since(bitmap_started);
        bitmap_write_elapsed += leaf_bitmap_elapsed;
        secure_leaf_profile_log(
            role,
            profile_level,
            SecureLeafProfileLevel::Leaf,
            {
                {"event", "sorted_group_leaf_bitmap"},
                {"group_index", static_cast<std::uint64_t>(group_index)},
                {"batch_index", static_cast<std::uint64_t>(batch_index)},
                {"leaf_index_in_batch", static_cast<std::uint64_t>(leaf_idx)},
                {"leaf_node_id", results[leaf_idx].leaf_node_id},
                {"rows", total_rows},
                {"cap", public_leaf_bucket_width},
                {"elapsed_seconds", leaf_bitmap_elapsed},
            });
    }

    secure_leaf_profile_log(
        role,
        profile_level,
        SecureLeafProfileLevel::Batch,
        {
            {"event", "sorted_group_batch"},
            {"group_index", static_cast<std::uint64_t>(group_index)},
            {"batch_index", static_cast<std::uint64_t>(batch_index)},
            {"batch_count", static_cast<std::uint64_t>(group_batch_count)},
            {"batch_leaf_count", static_cast<std::uint64_t>(leaf_count)},
            {"rows", total_rows},
            {"cap", public_leaf_bucket_width},
            {"read_membership_seconds", read_membership_elapsed},
            {"row_value_mask_seconds", row_value_mask_elapsed},
            {"sort_seconds", sort_elapsed},
            {"per_leaf_scan_seconds", per_leaf_scan_elapsed},
            {"sort_back_seconds", sort_back_elapsed},
            {"bitmap_write_seconds", bitmap_write_elapsed},
            {"elapsed_seconds", elapsed_seconds_since(batch_started)},
        });

    return results;
}

void write_secure_leaf_counts_manifest(
    const std::string& output_prefix,
    const std::vector<MaterializedLeafResult>& leaves,
    u64 total_rows)
{
    std::ostringstream content;
    content << "{\n";
    content << "  \"format_name\": \"BSPN_SECURE_MULTIPLIER_LEAF_COUNTS\",\n";
    content << "  \"format_version\": 2,\n";
    content << "  \"bool_share_encoding\": \"bitpacked_pair_lsb_v1\",\n";
    content << "  \"total_rows\": " << total_rows << ",\n";
    content << "  \"leaves\": [\n";
    for (std::size_t idx = 0; idx < leaves.size(); ++idx) {
        const auto& leaf = leaves[idx];
        content << "    {";
        content << "\"leaf_node_id\": " << leaf.leaf_node_id << ", ";
        content << "\"product_group_id\": " << leaf.product_group_id << ", ";
        content << "\"public_bucket_width\": " << leaf.public_bucket_width << ", ";
        if (leaf.bucket_mode != "fixed_cap") {
            content << "\"real_bucket_count\": " << leaf.real_bucket_count << ", ";
        }
        content << "\"secure_multiplier_leaf_bucket_mode\": \"" << leaf.bucket_mode << "\", ";
        content << "\"bucket_values_share_file\": \"leaf_" << leaf.leaf_node_id << ".bucket_values.shares.bin\", ";
        content << "\"bucket_lowers_share_file\": \"leaf_" << leaf.leaf_node_id << ".bucket_lowers.shares.bin\", ";
        content << "\"bucket_uppers_share_file\": \"leaf_" << leaf.leaf_node_id << ".bucket_uppers.shares.bin\", ";
        content << "\"leaf_bitmaps_share_file\": \"leaf_" << leaf.leaf_node_id << ".leaf_bitmaps.shares.bin\"";
        if (leaf.include_row_values) {
            content << ", \"row_values_share_file\": \"leaf_" << leaf.leaf_node_id << ".row_values.shares.bin\"";
        }
        content << "}" << (idx + 1 == leaves.size() ? "\n" : ",\n");
    }
    content << "  ],\n";
    content << "  \"role_share_dir\": \"role_{role}\"\n";
    content << "}\n";
    write_text_file(join_path(output_prefix, "secure_leaf_counts.json"), content.str());
}

void run_secure_leaf_materialize(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_leaf_materialize mode requires --role in {0,1,2}.");
    }
    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(config.role), ios, enc, eval, runtime);

    const auto plan = read_json_file(config.secure_leaf_plan_path);
    const u64 total_rows = plan.value("total_rows", std::uint64_t(0));
    const std::string bucket_mode = plan.value("secure_multiplier_leaf_bucket_mode", std::string("fixed_cap"));
    if (bucket_mode != "fixed_cap") {
        throw std::runtime_error("secure leaf materialization only supports fixed_cap mode.");
    }
    const u64 public_leaf_bucket_width = std::max<u64>(
        1,
        plan.value("multiplier_fixed_cap", plan.value("public_leaf_bucket_width", std::uint64_t(8))));
    const std::string plan_base_dir = dirname_from_path(config.secure_leaf_plan_path);
    const std::string plan_role_share_dir = plan.value("role_share_dir", std::string("secure_leaf_plan/role_{role}"));

    std::unordered_map<std::string, si64Matrix> row_value_cache;
    sbMatrix model_overflow = shared_false_bool_matrix(1, 1, config.role);
    const bool use_sorted_group = secure_leaf_sorted_group_enabled();
    const auto strategy_name = secure_leaf_materialize_strategy_name();
    const auto profile_level = secure_leaf_profile_level();
    const auto leaves = plan.value("leaves", json::array());
    std::vector<MaterializedLeafResult> results;

    secure_leaf_profile_log(
        config.role,
        profile_level,
        SecureLeafProfileLevel::Group,
        {
            {"event", "materialize_start"},
            {"strategy", strategy_name},
            {"rows", total_rows},
            {"cap", public_leaf_bucket_width},
            {"secure_leaf_count", static_cast<std::uint64_t>(leaves.size())},
        });

    if (use_sorted_group) {
        std::vector<std::string> group_order;
        std::unordered_map<std::string, std::vector<json>> leaves_by_row_value_source;
        for (const auto& leaf_doc : leaves) {
            const std::string row_value_key = secure_leaf_row_value_cache_key(
                plan,
                leaf_doc,
                plan_base_dir,
                plan_role_share_dir,
                config.role);
            auto insert_result = leaves_by_row_value_source.emplace(row_value_key, std::vector<json>{});
            if (insert_result.second) {
                group_order.push_back(row_value_key);
            }
            insert_result.first->second.push_back(leaf_doc);
        }

        const std::size_t batch_size = secure_leaf_sort_batch_size();
        secure_leaf_profile_log(
            config.role,
            profile_level,
            SecureLeafProfileLevel::Group,
            {
                {"event", "sorted_group_plan"},
                {"row_value_group_count", static_cast<std::uint64_t>(group_order.size())},
                {"sort_batch_size", static_cast<std::uint64_t>(batch_size)},
            });
        for (std::size_t group_idx = 0; group_idx < group_order.size(); ++group_idx) {
            const auto& row_value_key = group_order[group_idx];
            const auto group_started = SteadyClock::now();
            auto cache_iter = row_value_cache.find(row_value_key);
            if (cache_iter == row_value_cache.end()) {
                const auto load_started = SteadyClock::now();
                const auto& first_leaf_doc = leaves_by_row_value_source[row_value_key].front();
                auto row_values = load_secure_leaf_row_values(
                    plan,
                    first_leaf_doc,
                    plan_base_dir,
                    plan_role_share_dir,
                    total_rows,
                    config.role);
                cache_iter = row_value_cache.emplace(row_value_key, std::move(row_values)).first;
                secure_leaf_profile_log(
                    config.role,
                    profile_level,
                    SecureLeafProfileLevel::Batch,
                    {
                        {"event", "sorted_group_row_values_load"},
                        {"group_index", static_cast<std::uint64_t>(group_idx)},
                        {"relationship_id", first_leaf_doc.value("relationship_id", std::string())},
                        {"multiplier_kind", first_leaf_doc.value("multiplier_kind", std::string("mu"))},
                        {"rows", total_rows},
                        {"elapsed_seconds", elapsed_seconds_since(load_started)},
                    });
            }
            const auto& group_leaves = leaves_by_row_value_source[row_value_key];
            const std::size_t group_batch_count = (group_leaves.size() + batch_size - 1) / batch_size;
            secure_leaf_profile_log(
                config.role,
                profile_level,
                SecureLeafProfileLevel::Group,
                {
                    {"event", "sorted_group_start"},
                    {"group_index", static_cast<std::uint64_t>(group_idx)},
                    {"row_value_key_hash", static_cast<std::uint64_t>(std::hash<std::string>{}(row_value_key))},
                    {"relationship_id", group_leaves.front().value("relationship_id", std::string())},
                    {"multiplier_kind", group_leaves.front().value("multiplier_kind", std::string("mu"))},
                    {"leaf_count", static_cast<std::uint64_t>(group_leaves.size())},
                    {"batch_count", static_cast<std::uint64_t>(group_batch_count)},
                });
            for (std::size_t begin = 0; begin < group_leaves.size(); begin += batch_size) {
                const std::size_t end = std::min(group_leaves.size(), begin + batch_size);
                const std::size_t batch_idx = begin / batch_size;
                std::vector<json> batch(group_leaves.begin() + static_cast<std::ptrdiff_t>(begin),
                                        group_leaves.begin() + static_cast<std::ptrdiff_t>(end));
                auto batch_results = materialize_secure_leaf_sorted_group_batch(
                    batch,
                    plan_base_dir,
                    plan_role_share_dir,
                    config.output_prefix,
                    cache_iter->second,
                    total_rows,
                    public_leaf_bucket_width,
                    config.secure_sort_min_size,
                    config.role,
                    enc,
                    eval,
                    runtime,
                    model_overflow,
                    profile_level,
                    group_idx,
                    batch_idx,
                    group_batch_count);
                results.insert(results.end(), batch_results.begin(), batch_results.end());
            }
            secure_leaf_profile_log(
                config.role,
                profile_level,
                SecureLeafProfileLevel::Group,
                {
                    {"event", "sorted_group_end"},
                    {"group_index", static_cast<std::uint64_t>(group_idx)},
                    {"leaf_count", static_cast<std::uint64_t>(group_leaves.size())},
                    {"elapsed_seconds", elapsed_seconds_since(group_started)},
                });
        }
    } else {
        for (std::size_t leaf_idx = 0; leaf_idx < leaves.size(); ++leaf_idx) {
            const auto& leaf_doc = leaves[leaf_idx];
            const std::string row_value_key = secure_leaf_row_value_cache_key(
                plan,
                leaf_doc,
                plan_base_dir,
                plan_role_share_dir,
                config.role);
            auto cache_iter = row_value_cache.find(row_value_key);
            if (cache_iter == row_value_cache.end()) {
                auto row_values = load_secure_leaf_row_values(
                    plan,
                    leaf_doc,
                    plan_base_dir,
                    plan_role_share_dir,
                    total_rows,
                    config.role);
                cache_iter = row_value_cache.emplace(row_value_key, std::move(row_values)).first;
            }
            results.push_back(materialize_one_secure_leaf_fixed_cap(
                leaf_doc,
                plan_base_dir,
                plan_role_share_dir,
                config.output_prefix,
                cache_iter->second,
                total_rows,
                public_leaf_bucket_width,
                config.role,
                enc,
                eval,
                runtime,
                model_overflow,
                profile_level,
                leaf_idx,
                leaves.size()));
        }
    }
    i64Matrix plain_overflow(1, 1);
    enc.revealAll(runtime, model_overflow, plain_overflow).get();
    if ((plain_overflow(0, 0) & 1) != 0) {
        throw std::runtime_error("multiplier_fixed_cap too small");
    }
    if (config.role == 0) {
        ensure_dir(config.output_prefix);
        write_secure_leaf_counts_manifest(config.output_prefix, results, total_rows);
    }
    secure_leaf_profile_log(
        config.role,
        profile_level,
        SecureLeafProfileLevel::Group,
        {
            {"event", "materialize_end"},
            {"strategy", strategy_name},
            {"rows", total_rows},
            {"cap", public_leaf_bucket_width},
            {"secure_leaf_count", static_cast<std::uint64_t>(leaves.size())},
            {"status", "ok"},
        });
}

}  // namespace

int BSPN_multiplier_preprocess(const CLP& cmd)
{
    const auto config = parse_config(cmd);

    if (config.mode == "secure_shared_values") {
        run_secure_multiplier_shared_values(config);
        return 0;
    }

    if (config.mode == "secure_leaf_materialize") {
        run_secure_leaf_materialize(config);
        return 0;
    }

    throw std::runtime_error(
        "Unsupported --multiplier_mode. Use secure_shared_values or secure_leaf_materialize.");
}

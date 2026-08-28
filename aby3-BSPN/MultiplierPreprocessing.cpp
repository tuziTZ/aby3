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
#include <numeric>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
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

bool secure_multiplier_provider_separated_sorted_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_PROVIDER_SEPARATED_SORTED", false);
}

bool secure_multiplier_provider_separated_oblivious_merge_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_PROVIDER_SEPARATED_OBLIVIOUS_MERGE", false);
}

bool secure_multiplier_provider_separated_partitioned_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_PROVIDER_SEPARATED_PARTITIONED", false);
}

bool secure_multiplier_provider_local_fk_group_count_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_PROVIDER_LOCAL_FK_GROUP_COUNT", false);
}

bool secure_multiplier_layer_batch_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_LAYER_BATCH_COMPARE_EXCHANGE", true);
}

bool secure_multiplier_public_profile_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_PROFILE_PUBLIC", false);
}

bool secure_multiplier_streaming_artifact_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_STREAMING_ARTIFACT", false);
}

bool secure_multiplier_forbid_server_reveal_enabled()
{
    return env_flag_enabled("BSPN_MULTIPLIER_FORBID_SERVER_REVEAL", false);
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

std::vector<i64> parse_i64_csv_env(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return {};
    }
    std::vector<i64> values;
    std::stringstream stream(raw);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), token.end());
        if (token.empty()) {
            throw std::runtime_error(std::string("Empty integer token in ") + name + ".");
        }
        std::size_t parsed = 0;
        i64 value = 0;
        try {
            value = std::stoll(token, &parsed, 10);
        } catch (const std::exception&) {
            throw std::runtime_error(std::string("Invalid integer token in ") + name + ": " + token);
        }
        if (parsed != token.size()) {
            throw std::runtime_error(std::string("Invalid integer token in ") + name + ": " + token);
        }
        values.push_back(value);
    }
    return values;
}

std::vector<u64> parse_u64_csv_env(const char* name)
{
    const auto signed_values = parse_i64_csv_env(name);
    std::vector<u64> values;
    values.reserve(signed_values.size());
    for (i64 value : signed_values) {
        if (value < 0) {
            throw std::runtime_error(std::string("Negative capacity in ") + name + ".");
        }
        values.push_back(static_cast<u64>(value));
    }
    return values;
}

std::string join_i64_values(const std::vector<i64>& values)
{
    std::ostringstream out;
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            out << ",";
        }
        out << values[idx];
    }
    return out.str();
}

std::string join_u64_values(const std::vector<u64>& values)
{
    std::ostringstream out;
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (idx != 0) {
            out << ",";
        }
        out << values[idx];
    }
    return out.str();
}

bool public_domain_width_covers_capacities(
    const std::vector<i64>& boundaries,
    const std::vector<u64>& capacities)
{
    if (boundaries.size() < 2 || capacities.size() + 1 != boundaries.size()) {
        return false;
    }
    for (std::size_t idx = 0; idx < capacities.size(); ++idx) {
        const __int128 lower = static_cast<__int128>(boundaries[idx]);
        const __int128 upper = static_cast<__int128>(boundaries[idx + 1]);
        if (upper < lower) {
            return false;
        }
        __int128 width = upper - lower;
        if (idx + 1 == capacities.size()) {
            width += 1;
        }
        if (width < 0 || width > static_cast<__int128>(std::numeric_limits<u64>::max())) {
            return false;
        }
        if (static_cast<u64>(width) > capacities[idx]) {
            return false;
        }
    }
    return true;
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
sbMatrix local_bool_not_matrix(sbMatrix value, int role)
{
    sbMatrix out(value.rows(), value.bitCount());
    bool_cipher_not(role, value, out);
    return out;
}

sbMatrix local_bool_and_matrix(sbMatrix lhs, sbMatrix rhs, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime)
{
    sbMatrix out(lhs.rows(), lhs.bitCount());
    bool_cipher_and(role, lhs, rhs, out, enc, eval, runtime);
    return out;
}

sbMatrix local_bool_or_matrix(sbMatrix lhs, sbMatrix rhs, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime)
{
    sbMatrix out(lhs.rows(), lhs.bitCount());
    bool_cipher_or(role, lhs, rhs, out, enc, eval, runtime);
    return out;
}
sbMatrix shared_false_bool_matrix(u64 rows, u64 bit_count, int role);

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

template<typename Fn>
void for_each_key_column_csv(
    const std::string& csv_path,
    u64 key_column,
    bool has_header,
    Fn&& callback)
{
    std::ifstream input(csv_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_path);
    }

    std::string line;
    bool is_first_line = true;
    u64 row_index = 0;
    while (std::getline(input, line)) {
        if (is_first_line && has_header) {
            is_first_line = false;
            continue;
        }
        is_first_line = false;
        NormalizedKey key;
        if (!line.empty()) {
            auto fields = parse_csv_line(line);
            if (key_column >= fields.size()) {
                throw std::runtime_error(
                    "Requested key column out of range for CSV file: " + csv_path +
                    " column=" + std::to_string(key_column));
            }
            key = normalize_join_key_token(fields[key_column]);
        }
        callback(key, row_index);
        ++row_index;
    }
}

std::vector<NormalizedKey> load_key_column_csv(
    const std::string& csv_path,
    u64 key_column,
    bool has_header)
{
    std::vector<NormalizedKey> keys;
    for_each_key_column_csv(csv_path, key_column, has_header, [&](const NormalizedKey& key, u64) {
        keys.push_back(key);
    });
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

void write_share_pair_matrix_atomic(const std::string& path, const si64Matrix& values)
{
    const std::string tmp_path = path + ".tmp";
    write_share_pair_matrix(tmp_path, values);
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("Failed to publish share output file " + path + ": " + std::strerror(errno));
    }
}

void append_share_pair_matrix(const std::string& path, const si64Matrix& values)
{
    std::ofstream output(path, std::ios::binary | std::ios::app);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open share append file: " + path);
    }
    for (u64 row = 0; row < values.rows(); ++row) {
        const i64 lhs = values.mShares[0](row, 0);
        const i64 rhs = values.mShares[1](row, 0);
        output.write(reinterpret_cast<const char*>(&lhs), sizeof(i64));
        output.write(reinterpret_cast<const char*>(&rhs), sizeof(i64));
    }
}

std::uint64_t file_size_bytes(const std::string& path)
{
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("Failed to stat file " + path + ": " + std::strerror(errno));
    }
    return static_cast<std::uint64_t>(st.st_size);
}

std::string shell_quote_path(const std::string& path)
{
    std::string quoted = "'";
    for (char ch : path) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string sha256_file_external(const std::string& path)
{
    const std::string command = "sha256sum " + shell_quote_path(path);
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("Failed to run sha256sum for file: " + path);
    }
    char buffer[256];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    const int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("sha256sum failed for file: " + path);
    }
    std::istringstream stream(output);
    std::string digest;
    stream >> digest;
    if (digest.size() != 64) {
        throw std::runtime_error("sha256sum returned malformed digest for file: " + path);
    }
    return digest;
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
    const bool provider_separated_sorted = provider_separated && secure_multiplier_provider_separated_sorted_enabled();
    const bool provider_separated_oblivious = provider_separated &&
        secure_multiplier_provider_separated_oblivious_merge_enabled();
    const bool provider_separated_partitioned = provider_separated &&
        secure_multiplier_provider_separated_partitioned_enabled();
    const bool provider_local_fk_group_count = provider_separated_partitioned &&
        secure_multiplier_provider_local_fk_group_count_enabled();
    const std::string secure_core_status = fast_preprocess
        ? "legacy_plaintext_count_then_secret_share"
        : (provider_separated
            ? (provider_separated_partitioned
                ? (provider_local_fk_group_count
                    ? "provider_local_distinct_fk_partitioned_oblivious_merge"
                    : "provider_separated_partitioned_oblivious_merge_group_count")
                : (provider_separated_oblivious
                ? "provider_separated_oblivious_merge_group_count"
                : (provider_separated_sorted
                ? "diagnostic_provider_separated_sort_group_count_reveals_sort_comparisons"
                : "pairwise_secret_shared_key_equality_count_no_reveal")))
            : "centralized_plaintext_input_party_sorted_segmented_scan_count_no_reveal");
    const auto partition_boundaries = parse_i64_csv_env("BSPN_MULTIPLIER_PARTITION_BOUNDARIES");
    const auto partition_pk_capacities = parse_u64_csv_env("BSPN_MULTIPLIER_PARTITION_PK_CAPACITIES");
    const auto partition_fk_capacities = parse_u64_csv_env("BSPN_MULTIPLIER_PARTITION_FK_CAPACITIES");
    const bool grouped_capacity_is_domain_width =
        provider_local_fk_group_count &&
        public_domain_width_covers_capacities(partition_boundaries, partition_pk_capacities) &&
        public_domain_width_covers_capacities(partition_boundaries, partition_fk_capacities);
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
    content << "  \"construction_mode\": \"" << (
        provider_separated_sorted ? "diagnostic_non_private" : (provider_separated ? "privacy_aligned" : "legacy")
    ) << "\",\n";
    content << "  \"join_key_input_mode\": \"" << join_key_input_mode << "\",\n";
    content << "  \"provider_separated_key_inputs\": " << (provider_separated ? "true" : "false") << ",\n";
    content << "  \"provider_separated_sorted_group_count\": " << (provider_separated_sorted ? "true" : "false") << ",\n";
    content << "  \"provider_separated_oblivious_merge_group_count\": "
            << (provider_separated_oblivious ? "true" : "false") << ",\n";
    content << "  \"provider_separated_partitioned_oblivious_merge_group_count\": "
            << (provider_separated_partitioned ? "true" : "false") << ",\n";
    content << "  \"provider_local_fk_group_count\": " << (provider_local_fk_group_count ? "true" : "false") << ",\n";
    content << "  \"fk_group_count_exact\": " << (provider_local_fk_group_count ? "true" : "false") << ",\n";
    content << "  \"fk_group_multiplicity_secret_shared\": " << (provider_local_fk_group_count ? "true" : "false") << ",\n";
    content << "  \"provider_local_partition\": " << (provider_separated_partitioned ? "true" : "false") << ",\n";
    content << "  \"provider_local_sort\": " << ((provider_separated_oblivious || provider_separated_partitioned) ? "true" : "false") << ",\n";
    content << "  \"server_oblivious_merge\": " << ((provider_separated_oblivious || provider_separated_partitioned) ? "true" : "false") << ",\n";
    content << "  \"sort_core_reveals_comparisons\": " << (provider_separated_sorted ? "true" : "false") << ",\n";
    content << "  \"network_schedule_version\": \""
            << ((provider_separated_oblivious || provider_separated_partitioned) ? "bitonic_compare_exchange_v1" : "not_applicable") << "\",\n";
    content << "  \"partition_boundaries_public\": " << (provider_separated_partitioned ? "true" : "false") << ",\n";
    content << "  \"partition_boundaries_query_independent\": " << (provider_separated_partitioned ? "true" : "false") << ",\n";
    content << "  \"partition_real_cardinalities_hidden\": " << (provider_separated_partitioned ? "true" : "false") << ",\n";
    content << "  \"partition_capacities_public\": " << (provider_separated_partitioned ? "true" : "false") << ",\n";
    content << "  \"partition_boundary_rule\": \""
            << (provider_separated_partitioned ? "public_half_open_ranges_last_closed" : "not_applicable") << "\",\n";
    content << "  \"partition_boundaries\": \"" << json_escape(join_i64_values(partition_boundaries)) << "\",\n";
    content << "  \"partition_pk_capacities\": \"" << json_escape(join_u64_values(partition_pk_capacities)) << "\",\n";
    content << "  \"partition_fk_capacities\": \"" << json_escape(join_u64_values(partition_fk_capacities)) << "\",\n";
    content << "  \"capacity_policy\": \""
            << (grouped_capacity_is_domain_width ? "public_key_domain_width" :
                (provider_local_fk_group_count ? "explicit_public_group_capacity" :
                    (provider_separated_partitioned ? "explicit_public_capacity" : "not_applicable"))) << "\",\n";
    content << "  \"capacity_deterministic\": "
            << (grouped_capacity_is_domain_width ? "true" : "false") << ",\n";
    content << "  \"output_alignment_mode\": \""
            << (provider_separated_partitioned ? "provider_local_partitioned_model_order" : "original_pk_order") << "\",\n";
    content << "  \"overflow_policy\": \"" << (provider_separated_partitioned ? "fail_closed" : "not_applicable") << "\",\n";
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

si64Matrix int_rows_gather(const si64Matrix& src, const std::vector<u64>& rows)
{
    si64Matrix out(static_cast<u64>(rows.size()), src.cols());
    for (u64 out_row = 0; out_row < static_cast<u64>(rows.size()); ++out_row) {
        const u64 src_row = rows[static_cast<std::size_t>(out_row)];
        if (src_row >= src.rows()) {
            throw std::runtime_error("Integer matrix gather row index is out of bounds.");
        }
        for (u64 col = 0; col < src.cols(); ++col) {
            out.mShares[0](out_row, col) = src.mShares[0](src_row, col);
            out.mShares[1](out_row, col) = src.mShares[1](src_row, col);
        }
    }
    return out;
}

void int_rows_scatter(si64Matrix& dst, const std::vector<u64>& rows, const si64Matrix& values)
{
    if (values.rows() != static_cast<u64>(rows.size()) || values.cols() != dst.cols()) {
        throw std::runtime_error("Integer matrix scatter shape mismatch.");
    }
    for (u64 value_row = 0; value_row < static_cast<u64>(rows.size()); ++value_row) {
        const u64 dst_row = rows[static_cast<std::size_t>(value_row)];
        if (dst_row >= dst.rows()) {
            throw std::runtime_error("Integer matrix scatter row index is out of bounds.");
        }
        for (u64 col = 0; col < dst.cols(); ++col) {
            dst.mShares[0](dst_row, col) = values.mShares[0](value_row, col);
            dst.mShares[1](dst_row, col) = values.mShares[1](value_row, col);
        }
    }
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
    auto same_previous = local_bool_not_matrix(shared_false_bool_matrix(values.rows(), 1, role), role);
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

sbMatrix shared_true_bool_matrix(u64 rows, u64 bit_count, int role)
{
    sbMatrix out(rows, bit_count);
    bool_init_true(role, out);
    return out;
}

sbMatrix int_lt_matrix(
    const si64Matrix& lhs,
    const si64Matrix& rhs,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto lhs_copy = lhs;
    auto rhs_copy = rhs;
    sbMatrix out;
    arith_cipher_lt(role, lhs_copy, rhs_copy, out, enc, eval, runtime);
    return out;
}

sbMatrix int_eq_matrix(
    const si64Matrix& lhs,
    const si64Matrix& rhs,
    int role,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto lhs_copy = lhs;
    auto rhs_copy = rhs;
    sbMatrix out;
    cipher_eq(role, lhs_copy, rhs_copy, out, eval, runtime);
    return out;
}

sbMatrix lexicographic_less_row(
    const std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    u64 lhs_row,
    u64 rhs_row,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto less = shared_false_bool_matrix(1, 1, role);
    auto equal_prefix = shared_true_bool_matrix(1, 1, role);
    for (u64 field_idx : field_indices) {
        if (field_idx >= columns.size()) {
            throw std::runtime_error("Oblivious sort comparator field index is out of bounds.");
        }
        auto lhs_field = int_row_slice(columns[static_cast<std::size_t>(field_idx)], lhs_row, 1);
        auto rhs_field = int_row_slice(columns[static_cast<std::size_t>(field_idx)], rhs_row, 1);
        auto lt = int_lt_matrix(lhs_field, rhs_field, role, enc, eval, runtime);
        auto eq = int_eq_matrix(lhs_field, rhs_field, role, eval, runtime);
        auto term = local_bool_and_matrix(equal_prefix, lt, role, enc, eval, runtime);
        less = local_bool_or_matrix(less, term, role, enc, eval, runtime);
        equal_prefix = local_bool_and_matrix(equal_prefix, eq, role, enc, eval, runtime);
    }
    return less;
}

sbMatrix lexicographic_less_rows(
    const std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    const std::vector<u64>& lhs_rows,
    const std::vector<u64>& rhs_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    if (lhs_rows.size() != rhs_rows.size()) {
        throw std::runtime_error("Batched lexicographic comparator row count mismatch.");
    }
    const u64 rows = static_cast<u64>(lhs_rows.size());
    auto less = shared_false_bool_matrix(rows, 1, role);
    auto equal_prefix = shared_true_bool_matrix(rows, 1, role);
    for (u64 field_idx : field_indices) {
        if (field_idx >= columns.size()) {
            throw std::runtime_error("Batched oblivious sort comparator field index is out of bounds.");
        }
        const auto& column = columns[static_cast<std::size_t>(field_idx)];
        auto lhs_field = int_rows_gather(column, lhs_rows);
        auto rhs_field = int_rows_gather(column, rhs_rows);
        auto lt = int_lt_matrix(lhs_field, rhs_field, role, enc, eval, runtime);
        auto eq = int_eq_matrix(lhs_field, rhs_field, role, eval, runtime);
        auto term = local_bool_and_matrix(equal_prefix, lt, role, enc, eval, runtime);
        less = local_bool_or_matrix(less, term, role, enc, eval, runtime);
        equal_prefix = local_bool_and_matrix(equal_prefix, eq, role, enc, eval, runtime);
    }
    return less;
}

sbMatrix choose_public_bool_rows(
    const sbMatrix& if_true,
    const sbMatrix& if_false,
    const std::vector<unsigned char>& choose_true,
    int role)
{
    if (if_true.rows() != if_false.rows() ||
        if_true.bitCount() != if_false.bitCount() ||
        if_true.rows() != static_cast<u64>(choose_true.size())) {
        throw std::runtime_error("Public bool chooser shape mismatch.");
    }
    sbMatrix out(if_true.rows(), if_true.bitCount());
    for (u64 row = 0; row < if_true.rows(); ++row) {
        for (u64 col = 0; col < static_cast<u64>(if_true.mShares[0].cols()); ++col) {
            if (choose_true[static_cast<std::size_t>(row)] != 0) {
                out.mShares[0](row, col) = if_true.mShares[0](row, col);
                out.mShares[1](row, col) = if_true.mShares[1](row, col);
            } else {
                out.mShares[0](row, col) = if_false.mShares[0](row, col);
                out.mShares[1](row, col) = if_false.mShares[1](row, col);
            }
        }
    }
    return out;
}

void oblivious_compare_exchange_rows(
    std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    u64 lhs_row,
    u64 rhs_row,
    bool ascending,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto right_less_left = lexicographic_less_row(
        columns,
        field_indices,
        rhs_row,
        lhs_row,
        role,
        enc,
        eval,
        runtime);
    auto left_less_right = lexicographic_less_row(
        columns,
        field_indices,
        lhs_row,
        rhs_row,
        role,
        enc,
        eval,
        runtime);
    auto swap_flag = ascending ? right_less_left : left_less_right;

    for (auto& column : columns) {
        auto lhs_value = int_row_slice(column, lhs_row, 1);
        auto rhs_value = int_row_slice(column, rhs_row, 1);
        auto new_lhs = select_si64_by_bool(rhs_value, lhs_value, swap_flag, role, enc, eval, runtime);
        auto new_rhs = select_si64_by_bool(lhs_value, rhs_value, swap_flag, role, enc, eval, runtime);
        column.mShares[0](lhs_row, 0) = new_lhs.mShares[0](0, 0);
        column.mShares[1](lhs_row, 0) = new_lhs.mShares[1](0, 0);
        column.mShares[0](rhs_row, 0) = new_rhs.mShares[0](0, 0);
        column.mShares[1](rhs_row, 0) = new_rhs.mShares[1](0, 0);
    }
}

void assert_public_layer_pairs_disjoint(
    const std::vector<u64>& lhs_rows,
    const std::vector<u64>& rhs_rows,
    u64 row_count)
{
    std::vector<unsigned char> seen(static_cast<std::size_t>(row_count), 0);
    for (u64 idx = 0; idx < static_cast<u64>(lhs_rows.size()); ++idx) {
        const u64 lhs = lhs_rows[static_cast<std::size_t>(idx)];
        const u64 rhs = rhs_rows[static_cast<std::size_t>(idx)];
        if (lhs >= row_count || rhs >= row_count || lhs == rhs) {
            throw std::runtime_error("Oblivious network layer has invalid public pair.");
        }
        if (seen[static_cast<std::size_t>(lhs)] || seen[static_cast<std::size_t>(rhs)]) {
            throw std::runtime_error("Oblivious network layer has overlapping public pairs.");
        }
        seen[static_cast<std::size_t>(lhs)] = 1;
        seen[static_cast<std::size_t>(rhs)] = 1;
    }
}

struct StoredObliviousSwapLayer {
    std::vector<u64> lhs_rows;
    std::vector<u64> rhs_rows;
    sbMatrix swap_flags;
};

sbMatrix oblivious_compare_exchange_rows_batched_with_swaps(
    std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    const std::vector<u64>& lhs_rows,
    const std::vector<u64>& rhs_rows,
    const std::vector<unsigned char>& ascending,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    if (lhs_rows.empty()) {
        return shared_false_bool_matrix(0, 1, role);
    }
    if (lhs_rows.size() != rhs_rows.size() || lhs_rows.size() != ascending.size()) {
        throw std::runtime_error("Batched oblivious compare-exchange schedule shape mismatch.");
    }
    assert_public_layer_pairs_disjoint(lhs_rows, rhs_rows, columns.front().rows());
    auto right_less_left = lexicographic_less_rows(
        columns,
        field_indices,
        rhs_rows,
        lhs_rows,
        role,
        enc,
        eval,
        runtime);
    auto left_less_right = lexicographic_less_rows(
        columns,
        field_indices,
        lhs_rows,
        rhs_rows,
        role,
        enc,
        eval,
        runtime);
    auto swap_flags = choose_public_bool_rows(right_less_left, left_less_right, ascending, role);

    for (auto& column : columns) {
        auto lhs_values = int_rows_gather(column, lhs_rows);
        auto rhs_values = int_rows_gather(column, rhs_rows);
        auto new_lhs = select_si64_by_bool(rhs_values, lhs_values, swap_flags, role, enc, eval, runtime);
        auto new_rhs = select_si64_by_bool(lhs_values, rhs_values, swap_flags, role, enc, eval, runtime);
        int_rows_scatter(column, lhs_rows, new_lhs);
        int_rows_scatter(column, rhs_rows, new_rhs);
    }
    return swap_flags;
}

void oblivious_compare_exchange_rows_batched(
    std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    const std::vector<u64>& lhs_rows,
    const std::vector<u64>& rhs_rows,
    const std::vector<unsigned char>& ascending,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    (void)oblivious_compare_exchange_rows_batched_with_swaps(
        columns,
        field_indices,
        lhs_rows,
        rhs_rows,
        ascending,
        role,
        enc,
        eval,
        runtime);
}

void oblivious_apply_stored_swaps_rows_batched(
    std::vector<si64Matrix>& columns,
    const std::vector<u64>& lhs_rows,
    const std::vector<u64>& rhs_rows,
    const sbMatrix& stored_swap_flags,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    if (lhs_rows.empty()) {
        return;
    }
    if (lhs_rows.size() != rhs_rows.size() ||
        stored_swap_flags.rows() != static_cast<u64>(lhs_rows.size())) {
        throw std::runtime_error("Stored oblivious swap layer shape mismatch.");
    }
    assert_public_layer_pairs_disjoint(lhs_rows, rhs_rows, columns.front().rows());
    auto swap_flags = stored_swap_flags;
    for (auto& column : columns) {
        auto lhs_values = int_rows_gather(column, lhs_rows);
        auto rhs_values = int_rows_gather(column, rhs_rows);
        auto new_lhs = select_si64_by_bool(rhs_values, lhs_values, swap_flags, role, enc, eval, runtime);
        auto new_rhs = select_si64_by_bool(lhs_values, rhs_values, swap_flags, role, enc, eval, runtime);
        int_rows_scatter(column, lhs_rows, new_lhs);
        int_rows_scatter(column, rhs_rows, new_rhs);
    }
}

void oblivious_bitonic_sort_rows(
    std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    if (columns.empty()) {
        return;
    }
    const u64 rows = columns.front().rows();
    if (rows == 0) {
        return;
    }
    if ((rows & (rows - 1)) != 0) {
        throw std::runtime_error("Oblivious bitonic sort requires public power-of-two padding.");
    }
    for (const auto& column : columns) {
        if (column.rows() != rows || column.cols() != 1) {
            throw std::runtime_error("Oblivious sort payload columns must all be rows x 1.");
        }
    }

    const bool layer_batch = secure_multiplier_layer_batch_enabled();
    const bool profile = secure_multiplier_public_profile_enabled();
    u64 layer_count = 0;
    u64 comparator_count = 0;
    const auto started = SteadyClock::now();
    for (u64 k = 2; k <= rows; k <<= 1) {
        for (u64 j = k >> 1; j > 0; j >>= 1) {
            ++layer_count;
            std::vector<u64> lhs_rows;
            std::vector<u64> rhs_rows;
            std::vector<unsigned char> ascending_flags;
            if (layer_batch) {
                lhs_rows.reserve(static_cast<std::size_t>(rows / 2));
                rhs_rows.reserve(static_cast<std::size_t>(rows / 2));
                ascending_flags.reserve(static_cast<std::size_t>(rows / 2));
            }
            for (u64 i = 0; i < rows; ++i) {
                const u64 ixj = i ^ j;
                if (ixj <= i) {
                    continue;
                }
                const bool ascending = ((i & k) == 0);
                ++comparator_count;
                if (layer_batch) {
                    lhs_rows.push_back(i);
                    rhs_rows.push_back(ixj);
                    ascending_flags.push_back(ascending ? 1 : 0);
                } else {
                    oblivious_compare_exchange_rows(
                        columns,
                        field_indices,
                        i,
                        ixj,
                        ascending,
                        role,
                        enc,
                        eval,
                        runtime);
                }
            }
            if (layer_batch) {
                oblivious_compare_exchange_rows_batched(
                    columns,
                    field_indices,
                    lhs_rows,
                    rhs_rows,
                    ascending_flags,
                    role,
                    enc,
                    eval,
                    runtime);
            }
        }
    }
    if (profile && role == 0) {
        std::cerr << "bspn_multiplier_profile: event=bitonic_sort"
                  << " rows=" << rows
                  << " payload_columns=" << columns.size()
                  << " field_count=" << field_indices.size()
                  << " layer_batch=" << (layer_batch ? 1 : 0)
                  << " layers=" << layer_count
                  << " comparators=" << comparator_count
                  << " elapsed_seconds=" << elapsed_seconds_since(started)
                  << std::endl;
    }
}

std::vector<StoredObliviousSwapLayer> reversible_bitonic_forward_sort_rows(
    std::vector<si64Matrix>& columns,
    const std::vector<u64>& field_indices,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    u64& layer_count,
    u64& comparator_count)
{
    std::vector<StoredObliviousSwapLayer> stored_layers;
    layer_count = 0;
    comparator_count = 0;
    if (columns.empty()) {
        return stored_layers;
    }
    const u64 rows = columns.front().rows();
    if (rows == 0) {
        return stored_layers;
    }
    if ((rows & (rows - 1)) != 0) {
        throw std::runtime_error("Reversible oblivious bitonic lookup requires public power-of-two padding.");
    }
    for (const auto& column : columns) {
        if (column.rows() != rows || column.cols() != 1) {
            throw std::runtime_error("Reversible oblivious lookup payload columns must all be rows x 1.");
        }
    }

    for (u64 k = 2; k <= rows; k <<= 1) {
        for (u64 j = k >> 1; j > 0; j >>= 1) {
            std::vector<u64> lhs_rows;
            std::vector<u64> rhs_rows;
            std::vector<unsigned char> ascending_flags;
            lhs_rows.reserve(static_cast<std::size_t>(rows / 2));
            rhs_rows.reserve(static_cast<std::size_t>(rows / 2));
            ascending_flags.reserve(static_cast<std::size_t>(rows / 2));
            for (u64 i = 0; i < rows; ++i) {
                const u64 ixj = i ^ j;
                if (ixj <= i) {
                    continue;
                }
                const bool ascending = ((i & k) == 0);
                lhs_rows.push_back(i);
                rhs_rows.push_back(ixj);
                ascending_flags.push_back(ascending ? 1 : 0);
            }
            assert_public_layer_pairs_disjoint(lhs_rows, rhs_rows, rows);
            auto swap_flags = oblivious_compare_exchange_rows_batched_with_swaps(
                columns,
                field_indices,
                lhs_rows,
                rhs_rows,
                ascending_flags,
                role,
                enc,
                eval,
                runtime);
            comparator_count += static_cast<u64>(lhs_rows.size());
            ++layer_count;
            stored_layers.push_back(
                StoredObliviousSwapLayer{
                    std::move(lhs_rows),
                    std::move(rhs_rows),
                    std::move(swap_flags),
                });
        }
    }
    return stored_layers;
}

void reversible_bitonic_reverse_rows(
    std::vector<si64Matrix>& columns,
    const std::vector<StoredObliviousSwapLayer>& stored_layers,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    for (auto layer_it = stored_layers.rbegin(); layer_it != stored_layers.rend(); ++layer_it) {
        oblivious_apply_stored_swaps_rows_batched(
            columns,
            layer_it->lhs_rows,
            layer_it->rhs_rows,
            layer_it->swap_flags,
            role,
            enc,
            eval,
            runtime);
    }
}

struct ProviderLocalKeyRow {
    NormalizedKey key;
    u64 original_index = 0;
    i64 multiplicity = 1;
};

struct PartitionContract {
    std::vector<i64> boundaries;
    std::vector<u64> pk_capacities;
    std::vector<u64> fk_capacities;
};

struct ProviderDenseDomainState {
    i64 domain_min = 0;
    u64 domain_width = 0;
    u64 raw_row_count = 0;
    std::vector<std::uint8_t> pk_present;
    std::vector<i64> fk_counts;
    bool contract_failed = false;
};

PartitionContract parse_partition_contract()
{
    PartitionContract contract;
    contract.boundaries = parse_i64_csv_env("BSPN_MULTIPLIER_PARTITION_BOUNDARIES");
    contract.pk_capacities = parse_u64_csv_env("BSPN_MULTIPLIER_PARTITION_PK_CAPACITIES");
    contract.fk_capacities = parse_u64_csv_env("BSPN_MULTIPLIER_PARTITION_FK_CAPACITIES");
    if (contract.boundaries.size() < 2) {
        throw std::runtime_error("Partitioned multiplier requires at least two public boundaries.");
    }
    for (std::size_t idx = 1; idx < contract.boundaries.size(); ++idx) {
        if (contract.boundaries[idx] <= contract.boundaries[idx - 1]) {
            throw std::runtime_error("Partition boundaries must be strictly increasing public values.");
        }
    }
    const std::size_t partition_count = contract.boundaries.size() - 1;
    if (contract.pk_capacities.size() != partition_count ||
        contract.fk_capacities.size() != partition_count) {
        throw std::runtime_error("Partition capacity lists must match the public partition count.");
    }
    for (std::size_t idx = 0; idx < partition_count; ++idx) {
        if (contract.pk_capacities[idx] == 0 || contract.fk_capacities[idx] == 0) {
            throw std::runtime_error("Partition capacities must be positive public values.");
        }
    }
    return contract;
}

u64 partition_count(const PartitionContract& contract)
{
    return static_cast<u64>(contract.boundaries.size() - 1);
}

std::vector<u64> public_partition_domain_widths(const PartitionContract& contract)
{
    std::vector<u64> widths;
    const u64 partitions = partition_count(contract);
    widths.reserve(static_cast<std::size_t>(partitions));
    for (u64 part = 0; part < partitions; ++part) {
        const i64 lower = contract.boundaries[static_cast<std::size_t>(part)];
        const i64 upper = contract.boundaries[static_cast<std::size_t>(part + 1)];
        __int128 signed_width = static_cast<__int128>(upper) - static_cast<__int128>(lower);
        if (part + 1 == partitions) {
            signed_width += 1;
        }
        if (signed_width <= 0) {
            throw std::runtime_error("Invalid public partition domain width.");
        }
        const unsigned __int128 width = static_cast<unsigned __int128>(signed_width);
        if (width > std::numeric_limits<u64>::max()) {
            throw std::runtime_error("Public partition domain width exceeds uint64 capacity.");
        }
        widths.push_back(static_cast<u64>(width));
    }
    return widths;
}

bool capacities_cover_public_domain_widths(
    const PartitionContract& contract,
    const std::vector<u64>& capacities)
{
    const auto widths = public_partition_domain_widths(contract);
    if (widths.size() != capacities.size()) {
        return false;
    }
    for (std::size_t idx = 0; idx < widths.size(); ++idx) {
        if (capacities[idx] < widths[idx]) {
            return false;
        }
    }
    return true;
}

u64 sum_u64(const std::vector<u64>& values)
{
    u64 total = 0;
    for (u64 value : values) {
        total += value;
    }
    return total;
}

u64 public_domain_width_inclusive(const PartitionContract& contract)
{
    if (contract.boundaries.size() < 2) {
        throw std::runtime_error("Partition contract has no public domain.");
    }
    const __int128 lower = static_cast<__int128>(contract.boundaries.front());
    const __int128 upper = static_cast<__int128>(contract.boundaries.back());
    const __int128 width = upper - lower + 1;
    if (width <= 0 || width > static_cast<__int128>(std::numeric_limits<u64>::max())) {
        throw std::runtime_error("Public domain width is invalid.");
    }
    return static_cast<u64>(width);
}

u64 public_domain_offset(const PartitionContract& contract, const NormalizedKey& key)
{
    if (key.is_null) {
        throw std::runtime_error("null key has no public domain offset");
    }
    const i64 domain_min = contract.boundaries.front();
    const i64 domain_max = contract.boundaries.back();
    if (key.value < domain_min || key.value > domain_max) {
        throw std::runtime_error("public capacity contract not satisfied");
    }
    const unsigned __int128 offset =
        static_cast<unsigned __int128>(static_cast<__int128>(key.value) - static_cast<__int128>(domain_min));
    if (offset > std::numeric_limits<u64>::max()) {
        throw std::runtime_error("Public domain offset exceeds uint64 capacity.");
    }
    return static_cast<u64>(offset);
}

std::pair<u64, u64> public_partition_offset_range(const PartitionContract& contract, u64 part)
{
    const u64 partitions = partition_count(contract);
    if (part >= partitions) {
        throw std::runtime_error("Public partition index out of range.");
    }
    const i64 domain_min = contract.boundaries.front();
    const i64 lower = contract.boundaries[static_cast<std::size_t>(part)];
    const i64 upper = contract.boundaries[static_cast<std::size_t>(part + 1)];
    const __int128 begin_signed = static_cast<__int128>(lower) - static_cast<__int128>(domain_min);
    __int128 end_signed = static_cast<__int128>(upper) - static_cast<__int128>(domain_min);
    if (part + 1 == partitions) {
        end_signed += 1;
    }
    if (begin_signed < 0 || end_signed < begin_signed ||
        end_signed > static_cast<__int128>(std::numeric_limits<u64>::max())) {
        throw std::runtime_error("Invalid public partition offset range.");
    }
    return {static_cast<u64>(begin_signed), static_cast<u64>(end_signed)};
}

u64 assign_public_partition(const PartitionContract& contract, const NormalizedKey& key)
{
    if (key.is_null) {
        return 0;
    }
    const i64 domain_min = contract.boundaries.front();
    const i64 domain_max = contract.boundaries.back();
    if (key.value < domain_min || key.value > domain_max) {
        throw std::runtime_error("public capacity contract not satisfied");
    }
    const u64 partitions = partition_count(contract);
    for (u64 part = 0; part < partitions; ++part) {
        const i64 lower = contract.boundaries[static_cast<std::size_t>(part)];
        const i64 upper = contract.boundaries[static_cast<std::size_t>(part + 1)];
        const bool in_range = (part + 1 == partitions)
            ? (key.value >= lower && key.value <= upper)
            : (key.value >= lower && key.value < upper);
        if (in_range) {
            return part;
        }
    }
    throw std::runtime_error("public capacity contract not satisfied");
}

std::vector<ProviderLocalKeyRow> provider_local_sorted_rows(const std::vector<NormalizedKey>& keys)
{
    std::vector<ProviderLocalKeyRow> rows;
    rows.reserve(keys.size());
    for (u64 idx = 0; idx < static_cast<u64>(keys.size()); ++idx) {
        rows.push_back({keys[static_cast<std::size_t>(idx)], idx, 1});
    }
    std::sort(rows.begin(), rows.end(), [](const ProviderLocalKeyRow& lhs, const ProviderLocalKeyRow& rhs) {
        if (lhs.key.is_null != rhs.key.is_null) {
            return !lhs.key.is_null && rhs.key.is_null;
        }
        if (lhs.key.value != rhs.key.value) {
            return lhs.key.value < rhs.key.value;
        }
        return lhs.original_index < rhs.original_index;
    });
    return rows;
}

std::vector<ProviderLocalKeyRow> provider_local_partitioned_rows(
    const std::vector<NormalizedKey>& keys,
    const PartitionContract& contract,
    const std::vector<u64>& capacities,
    bool& overflow)
{
    const u64 partitions = partition_count(contract);
    std::vector<std::vector<ProviderLocalKeyRow>> per_partition(static_cast<std::size_t>(partitions));
    overflow = false;
    for (u64 idx = 0; idx < static_cast<u64>(keys.size()); ++idx) {
        u64 part = 0;
        try {
            part = assign_public_partition(contract, keys[static_cast<std::size_t>(idx)]);
        } catch (const std::exception&) {
            overflow = true;
            continue;
        }
        auto& rows = per_partition[static_cast<std::size_t>(part)];
        rows.push_back({keys[static_cast<std::size_t>(idx)], idx});
        if (rows.size() > capacities[static_cast<std::size_t>(part)]) {
            overflow = true;
        }
    }
    std::vector<ProviderLocalKeyRow> out;
    out.reserve(sum_u64(capacities));
    for (u64 part = 0; part < partitions; ++part) {
        auto& rows = per_partition[static_cast<std::size_t>(part)];
        std::sort(rows.begin(), rows.end(), [](const ProviderLocalKeyRow& lhs, const ProviderLocalKeyRow& rhs) {
            if (lhs.key.is_null != rhs.key.is_null) {
                return !lhs.key.is_null && rhs.key.is_null;
            }
            if (lhs.key.value != rhs.key.value) {
                return lhs.key.value < rhs.key.value;
            }
            return lhs.original_index < rhs.original_index;
        });
        u64 slot = 0;
        for (; slot < static_cast<u64>(rows.size()) &&
               slot < capacities[static_cast<std::size_t>(part)]; ++slot) {
            auto row = rows[static_cast<std::size_t>(slot)];
            row.original_index = slot;
            out.push_back(row);
        }
        for (; slot < capacities[static_cast<std::size_t>(part)]; ++slot) {
            out.push_back({NormalizedKey{0, true}, slot, 0});
        }
    }
    return out;
}

bool provider_local_keys_unique_in_public_domain(
    const std::vector<NormalizedKey>& keys,
    const PartitionContract& contract)
{
    std::unordered_map<i64, bool> seen;
    for (const auto& key : keys) {
        if (key.is_null) {
            continue;
        }
        try {
            (void) assign_public_partition(contract, key);
        } catch (const std::exception&) {
            return false;
        }
        auto inserted = seen.emplace(key.value, true);
        if (!inserted.second) {
            return false;
        }
    }
    return true;
}

ProviderDenseDomainState provider_local_pk_dense_domain_state(
    const std::string& csv_path,
    u64 key_column,
    bool has_header,
    const PartitionContract& contract)
{
    ProviderDenseDomainState state;
    state.domain_min = contract.boundaries.front();
    state.domain_width = public_domain_width_inclusive(contract);
    state.pk_present.assign(static_cast<std::size_t>(state.domain_width), 0);
    for_each_key_column_csv(csv_path, key_column, has_header, [&](const NormalizedKey& key, u64) {
        ++state.raw_row_count;
        if (key.is_null) {
            state.contract_failed = true;
            return;
        }
        try {
            const u64 offset = public_domain_offset(contract, key);
            auto& present = state.pk_present[static_cast<std::size_t>(offset)];
            if (present != 0) {
                state.contract_failed = true;
            }
            present = 1;
        } catch (const std::exception&) {
            state.contract_failed = true;
        }
    });
    return state;
}

ProviderDenseDomainState provider_local_fk_dense_domain_state(
    const std::string& csv_path,
    u64 key_column,
    bool has_header,
    const PartitionContract& contract)
{
    ProviderDenseDomainState state;
    state.domain_min = contract.boundaries.front();
    state.domain_width = public_domain_width_inclusive(contract);
    state.fk_counts.assign(static_cast<std::size_t>(state.domain_width), 0);
    for_each_key_column_csv(csv_path, key_column, has_header, [&](const NormalizedKey& key, u64) {
        ++state.raw_row_count;
        if (key.is_null) {
            return;
        }
        try {
            const u64 offset = public_domain_offset(contract, key);
            auto& count = state.fk_counts[static_cast<std::size_t>(offset)];
            if (count == std::numeric_limits<i64>::max()) {
                state.contract_failed = true;
                return;
            }
            ++count;
        } catch (const std::exception&) {
            state.contract_failed = true;
        }
    });
    return state;
}

std::vector<ProviderLocalKeyRow> provider_local_pk_rows_from_dense_state_for_part(
    const ProviderDenseDomainState& state,
    const PartitionContract& contract,
    u64 part,
    u64 capacity,
    bool& overflow)
{
    overflow = false;
    const auto range = public_partition_offset_range(contract, part);
    std::vector<ProviderLocalKeyRow> out;
    out.reserve(static_cast<std::size_t>(capacity));
    u64 slot = 0;
    for (u64 offset = range.first; offset < range.second; ++offset) {
        if (offset >= static_cast<u64>(state.pk_present.size())) {
            overflow = true;
            break;
        }
        if (state.pk_present[static_cast<std::size_t>(offset)] == 0) {
            continue;
        }
        if (slot >= capacity) {
            overflow = true;
            break;
        }
        out.push_back({NormalizedKey{state.domain_min + static_cast<i64>(offset), false}, slot, 1});
        ++slot;
    }
    for (; slot < capacity; ++slot) {
        out.push_back({NormalizedKey{0, true}, slot, 0});
    }
    return out;
}

std::vector<ProviderLocalKeyRow> provider_local_fk_group_rows_from_dense_state_for_part(
    const ProviderDenseDomainState& state,
    const PartitionContract& contract,
    u64 part,
    u64 capacity,
    bool& overflow)
{
    overflow = false;
    const auto range = public_partition_offset_range(contract, part);
    std::vector<ProviderLocalKeyRow> out;
    out.reserve(static_cast<std::size_t>(capacity));
    u64 slot = 0;
    for (u64 offset = range.first; offset < range.second; ++offset) {
        if (offset >= static_cast<u64>(state.fk_counts.size())) {
            overflow = true;
            break;
        }
        const i64 count = state.fk_counts[static_cast<std::size_t>(offset)];
        if (count == 0) {
            continue;
        }
        if (slot >= capacity) {
            overflow = true;
            break;
        }
        out.push_back({NormalizedKey{state.domain_min + static_cast<i64>(offset), false}, slot, count});
        ++slot;
    }
    for (; slot < capacity; ++slot) {
        out.push_back({NormalizedKey{0, true}, slot, 0});
    }
    return out;
}

std::vector<ProviderLocalKeyRow> provider_local_partitioned_fk_group_rows(
    const std::vector<NormalizedKey>& keys,
    const PartitionContract& contract,
    const std::vector<u64>& capacities,
    bool& overflow)
{
    const u64 partitions = partition_count(contract);
    std::vector<std::unordered_map<i64, i64>> counts(static_cast<std::size_t>(partitions));
    overflow = false;
    for (const auto& key : keys) {
        if (key.is_null) {
            continue;
        }
        u64 part = 0;
        try {
            part = assign_public_partition(contract, key);
        } catch (const std::exception&) {
            overflow = true;
            continue;
        }
        auto& part_counts = counts[static_cast<std::size_t>(part)];
        i64& value = part_counts[key.value];
        if (value == std::numeric_limits<i64>::max()) {
            overflow = true;
            continue;
        }
        ++value;
        if (part_counts.size() > capacities[static_cast<std::size_t>(part)]) {
            overflow = true;
        }
    }

    std::vector<ProviderLocalKeyRow> out;
    out.reserve(sum_u64(capacities));
    for (u64 part = 0; part < partitions; ++part) {
        std::vector<ProviderLocalKeyRow> rows;
        rows.reserve(counts[static_cast<std::size_t>(part)].size());
        for (const auto& item : counts[static_cast<std::size_t>(part)]) {
            rows.push_back({NormalizedKey{item.first, false}, 0, item.second});
        }
        std::sort(rows.begin(), rows.end(), [](const ProviderLocalKeyRow& lhs, const ProviderLocalKeyRow& rhs) {
            return lhs.key.value < rhs.key.value;
        });
        u64 slot = 0;
        for (; slot < static_cast<u64>(rows.size()) &&
               slot < capacities[static_cast<std::size_t>(part)]; ++slot) {
            auto row = rows[static_cast<std::size_t>(slot)];
            row.original_index = slot;
            out.push_back(row);
        }
        for (; slot < capacities[static_cast<std::size_t>(part)]; ++slot) {
            out.push_back({NormalizedKey{0, true}, slot, 0});
        }
    }
    return out;
}

std::vector<ProviderLocalKeyRow> provider_local_partition_rows_for_part(
    const std::vector<NormalizedKey>& keys,
    const PartitionContract& contract,
    u64 part,
    u64 capacity,
    bool& overflow)
{
    std::vector<ProviderLocalKeyRow> rows;
    overflow = false;
    for (u64 idx = 0; idx < static_cast<u64>(keys.size()); ++idx) {
        if (keys[static_cast<std::size_t>(idx)].is_null) {
            continue;
        }
        u64 key_part = 0;
        try {
            key_part = assign_public_partition(contract, keys[static_cast<std::size_t>(idx)]);
        } catch (const std::exception&) {
            overflow = true;
            continue;
        }
        if (key_part != part) {
            continue;
        }
        rows.push_back({keys[static_cast<std::size_t>(idx)], idx, 1});
        if (rows.size() > capacity) {
            overflow = true;
        }
    }
    std::sort(rows.begin(), rows.end(), [](const ProviderLocalKeyRow& lhs, const ProviderLocalKeyRow& rhs) {
        if (lhs.key.value != rhs.key.value) {
            return lhs.key.value < rhs.key.value;
        }
        return lhs.original_index < rhs.original_index;
    });
    std::vector<ProviderLocalKeyRow> out;
    out.reserve(static_cast<std::size_t>(capacity));
    u64 slot = 0;
    for (; slot < static_cast<u64>(rows.size()) && slot < capacity; ++slot) {
        auto row = rows[static_cast<std::size_t>(slot)];
        row.original_index = slot;
        out.push_back(row);
    }
    for (; slot < capacity; ++slot) {
        out.push_back({NormalizedKey{0, true}, slot, 0});
    }
    return out;
}

std::vector<ProviderLocalKeyRow> provider_local_fk_group_rows_for_part(
    const std::vector<NormalizedKey>& keys,
    const PartitionContract& contract,
    u64 part,
    u64 capacity,
    bool& overflow)
{
    std::unordered_map<i64, i64> counts;
    overflow = false;
    for (const auto& key : keys) {
        if (key.is_null) {
            continue;
        }
        u64 key_part = 0;
        try {
            key_part = assign_public_partition(contract, key);
        } catch (const std::exception&) {
            overflow = true;
            continue;
        }
        if (key_part != part) {
            continue;
        }
        i64& value = counts[key.value];
        if (value == std::numeric_limits<i64>::max()) {
            overflow = true;
            continue;
        }
        ++value;
        if (counts.size() > capacity) {
            overflow = true;
        }
    }
    std::vector<ProviderLocalKeyRow> rows;
    rows.reserve(counts.size());
    for (const auto& item : counts) {
        rows.push_back({NormalizedKey{item.first, false}, 0, item.second});
    }
    std::sort(rows.begin(), rows.end(), [](const ProviderLocalKeyRow& lhs, const ProviderLocalKeyRow& rhs) {
        return lhs.key.value < rhs.key.value;
    });
    std::vector<ProviderLocalKeyRow> out;
    out.reserve(static_cast<std::size_t>(capacity));
    u64 slot = 0;
    for (; slot < static_cast<u64>(rows.size()) && slot < capacity; ++slot) {
        auto row = rows[static_cast<std::size_t>(slot)];
        row.original_index = slot;
        out.push_back(row);
    }
    for (; slot < capacity; ++slot) {
        out.push_back({NormalizedKey{0, true}, slot, 0});
    }
    return out;
}

void populate_plain_rows(
    const std::vector<ProviderLocalKeyRow>& rows,
    i64Matrix& key,
    i64Matrix& is_null,
    i64Matrix& original_index,
    i64Matrix* multiplicity = nullptr)
{
    if (static_cast<u64>(rows.size()) != static_cast<u64>(key.rows())) {
        throw std::runtime_error("Provider-local partition row count does not match public capacity.");
    }
    for (u64 row = 0; row < static_cast<u64>(rows.size()); ++row) {
        const auto& local_row = rows[static_cast<std::size_t>(row)];
        key(row, 0) = local_row.key.value;
        is_null(row, 0) = local_row.key.is_null ? 1 : 0;
        original_index(row, 0) = static_cast<i64>(local_row.original_index);
        if (multiplicity != nullptr) {
            (*multiplicity)(row, 0) = local_row.multiplicity;
        }
    }
}

si64Matrix provider_separated_oblivious_merge_counts(
    const MultiplierPreprocessConfig& config,
    const si64Matrix& pk_key_shared,
    const si64Matrix& pk_null_shared,
    const si64Matrix& pk_original_index_shared,
    const si64Matrix& fk_key_shared,
    const si64Matrix& fk_null_shared,
    const si64Matrix& fk_original_index_shared,
    const si64Matrix& fk_multiplicity_shared,
    u64 n_pk,
    u64 n_fk,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const u64 padded = roundUpToPowerOfTwo(std::max<u64>(1, n_pk + n_fk));
    const u64 pad_rows = padded - n_pk - n_fk;

    auto pk_not_null_flag = int_eq_public(pk_null_shared, 0, config.role, eval, runtime);
    auto fk_not_null_flag = int_eq_public(fk_null_shared, 0, config.role, eval, runtime);
    auto pk_not_null = bool_to_si64(pk_not_null_flag, config.role, enc, eval, runtime);
    auto pk_invalid = public_i64_column(1, n_pk, config.role) - pk_not_null;
    auto fk_not_null = bool_to_si64(fk_not_null_flag, config.role, enc, eval, runtime);
    auto fk_invalid = public_i64_column(1, n_fk, config.role) - fk_not_null;
    si64Matrix fk_mass(n_fk, 1);
    fk_mass.mShares[0].setZero();
    fk_mass.mShares[1].setZero();
    cipher_mul(config.role, fk_multiplicity_shared, fk_not_null, fk_mass, eval, enc, runtime);

    auto pk_table_id = public_i64_column(1, n_pk, config.role);
    auto fk_table_id = public_i64_column(0, n_fk, config.role);
    auto pad_table_id = public_i64_column(2, pad_rows, config.role);
    auto pad_invalid = public_i64_column(1, pad_rows, config.role);
    auto pad_key = public_i64_column(0, pad_rows, config.role);
    auto pad_original_index = public_i64_row_ids(pad_rows, config.role);
    auto pad_fk_contrib = public_i64_column(0, pad_rows, config.role);

    auto key = concat_shared_row_blocks({pk_key_shared, fk_key_shared, pad_key});
    auto invalid_rank = concat_shared_row_blocks({pk_invalid, fk_invalid, pad_invalid});
    auto origin_order = concat_shared_row_blocks({pk_table_id, fk_table_id, pad_table_id});
    auto original_index = concat_shared_row_blocks({pk_original_index_shared, fk_original_index_shared, pad_original_index});
    auto fk_contrib = concat_shared_row_blocks({
        public_i64_column(0, n_pk, config.role),
        fk_mass,
        pad_fk_contrib,
    });

    std::vector<si64Matrix> columns = {
        invalid_rank,
        key,
        origin_order,
        original_index,
        fk_contrib,
    };
    constexpr u64 kInvalidRankCol = 0;
    constexpr u64 kKeyCol = 1;
    constexpr u64 kOriginOrderCol = 2;
    constexpr u64 kOriginalIndexCol = 3;
    constexpr u64 kFkContribCol = 4;

    // Public bitonic network, secret lexicographic comparator:
    // valid rows first, then key, then FK before PK, then local tie breaker.
    oblivious_bitonic_sort_rows(
        columns,
        {kInvalidRankCol, kKeyCol, kOriginOrderCol, kOriginalIndexCol},
        config.role,
        enc,
        eval,
        runtime);

    auto zero = public_i64_column(0, padded, config.role);
    auto one = public_i64_column(1, padded, config.role);
    auto is_valid_sorted = int_eq_matrix(columns[kInvalidRankCol], zero, config.role, eval, runtime);
    auto is_fk_sorted = int_eq_matrix(columns[kOriginOrderCol], zero, config.role, eval, runtime);
    auto is_pk_sorted = int_eq_matrix(columns[kOriginOrderCol], one, config.role, eval, runtime);
    is_fk_sorted = local_bool_and_matrix(is_fk_sorted, is_valid_sorted, config.role, enc, eval, runtime);

    si64Matrix running_count(padded, 1);
    running_count.mShares[0].setZero();
    running_count.mShares[1].setZero();
    for (u64 row = 0; row < padded; ++row) {
        auto current_fk = int_row_slice(columns[kFkContribCol], row, 1);
        if (row == 0) {
            running_count.mShares[0](row, 0) = current_fk.mShares[0](0, 0);
            running_count.mShares[1](row, 0) = current_fk.mShares[1](0, 0);
            continue;
        }
        auto lhs_key = int_row_slice(columns[kKeyCol], row, 1);
        auto rhs_key = int_row_slice(columns[kKeyCol], row - 1, 1);
        auto same_key = int_eq_matrix(lhs_key, rhs_key, config.role, eval, runtime);
        auto current_valid = bool_row_slice(is_valid_sorted, row, 1);
        auto prev_valid = bool_row_slice(is_valid_sorted, row - 1, 1);
        same_key = local_bool_and_matrix(same_key, current_valid, config.role, enc, eval, runtime);
        same_key = local_bool_and_matrix(same_key, prev_valid, config.role, enc, eval, runtime);
        auto previous_count = int_row_slice(running_count, row - 1, 1);
        auto candidate = previous_count + current_fk;
        auto selected = select_si64_by_bool(
            candidate,
            current_fk,
            same_key,
            config.role,
            enc,
            eval,
            runtime);
        running_count.mShares[0](row, 0) = selected.mShares[0](0, 0);
        running_count.mShares[1](row, 0) = selected.mShares[1](0, 0);
    }

    auto selected_counts = select_si64_by_bool(
        running_count,
        shared_zero_int_matrix(padded, 1),
        is_pk_sorted,
        config.role,
        enc,
        eval,
        runtime);
    auto is_pk_int = bool_to_si64(is_pk_sorted, config.role, enc, eval, runtime);
    auto non_pk_order = public_i64_column(1, padded, config.role) - is_pk_int;

    std::vector<si64Matrix> output_columns = {
        non_pk_order,
        columns[kOriginalIndexCol],
        selected_counts,
    };
    constexpr u64 kOutputOrderCol = 0;
    constexpr u64 kOutputOriginalIndexCol = 1;
    constexpr u64 kOutputCountCol = 2;
    oblivious_bitonic_sort_rows(
        output_columns,
        {kOutputOrderCol, kOutputOriginalIndexCol},
        config.role,
        enc,
        eval,
        runtime);

    si64Matrix counts(n_pk, 1);
    counts.mShares[0].setZero();
    counts.mShares[1].setZero();
    for (u64 row = 0; row < n_pk; ++row) {
        counts.mShares[0](row, 0) = output_columns[kOutputCountCol].mShares[0](row, 0);
        counts.mShares[1](row, 0) = output_columns[kOutputCountCol].mShares[1](row, 0);
    }
    return counts;
}

si64Matrix provider_separated_partitioned_oblivious_merge_counts(
    const MultiplierPreprocessConfig& config,
    const PartitionContract& contract,
    const si64Matrix& pk_key_shared,
    const si64Matrix& pk_null_shared,
    const si64Matrix& pk_original_index_shared,
    const si64Matrix& fk_key_shared,
    const si64Matrix& fk_null_shared,
    const si64Matrix& fk_original_index_shared,
    const si64Matrix& fk_multiplicity_shared,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const u64 partitions = partition_count(contract);
    const u64 total_pk_capacity = sum_u64(contract.pk_capacities);
    si64Matrix counts(total_pk_capacity, 1);
    counts.mShares[0].setZero();
    counts.mShares[1].setZero();

    u64 pk_offset = 0;
    u64 fk_offset = 0;
    for (u64 part = 0; part < partitions; ++part) {
        const u64 pk_cap = contract.pk_capacities[static_cast<std::size_t>(part)];
        const u64 fk_cap = contract.fk_capacities[static_cast<std::size_t>(part)];
        auto part_counts = provider_separated_oblivious_merge_counts(
            config,
            int_row_slice(pk_key_shared, pk_offset, pk_cap),
            int_row_slice(pk_null_shared, pk_offset, pk_cap),
            int_row_slice(pk_original_index_shared, pk_offset, pk_cap),
            int_row_slice(fk_key_shared, fk_offset, fk_cap),
            int_row_slice(fk_null_shared, fk_offset, fk_cap),
            int_row_slice(fk_original_index_shared, fk_offset, fk_cap),
            int_row_slice(fk_multiplicity_shared, fk_offset, fk_cap),
            pk_cap,
            fk_cap,
            enc,
            eval,
            runtime);
        for (u64 row = 0; row < pk_cap; ++row) {
            counts.mShares[0](pk_offset + row, 0) = part_counts.mShares[0](row, 0);
            counts.mShares[1](pk_offset + row, 0) = part_counts.mShares[1](row, 0);
        }
        pk_offset += pk_cap;
        fk_offset += fk_cap;
    }
    return counts;
}

struct StreamingShareFileMeta {
    std::string path;
    std::uint64_t bytes = 0;
    std::string sha256;
};

struct StreamingPartitionMeta {
    u64 partition_id = 0;
    u64 pk_capacity = 0;
    u64 fk_capacity = 0;
    std::array<StreamingShareFileMeta, 3> mu;
    std::array<StreamingShareFileMeta, 3> mu_nn;
};

void multiplier_public_barrier(int role, Sh3Runtime& runtime)
{
    for (int owner = 0; owner < 3; ++owner) {
        u64 marker = static_cast<u64>(owner + 1);
        sync_value_from_party(role, owner, runtime, marker);
    }
}

std::string partition_dir_name(u64 part)
{
    std::ostringstream out;
    out << "partition_" << std::setw(4) << std::setfill('0') << part;
    return out.str();
}

void maybe_fail_streaming_writer(const std::string& point, u64 part = std::numeric_limits<u64>::max())
{
    const char* value = std::getenv("BSPN_MULTIPLIER_STREAMING_FAIL_POINT");
    if (value == nullptr || *value == '\0') {
        return;
    }
    const std::string requested(value);
    if (requested == "0" || requested == "false" || requested == "off") {
        return;
    }
    if (requested == point) {
        throw std::runtime_error("injected public streaming writer failure: " + point);
    }
    if (part != std::numeric_limits<u64>::max()) {
        const std::string after_partition = "after_partition_" + std::to_string(part);
        if (requested == after_partition) {
            throw std::runtime_error("injected public streaming writer failure: " + after_partition);
        }
    }
}

json streaming_manifest_base(
    const MultiplierPreprocessConfig& config,
    size_t pk_row_count,
    size_t fk_row_count,
    const PartitionContract& contract,
    const std::vector<StreamingPartitionMeta>& partitions,
    const std::string& artifact_dir)
{
    const auto partition_boundaries = parse_i64_csv_env("BSPN_MULTIPLIER_PARTITION_BOUNDARIES");
    const auto partition_pk_capacities = parse_u64_csv_env("BSPN_MULTIPLIER_PARTITION_PK_CAPACITIES");
    const auto partition_fk_capacities = parse_u64_csv_env("BSPN_MULTIPLIER_PARTITION_FK_CAPACITIES");
    const bool grouped_capacity_is_domain_width =
        public_domain_width_covers_capacities(partition_boundaries, partition_pk_capacities) &&
        public_domain_width_covers_capacities(partition_boundaries, partition_fk_capacities);

    json doc;
    doc["format_name"] = "BSPN_MULTIPLIER_PAYLOAD";
    doc["format_version"] = 3;
    doc["mode"] = "secure_shared_values";
    doc["relationship_id"] = config.relationship_id;
    doc["pk_csv_path"] = config.pk_csv_path;
    doc["fk_csv_path"] = config.fk_csv_path;
    doc["pk_key_column"] = config.pk_key_column;
    doc["fk_key_column"] = config.fk_key_column;
    doc["fk_sample_rate"] = config.fk_sample_rate;
    doc["pk_row_count"] = pk_row_count;
    doc["fk_row_count"] = fk_row_count;
    doc["fixed_decimal_bits"] = kMultiplierFixedDecimalBits;
    doc["pk_input_party"] = config.pk_input_party;
    doc["fk_input_party"] = config.fk_input_party;
    doc["share_kind"] = "ABY3_REPLICATED_PAIR_I64";
    doc["construction_mode"] = "privacy_aligned";
    doc["join_key_input_mode"] = "provider_separated_secret_shares";
    doc["provider_separated_key_inputs"] = true;
    doc["provider_separated_sorted_group_count"] = false;
    doc["provider_separated_oblivious_merge_group_count"] = false;
    doc["provider_separated_partitioned_oblivious_merge_group_count"] = true;
    doc["provider_local_fk_group_count"] = true;
    doc["fk_group_count_exact"] = true;
    doc["fk_group_multiplicity_secret_shared"] = true;
    doc["provider_local_partition"] = true;
    doc["provider_local_sort"] = true;
    doc["server_oblivious_merge"] = true;
    doc["sort_core_reveals_comparisons"] = false;
    doc["network_execution_mode"] = secure_multiplier_layer_batch_enabled() ? "public_layer_batched" : "scalar_reference";
    doc["network_schedule_version"] = "bitonic_compare_exchange_v1";
    doc["record_schema_version"] = "streaming_multiplier_partition_v1";
    doc["partition_boundaries_public"] = true;
    doc["partition_boundaries_query_independent"] = true;
    doc["partition_real_cardinalities_hidden"] = true;
    doc["partition_capacities_public"] = true;
    doc["partition_boundary_rule"] = "public_half_open_ranges_last_closed";
    doc["partition_boundaries"] = join_i64_values(partition_boundaries);
    doc["partition_pk_capacities"] = join_u64_values(partition_pk_capacities);
    doc["partition_fk_capacities"] = join_u64_values(partition_fk_capacities);
    doc["capacity_policy"] = grouped_capacity_is_domain_width ? "public_key_domain_width" : "explicit_public_group_capacity";
    doc["capacity_deterministic"] = grouped_capacity_is_domain_width;
    doc["output_alignment_mode"] = "provider_local_partitioned_model_order";
    doc["overflow_policy"] = "fail_closed";
    doc["plaintext_count_then_share"] = false;
    doc["secure_core_status"] = "provider_local_distinct_fk_partitioned_oblivious_merge";
    doc["artifact_state"] = "complete";
    doc["streaming_partition_execution"] = true;
    doc["fixed_partition_file_schedule"] = true;
    doc["partial_artifact_fail_closed"] = true;
    doc["atomic_acceptance_finalization"] = true;
    doc["expected_role_count"] = 3;
    doc["partition_count"] = static_cast<std::uint64_t>(partitions.size());
    doc["role_paths"] = json::object();
    for (int role = 0; role < 3; ++role) {
        doc["role_paths"][std::to_string(role)] = {
            {"mu_path", artifact_dir + "/role_" + std::to_string(role) + "/mu.shares.bin"},
            {"mu_nn_path", artifact_dir + "/role_" + std::to_string(role) + "/mu_nn.shares.bin"},
        };
    }
    doc["streaming_partitions"] = json::array();
    for (const auto& part : partitions) {
        json part_doc;
        part_doc["partition_id"] = part.partition_id;
        part_doc["pk_capacity"] = part.pk_capacity;
        part_doc["fk_group_capacity"] = part.fk_capacity;
        part_doc["expected_mu_bytes"] = part.pk_capacity * sizeof(i64) * 2;
        part_doc["expected_mu_nn_bytes"] = part.pk_capacity * sizeof(i64) * 2;
        part_doc["role_shares"] = json::object();
        for (int role = 0; role < 3; ++role) {
            part_doc["role_shares"][std::to_string(role)] = {
                {"mu_path", part.mu[role].path},
                {"mu_bytes", part.mu[role].bytes},
                {"mu_sha256", part.mu[role].sha256},
                {"mu_nn_path", part.mu_nn[role].path},
                {"mu_nn_bytes", part.mu_nn[role].bytes},
                {"mu_nn_sha256", part.mu_nn[role].sha256},
            };
        }
        doc["streaming_partitions"].push_back(part_doc);
    }
    return doc;
}

void write_streaming_manifest_atomic(const std::string& path, const json& doc)
{
    const std::string tmp_path = path + ".tmp";
    std::ofstream output(tmp_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open streaming manifest temp file: " + tmp_path);
    }
    output << doc.dump(2) << "\n";
    output.close();
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("Failed to publish streaming manifest " + path + ": " + std::strerror(errno));
    }
}

si64Matrix provider_separated_sorted_counts(
    const MultiplierPreprocessConfig& config,
    const si64Matrix& pk_key_shared,
    const si64Matrix& pk_null_shared,
    const si64Matrix& fk_key_shared,
    const si64Matrix& fk_null_shared,
    u64 n_pk,
    u64 n_fk,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    if (secure_multiplier_forbid_server_reveal_enabled()) {
        throw std::runtime_error("privacy-aligned multiplier preprocessing forbids comparison-revealing sorted core");
    }
    const u64 padded = roundUpToPowerOfTwo(n_pk + n_fk);
    const u64 pad_rows = padded - n_pk - n_fk;

    auto pk_table_id = public_i64_column(0, n_pk, config.role);
    auto fk_table_id = public_i64_column(1, n_fk, config.role);
    auto pad_table_id = public_i64_column(2, pad_rows, config.role);
    auto pk_row_id = public_i64_row_ids(n_pk, config.role);
    auto fk_row_id = public_i64_row_ids(n_fk, config.role);
    auto pad_row_id = public_i64_row_ids(pad_rows, config.role);

    auto pk_not_null_flag = int_eq_public(pk_null_shared, 0, config.role, eval, runtime);
    auto fk_not_null_flag = int_eq_public(fk_null_shared, 0, config.role, eval, runtime);
    auto pk_not_null = bool_to_si64(pk_not_null_flag, config.role, enc, eval, runtime);
    auto fk_contrib = bool_to_si64(fk_not_null_flag, config.role, enc, eval, runtime);

    auto pad_key = public_i64_column(0, pad_rows, config.role);
    auto pad_not_null = public_i64_column(0, pad_rows, config.role);
    auto pad_fk_contrib = public_i64_column(0, pad_rows, config.role);

    si64Matrix group_key = concat_shared_row_blocks({pk_key_shared, fk_key_shared, pad_key});
    auto table_id = concat_shared_row_blocks({pk_table_id, fk_table_id, pad_table_id});
    auto row_id = concat_shared_row_blocks({pk_row_id, fk_row_id, pad_row_id});
    auto pk_not_null_rows = concat_shared_row_blocks({
        pk_not_null,
        public_i64_column(0, n_fk, config.role),
        pad_not_null,
    });
    auto fk_contrib_rows = concat_shared_row_blocks({
        public_i64_column(0, n_pk, config.role),
        fk_contrib,
        pad_fk_contrib,
    });

    si64Matrix public_tag(padded, 1);
    public_tag.mShares[0].setZero();
    public_tag.mShares[1].setZero();
    for (u64 row = 0; row < padded; ++row) {
        if (config.role == 0) {
            public_tag.mShares[0](row, 0) = static_cast<i64>(row);
        } else if (config.role == 1) {
            public_tag.mShares[1](row, 0) = static_cast<i64>(row);
        }
    }
    si64Matrix sec_key = group_key;
    for (u64 row = 0; row < padded; ++row) {
        sec_key.mShares[0](row, 0) *= static_cast<i64>(std::max<u64>(1, padded));
        sec_key.mShares[1](row, 0) *= static_cast<i64>(std::max<u64>(1, padded));
    }
    sec_key = sec_key + public_tag;

    std::vector<si64Matrix> payloads(static_cast<std::size_t>(padded));
    for (u64 row = 0; row < padded; ++row) {
        si64Matrix payload(5, 1);
        payload.mShares[0](0, 0) = table_id.mShares[0](row, 0);
        payload.mShares[1](0, 0) = table_id.mShares[1](row, 0);
        payload.mShares[0](1, 0) = row_id.mShares[0](row, 0);
        payload.mShares[1](1, 0) = row_id.mShares[1](row, 0);
        payload.mShares[0](2, 0) = pk_not_null_rows.mShares[0](row, 0);
        payload.mShares[1](2, 0) = pk_not_null_rows.mShares[1](row, 0);
        payload.mShares[0](3, 0) = fk_contrib_rows.mShares[0](row, 0);
        payload.mShares[1](3, 0) = fk_contrib_rows.mShares[1](row, 0);
        payload.mShares[0](4, 0) = group_key.mShares[0](row, 0);
        payload.mShares[1](4, 0) = group_key.mShares[1](row, 0);
        payloads[static_cast<std::size_t>(row)] = std::move(payload);
    }

    quick_sort_with_other_elements(
        sec_key,
        payloads,
        config.role,
        enc,
        eval,
        runtime,
        config.secure_sort_min_size);

    auto sorted_table_id = payload_column_to_matrix(payloads, 0);
    auto sorted_row_id = payload_column_to_matrix(payloads, 1);
    auto sorted_pk_not_null = payload_column_to_matrix(payloads, 2);
    auto sorted_fk_contrib = payload_column_to_matrix(payloads, 3);
    auto sorted_group_key = payload_column_to_matrix(payloads, 4);

    auto same_previous = adjacent_group_equal_flags(sorted_group_key, true, config.role, eval, runtime);
    auto same_next = adjacent_group_equal_flags(sorted_group_key, false, config.role, eval, runtime);
    auto prefix_counts = segmented_prefix_sum(sorted_fk_contrib, same_previous, config.role, enc, eval, runtime);
    auto suffix_counts = segmented_suffix_sum(sorted_fk_contrib, same_next, config.role, enc, eval, runtime);
    auto sorted_group_counts = prefix_counts + suffix_counts - sorted_fk_contrib;

    auto zero_sorted = public_i64_column(0, padded, config.role);
    sbMatrix is_pk_row;
    cipher_eq(config.role, sorted_table_id, zero_sorted, is_pk_row, eval, runtime);
    auto sorted_pk_is_non_null = int_eq_public(sorted_pk_not_null, 1, config.role, eval, runtime);
    auto output_pk_row = local_bool_and_matrix(is_pk_row, sorted_pk_is_non_null, config.role, enc, eval, runtime);
    auto selected_counts = select_si64_by_bool(
        sorted_group_counts,
        zero_sorted,
        output_pk_row,
        config.role,
        enc,
        eval,
        runtime);

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
    auto sort_back_key = select_si64_by_bool(
        sorted_row_id,
        non_pk_sort_key,
        is_pk_row,
        config.role,
        enc,
        eval,
        runtime);
    auto count_payloads = column_to_payload_vector(selected_counts);
    quick_sort_with_other_elements(
        sort_back_key,
        count_payloads,
        config.role,
        enc,
        eval,
        runtime,
        config.secure_sort_min_size);

    si64Matrix counts(n_pk, 1);
    counts.mShares[0].setZero();
    counts.mShares[1].setZero();
    for (u64 row = 0; row < n_pk; ++row) {
        counts.mShares[0](row, 0) = count_payloads[static_cast<std::size_t>(row)].mShares[0](0, 0);
        counts.mShares[1](row, 0) = count_payloads[static_cast<std::size_t>(row)].mShares[1](0, 0);
    }
    return counts;
}

void run_secure_multiplier_shared_values_fast(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_shared_values mode requires --role in {0,1,2}.");
    }
    if (config.fk_sample_rate <= 0.0) {
        throw std::runtime_error("FK sample rate must be positive.");
    }
    if (secure_multiplier_provider_separated_sorted_enabled() &&
        (secure_multiplier_provider_separated_oblivious_merge_enabled() ||
         secure_multiplier_provider_separated_partitioned_enabled())) {
        throw std::runtime_error(
            "Provider-separated diagnostic sorted path is mutually exclusive with privacy-aligned oblivious paths.");
    }
    const bool partitioned = secure_multiplier_provider_separated_partitioned_enabled();
    const bool provider_local_fk_group_count = secure_multiplier_provider_local_fk_group_count_enabled();
    if (provider_local_fk_group_count && !partitioned) {
        throw std::runtime_error("Provider-local FK group/count requires partitioned provider-separated preprocessing.");
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

void run_secure_multiplier_shared_values_provider_separated_streaming(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_shared_values mode requires --role in {0,1,2}.");
    }
    if (config.pk_input_party < 0 || config.pk_input_party > 2 ||
        config.fk_input_party < 0 || config.fk_input_party > 2 ||
        config.pk_input_party == config.fk_input_party) {
        throw std::runtime_error("streaming provider-separated preprocessing requires distinct input parties.");
    }
    if (!secure_multiplier_provider_separated_partitioned_enabled() ||
        !secure_multiplier_provider_local_fk_group_count_enabled()) {
        throw std::runtime_error(
            "streaming multiplier artifacts require partitioned provider-local FK group/count preprocessing.");
    }
    if (config.fk_sample_rate <= 0.0) {
        throw std::runtime_error("FK sample rate must be positive.");
    }

    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(config.role), ios, enc, eval, runtime);

    const PartitionContract partition_contract = parse_partition_contract();
    const u64 partitions = partition_count(partition_contract);
    const u64 total_pk_capacity = sum_u64(partition_contract.pk_capacities);
    const u64 total_fk_capacity = sum_u64(partition_contract.fk_capacities);

    ProviderDenseDomainState pk_dense;
    ProviderDenseDomainState fk_dense;
    u64 n_pk = 0;
    u64 n_fk = 0;
    if (config.role == config.pk_input_party) {
        const auto started = SteadyClock::now();
        pk_dense = provider_local_pk_dense_domain_state(
            config.pk_csv_path,
            config.pk_key_column,
            config.pk_has_header,
            partition_contract);
        n_pk = pk_dense.raw_row_count;
        if (secure_multiplier_public_profile_enabled()) {
            std::cerr << "bspn_multiplier_profile: event=provider_dense_pk"
                      << " public_domain_width=" << pk_dense.domain_width
                      << " elapsed_seconds=" << elapsed_seconds_since(started)
                      << std::endl;
        }
    }
    if (config.role == config.fk_input_party) {
        const auto started = SteadyClock::now();
        fk_dense = provider_local_fk_dense_domain_state(
            config.fk_csv_path,
            config.fk_key_column,
            config.fk_has_header,
            partition_contract);
        n_fk = fk_dense.raw_row_count;
        if (secure_multiplier_public_profile_enabled()) {
            std::cerr << "bspn_multiplier_profile: event=provider_dense_fk"
                      << " public_domain_width=" << fk_dense.domain_width
                      << " elapsed_seconds=" << elapsed_seconds_since(started)
                      << std::endl;
        }
    }
    sync_value_from_party(config.role, config.pk_input_party, runtime, n_pk);
    sync_value_from_party(config.role, config.fk_input_party, runtime, n_fk);

    u64 pk_constraint_failed = 0;
    if (config.role == config.pk_input_party) {
        pk_constraint_failed = pk_dense.contract_failed ? 1 : 0;
    }
    sync_value_from_party(config.role, config.pk_input_party, runtime, pk_constraint_failed);
    if (pk_constraint_failed != 0) {
        throw std::runtime_error("public capacity contract not satisfied");
    }
    u64 fk_constraint_failed = 0;
    if (config.role == config.fk_input_party) {
        fk_constraint_failed = fk_dense.contract_failed ? 1 : 0;
    }
    sync_value_from_party(config.role, config.fk_input_party, runtime, fk_constraint_failed);
    if (fk_constraint_failed != 0) {
        throw std::runtime_error("public capacity contract not satisfied");
    }

    ensure_dir(config.output_prefix);
    const std::string incomplete_path = config.output_prefix + "/INCOMPLETE";
    if (config.role == 0) {
        if (::access((config.output_prefix + "/manifest.json").c_str(), F_OK) == 0) {
            throw std::runtime_error("streaming final artifact already exists");
        }
        write_text_file(incomplete_path, "incomplete\n");
    }
    multiplier_public_barrier(config.role, runtime);
    maybe_fail_streaming_writer("before_partition_0");

    const std::string role_dir = config.output_prefix + "/role_" + std::to_string(config.role);
    ensure_dir(role_dir);
    const std::string top_mu_tmp = role_dir + "/mu.shares.bin.tmp";
    const std::string top_mu_nn_tmp = role_dir + "/mu_nn.shares.bin.tmp";
    {
        std::ofstream(top_mu_tmp, std::ios::binary | std::ios::trunc).close();
        std::ofstream(top_mu_nn_tmp, std::ios::binary | std::ios::trunc).close();
    }

    const i64 fixed_scale = static_cast<i64>(std::llround(
        static_cast<double>(kMultiplierFixedOne) / config.fk_sample_rate));

    u64 pk_offset = 0;
    u64 fk_offset = 0;
    for (u64 part = 0; part < partitions; ++part) {
        const u64 pk_cap = partition_contract.pk_capacities[static_cast<std::size_t>(part)];
        const u64 fk_cap = partition_contract.fk_capacities[static_cast<std::size_t>(part)];

        i64Matrix plain_pk_key(pk_cap, 1);
        i64Matrix plain_pk_is_null(pk_cap, 1);
        i64Matrix plain_pk_original_index(pk_cap, 1);
        i64Matrix plain_fk_key(fk_cap, 1);
        i64Matrix plain_fk_is_null(fk_cap, 1);
        i64Matrix plain_fk_original_index(fk_cap, 1);
        i64Matrix plain_fk_multiplicity(fk_cap, 1);
        plain_pk_key.setZero();
        plain_pk_is_null.setConstant(1);
        plain_pk_original_index.setZero();
        plain_fk_key.setZero();
        plain_fk_is_null.setConstant(1);
        plain_fk_original_index.setZero();
        plain_fk_multiplicity.setZero();

        u64 pk_overflow = 0;
        u64 fk_overflow = 0;
        if (config.role == config.pk_input_party) {
            bool overflow = false;
            const auto rows = provider_local_pk_rows_from_dense_state_for_part(
                pk_dense, partition_contract, part, pk_cap, overflow);
            populate_plain_rows(rows, plain_pk_key, plain_pk_is_null, plain_pk_original_index);
            pk_overflow = overflow ? 1 : 0;
        }
        if (config.role == config.fk_input_party) {
            bool overflow = false;
            const auto rows = provider_local_fk_group_rows_from_dense_state_for_part(
                fk_dense, partition_contract, part, fk_cap, overflow);
            populate_plain_rows(rows, plain_fk_key, plain_fk_is_null, plain_fk_original_index, &plain_fk_multiplicity);
            fk_overflow = overflow ? 1 : 0;
        }
        sync_value_from_party(config.role, config.pk_input_party, runtime, pk_overflow);
        sync_value_from_party(config.role, config.fk_input_party, runtime, fk_overflow);
        if (pk_overflow != 0 || fk_overflow != 0) {
            throw std::runtime_error("public capacity contract not satisfied");
        }

        si64Matrix pk_key_shared;
        si64Matrix pk_null_shared;
        si64Matrix pk_original_index_shared;
        si64Matrix fk_key_shared;
        si64Matrix fk_null_shared;
        si64Matrix fk_original_index_shared;
        si64Matrix fk_multiplicity_shared;
        secure_share_i64_column(config.role, config.pk_input_party, plain_pk_key, pk_key_shared, enc, runtime);
        secure_share_i64_column(config.role, config.pk_input_party, plain_pk_is_null, pk_null_shared, enc, runtime);
        secure_share_i64_column(config.role, config.pk_input_party, plain_pk_original_index, pk_original_index_shared, enc, runtime);
        secure_share_i64_column(config.role, config.fk_input_party, plain_fk_key, fk_key_shared, enc, runtime);
        secure_share_i64_column(config.role, config.fk_input_party, plain_fk_is_null, fk_null_shared, enc, runtime);
        secure_share_i64_column(config.role, config.fk_input_party, plain_fk_original_index, fk_original_index_shared, enc, runtime);
        secure_share_i64_column(config.role, config.fk_input_party, plain_fk_multiplicity, fk_multiplicity_shared, enc, runtime);

        auto counts = provider_separated_oblivious_merge_counts(
            config,
            pk_key_shared,
            pk_null_shared,
            pk_original_index_shared,
            fk_key_shared,
            fk_null_shared,
            fk_original_index_shared,
            fk_multiplicity_shared,
            pk_cap,
            fk_cap,
            enc,
            eval,
            runtime);

        si64Matrix mu_fixed(pk_cap, 1);
        mu_fixed.mShares[0].setZero();
        mu_fixed.mShares[1].setZero();
        for (u64 row = 0; row < pk_cap; ++row) {
            mu_fixed.mShares[0](row, 0) = counts.mShares[0](row, 0) * fixed_scale;
            mu_fixed.mShares[1](row, 0) = counts.mShares[1](row, 0) * fixed_scale;
        }
        auto zero_counts = shared_zero_int_matrix(pk_cap, 1);
        sbMatrix is_zero_count;
        cipher_eq(config.role, counts, zero_counts, is_zero_count, eval, runtime);
        const auto one_scalar = share_int_scalar(kMultiplierFixedOne, 0, enc, runtime, config.role);
        const auto one_rows = repeat_int_scalar_rows(one_scalar, pk_cap);
        si64Matrix mu_nn_fixed = select_si64_by_bool(
            one_rows,
            mu_fixed,
            is_zero_count,
            config.role,
            enc,
            eval,
            runtime);

        const std::string partition_dir = config.output_prefix + "/partitions/" + partition_dir_name(part);
        ensure_dir(config.output_prefix + "/partitions");
        ensure_dir(partition_dir);
        const std::string part_role_dir = partition_dir + "/role_" + std::to_string(config.role);
        ensure_dir(part_role_dir);
        write_share_pair_matrix_atomic(part_role_dir + "/mu.shares.bin", mu_fixed);
        write_share_pair_matrix_atomic(part_role_dir + "/mu_nn.shares.bin", mu_nn_fixed);
        append_share_pair_matrix(top_mu_tmp, mu_fixed);
        append_share_pair_matrix(top_mu_nn_tmp, mu_nn_fixed);

        pk_offset += pk_cap;
        fk_offset += fk_cap;
        maybe_fail_streaming_writer("after_partition", part);
    }
    (void) pk_offset;
    (void) fk_offset;

    if (std::rename(top_mu_tmp.c_str(), (role_dir + "/mu.shares.bin").c_str()) != 0) {
        throw std::runtime_error("Failed to publish role mu shares: " + std::string(std::strerror(errno)));
    }
    if (std::rename(top_mu_nn_tmp.c_str(), (role_dir + "/mu_nn.shares.bin").c_str()) != 0) {
        throw std::runtime_error("Failed to publish role mu_nn shares: " + std::string(std::strerror(errno)));
    }
    maybe_fail_streaming_writer("after_role_publish");
    multiplier_public_barrier(config.role, runtime);

    if (config.role == 0) {
        std::vector<StreamingPartitionMeta> metas;
        metas.reserve(static_cast<std::size_t>(partitions));
        for (u64 part = 0; part < partitions; ++part) {
            StreamingPartitionMeta meta;
            meta.partition_id = part;
            meta.pk_capacity = partition_contract.pk_capacities[static_cast<std::size_t>(part)];
            meta.fk_capacity = partition_contract.fk_capacities[static_cast<std::size_t>(part)];
            for (int role = 0; role < 3; ++role) {
                const std::string base_rel = "partitions/" + partition_dir_name(part) + "/role_" + std::to_string(role);
                const std::string mu_rel = base_rel + "/mu.shares.bin";
                const std::string mu_nn_rel = base_rel + "/mu_nn.shares.bin";
                const std::string mu_path = config.output_prefix + "/" + mu_rel;
                const std::string mu_nn_path = config.output_prefix + "/" + mu_nn_rel;
                meta.mu[role] = {mu_rel, file_size_bytes(mu_path), sha256_file_external(mu_path)};
                meta.mu_nn[role] = {mu_nn_rel, file_size_bytes(mu_nn_path), sha256_file_external(mu_nn_path)};
            }
            metas.push_back(std::move(meta));
        }
        maybe_fail_streaming_writer("before_manifest_publish");
        const auto manifest = streaming_manifest_base(
            config,
            static_cast<size_t>(total_pk_capacity),
            static_cast<size_t>(total_fk_capacity),
            partition_contract,
            metas,
            config.output_prefix);
        write_streaming_manifest_atomic(config.output_prefix + "/manifest.json", manifest);
        maybe_fail_streaming_writer("after_manifest_publish");
        if (std::remove(incomplete_path.c_str()) != 0) {
            throw std::runtime_error("Failed to remove INCOMPLETE marker: " + std::string(std::strerror(errno)));
        }
    }
    multiplier_public_barrier(config.role, runtime);
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
    const bool partitioned = secure_multiplier_provider_separated_partitioned_enabled();
    const bool provider_local_fk_group_count = secure_multiplier_provider_local_fk_group_count_enabled();
    if (provider_local_fk_group_count && !partitioned) {
        throw std::runtime_error("Provider-local FK group/count requires partitioned provider-separated preprocessing.");
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

    const PartitionContract partition_contract = partitioned ? parse_partition_contract() : PartitionContract{};
    const u64 pk_share_rows = partitioned ? sum_u64(partition_contract.pk_capacities) : n_pk;
    const u64 fk_share_rows = partitioned ? sum_u64(partition_contract.fk_capacities) : n_fk;
    u64 pk_overflow = 0;
    u64 fk_overflow = 0;

    i64Matrix plain_pk_key(pk_share_rows, 1);
    i64Matrix plain_pk_is_null(pk_share_rows, 1);
    i64Matrix plain_pk_original_index(pk_share_rows, 1);
    i64Matrix plain_fk_key(fk_share_rows, 1);
    i64Matrix plain_fk_is_null(fk_share_rows, 1);
    i64Matrix plain_fk_original_index(fk_share_rows, 1);
    i64Matrix plain_fk_multiplicity(fk_share_rows, 1);
    plain_pk_key.setZero();
    plain_pk_is_null.setConstant(1);
    plain_pk_original_index.setZero();
    plain_fk_key.setZero();
    plain_fk_is_null.setConstant(1);
    plain_fk_original_index.setZero();
    plain_fk_multiplicity.setConstant(1);

    if (config.role == config.pk_input_party) {
        if (static_cast<u64>(pk_keys.size()) != n_pk) {
            throw std::runtime_error("PK owner has inconsistent key row count.");
        }
        bool overflow = false;
        const bool pk_constraint_failed = provider_local_fk_group_count &&
            !provider_local_keys_unique_in_public_domain(pk_keys, partition_contract);
        const auto rows = partitioned
            ? provider_local_partitioned_rows(
                pk_keys,
                partition_contract,
                partition_contract.pk_capacities,
                overflow)
            : (secure_multiplier_provider_separated_oblivious_merge_enabled()
                ? provider_local_sorted_rows(pk_keys)
                : std::vector<ProviderLocalKeyRow>{});
        pk_overflow = (overflow || pk_constraint_failed) ? 1 : 0;
        for (u64 row = 0; row < pk_share_rows; ++row) {
            const auto& local_row = (partitioned || secure_multiplier_provider_separated_oblivious_merge_enabled())
                ? rows[static_cast<std::size_t>(row)]
                : ProviderLocalKeyRow{pk_keys[static_cast<std::size_t>(row)], row};
            plain_pk_key(row, 0) = local_row.key.value;
            plain_pk_is_null(row, 0) = local_row.key.is_null ? 1 : 0;
            plain_pk_original_index(row, 0) = static_cast<i64>(local_row.original_index);
        }
    }
    if (config.role == config.fk_input_party) {
        if (static_cast<u64>(fk_keys.size()) != n_fk) {
            throw std::runtime_error("FK owner has inconsistent key row count.");
        }
        bool overflow = false;
        const auto rows = partitioned
            ? (provider_local_fk_group_count
                ? provider_local_partitioned_fk_group_rows(
                    fk_keys,
                    partition_contract,
                    partition_contract.fk_capacities,
                    overflow)
                : provider_local_partitioned_rows(
                    fk_keys,
                    partition_contract,
                    partition_contract.fk_capacities,
                    overflow))
            : (secure_multiplier_provider_separated_oblivious_merge_enabled()
                ? provider_local_sorted_rows(fk_keys)
                : std::vector<ProviderLocalKeyRow>{});
        fk_overflow = overflow ? 1 : 0;
        for (u64 row = 0; row < fk_share_rows; ++row) {
            const auto& local_row = (partitioned || secure_multiplier_provider_separated_oblivious_merge_enabled())
                ? rows[static_cast<std::size_t>(row)]
                : ProviderLocalKeyRow{fk_keys[static_cast<std::size_t>(row)], row};
            plain_fk_key(row, 0) = local_row.key.value;
            plain_fk_is_null(row, 0) = local_row.key.is_null ? 1 : 0;
            plain_fk_original_index(row, 0) = static_cast<i64>(local_row.original_index);
            plain_fk_multiplicity(row, 0) = local_row.multiplicity;
        }
    }

    if (partitioned) {
        sync_value_from_party(config.role, config.pk_input_party, runtime, pk_overflow);
        sync_value_from_party(config.role, config.fk_input_party, runtime, fk_overflow);
        if (pk_overflow != 0 || fk_overflow != 0) {
            throw std::runtime_error("public capacity contract not satisfied");
        }
    }

    si64Matrix pk_key_shared;
    si64Matrix pk_null_shared;
    si64Matrix pk_original_index_shared;
    si64Matrix fk_key_shared;
    si64Matrix fk_null_shared;
    si64Matrix fk_original_index_shared;
    si64Matrix fk_multiplicity_shared;
    secure_share_i64_column(config.role, config.pk_input_party, plain_pk_key, pk_key_shared, enc, runtime);
    secure_share_i64_column(config.role, config.pk_input_party, plain_pk_is_null, pk_null_shared, enc, runtime);
    secure_share_i64_column(config.role, config.pk_input_party, plain_pk_original_index, pk_original_index_shared, enc, runtime);
    secure_share_i64_column(config.role, config.fk_input_party, plain_fk_key, fk_key_shared, enc, runtime);
    secure_share_i64_column(config.role, config.fk_input_party, plain_fk_is_null, fk_null_shared, enc, runtime);
    secure_share_i64_column(config.role, config.fk_input_party, plain_fk_original_index, fk_original_index_shared, enc, runtime);
    secure_share_i64_column(config.role, config.fk_input_party, plain_fk_multiplicity, fk_multiplicity_shared, enc, runtime);

    si64Matrix counts(n_pk, 1);
    counts.mShares[0].setZero();
    counts.mShares[1].setZero();

    if (partitioned) {
        counts = provider_separated_partitioned_oblivious_merge_counts(
            config,
            partition_contract,
            pk_key_shared,
            pk_null_shared,
            pk_original_index_shared,
            fk_key_shared,
            fk_null_shared,
            fk_original_index_shared,
            fk_multiplicity_shared,
            enc,
            eval,
            runtime);
        n_pk = pk_share_rows;
        n_fk = fk_share_rows;
    } else if (secure_multiplier_provider_separated_oblivious_merge_enabled()) {
        counts = provider_separated_oblivious_merge_counts(
            config,
            pk_key_shared,
            pk_null_shared,
            pk_original_index_shared,
            fk_key_shared,
            fk_null_shared,
            fk_original_index_shared,
            fk_multiplicity_shared,
            n_pk,
            n_fk,
            enc,
            eval,
            runtime);
    } else if (secure_multiplier_provider_separated_sorted_enabled()) {
        counts = provider_separated_sorted_counts(
            config,
            pk_key_shared,
            pk_null_shared,
            fk_key_shared,
            fk_null_shared,
            n_pk,
            n_fk,
            enc,
            eval,
            runtime);
    } else if (n_fk > 0) {
        auto fk_not_null = int_eq_public(fk_null_shared, 0, config.role, eval, runtime);
        for (u64 pk_row = 0; pk_row < n_pk; ++pk_row) {
            auto pk_key_rows = repeat_int_scalar_rows(int_row_slice(pk_key_shared, pk_row, 1), n_fk);
            auto pk_null_rows = repeat_int_scalar_rows(int_row_slice(pk_null_shared, pk_row, 1), n_fk);
            sbMatrix key_equal;
            cipher_eq(config.role, pk_key_rows, fk_key_shared, key_equal, eval, runtime);
            auto pk_not_null = int_eq_public(pk_null_rows, 0, config.role, eval, runtime);
            auto valid = local_bool_and_matrix(key_equal, pk_not_null, config.role, enc, eval, runtime);
            valid = local_bool_and_matrix(valid, fk_not_null, config.role, enc, eval, runtime);
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
        if (secure_multiplier_streaming_artifact_enabled()) {
            run_secure_multiplier_shared_values_provider_separated_streaming(config);
            return;
        }
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
        prefix = local_bool_or_matrix(prefix, shifted, role, enc, eval, runtime);
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

std::vector<u64> json_u64_array_or_csv(const json& doc, const std::string& key)
{
    if (!doc.contains(key)) {
        throw std::runtime_error("Secure leaf reversible lookup plan is missing " + key + ".");
    }
    std::vector<u64> out;
    const auto& value = doc[key];
    if (value.is_array()) {
        for (const auto& item : value) {
            out.push_back(item.get<std::uint64_t>());
        }
        return out;
    }
    if (value.is_string()) {
        std::stringstream stream(value.get<std::string>());
        std::string part;
        while (std::getline(stream, part, ',')) {
            const auto trimmed = trim_copy(part);
            if (!trimmed.empty()) {
                out.push_back(static_cast<u64>(std::stoull(trimmed)));
            }
        }
        return out;
    }
    throw std::runtime_error("Secure leaf reversible lookup plan key is not an array/csv: " + key + ".");
}

u64 next_power_of_two_u64(u64 value)
{
    if (value <= 1) {
        return 1;
    }
    u64 out = 1;
    while (out < value) {
        if (out > (std::numeric_limits<u64>::max() >> 1)) {
            throw std::runtime_error("Public padded network size overflow.");
        }
        out <<= 1;
    }
    return out;
}

si64Matrix load_secure_leaf_row_values_reversible_lookup(
    const json& plan,
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    u64 total_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const std::string mapping_mode = plan.value("mapping_mode", std::string());
    if (mapping_mode != "partition_replicated_reversible_bitonic_lookup") {
        throw std::runtime_error("Unsupported secure leaf secret mapping mode: " + mapping_mode);
    }
    const std::string artifact_dir = leaf_doc.value("multiplier_artifact_dir", std::string());
    if (artifact_dir.empty()) {
        throw std::runtime_error("Secure leaf reversible lookup plan is missing multiplier_artifact_dir.");
    }
    const std::string kind = leaf_doc.value("multiplier_kind", std::string("mu"));
    if (kind != "mu" && kind != "mu_nn") {
        throw std::runtime_error("Unsupported secure multiplier leaf multiplier_kind: " + kind);
    }
    const auto source_capacities = json_u64_array_or_csv(plan, "partition_source_capacities");
    const auto source_offsets = json_u64_array_or_csv(plan, "partition_source_offsets");
    if (source_capacities.empty() || source_capacities.size() != source_offsets.size()) {
        throw std::runtime_error("Secure leaf reversible lookup partition capacity/offset mismatch.");
    }
    const u64 partition_count = static_cast<u64>(source_capacities.size());
    if (plan.value("partition_count", partition_count) != partition_count) {
        throw std::runtime_error("Secure leaf reversible lookup partition_count mismatch.");
    }
    if (plan.value("public_query_slots", total_rows) != total_rows) {
        throw std::runtime_error("Secure leaf reversible lookup query slot count mismatch.");
    }
    u64 source_domain_size = 0;
    for (const auto cap : source_capacities) {
        source_domain_size += cap;
    }
    if (plan.value("source_domain_size", source_domain_size) != source_domain_size) {
        throw std::runtime_error("Secure leaf reversible lookup source domain size mismatch.");
    }

    const auto load_started = SteadyClock::now();
    const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
    const std::string valid_file = plan.value("mapping_valid_in_partition_share_file", std::string());
    const std::string dense_file = plan.value("mapping_dense_key_share_file", std::string());
    if (valid_file.empty() || dense_file.empty()) {
        throw std::runtime_error("Secure leaf reversible lookup mapping share files are missing.");
    }
    const u64 mapping_rows = partition_count * total_rows;
    auto mapping_valid = read_bool_share_pair_matrix(join_path(plan_role_dir, valid_file), mapping_rows, 1);
    auto mapping_dense_key = read_share_pair_matrix(join_path(plan_role_dir, dense_file), mapping_rows);

    const std::string resolved_artifact_dir = resolve_plan_relative_path(plan_base_dir, artifact_dir);
    const std::string role_dir = join_path(resolved_artifact_dir, "role_" + std::to_string(role));
    const auto full_values = read_share_pair_matrix_auto_rows(join_path(role_dir, kind + ".shares.bin"));
    if (full_values.rows() != source_domain_size) {
        throw std::runtime_error("Secure leaf reversible lookup source artifact row count does not match public domain size.");
    }
    const double load_seconds = elapsed_seconds_since(load_started);

    si64Matrix final_values(total_rows, 1);
    final_values.mShares[0].setZero();
    final_values.mShares[1].setZero();
    const bool profile = secure_multiplier_public_profile_enabled();
    double forward_seconds = 0.0;
    double propagate_seconds = 0.0;
    double reverse_seconds = 0.0;
    double mask_sum_seconds = 0.0;
    u64 total_forward_comparators = 0;
    u64 total_forward_layers = 0;

    for (u64 partition_id = 0; partition_id < partition_count; ++partition_id) {
        const u64 source_capacity = source_capacities[static_cast<std::size_t>(partition_id)];
        const u64 source_offset = source_offsets[static_cast<std::size_t>(partition_id)];
        const u64 real_rows = source_capacity + total_rows;
        const u64 padded_rows = next_power_of_two_u64(real_rows);
        const u64 pad_rows = padded_rows - real_rows;
        const u64 mapping_offset = partition_id * total_rows;

        auto source_invalid = public_i64_column(0, source_capacity, role);
        si64Matrix source_key(source_capacity, 1);
        source_key.mShares[0].setZero();
        source_key.mShares[1].setZero();
        for (u64 row = 0; row < source_capacity; ++row) {
            set_public_i64_cell(source_key, row, static_cast<i64>(source_offset + row), role);
        }
        auto source_origin = public_i64_column(0, source_capacity, role);
        auto source_tie = public_i64_row_ids(source_capacity, role);
        auto source_value = int_row_slice(full_values, source_offset, source_capacity);
        auto source_fetched = shared_zero_int_matrix(source_capacity, 1);

        auto query_valid_bool = bool_row_slice(mapping_valid, mapping_offset, total_rows);
        auto query_valid_int = bool_to_si64(query_valid_bool, role, enc, eval, runtime);
        auto query_invalid = public_i64_column(1, total_rows, role) - query_valid_int;
        auto query_key = int_row_slice(mapping_dense_key, mapping_offset, total_rows);
        auto query_origin = public_i64_column(1, total_rows, role);
        auto query_tie = public_i64_row_ids(total_rows, role);
        auto query_value = shared_zero_int_matrix(total_rows, 1);
        auto query_fetched = shared_zero_int_matrix(total_rows, 1);

        auto pad_invalid = public_i64_column(1, pad_rows, role);
        auto pad_key = public_i64_column(0, pad_rows, role);
        auto pad_origin = public_i64_column(2, pad_rows, role);
        auto pad_tie = public_i64_row_ids(pad_rows, role);
        auto pad_value = shared_zero_int_matrix(pad_rows, 1);
        auto pad_fetched = shared_zero_int_matrix(pad_rows, 1);

        enum ReversibleLookupColumn : u64 {
            kInvalidRank = 0,
            kDenseKey = 1,
            kOrigin = 2,
            kTie = 3,
            kValue = 4,
            kFetched = 5,
        };
        std::vector<si64Matrix> columns;
        columns.push_back(concat_shared_row_blocks({source_invalid, query_invalid, pad_invalid}));
        columns.push_back(concat_shared_row_blocks({source_key, query_key, pad_key}));
        columns.push_back(concat_shared_row_blocks({source_origin, query_origin, pad_origin}));
        columns.push_back(concat_shared_row_blocks({source_tie, query_tie, pad_tie}));
        columns.push_back(concat_shared_row_blocks({source_value, query_value, pad_value}));
        columns.push_back(concat_shared_row_blocks({source_fetched, query_fetched, pad_fetched}));
        const std::vector<u64> comparator_fields = {kInvalidRank, kDenseKey, kOrigin, kTie};

        u64 layer_count = 0;
        u64 comparator_count = 0;
        const auto forward_started = SteadyClock::now();
        auto stored_layers = reversible_bitonic_forward_sort_rows(
            columns,
            comparator_fields,
            role,
            enc,
            eval,
            runtime,
            layer_count,
            comparator_count);
        const double partition_forward_seconds = elapsed_seconds_since(forward_started);
        forward_seconds += partition_forward_seconds;
        total_forward_layers += layer_count;
        total_forward_comparators += comparator_count;

        const auto propagate_started = SteadyClock::now();
        auto is_source = int_eq_public(columns[kOrigin], 0, role, eval, runtime);
        auto is_query = int_eq_public(columns[kOrigin], 1, role, eval, runtime);
        auto is_active = int_eq_public(columns[kInvalidRank], 0, role, eval, runtime);
        is_source = local_bool_and_matrix(is_source, is_active, role, enc, eval, runtime);
        is_query = local_bool_and_matrix(is_query, is_active, role, enc, eval, runtime);
        si64Matrix running(padded_rows, 1);
        running.mShares[0].setZero();
        running.mShares[1].setZero();
        si64Matrix fetched(padded_rows, 1);
        fetched.mShares[0].setZero();
        fetched.mShares[1].setZero();
        auto zero_row = shared_zero_int_matrix(1, 1);
        for (u64 row = 0; row < padded_rows; ++row) {
            auto value_row = int_row_slice(columns[kValue], row, 1);
            auto source_flag = bool_row_slice(is_source, row, 1);
            auto query_flag = bool_row_slice(is_query, row, 1);
            auto source_payload = select_si64_by_bool(value_row, zero_row, source_flag, role, enc, eval, runtime);
            si64Matrix current_running(1, 1);
            if (row == 0) {
                current_running = source_payload;
            } else {
                auto lhs_key = int_row_slice(columns[kDenseKey], row, 1);
                auto rhs_key = int_row_slice(columns[kDenseKey], row - 1, 1);
                auto same_key = int_eq_matrix(lhs_key, rhs_key, role, eval, runtime);
                auto current_active = bool_row_slice(is_active, row, 1);
                auto prev_active = bool_row_slice(is_active, row - 1, 1);
                same_key = local_bool_and_matrix(same_key, current_active, role, enc, eval, runtime);
                same_key = local_bool_and_matrix(same_key, prev_active, role, enc, eval, runtime);
                auto previous_running = int_row_slice(running, row - 1, 1);
                auto same_group_value = select_si64_by_bool(
                    source_payload,
                    previous_running,
                    source_flag,
                    role,
                    enc,
                    eval,
                    runtime);
                current_running = select_si64_by_bool(
                    same_group_value,
                    source_payload,
                    same_key,
                    role,
                    enc,
                    eval,
                    runtime);
            }
            running.mShares[0](row, 0) = current_running.mShares[0](0, 0);
            running.mShares[1](row, 0) = current_running.mShares[1](0, 0);
            auto fetched_row = select_si64_by_bool(current_running, zero_row, query_flag, role, enc, eval, runtime);
            fetched.mShares[0](row, 0) = fetched_row.mShares[0](0, 0);
            fetched.mShares[1](row, 0) = fetched_row.mShares[1](0, 0);
        }
        columns[kFetched] = std::move(fetched);
        const double partition_propagate_seconds = elapsed_seconds_since(propagate_started);
        propagate_seconds += partition_propagate_seconds;

        const auto reverse_started = SteadyClock::now();
        reversible_bitonic_reverse_rows(columns, stored_layers, role, enc, eval, runtime);
        const double partition_reverse_seconds = elapsed_seconds_since(reverse_started);
        reverse_seconds += partition_reverse_seconds;
        stored_layers.clear();

        const auto mask_started = SteadyClock::now();
        auto partition_values = int_row_slice(columns[kFetched], source_capacity, total_rows);
        auto partition_invalid = int_row_slice(columns[kInvalidRank], source_capacity, total_rows);
        auto partition_valid = int_eq_public(partition_invalid, 0, role, eval, runtime);
        auto zero_slots = shared_zero_int_matrix(total_rows, 1);
        auto selected_values = select_si64_by_bool(
            partition_values,
            zero_slots,
            partition_valid,
            role,
            enc,
            eval,
            runtime);
        final_values = final_values + selected_values;
        const double partition_mask_seconds = elapsed_seconds_since(mask_started);
        mask_sum_seconds += partition_mask_seconds;

        if (profile && role == 0) {
            std::cerr << "bspn_multiplier_profile: event=reversible_lookup_partition"
                      << " partition=" << partition_id
                      << " source_capacity=" << source_capacity
                      << " query_slots=" << total_rows
                      << " padded_rows=" << padded_rows
                      << " layers=" << layer_count
                      << " forward_comparators=" << comparator_count
                      << " reverse_swaps=" << comparator_count
                      << " swap_bit_shares=" << comparator_count
                      << " swap_bit_pair_bytes_estimate=" << (comparator_count * sizeof(i64) * 2)
                      << " forward_seconds=" << partition_forward_seconds
                      << " propagation_seconds=" << partition_propagate_seconds
                      << " reverse_seconds=" << partition_reverse_seconds
                      << " mask_sum_seconds=" << partition_mask_seconds
                      << std::endl;
        }
    }

    if (profile && role == 0) {
        std::cerr << "bspn_multiplier_profile: event=reversible_lookup_total"
                  << " partitions=" << partition_count
                  << " query_slots=" << total_rows
                  << " forward_layers=" << total_forward_layers
                  << " forward_comparators=" << total_forward_comparators
                  << " reverse_swaps=" << total_forward_comparators
                  << " swap_bit_shares=" << total_forward_comparators
                  << " mapping_load_seconds=" << load_seconds
                  << " forward_seconds=" << forward_seconds
                  << " propagation_seconds=" << propagate_seconds
                  << " reverse_seconds=" << reverse_seconds
                  << " mask_sum_seconds=" << mask_sum_seconds
                  << std::endl;
    }
    return final_values;
}

si64Matrix load_secure_leaf_row_values(
    const json& plan,
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    u64 total_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const std::string sampled_row_values_file = leaf_doc.value("sampled_row_values_share_file", std::string());
    if (!sampled_row_values_file.empty()) {
        const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, role);
        return read_share_pair_matrix(join_path(plan_role_dir, sampled_row_values_file), total_rows);
    }

    if (plan.value("mapping_mode", std::string()) == "partition_replicated_reversible_bitonic_lookup") {
        return load_secure_leaf_row_values_reversible_lookup(
            plan,
            leaf_doc,
            plan_base_dir,
            plan_role_share_dir,
            total_rows,
            role,
            enc,
            eval,
            runtime);
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
        auto no_prior = local_bool_not_matrix(has_prior, role);
        auto representative_rows = local_bool_and_matrix(remaining, no_prior, role, enc, eval, runtime);

        auto selected_values = arith_mul_bool(values, representative_rows, role, enc, eval, runtime);
        bucket_values.mShares[0](bucket_idx, 0) = selected_values.mShares[0].sum();
        bucket_values.mShares[1](bucket_idx, 0) = selected_values.mShares[1].sum();

        auto repeated_bucket_value = repeat_shared_row(bucket_values, bucket_idx, total_rows);
        sbMatrix value_matches;
        auto values_copy = values;
        cipher_eq(role, values_copy, repeated_bucket_value, value_matches, eval, runtime);
        auto bucket_rows = local_bool_and_matrix(value_matches, remaining, role, enc, eval, runtime);
        const u64 out_begin = bucket_idx * total_rows;
        for (u64 row = 0; row < total_rows; ++row) {
            bucket_bitmaps.mShares[0](out_begin + row, 0) = bucket_rows.mShares[0](row, 0);
            bucket_bitmaps.mShares[1](out_begin + row, 0) = bucket_rows.mShares[1](row, 0);
        }

        auto not_bucket_rows = local_bool_not_matrix(bucket_rows, role);
        remaining = local_bool_and_matrix(remaining, not_bucket_rows, role, enc, eval, runtime);
    }
    const double bucket_loop_elapsed = elapsed_seconds_since(bucket_loop_started);

    const auto overflow_started = SteadyClock::now();
    auto leaf_overflow = bool_reduce_or_matrix(remaining, role, enc, eval, runtime);
    model_overflow = local_bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);
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
    auto group_has_member = local_bool_not_matrix(group_empty, role);
    auto group_start = local_bool_not_matrix(same_previous, role);
    auto active_group_start = local_bool_and_matrix(group_start, group_has_member, role, enc, eval, runtime);
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
        auto bucket_start = local_bool_and_matrix(rank_match, active_group_start, role, enc, eval, runtime);
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
    auto overflow_starts = local_bool_and_matrix(
        overflow_rank_match,
        active_group_start,
        role,
        enc,
        eval,
        runtime);
    auto leaf_overflow = bool_reduce_or_matrix(overflow_starts, role, enc, eval, runtime);
    model_overflow = local_bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);

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
        auto bucket_rows = local_bool_and_matrix(rank_match, membership_by_row_bool, role, enc, eval, runtime);
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
        auto group_has_member = local_bool_not_matrix(group_empty, role);
        auto group_start = local_bool_not_matrix(same_previous, role);
        auto active_group_start = local_bool_and_matrix(group_start, group_has_member, role, enc, eval, runtime);
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
            auto bucket_start = local_bool_and_matrix(rank_match, active_group_start, role, enc, eval, runtime);
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
        auto overflow_starts = local_bool_and_matrix(
            overflow_rank_match,
            active_group_start,
            role,
            enc,
            eval,
            runtime);
        auto leaf_overflow = bool_reduce_or_matrix(overflow_starts, role, enc, eval, runtime);
        model_overflow = local_bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);

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
            auto bucket_rows = local_bool_and_matrix(rank_match, membership_by_row_bool, role, enc, eval, runtime);
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
                    config.role,
                    enc,
                    eval,
                    runtime);
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
                    config.role,
                    enc,
                    eval,
                    runtime);
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

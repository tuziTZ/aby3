#include "MultiplierPreprocessing.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
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
    std::string mode = "reference";
    int role = -1;
    int input_party = 0;
    int pk_input_party = 0;
    int fk_input_party = 1;
    u64 secure_sort_min_size = 32;
};

void secure_share_i64_column(
    int role,
    int input_party,
    const i64Matrix& plain,
    si64Matrix& shared,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime);

std::string json_escape(const std::string& input);
void write_text_file(const std::string& path, const std::string& content);

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

void compute_pk_side_multiplier_reference(
    const std::vector<NormalizedKey>& pk_keys,
    const std::vector<NormalizedKey>& fk_keys,
    double fk_sample_rate,
    std::vector<double>& mu_out,
    std::vector<double>& mu_nn_out)
{
    if (fk_sample_rate <= 0.0) {
        throw std::runtime_error("FK sample rate must be positive.");
    }

    std::unordered_map<i64, i64> fk_counts;
    for (const auto& fk : fk_keys) {
        if (!fk.is_null) {
            ++fk_counts[fk.value];
        }
    }

    mu_out.assign(pk_keys.size(), 0.0);
    mu_nn_out.assign(pk_keys.size(), 1.0);

    const double inv_rate = 1.0 / fk_sample_rate;
    for (size_t i = 0; i < pk_keys.size(); ++i) {
        const auto& pk = pk_keys[i];
        if (pk.is_null) {
            mu_out[i] = 0.0;
            mu_nn_out[i] = 1.0;
            continue;
        }

        auto iter = fk_counts.find(pk.value);
        const double mu = (iter == fk_counts.end()) ? 0.0 : static_cast<double>(iter->second) * inv_rate;
        mu_out[i] = mu;
        mu_nn_out[i] = (mu == 0.0) ? 1.0 : mu;
    }
}

template <typename T>
void write_binary_vector(const std::string& path, const std::vector<T>& values)
{
    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
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

std::vector<std::pair<i64, i64>> read_share_pairs_file(const std::string& path, u64 rows)
{
    const auto raw = read_i64_records(path);
    if (raw.size() != static_cast<std::size_t>(rows) * 2) {
        throw std::runtime_error("Share pair file row count mismatch: " + path);
    }
    std::vector<std::pair<i64, i64>> out(static_cast<std::size_t>(rows));
    for (u64 row = 0; row < rows; ++row) {
        out[static_cast<std::size_t>(row)] = {
            raw[static_cast<std::size_t>(row) * 2],
            raw[static_cast<std::size_t>(row) * 2 + 1],
        };
    }
    return out;
}

std::vector<i64> reconstruct_arithmetic_share_pair_files(
    const std::string& role0_path,
    const std::string& role1_path,
    const std::string& role2_path,
    u64 rows)
{
    const auto role0 = read_share_pairs_file(role0_path, rows);
    const auto role1 = read_share_pairs_file(role1_path, rows);
    const auto role2 = read_share_pairs_file(role2_path, rows);
    std::vector<i64> out(static_cast<std::size_t>(rows), 0);
    for (u64 row = 0; row < rows; ++row) {
        const auto a0 = static_cast<std::uint64_t>(role0[static_cast<std::size_t>(row)].first);
        const auto a1 = static_cast<std::uint64_t>(role1[static_cast<std::size_t>(row)].first);
        const auto a2 = static_cast<std::uint64_t>(role2[static_cast<std::size_t>(row)].first);
        out[static_cast<std::size_t>(row)] = static_cast<i64>(a0 + a1 + a2);
    }
    return out;
}

std::vector<i64> reconstruct_boolean_share_pair_files(
    const std::string& role0_path,
    const std::string& role1_path,
    const std::string& role2_path,
    u64 rows)
{
    const auto role0 = read_share_pairs_file(role0_path, rows);
    const auto role1 = read_share_pairs_file(role1_path, rows);
    const auto role2 = read_share_pairs_file(role2_path, rows);
    std::vector<i64> out(static_cast<std::size_t>(rows), 0);
    for (u64 row = 0; row < rows; ++row) {
        const auto a0 = static_cast<std::uint64_t>(role0[static_cast<std::size_t>(row)].first);
        const auto a1 = static_cast<std::uint64_t>(role1[static_cast<std::size_t>(row)].first);
        const auto a2 = static_cast<std::uint64_t>(role2[static_cast<std::size_t>(row)].first);
        out[static_cast<std::size_t>(row)] = static_cast<i64>((a0 ^ a1 ^ a2) & 1ULL);
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

void write_reference_manifest(
    const MultiplierPreprocessConfig& config,
    size_t pk_row_count,
    size_t fk_row_count,
    const std::string& manifest_path,
    const std::string& mu_path,
    const std::string& mu_nn_path)
{
    std::ostringstream content;
    content << "{\n";
    content << "  \"format_name\": \"BSPN_MULTIPLIER_PAYLOAD\",\n";
    content << "  \"format_version\": 1,\n";
    content << "  \"mode\": \"reference\",\n";
    content << "  \"relationship_id\": \"" << json_escape(config.relationship_id) << "\",\n";
    content << "  \"pk_csv_path\": \"" << json_escape(config.pk_csv_path) << "\",\n";
    content << "  \"fk_csv_path\": \"" << json_escape(config.fk_csv_path) << "\",\n";
    content << "  \"pk_key_column\": " << config.pk_key_column << ",\n";
    content << "  \"fk_key_column\": " << config.fk_key_column << ",\n";
    content << "  \"fk_sample_rate\": " << std::setprecision(17) << config.fk_sample_rate << ",\n";
    content << "  \"pk_row_count\": " << pk_row_count << ",\n";
    content << "  \"fk_row_count\": " << fk_row_count << ",\n";
    content << "  \"mu_dtype\": \"float64\",\n";
    content << "  \"mu_nn_dtype\": \"float64\",\n";
    content << "  \"mu_file\": \"" << json_escape(mu_path) << "\",\n";
    content << "  \"mu_nn_file\": \"" << json_escape(mu_nn_path) << "\"\n";
    content << "}\n";
    write_text_file(manifest_path, content.str());
}

void write_secure_shared_values_manifest(
    const MultiplierPreprocessConfig& config,
    size_t pk_row_count,
    size_t fk_row_count,
    const std::string& artifact_dir)
{
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
    content << "  \"secure_core_status\": \"sorted_segmented_scan_count_no_reveal\",\n";
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

void run_reference_multiplier_preprocess(const MultiplierPreprocessConfig& config)
{
    const auto pk_keys = load_key_column_csv(config.pk_csv_path, config.pk_key_column, config.pk_has_header);
    const auto fk_keys = load_key_column_csv(config.fk_csv_path, config.fk_key_column, config.fk_has_header);

    std::vector<double> mu;
    std::vector<double> mu_nn;
    compute_pk_side_multiplier_reference(pk_keys, fk_keys, config.fk_sample_rate, mu, mu_nn);

    const std::string mu_path = config.output_prefix + ".mu.bin";
    const std::string mu_nn_path = config.output_prefix + ".mu_nn.bin";
    const std::string manifest_path = config.output_prefix + ".manifest.json";

    write_binary_vector(mu_path, mu);
    write_binary_vector(mu_nn_path, mu_nn);
    write_reference_manifest(config, pk_keys.size(), fk_keys.size(), manifest_path, mu_path, mu_nn_path);
}

void sync_scaffold_shape(int role, Sh3Runtime& runtime, std::array<u64, 3>& shape)
{
    if (role == 0) {
        runtime.mComm.mNext.asyncSendCopy(shape.data(), shape.size());
        runtime.mComm.mPrev.asyncSendCopy(shape.data(), shape.size());
    } else if (role == 1) {
        runtime.mComm.mPrev.recv(shape.data(), shape.size());
    } else if (role == 2) {
        runtime.mComm.mNext.recv(shape.data(), shape.size());
    } else {
        throw std::runtime_error("Invalid ABY3 role.");
    }
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

void write_secure_scaffold_manifest(
    const MultiplierPreprocessConfig& config,
    const std::array<u64, 3>& shape,
    const std::string& manifest_path)
{
    std::ostringstream content;
    content << "{\n";
    content << "  \"format_name\": \"BSPN_MULTIPLIER_PAYLOAD\",\n";
    content << "  \"format_version\": 1,\n";
    content << "  \"mode\": \"secure_scaffold\",\n";
    content << "  \"relationship_id\": \"" << json_escape(config.relationship_id) << "\",\n";
    content << "  \"pk_row_count\": " << shape[0] << ",\n";
    content << "  \"fk_row_count\": " << shape[1] << ",\n";
    content << "  \"padded_row_count\": " << shape[2] << ",\n";
    content << "  \"input_party\": " << config.input_party << ",\n";
    content << "  \"approved_helpers\": [\n";
    content << "    \"basic_setup\",\n";
    content << "    \"localIntMatrix/remoteIntMatrix\",\n";
    content << "    \"quick_sort_with_other_elements\"\n";
    content << "  ],\n";
    content << "  \"secure_core_status\": \"sorted_combined_rows_ready\",\n";
    content << "  \"next_stage\": \"replace scaffold manifest with secret multiplier payload emission\"\n";
    content << "}\n";
    write_text_file(manifest_path, content.str());
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

    i64 min_non_null_key = 0;
    bool have_non_null_key = false;
    if (role == sort_input_party) {
        auto update_min = [&](const std::vector<NormalizedKey>& keys) {
            for (const auto& key : keys) {
                if (key.is_null) {
                    continue;
                }
                if (!have_non_null_key) {
                    min_non_null_key = key.value;
                    have_non_null_key = true;
                    continue;
                }
                min_non_null_key = std::min(min_non_null_key, key.value);
            }
        };
        update_min(pk_keys);
        update_min(fk_keys);
    }

    const __int128 stride = static_cast<__int128>(std::max<u64>(1, padded));
    auto encode_key = [&](const NormalizedKey& key, i64 tag) -> i64 {
        if (key.is_null || !have_non_null_key) {
            return tag;
        }

        const __int128 base = static_cast<__int128>(1) +
            static_cast<__int128>(key.value) -
            static_cast<__int128>(min_non_null_key);
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
        const __int128 base = static_cast<__int128>(1) +
            static_cast<__int128>(key.value) -
            static_cast<__int128>(min_non_null_key);
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

void run_secure_multiplier_scaffold(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_scaffold mode requires --role in {0,1,2}.");
    }

    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(config.role), ios, enc, eval, runtime);

    std::vector<NormalizedKey> pk_keys;
    std::vector<NormalizedKey> fk_keys;
    std::array<u64, 3> shape = {0, 0, 0};

    if (config.role == config.input_party) {
        pk_keys = load_key_column_csv(config.pk_csv_path, config.pk_key_column, config.pk_has_header);
        fk_keys = load_key_column_csv(config.fk_csv_path, config.fk_key_column, config.fk_has_header);
        shape[0] = static_cast<u64>(pk_keys.size());
        shape[1] = static_cast<u64>(fk_keys.size());
        shape[2] = static_cast<u64>(roundUpToPowerOfTwo(pk_keys.size() + fk_keys.size()));
    }

    sync_scaffold_shape(config.role, runtime, shape);

    const size_t n_pk = static_cast<size_t>(shape[0]);
    const size_t n_fk = static_cast<size_t>(shape[1]);
    const size_t padded = static_cast<size_t>(shape[2]);

    i64Matrix plain_key(padded, 1);
    i64Matrix plain_table_id(padded, 1);
    i64Matrix plain_row_id(padded, 1);
    i64Matrix plain_is_null(padded, 1);
    i64Matrix plain_valid(padded, 1);
    i64Matrix plain_fk_contrib(padded, 1);

    plain_key.setZero();
    plain_table_id.setZero();
    plain_row_id.setZero();
    plain_is_null.setConstant(1);
    plain_valid.setZero();
    plain_fk_contrib.setZero();

    if (config.role == config.input_party) {
        for (size_t i = 0; i < n_pk; ++i) {
            plain_key(i, 0) = pk_keys[i].value;
            plain_table_id(i, 0) = 0;
            plain_row_id(i, 0) = static_cast<i64>(i);
            plain_is_null(i, 0) = pk_keys[i].is_null ? 1 : 0;
            plain_valid(i, 0) = 1;
        }

        for (size_t i = 0; i < n_fk; ++i) {
            const size_t row = n_pk + i;
            plain_key(row, 0) = fk_keys[i].value;
            plain_table_id(row, 0) = 1;
            plain_row_id(row, 0) = static_cast<i64>(i);
            plain_is_null(row, 0) = fk_keys[i].is_null ? 1 : 0;
            plain_valid(row, 0) = 1;
            plain_fk_contrib(row, 0) = fk_keys[i].is_null ? 0 : 1;
        }

        for (size_t row = n_pk + n_fk; row < padded; ++row) {
            plain_table_id(row, 0) = 2;
            plain_row_id(row, 0) = static_cast<i64>(row);
        }
    }

    si64Matrix sec_key;
    si64Matrix sec_table_id;
    si64Matrix sec_row_id;
    si64Matrix sec_is_null;
    si64Matrix sec_valid;
    si64Matrix sec_fk_contrib;

    secure_share_i64_column(config.role, config.input_party, plain_key, sec_key, enc, runtime);
    secure_share_i64_column(config.role, config.input_party, plain_table_id, sec_table_id, enc, runtime);
    secure_share_i64_column(config.role, config.input_party, plain_row_id, sec_row_id, enc, runtime);
    secure_share_i64_column(config.role, config.input_party, plain_is_null, sec_is_null, enc, runtime);
    secure_share_i64_column(config.role, config.input_party, plain_valid, sec_valid, enc, runtime);
    secure_share_i64_column(config.role, config.input_party, plain_fk_contrib, sec_fk_contrib, enc, runtime);

    std::vector<si64Matrix> payloads = {
        sec_table_id,
        sec_row_id,
        sec_is_null,
        sec_valid,
        sec_fk_contrib,
    };
    quick_sort_with_other_elements(sec_key, payloads, config.role, enc, eval, runtime, config.secure_sort_min_size);

    if (config.role == 0) {
        const std::string scaffold_path = config.output_prefix + ".secure_scaffold.json";
        write_secure_scaffold_manifest(config, shape, scaffold_path);
    }
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

void run_secure_multiplier_shared_values(const MultiplierPreprocessConfig& config)
{
    run_secure_multiplier_shared_values_fast(config);
    return;

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

si64Matrix secure_lookup_by_secret_index(
    const si64Matrix& values_by_pk,
    const si64Matrix& sample_pk_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    si64Matrix out(sample_pk_rows.rows(), 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    for (u64 pk_row = 0; pk_row < values_by_pk.rows(); ++pk_row) {
        auto sample_pk_rows_copy = sample_pk_rows;
        auto public_pk = public_i64_column(static_cast<i64>(pk_row), sample_pk_rows.rows(), role);
        sbMatrix is_row;
        cipher_eq(role, sample_pk_rows_copy, public_pk, is_row, eval, runtime);
        auto repeated_value = repeat_shared_row(values_by_pk, pk_row, sample_pk_rows.rows());
        auto contribution = arith_mul_bool(repeated_value, is_row, role, enc, eval, runtime);
        out = out + contribution;
    }
    return out;
}

u64 secure_lookup_batch_rows(u64 value_rows)
{
    const char* raw = std::getenv("SECURE_LEAF_LOOKUP_BATCH_ROWS");
    u64 configured = 16;
    if (raw != nullptr && std::strlen(raw) > 0) {
        const auto parsed = std::strtoull(raw, nullptr, 10);
        if (parsed > 0) {
            configured = static_cast<u64>(parsed);
        }
    }
    if (value_rows == 0) {
        return 1;
    }
    const u64 max_by_int_offset = static_cast<u64>(std::numeric_limits<int>::max() - 1) / value_rows;
    return std::max<u64>(1, std::min<u64>(configured, std::max<u64>(1, max_by_int_offset)));
}

u64 secure_lookup_chunked_min_expanded()
{
    const char* raw = std::getenv("SECURE_LEAF_LOOKUP_CHUNKED_MIN_EXPANDED");
    if (raw != nullptr && std::strlen(raw) > 0) {
        const auto parsed = std::strtoull(raw, nullptr, 10);
        if (parsed > 0) {
            return static_cast<u64>(parsed);
        }
    }
    return 1000000;
}

si64Matrix secure_lookup_by_secret_index_chunked(
    si64Matrix& values_by_pk,
    const si64Matrix& sample_pk_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const u64 total_rows = sample_pk_rows.rows();
    si64Matrix out(total_rows, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    if (total_rows == 0 || values_by_pk.rows() == 0) {
        return out;
    }
    if (values_by_pk.rows() <= std::numeric_limits<u64>::max() / total_rows &&
        values_by_pk.rows() * total_rows < secure_lookup_chunked_min_expanded()) {
        return secure_lookup_by_secret_index(values_by_pk, sample_pk_rows, role, enc, eval, runtime);
    }

    const u64 index_batch_rows = secure_lookup_batch_rows(values_by_pk.rows());
    constexpr u64 kLookupBlockSize = 50000;
    for (u64 row_begin = 0; row_begin < total_rows; row_begin += index_batch_rows) {
        const u64 row_count = std::min<u64>(index_batch_rows, total_rows - row_begin);
        si64Matrix batch_out(row_count, 1);
        batch_out.mShares[0].setZero();
        batch_out.mShares[1].setZero();
        const u64 expanded = values_by_pk.rows() * row_count;
        for (u64 offset = 0; offset < expanded; offset += kLookupBlockSize) {
            const u64 end = std::min<u64>(expanded, offset + kLookupBlockSize);
            const u64 block_rows = end - offset;
            si64Matrix sample_expanded(block_rows, 1);
            si64Matrix value_expanded(block_rows, 1);
            si64Matrix pk_public(block_rows, 1);
            pk_public.mShares[0].setZero();
            pk_public.mShares[1].setZero();
            for (u64 p = 0; p < block_rows; ++p) {
                const u64 flat = offset + p;
                const u64 local_row = flat / values_by_pk.rows();
                const u64 pk_row = flat % values_by_pk.rows();
                sample_expanded.mShares[0](p, 0) = sample_pk_rows.mShares[0](row_begin + local_row, 0);
                sample_expanded.mShares[1](p, 0) = sample_pk_rows.mShares[1](row_begin + local_row, 0);
                value_expanded.mShares[0](p, 0) = values_by_pk.mShares[0](pk_row, 0);
                value_expanded.mShares[1](p, 0) = values_by_pk.mShares[1](pk_row, 0);
                if (role == 0) {
                    pk_public.mShares[0](p, 0) = static_cast<i64>(pk_row);
                } else if (role == 1) {
                    pk_public.mShares[1](p, 0) = static_cast<i64>(pk_row);
                } else if (role != 2) {
                    throw std::runtime_error("Invalid ABY3 role.");
                }
            }
            sbMatrix is_row;
            cipher_eq(role, sample_expanded, pk_public, is_row, eval, runtime);
            auto contribution = arith_mul_bool(value_expanded, is_row, role, enc, eval, runtime);
            for (u64 p = 0; p < block_rows; ++p) {
                const u64 local_row = (offset + p) / values_by_pk.rows();
                batch_out.mShares[0](local_row, 0) += contribution.mShares[0](p, 0);
                batch_out.mShares[1](local_row, 0) += contribution.mShares[1](p, 0);
            }
        }
        out.mShares[0].block(row_begin, 0, row_count, 1) = batch_out.mShares[0];
        out.mShares[1].block(row_begin, 0, row_count, 1) = batch_out.mShares[1];
    }
    return out;
}

std::string secure_leaf_lookup_mode()
{
    const char* raw = std::getenv("SECURE_LEAF_LOOKUP_MODE");
    if (raw == nullptr || std::strlen(raw) == 0) {
        return "sort_join";
    }
    std::string mode(raw);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return mode;
}

int secure_leaf_oram_stash_size(u64 rows)
{
    const char* raw = std::getenv("SECURE_LEAF_ORAM_STASH_SIZE");
    if (raw != nullptr && std::strlen(raw) > 0) {
        const auto parsed = std::strtoull(raw, nullptr, 10);
        if (parsed > 0) {
            return static_cast<int>(parsed);
        }
    }
    return static_cast<int>(std::max<u64>(1, static_cast<u64>(std::ceil(std::sqrt(static_cast<double>(std::max<u64>(1, rows)))))));
}

int secure_leaf_oram_pack_size()
{
    const char* raw = std::getenv("SECURE_LEAF_ORAM_PACK_SIZE");
    if (raw != nullptr && std::strlen(raw) > 0) {
        const auto parsed = std::strtoull(raw, nullptr, 10);
        if (parsed > 0) {
            return static_cast<int>(parsed);
        }
    }
    return 16;
}

si64Matrix secure_lookup_by_secret_index_oram(
    si64Matrix& values_by_pk,
    si64Matrix sample_pk_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const u64 pk_rows = values_by_pk.rows();
    const u64 sample_rows = sample_pk_rows.rows();
    if (pk_rows > static_cast<u64>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("ORAM secure lookup does not support pk_row_count above int32 range.");
    }
    si64Matrix out(sample_rows, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    if (pk_rows == 0 || sample_rows == 0) {
        return out;
    }

    sbMatrix value_bits(pk_rows, 64);
    arith2bool(role, values_by_pk, value_bits, enc, eval, runtime);
    std::vector<sbMatrix> data(static_cast<std::size_t>(pk_rows));
    for (u64 row = 0; row < pk_rows; ++row) {
        data[static_cast<std::size_t>(row)] = bool_row_slice(value_bits, row, 1);
    }

    ABY3SqrtOram oram(
        static_cast<int>(pk_rows),
        secure_leaf_oram_stash_size(pk_rows),
        secure_leaf_oram_pack_size(),
        role,
        enc,
        eval,
        runtime);
    oram.initiate(data);

    sbMatrix index_bits(sample_rows, 64);
    arith2bool(role, sample_pk_rows, index_bits, enc, eval, runtime);
    sbMatrix out_bits(sample_rows, 64);
    out_bits.mShares[0].setZero();
    out_bits.mShares[1].setZero();
    for (u64 row = 0; row < sample_rows; ++row) {
        auto index_row = bool_row_slice(index_bits, row, 1);
        boolIndex index(index_row);
        auto value = oram.access(index);
        out_bits.mShares[0](row, 0) = value.mShares[0](0, 0);
        out_bits.mShares[1](row, 0) = value.mShares[1](0, 0);
    }
    bool2arith(role, out_bits, out, enc, eval, runtime);
    return out;
}

void assign_shared_row(si64Matrix& dst, u64 dst_row, const si64Matrix& src, u64 src_row)
{
    for (u64 col = 0; col < static_cast<u64>(dst.mShares[0].cols()); ++col) {
        dst.mShares[0](dst_row, col) = src.mShares[0](src_row, col);
        dst.mShares[1](dst_row, col) = src.mShares[1](src_row, col);
    }
}

void private_compare_swap_rows(
    si64Matrix& key,
    std::vector<si64Matrix>& payloads,
    u64 lhs_row,
    u64 rhs_row,
    bool ascending,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    auto lhs_key = int_row_slice(key, lhs_row, 1);
    auto rhs_key = int_row_slice(key, rhs_row, 1);
    sbMatrix should_swap;
    if (ascending) {
        cipher_gt(role, lhs_key, rhs_key, should_swap, eval, runtime);
    } else {
        cipher_gt(role, rhs_key, lhs_key, should_swap, eval, runtime);
    }

    auto new_lhs_key = select_si64_by_bool(rhs_key, lhs_key, should_swap, role, enc, eval, runtime);
    auto new_rhs_key = select_si64_by_bool(lhs_key, rhs_key, should_swap, role, enc, eval, runtime);
    assign_shared_row(key, lhs_row, new_lhs_key, 0);
    assign_shared_row(key, rhs_row, new_rhs_key, 0);

    auto payload_flag = bool_repeat_row(should_swap, 0, payloads[static_cast<std::size_t>(lhs_row)].rows());
    auto lhs_payload = payloads[static_cast<std::size_t>(lhs_row)];
    auto rhs_payload = payloads[static_cast<std::size_t>(rhs_row)];
    payloads[static_cast<std::size_t>(lhs_row)] =
        select_si64_by_bool(rhs_payload, lhs_payload, payload_flag, role, enc, eval, runtime);
    payloads[static_cast<std::size_t>(rhs_row)] =
        select_si64_by_bool(lhs_payload, rhs_payload, payload_flag, role, enc, eval, runtime);
}

void private_bitonic_sort_with_payload(
    si64Matrix& key,
    std::vector<si64Matrix>& payloads,
    bool ascending,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const u64 rows = key.rows();
    if (rows == 0) {
        return;
    }
    if ((rows & (rows - 1)) != 0) {
        throw std::runtime_error("private bitonic sort requires power-of-two row count.");
    }
    if (payloads.size() != static_cast<std::size_t>(rows)) {
        throw std::runtime_error("private bitonic sort key/payload row count mismatch.");
    }

    for (u64 k = 2; k <= rows; k <<= 1) {
        for (u64 j = k >> 1; j > 0; j >>= 1) {
            for (u64 i = 0; i < rows; ++i) {
                const u64 ixj = i ^ j;
                if (ixj <= i) {
                    continue;
                }
                const bool local_ascending = ((i & k) == 0) ? ascending : !ascending;
                private_compare_swap_rows(
                    key,
                    payloads,
                    i,
                    ixj,
                    local_ascending,
                    role,
                    enc,
                    eval,
                    runtime);
            }
        }
    }
}

si64Matrix secure_lookup_by_secret_index_sort_join(
    const si64Matrix& values_by_pk,
    const si64Matrix& sample_pk_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    u64 secure_sort_min_size)
{
    (void)secure_sort_min_size;
    const u64 pk_rows = values_by_pk.rows();
    const u64 sample_rows = sample_pk_rows.rows();
    const u64 real_rows = pk_rows + sample_rows;
    const u64 padded_rows = std::max<u64>(1, roundUpToPowerOfTwo(real_rows));

    si64Matrix sort_key(padded_rows, 1);
    sort_key.mShares[0].setZero();
    sort_key.mShares[1].setZero();
    std::vector<si64Matrix> payloads(static_cast<std::size_t>(padded_rows));

    for (u64 row = 0; row < padded_rows; ++row) {
        si64Matrix payload(4, 1);
        payload.mShares[0].setZero();
        payload.mShares[1].setZero();
        payloads[static_cast<std::size_t>(row)] = std::move(payload);
    }

    for (u64 pk = 0; pk < pk_rows; ++pk) {
        set_public_i64_cell(sort_key, pk, static_cast<i64>(pk * 2), role);
        auto& payload = payloads[static_cast<std::size_t>(pk)];
        set_public_i64_cell(payload, 0, 0, role);                         // table_id: pk
        set_public_i64_cell(payload, 1, static_cast<i64>(sample_rows + pk), role);
        payload.mShares[0](2, 0) = values_by_pk.mShares[0](pk, 0);        // lookup value
        payload.mShares[1](2, 0) = values_by_pk.mShares[1](pk, 0);
        set_public_i64_cell(payload, 3, static_cast<i64>(pk), role);       // raw group key
    }

    for (u64 sample = 0; sample < sample_rows; ++sample) {
        const u64 out_row = pk_rows + sample;
        sort_key.mShares[0](out_row, 0) = sample_pk_rows.mShares[0](sample, 0) * 2;
        sort_key.mShares[1](out_row, 0) = sample_pk_rows.mShares[1](sample, 0) * 2;
        if (role == 0) {
            sort_key.mShares[0](out_row, 0) += 1;
        } else if (role == 1) {
            sort_key.mShares[1](out_row, 0) += 1;
        } else if (role != 2) {
            throw std::runtime_error("Invalid ABY3 role.");
        }
        auto& payload = payloads[static_cast<std::size_t>(out_row)];
        set_public_i64_cell(payload, 0, 1, role);                         // table_id: sample
        set_public_i64_cell(payload, 1, static_cast<i64>(sample), role);   // original sample position
        payload.mShares[0](3, 0) = sample_pk_rows.mShares[0](sample, 0);   // raw group key
        payload.mShares[1](3, 0) = sample_pk_rows.mShares[1](sample, 0);
    }

    for (u64 pad = real_rows; pad < padded_rows; ++pad) {
        set_public_i64_cell(sort_key, pad, static_cast<i64>((pk_rows + pad) * 2), role);
        auto& payload = payloads[static_cast<std::size_t>(pad)];
        set_public_i64_cell(payload, 0, 2, role);                         // table_id: pad
        set_public_i64_cell(payload, 1, static_cast<i64>(sample_rows + pad), role);
        set_public_i64_cell(payload, 3, static_cast<i64>(pk_rows + pad), role);
    }

    private_bitonic_sort_with_payload(sort_key, payloads, true, role, enc, eval, runtime);

    auto table_id = payload_column_to_matrix(payloads, 0);
    auto sample_pos = payload_column_to_matrix(payloads, 1);
    auto seed_value = payload_column_to_matrix(payloads, 2);
    auto group_key = payload_column_to_matrix(payloads, 3);
    auto same_previous = adjacent_group_equal_flags(group_key, true, role, eval, runtime);
    auto propagated_value = segmented_prefix_sum(seed_value, same_previous, role, enc, eval, runtime);

    auto sample_table_id = public_i64_column(1, padded_rows, role);
    sbMatrix is_sample;
    cipher_eq(role, table_id, sample_table_id, is_sample, eval, runtime);

    si64Matrix non_sample_sort_key(padded_rows, 1);
    non_sample_sort_key.mShares[0].setZero();
    non_sample_sort_key.mShares[1].setZero();
    for (u64 row = 0; row < padded_rows; ++row) {
        set_public_i64_cell(non_sample_sort_key, row, static_cast<i64>(sample_rows + row), role);
    }
    auto sort_back_key = select_si64_by_bool(
        sample_pos,
        non_sample_sort_key,
        is_sample,
        role,
        enc,
        eval,
        runtime);
    auto back_payloads = column_to_payload_vector(propagated_value);
    private_bitonic_sort_with_payload(sort_back_key, back_payloads, true, role, enc, eval, runtime);

    si64Matrix out(sample_rows, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    for (u64 row = 0; row < sample_rows; ++row) {
        out.mShares[0](row, 0) = back_payloads[static_cast<std::size_t>(row)].mShares[0](0, 0);
        out.mShares[1](row, 0) = back_payloads[static_cast<std::size_t>(row)].mShares[1](0, 0);
    }
    return out;
}

si64Matrix secure_lookup_by_secret_index_private(
    si64Matrix& values_by_pk,
    const si64Matrix& sample_pk_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    u64 secure_sort_min_size)
{
    const auto mode = secure_leaf_lookup_mode();
    if (mode == "equality_scan" || mode == "scan" || mode == "chunked_scan") {
        return secure_lookup_by_secret_index_chunked(values_by_pk, sample_pk_rows, role, enc, eval, runtime);
    }
    if (mode == "oram" || mode == "sqrt_oram" || mode == "sqrt-oram") {
        return secure_lookup_by_secret_index_oram(values_by_pk, sample_pk_rows, role, enc, eval, runtime);
    }
    if (mode != "sort_join" && mode != "sort-join" && mode != "private_lookup") {
        throw std::runtime_error("Unsupported SECURE_LEAF_LOOKUP_MODE: " + mode);
    }
    return secure_lookup_by_secret_index_sort_join(
        values_by_pk,
        sample_pk_rows,
        role,
        enc,
        eval,
        runtime,
        secure_sort_min_size);
}

struct MaterializedLeafResult {
    std::uint64_t leaf_node_id = 0;
    std::uint64_t product_group_id = 0;
    std::uint64_t public_bucket_width = 0;
    std::uint64_t real_bucket_count = 0;
    std::string bucket_mode = "exact";
};

u64 public_leaf_bucket_width_from_plan(const json& plan, u64 total_rows, const std::string& bucket_mode);
void write_secure_leaf_counts_manifest(
    const std::string& output_prefix,
    const std::vector<MaterializedLeafResult>& leaves,
    u64 total_rows);

bool allow_insecure_fast_secure_leaf_materialize()
{
    const char* value = std::getenv("SECURE_ALLOW_INSECURE_LOCAL_MATERIALIZE");
    return value != nullptr && std::string(value) == "1";
}

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

std::vector<i64> reconstruct_plan_arithmetic_vector(
    const std::string& plan_base_dir,
    const std::string& role_share_dir,
    const std::string& filename,
    u64 rows)
{
    return reconstruct_arithmetic_share_pair_files(
        join_path(role_dir_from_template(plan_base_dir, role_share_dir, 0), filename),
        join_path(role_dir_from_template(plan_base_dir, role_share_dir, 1), filename),
        join_path(role_dir_from_template(plan_base_dir, role_share_dir, 2), filename),
        rows);
}

std::vector<i64> reconstruct_plan_boolean_vector(
    const std::string& plan_base_dir,
    const std::string& role_share_dir,
    const std::string& filename,
    u64 rows)
{
    return reconstruct_boolean_share_pair_files(
        join_path(role_dir_from_template(plan_base_dir, role_share_dir, 0), filename),
        join_path(role_dir_from_template(plan_base_dir, role_share_dir, 1), filename),
        join_path(role_dir_from_template(plan_base_dir, role_share_dir, 2), filename),
        rows);
}

std::vector<i64> reconstruct_artifact_multiplier(
    const std::string& artifact_dir,
    const std::string& multiplier_file,
    u64 pk_row_count)
{
    return reconstruct_arithmetic_share_pair_files(
        join_path(join_path(artifact_dir, "role_0"), multiplier_file),
        join_path(join_path(artifact_dir, "role_1"), multiplier_file),
        join_path(join_path(artifact_dir, "role_2"), multiplier_file),
        pk_row_count);
}

void write_fast_materialized_leaf_role_files(
    const std::string& role_dir,
    std::uint64_t leaf_node_id,
    const std::vector<i64>& bucket_values_fixed,
    const std::vector<i64>& flat_bitmap_bits,
    const std::vector<i64>& row_values_fixed,
    int role)
{
    ensure_dir(role_dir);
    const std::string leaf_prefix = "leaf_" + std::to_string(leaf_node_id);
    const auto value_shares = split_arithmetic_replicated_fast(bucket_values_fixed, 5100 + leaf_node_id);
    const auto lower_shares = split_arithmetic_replicated_fast(bucket_values_fixed, 6100 + leaf_node_id);
    const auto upper_shares = split_arithmetic_replicated_fast(bucket_values_fixed, 7100 + leaf_node_id);
    const auto bitmap_shares = split_boolean_replicated_fast(flat_bitmap_bits, 8100 + leaf_node_id);
    const auto row_value_shares = split_arithmetic_replicated_fast(row_values_fixed, 9100 + leaf_node_id);

    write_role_share_pairs(join_path(role_dir, leaf_prefix + ".bucket_values.shares.bin"), value_shares, role);
    write_role_share_pairs(join_path(role_dir, leaf_prefix + ".bucket_lowers.shares.bin"), lower_shares, role);
    write_role_share_pairs(join_path(role_dir, leaf_prefix + ".bucket_uppers.shares.bin"), upper_shares, role);
    write_role_share_pairs(join_path(role_dir, leaf_prefix + ".leaf_bitmaps.shares.bin"), bitmap_shares, role);
    write_role_share_pairs(join_path(role_dir, leaf_prefix + ".row_values.shares.bin"), row_value_shares, role);
}

MaterializedLeafResult materialize_one_secure_leaf_fast(
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& plan_role_share_dir,
    const std::string& output_prefix,
    const std::vector<i64>& global_sample_pk_rows,
    u64 total_rows,
    u64 public_leaf_bucket_width,
    const std::string& bucket_mode,
    int role,
    std::unordered_map<std::string, std::vector<i64>>& multiplier_cache)
{
    const std::uint64_t leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
    const auto product_group_signed = leaf_doc.value("product_group_id", std::int64_t(-1));
    const std::uint64_t product_group_id = static_cast<std::uint64_t>(product_group_signed);
    const std::string artifact_dir = leaf_doc.value("multiplier_artifact_dir", std::string());
    if (artifact_dir.empty()) {
        throw std::runtime_error("Secure leaf plan is missing multiplier_artifact_dir.");
    }
    const auto artifact_manifest = read_json_file(join_path(artifact_dir, "manifest.json"));
    const u64 pk_row_count = artifact_manifest.value("pk_row_count", std::uint64_t(0));
    const std::string kind = leaf_doc.value("multiplier_kind", std::string("mu"));
    const std::string multiplier_file = kind == "mu_nn" ? "mu_nn.shares.bin" : "mu.shares.bin";
    const std::string cache_key = artifact_dir + "\n" + multiplier_file;
    auto cache_iter = multiplier_cache.find(cache_key);
    if (cache_iter == multiplier_cache.end()) {
        cache_iter = multiplier_cache.emplace(
            cache_key,
            reconstruct_artifact_multiplier(artifact_dir, multiplier_file, pk_row_count)).first;
    }
    const auto& multiplier_values = cache_iter->second;

    const std::string membership_file = leaf_doc.value("membership_share_file", std::string());
    const auto membership = reconstruct_plan_boolean_vector(
        plan_base_dir,
        plan_role_share_dir,
        membership_file,
        total_rows);

    std::vector<i64> sample_pk_rows = global_sample_pk_rows;
    if (sample_pk_rows.empty()) {
        const std::string sample_file = leaf_doc.value("sample_pk_rows_share_file", std::string());
        if (sample_file.empty()) {
            throw std::runtime_error("Secure leaf plan entry is missing sample pk rows share file.");
        }
        sample_pk_rows = reconstruct_plan_arithmetic_vector(
            plan_base_dir,
            plan_role_share_dir,
            sample_file,
            total_rows);
    }
    if (sample_pk_rows.size() != static_cast<std::size_t>(total_rows)) {
        throw std::runtime_error("Secure leaf sample pk row vector size mismatch.");
    }

    std::vector<i64> row_values_fixed(static_cast<std::size_t>(total_rows), 0);
    std::vector<i64> bucket_values_fixed;
    std::vector<std::vector<i64>> bucket_bitmaps;
    std::unordered_map<i64, u64> bucket_index_by_value;

    for (u64 row = 0; row < total_rows; ++row) {
        if ((static_cast<std::uint64_t>(membership[static_cast<std::size_t>(row)]) & 1ULL) == 0) {
            continue;
        }
        const i64 pk_row = sample_pk_rows[static_cast<std::size_t>(row)];
        if (pk_row < 0 || static_cast<u64>(pk_row) >= pk_row_count ||
            static_cast<std::size_t>(pk_row) >= multiplier_values.size()) {
            throw std::runtime_error(
                "Secure leaf references sample_pk_row outside multiplier artifact range.");
        }
        const i64 fixed_value = multiplier_values[static_cast<std::size_t>(pk_row)];
        row_values_fixed[static_cast<std::size_t>(row)] = fixed_value;
        auto iter = bucket_index_by_value.find(fixed_value);
        if (iter == bucket_index_by_value.end()) {
            const u64 bucket_idx = static_cast<u64>(bucket_values_fixed.size());
            bucket_index_by_value.emplace(fixed_value, bucket_idx);
            bucket_values_fixed.push_back(fixed_value);
            bucket_bitmaps.emplace_back(static_cast<std::size_t>(total_rows), 0);
            iter = bucket_index_by_value.find(fixed_value);
        }
        bucket_bitmaps[static_cast<std::size_t>(iter->second)][static_cast<std::size_t>(row)] = 1;
    }

    const u64 real_bucket_count = static_cast<u64>(bucket_values_fixed.size());
    u64 leaf_bucket_width = public_leaf_bucket_width;
    if (bucket_mode == "exact") {
        leaf_bucket_width = std::max<u64>(1, real_bucket_count);
    }
    if (real_bucket_count > leaf_bucket_width) {
        throw std::runtime_error("Secure leaf produced more buckets than the public bucket width.");
    }
    while (bucket_values_fixed.size() < static_cast<std::size_t>(leaf_bucket_width)) {
        bucket_values_fixed.push_back(0);
        bucket_bitmaps.emplace_back(static_cast<std::size_t>(total_rows), 0);
    }

    std::vector<i64> flat_bitmap_bits;
    flat_bitmap_bits.reserve(static_cast<std::size_t>(leaf_bucket_width * total_rows));
    for (const auto& bitmap : bucket_bitmaps) {
        flat_bitmap_bits.insert(flat_bitmap_bits.end(), bitmap.begin(), bitmap.end());
    }

    ensure_dir(output_prefix);
    const std::string role_dir = join_path(output_prefix, "role_" + std::to_string(role));
    write_fast_materialized_leaf_role_files(
        role_dir,
        leaf_node_id,
        bucket_values_fixed,
        flat_bitmap_bits,
        row_values_fixed,
        role);

    MaterializedLeafResult result;
    result.leaf_node_id = leaf_node_id;
    result.product_group_id = product_group_id;
    result.public_bucket_width = leaf_bucket_width;
    result.real_bucket_count = real_bucket_count;
    result.bucket_mode = bucket_mode;
    return result;
}

void run_secure_leaf_materialize_fast(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_leaf_materialize mode requires --role in {0,1,2}.");
    }

    const auto plan = read_json_file(config.secure_leaf_plan_path);
    const u64 total_rows = plan.value("total_rows", std::uint64_t(0));
    std::string bucket_mode = plan.value(
        "secure_multiplier_leaf_bucket_mode",
        plan.contains("public_leaf_bucket_width") ? std::string("padded_public") : std::string("exact"));
    if (bucket_mode == "reveal" || bucket_mode == "reveal_count") {
        bucket_mode = "exact";
    } else if (bucket_mode == "public" || bucket_mode == "padded" ||
               bucket_mode == "hide_count" || bucket_mode == "hide_counts") {
        bucket_mode = "padded_public";
    }
    if (bucket_mode != "exact" && bucket_mode != "padded_public" && bucket_mode != "fixed_cap") {
        throw std::runtime_error("Unsupported secure_multiplier_leaf_bucket_mode in secure leaf plan: " + bucket_mode);
    }

    const u64 public_leaf_bucket_width = public_leaf_bucket_width_from_plan(plan, total_rows, bucket_mode);
    const std::string plan_base_dir = dirname_from_path(config.secure_leaf_plan_path);
    const std::string plan_role_share_dir = plan.value("role_share_dir", std::string("secure_leaf_plan/role_{role}"));

    std::vector<i64> global_sample_pk_rows;
    const std::string global_sample_file = plan.value("global_sample_pk_rows_share_file", std::string());
    if (!global_sample_file.empty()) {
        global_sample_pk_rows = reconstruct_plan_arithmetic_vector(
            plan_base_dir,
            plan_role_share_dir,
            global_sample_file,
            total_rows);
    }

    std::unordered_map<std::string, std::vector<i64>> multiplier_cache;
    std::vector<MaterializedLeafResult> results;
    for (const auto& leaf_doc : plan.value("leaves", json::array())) {
        results.push_back(materialize_one_secure_leaf_fast(
            leaf_doc,
            plan_base_dir,
            plan_role_share_dir,
            config.output_prefix,
            global_sample_pk_rows,
            total_rows,
            public_leaf_bucket_width,
            bucket_mode,
            config.role,
            multiplier_cache));
    }
    if (config.role == 0) {
        ensure_dir(config.output_prefix);
        write_secure_leaf_counts_manifest(config.output_prefix, results, total_rows);
    }
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
    sbMatrix& model_overflow)
{
    const std::uint64_t leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
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

    si64Matrix bucket_values(public_leaf_bucket_width, 1);
    bucket_values.mShares[0].setZero();
    bucket_values.mShares[1].setZero();
    si64Matrix row_values(total_rows, 1);
    row_values.mShares[0].setZero();
    row_values.mShares[1].setZero();
    row_values = arith_mul_bool(values, membership, role, enc, eval, runtime);

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

    auto leaf_overflow = bool_reduce_or_matrix(remaining, role, enc, eval, runtime);
    model_overflow = bool_or_matrix(model_overflow, leaf_overflow, role, enc, eval, runtime);

    ensure_dir(output_prefix);
    const std::string role_dir = join_path(output_prefix, role_name);
    ensure_dir(role_dir);
    const std::string leaf_prefix = "leaf_" + std::to_string(leaf_node_id);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_values.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_lowers.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_uppers.shares.bin"), bucket_values);
    write_bool_share_pair_matrix(join_path(role_dir, leaf_prefix + ".leaf_bitmaps.shares.bin"), bucket_bitmaps);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".row_values.shares.bin"), row_values);

    MaterializedLeafResult result;
    result.leaf_node_id = leaf_node_id;
    result.product_group_id = product_group_id;
    result.public_bucket_width = public_leaf_bucket_width;
    result.real_bucket_count = 0;
    result.bucket_mode = "fixed_cap";
    return result;
}

MaterializedLeafResult materialize_one_secure_leaf(
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& output_prefix,
    u64 total_rows,
    u64 public_leaf_bucket_width,
    const std::string& bucket_mode,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const std::uint64_t leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
    const auto product_group_signed = leaf_doc.value("product_group_id", std::int64_t(-1));
    const std::uint64_t product_group_id = static_cast<std::uint64_t>(product_group_signed);
    const std::string artifact_dir = leaf_doc.value("multiplier_artifact_dir", std::string());
    if (artifact_dir.empty()) {
        throw std::runtime_error("Secure leaf plan is missing multiplier_artifact_dir.");
    }
    const auto artifact_manifest = read_json_file(join_path(artifact_dir, "manifest.json"));
    const u64 pk_row_count = artifact_manifest.value("pk_row_count", std::uint64_t(0));
    const std::string kind = leaf_doc.value("multiplier_kind", std::string("mu"));
    const std::string multiplier_file = kind == "mu_nn" ? "mu_nn.shares.bin" : "mu.shares.bin";
    const std::string role_name = "role_" + std::to_string(role);
    const auto multiplier_values = read_share_pair_matrix(
        join_path(join_path(artifact_dir, role_name), multiplier_file),
        pk_row_count);

    const std::string role_share_template = leaf_doc.value("role_share_dir", std::string());
    (void)role_share_template;
    const std::string plan_role_dir = join_path(join_path(plan_base_dir, "secure_leaf_plan"), role_name);
    const auto membership = read_bool_share_pair_matrix(
        join_path(plan_role_dir, leaf_doc.value("membership_share_file", std::string())),
        total_rows,
        1);
    const auto sample_pk_rows = read_share_pair_matrix(
        join_path(plan_role_dir, leaf_doc.value("sample_pk_rows_share_file", std::string())),
        total_rows);
    auto values = secure_lookup_by_secret_index(
        multiplier_values,
        sample_pk_rows,
        role,
        enc,
        eval,
        runtime);

    sbMatrix boundaries(total_rows, 1);
    for (u64 row_i = 0; row_i < total_rows; ++row_i) {
        auto active_i = bool_row_slice(membership, row_i, 1);
        sbMatrix has_prior(1, 1);
        bool_init_false(role, has_prior);
        auto value_i = int_row_slice(values, row_i, 1);
        for (u64 row_j = 0; row_j < row_i; ++row_j) {
            auto value_j = int_row_slice(values, row_j, 1);
            sbMatrix same_value;
            cipher_eq(role, value_i, value_j, same_value, eval, runtime);
            auto active_j = bool_row_slice(membership, row_j, 1);
            auto active_same = bool_and_matrix(same_value, active_j, role, enc, eval, runtime);
            has_prior = bool_or_matrix(has_prior, active_same, role, enc, eval, runtime);
        }
        auto no_prior = bool_not_matrix(has_prior, role);
        auto boundary = bool_and_matrix(active_i, no_prior, role, enc, eval, runtime);
        boundaries.mShares[0](row_i, 0) = boundary.mShares[0](0, 0);
        boundaries.mShares[1](row_i, 0) = boundary.mShares[1](0, 0);
    }

    auto boundary_arith = bool_to_arith_matrix(boundaries, role, enc, eval, runtime);
    si64Matrix boundary_count(1, 1);
    boundary_count.mShares[0](0, 0) = boundary_arith.mShares[0].sum();
    boundary_count.mShares[1](0, 0) = boundary_arith.mShares[1].sum();
    u64 leaf_bucket_width = public_leaf_bucket_width;
    u64 real_bucket_count = 0;
    if (bucket_mode == "exact") {
        i64Matrix plain_boundary_count(1, 1);
        enc.revealAll(runtime, boundary_count, plain_boundary_count).get();
        real_bucket_count = static_cast<u64>(std::max<i64>(0, plain_boundary_count(0, 0)));
        leaf_bucket_width = std::max<u64>(1, real_bucket_count);
    } else if (bucket_mode == "fixed_cap") {
        auto public_cap = public_i64_scalar(static_cast<i64>(leaf_bucket_width), role);
        sbMatrix overflow(1, 1);
        cipher_gt(role, boundary_count, public_cap, overflow, eval, runtime);
        i64Matrix plain_overflow(1, 1);
        enc.revealAll(runtime, overflow, plain_overflow).get();
        if ((plain_overflow(0, 0) & 1) != 0) {
            throw std::runtime_error("multiplier_fixed_cap too small");
        }
    }

    si64Matrix ranks(total_rows, 1);
    ranks.mShares[0].setZero();
    ranks.mShares[1].setZero();
    i64 lhs_running = 0;
    i64 rhs_running = 0;
    for (u64 row = 0; row < total_rows; ++row) {
        lhs_running += boundary_arith.mShares[0](row, 0);
        rhs_running += boundary_arith.mShares[1](row, 0);
        ranks.mShares[0](row, 0) = lhs_running;
        ranks.mShares[1](row, 0) = rhs_running;
    }

    si64Matrix bucket_values(leaf_bucket_width, 1);
    bucket_values.mShares[0].setZero();
    bucket_values.mShares[1].setZero();
    si64Matrix row_values(total_rows, 1);
    row_values.mShares[0].setZero();
    row_values.mShares[1].setZero();
    for (u64 row = 0; row < total_rows; ++row) {
        auto active = bool_row_slice(membership, row, 1);
        auto value_row = int_row_slice(values, row, 1);
        auto selected = arith_mul_bool(value_row, active, role, enc, eval, runtime);
        row_values.mShares[0](row, 0) = selected.mShares[0](0, 0);
        row_values.mShares[1](row, 0) = selected.mShares[1](0, 0);
    }
    sbMatrix bucket_bitmaps(leaf_bucket_width * total_rows, 1);
    for (u64 row = 0; row < bucket_bitmaps.rows(); ++row) {
        bucket_bitmaps.mShares[0](row, 0) = 0;
        bucket_bitmaps.mShares[1](row, 0) = 0;
    }

    for (u64 bucket_idx = 0; bucket_idx < leaf_bucket_width; ++bucket_idx) {
        auto public_rank = public_i64_column(static_cast<i64>(bucket_idx + 1), total_rows, role);
        sbMatrix rank_matches;
        cipher_eq(role, ranks, public_rank, rank_matches, eval, runtime);
        auto representative_rows = bool_and_matrix(rank_matches, boundaries, role, enc, eval, runtime);
        sbMatrix bucket_active(1, 1);
        bool_init_false(role, bucket_active);
        for (u64 row = 0; row < total_rows; ++row) {
            auto representative_row = bool_row_slice(representative_rows, row, 1);
            bucket_active = bool_or_matrix(bucket_active, representative_row, role, enc, eval, runtime);
        }
        auto selected_values = arith_mul_bool(values, representative_rows, role, enc, eval, runtime);
        bucket_values.mShares[0](bucket_idx, 0) = selected_values.mShares[0].sum();
        bucket_values.mShares[1](bucket_idx, 0) = selected_values.mShares[1].sum();

        auto repeated_bucket_value = repeat_shared_row(bucket_values, bucket_idx, total_rows);
        sbMatrix value_matches;
        cipher_eq(role, values, repeated_bucket_value, value_matches, eval, runtime);
        auto active_value_matches = bool_and_matrix(value_matches, membership, role, enc, eval, runtime);
        auto active_bucket_rows = bool_repeat_row(bucket_active, 0, total_rows);
        auto active_bucket_value_matches = bool_and_matrix(
            active_value_matches,
            active_bucket_rows,
            role,
            enc,
            eval,
            runtime);
        const u64 out_begin = bucket_idx * total_rows;
        for (u64 row = 0; row < total_rows; ++row) {
            bucket_bitmaps.mShares[0](out_begin + row, 0) = active_bucket_value_matches.mShares[0](row, 0);
            bucket_bitmaps.mShares[1](out_begin + row, 0) = active_bucket_value_matches.mShares[1](row, 0);
        }
    }

    ensure_dir(output_prefix);
    const std::string role_dir = join_path(output_prefix, role_name);
    ensure_dir(role_dir);
    const std::string leaf_prefix = "leaf_" + std::to_string(leaf_node_id);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_values.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_lowers.shares.bin"), bucket_values);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".bucket_uppers.shares.bin"), bucket_values);
    write_bool_share_pair_matrix(join_path(role_dir, leaf_prefix + ".leaf_bitmaps.shares.bin"), bucket_bitmaps);
    write_share_pair_matrix(join_path(role_dir, leaf_prefix + ".row_values.shares.bin"), row_values);

    MaterializedLeafResult result;
    result.leaf_node_id = leaf_node_id;
    result.product_group_id = product_group_id;
    result.public_bucket_width = leaf_bucket_width;
    result.real_bucket_count = real_bucket_count;
    result.bucket_mode = bucket_mode;
    return result;
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
        content << ", \"row_values_share_file\": \"leaf_" << leaf.leaf_node_id << ".row_values.shares.bin\"";
        content << "}" << (idx + 1 == leaves.size() ? "\n" : ",\n");
    }
    content << "  ],\n";
    content << "  \"role_share_dir\": \"role_{role}\"\n";
    content << "}\n";
    write_text_file(join_path(output_prefix, "secure_leaf_counts.json"), content.str());
}

u64 public_leaf_bucket_width_from_plan(const json& plan, u64 total_rows, const std::string& bucket_mode)
{
    u64 width = plan.value("public_leaf_bucket_width", std::uint64_t(0));
    if (width == 0 && bucket_mode == "fixed_cap") {
        width = plan.value("multiplier_fixed_cap", std::uint64_t(0));
    }
    if (width == 0) {
        for (const auto& leaf_doc : plan.value("leaves", json::array())) {
            const u64 cardinality = leaf_doc.value("cardinality", std::uint64_t(0));
            width = std::max<u64>(width, cardinality);
        }
    }
    width = std::max<u64>(1, width);
    if (bucket_mode != "fixed_cap" && total_rows != 0) {
        width = std::min<u64>(width, total_rows);
    }
    return width;
}

void run_secure_leaf_materialize(const MultiplierPreprocessConfig& config)
{
    if (allow_insecure_fast_secure_leaf_materialize()) {
        run_secure_leaf_materialize_fast(config);
        return;
    }

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
    std::string bucket_mode = plan.value("secure_multiplier_leaf_bucket_mode", std::string("exact"));
    if (bucket_mode == "reveal" || bucket_mode == "reveal_count") {
        bucket_mode = "exact";
    } else if (bucket_mode == "public" || bucket_mode == "padded" ||
               bucket_mode == "hide_count" || bucket_mode == "hide_counts") {
        bucket_mode = "padded_public";
    }
    if (bucket_mode != "exact" && bucket_mode != "padded_public" && bucket_mode != "fixed_cap") {
        throw std::runtime_error("Unsupported secure_multiplier_leaf_bucket_mode in secure leaf plan: " + bucket_mode);
    }
    const u64 public_leaf_bucket_width = public_leaf_bucket_width_from_plan(plan, total_rows, bucket_mode);
    const std::string plan_base_dir = dirname_from_path(config.secure_leaf_plan_path);
    const std::string plan_role_share_dir = plan.value("role_share_dir", std::string("secure_leaf_plan/role_{role}"));
    si64Matrix global_sample_pk_rows;
    bool has_global_sample_pk_rows = false;
    if (bucket_mode == "fixed_cap") {
        const std::string global_sample_file = plan.value("global_sample_pk_rows_share_file", std::string());
        if (!global_sample_file.empty()) {
            const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, config.role);
            global_sample_pk_rows = read_share_pair_matrix(
                join_path(plan_role_dir, global_sample_file),
                total_rows);
            has_global_sample_pk_rows = true;
        }
    }

    std::unordered_map<std::string, si64Matrix> row_value_cache;
    sbMatrix model_overflow = shared_false_bool_matrix(1, 1, config.role);
    std::vector<MaterializedLeafResult> results;
    for (const auto& leaf_doc : plan.value("leaves", json::array())) {
        if (bucket_mode == "fixed_cap") {
            const std::string artifact_dir = leaf_doc.value("multiplier_artifact_dir", std::string());
            const std::string kind = leaf_doc.value("multiplier_kind", std::string("mu"));
            const std::string multiplier_file = kind == "mu_nn" ? "mu_nn.shares.bin" : "mu.shares.bin";
            const std::string sampled_row_values_file = leaf_doc.value("sampled_row_values_share_file", std::string());
            std::string sample_cache_suffix = "global";
            si64Matrix local_sample_pk_rows;
            const si64Matrix* sample_pk_rows_for_lookup = &global_sample_pk_rows;
            const std::string plan_role_dir = role_dir_from_template(plan_base_dir, plan_role_share_dir, config.role);
            if (sampled_row_values_file.empty() && !has_global_sample_pk_rows) {
                const std::string sample_file = leaf_doc.value("sample_pk_rows_share_file", std::string());
                if (sample_file.empty()) {
                    throw std::runtime_error("Secure leaf plan entry is missing sample pk rows share file.");
                }
                local_sample_pk_rows = read_share_pair_matrix(
                    join_path(plan_role_dir, sample_file),
                    total_rows);
                sample_pk_rows_for_lookup = &local_sample_pk_rows;
                sample_cache_suffix = sample_file;
            }
            const std::string cache_key = sampled_row_values_file.empty()
                ? artifact_dir + "\n" + multiplier_file + "\n" + sample_cache_suffix
                : std::string("sampled\n") + sampled_row_values_file;
            auto cache_iter = row_value_cache.find(cache_key);
            if (cache_iter == row_value_cache.end()) {
                si64Matrix row_values;
                if (!sampled_row_values_file.empty()) {
                    row_values = read_share_pair_matrix(
                        join_path(plan_role_dir, sampled_row_values_file),
                        total_rows);
                } else {
                    if (artifact_dir.empty()) {
                        throw std::runtime_error("Secure leaf plan is missing multiplier_artifact_dir.");
                    }
                    const auto artifact_manifest = read_json_file(join_path(artifact_dir, "manifest.json"));
                    const u64 pk_row_count = artifact_manifest.value("pk_row_count", std::uint64_t(0));
                    const std::string role_name = "role_" + std::to_string(config.role);
                    auto multiplier_values = read_share_pair_matrix(
                        join_path(join_path(artifact_dir, role_name), multiplier_file),
                        pk_row_count);
                    row_values = secure_lookup_by_secret_index_private(
                        multiplier_values,
                        *sample_pk_rows_for_lookup,
                        config.role,
                        enc,
                        eval,
                        runtime,
                        config.secure_sort_min_size);
                }
                cache_iter = row_value_cache.emplace(cache_key, std::move(row_values)).first;
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
                model_overflow));
        } else {
            results.push_back(materialize_one_secure_leaf(
                leaf_doc,
                plan_base_dir,
                config.output_prefix,
                total_rows,
                public_leaf_bucket_width,
                bucket_mode,
                config.role,
                enc,
                eval,
                runtime));
        }
    }
    if (bucket_mode == "fixed_cap") {
        i64Matrix plain_overflow(1, 1);
        enc.revealAll(runtime, model_overflow, plain_overflow).get();
        if ((plain_overflow(0, 0) & 1) != 0) {
            throw std::runtime_error("multiplier_fixed_cap too small");
        }
    }
    if (config.role == 0) {
        ensure_dir(config.output_prefix);
        write_secure_leaf_counts_manifest(config.output_prefix, results, total_rows);
    }
}

}  // namespace

int BSPN_multiplier_preprocess(const CLP& cmd)
{
    const auto config = parse_config(cmd);

    if (config.mode == "reference") {
        run_reference_multiplier_preprocess(config);
        return 0;
    }

    if (config.mode == "secure_scaffold") {
        run_secure_multiplier_scaffold(config);
        return 0;
    }

    if (config.mode == "secure_shared_values") {
        run_secure_multiplier_shared_values(config);
        return 0;
    }

    if (config.mode == "secure_leaf_materialize") {
        run_secure_leaf_materialize(config);
        return 0;
    }

    throw std::runtime_error(
        "Unsupported --multiplier_mode. Use reference, secure_scaffold, secure_shared_values, or secure_leaf_materialize.");
}

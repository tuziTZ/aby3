#include "MultiplierPreprocessing.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "../aby3-Basic/BuildingBlocks.h"
#include "../aby3-GORAM-Core/Basics.h"
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

std::vector<double> reveal_shared_int_column_as_f64(
    si64Matrix& shared,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime)
{
    i64Matrix plain(shared.rows(), shared.cols());
    enc.revealAll(runtime, shared, plain).get();

    std::vector<double> values(static_cast<size_t>(plain.rows()), 0.0);
    for (u64 row = 0; row < plain.rows(); ++row) {
        values[static_cast<size_t>(row)] = static_cast<double>(plain(row, 0));
    }
    return values;
}

i64Matrix reveal_shared_int_matrix(
    si64Matrix& shared,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime)
{
    i64Matrix plain(shared.rows(), shared.cols());
    enc.revealAll(runtime, shared, plain).get();
    return plain;
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

void write_revealed_manifest(
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
    content << "  \"mode\": \"secure_reveal\",\n";
    content << "  \"relationship_id\": \"" << json_escape(config.relationship_id) << "\",\n";
    content << "  \"pk_csv_path\": \"" << json_escape(config.pk_csv_path) << "\",\n";
    content << "  \"fk_csv_path\": \"" << json_escape(config.fk_csv_path) << "\",\n";
    content << "  \"pk_key_column\": " << config.pk_key_column << ",\n";
    content << "  \"fk_key_column\": " << config.fk_key_column << ",\n";
    content << "  \"fk_sample_rate\": " << std::setprecision(17) << config.fk_sample_rate << ",\n";
    content << "  \"pk_row_count\": " << pk_row_count << ",\n";
    content << "  \"fk_row_count\": " << fk_row_count << ",\n";
    content << "  \"pk_input_party\": " << config.pk_input_party << ",\n";
    content << "  \"fk_input_party\": " << config.fk_input_party << ",\n";
    content << "  \"approved_helpers\": [\n";
    content << "    \"basic_setup\",\n";
    content << "    \"localIntMatrix/remoteIntMatrix\",\n";
    content << "    \"quick_sort_with_other_elements\",\n";
    content << "    \"revealAll\"\n";
    content << "  ],\n";
    content << "  \"secure_core_status\": \"sorted_combined_rows_revealed_for_grouped_count\",\n";
    content << "  \"mu_dtype\": \"float64\",\n";
    content << "  \"mu_nn_dtype\": \"float64\",\n";
    content << "  \"mu_file\": \"" << json_escape(mu_path) << "\",\n";
    content << "  \"mu_nn_file\": \"" << json_escape(mu_nn_path) << "\"\n";
    content << "}\n";
    write_text_file(manifest_path, content.str());
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
            throw std::runtime_error("secure_reveal sort key overflowed int64 range.");
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

void run_secure_multiplier_shared_values(const MultiplierPreprocessConfig& config)
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

void run_secure_multiplier_reveal(const MultiplierPreprocessConfig& config)
{
    if (config.role < 0 || config.role > 2) {
        throw std::runtime_error("secure_reveal mode requires --role in {0,1,2}.");
    }
    if (config.pk_input_party < 0 || config.pk_input_party > 2) {
        throw std::runtime_error("secure_reveal mode requires --pk_input_party in {0,1,2}.");
    }
    if (config.fk_input_party < 0 || config.fk_input_party > 2) {
        throw std::runtime_error("secure_reveal mode requires --fk_input_party in {0,1,2}.");
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
        sorted_payloads);

    auto plain_sorted_key = reveal_shared_int_matrix(sec_key, enc, runtime);
    si64Matrix sorted_payload_matrix(padded * 5, 1);
    for (u64 row = 0; row < padded; ++row) {
        const auto& payload = sorted_payloads[static_cast<size_t>(row)];
        sorted_payload_matrix.mShares[0].block(row * 5, 0, 5, 1) = payload.mShares[0];
        sorted_payload_matrix.mShares[1].block(row * 5, 0, 5, 1) = payload.mShares[1];
    }
    auto plain_sorted_payload = reveal_shared_int_matrix(sorted_payload_matrix, enc, runtime);

    std::vector<double> revealed_counts(static_cast<size_t>(n_pk), 0.0);
    const i64 stride_i64 = static_cast<i64>(std::max<u64>(1, padded));
    for (u64 group_begin = 0; group_begin < padded;) {
        const i64 key_bucket = plain_sorted_key(group_begin, 0) / stride_i64;
        u64 group_end = group_begin + 1;
        while (group_end < padded && plain_sorted_key(group_end, 0) / stride_i64 == key_bucket) {
            ++group_end;
        }

        i64 fk_count = 0;
        for (u64 row = group_begin; row < group_end; ++row) {
            const i64 table_id = plain_sorted_payload(row * 5 + 0, 0);
            const i64 valid = plain_sorted_payload(row * 5 + 3, 0);
            const i64 fk_contrib = plain_sorted_payload(row * 5 + 4, 0);
            if (valid != 0 && table_id == 1) {
                fk_count += fk_contrib;
            }
        }

        for (u64 row = group_begin; row < group_end; ++row) {
            const i64 table_id = plain_sorted_payload(row * 5 + 0, 0);
            const i64 row_id = plain_sorted_payload(row * 5 + 1, 0);
            const i64 is_null = plain_sorted_payload(row * 5 + 2, 0);
            const i64 valid = plain_sorted_payload(row * 5 + 3, 0);
            const bool is_pk_row = valid != 0 && table_id == 0;
            const bool is_non_null = is_null == 0;
            if (!is_pk_row || !is_non_null) {
                continue;
            }
            if (row_id < 0 || static_cast<u64>(row_id) >= n_pk) {
                throw std::runtime_error("Sorted PK row id out of range during secure_reveal multiplier scan.");
            }
            revealed_counts[static_cast<size_t>(row_id)] = static_cast<double>(fk_count);
        }

        group_begin = group_end;
    }

    std::vector<double> mu(revealed_counts.size(), 0.0);
    std::vector<double> mu_nn(revealed_counts.size(), 1.0);
    const double inv_rate = 1.0 / config.fk_sample_rate;
    for (size_t i = 0; i < revealed_counts.size(); ++i) {
        mu[i] = revealed_counts[i] * inv_rate;
        mu_nn[i] = (mu[i] == 0.0) ? 1.0 : mu[i];
    }

    if (config.role == 0) {
        const std::string mu_path = config.output_prefix + ".mu.bin";
        const std::string mu_nn_path = config.output_prefix + ".mu_nn.bin";
        const std::string manifest_path = config.output_prefix + ".manifest.json";
        write_binary_vector(mu_path, mu);
        write_binary_vector(mu_nn_path, mu_nn);
        write_revealed_manifest(config, static_cast<size_t>(n_pk), static_cast<size_t>(n_fk), manifest_path, mu_path, mu_nn_path);
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

struct MaterializedLeafResult {
    std::uint64_t leaf_node_id = 0;
    std::uint64_t product_group_id = 0;
    std::uint64_t bucket_count = 0;
};

MaterializedLeafResult materialize_one_secure_leaf(
    const json& leaf_doc,
    const std::string& plan_base_dir,
    const std::string& output_prefix,
    u64 total_rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime)
{
    const std::uint64_t leaf_node_id = leaf_doc.value("leaf_node_id", std::uint64_t(0));
    const std::uint64_t product_group_id = leaf_doc.value("product_group_id", std::uint64_t(0));
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
    si64Matrix bucket_count_shared(1, 1);
    bucket_count_shared.mShares[0].setZero();
    bucket_count_shared.mShares[1].setZero();
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
    bucket_count_shared.mShares[0](0, 0) = lhs_running;
    bucket_count_shared.mShares[1](0, 0) = rhs_running;
    i64Matrix bucket_count_plain(1, 1);
    enc.revealAll(runtime, bucket_count_shared, bucket_count_plain).get();
    const i64 revealed_count = bucket_count_plain(0, 0);
    if (revealed_count < 0 || static_cast<u64>(revealed_count) > total_rows) {
        throw std::runtime_error("Materialized secure leaf produced invalid bucket count.");
    }
    const u64 bucket_count = static_cast<u64>(revealed_count);

    si64Matrix bucket_values(bucket_count, 1);
    bucket_values.mShares[0].setZero();
    bucket_values.mShares[1].setZero();
    sbMatrix bucket_bitmaps(bucket_count * total_rows, 1);
    for (u64 row = 0; row < bucket_bitmaps.rows(); ++row) {
        bucket_bitmaps.mShares[0](row, 0) = 0;
        bucket_bitmaps.mShares[1](row, 0) = 0;
    }

    for (u64 bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
        auto public_rank = public_i64_column(static_cast<i64>(bucket_idx + 1), total_rows, role);
        sbMatrix rank_matches;
        cipher_eq(role, ranks, public_rank, rank_matches, eval, runtime);
        auto representative_rows = bool_and_matrix(rank_matches, boundaries, role, enc, eval, runtime);
        auto selected_values = arith_mul_bool(values, representative_rows, role, enc, eval, runtime);
        bucket_values.mShares[0](bucket_idx, 0) = selected_values.mShares[0].sum();
        bucket_values.mShares[1](bucket_idx, 0) = selected_values.mShares[1].sum();

        auto repeated_bucket_value = repeat_shared_row(bucket_values, bucket_idx, total_rows);
        sbMatrix value_matches;
        cipher_eq(role, values, repeated_bucket_value, value_matches, eval, runtime);
        auto active_value_matches = bool_and_matrix(value_matches, membership, role, enc, eval, runtime);
        const u64 out_begin = bucket_idx * total_rows;
        for (u64 row = 0; row < total_rows; ++row) {
            bucket_bitmaps.mShares[0](out_begin + row, 0) = active_value_matches.mShares[0](row, 0);
            bucket_bitmaps.mShares[1](out_begin + row, 0) = active_value_matches.mShares[1](row, 0);
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

    MaterializedLeafResult result;
    result.leaf_node_id = leaf_node_id;
    result.product_group_id = product_group_id;
    result.bucket_count = bucket_count;
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
    content << "  \"format_version\": 1,\n";
    content << "  \"total_rows\": " << total_rows << ",\n";
    content << "  \"leaves\": [\n";
    for (std::size_t idx = 0; idx < leaves.size(); ++idx) {
        const auto& leaf = leaves[idx];
        content << "    {";
        content << "\"leaf_node_id\": " << leaf.leaf_node_id << ", ";
        content << "\"product_group_id\": " << leaf.product_group_id << ", ";
        content << "\"real_bucket_count\": " << leaf.bucket_count << ", ";
        content << "\"bucket_values_share_file\": \"leaf_" << leaf.leaf_node_id << ".bucket_values.shares.bin\", ";
        content << "\"bucket_lowers_share_file\": \"leaf_" << leaf.leaf_node_id << ".bucket_lowers.shares.bin\", ";
        content << "\"bucket_uppers_share_file\": \"leaf_" << leaf.leaf_node_id << ".bucket_uppers.shares.bin\", ";
        content << "\"leaf_bitmaps_share_file\": \"leaf_" << leaf.leaf_node_id << ".leaf_bitmaps.shares.bin\"";
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
    const std::string plan_base_dir = dirname_from_path(config.secure_leaf_plan_path);
    std::vector<MaterializedLeafResult> results;
    for (const auto& leaf_doc : plan.value("leaves", json::array())) {
        results.push_back(materialize_one_secure_leaf(
            leaf_doc,
            plan_base_dir,
            config.output_prefix,
            total_rows,
            config.role,
            enc,
            eval,
            runtime));
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

    if (config.mode == "secure_reveal") {
        run_secure_multiplier_reveal(config);
        return 0;
    }

    if (config.mode == "secure_leaf_materialize") {
        run_secure_leaf_materialize(config);
        return 0;
    }

    throw std::runtime_error(
        "Unsupported --multiplier_mode. Use reference, secure_scaffold, secure_shared_values, secure_leaf_materialize, or secure_reveal.");
}

#include "MultiplierPreprocessing.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../aby3-Basic/BuildingBlocks.h"
#include "../aby3-GORAM-Core/Basics.h"
#include "../aby3-GORAM-Core/Sort.h"

using namespace aby3;
using namespace oc;

namespace {

struct NormalizedKey {
    i64 value = 0;
    bool is_null = true;
};

struct MultiplierPreprocessConfig {
    std::string pk_csv_path;
    std::string fk_csv_path;
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
    u64 secure_sort_min_size = 32;
};

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

    config.pk_csv_path = read_required_string("pk_csv");
    config.fk_csv_path = read_required_string("fk_csv");
    config.output_prefix = read_required_string("output_prefix");
    config.relationship_id = cmd.isSet("relationship_id")
        ? cmd.getMany<std::string>("relationship_id")[0]
        : "unknown_relationship";

    config.pk_key_column = read_optional_u64("pk_key_column", 0);
    config.fk_key_column = read_optional_u64("fk_key_column", 0);

    if (cmd.isSet("fk_sample_rate")) {
        config.fk_sample_rate = cmd.getMany<double>("fk_sample_rate")[0];
    }

    if (cmd.isSet("multiplier_mode")) {
        config.mode = cmd.getMany<std::string>("multiplier_mode")[0];
    }

    if (cmd.isSet("role")) {
        config.role = cmd.getMany<int>("role")[0];
    }

    if (cmd.isSet("input_party")) {
        config.input_party = cmd.getMany<int>("input_party")[0];
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

    throw std::runtime_error("Unsupported --multiplier_mode. Use reference or secure_scaffold.");
}

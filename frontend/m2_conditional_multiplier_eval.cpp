#include "m2_conditional_multiplier_eval.h"

#include "aby3-Basic/Basics.h"
#include "aby3-Basic/BuildingBlocks.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace aby3;
using namespace oc;

namespace {

constexpr i64 kFixedOne = static_cast<i64>(1) << 16;
constexpr const char* kFixedParentPayloadMode = "fixed_parent_payload_v1";
constexpr const char* kSampleMarkerStoredReverseMode = "sample_marker_stored_reverse_v1";
constexpr const char* kSampleMarkerCapReorderMode = "sample_marker_partition_cap_reorder_v1";
constexpr const char* kConditionalDensePayloadMode = "fixed_dictionary_dense_counts_v1";

void require_bool(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

json read_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("missing JSON file: " + path);
    }
    json value;
    in >> value;
    return value;
}

std::vector<unsigned char> read_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("missing binary payload: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot stat binary payload: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (static_cast<std::size_t>(in.gcount()) != bytes.size()) {
        throw std::runtime_error("short read for binary payload: " + path);
    }
    return bytes;
}

template <typename T>
T read_plain(const std::vector<unsigned char>& bytes, std::size_t& offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("binary payload truncated");
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

bool path_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

std::string dirname_of(const std::string& path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return ".";
    }
    return path.substr(0, pos);
}

std::string join_path(const std::string& left, const std::string& right) {
    if (left.empty() || left == ".") {
        return right;
    }
    if (left.back() == '/') {
        return left + right;
    }
    return left + "/" + right;
}

std::string role_path(const std::string& path, int role) {
    std::string out = path;
    const std::string token = "{role}";
    std::size_t pos = 0;
    while ((pos = out.find(token, pos)) != std::string::npos) {
        out.replace(pos, token.size(), std::to_string(role));
        pos += std::to_string(role).size();
    }
    return out;
}

si64Matrix load_share_pair_vector(const std::string& path, std::size_t rows) {
    const auto bytes = read_binary(path);
    require_bool(bytes.size() == rows * sizeof(i64) * 2, "bad share-pair vector size: " + path);
    si64Matrix out(rows, 1);
    std::size_t offset = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        out.mShares[0](static_cast<u64>(row), 0) = read_plain<i64>(bytes, offset);
        out.mShares[1](static_cast<u64>(row), 0) = read_plain<i64>(bytes, offset);
    }
    return out;
}

std::size_t share_pair_rows(const std::string& path) {
    const auto bytes = read_binary(path);
    require_bool(bytes.size() % (sizeof(i64) * 2) == 0, "bad share-pair file size: " + path);
    return bytes.size() / (sizeof(i64) * 2);
}

std::vector<std::uint64_t> load_public_sample_mapping(const std::string& path) {
    const auto doc = read_json_file(path);
    const auto& arr = doc.is_array() ? doc : doc.at("sample_row_id_by_position");
    std::vector<std::uint64_t> out;
    for (const auto& item : arr) {
        out.push_back(item.get<std::uint64_t>());
    }
    require_bool(!out.empty(), "sample mapping is empty");
    return out;
}

si64Matrix gather_public_rows(const si64Matrix& parent_values, const std::vector<std::uint64_t>& mapping) {
    si64Matrix out(mapping.size(), 1);
    for (std::size_t row = 0; row < mapping.size(); ++row) {
        const auto src = static_cast<u64>(mapping[row]);
        require_bool(src < parent_values.rows(), "sample mapping row id outside parent value shares");
        out.mShares[0](static_cast<u64>(row), 0) = parent_values.mShares[0](src, 0);
        out.mShares[1](static_cast<u64>(row), 0) = parent_values.mShares[1](src, 0);
    }
    return out;
}

si64Matrix repeat_secret_scalar(const si64Matrix& scalar, std::size_t rows) {
    require_bool(scalar.rows() == 1, "expected one-row secret scalar");
    si64Matrix out(rows, 1);
    for (std::size_t row = 0; row < rows; ++row) {
        out.mShares[0](static_cast<u64>(row), 0) = scalar.mShares[0](0, 0);
        out.mShares[1](static_cast<u64>(row), 0) = scalar.mShares[1](0, 0);
    }
    return out;
}

si64Matrix share_public_i64_column(const std::vector<i64>& values, int role, Sh3Encryptor& enc, Sh3Runtime& runtime) {
    i64Matrix plain(values.size(), 1);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        plain(static_cast<u64>(idx), 0) = values[idx];
    }
    si64Matrix shared;
    plain_mat_to_cipher_mat(role, enc, runtime, plain, shared);
    return shared;
}

si64Matrix bool_to_arith(sbMatrix value, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime) {
    si64Matrix out(value.rows(), 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    bool2arith(role, value, out, enc, eval, runtime);
    return out;
}

void write_share_pair_vector(const std::string& path, const si64Matrix& value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write share file: " + path);
    }
    for (u64 row = 0; row < value.rows(); ++row) {
        const i64 lhs = value.mShares[0](row, 0);
        const i64 rhs = value.mShares[1](row, 0);
        out.write(reinterpret_cast<const char*>(&lhs), sizeof(lhs));
        out.write(reinterpret_cast<const char*>(&rhs), sizeof(rhs));
    }
}

json transport_stats(Sh3Runtime& runtime) {
    const auto next_sent = runtime.mComm.mNext.getTotalDataSent();
    const auto prev_sent = runtime.mComm.mPrev.getTotalDataSent();
    const auto next_recv = runtime.mComm.mNext.getTotalDataRecv();
    const auto prev_recv = runtime.mComm.mPrev.getTotalDataRecv();
    return json{
        {"counter_source", "cryptoTools::Channel"},
        {"bytes_sent_next", next_sent},
        {"bytes_sent_prev", prev_sent},
        {"bytes_sent_total", next_sent + prev_sent},
        {"bytes_received_next", next_recv},
        {"bytes_received_prev", prev_recv},
        {"bytes_received_total", next_recv + prev_recv}
    };
}

sf64Matrix<D16> as_fixed(const si64Matrix& value) {
    sf64Matrix<D16> out(value.rows(), 1);
    out.i64Cast().mShares[0] = value.mShares[0];
    out.i64Cast().mShares[1] = value.mShares[1];
    return out;
}

si64Matrix multiply_fixed(
    const si64Matrix& lhs,
    const si64Matrix& rhs,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime) {
    auto left = as_fixed(lhs);
    auto right = as_fixed(rhs);
    sf64Matrix<D16> product(lhs.rows(), 1);
    eval.asyncMul(runtime, left, right, product).get();
    si64Matrix out(lhs.rows(), 1);
    out.mShares[0] = product.i64Cast().mShares[0];
    out.mShares[1] = product.i64Cast().mShares[1];
    return out;
}

si64Matrix sum_rows(const si64Matrix& value) {
    si64Matrix out(1, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    for (u64 row = 0; row < value.rows(); ++row) {
        out.mShares[0](0, 0) += value.mShares[0](row, 0);
        out.mShares[1](0, 0) += value.mShares[1](row, 0);
    }
    return out;
}

si64Matrix slice_si64_rows(const si64Matrix& value, u64 offset, u64 rows) {
    if (offset + rows > value.rows()) {
        throw std::runtime_error("secret row slice out of bounds");
    }
    si64Matrix out(rows, 1);
    out.mShares[0] = value.mShares[0].block(offset, 0, rows, 1);
    out.mShares[1] = value.mShares[1].block(offset, 0, rows, 1);
    return out;
}

si64Matrix load_or_public_ones(
    const json& query,
    const std::string& key,
    std::size_t rows,
    int role,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime) {
    const std::string path = query.value(key, std::string());
    if (!path.empty()) {
        return load_share_pair_vector(role_path(path, role), rows);
    }
    return share_public_i64_column(std::vector<i64>(rows, kFixedOne), role, enc, runtime);
}

void require_manifest(
    const json& manifest,
    const std::string& edge_id,
    const std::string& column_id,
    std::uint64_t cap_l,
    const std::string& mode) {
    if (mode == kFixedParentPayloadMode) {
        require_bool(manifest.value("target", std::string()) == "M2_CONDITIONAL_MULTIPLIER_FIXED_PAYLOAD_V1", "bad conditional target");
        require_bool(manifest.value("schema_version", std::string()) == "M2CMFIXED01", "bad conditional schema");
    } else if (mode == kSampleMarkerStoredReverseMode || mode == kSampleMarkerCapReorderMode) {
        const bool dense_payload =
            manifest.value("conditional_payload_mode", std::string()) == kConditionalDensePayloadMode;
        if (mode == kSampleMarkerCapReorderMode) {
            require_bool(manifest.value("target", std::string()) == "M2_CONDITIONAL_MULTIPLIER_SAMPLE_MARKER_PARTITION_CAP_REORDER_V1", "bad conditional target");
            require_bool(
                manifest.value("schema_version", std::string()) ==
                    (dense_payload ? "M2CMSMALLREORDERDENSE01" : "M2CMSMALLREORDER01"),
                "bad conditional schema");
        } else {
            require_bool(manifest.value("target", std::string()) == "M2_CONDITIONAL_MULTIPLIER_SAMPLE_MARKER_STORED_REVERSE_V1", "bad conditional target");
            require_bool(
                manifest.value("schema_version", std::string()) ==
                    (dense_payload ? "M2CMSMALLREORDERDENSE01" : "M2CMSAMPLE01"),
                "bad conditional schema");
        }
        require_bool(!manifest.value("contains_real_row_id", true), "sample-marker artifact exposes real row id");
        require_bool(!manifest.value("contains_join_key", true), "sample-marker artifact exposes join key");
        require_bool(!manifest.value("contains_dense_offset", true), "sample-marker artifact exposes dense offset");
        require_bool(!manifest.value("contains_sample_row_id_by_position", true), "sample-marker artifact exposes sample_row_id_by_position");
    } else {
        throw std::runtime_error("bad conditional mode");
    }
    require_bool(manifest.value("conditional_multiplier_mode", std::string()) == mode, "bad conditional mode");
    require_bool(manifest.value("edge_id", std::string()) == edge_id, "conditional edge mismatch");
    require_bool(manifest.value("column_id", std::string()) == column_id, "conditional column mismatch");
    require_bool(manifest.value("public_cap_L", std::uint64_t(0)) == cap_l, "conditional cap L mismatch");
    require_bool(manifest.value("artifact_state", std::string()) == "complete", "conditional artifact is not complete");
    require_bool(manifest.value("selected_overflow_authorized", false), "conditional selected overflow not authorized");
}

}  // namespace

int M2_conditional_multiplier_eval(const oc::CLP& cmd) {
    const int role = cmd.getOr<int>("role", -1);
    if (role < 0 || role > 2) {
        throw std::runtime_error("m2_conditional_multiplier_eval requires --role in {0,1,2}");
    }
    const std::string mode = cmd.getOr<std::string>("conditional_multiplier_mode", "off");
    require_bool(
        mode == "off" || mode == kFixedParentPayloadMode || mode == kSampleMarkerStoredReverseMode ||
            mode == kSampleMarkerCapReorderMode,
        "unsupported conditional multiplier eval mode");
    const std::string mapping_path = cmd.getOr<std::string>("sample_mapping_json", "");
    const std::string scalar_artifact_dir = cmd.getOr<std::string>("scalar_artifact_dir", "");
    const std::string query_path = cmd.getOr<std::string>("conditional_query_json", "");
    const std::string output_path = cmd.getOr<std::string>("conditional_output_share_bin", "");
    require_bool(
        mode == kSampleMarkerStoredReverseMode || mode == kSampleMarkerCapReorderMode || !mapping_path.empty(),
        "missing --sample_mapping_json");
    require_bool(!scalar_artifact_dir.empty(), "missing --scalar_artifact_dir");
    require_bool(!query_path.empty(), "missing --conditional_query_json");
    require_bool(!output_path.empty(), "missing --conditional_output_share_bin");

    const auto started = std::chrono::steady_clock::now();
    std::vector<std::uint64_t> mapping;
    if (mode != kSampleMarkerStoredReverseMode && mode != kSampleMarkerCapReorderMode) {
        mapping = load_public_sample_mapping(mapping_path);
    }
    const auto query = read_json_file(query_path);
    const std::string operation = query.value("operation", std::string("COUNT"));
    require_bool(operation == "COUNT" || operation == "SUM" || operation == "AVG", "unsupported operation");
    const std::string edge_id = query.value("edge_id", std::string());
    const std::string column_id = query.value("column_id", std::string());
    const std::uint64_t cap_l = query.value("public_cap_L", std::uint64_t(0));
    require_bool(
        (edge_id == "movie_companies" && column_id == "company_type_id" && cap_l == 2) ||
        (edge_id == "movie_info_idx" && column_id == "info_type_id" && cap_l == 4) ||
        (edge_id == "movie_info" && column_id == "info_type_id" && cap_l == 16),
        "unsupported conditional edge/column/cap");

    const std::string role_dir = join_path(scalar_artifact_dir, "role_" + std::to_string(role));
    const auto parent_mu = load_share_pair_vector(join_path(role_dir, "mu.shares.bin"), share_pair_rows(join_path(role_dir, "mu.shares.bin")));
    if (mode == kSampleMarkerStoredReverseMode || mode == kSampleMarkerCapReorderMode) {
        mapping.resize(static_cast<std::size_t>(parent_mu.rows()));
        for (std::size_t idx = 0; idx < mapping.size(); ++idx) {
            mapping[idx] = idx;
        }
    }
    const auto baseline_factor = (mode == kSampleMarkerStoredReverseMode)
        ? parent_mu
        : gather_public_rows(parent_mu, mapping);

    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(role), ios, enc, eval, runtime);
    runtime.mComm.mNext.resetStats();
    runtime.mComm.mPrev.resetStats();

    si64Matrix factor(mapping.size(), 1);
    std::uint64_t equality_count = 0;
    std::uint64_t multiplication_count = 0;
    std::uint64_t conversion_count = 0;
    std::uint64_t mux_count = 0;

    if (mode == "off") {
        factor = baseline_factor;
    } else {
        const std::string manifest_path = cmd.getOr<std::string>("conditional_manifest_json", "");
        require_bool(!manifest_path.empty(), "missing --conditional_manifest_json");
        const std::string sidecar_dir = dirname_of(manifest_path);
        require_bool(path_exists(join_path(sidecar_dir, "COMPLETE")), "conditional artifact missing COMPLETE marker");
        const auto manifest = read_json_file(manifest_path);
        require_manifest(manifest, edge_id, column_id, cap_l, mode);
        if (mode == kSampleMarkerStoredReverseMode || mode == kSampleMarkerCapReorderMode) {
            require_bool(
                manifest.value("sample_rows", std::uint64_t(0)) == static_cast<std::uint64_t>(mapping.size()),
                "sample-marker sample_rows mismatch");
        }
        const std::string literal_path = query.value("literal_share_bin", std::string());
        const std::string use_cm_path = query.value("use_cm_share_bin", std::string());
        require_bool(!literal_path.empty(), "missing secret literal share file");
        require_bool(!use_cm_path.empty(), "missing secret use-cm share file");
        const auto literal = load_share_pair_vector(role_path(literal_path, role), 1);
        const auto use_cm = load_share_pair_vector(role_path(use_cm_path, role), 1);
        auto use_cm_vec = repeat_secret_scalar(use_cm, mapping.size());
        const bool fixed_dictionary_fallback_on_miss =
            query.value("fixed_dictionary_fallback_on_miss", false);
        const std::string sidecar_role_dir = join_path(sidecar_dir, "role_" + std::to_string(role));
        const bool dense_payload =
            manifest.value("conditional_payload_mode", std::string()) == kConditionalDensePayloadMode;

        si64Matrix conditional_count(mapping.size(), 1);
        conditional_count.mShares[0].setZero();
        conditional_count.mShares[1].setZero();
        si64Matrix matched_any(mapping.size(), 1);
        matched_any.mShares[0].setZero();
        matched_any.mShares[1].setZero();
        if (dense_payload) {
            auto dictionary = load_share_pair_vector(join_path(sidecar_role_dir, "dictionary.shares.bin"), cap_l);
            const std::size_t flat_rows = share_pair_rows(join_path(sidecar_role_dir, "dense_count.shares.bin"));
            require_bool(flat_rows % cap_l == 0, "conditional dense count row count is not divisible by L");
            const std::size_t parent_rows = flat_rows / static_cast<std::size_t>(cap_l);
            const auto slot_count_parent = load_share_pair_vector(join_path(sidecar_role_dir, "dense_count.shares.bin"), flat_rows);
            (void)load_share_pair_vector(join_path(sidecar_role_dir, "selected_overflow.shares.bin"), parent_rows);

            auto literal_dict_vec = repeat_secret_scalar(literal, cap_l);
            sbMatrix dict_eq;
            cipher_eq(role, dictionary, literal_dict_vec, dict_eq, eval, runtime);
            auto dict_match = bool_to_arith(dict_eq, role, enc, eval, runtime);
            si64Matrix matched_any_scalar(1, 1);
            matched_any_scalar.mShares[0].setZero();
            matched_any_scalar.mShares[1].setZero();
            for (std::uint64_t slot = 0; slot < cap_l; ++slot) {
                matched_any_scalar.mShares[0](0, 0) += dict_match.mShares[0](slot, 0);
                matched_any_scalar.mShares[1](0, 0) += dict_match.mShares[1](slot, 0);
            }
            matched_any = repeat_secret_scalar(matched_any_scalar, mapping.size());
            equality_count += cap_l;
            conversion_count += cap_l;

            for (std::uint64_t slot = 0; slot < cap_l; ++slot) {
                si64Matrix count(mapping.size(), 1);
                for (std::size_t row = 0; row < mapping.size(); ++row) {
                    const auto parent = mapping[row];
                    require_bool(parent < parent_rows, "sample mapping row id outside conditional sidecar");
                    const auto src = static_cast<u64>(parent * cap_l + slot);
                    count.mShares[0](static_cast<u64>(row), 0) = slot_count_parent.mShares[0](src, 0);
                    count.mShares[1](static_cast<u64>(row), 0) = slot_count_parent.mShares[1](src, 0);
                }
                auto match_scalar = slice_si64_rows(dict_match, slot, 1);
                auto match_vec = repeat_secret_scalar(match_scalar, mapping.size());
                si64Matrix selected(mapping.size(), 1);
                selected.mShares[0].setZero();
                selected.mShares[1].setZero();
                cipher_mul(role, count, match_vec, selected, eval, enc, runtime);
                conditional_count.mShares[0] += selected.mShares[0];
                conditional_count.mShares[1] += selected.mShares[1];
                multiplication_count += mapping.size();
            }
        } else {
            const std::size_t flat_rows = share_pair_rows(join_path(sidecar_role_dir, "slot_category.shares.bin"));
            require_bool(flat_rows % cap_l == 0, "conditional slot file row count is not divisible by L");
            const std::size_t parent_rows = flat_rows / static_cast<std::size_t>(cap_l);
            const auto slot_category_parent = load_share_pair_vector(join_path(sidecar_role_dir, "slot_category.shares.bin"), flat_rows);
            const auto slot_count_parent = load_share_pair_vector(join_path(sidecar_role_dir, "slot_count.shares.bin"), flat_rows);
            const auto slot_valid_parent = load_share_pair_vector(join_path(sidecar_role_dir, "slot_valid.shares.bin"), flat_rows);
            (void)load_share_pair_vector(join_path(sidecar_role_dir, "selected_overflow.shares.bin"), parent_rows);
            auto literal_vec = repeat_secret_scalar(literal, mapping.size());
            for (std::uint64_t slot = 0; slot < cap_l; ++slot) {
                si64Matrix category(mapping.size(), 1);
                si64Matrix count(mapping.size(), 1);
                si64Matrix valid(mapping.size(), 1);
                for (std::size_t row = 0; row < mapping.size(); ++row) {
                    const auto parent = mapping[row];
                    require_bool(parent < parent_rows, "sample mapping row id outside conditional sidecar");
                    const auto src = static_cast<u64>(parent * cap_l + slot);
                    category.mShares[0](static_cast<u64>(row), 0) = slot_category_parent.mShares[0](src, 0);
                    category.mShares[1](static_cast<u64>(row), 0) = slot_category_parent.mShares[1](src, 0);
                    count.mShares[0](static_cast<u64>(row), 0) = slot_count_parent.mShares[0](src, 0);
                    count.mShares[1](static_cast<u64>(row), 0) = slot_count_parent.mShares[1](src, 0);
                    valid.mShares[0](static_cast<u64>(row), 0) = slot_valid_parent.mShares[0](src, 0);
                    valid.mShares[1](static_cast<u64>(row), 0) = slot_valid_parent.mShares[1](src, 0);
                }
                sbMatrix eq;
                cipher_eq(role, category, literal_vec, eq, eval, runtime);
                auto eq_arith = bool_to_arith(eq, role, enc, eval, runtime);
                si64Matrix valid_match(mapping.size(), 1);
                valid_match.mShares[0].setZero();
                valid_match.mShares[1].setZero();
                cipher_mul(role, eq_arith, valid, valid_match, eval, enc, runtime);
                if (fixed_dictionary_fallback_on_miss) {
                    matched_any.mShares[0] += valid_match.mShares[0];
                    matched_any.mShares[1] += valid_match.mShares[1];
                }
                si64Matrix selected(mapping.size(), 1);
                selected.mShares[0].setZero();
                selected.mShares[1].setZero();
                cipher_mul(role, count, valid_match, selected, eval, enc, runtime);
                conditional_count.mShares[0] += selected.mShares[0];
                conditional_count.mShares[1] += selected.mShares[1];
                equality_count += mapping.size();
                conversion_count += mapping.size();
                multiplication_count += mapping.size() * 2;
            }
        }
        si64Matrix conditional_factor(mapping.size(), 1);
        conditional_factor.mShares[0] = conditional_count.mShares[0] * kFixedOne;
        conditional_factor.mShares[1] = conditional_count.mShares[1] * kFixedOne;

        auto one_vec = share_public_i64_column(std::vector<i64>(mapping.size(), 1), role, enc, runtime);
        si64Matrix cm_selector = use_cm_vec;
        if (dense_payload || fixed_dictionary_fallback_on_miss) {
            si64Matrix active(mapping.size(), 1);
            active.mShares[0].setZero();
            active.mShares[1].setZero();
            cipher_mul(role, use_cm_vec, matched_any, active, eval, enc, runtime);
            cm_selector = active;
            multiplication_count += mapping.size();
        }
        si64Matrix not_use_cm(mapping.size(), 1);
        not_use_cm.mShares[0] = one_vec.mShares[0] - cm_selector.mShares[0];
        not_use_cm.mShares[1] = one_vec.mShares[1] - cm_selector.mShares[1];
        si64Matrix cm_part(mapping.size(), 1);
        cm_part.mShares[0].setZero();
        cm_part.mShares[1].setZero();
        si64Matrix base_part(mapping.size(), 1);
        base_part.mShares[0].setZero();
        base_part.mShares[1].setZero();
        cipher_mul(role, conditional_factor, cm_selector, cm_part, eval, enc, runtime);
        cipher_mul(role, baseline_factor, not_use_cm, base_part, eval, enc, runtime);
        factor.mShares[0] = cm_part.mShares[0] + base_part.mShares[0];
        factor.mShares[1] = cm_part.mShares[1] + base_part.mShares[1];
        multiplication_count += mapping.size() * 2;
        mux_count += mapping.size();
    }

    if (cmd.isSet("conditional_factor_output_share_bin")) {
        write_share_pair_vector(cmd.get<std::string>("conditional_factor_output_share_bin"), factor);
    }

    si64Matrix combined_factor = factor;
    if (!query.value("other_factor_share_bin", std::string()).empty()) {
        const auto other_factor = load_share_pair_vector(role_path(query.value("other_factor_share_bin", std::string()), role), mapping.size());
        combined_factor = multiply_fixed(factor, other_factor, eval, runtime);
        multiplication_count += mapping.size();
    }

    si64Matrix weighted = combined_factor;
    if (!query.value("weight_share_bin", std::string()).empty()) {
        const auto weights = load_or_public_ones(query, "weight_share_bin", mapping.size(), role, enc, runtime);
        weighted = multiply_fixed(factor, weights, eval, runtime);
        multiplication_count += mapping.size();
    }
    si64Matrix numerator = weighted;
    si64Matrix denominator = weighted;
    if (operation == "SUM" || operation == "AVG") {
        const std::string measure_path = query.value("measure_share_bin", std::string());
        require_bool(!measure_path.empty(), "SUM/AVG requires measure_share_bin");
        const auto measure = load_share_pair_vector(role_path(measure_path, role), mapping.size());
        numerator = multiply_fixed(weighted, measure, eval, runtime);
        multiplication_count += mapping.size();
    }

    if (operation == "AVG") {
        const std::string num_path = cmd.getOr<std::string>("conditional_numerator_output_share_bin", "");
        const std::string den_path = cmd.getOr<std::string>("conditional_denominator_output_share_bin", "");
        require_bool(!num_path.empty() && !den_path.empty(), "AVG requires numerator and denominator output paths");
        write_share_pair_vector(num_path, sum_rows(numerator));
        write_share_pair_vector(den_path, sum_rows(denominator));
        write_share_pair_vector(output_path, sum_rows(numerator));
    } else {
        write_share_pair_vector(output_path, sum_rows(numerator));
    }

    const auto ended = std::chrono::steady_clock::now();
    const auto comm = transport_stats(runtime);
    std::cout << json{
        {"mode", "m2_conditional_multiplier_eval"},
        {"conditional_multiplier_mode", mode},
        {"role", role},
        {"operation", operation},
        {"edge_id", edge_id},
        {"column_id", column_id},
        {"public_cap_L", cap_l},
        {"sample_rows", mapping.size()},
        {"public_sample_mapping_used", mode != kSampleMarkerStoredReverseMode && mode != kSampleMarkerCapReorderMode},
        {"sample_order_sidecar_used", mode == kSampleMarkerStoredReverseMode || mode == kSampleMarkerCapReorderMode},
        {"secure_evaluation_executed", true},
        {"three_role_protocol_executed", true},
        {"plaintext_fallback_used", false},
        {"reveal_count", 0},
        {"open_count", 0},
        {"wall_seconds", std::chrono::duration<double>(ended - started).count()},
        {"operation_counts", {
            {"equality", equality_count},
            {"conversion", conversion_count},
            {"multiplication", multiplication_count},
            {"mux", mux_count}
        }},
        {"communication", comm},
        {"bytes_sent_total", comm.value("bytes_sent_total", static_cast<std::uint64_t>(0))},
        {"bytes_received_total", comm.value("bytes_received_total", static_cast<std::uint64_t>(0))},
        {"output_share_path", output_path}
    }.dump() << std::endl;
    return 0;
}

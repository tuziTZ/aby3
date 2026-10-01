#include "m2_fixed_block_eval.h"

#include "aby3-Basic/Basics.h"
#include "aby3-Basic/BuildingBlocks.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace aby3;
using namespace oc;

namespace {

constexpr const char kMagic[16] = {
    'M', '2', 'F', 'B', 'L', 'O', 'C', 'K',
    'P', 'A', 'Y', 'L', 'O', 'A', 'D', '\0'};
constexpr const char kShareMagic[16] = {
    'M', '2', 'F', 'B', 'L', 'O', 'C', 'K',
    'S', 'H', 'A', 'R', 'E', '0', '2', '\0'};
constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kShareSchemaVersion = 2;
constexpr std::uint32_t kAnchorPredicateEdge = std::numeric_limits<std::uint32_t>::max();
constexpr i64 kFixedOneD16 = static_cast<i64>(1) << 16;
constexpr const char* kConditionalDenseMode = "sample_marker_partition_cap_reorder_v1";
constexpr const char* kConditionalDensePayloadMode = "fixed_dictionary_dense_counts_v1";
constexpr const char* kConditionalSidecarName = "conditional_multiplier_sample_marker_partition_cap_reorder_v1";

struct Header {
    char magic[16];
    std::uint32_t schema_version;
    std::uint32_t role_id;
    std::uint32_t anchor_slots;
    std::uint32_t edge_count;
    std::uint32_t record_count;
    std::uint32_t fixed_decimal_bits;
    std::uint64_t payload_checksum;
    std::uint64_t reserved;
};

struct Record {
    std::uint32_t anchor_slot;
    std::uint32_t edge_index;
    std::uint32_t basis_id;
    std::uint32_t flags;
    std::int64_t value_raw;
    std::int64_t weight_raw;
};

struct SecureShareRecord {
    std::uint32_t anchor_slot;
    std::uint32_t edge_index;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
    std::int64_t basis_share0;
    std::int64_t basis_share1;
    std::int64_t flags_share0;
    std::int64_t flags_share1;
    std::int64_t value_share0;
    std::int64_t value_share1;
    std::int64_t weight_share0;
    std::int64_t weight_share1;
};

struct Predicate {
    std::uint32_t edge_index;
    std::uint32_t basis_id;
};

struct Query {
    std::string query_id;
    std::vector<std::uint32_t> required_edges;
    std::vector<Predicate> predicates;
    std::vector<std::int64_t> anchor_mask_values;
};

struct SecurePredicate {
    std::uint32_t edge_index;
    std::int64_t basis_share0;
    std::int64_t basis_share1;
};

struct ConditionalEdge {
    std::uint32_t edge_index;
    std::string edge_id;
    std::string column_id;
    std::uint32_t public_cap_l;
    std::string scalar_artifact_dir;
    std::string conditional_manifest_json;
    std::int64_t literal_share0;
    std::int64_t literal_share1;
    std::int64_t use_cm_share0;
    std::int64_t use_cm_share1;
};

struct SecureQuery {
    std::string query_id;
    std::vector<std::uint32_t> required_edges;
    std::vector<SecurePredicate> predicates;
    std::vector<SecurePredicate> anchor_predicates;
    std::vector<ConditionalEdge> conditional_edges;
    std::string anchor_mask_share_bin;
};

struct SecurePayload {
    std::uint32_t anchor_slots = 0;
    std::uint32_t edge_count = 0;
    std::uint32_t record_count = 0;
    std::uint32_t fixed_decimal_bits = 16;
    std::vector<std::uint32_t> anchor_slots_by_record;
    std::vector<std::uint32_t> edge_index_by_record;
    si64Matrix basis;
    si64Matrix flags;
    si64Matrix value_raw;
    si64Matrix weight_raw;
};

std::uint64_t fnv1a64(const unsigned char* data, std::size_t size) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(data[i]);
        h *= 1099511628211ull;
    }
    return h;
}

template <typename T>
T read_plain(const std::vector<unsigned char>& bytes, std::size_t& offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("fixed-block payload truncated");
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
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

void require_bool(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void reset_transport_stats(Sh3Runtime& runtime) {
    runtime.mComm.mNext.resetStats();
    runtime.mComm.mPrev.resetStats();
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

std::string require_manifest_string(const json& manifest, const std::string& key) {
    require_bool(manifest.contains(key), "missing manifest field: " + key);
    require_bool(manifest.at(key).is_string(), "manifest field is not a string: " + key);
    const std::string value = manifest.at(key).get<std::string>();
    require_bool(!value.empty(), "empty manifest field: " + key);
    return value;
}

void require_privacy_aligned_offline_manifest(const json& manifest) {
    require_bool(
        manifest.value("offline_materialization_privacy_status", std::string()) == "privacy_aligned_production",
        "offline materialization is not privacy-aligned production");
    require_bool(
        manifest.value("offline_materializer_id", std::string()) == "m2_fixed_block_materialize",
        "offline materializer id is not accepted for production");
    require_bool(
        manifest.value("production_clear_payload_present", true) == false,
        "production clear payload provenance is forbidden");
    require_bool(
        manifest.value("materializer_clear_coordinator", true) == false,
        "central clear coordinator provenance is forbidden");
    require_bool(
        manifest.value("provider_separation", false) == true,
        "provider separation provenance missing");
    require_bool(
        manifest.value("no_cross_provider_plaintext", false) == true,
        "no-cross-provider-plaintext provenance missing");
    require_bool(
        manifest.value("no_combined_clear_model", false) == true,
        "combined clear model provenance missing");
    require_bool(
        manifest.value("share_only_cross_relation_alignment", false) == true,
        "share-only cross-relation alignment provenance missing");
    require_bool(
        manifest.value("secure_m2_statistics", false) == true,
        "secure M2 statistics provenance missing");
    require_bool(
        manifest.value("secure_anchor_selection", false) == true,
        "secure anchor selection provenance missing");
    require_bool(
        manifest.value("secure_fixed_block_materialization", false) == true,
        "secure fixed-block materialization provenance missing");
    require_bool(
        manifest.value("fixed_public_capacity", false) == true,
        "fixed public capacity provenance missing");
    require_bool(
        manifest.value("secret_occupancy", false) == true,
        "secret occupancy provenance missing");
    require_bool(
        manifest.value("no_reveal_open", false) == true,
        "no reveal/open provenance missing");
    require_bool(
        manifest.value("no_plaintext_fallback", false) == true,
        "no plaintext fallback provenance missing");
    (void)require_manifest_string(manifest, "privacy_provenance_hash");
}

std::vector<Record> load_records(
    const std::string& payload_path,
    const json& manifest,
    std::uint32_t role) {
    const auto bytes = read_binary(payload_path);
    std::size_t offset = 0;
    Header header = read_plain<Header>(bytes, offset);
    require_bool(std::memcmp(header.magic, kMagic, sizeof(kMagic)) == 0, "bad fixed-block magic");
    require_bool(header.schema_version == kSchemaVersion, "bad fixed-block schema version");
    require_bool(header.role_id == role, "wrong fixed-block role id");
    require_bool(header.fixed_decimal_bits == manifest.value("fixed_decimal_bits", 16), "wrong fixed-point scale");
    require_bool(header.anchor_slots == manifest.value("anchor_slots", 0), "bad anchor slot count");
    require_bool(header.edge_count == manifest.value("edge_count", 0), "bad edge count");
    require_bool(header.record_count == manifest.value("record_count", 0), "bad record count");
    require_bool(manifest.value("record_bytes", 0) == static_cast<int>(sizeof(Record)), "bad record size");
    const std::size_t expected_size = sizeof(Header) + static_cast<std::size_t>(header.record_count) * sizeof(Record);
    require_bool(bytes.size() == expected_size, "extra trailing payload or truncated record area");
    const bool external_checksum = manifest.value("payload_checksum_algorithm", std::string("fnv1a64")) == "external_sha256";
    if (external_checksum) {
        require_bool(header.payload_checksum == 0, "external checksum payload must set header checksum to zero");
    } else {
        const std::uint64_t actual_checksum = fnv1a64(bytes.data() + sizeof(Header), bytes.size() - sizeof(Header));
        require_bool(actual_checksum == header.payload_checksum, "bad payload checksum");
    }

    std::vector<Record> records;
    records.reserve(header.record_count);
    for (std::uint32_t idx = 0; idx < header.record_count; ++idx) {
        Record record = read_plain<Record>(bytes, offset);
        require_bool(record.anchor_slot < header.anchor_slots, "record anchor_slot outside public shape");
        require_bool(record.edge_index < header.edge_count, "record edge_index outside public shape");
        const bool occupied = (record.flags & 1u) != 0;
        const bool dummy = (record.flags & 2u) != 0;
        require_bool(!(occupied && dummy), "record cannot be both occupied and dummy");
        if (dummy) {
            require_bool(record.value_raw == 0, "dummy record has nonzero value");
        }
        records.push_back(record);
    }
    return records;
}

SecurePayload load_secure_share_payload(
    const std::string& payload_path,
    const json& manifest,
    std::uint32_t role) {
    const auto bytes = read_binary(payload_path);
    std::size_t offset = 0;
    Header header = read_plain<Header>(bytes, offset);
    require_bool(std::memcmp(header.magic, kShareMagic, sizeof(kShareMagic)) == 0, "bad fixed-block share magic");
    require_bool(header.schema_version == kShareSchemaVersion, "bad fixed-block share schema version");
    require_bool(header.role_id == role, "wrong fixed-block share role id");
    require_bool(header.fixed_decimal_bits == manifest.value("fixed_decimal_bits", 16), "wrong fixed-block share fixed-point scale");
    require_bool(header.anchor_slots == manifest.value("anchor_slots", 0), "bad fixed-block share anchor slot count");
    require_bool(header.edge_count == manifest.value("edge_count", 0), "bad fixed-block share edge count");
    require_bool(header.record_count == manifest.value("record_count", 0), "bad fixed-block share record count");
    require_bool(manifest.value("record_bytes", 0) == static_cast<int>(sizeof(SecureShareRecord)), "bad fixed-block share record size");
    require_bool(manifest.value("overflow_count", 0) == 0, "nonzero fixed-block share overflow marker");
    const std::size_t expected_size = sizeof(Header) + static_cast<std::size_t>(header.record_count) * sizeof(SecureShareRecord);
    require_bool(bytes.size() == expected_size, "extra trailing fixed-block share payload or truncated record area");
    const bool external_checksum = manifest.value("payload_checksum_algorithm", std::string("fnv1a64")) == "external_sha256";
    if (external_checksum) {
        require_bool(header.payload_checksum == 0, "external checksum share payload must set header checksum to zero");
    } else {
        const std::uint64_t actual_checksum = fnv1a64(bytes.data() + sizeof(Header), bytes.size() - sizeof(Header));
        require_bool(actual_checksum == header.payload_checksum, "bad fixed-block share payload checksum");
    }
    SecurePayload payload;
    payload.anchor_slots = header.anchor_slots;
    payload.edge_count = header.edge_count;
    payload.record_count = header.record_count;
    payload.fixed_decimal_bits = header.fixed_decimal_bits;
    payload.anchor_slots_by_record.reserve(header.record_count);
    payload.edge_index_by_record.reserve(header.record_count);
    payload.basis.resize(header.record_count, 1);
    payload.flags.resize(header.record_count, 1);
    payload.value_raw.resize(header.record_count, 1);
    payload.weight_raw.resize(header.record_count, 1);
    for (std::uint32_t idx = 0; idx < header.record_count; ++idx) {
        SecureShareRecord record = read_plain<SecureShareRecord>(bytes, offset);
        require_bool(record.anchor_slot < header.anchor_slots, "share record anchor_slot outside public shape");
        require_bool(
            record.edge_index < header.edge_count || record.edge_index == kAnchorPredicateEdge,
            "share record edge_index outside public shape");
        require_bool(record.reserved0 == 0 && record.reserved1 == 0, "share record reserved fields must be zero");
        payload.anchor_slots_by_record.push_back(record.anchor_slot);
        payload.edge_index_by_record.push_back(record.edge_index);
        payload.basis.mShares[0](idx, 0) = record.basis_share0;
        payload.basis.mShares[1](idx, 0) = record.basis_share1;
        payload.flags.mShares[0](idx, 0) = record.flags_share0;
        payload.flags.mShares[1](idx, 0) = record.flags_share1;
        payload.value_raw.mShares[0](idx, 0) = record.value_share0;
        payload.value_raw.mShares[1](idx, 0) = record.value_share1;
        payload.weight_raw.mShares[0](idx, 0) = record.weight_share0;
        payload.weight_raw.mShares[1](idx, 0) = record.weight_share1;
    }
    return payload;
}

Query load_query_from_json(const json& q, std::uint32_t edge_count) {
    Query query;
    query.query_id = q.value("query_id", std::string("fixture"));
    for (const auto& item : q.at("required_edges")) {
        const auto edge = item.get<std::uint32_t>();
        require_bool(edge < edge_count, "query references unknown edge");
        query.required_edges.push_back(edge);
    }
    std::sort(query.required_edges.begin(), query.required_edges.end());
    require_bool(
        std::adjacent_find(query.required_edges.begin(), query.required_edges.end()) == query.required_edges.end(),
        "duplicate required edge");
    for (const auto& item : q.at("predicates")) {
        Predicate pred{item.at("edge_index").get<std::uint32_t>(), item.at("basis_id").get<std::uint32_t>()};
        require_bool(pred.edge_index < edge_count, "predicate references unknown edge");
        query.predicates.push_back(pred);
    }
    if (q.contains("anchor_mask_values")) {
        for (const auto& item : q.at("anchor_mask_values")) {
            query.anchor_mask_values.push_back(item.get<std::int64_t>());
        }
    }
    return query;
}

std::vector<Query> load_query_bundle(const std::string& query_path, std::uint32_t edge_count) {
    const auto bundle = read_json_file(query_path);
    std::vector<Query> queries;
    const auto& items = bundle.is_array() ? bundle : bundle.at("queries");
    for (const auto& item : items) {
        queries.push_back(load_query_from_json(item, edge_count));
    }
    require_bool(!queries.empty(), "query bundle is empty");
    return queries;
}

Query load_query(const std::string& query_path, std::uint32_t edge_count) {
    const auto q = read_json_file(query_path);
    return load_query_from_json(q, edge_count);
}

SecureQuery load_secure_query_from_json(const json& q, std::uint32_t edge_count) {
    SecureQuery query;
    query.query_id = q.value("query_id", std::string("secure_fixture"));
    require_bool(q.value("predicate_literals_in_public_query", true) == false, "secure query must not expose predicate literals");
    for (const auto& item : q.at("required_edges")) {
        const auto edge = item.get<std::uint32_t>();
        require_bool(edge < edge_count, "secure query references unknown edge");
        query.required_edges.push_back(edge);
    }
    std::sort(query.required_edges.begin(), query.required_edges.end());
    require_bool(
        std::adjacent_find(query.required_edges.begin(), query.required_edges.end()) == query.required_edges.end(),
        "duplicate secure required edge");
    for (const auto& item : q.at("predicates")) {
        SecurePredicate pred{
            item.at("edge_index").get<std::uint32_t>(),
            item.at("basis_share0").get<std::int64_t>(),
            item.at("basis_share1").get<std::int64_t>()};
        require_bool(pred.edge_index < edge_count, "secure predicate references unknown edge");
        query.predicates.push_back(pred);
    }
    if (q.contains("anchor_mask_share_bin")) {
        query.anchor_mask_share_bin = q.at("anchor_mask_share_bin").get<std::string>();
    }
    if (q.contains("anchor_predicates")) {
        for (const auto& item : q.at("anchor_predicates")) {
            SecurePredicate pred{
                kAnchorPredicateEdge,
                item.at("basis_share0").get<std::int64_t>(),
                item.at("basis_share1").get<std::int64_t>()};
            query.anchor_predicates.push_back(pred);
        }
    }
    if (q.contains("conditional_edges")) {
        for (const auto& item : q.at("conditional_edges")) {
            ConditionalEdge edge;
            edge.edge_index = item.at("edge_index").get<std::uint32_t>();
            require_bool(edge.edge_index < edge_count, "conditional edge references unknown edge");
            edge.edge_id = item.at("edge_id").get<std::string>();
            edge.column_id = item.at("column_id").get<std::string>();
            edge.public_cap_l = item.at("public_cap_L").get<std::uint32_t>();
            edge.scalar_artifact_dir = item.at("scalar_artifact_dir").get<std::string>();
            edge.conditional_manifest_json = item.at("conditional_manifest_json").get<std::string>();
            edge.literal_share0 = item.at("literal_share0").get<std::int64_t>();
            edge.literal_share1 = item.at("literal_share1").get<std::int64_t>();
            edge.use_cm_share0 = item.at("use_cm_share0").get<std::int64_t>();
            edge.use_cm_share1 = item.at("use_cm_share1").get<std::int64_t>();
            require_bool(!edge.scalar_artifact_dir.empty(), "conditional scalar artifact dir is empty");
            require_bool(!edge.conditional_manifest_json.empty(), "conditional manifest path is empty");
            query.conditional_edges.push_back(edge);
        }
    }
    require_bool(!query.anchor_mask_share_bin.empty() || !q.contains("anchor_mask_required"), "missing secure anchor mask share file");
    require_bool(q.value("anchor_mask_public", false) == false, "secure query must not expose anchor mask values");
    require_bool(q.value("legacy_evaluator_allowed", false) == false, "legacy evaluator must not be allowed");
    require_bool(q.value("validate_only_for_production", false) == false, "validate-only mode forbidden for production");
    require_bool(q.value("evaluator_id", std::string("m2_fixed_block_v2")) == "m2_fixed_block_v2", "wrong evaluator id");
    return query;
}

SecureQuery load_secure_query(const std::string& query_path, std::uint32_t edge_count) {
    const auto q = read_json_file(query_path);
    return load_secure_query_from_json(q, edge_count);
}

std::vector<SecureQuery> load_secure_query_bundle(const std::string& query_path, std::uint32_t edge_count) {
    const auto bundle = read_json_file(query_path);
    std::vector<SecureQuery> queries;
    const auto& items = bundle.is_array() ? bundle : bundle.at("queries");
    for (const auto& item : items) {
        queries.push_back(load_secure_query_from_json(item, edge_count));
    }
    require_bool(!queries.empty(), "secure query bundle is empty");
    return queries;
}

si64Matrix share_public_i64_column(
    const std::vector<i64>& values,
    int role,
    Sh3Encryptor& enc,
    Sh3Runtime& runtime) {
    i64Matrix plain(values.size(), 1);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        plain(static_cast<u64>(idx), 0) = values[idx];
    }
    si64Matrix shared;
    plain_mat_to_cipher_mat(role, enc, runtime, plain, shared);
    return shared;
}

si64Matrix slice_si64_rows(const si64Matrix& src, const std::vector<std::uint32_t>& indices) {
    si64Matrix out(indices.size(), 1);
    for (std::size_t row = 0; row < indices.size(); ++row) {
        const auto src_row = static_cast<u64>(indices[row]);
        out.mShares[0](static_cast<u64>(row), 0) = src.mShares[0](src_row, 0);
        out.mShares[1](static_cast<u64>(row), 0) = src.mShares[1](src_row, 0);
    }
    return out;
}

si64Matrix repeat_secret_scalar(const SecurePredicate& pred, std::size_t rows) {
    si64Matrix out(rows, 1);
    for (std::size_t row = 0; row < rows; ++row) {
        out.mShares[0](static_cast<u64>(row), 0) = pred.basis_share0;
        out.mShares[1](static_cast<u64>(row), 0) = pred.basis_share1;
    }
    return out;
}

si64Matrix load_share_pair_vector(const std::string& path, std::size_t rows) {
    const auto bytes = read_binary(path);
    require_bool(bytes.size() == rows * sizeof(i64) * 2, "bad share-pair vector size");
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
    require_bool(bytes.size() % (sizeof(i64) * 2) == 0, "bad share-pair vector size: " + path);
    return bytes.size() / (sizeof(i64) * 2);
}

si64Matrix bool_to_arith(sbMatrix value, int role, Sh3Encryptor& enc, Sh3Evaluator& eval, Sh3Runtime& runtime) {
    si64Matrix out(value.rows(), 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    bool2arith(role, value, out, enc, eval, runtime);
    return out;
}

si64Matrix repeat_secret_i64_pair(std::int64_t share0, std::int64_t share1, std::size_t rows) {
    si64Matrix out(rows, 1);
    for (std::size_t row = 0; row < rows; ++row) {
        out.mShares[0](static_cast<u64>(row), 0) = share0;
        out.mShares[1](static_cast<u64>(row), 0) = share1;
    }
    return out;
}

si64Matrix multiply_i64(
    const si64Matrix& lhs,
    const si64Matrix& rhs,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime) {
    require_bool(lhs.rows() == rhs.rows(), "i64 multiply row mismatch");
    si64Matrix out(lhs.rows(), 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    cipher_mul(role, lhs, rhs, out, eval, enc, runtime);
    return out;
}

si64Matrix multiply_fixed_d16(const si64Matrix& lhs, const si64Matrix& rhs, Sh3Evaluator& eval, Sh3Runtime& runtime) {
    require_bool(lhs.rows() == rhs.rows(), "D16 multiply row mismatch");
    sf64Matrix<D16> left(lhs.rows(), 1);
    sf64Matrix<D16> right(lhs.rows(), 1);
    sf64Matrix<D16> product(lhs.rows(), 1);
    left.i64Cast().mShares[0] = lhs.mShares[0];
    left.i64Cast().mShares[1] = lhs.mShares[1];
    right.i64Cast().mShares[0] = rhs.mShares[0];
    right.i64Cast().mShares[1] = rhs.mShares[1];
    eval.asyncMul(runtime, left, right, product).get();
    si64Matrix out(lhs.rows(), 1);
    out.mShares[0] = product.i64Cast().mShares[0];
    out.mShares[1] = product.i64Cast().mShares[1];
    return out;
}

void require_conditional_manifest(
    const json& manifest,
    const ConditionalEdge& edge,
    std::size_t anchor_slots,
    std::uint32_t fixed_decimal_bits) {
    require_bool(manifest.value("target", std::string()) == "M2_CONDITIONAL_MULTIPLIER_SAMPLE_MARKER_PARTITION_CAP_REORDER_V1", "bad conditional target");
    require_bool(manifest.value("schema_version", std::string()) == "M2CMSMALLREORDERDENSE01", "bad conditional schema");
    require_bool(manifest.value("conditional_multiplier_mode", std::string()) == kConditionalDenseMode, "bad conditional mode");
    require_bool(manifest.value("conditional_payload_mode", std::string()) == kConditionalDensePayloadMode, "bad conditional payload mode");
    require_bool(manifest.value("edge_id", std::string()) == edge.edge_id, "conditional edge id mismatch");
    require_bool(manifest.value("column_id", std::string()) == edge.column_id, "conditional column id mismatch");
    require_bool(manifest.value("public_cap_L", std::uint32_t(0)) == edge.public_cap_l, "conditional L mismatch");
    require_bool(manifest.value("sample_rows", std::uint64_t(0)) == static_cast<std::uint64_t>(anchor_slots), "conditional sample_rows mismatch");
    require_bool(manifest.value("artifact_state", std::string()) == "complete", "conditional artifact is not complete");
    require_bool(manifest.value("selected_overflow_authorized", false), "conditional overflow authorization missing");
    require_bool(!manifest.value("contains_real_row_id", true), "conditional artifact exposes real row id");
    require_bool(!manifest.value("contains_join_key", true), "conditional artifact exposes join key");
    require_bool(!manifest.value("contains_dense_offset", true), "conditional artifact exposes dense offset");
    require_bool(!manifest.value("contains_sample_row_id_by_position", true), "conditional artifact exposes sample row id by position");
    if (manifest.contains("fixed_point")) {
        const auto fp = manifest.at("fixed_point");
        require_bool(fp.value("fixed_decimal_bits", static_cast<int>(fixed_decimal_bits)) == static_cast<int>(fixed_decimal_bits), "conditional fixed-point mismatch");
    }
}

si64Matrix select_conditional_factor(
    const ConditionalEdge& edge,
    std::size_t anchor_slots,
    std::uint32_t fixed_decimal_bits,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    std::uint64_t& equality_count,
    std::uint64_t& mux_count,
    std::uint64_t& multiply_count,
    std::uint64_t& conversion_count,
    std::uint64_t& conditional_edge_count) {
    require_bool(fixed_decimal_bits == 16, "conditional dense bridge currently requires D16 fixed point");
    const auto manifest = read_json_file(edge.conditional_manifest_json);
    require_conditional_manifest(manifest, edge, anchor_slots, fixed_decimal_bits);
    const std::string sidecar_dir = dirname_of(edge.conditional_manifest_json);
    require_bool(path_exists(join_path(sidecar_dir, "COMPLETE")), "conditional artifact missing COMPLETE marker");
    const std::string role_dir = join_path(edge.scalar_artifact_dir, "role_" + std::to_string(role));
    const std::string sidecar_role_dir = join_path(sidecar_dir, "role_" + std::to_string(role));
    auto baseline_factor = load_share_pair_vector(join_path(role_dir, "mu.shares.bin"), anchor_slots);
    auto dictionary = load_share_pair_vector(join_path(sidecar_role_dir, "dictionary.shares.bin"), edge.public_cap_l);
    const std::size_t flat_rows = share_pair_rows(join_path(sidecar_role_dir, "dense_count.shares.bin"));
    require_bool(flat_rows == anchor_slots * static_cast<std::size_t>(edge.public_cap_l), "conditional dense row count mismatch");
    const auto dense_count = load_share_pair_vector(join_path(sidecar_role_dir, "dense_count.shares.bin"), flat_rows);
    (void)load_share_pair_vector(join_path(sidecar_role_dir, "selected_overflow.shares.bin"), anchor_slots);
    (void)load_share_pair_vector(join_path(sidecar_role_dir, "sample_order_integrity.shares.bin"), anchor_slots + 3);

    auto literal_dict_vec = repeat_secret_i64_pair(edge.literal_share0, edge.literal_share1, edge.public_cap_l);
    sbMatrix dict_eq;
    cipher_eq(role, dictionary, literal_dict_vec, dict_eq, eval, runtime);
    auto dict_match = bool_to_arith(dict_eq, role, enc, eval, runtime);
    equality_count += edge.public_cap_l;
    conversion_count += edge.public_cap_l;

    si64Matrix conditional_count(anchor_slots, 1);
    conditional_count.mShares[0].setZero();
    conditional_count.mShares[1].setZero();
    si64Matrix matched_any_scalar(1, 1);
    matched_any_scalar.mShares[0].setZero();
    matched_any_scalar.mShares[1].setZero();
    for (std::uint32_t slot = 0; slot < edge.public_cap_l; ++slot) {
        matched_any_scalar.mShares[0](0, 0) += dict_match.mShares[0](slot, 0);
        matched_any_scalar.mShares[1](0, 0) += dict_match.mShares[1](slot, 0);
        auto match_vec = repeat_secret_i64_pair(dict_match.mShares[0](slot, 0), dict_match.mShares[1](slot, 0), anchor_slots);
        si64Matrix count(anchor_slots, 1);
        for (std::size_t row = 0; row < anchor_slots; ++row) {
            const auto src = static_cast<u64>(row * edge.public_cap_l + slot);
            count.mShares[0](static_cast<u64>(row), 0) = dense_count.mShares[0](src, 0);
            count.mShares[1](static_cast<u64>(row), 0) = dense_count.mShares[1](src, 0);
        }
        auto selected = multiply_i64(count, match_vec, role, enc, eval, runtime);
        conditional_count.mShares[0] += selected.mShares[0];
        conditional_count.mShares[1] += selected.mShares[1];
        multiply_count += anchor_slots;
    }
    auto matched_any = repeat_secret_i64_pair(matched_any_scalar.mShares[0](0, 0), matched_any_scalar.mShares[1](0, 0), anchor_slots);
    auto use_cm = repeat_secret_i64_pair(edge.use_cm_share0, edge.use_cm_share1, anchor_slots);
    auto cm_selector = multiply_i64(use_cm, matched_any, role, enc, eval, runtime);
    multiply_count += anchor_slots;

    si64Matrix conditional_factor(anchor_slots, 1);
    conditional_factor.mShares[0] = conditional_count.mShares[0] * kFixedOneD16;
    conditional_factor.mShares[1] = conditional_count.mShares[1] * kFixedOneD16;
    auto one_vec = share_public_i64_column(std::vector<i64>(anchor_slots, 1), role, enc, runtime);
    si64Matrix not_use_cm(anchor_slots, 1);
    not_use_cm.mShares[0] = one_vec.mShares[0] - cm_selector.mShares[0];
    not_use_cm.mShares[1] = one_vec.mShares[1] - cm_selector.mShares[1];
    auto cm_part = multiply_i64(conditional_factor, cm_selector, role, enc, eval, runtime);
    auto base_part = multiply_i64(baseline_factor, not_use_cm, role, enc, eval, runtime);
    si64Matrix out(anchor_slots, 1);
    out.mShares[0] = cm_part.mShares[0] + base_part.mShares[0];
    out.mShares[1] = cm_part.mShares[1] + base_part.mShares[1];
    multiply_count += anchor_slots * 2;
    mux_count += anchor_slots;
    conditional_edge_count += 1;
    return out;
}

const ConditionalEdge* find_conditional_edge(const SecureQuery& query, std::uint32_t edge_index) {
    for (const auto& edge : query.conditional_edges) {
        if (edge.edge_index == edge_index) {
            return &edge;
        }
    }
    return nullptr;
}

void add_row_to_anchor(si64Matrix& anchors, std::uint32_t anchor, const si64Matrix& row_values, std::size_t row) {
    anchors.mShares[0](anchor, 0) += row_values.mShares[0](static_cast<u64>(row), 0);
    anchors.mShares[1](anchor, 0) += row_values.mShares[1](static_cast<u64>(row), 0);
}

void write_result_share_pair(const std::string& path, const si64Matrix& result) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write result share file: " + path);
    }
    const i64 lhs = result.mShares[0](0, 0);
    const i64 rhs = result.mShares[1](0, 0);
    out.write(reinterpret_cast<const char*>(&lhs), sizeof(lhs));
    out.write(reinterpret_cast<const char*>(&rhs), sizeof(rhs));
}

json evaluate_secure(
    const SecurePayload& payload,
    const SecureQuery& query,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    const std::string& output_share_path) {
    require_bool(
        payload.fixed_decimal_bits == 0 || payload.fixed_decimal_bits == 16,
        "secure evaluator currently requires D0 or D16 fixed point");
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t equality_count = 0;
    std::uint64_t mux_count = 0;
    std::uint64_t multiply_count = 0;
    std::uint64_t conversion_count = 0;
    std::uint64_t secure_gate_count = 0;
    std::uint64_t conditional_edge_count = 0;

    std::vector<si64Matrix> selected_by_edge;
    selected_by_edge.reserve(query.required_edges.size());
    si64Matrix anchor_weight(payload.anchor_slots, 1);
    anchor_weight.mShares[0].setZero();
    anchor_weight.mShares[1].setZero();
    si64Matrix anchor_mask = share_public_i64_column(std::vector<i64>(payload.anchor_slots, 1), role, enc, runtime);
    if (!query.anchor_mask_share_bin.empty()) {
        anchor_mask = load_share_pair_vector(query.anchor_mask_share_bin, payload.anchor_slots);
    }
    for (const auto& anchor_pred : query.anchor_predicates) {
        std::vector<std::uint32_t> record_indices;
        for (std::uint32_t idx = 0; idx < payload.record_count; ++idx) {
            if (payload.edge_index_by_record[idx] == kAnchorPredicateEdge) {
                record_indices.push_back(idx);
            }
        }
        require_bool(!record_indices.empty(), "anchor predicate has no public fixed-block records");
        auto basis = slice_si64_rows(payload.basis, record_indices);
        auto flags = slice_si64_rows(payload.flags, record_indices);
        auto values = slice_si64_rows(payload.value_raw, record_indices);
        auto predicate_basis = repeat_secret_scalar(anchor_pred, record_indices.size());
        sbMatrix basis_eq;
        cipher_eq(role, basis, predicate_basis, basis_eq, eval, runtime);
        auto one = share_public_i64_column(std::vector<i64>(record_indices.size(), 1), role, enc, runtime);
        sbMatrix flag_eq_one;
        cipher_eq(role, flags, one, flag_eq_one, eval, runtime);
        auto basis_mask = bool_to_arith(basis_eq, role, enc, eval, runtime);
        auto flag_mask = bool_to_arith(flag_eq_one, role, enc, eval, runtime);
        si64Matrix valid_mask(record_indices.size(), 1);
        valid_mask.mShares[0].setZero();
        valid_mask.mShares[1].setZero();
        cipher_mul(role, basis_mask, flag_mask, valid_mask, eval, enc, runtime);
        si64Matrix selected_anchor(payload.anchor_slots, 1);
        selected_anchor.mShares[0].setZero();
        selected_anchor.mShares[1].setZero();
        si64Matrix selected_records(record_indices.size(), 1);
        selected_records.mShares[0].setZero();
        selected_records.mShares[1].setZero();
        cipher_mul(role, values, valid_mask, selected_records, eval, enc, runtime);
        for (std::size_t row = 0; row < record_indices.size(); ++row) {
            add_row_to_anchor(selected_anchor, payload.anchor_slots_by_record[record_indices[row]], selected_records, row);
        }
        si64Matrix next_mask(payload.anchor_slots, 1);
        next_mask.mShares[0].setZero();
        next_mask.mShares[1].setZero();
        cipher_mul(role, anchor_mask, selected_anchor, next_mask, eval, enc, runtime);
        anchor_mask = next_mask;
        equality_count += record_indices.size() * 2;
        mux_count += record_indices.size();
        multiply_count += record_indices.size() * 2 + payload.anchor_slots;
        conversion_count += record_indices.size() * 2;
        secure_gate_count += record_indices.size() * 5 + payload.anchor_slots;
    }

    for (std::size_t edge_pos = 0; edge_pos < query.required_edges.size(); ++edge_pos) {
        const auto edge = query.required_edges[edge_pos];
        if (const auto* conditional = find_conditional_edge(query, edge)) {
            selected_by_edge.push_back(select_conditional_factor(
                *conditional,
                payload.anchor_slots,
                payload.fixed_decimal_bits,
                role,
                enc,
                eval,
                runtime,
                equality_count,
                mux_count,
                multiply_count,
                conversion_count,
                conditional_edge_count));
            if (edge_pos == 0) {
                const i64 unit_weight = payload.fixed_decimal_bits == 16 ? kFixedOneD16 : 1;
                anchor_weight = share_public_i64_column(std::vector<i64>(payload.anchor_slots, unit_weight), role, enc, runtime);
            }
            continue;
        }
        auto pred_it = std::find_if(query.predicates.begin(), query.predicates.end(), [&](const SecurePredicate& pred) {
            return pred.edge_index == edge;
        });
        require_bool(pred_it != query.predicates.end(), "missing secure predicate for required edge");
        std::vector<std::uint32_t> record_indices;
        for (std::uint32_t idx = 0; idx < payload.record_count; ++idx) {
            if (payload.edge_index_by_record[idx] == edge) {
                record_indices.push_back(idx);
            }
        }
        require_bool(!record_indices.empty(), "required edge has no public fixed-block records");
        auto basis = slice_si64_rows(payload.basis, record_indices);
        auto flags = slice_si64_rows(payload.flags, record_indices);
        auto values = slice_si64_rows(payload.value_raw, record_indices);
        auto predicate_basis = repeat_secret_scalar(*pred_it, record_indices.size());
        sbMatrix basis_eq;
        cipher_eq(role, basis, predicate_basis, basis_eq, eval, runtime);
        auto one = share_public_i64_column(std::vector<i64>(record_indices.size(), 1), role, enc, runtime);
        sbMatrix flag_eq_one;
        cipher_eq(role, flags, one, flag_eq_one, eval, runtime);
        auto basis_mask = bool_to_arith(basis_eq, role, enc, eval, runtime);
        auto flag_mask = bool_to_arith(flag_eq_one, role, enc, eval, runtime);
        si64Matrix valid_mask(record_indices.size(), 1);
        valid_mask.mShares[0].setZero();
        valid_mask.mShares[1].setZero();
        cipher_mul(role, basis_mask, flag_mask, valid_mask, eval, enc, runtime);
        si64Matrix selected_records(record_indices.size(), 1);
        selected_records.mShares[0].setZero();
        selected_records.mShares[1].setZero();
        cipher_mul(role, values, valid_mask, selected_records, eval, enc, runtime);
        si64Matrix selected_anchor(payload.anchor_slots, 1);
        selected_anchor.mShares[0].setZero();
        selected_anchor.mShares[1].setZero();
        for (std::size_t row = 0; row < record_indices.size(); ++row) {
            add_row_to_anchor(selected_anchor, payload.anchor_slots_by_record[record_indices[row]], selected_records, row);
            if (edge_pos == 0) {
                anchor_weight.mShares[0](payload.anchor_slots_by_record[record_indices[row]], 0) =
                    payload.weight_raw.mShares[0](record_indices[row], 0);
                anchor_weight.mShares[1](payload.anchor_slots_by_record[record_indices[row]], 0) =
                    payload.weight_raw.mShares[1](record_indices[row], 0);
            }
        }
        selected_by_edge.push_back(selected_anchor);
        equality_count += record_indices.size() * 2;
        mux_count += record_indices.size();
        multiply_count += record_indices.size() * 2;
        conversion_count += record_indices.size() * 2;
        secure_gate_count += record_indices.size() * 5;
    }

    si64Matrix weighted(payload.anchor_slots, 1);
    if (payload.fixed_decimal_bits == 0) {
        si64Matrix product = selected_by_edge.front();
        for (std::size_t idx = 1; idx < selected_by_edge.size(); ++idx) {
            si64Matrix out(payload.anchor_slots, 1);
            out.mShares[0].setZero();
            out.mShares[1].setZero();
            cipher_mul(role, product, selected_by_edge[idx], out, eval, enc, runtime);
            product = out;
            multiply_count += payload.anchor_slots;
            secure_gate_count += payload.anchor_slots;
        }
        weighted.mShares[0].setZero();
        weighted.mShares[1].setZero();
        cipher_mul(role, product, anchor_weight, weighted, eval, enc, runtime);
        if (!query.anchor_predicates.empty()) {
            si64Matrix masked(payload.anchor_slots, 1);
            masked.mShares[0].setZero();
            masked.mShares[1].setZero();
            cipher_mul(role, weighted, anchor_mask, masked, eval, enc, runtime);
            weighted = masked;
            multiply_count += payload.anchor_slots;
            secure_gate_count += payload.anchor_slots;
        }
    } else {
        sf64Matrix<D16> product(payload.anchor_slots, 1);
        product.i64Cast().mShares[0] = selected_by_edge.front().mShares[0];
        product.i64Cast().mShares[1] = selected_by_edge.front().mShares[1];
        for (std::size_t idx = 1; idx < selected_by_edge.size(); ++idx) {
            sf64Matrix<D16> rhs(payload.anchor_slots, 1);
            rhs.i64Cast().mShares[0] = selected_by_edge[idx].mShares[0];
            rhs.i64Cast().mShares[1] = selected_by_edge[idx].mShares[1];
            sf64Matrix<D16> out(payload.anchor_slots, 1);
            eval.asyncMul(runtime, product, rhs, out).get();
            product = out;
            multiply_count += payload.anchor_slots;
            secure_gate_count += payload.anchor_slots;
        }
        sf64Matrix<D16> weights(payload.anchor_slots, 1);
        weights.i64Cast().mShares[0] = anchor_weight.mShares[0];
        weights.i64Cast().mShares[1] = anchor_weight.mShares[1];
        sf64Matrix<D16> weighted_fixed(payload.anchor_slots, 1);
        eval.asyncMul(runtime, product, weights, weighted_fixed).get();
        weighted.mShares[0] = weighted_fixed.i64Cast().mShares[0];
        weighted.mShares[1] = weighted_fixed.i64Cast().mShares[1];
        if (!query.anchor_predicates.empty() || !query.anchor_mask_share_bin.empty()) {
            sf64Matrix<D16> mask(payload.anchor_slots, 1);
            mask.i64Cast().mShares[0] = anchor_mask.mShares[0];
            mask.i64Cast().mShares[1] = anchor_mask.mShares[1];
            sf64Matrix<D16> masked(payload.anchor_slots, 1);
            eval.asyncMul(runtime, weighted_fixed, mask, masked).get();
            weighted.mShares[0] = masked.i64Cast().mShares[0];
            weighted.mShares[1] = masked.i64Cast().mShares[1];
            multiply_count += payload.anchor_slots;
            secure_gate_count += payload.anchor_slots;
        }
    }
    multiply_count += payload.anchor_slots;
    secure_gate_count += payload.anchor_slots;

    si64Matrix result(1, 1);
    result.mShares[0].setZero();
    result.mShares[1].setZero();
    for (u64 row = 0; row < weighted.rows(); ++row) {
        result.mShares[0](0, 0) += weighted.mShares[0](row, 0);
        result.mShares[1](0, 0) += weighted.mShares[1](row, 0);
    }
    write_result_share_pair(output_share_path, result);
    const auto ended = std::chrono::steady_clock::now();
    return json{
        {"mode", "secure_evaluate"},
        {"query_id", query.query_id},
        {"role", role},
        {"secure_evaluation_executed", true},
        {"three_role_protocol_executed", true},
        {"client_reconstruction_executed", false},
        {"plaintext_fallback_used", false},
        {"reveal_count", 0},
        {"open_count", 0},
        {"secure_gate_count", secure_gate_count},
        {"conditional_edge_count", conditional_edge_count},
        {"conditional_bridge_mode", conditional_edge_count > 0 ? "inline_dense_sidecar_v1" : "off"},
        {"online_producer_invocation_count", 0},
        {"online_secret_sort_count", 0},
        {"online_stored_reverse_count", 0},
        {"record_count", payload.record_count},
        {"anchor_slots", payload.anchor_slots},
        {"edge_count", payload.edge_count},
        {"fixed_decimal_bits", payload.fixed_decimal_bits},
        {"wall_seconds", std::chrono::duration<double>(ended - started).count()},
        {"operation_counts", {
            {"equality", equality_count},
            {"mux", mux_count},
            {"multiplication", multiply_count},
            {"conversion", conversion_count},
            {"rounds", 0}
        }},
        {"result_share_file_bytes", 16},
        {"output_share_path", output_share_path}
    };
}

double decode(std::int64_t raw, std::uint32_t fixed_bits) {
    return static_cast<double>(raw) / static_cast<double>(std::uint64_t{1} << fixed_bits);
}

std::int64_t fixed_mul_raw(std::int64_t lhs, std::int64_t rhs, std::uint32_t fixed_bits) {
    const __int128 product = static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
    const __int128 shifted = product >> fixed_bits;
    require_bool(
        shifted >= static_cast<__int128>(std::numeric_limits<std::int64_t>::min()) &&
            shifted <= static_cast<__int128>(std::numeric_limits<std::int64_t>::max()),
        "fixed-block clear multiplication overflow");
    return static_cast<std::int64_t>(shifted);
}

json evaluate_clear(const std::vector<Record>& records, const json& manifest, const Query& query) {
    const auto started = std::chrono::steady_clock::now();
    const std::uint32_t anchors = manifest.at("anchor_slots").get<std::uint32_t>();
    const std::uint32_t fixed_bits = manifest.at("fixed_decimal_bits").get<std::uint32_t>();
    const double scale = static_cast<double>(std::uint64_t{1} << fixed_bits);
    std::vector<std::vector<const Record*>> by_anchor(anchors);
    for (const auto& record : records) {
        by_anchor[record.anchor_slot].push_back(&record);
    }
    std::int64_t total_raw = 0;
    std::uint64_t equality_count = 0;
    std::uint64_t mux_count = 0;
    std::uint64_t multiply_count = 0;
    std::uint64_t dummy_contribution_count = 0;
    std::vector<std::int64_t> anchor_mask = query.anchor_mask_values;
    if (anchor_mask.empty()) {
        anchor_mask.assign(anchors, static_cast<std::int64_t>(std::uint64_t{1} << fixed_bits));
    }
    require_bool(anchor_mask.size() == anchors, "bad clear anchor mask size");
    for (std::uint32_t anchor = 0; anchor < anchors; ++anchor) {
        std::int64_t anchor_weight = static_cast<std::int64_t>(std::uint64_t{1} << fixed_bits);
        std::int64_t product = static_cast<std::int64_t>(std::uint64_t{1} << fixed_bits);
        bool all_edges_present = true;
        for (const auto edge : query.required_edges) {
            bool matched = false;
            std::int64_t selected = 0;
            for (const auto* record : by_anchor[anchor]) {
                const bool occupied = (record->flags & 1u) != 0;
                const bool dummy = (record->flags & 2u) != 0;
                if (record->edge_index == edge) {
                    anchor_weight = record->weight_raw;
                }
                for (const auto& pred : query.predicates) {
                    if (pred.edge_index != edge) {
                        continue;
                    }
                    equality_count += 1;
                    const bool hit = occupied && !dummy && record->edge_index == edge && record->basis_id == pred.basis_id;
                    mux_count += 1;
                    if (hit) {
                        matched = true;
                        selected += record->value_raw;
                    }
                    if (dummy && record->value_raw != 0) {
                        dummy_contribution_count += 1;
                    }
                }
            }
            if (!matched) {
                all_edges_present = false;
                break;
            }
            product = fixed_mul_raw(product, selected, fixed_bits);
            multiply_count += 1;
        }
        if (all_edges_present) {
            const auto masked_product = fixed_mul_raw(product, anchor_mask[anchor], fixed_bits);
            total_raw += fixed_mul_raw(anchor_weight, masked_product, fixed_bits);
            multiply_count += 1;
        }
    }
    const auto ended = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(ended - started).count();
    const double total = decode(total_raw, fixed_bits);
    const std::int64_t encoded = total_raw;
    return json{
        {"mode", "cleartext_mirror"},
        {"query_id", query.query_id},
        {"result", total},
        {"encoded_result", encoded},
        {"fixed_decimal_bits", fixed_bits},
        {"anchor_slots", anchors},
        {"record_count", records.size()},
        {"wall_seconds", wall},
        {"operation_counts", {
            {"equality", equality_count},
            {"mux", mux_count},
            {"multiplication", multiply_count},
            {"conversion", 0},
            {"rounds", 0}
        }},
        {"dummy_contribution_count", dummy_contribution_count},
        {"privacy", {
            {"cleartext_fixture_only", true},
            {"reveals_secret_state", false},
            {"predicate_literals_logged", false}
        }}
    };
}

json evaluate_clear_bundle(const std::vector<Record>& records, const json& manifest, const std::vector<Query>& queries) {
    json rows = json::array();
    double max_abs_delta = 0.0;
    for (const auto& query : queries) {
        rows.push_back(evaluate_clear(records, manifest, query));
    }
    return json{
        {"mode", "cleartext_bundle"},
        {"evaluator_id", "m2_fixed_block_v2"},
        {"query_count", rows.size()},
        {"queries", rows},
        {"max_abs_delta_internal", max_abs_delta},
        {"plaintext_fallback_used", false},
        {"legacy_evaluator_used", false}
    };
}

json evaluate_secure_bundle(
    const SecurePayload& payload,
    const std::vector<SecureQuery>& queries,
    int role,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime,
    const std::string& output_share_dir) {
    json rows = json::array();
    std::uint64_t secure_gate_count = 0;
    std::uint64_t communication_count = 0;
    std::uint64_t bytes_sent_total = 0;
    std::uint64_t bytes_received_total = 0;
    for (const auto& query : queries) {
        const auto output_path = output_share_dir + "/" + query.query_id + ".role_" + std::to_string(role) + ".shares.bin";
        reset_transport_stats(runtime);
        auto row = evaluate_secure(payload, query, role, enc, eval, runtime, output_path);
        const auto comm = transport_stats(runtime);
        row["communication"] = comm;
        row["bytes_sent_total"] = comm.value("bytes_sent_total", static_cast<std::uint64_t>(0));
        row["bytes_received_total"] = comm.value("bytes_received_total", static_cast<std::uint64_t>(0));
        secure_gate_count += row.value("secure_gate_count", static_cast<std::uint64_t>(0));
        if (row.contains("operation_counts")) {
            communication_count += row.at("operation_counts").value("multiplication", static_cast<std::uint64_t>(0));
        }
        bytes_sent_total += comm.value("bytes_sent_total", static_cast<std::uint64_t>(0));
        bytes_received_total += comm.value("bytes_received_total", static_cast<std::uint64_t>(0));
        rows.push_back(row);
    }
    return json{
        {"mode", "secure_evaluate_bundle"},
        {"evaluator_id", "m2_fixed_block_v2"},
        {"query_count", rows.size()},
        {"role", role},
        {"secure_evaluation_executed", true},
        {"three_role_protocol_executed", true},
        {"client_reconstruction_executed", false},
        {"plaintext_fallback_used", false},
        {"legacy_evaluator_used", false},
        {"reveal_count", 0},
        {"open_count", 0},
        {"secure_gate_count", secure_gate_count},
        {"communication_counter_source", "cryptoTools::Channel"},
        {"bytes_sent_total", bytes_sent_total},
        {"bytes_received_total", bytes_received_total},
        {"communication_count_proxy", communication_count},
        {"queries", rows}
    };
}

}  // namespace

int M2_fixed_block_eval(const oc::CLP& cmd) {
    const int role = cmd.getOr<int>("role", -1);
    if (role < 0 || role > 2) {
        throw std::runtime_error("m2_fixed_block_eval requires --role in {0,1,2}");
    }
    const std::string manifest_path = cmd.getOr<std::string>("fixed_block_manifest_json", "");
    const std::string payload_path = cmd.getOr<std::string>("fixed_block_payload_bin", "");
    const std::string query_path = cmd.getOr<std::string>("fixed_block_query_json", "");
    const std::string query_bundle_path = cmd.getOr<std::string>("fixed_block_query_bundle_json", "");
    if (manifest_path.empty() || payload_path.empty() || (query_path.empty() && query_bundle_path.empty())) {
        throw std::runtime_error("m2_fixed_block_eval requires manifest, payload, and query paths");
    }
    const bool clear = cmd.isSet("cleartext_mirror");
    const bool secure_validate_only = cmd.isSet("secure_payload_validate_only");
    const bool secure_evaluate = cmd.isSet("secure_evaluate");
    const bool require_privacy_aligned_offline = cmd.isSet("require_privacy_aligned_offline");
    const auto manifest = read_json_file(manifest_path);
    require_bool(manifest.value("endianness", std::string()) == "little", "bad manifest endianness");
    require_bool(manifest.value("overflow_count", 0) == 0, "nonzero overflow marker");
    require_bool(manifest.value("role_count", 3) == 3, "bad role count");
    (void)require_manifest_string(manifest, "dataset_schema_hash");
    (void)require_manifest_string(manifest, "frozen_config_hash");
    (void)require_manifest_string(manifest, "sample_provenance_hash");
    (void)require_manifest_string(manifest, "multiplier_provenance_hash");
    if (require_privacy_aligned_offline) {
        require_privacy_aligned_offline_manifest(manifest);
    }
    if (secure_validate_only || secure_evaluate) {
        require_bool(manifest.value("magic", std::string()) == "M2FBLOCKSHARE02", "bad manifest share magic");
        require_bool(manifest.value("schema_version", 0) == static_cast<int>(kShareSchemaVersion), "bad manifest share schema version");
        const auto load_started = std::chrono::steady_clock::now();
        const auto payload = load_secure_share_payload(payload_path, manifest, static_cast<std::uint32_t>(role));
        const auto load_ended = std::chrono::steady_clock::now();
        if (secure_evaluate) {
            const std::string output_share_path = cmd.getOr<std::string>("fixed_block_output_share_bin", "");
            const std::string output_share_dir = cmd.getOr<std::string>("fixed_block_output_share_dir", "");
            require_bool(!output_share_path.empty() || !output_share_dir.empty(), "secure evaluation requires --fixed_block_output_share_bin or --fixed_block_output_share_dir");
            IOService ios;
            Sh3Encryptor enc;
            Sh3Evaluator eval;
            Sh3Runtime runtime;
            basic_setup(static_cast<u64>(role), ios, enc, eval, runtime);
            if (!query_bundle_path.empty()) {
                require_bool(!output_share_dir.empty(), "secure query bundle requires --fixed_block_output_share_dir");
                const auto queries = load_secure_query_bundle(query_bundle_path, payload.edge_count);
                std::cout << evaluate_secure_bundle(payload, queries, role, enc, eval, runtime, output_share_dir).dump() << std::endl;
            } else {
                const auto query = load_secure_query(query_path, payload.edge_count);
                reset_transport_stats(runtime);
                auto row = evaluate_secure(payload, query, role, enc, eval, runtime, output_share_path);
                const auto comm = transport_stats(runtime);
                row["communication"] = comm;
                row["bytes_sent_total"] = comm.value("bytes_sent_total", static_cast<std::uint64_t>(0));
                row["bytes_received_total"] = comm.value("bytes_received_total", static_cast<std::uint64_t>(0));
                std::cout << row.dump() << std::endl;
            }
            return 0;
        }
        std::cout << json{
            {"mode", "secure_payload_validate_only"},
            {"role", role},
            {"record_count", payload.record_count},
            {"anchor_slots", manifest.at("anchor_slots")},
            {"edge_count", manifest.at("edge_count")},
            {"wall_seconds", std::chrono::duration<double>(load_ended - load_started).count()},
            {"privacy", {
                {"reveals_secret_state", false},
                {"predicate_literals_logged", false}
            }}
        }.dump() << std::endl;
        return 0;
    }
    if (!clear) {
        throw std::runtime_error("m2_fixed_block_eval requires either -cleartext_mirror, -secure_payload_validate_only, or -secure_evaluate");
    }
    require_bool(!require_privacy_aligned_offline, "cleartext mirror cannot satisfy privacy-aligned offline production");
    require_bool(manifest.value("magic", std::string()) == "M2FBLOCKPAYLOAD", "bad manifest magic");
    require_bool(manifest.value("schema_version", 0) == static_cast<int>(kSchemaVersion), "bad manifest schema version");
    const auto records = load_records(payload_path, manifest, static_cast<std::uint32_t>(role));
    if (!query_bundle_path.empty()) {
        const auto queries = load_query_bundle(query_bundle_path, manifest.at("edge_count").get<std::uint32_t>());
        std::cout << evaluate_clear_bundle(records, manifest, queries).dump() << std::endl;
    } else {
        const auto query = load_query(query_path, manifest.at("edge_count").get<std::uint32_t>());
        std::cout << evaluate_clear(records, manifest, query).dump() << std::endl;
    }
    return 0;
}

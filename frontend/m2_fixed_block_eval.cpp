#include "m2_fixed_block_eval.h"

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

namespace {

constexpr const char kMagic[16] = {
    'M', '2', 'F', 'B', 'L', 'O', 'C', 'K',
    'P', 'A', 'Y', 'L', 'O', 'A', 'D', '\0'};
constexpr const char kShareMagic[16] = {
    'M', '2', 'F', 'B', 'L', 'O', 'C', 'K',
    'S', 'H', 'A', 'R', 'E', '0', '2', '\0'};
constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kShareSchemaVersion = 2;

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

std::string require_manifest_string(const json& manifest, const std::string& key) {
    require_bool(manifest.contains(key), "missing manifest field: " + key);
    require_bool(manifest.at(key).is_string(), "manifest field is not a string: " + key);
    const std::string value = manifest.at(key).get<std::string>();
    require_bool(!value.empty(), "empty manifest field: " + key);
    return value;
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
    const std::uint64_t actual_checksum = fnv1a64(bytes.data() + sizeof(Header), bytes.size() - sizeof(Header));
    require_bool(actual_checksum == header.payload_checksum, "bad payload checksum");

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

std::uint32_t validate_secure_share_payload(
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
    const std::uint64_t actual_checksum = fnv1a64(bytes.data() + sizeof(Header), bytes.size() - sizeof(Header));
    require_bool(actual_checksum == header.payload_checksum, "bad fixed-block share payload checksum");
    for (std::uint32_t idx = 0; idx < header.record_count; ++idx) {
        SecureShareRecord record = read_plain<SecureShareRecord>(bytes, offset);
        require_bool(record.anchor_slot < header.anchor_slots, "share record anchor_slot outside public shape");
        require_bool(record.edge_index < header.edge_count, "share record edge_index outside public shape");
        require_bool(record.reserved0 == 0 && record.reserved1 == 0, "share record reserved fields must be zero");
    }
    return header.record_count;
}

Query load_query(const std::string& query_path, std::uint32_t edge_count) {
    const auto q = read_json_file(query_path);
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
    return query;
}

double decode(std::int64_t raw, std::uint32_t fixed_bits) {
    return static_cast<double>(raw) / static_cast<double>(std::uint64_t{1} << fixed_bits);
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
    double total = 0.0;
    std::uint64_t equality_count = 0;
    std::uint64_t mux_count = 0;
    std::uint64_t multiply_count = 0;
    std::uint64_t dummy_contribution_count = 0;
    for (std::uint32_t anchor = 0; anchor < anchors; ++anchor) {
        double anchor_weight = 1.0;
        double product = 1.0;
        bool all_edges_present = true;
        for (const auto edge : query.required_edges) {
            bool matched = false;
            double selected = 0.0;
            for (const auto* record : by_anchor[anchor]) {
                const bool occupied = (record->flags & 1u) != 0;
                const bool dummy = (record->flags & 2u) != 0;
                if (record->edge_index == edge) {
                    anchor_weight = decode(record->weight_raw, fixed_bits);
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
                        selected = decode(record->value_raw, fixed_bits);
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
            product *= selected;
            multiply_count += 1;
        }
        if (all_edges_present) {
            total += anchor_weight * product;
            multiply_count += 1;
        }
    }
    const auto ended = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(ended - started).count();
    const std::int64_t encoded = static_cast<std::int64_t>(std::llround(total * scale));
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

}  // namespace

int M2_fixed_block_eval(const oc::CLP& cmd) {
    const int role = cmd.getOr<int>("role", -1);
    if (role < 0 || role > 2) {
        throw std::runtime_error("m2_fixed_block_eval requires --role in {0,1,2}");
    }
    const std::string manifest_path = cmd.getOr<std::string>("fixed_block_manifest_json", "");
    const std::string payload_path = cmd.getOr<std::string>("fixed_block_payload_bin", "");
    const std::string query_path = cmd.getOr<std::string>("fixed_block_query_json", "");
    if (manifest_path.empty() || payload_path.empty() || query_path.empty()) {
        throw std::runtime_error("m2_fixed_block_eval requires manifest, payload, and query paths");
    }
    const bool clear = cmd.isSet("cleartext_mirror");
    const bool secure_validate_only = cmd.isSet("secure_payload_validate_only");
    const auto manifest = read_json_file(manifest_path);
    require_bool(manifest.value("endianness", std::string()) == "little", "bad manifest endianness");
    require_bool(manifest.value("overflow_count", 0) == 0, "nonzero overflow marker");
    require_bool(manifest.value("role_count", 3) == 3, "bad role count");
    (void)require_manifest_string(manifest, "dataset_schema_hash");
    (void)require_manifest_string(manifest, "frozen_config_hash");
    (void)require_manifest_string(manifest, "sample_provenance_hash");
    (void)require_manifest_string(manifest, "multiplier_provenance_hash");
    if (secure_validate_only) {
        require_bool(manifest.value("magic", std::string()) == "M2FBLOCKSHARE02", "bad manifest share magic");
        require_bool(manifest.value("schema_version", 0) == static_cast<int>(kShareSchemaVersion), "bad manifest share schema version");
        const auto started = std::chrono::steady_clock::now();
        const auto record_count = validate_secure_share_payload(payload_path, manifest, static_cast<std::uint32_t>(role));
        const auto ended = std::chrono::steady_clock::now();
        std::cout << json{
            {"mode", "secure_payload_validate_only"},
            {"role", role},
            {"record_count", record_count},
            {"anchor_slots", manifest.at("anchor_slots")},
            {"edge_count", manifest.at("edge_count")},
            {"wall_seconds", std::chrono::duration<double>(ended - started).count()},
            {"privacy", {
                {"reveals_secret_state", false},
                {"predicate_literals_logged", false}
            }}
        }.dump() << std::endl;
        return 0;
    }
    if (!clear) {
        throw std::runtime_error("production ABY3 fixed-block evaluator is not implemented; cleartext_mirror only");
    }
    require_bool(manifest.value("magic", std::string()) == "M2FBLOCKPAYLOAD", "bad manifest magic");
    require_bool(manifest.value("schema_version", 0) == static_cast<int>(kSchemaVersion), "bad manifest schema version");
    const auto records = load_records(payload_path, manifest, static_cast<std::uint32_t>(role));
    const auto query = load_query(query_path, manifest.at("edge_count").get<std::uint32_t>());
    std::cout << evaluate_clear(records, manifest, query).dump() << std::endl;
    return 0;
}

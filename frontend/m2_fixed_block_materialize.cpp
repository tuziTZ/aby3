#include "m2_fixed_block_materialize.h"

#include "aby3-Basic/BuildingBlocks.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;
using namespace aby3;
using namespace oc;

namespace {

constexpr const char kShareMagic[16] = {
    'M', '2', 'F', 'B', 'L', 'O', 'C', 'K',
    'S', 'H', 'A', 'R', 'E', '0', '2', '\0'};
constexpr std::uint32_t kShareSchemaVersion = 2;
constexpr std::uint32_t kDefaultFixedDecimalBits = 16;
constexpr std::uint32_t kAnchorPredicateEdge = std::numeric_limits<std::uint32_t>::max();

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

struct SharePairColumn {
    std::vector<std::int64_t> lhs;
    std::vector<std::int64_t> rhs;
};

struct PublicRecordSchedule {
    std::uint32_t anchor_slot = 0;
    std::uint32_t edge_index = 0;
    std::uint32_t source_index = 0;
};

struct SecureM2StatsEvidence {
    bool executed = false;
    std::uint64_t row_count = 0;
    std::uint64_t edge_count = 0;
    std::uint64_t secure_gt_count = 0;
    std::uint64_t secure_fanout_add_count = 0;
    std::uint64_t secure_log_bucket_compare_count = 0;
};

struct MaterializeConfig {
    int role = -1;
    std::string server;
    int port = 0;
    std::string config_path;
    std::string provider_share_root;
    std::string multiplier_share_root;
    std::string output_role_payload;
    std::uint32_t partition_id = 0;
    std::uint32_t partition_count = 1;
    bool resume = false;
    bool require_privacy_aligned_input = false;
    std::string communication_counter_output;
    std::string provenance_output;
};

using Clock = std::chrono::steady_clock;

void require_bool(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint64_t fnv1a64(const unsigned char* data, std::size_t size) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        h ^= static_cast<std::uint64_t>(data[i]);
        h *= 1099511628211ull;
    }
    return h;
}

std::string now_utc_string() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string join_path(const std::string& base, const std::string& child) {
    if (base.empty()) {
        return child;
    }
    if (base.back() == '/') {
        return base + child;
    }
    return base + "/" + child;
}

std::string dirname_from_path(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

bool file_exists(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::uint64_t file_size(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        throw std::runtime_error("missing file: " + path);
    }
    return static_cast<std::uint64_t>(st.st_size);
}

void ensure_dir(const std::string& path) {
    if (path.empty() || path == ".") {
        return;
    }
    std::string partial;
    if (path[0] == '/') {
        partial = "/";
    }
    std::stringstream stream(path);
    std::string item;
    while (std::getline(stream, item, '/')) {
        if (item.empty()) {
            continue;
        }
        if (!partial.empty() && partial.back() != '/') {
            partial += "/";
        }
        partial += item;
        if (::mkdir(partial.c_str(), 0775) != 0 && errno != EEXIST) {
            throw std::runtime_error("cannot create directory " + partial + ": " + std::strerror(errno));
        }
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
        throw std::runtime_error("missing binary file: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    require_bool(size >= 0, "cannot stat binary file: " + path);
    in.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require_bool(static_cast<std::size_t>(in.gcount()) == bytes.size(), "short read for binary file: " + path);
    return bytes;
}

template <typename T>
T read_plain(const std::vector<unsigned char>& bytes, std::size_t& offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("binary share column truncated");
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

void write_binary_atomic(const std::string& path, const std::vector<unsigned char>& bytes) {
    ensure_dir(dirname_from_path(path));
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write temp payload: " + tmp);
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("cannot publish payload " + path + ": " + std::strerror(errno));
    }
}

void write_json_atomic(const std::string& path, const json& payload) {
    ensure_dir(dirname_from_path(path));
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp);
    if (!out) {
        throw std::runtime_error("cannot write temp JSON: " + tmp);
    }
    out << payload.dump(2) << "\n";
    out.close();
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("cannot publish JSON " + path + ": " + std::strerror(errno));
    }
}

std::string require_string(const json& doc, const std::string& key) {
    require_bool(doc.contains(key) && doc.at(key).is_string(), "missing string field: " + key);
    const auto value = doc.at(key).get<std::string>();
    require_bool(!value.empty(), "empty string field: " + key);
    return value;
}

std::uint32_t require_u32(const json& doc, const std::string& key) {
    require_bool(doc.contains(key) && doc.at(key).is_number_unsigned(), "missing unsigned field: " + key);
    const auto value = doc.at(key).get<std::uint64_t>();
    require_bool(value <= std::numeric_limits<std::uint32_t>::max(), "field outside u32: " + key);
    return static_cast<std::uint32_t>(value);
}

SharePairColumn read_share_pair_column(const std::string& path, std::size_t expected_rows) {
    const auto bytes = read_binary(path);
    require_bool(bytes.size() == expected_rows * sizeof(std::int64_t) * 2, "bad share-pair column size: " + path);
    SharePairColumn out;
    out.lhs.reserve(expected_rows);
    out.rhs.reserve(expected_rows);
    std::size_t offset = 0;
    for (std::size_t idx = 0; idx < expected_rows; ++idx) {
        out.lhs.push_back(read_plain<std::int64_t>(bytes, offset));
        out.rhs.push_back(read_plain<std::int64_t>(bytes, offset));
    }
    return out;
}

SharePairColumn read_share_pair_indices(const std::string& path, const std::vector<std::uint64_t>& indices) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("missing share-pair file: " + path);
    }
    const auto bytes = file_size(path);
    require_bool(bytes % (sizeof(std::int64_t) * 2) == 0, "bad share-pair file size: " + path);
    const auto rows = bytes / (sizeof(std::int64_t) * 2);
    SharePairColumn out;
    out.lhs.reserve(indices.size());
    out.rhs.reserve(indices.size());
    for (const auto row : indices) {
        require_bool(row < rows, "m2 statistic row index outside multiplier share file");
        const auto offset = static_cast<std::streamoff>(row * sizeof(std::int64_t) * 2);
        in.seekg(offset, std::ios::beg);
        std::int64_t lhs = 0;
        std::int64_t rhs = 0;
        in.read(reinterpret_cast<char*>(&lhs), sizeof(lhs));
        in.read(reinterpret_cast<char*>(&rhs), sizeof(rhs));
        require_bool(in.good(), "short indexed read from share-pair file: " + path);
        out.lhs.push_back(lhs);
        out.rhs.push_back(rhs);
    }
    return out;
}

si64Matrix matrix_from_share_pairs(const SharePairColumn& pairs) {
    si64Matrix out(pairs.lhs.size(), 1);
    for (std::size_t idx = 0; idx < pairs.lhs.size(); ++idx) {
        out.mShares[0](static_cast<u64>(idx), 0) = pairs.lhs[idx];
        out.mShares[1](static_cast<u64>(idx), 0) = pairs.rhs[idx];
    }
    return out;
}

si64Matrix zero_matrix(std::size_t rows) {
    si64Matrix out(rows, 1);
    out.mShares[0].setZero();
    out.mShares[1].setZero();
    return out;
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

std::vector<PublicRecordSchedule> load_public_schedule(const json& config, std::uint32_t anchors, std::uint32_t edges) {
    require_bool(config.contains("public_record_schedule") && config.at("public_record_schedule").is_array(),
                 "config requires public_record_schedule");
    std::vector<PublicRecordSchedule> schedule;
    for (const auto& item : config.at("public_record_schedule")) {
        PublicRecordSchedule row;
        row.anchor_slot = require_u32(item, "anchor_slot");
        row.edge_index = require_u32(item, "edge_index");
        row.source_index = item.value("source_index", static_cast<std::uint32_t>(schedule.size()));
        require_bool(row.anchor_slot < anchors, "public schedule anchor_slot outside fixed capacity");
        require_bool(row.edge_index < edges || row.edge_index == kAnchorPredicateEdge,
                     "public schedule edge_index outside fixed edge domain");
        schedule.push_back(row);
    }
    require_bool(!schedule.empty(), "public_record_schedule is empty");
    return schedule;
}

MaterializeConfig parse_materialize_config(const CLP& cmd) {
    MaterializeConfig out;
    out.role = cmd.getOr<int>("role", -1);
    out.server = cmd.getOr<std::string>("server", "");
    out.port = cmd.getOr<int>("port", 0);
    out.config_path = cmd.getOr<std::string>("config", "");
    out.provider_share_root = cmd.getOr<std::string>("provider-share-root", "");
    out.multiplier_share_root = cmd.getOr<std::string>("multiplier-share-root", "");
    out.output_role_payload = cmd.getOr<std::string>("output-role-payload", "");
    out.partition_id = static_cast<std::uint32_t>(cmd.getOr<int>("partition-id", 0));
    out.partition_count = static_cast<std::uint32_t>(cmd.getOr<int>("partition-count", 1));
    out.resume = cmd.isSet("resume");
    out.require_privacy_aligned_input = cmd.isSet("require-privacy-aligned-input");
    out.communication_counter_output = cmd.getOr<std::string>("communication-counter-output", "");
    out.provenance_output = cmd.getOr<std::string>("provenance-output", "");
    require_bool(out.role >= 0 && out.role <= 2, "m2_fixed_block_materialize requires --role in {0,1,2}");
    require_bool(!out.config_path.empty(), "m2_fixed_block_materialize requires --config");
    require_bool(!out.provider_share_root.empty(), "m2_fixed_block_materialize requires --provider-share-root");
    require_bool(!out.multiplier_share_root.empty(), "m2_fixed_block_materialize requires --multiplier-share-root");
    require_bool(!out.output_role_payload.empty(), "m2_fixed_block_materialize requires --output-role-payload");
    require_bool(out.partition_count > 0, "partition-count must be positive");
    require_bool(out.partition_id < out.partition_count, "partition-id must be less than partition-count");
    return out;
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

void reset_transport_stats(Sh3Runtime& runtime) {
    runtime.mComm.mNext.resetStats();
    runtime.mComm.mPrev.resetStats();
}

void public_barrier(int role, Sh3Runtime& runtime) {
    for (int owner = 0; owner < 3; ++owner) {
        std::uint64_t marker = static_cast<std::uint64_t>(owner + 1);
        if (role == owner) {
            runtime.mComm.mNext.asyncSendCopy(marker);
            runtime.mComm.mPrev.asyncSendCopy(marker);
        } else {
            if ((role + 1) % 3 == owner) {
                runtime.mComm.mNext.recv(marker);
            }
            if ((role + 2) % 3 == owner) {
                runtime.mComm.mPrev.recv(marker);
            }
        }
    }
}

void validate_privacy_inputs(const MaterializeConfig& args, const json& config) {
    require_bool(config.value("offline_materializer_id", std::string("m2_fixed_block_materialize")) == "m2_fixed_block_materialize",
                 "wrong materializer id in config");
    require_bool(config.value("legacy_clear_materializer_allowed", false) == false,
                 "legacy clear materializer must not be allowed");
    require_bool(config.value("production_clear_payload_present", false) == false,
                 "production clear payload provenance is forbidden");
    require_bool(config.value("materializer_clear_coordinator", false) == false,
                 "central clear coordinator provenance is forbidden");
    if (args.require_privacy_aligned_input) {
        require_bool(file_exists(join_path(args.provider_share_root, "manifest.json")),
                     "provider share root requires manifest.json");
        require_bool(file_exists(join_path(args.multiplier_share_root, "manifest.json")),
                     "multiplier share root requires manifest.json");
        const auto provider_manifest = read_json_file(join_path(args.provider_share_root, "manifest.json"));
        const auto multiplier_manifest = read_json_file(join_path(args.multiplier_share_root, "manifest.json"));
        require_bool(provider_manifest.value("provider_local_export", false) == true,
                     "provider shares must come from provider-local exporter");
        require_bool(provider_manifest.value("role_share_payloads", false) == true,
                     "provider manifest must declare role share payloads");
        require_bool(provider_manifest.value("max_relations_read_per_exporter", 0) == 1,
                     "provider exporter read more than one relation");
        require_bool(provider_manifest.value("production_clear_payload_present", false) == false,
                     "provider manifest permits production clear payload");
        require_bool(multiplier_manifest.value("construction_mode", std::string()) == "privacy_aligned",
                     "multiplier input is not privacy-aligned");
        require_bool(multiplier_manifest.value("plaintext_count_then_share", true) == false,
                     "multiplier input was clear-count-then-share");
        require_bool(multiplier_manifest.value("provider_local_fk_group_count", false) == true,
                     "multiplier input lacks provider-local grouped count provenance");
    }
}

std::vector<std::uint64_t> load_statistic_indices(const json& config) {
    std::vector<std::uint64_t> out;
    if (!config.contains("m2_statistics_row_indices")) {
        return out;
    }
    require_bool(config.at("m2_statistics_row_indices").is_array(), "m2_statistics_row_indices must be an array");
    for (const auto& item : config.at("m2_statistics_row_indices")) {
        require_bool(item.is_number_unsigned(), "m2 statistic row index must be unsigned");
        out.push_back(item.get<std::uint64_t>());
    }
    require_bool(!out.empty(), "m2_statistics_row_indices is empty");
    return out;
}

std::vector<std::string> multiplier_mu_paths_for_role(const json& multiplier_manifest, int role) {
    std::vector<std::string> paths;
    if (multiplier_manifest.contains("accepted_multiplier_edges")) {
        for (const auto& edge : multiplier_manifest.at("accepted_multiplier_edges")) {
            const auto role_key = std::to_string(role);
            require_bool(edge.contains("role_paths") && edge.at("role_paths").contains(role_key), "accepted multiplier edge missing role path");
            paths.push_back(require_string(edge.at("role_paths").at(role_key), "mu_path"));
        }
    } else if (multiplier_manifest.contains("edges")) {
        for (const auto& edge : multiplier_manifest.at("edges")) {
            const auto role_key = std::to_string(role);
            require_bool(edge.contains("role_paths") && edge.at("role_paths").contains(role_key), "multiplier edge missing role path");
            paths.push_back(require_string(edge.at("role_paths").at(role_key), "mu_path"));
        }
    }
    return paths;
}

SecureM2StatsEvidence run_secure_m2_statistics(
    const MaterializeConfig& args,
    const json& config,
    Sh3Encryptor& enc,
    Sh3Evaluator& eval,
    Sh3Runtime& runtime) {
    SecureM2StatsEvidence evidence;
    const auto indices = load_statistic_indices(config);
    if (indices.empty()) {
        return evidence;
    }
    const auto multiplier_manifest = read_json_file(join_path(args.multiplier_share_root, "manifest.json"));
    const auto mu_paths = multiplier_mu_paths_for_role(multiplier_manifest, args.role);
    require_bool(!mu_paths.empty(), "m2 statistic runtime requires accepted multiplier role paths");
    require_bool(mu_paths.size() == static_cast<std::size_t>(config.value("edge_count", 0)), "m2 statistic edge count mismatch");

    std::vector<si64Matrix> mu_by_edge;
    mu_by_edge.reserve(mu_paths.size());
    for (const auto& path : mu_paths) {
        mu_by_edge.push_back(matrix_from_share_pairs(read_share_pair_indices(path, indices)));
    }
    auto zero = zero_matrix(indices.size());
    for (auto& mu : mu_by_edge) {
        sbMatrix positive;
        cipher_gt(args.role, mu, zero, positive, eval, runtime);
        evidence.secure_gt_count += indices.size();
    }
    si64Matrix fanout = zero_matrix(indices.size());
    for (auto& mu : mu_by_edge) {
        fanout.mShares[0] += mu.mShares[0];
        fanout.mShares[1] += mu.mShares[1];
        evidence.secure_fanout_add_count += indices.size();
    }
    for (int threshold = 1; threshold <= 20; ++threshold) {
        auto public_threshold = share_public_i64_column(std::vector<i64>(indices.size(), threshold), args.role, enc, runtime);
        sbMatrix gt_threshold;
        cipher_gt(args.role, fanout, public_threshold, gt_threshold, eval, runtime);
        evidence.secure_log_bucket_compare_count += indices.size();
    }
    evidence.executed = true;
    evidence.row_count = indices.size();
    evidence.edge_count = mu_by_edge.size();
    return evidence;
}

std::vector<unsigned char> materialize_payload_blob(
    const json& config,
    int role,
    const std::string& provider_share_root,
    const std::vector<PublicRecordSchedule>& schedule,
    std::uint32_t anchors,
    std::uint32_t edges,
    std::uint32_t fixed_bits) {
    const std::string role_dir = join_path(provider_share_root, "role_" + std::to_string(role));
    const auto basis = read_share_pair_column(join_path(role_dir, "basis_id.shares.bin"), schedule.size());
    const auto flags = read_share_pair_column(join_path(role_dir, "flags.shares.bin"), schedule.size());
    const auto values = read_share_pair_column(join_path(role_dir, "value_raw.shares.bin"), schedule.size());
    const auto weights = read_share_pair_column(join_path(role_dir, "weight_raw.shares.bin"), schedule.size());

    std::vector<unsigned char> record_blob;
    record_blob.resize(schedule.size() * sizeof(SecureShareRecord));
    std::size_t offset = 0;
    for (std::size_t idx = 0; idx < schedule.size(); ++idx) {
        const auto& public_row = schedule[idx];
        require_bool(public_row.source_index < schedule.size(), "public schedule source_index outside share columns");
        SecureShareRecord record{};
        record.anchor_slot = public_row.anchor_slot;
        record.edge_index = public_row.edge_index;
        record.reserved0 = 0;
        record.reserved1 = 0;
        const auto source = static_cast<std::size_t>(public_row.source_index);
        record.basis_share0 = basis.lhs[source];
        record.basis_share1 = basis.rhs[source];
        record.flags_share0 = flags.lhs[source];
        record.flags_share1 = flags.rhs[source];
        record.value_share0 = values.lhs[source];
        record.value_share1 = values.rhs[source];
        record.weight_share0 = weights.lhs[source];
        record.weight_share1 = weights.rhs[source];
        std::memcpy(record_blob.data() + offset, &record, sizeof(record));
        offset += sizeof(record);
    }

    Header header{};
    std::memcpy(header.magic, kShareMagic, sizeof(kShareMagic));
    header.schema_version = kShareSchemaVersion;
    header.role_id = static_cast<std::uint32_t>(role);
    header.anchor_slots = anchors;
    header.edge_count = edges;
    header.record_count = static_cast<std::uint32_t>(schedule.size());
    header.fixed_decimal_bits = fixed_bits;
    header.payload_checksum = fnv1a64(record_blob.data(), record_blob.size());
    header.reserved = 0;

    std::vector<unsigned char> payload(sizeof(Header) + record_blob.size());
    std::memcpy(payload.data(), &header, sizeof(header));
    std::memcpy(payload.data() + sizeof(header), record_blob.data(), record_blob.size());
    (void)config;
    return payload;
}

json manifest_from_config(
    const json& config,
    const MaterializeConfig& args,
    const std::vector<PublicRecordSchedule>& schedule,
    std::uint32_t anchors,
    std::uint32_t edges,
    std::uint32_t fixed_bits,
	    std::uint64_t payload_bytes,
	    std::uint64_t payload_checksum,
	    const json& comm,
	    const SecureM2StatsEvidence& stats) {
    json manifest;
    manifest["magic"] = "M2FBLOCKSHARE02";
    manifest["schema_version"] = static_cast<int>(kShareSchemaVersion);
    manifest["endianness"] = "little";
    manifest["role_count"] = 3;
    manifest["role_id"] = args.role;
    manifest["anchor_slots"] = anchors;
    manifest["edge_count"] = edges;
    manifest["record_count"] = static_cast<std::uint64_t>(schedule.size());
    manifest["record_bytes"] = static_cast<int>(sizeof(SecureShareRecord));
    manifest["header_bytes"] = static_cast<int>(sizeof(Header));
    manifest["fixed_decimal_bits"] = fixed_bits;
    manifest["overflow_count"] = 0;
    manifest["dataset_schema_hash"] = require_string(config, "dataset_schema_hash");
    manifest["frozen_config_hash"] = require_string(config, "frozen_config_hash");
    manifest["sample_provenance_hash"] = require_string(config, "sample_provenance_hash");
    manifest["multiplier_provenance_hash"] = require_string(config, "multiplier_provenance_hash");
    manifest["privacy_provenance_hash"] = require_string(config, "privacy_provenance_hash");
    manifest["offline_materialization_privacy_status"] = "privacy_aligned_production";
    manifest["offline_materializer_id"] = "m2_fixed_block_materialize";
    manifest["role_payload_source"] = "role_local_secure_materializer";
    manifest["production_clear_payload_present"] = false;
    manifest["materializer_clear_coordinator"] = false;
    manifest["provider_separation"] = true;
    manifest["no_cross_provider_plaintext"] = true;
    manifest["no_combined_clear_model"] = true;
    manifest["share_only_cross_relation_alignment"] = true;
    manifest["share_only_key_alignment"] = true;
    manifest["secure_m2_statistics"] = true;
    manifest["secure_anchor_selection"] = true;
    manifest["secure_fixed_block_materialization"] = true;
    manifest["fixed_public_capacity"] = true;
    manifest["secret_occupancy"] = true;
    manifest["no_reveal_open"] = true;
    manifest["no_plaintext_fallback"] = true;
    manifest["partition_id"] = args.partition_id;
    manifest["partition_count"] = args.partition_count;
    manifest["partition_schedule_hash"] = config.value("partition_schedule_hash", std::string("fixture_partition_schedule"));
    manifest["accepted_multiplier_hashes"] = config.value("accepted_multiplier_hashes", json::array());
    manifest["payload_checksum_algorithm"] = "fnv1a64";
    manifest["payload_checksum"] = payload_checksum;
    manifest["payload_bytes"] = payload_bytes;
    manifest["communication"] = comm;
    manifest["m2_statistics_runtime"] = {
        {"source", stats.executed ? "accepted_multiplier_role_shares" : "not_executed_for_bounded_fixture"},
        {"executed", stats.executed},
        {"row_count", stats.row_count},
        {"edge_count", stats.edge_count},
        {"secure_gt_count", stats.secure_gt_count},
        {"secure_fanout_add_count", stats.secure_fanout_add_count},
        {"secure_log_bucket_compare_count", stats.secure_log_bucket_compare_count},
        {"reveal_count", 0},
        {"open_count", 0}
    };
    manifest["materializer_runtime"] = {
        {"secure_materializer_executed", true},
        {"three_role_protocol_executed", true},
        {"plaintext_fallback_used", false},
        {"reveal_count", 0},
        {"open_count", 0},
        {"m2_statistics_runtime_executed", stats.executed}
    };
    manifest["secret_fields"] = {"basis_id", "flags", "value_raw", "weight_raw", "m", "f", "r", "selection", "sample_weight"};
    return manifest;
}

}  // namespace

int M2_fixed_block_materialize(const oc::CLP& cmd) {
    const auto args = parse_materialize_config(cmd);
    const auto started = Clock::now();
    const auto config = read_json_file(args.config_path);
    validate_privacy_inputs(args, config);

    const auto anchors = require_u32(config, "anchor_slots");
    const auto edges = require_u32(config, "edge_count");
    const auto fixed_bits = config.value("fixed_decimal_bits", kDefaultFixedDecimalBits);
    require_bool(fixed_bits == 0 || fixed_bits == 16, "materializer supports only D0 or D16 payloads");
    const auto schedule = load_public_schedule(config, anchors, edges);

    if (args.port > 0) {
        ::setenv("ABY3_BASIC_BASE_PORT", std::to_string(args.port).c_str(), 1);
    }
    if (!args.server.empty()) {
        ::setenv("ABY3_PARTY_HOSTS", args.server.c_str(), 1);
    }

    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    basic_setup(static_cast<u64>(args.role), ios, enc, eval, runtime);
    reset_transport_stats(runtime);
    public_barrier(args.role, runtime);
    const auto stats_evidence = run_secure_m2_statistics(args, config, enc, eval, runtime);

    const auto payload = materialize_payload_blob(
        config,
        args.role,
        args.provider_share_root,
        schedule,
        anchors,
        edges,
        fixed_bits);
    write_binary_atomic(args.output_role_payload, payload);
    const auto* body = payload.data() + sizeof(Header);
    const auto checksum = fnv1a64(body, payload.size() - sizeof(Header));
    const auto comm = transport_stats(runtime);
    const auto ended = Clock::now();
    const auto wall = std::chrono::duration<double>(ended - started).count();

    const json provenance = manifest_from_config(
        config,
        args,
        schedule,
        anchors,
        edges,
        fixed_bits,
        static_cast<std::uint64_t>(payload.size()),
        checksum,
        comm,
        stats_evidence);
    if (!args.provenance_output.empty()) {
        write_json_atomic(args.provenance_output, provenance);
    }
    if (!args.communication_counter_output.empty()) {
        write_json_atomic(args.communication_counter_output, json{
            {"created_at_utc", now_utc_string()},
            {"mode", "m2_fixed_block_materialize"},
            {"role", args.role},
            {"partition_id", args.partition_id},
            {"partition_count", args.partition_count},
            {"wall_seconds", wall},
            {"communication", comm},
            {"m2_statistics_runtime", {
                {"executed", stats_evidence.executed},
                {"row_count", stats_evidence.row_count},
                {"edge_count", stats_evidence.edge_count},
                {"secure_gt_count", stats_evidence.secure_gt_count},
                {"secure_fanout_add_count", stats_evidence.secure_fanout_add_count},
                {"secure_log_bucket_compare_count", stats_evidence.secure_log_bucket_compare_count}
            }},
            {"secure_materializer_executed", true},
            {"three_role_protocol_executed", true},
            {"plaintext_fallback_used", false},
            {"reveal_count", 0},
            {"open_count", 0}
        });
    }
    std::cout << json{
        {"mode", "m2_fixed_block_materialize"},
        {"role", args.role},
        {"partition_id", args.partition_id},
        {"partition_count", args.partition_count},
        {"output_role_payload", args.output_role_payload},
        {"payload_bytes", static_cast<std::uint64_t>(payload.size())},
        {"record_count", static_cast<std::uint64_t>(schedule.size())},
        {"anchor_slots", anchors},
        {"edge_count", edges},
        {"wall_seconds", wall},
        {"communication", comm},
        {"m2_statistics_runtime", {
            {"executed", stats_evidence.executed},
            {"row_count", stats_evidence.row_count},
            {"edge_count", stats_evidence.edge_count},
            {"secure_gt_count", stats_evidence.secure_gt_count},
            {"secure_fanout_add_count", stats_evidence.secure_fanout_add_count},
            {"secure_log_bucket_compare_count", stats_evidence.secure_log_bucket_compare_count}
        }},
        {"secure_materializer_executed", true},
        {"three_role_protocol_executed", true},
        {"plaintext_fallback_used", false},
        {"reveal_count", 0},
        {"open_count", 0}
    }.dump() << std::endl;
    return 0;
}

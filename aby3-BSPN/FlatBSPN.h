#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>

namespace aby3 {

enum class FlatBSPNNodeType : std::uint8_t {
    SUM = 0,
    PRODUCT = 1,
    LEAF = 2,
    DUMMY = 3,
};

struct FlatBSPNNodeRecord {
    std::uint32_t node_id = 0;
    FlatBSPNNodeType node_type = FlatBSPNNodeType::SUM;
    std::uint64_t cardinality = 0;
    std::uint32_t child_begin = 0;
    std::uint32_t child_count = 0;
    std::uint32_t scope_bitmap_begin = 0;
    std::uint32_t scope_bitmap_len = 0;
    std::uint32_t bucket_begin = 0;
    std::uint32_t bucket_count = 0;
    std::uint32_t weight_begin = 0;
    std::uint32_t weight_count = 0;
    std::int32_t leaf_column_id = -1;
    std::vector<std::uint8_t> scope_mask;
};

struct FlatBSPNBucketRecord {
    std::uint32_t bucket_id = 0;
    std::uint64_t bitmap_begin = 0;
    std::uint32_t bitmap_len = 0;
    std::uint32_t value_index = 0;
    std::uint32_t lower_bound_index = 0;
    std::uint32_t upper_bound_index = 0;
};

struct FlatSecureBucketMapping {
    std::uint64_t secure_local_idx = 0;
    std::uint32_t global_bucket_id = 0;
    std::uint32_t leaf_node_id = 0;
    std::uint64_t leaf_local_bucket_idx = 0;
};

struct FlatBSPNManifest {
    std::string model_id;
    std::string table_name;
    std::vector<std::string> column_names;
    std::vector<std::string> column_types;
    std::uint64_t node_count = 0;
    std::uint64_t root_node_id = 0;
    std::uint64_t total_rows = 0;
    std::uint64_t sample_total_rows = 0;
    std::uint64_t actual_total_rows = 0;
    double sample_scale = 1.0;
    std::uint64_t scope_bitmap_bytes = 0;
    std::uint64_t children_count = 0;
    std::uint64_t bucket_count = 0;
    std::uint64_t weights_count = 0;
    std::uint64_t leaf_bitmap_bytes = 0;
    std::uint64_t leaf_bucket_width = 0;
    std::uint64_t max_leaf_bucket_width = 0;
    std::uint64_t leaf_node_count = 0;
    std::uint64_t real_bucket_count = 0;
    bool has_real_bucket_count = false;
    std::uint64_t padding_bucket_count = 0;
    std::uint64_t node_cardinality_count = 0;
    std::uint64_t node_inv_cardinality_count = 0;
    std::uint64_t payload_layout_version = 0;
    std::string bucket_padding_scope = "global";
    std::string secret_payload_dir = "secret";
    std::string secret_payload_encoding;
    bool secure_multiplier_materialized = false;
    std::string secure_share_payload_dir = "secret_shares";
    std::string secure_share_payload_layout = "consolidated_v1";
    std::string secure_share_bool_encoding = "bitpacked_pair_lsb_v1";
    std::vector<std::uint32_t> secure_multiplier_bucket_indices;
    std::vector<FlatSecureBucketMapping> secure_multiplier_bucket_mapping;
    bool has_leaf_row_values = false;
    std::vector<std::uint32_t> leaf_row_value_node_ids;
    std::vector<std::uint64_t> leaf_row_value_offsets;
    std::uint64_t leaf_row_value_total_rows = 0;
    bool has_row_weights = false;
    std::uint64_t row_weight_count = 0;
    std::string row_weight_encoding;
    bool has_leaf_bucket_weight_sums = false;
    std::uint64_t leaf_bucket_weight_sum_count = 0;
    std::string leaf_bucket_weight_sum_encoding;
    std::vector<std::uint32_t> secure_multiplier_leaf_row_value_node_ids;
    std::vector<std::uint64_t> secure_multiplier_leaf_row_value_offsets;
    std::uint64_t secure_multiplier_leaf_row_value_total_rows = 0;
};

struct FlatEvidenceInterval {
    double lower = 0.0;
    double upper = 0.0;
    bool has_lower = false;
    bool has_upper = false;
    bool open_lower = false;
    bool open_upper = false;
};

struct FlatFactorSpec {
    int factor_index = -1;
    std::string factor_kind;
    bool inverse = false;
    double public_constant_value = 0.0;
    std::vector<int> feature_scope;
    std::vector<int> relevant_scope;
    std::vector<int> feature_inverted_scope;
    std::vector<std::vector<FlatEvidenceInterval>> evidence;
    std::uint64_t public_feature_count = 0;
    std::uint64_t public_evidence_count = 0;
    std::uint64_t total_rows = 0;
    bool weighted_count_direct = false;
};

inline constexpr Decimal kFlatBSPNDecimal = D16;

struct FlatBSPNSecureContext {
    int role = -1;
    int model_owner_party = 0;
    int query_owner_party = 0;
    bool debug_reveal = false;
    bool debug_internal_reveal = false;
    oc::IOService* io_service = nullptr;
    Sh3Encryptor* enc = nullptr;
    Sh3Evaluator* eval = nullptr;
    Sh3Runtime* runtime = nullptr;

    bool has_runtime() const { return enc != nullptr && eval != nullptr && runtime != nullptr; }
};

struct FlatBSPNSecretHostPayload {
    f64Matrix<kFlatBSPNDecimal> node_cardinalities;
    f64Matrix<kFlatBSPNDecimal> node_inv_cardinalities;
    i64Matrix node_scopes;
    f64Matrix<kFlatBSPNDecimal> weights;
    f64Matrix<kFlatBSPNDecimal> bucket_values;
    f64Matrix<kFlatBSPNDecimal> bucket_lowers;
    f64Matrix<kFlatBSPNDecimal> bucket_uppers;
    f64Matrix<kFlatBSPNDecimal> leaf_row_values;
    f64Matrix<kFlatBSPNDecimal> row_weights;
    f64Matrix<kFlatBSPNDecimal> leaf_bucket_weight_sums;
    i64Matrix leaf_bitmaps;
};

struct FlatBSPNSecretSharedPayload {
    sf64Matrix<kFlatBSPNDecimal> node_cardinalities;
    sf64Matrix<kFlatBSPNDecimal> node_inv_cardinalities;
    si64Matrix node_scopes;
    sf64Matrix<kFlatBSPNDecimal> weights;
    sf64Matrix<kFlatBSPNDecimal> bucket_values;
    sf64Matrix<kFlatBSPNDecimal> bucket_lowers;
    sf64Matrix<kFlatBSPNDecimal> bucket_uppers;
    sf64Matrix<kFlatBSPNDecimal> leaf_row_values;
    sf64Matrix<kFlatBSPNDecimal> row_weights;
    bool row_weights_loaded = false;
    sf64Matrix<kFlatBSPNDecimal> leaf_bucket_weight_sums;
    bool leaf_bucket_weight_sums_loaded = false;
    std::vector<std::int64_t> leaf_row_value_offset_by_node;
    bool leaf_row_values_loaded = false;
    si64Matrix leaf_bitmaps;
    std::vector<sbMatrix> dense_bucket_bitmaps;
    bool dense_bucket_bitmaps_loaded = false;
    bool loaded = false;
};

struct FlatDenseSecretFactorBinding {
    std::string secret_factor_id;
    int factor_index = -1;
    std::uint64_t column_count = 0;
    std::vector<std::uint8_t> feature_scope;
    std::vector<std::uint8_t> relevant_scope;
    std::vector<std::uint8_t> feature_inverted_scope;
    std::vector<std::uint8_t> has_evidence;
    std::vector<std::uint64_t> interval_counts;
    std::vector<std::vector<double>> lower_bounds;
    std::vector<std::vector<double>> upper_bounds;
    std::vector<std::vector<std::uint8_t>> has_lower;
    std::vector<std::vector<std::uint8_t>> has_upper;
    std::vector<std::vector<std::uint8_t>> open_lower;
    std::vector<std::vector<std::uint8_t>> open_upper;
};

struct FlatSecureQueryPayload {
    std::string payload_version;
    std::string query_skeleton_id;
    std::string binding_layout_kind;
    std::uint64_t max_interval_count = 0;
    std::uint64_t factor_count = 0;
    std::uint64_t max_factor_column_count = 0;
    std::vector<FlatDenseSecretFactorBinding> secret_factor_bindings;
};

struct FlatSecureQueryTensorPayload {
    f64Matrix<kFlatBSPNDecimal> lower_bounds;
    f64Matrix<kFlatBSPNDecimal> upper_bounds;
    i64Matrix has_lower;
    i64Matrix has_upper;
    i64Matrix open_lower;
    i64Matrix open_upper;
    i64Matrix has_evidence;
    i64Matrix interval_counts;
    i64Matrix feature_scope;
    i64Matrix relevant_scope;
    i64Matrix feature_inverted_scope;
    i64Matrix factor_column_counts;

    sf64Matrix<kFlatBSPNDecimal> lower_bounds_shared;
    sf64Matrix<kFlatBSPNDecimal> upper_bounds_shared;
    si64Matrix has_lower_shared;
    si64Matrix has_upper_shared;
    si64Matrix open_lower_shared;
    si64Matrix open_upper_shared;
    si64Matrix has_evidence_shared;
    si64Matrix interval_counts_shared;
    si64Matrix feature_scope_shared;
    si64Matrix relevant_scope_shared;
    si64Matrix feature_inverted_scope_shared;
    si64Matrix factor_column_counts_shared;
    bool shared_loaded = false;
};

struct SecureRational {
    double numerator = 0.0;
    double denominator = 1.0;
};

using FlatRationalValue = SecureRational;

class FlatBSPNModel {
public:
    void load_public_manifest(const std::string& manifest_path);
    void load_secret_payload();
    void load_secret_payload(const FlatBSPNSecureContext& context);
    sbMatrix share_bucket_bitmap_stack(
        const std::vector<std::uint32_t>& bucket_indices,
        const FlatBSPNSecureContext& context) const;

    const FlatBSPNManifest& manifest() const { return manifest_; }
    const std::vector<FlatBSPNNodeRecord>& nodes() const { return nodes_; }
    const std::vector<std::uint32_t>& children() const { return children_; }
    const FlatBSPNSecretHostPayload& secret_host_payload() const { return secret_host_payload_; }
    const FlatBSPNSecretSharedPayload& secret_shared_payload() const { return secret_shared_payload_; }

private:
    std::string manifest_path_;
    std::string base_dir_;
    FlatBSPNManifest manifest_;
    std::vector<FlatBSPNNodeRecord> nodes_;
    std::vector<std::uint32_t> children_;
    std::vector<FlatBSPNBucketRecord> buckets_;
    std::vector<double> weights_;
    std::vector<double> bucket_values_;
    std::vector<double> node_cardinalities_;
    std::vector<double> node_inv_cardinalities_;
    std::vector<std::uint8_t> node_scopes_;
    std::vector<std::uint8_t> leaf_bitmaps_;
    std::vector<double> row_weights_;
    std::vector<double> leaf_bucket_weight_sums_;
    std::vector<int> secure_bucket_offsets_;
    std::vector<sbMatrix> secure_multiplier_bucket_bitmaps_;
    bool secure_share_payload_loaded_ = false;
    FlatBSPNSecretHostPayload secret_host_payload_;
    FlatBSPNSecretSharedPayload secret_shared_payload_;
};

FlatSecureQueryPayload load_secure_query_payload_json(const std::string& payload_json_path);
FlatSecureQueryTensorPayload build_secure_query_tensor_payload(const FlatSecureQueryPayload& payload);
FlatSecureQueryTensorPayload share_secure_query_tensor_payload(
    const FlatSecureQueryPayload& payload,
    const FlatBSPNSecureContext& context);

void BSPN_secure_bundle_eval(const oc::CLP& cmd);
void BSPN_flat_eval(const oc::CLP& cmd);

}  // namespace aby3

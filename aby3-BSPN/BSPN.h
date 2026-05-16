#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>
#include <aby3/sh3/Sh3FixedPoint.h> 

namespace aby3 {

enum class BSPNNodeType {
    SUM,
    PRODUCT,
    NUMERIC_LEAF,     // IdentityNumericLeaf
    CATEGORICAL_LEAF  // Categorical
};

struct BSPNNode {
    int id;
    BSPNNodeType type;
    std::vector<int> children_ids;
    std::vector<int> scope;

    // For Sum nodes
    std::vector<double> weights; 
    
    // For Numeric / IdentityNumeric Leaves
    // intervals: start:end pairs
    std::vector<std::pair<double, double>> intervals;

    // For Categorical Leaves
    std::vector<double> cat_values;

    // For both Leaf types
    // Corresponding bitmaps for each interval or category value
    std::vector<std::vector<u64>> bin_bitmaps;
    
    // Total number of rows (cardinality)
    long long cardinality = 0;

    BSPNNode(int id = -1, BSPNNodeType type = BSPNNodeType::SUM) 
        : id(id), type(type) {}
};

struct PlainBSPN {
    std::vector<BSPNNode> nodes;
    int root_id;

    void load_from_file(const std::string& filepath);
};

struct SecureBSPN {
    struct SecureNode {
        int id;
        BSPNNodeType type;
        std::vector<int> children_ids;
        
        // 秘密共享的权重（用于 Sum 节点）
        // 以算术分享形式存储（si64 或 sf64）
        // 我们使用一个矩阵来存储，每一行对应一个权重（或打包后的权重）。
        si64Matrix secret_weights; 

        // Secret Shared Bitmap (for Leaf nodes)
        // Multiple secret bitmaps for Numeric/Categorical Leaf nodes
        std::vector<sbMatrix> secret_bin_bitmaps; 
        
        // 这些range和类别value可以把范围限定一下，转换成密文，并且应该不能暴露LEAF节点类型，应该把cat_values转换为[value,value]这样的range
        std::vector<std::pair<double, double>> intervals; 
        std::vector<double> cat_values;
        //这个也应该是密文来着
        std::vector<int> scope;
        long long cardinality;
    };

    std::vector<SecureNode> nodes;
    int root_id;

    // Encrypt a plaintext BSPN into secret shares
    void encrypt(const PlainBSPN& plain, int pIdx, Sh3Encryptor& enc, Sh3Runtime& runtime);

    // 查询上下文：feature_scope、relevant_scope表示查询的条件限定列
    // evidence_ranges对于每个relevant列，给出一个或多个范围
    // evidence_inclusive在这里是evidence_ranges的补充，表示范围是包含还是不包含（比如年龄在[20,30)还是(20,30)）
    sf64<D16> compute_expectation(int feature_scope, 
                             const std::vector<int>& relevant_scope, 
                             const std::map<int, std::vector<double>>& evidence_ranges,
                             const std::map<int, std::vector<int>>& evidence_inclusive,
                             Sh3Encryptor& enc,
                             Sh3Evaluator& eval, 
                             Sh3Runtime& runtime);
    
private:
    struct ComputationResult {
        sf64<D16> value;      // 值（应该是小数）
        sbMatrix filter; // 一个位图，表示哪些列符合条件
        bool has_filter; // 辨别是否有效的位
    };

    //还缺一个total_rows的参数
    ComputationResult compute_recursive(int node_id,
                                      int feature_scope,
                                      const std::vector<int>& relevant_scope,
                                      const std::map<int, std::vector<double>>& evidence_ranges,
                                      const std::map<int, std::vector<int>>& evidence_inclusive,
                                      const sbMatrix* c_ids_vec,
                                      Sh3Encryptor& enc,
                                      Sh3Evaluator& eval, 
                                      Sh3Runtime& runtime);
};

} // namespace aby3

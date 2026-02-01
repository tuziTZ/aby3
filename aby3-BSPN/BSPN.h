#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>

namespace aby3 {

enum class BSPNNodeType {
    SUM,
    PRODUCT,
    LEAF, // Generic/Legacy Leaf
    NUMERIC_LEAF,
    CATEGORICAL_LEAF
};

struct BSPNNode {
    int id;
    BSPNNodeType type;
    std::vector<int> children_ids;
    std::vector<int> scope;

    // For Sum nodes
    std::vector<double> weights; 
    
    // For Numeric Leaf nodes
    std::vector<double> buckets;
    std::vector<double> bin_values;

    // For Categorical Leaf nodes
    std::vector<double> probabilities;
    std::vector<double> cat_values;

    // For Leaf nodes (Numeric & Categorical)
    // Stores the bitmap for each bin/category.
    // Each inner vector is a list of IDs (integers) representing the bitmap.
    std::vector<std::vector<int>> bin_bitmaps;

    // Legacy/Simple Leaf bitmap (single bitmap)
    // Stored as a vector of 64-bit integers (bits).
    std::vector<u64> bitmap; 

    BSPNNode(int id = -1, BSPNNodeType type = BSPNNodeType::LEAF) 
        : id(id), type(type) {}
};

struct PlainBSPN {
    std::vector<BSPNNode> nodes;
    int root_id;

    // Load from the text file format (requires pre-processed bitmaps)
    void load_from_file(const std::string& filepath);
};

struct SecureBSPN {
    // The topology is public, so we keep the structure of nodes.
    // But we replace the data (weights, bitmaps) with secret shares.
    
    struct SecureNode {
        int id;
        BSPNNodeType type;
        std::vector<int> children_ids; // Public topology
        
        // Secret Shared Weights (for Sum nodes)
        // Stored as Arithmetic Shares (si64 or sf64)
        // We use a matrix where each row corresponds to a weight (or packed).
        // Here we use a simple vector of shares for simplicity, or a 1xN matrix.
        si64Matrix secret_weights; 

        // Secret Shared Bitmap (for Leaf nodes)
        // Stored as Boolean Shares (sbMatrix)
        // Each row is a 64-bit chunk of the bitmap.
        sbMatrix secret_bitmap;
    };

    std::vector<SecureNode> nodes;
    int root_id;

    // Encrypt a plaintext BSPN into secret shares
    void encrypt(const PlainBSPN& plain, int pIdx, Sh3Encryptor& enc, Sh3Runtime& runtime);
};

} // namespace aby3

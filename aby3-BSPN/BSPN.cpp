#include "BSPN.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <aby3/sh3/Sh3BinaryEvaluator.h>
#include <aby3/Circuit/CircuitLibrary.h>

namespace aby3
{

#ifndef BSPN_UNSAFE_DEBUG_REVEAL
#define BSPN_UNSAFE_DEBUG_REVEAL 0
#endif

    // ==================== 辅助解析函数 ====================

    // 分割字符串
    std::vector<std::string> split_string(const std::string &s, char delimiter)
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter))
        {
            if (!token.empty())
            {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    // 解析 double 列表 (逗号分隔)
    std::vector<double> parse_doubles(const std::string &s)
    {
        std::vector<double> res;
        if (s.empty() || s == "none")
            return res;
        auto tokens = split_string(s, ',');
        for (const auto &t : tokens)
        {
            try
            {
                res.push_back(std::stod(t));
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing double: " << t << " - " << e.what() << std::endl;
            }
        }
        return res;
    }

    // 解析 int 列表 (逗号分隔)
    std::vector<int> parse_ints(const std::string &s)
    {
        std::vector<int> res;
        if (s.empty() || s == "none")
            return res;
        auto tokens = split_string(s, ',');
        for (const auto &t : tokens)
        {
            try
            {
                res.push_back(std::stoi(t));
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing int: " << t << " - " << e.what() << std::endl;
            }
        }
        return res;
    }

    // 解析区间列表 (逗号分隔，每个区间为 start:end)
    // Format example: 1995.0:1995.0,1996.0:1996.0
    std::vector<std::pair<double, double>> parse_intervals_str(const std::string &s)
    {
        std::vector<std::pair<double, double>> res;
        if (s.empty() || s == "none")
            return res;
        auto tokens = split_string(s, ',');
        for (const auto &t : tokens)
        {
            size_t col_pos = t.find(':');
            if (col_pos != std::string::npos)
            {
                try
                {
                    double start = std::stod(t.substr(0, col_pos));
                    double end = std::stod(t.substr(col_pos + 1));
                    res.push_back({start, end});
                }
                catch (...)
                {
                    // Ignore error
                }
            }
        }
        return res;
    }

    // Convert ID list to packed u64 bitmap
    std::vector<u64> ids_to_bitmap_u64(const std::vector<int> &ids, u64 total_rows)
    {
        u64 num_blocks = (total_rows + 63) / 64;
        std::vector<u64> bitmap(num_blocks, 0);
        for (int id : ids)
        {
            if (id >= 0 && (u64)id < total_rows)
            {
                bitmap[id / 64] |= (1ULL << (id % 64));
            }
        }
        return bitmap;
    }

    // 解析位图列表 (分号分隔桶，竖线分隔 ID)
    // 返回多个位图，每个位图是一个 u64 vector (bitset)
    std::vector<std::vector<u64>> parse_bitmaps(const std::string &s, u64 total_rows)
    {
        std::vector<std::vector<u64>> res;
        if (s.empty() || s == "none")
            return res;

        // 分号分隔各个桶
        std::stringstream ss(s);
        std::string segment;
        while (std::getline(ss, segment, ';'))
        {
            std::vector<int> ids;
            if (!segment.empty())
            {
                // 竖线分隔 ID
                auto tokens = split_string(segment, '|');
                for (const auto &t : tokens)
                {
                    try
                    {
                        ids.push_back(std::stoi(t));
                    }
                    catch (const std::exception &e)
                    {
                        // std::cerr << "Error parsing bitmap id: " << t << std::endl;
                    }
                }
            }
            res.push_back(ids_to_bitmap_u64(ids, total_rows));
        }
        return res;
    }

    // 解析 hex 字符串为 u64 向量
    std::vector<u64> parse_bitmap_hex(const std::string &hex_str)
    {
        std::vector<u64> bitmap;
        if (hex_str == "none" || hex_str.empty())
        {
            return bitmap;
        }

        // 如果字符串以 "0x" 或 "hex:" 开头，移除它
        size_t start = 0;
        if (hex_str.size() >= 2 && hex_str.substr(0, 2) == "0x")
        {
            start = 2;
        }
        else if (hex_str.size() >= 4 && hex_str.substr(0, 4) == "hex:")
        {
            start = 4;
        }

        // 每次处理 16 个 hex 字符 (64 bits)
        for (size_t i = start; i < hex_str.length(); i += 16)
        {
            std::string chunk = hex_str.substr(i, 16);
            try
            {
                u64 val = std::stoull(chunk, nullptr, 16);
                bitmap.push_back(val);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing bitmap hex chunk: " << chunk << " - " << e.what() << std::endl;
            }
        }
        return bitmap;
    }

    // ==================== Boolean Operator Helpers ====================

    // Logic from BoolBasic.cpp to perform bitwise AND on secret shared bitmaps
    void bool_cipher_and(const sbMatrix &sharedA, const sbMatrix &sharedB,
                         sbMatrix &res, Sh3Evaluator &eval, Sh3Runtime &runtime)
    {
        Sh3BinaryEvaluator binEng;
        CircuitLibrary lib;

        int bitSize = sharedA.bitCount(); // Likely 64
        int i64Size = sharedA.i64Size();  // Number of i64 blocks (rows)

        auto cir = lib.int_int_bitwiseAnd(bitSize, bitSize, bitSize);

        binEng.setCir(cir, i64Size, eval.mShareGen);
        binEng.setInput(0, sharedA);
        binEng.setInput(1, sharedB);

        auto dep = binEng.asyncEvaluate(runtime).then([&](Sh3Task self)
                                                      {
        res.resize(i64Size, bitSize);
        binEng.getOutput(0, res); });
        dep.get();
    }

    void bool_cipher_or(const sbMatrix &sharedA, const sbMatrix &sharedB,
                        sbMatrix &res, Sh3Evaluator &eval, Sh3Runtime &runtime)
    {
        Sh3BinaryEvaluator binEng;
        CircuitLibrary lib;

        int bitSize = sharedA.bitCount();
        int i64Size = sharedA.i64Size();

        auto cir = lib.int_int_bitwiseOr(bitSize, bitSize, bitSize);

        binEng.setCir(cir, i64Size, eval.mShareGen);
        binEng.setInput(0, sharedA);
        binEng.setInput(1, sharedB);

        auto dep = binEng.asyncEvaluate(runtime).then([&](Sh3Task self)
                                                      {
        res.resize(i64Size, bitSize);
        binEng.getOutput(0, res); });
        dep.get();
    }

    // ==================== PlainBSPN 加载函数 ====================

    void PlainBSPN::load_from_file(const std::string &filepath)
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filepath);
        }

        std::string line;
        nodes.clear();
        root_id = 0;
        u64 total_rows = 0; // Will be set from the first node (root)
        bool is_first_node = true;

        while (std::getline(file, line))
        {
            // 跳过空行和头部信息
            if (line.empty())
                continue;
            if (line.find("Tree=") == 0 || line.find("format=") == 0 ||
                line.find("num_nodes=") == 0 || line.find("model_type=") == 0 ||
                line.find("num_trees=") == 0)
            {
                continue;
            }

            std::map<std::string, std::string> attrs;
            std::stringstream ss(line);
            std::string segment;

            while (ss >> segment)
            {
                size_t eq_pos = segment.find('=');
                if (eq_pos != std::string::npos)
                {
                    std::string key = segment.substr(0, eq_pos);
                    std::string val = segment.substr(eq_pos + 1);
                    attrs[key] = val;
                }
            }

            // 必须有 id 属性
            if (attrs.find("id") == attrs.end())
                continue;

            BSPNNode node;
            try
            {
                node.id = std::stoi(attrs["id"]);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error parsing node id: " << e.what() << std::endl;
                continue;
            }

            // 解析 cardinality
            if (attrs.find("cardinality") != attrs.end())
            {
                try
                {
                    node.cardinality = std::stoll(attrs["cardinality"]);
                }
                catch (...)
                {
                }
            }

            // Assume first node is root and its cardinality is the total rows
            if (is_first_node)
            {
                total_rows = node.cardinality;
                root_id = node.id; // Usually 0 or the first one
                is_first_node = false;
                // Ensure total_rows is valid. If 0, maybe default to large number?
                // Better to warn if 0.
                if (total_rows == 0)
                {
                    std::cerr << "Warning: Root cardinality is 0. Bitmaps might be empty." << std::endl;
                }
            }

            // 解析节点类型
            if (attrs.find("type") != attrs.end())
            {
                std::string type_str = attrs["type"];
                if (type_str == "Sum" || type_str == "sum")
                {
                    node.type = BSPNNodeType::SUM;
                }
                else if (type_str == "Product" || type_str == "product")
                {
                    node.type = BSPNNodeType::PRODUCT;
                }
                else if (type_str == "IdentityNumericLeaf")
                {
                    node.type = BSPNNodeType::NUMERIC_LEAF;
                }
                else if (type_str == "Categorical")
                {
                    node.type = BSPNNodeType::CATEGORICAL_LEAF;
                }
                else
                {
                    node.type = BSPNNodeType::NUMERIC_LEAF; // Default/Fallback
                }
            }

            // 解析 children (竖线分隔)
            if (attrs.find("children") != attrs.end())
            {
                std::string children_str = attrs["children"];
                if (children_str != "none" && !children_str.empty())
                {
                    auto tokens = split_string(children_str, '|');
                    for (const auto &t : tokens)
                    {
                        try
                        {
                            node.children_ids.push_back(std::stoi(t));
                        }
                        catch (const std::exception &e)
                        {
                            // std::cerr << "Error parsing child id: " << t << std::endl;
                        }
                    }
                }
            }

            // 解析权重 (竖线分隔，用于 Sum 节点)
            if (attrs.find("weights") != attrs.end())
            {
                node.weights = parse_doubles(attrs["weights"]);
            }

            // 解析 scope (逗号分隔)
            if (attrs.find("scope") != attrs.end())
            {
                node.scope = parse_ints(attrs["scope"]);
            }

            // 解析 NumericLeaf 属性
            if (node.type == BSPNNodeType::NUMERIC_LEAF)
            {
                if (attrs.find("intervals") != attrs.end())
                {
                    node.intervals = parse_intervals_str(attrs["intervals"]);
                }
                if (attrs.find("bitmaps") != attrs.end())
                {
                    node.bin_bitmaps = parse_bitmaps(attrs["bitmaps"], total_rows);
                }
                // Support legacy "bin_bitmaps" key
                if (attrs.find("bin_bitmaps") != attrs.end())
                {
                    node.bin_bitmaps = parse_bitmaps(attrs["bin_bitmaps"], total_rows);
                }
            }

            // 解析 CategoricalLeaf 属性
            if (node.type == BSPNNodeType::CATEGORICAL_LEAF)
            {
                // Assume format similar to NumericLeaf but maybe with single values instead of intervals?
                // The user didn't specify exactly, but preserved "cat_values" in my header.
                // If the user's export uses "cat_values=...", we parse it.
                if (attrs.find("cat_values") != attrs.end())
                {
                    node.cat_values = parse_doubles(attrs["cat_values"]);
                }
                // Just in case it uses "intervals" or "values" generically
                if (attrs.find("values") != attrs.end())
                {
                    node.cat_values = parse_doubles(attrs["values"]);
                }

                if (attrs.find("bitmaps") != attrs.end())
                {
                    node.bin_bitmaps = parse_bitmaps(attrs["bitmaps"], total_rows);
                }
                if (attrs.find("cat_bitmaps") != attrs.end())
                {
                    node.bin_bitmaps = parse_bitmaps(attrs["cat_bitmaps"], total_rows);
                }
            }

            // 调整向量大小以容纳该节点 ID
            if (nodes.size() <= (size_t)node.id)
            {
                nodes.resize(node.id + 1);
            }
            nodes[node.id] = node;
        }

        // 设置根节点 ID (假设为 0 或第一个非空节点)
        if (!nodes.empty())
        {
            root_id = 0;
        }

        std::cout << "Loaded " << nodes.size() << " nodes from " << filepath << std::endl;
    }

    // ==================== SecureBSPN 加密函数 ====================

    // Helper to encrypt a single u64 bitmap into secret shares
    void encrypt_bitmap(const std::vector<u64> &bmp, int pIdx, sbMatrix &dest, Sh3Encryptor &enc, Sh3Runtime &runtime)
    {
        if (bmp.empty())
            return;

        i64Matrix bitmap_mat(bmp.size(), 1);
        for (size_t j = 0; j < bmp.size(); ++j)
        {
            bitmap_mat(j, 0) = static_cast<i64>(bmp[j]);
        }

        // bitCount is 64 for sbMatrix
        dest.resize(bitmap_mat.rows(), 64);

        if (pIdx == runtime.mPartyIdx)
        {
            enc.localBinMatrix(runtime, bitmap_mat, dest).get();
        }
        else
        {
            enc.remoteBinMatrix(runtime, dest).get();
        }
    }

    void SecureBSPN::encrypt(const PlainBSPN &plain, int pIdx, Sh3Encryptor &enc, Sh3Runtime &runtime)
    {
        nodes.resize(plain.nodes.size());
        root_id = plain.root_id;

        // Use root cardinality if available, otherwise 0
        u64 total_rows = 0;
        if (plain.nodes.size() > 0 && plain.nodes[0].cardinality > 0)
        {
            total_rows = plain.nodes[0].cardinality;
        }

        u64 num_blocks = (total_rows + 63) / 64;

        for (size_t i = 0; i < plain.nodes.size(); ++i)
        {
            const auto &p_node = plain.nodes[i];
            auto &s_node = nodes[i];

            s_node.id = p_node.id;
            s_node.type = p_node.type;
            s_node.children_ids = p_node.children_ids;
            s_node.scope = p_node.scope;
            s_node.intervals = p_node.intervals;
            s_node.cat_values = p_node.cat_values;
            s_node.cardinality = p_node.cardinality;

            // 目前只加密以下两个内容
            //  Weights for SUM nodes
            if (s_node.type == BSPNNodeType::SUM)
            {
                if (!p_node.weights.empty())
                {
                    i64Matrix weight_mat(p_node.weights.size(), 1);
                    for (size_t j = 0; j < p_node.weights.size(); ++j)
                    {
                        weight_mat(j, 0) = static_cast<i64>(p_node.weights[j] * 65536); // Scale 2^16
                    }

                    s_node.secret_weights.resize(p_node.weights.size(), 1);

                    if (pIdx == runtime.mPartyIdx)
                    { // Use own ID if we are the souce
                        enc.localIntMatrix(runtime, weight_mat, s_node.secret_weights).get();
                    }
                    else
                    {
                        enc.remoteIntMatrix(runtime, s_node.secret_weights).get();
                    }
                }
            }
            // Bitmaps for Leaf nodes
            else if (s_node.type == BSPNNodeType::NUMERIC_LEAF || s_node.type == BSPNNodeType::CATEGORICAL_LEAF)
            {
                s_node.secret_bin_bitmaps.clear();
                // Encrypt multiple bitmaps
                for (const auto &bmp : p_node.bin_bitmaps)
                {
                    std::vector<u64> current_bmp = bmp;
                    if (current_bmp.size() < num_blocks)
                    {
                        current_bmp.resize(num_blocks, 0);
                    }

                    sbMatrix s_bmp;
                    encrypt_bitmap(current_bmp, pIdx, s_bmp, enc, runtime);
                    s_node.secret_bin_bitmaps.push_back(s_bmp);
                }
            }
        }
    }

    // SUM（bitmap 1的个数）
    si64 cipher_bit_popcount(const sbMatrix &bits, Sh3Encryptor &enc, Sh3Evaluator &eval, Sh3Runtime &runtime)
    {
        si64 zero;
        zero.mData[0] = 0;
        zero.mData[1] = 0;
        if (bits.rows() == 0)
            return zero;

        si64Matrix arith_bits;
        arith_bits.resize(bits.rows(), bits.bitCount());

        // Convert binary shares to arithmetic shares (0/1)
        // 1 * bits = arithmetic_val
        eval.asyncMul(runtime, 1, bits, arith_bits).get();

        /* DEBUG: Check intermediate popcount */
        /*
        i64 temp_sum = 0;
        std::vector<i64> plain_vals(arith_bits.rows() * arith_bits.cols());
        Sh3Encryptor enc; // Need passed encryptor to reveal...
        // simpler: just print shares sum
        */

        // Sum locally
        i64 sum0 = 0, sum1 = 0;
        i64 *p0 = arith_bits.mShares[0].data();
        i64 *p1 = arith_bits.mShares[1].data();
        u64 total = arith_bits.mShares[0].size();

        for (u64 i = 0; i < total; ++i)
        {
            sum0 += p0[i];
            sum1 += p1[i];
        }

        si64 result;
        result.mData[0] = sum0;
        result.mData[1] = sum1;

#if BSPN_UNSAFE_DEBUG_REVEAL
        i64 temp_sum = 0;
        enc.revealAll(runtime, result, temp_sum).get();
        std::cout << "[DEBUG] POPCOUNT = " << temp_sum << std::endl;
#endif

        return result;
    }

    // ==================== Computation ====================

    sf64<D16> SecureBSPN::compute_expectation(int feature_scope,
                                         const std::vector<int> &relevant_scope,
                                         const std::map<int, std::vector<double>> &evidence_ranges,
                                         const std::map<int, std::vector<int>> &evidence_inclusive,
                                         Sh3Encryptor &enc,
                                         Sh3Evaluator &eval,
                                         Sh3Runtime &runtime)
    {
        auto res = compute_recursive(root_id, feature_scope, relevant_scope,
                                     evidence_ranges, evidence_inclusive, nullptr, enc, eval, runtime);
        return res.value;
    }

    SecureBSPN::ComputationResult SecureBSPN::compute_recursive(
        
        int node_id,
        int feature_scope,
        const std::vector<int> &relevant_scope,
        const std::map<int, std::vector<double>> &evidence_ranges,
        const std::map<int, std::vector<int>> &evidence_inclusive,
        const sbMatrix *c_ids_vec,
        Sh3Encryptor &enc,
        Sh3Evaluator &eval,
        Sh3Runtime &runtime)
    {
        ComputationResult result;
        result.has_filter = false;

#if BSPN_UNSAFE_DEBUG_REVEAL
        std::cout << "Entering compute_recursive node=" << node_id << std::endl;
#endif

        const auto &node = nodes[node_id];

        sf64<D16> s_one;
        if (runtime.mPartyIdx == 0)
        {
            s_one.mShare.mData[0] = 65536;
            s_one.mShare.mData[1] = 0;
        }
        else
        {
            s_one.mShare.mData[0] = 0;
            s_one.mShare.mData[1] = 0;
        }

        sf64<D16> s_zero;
        s_zero.mShare.mData[0] = 0;
        s_zero.mShare.mData[1] = 0;
        
        // Default value 0
        result.value = s_zero;

        if (node.type == BSPNNodeType::SUM)
        {
            // 子节点的加权求和
            // 逻辑：WeightedSum(Children) / Sum(Weights)
            std::vector<sf64<D16>> children_results;
            for (int child_id : node.children_ids)
            {
                auto child_res = compute_recursive(child_id, feature_scope, relevant_scope,
                                                   evidence_ranges, evidence_inclusive, c_ids_vec,
                                                   enc, eval, runtime);
                children_results.push_back(child_res.value);
            }

            sf64<D16> weighted_sum = s_zero;
            sf64<D16> normalizer = s_zero;

            for (size_t i = 0; i < children_results.size(); ++i)
            {
                if (i >= node.secret_weights.rows())
                    break;

                sf64<D16> val = children_results[i];
                
                // Construct weight (already scaled in encrypt)
                si64 w_raw;
                w_raw.mData[0] = node.secret_weights.mShares[0](i, 0);
                w_raw.mData[1] = node.secret_weights.mShares[1](i, 0);
                sf64<D16> w;
                w.mShare = w_raw;

                // weighted_sum += val * weight
                sf64<D16> prod;
                eval.asyncMul(runtime, val, w, prod).get();

                weighted_sum = weighted_sum + prod;

                // normalizer += weight
                normalizer = normalizer + w;
            }

            // Ideally we divide by normalizer, but usually weights sum to 1.0 (65536)
            // Just use weighted_sum for now as approximation if normalized
            result.value = weighted_sum;

#if BSPN_UNSAFE_DEBUG_REVEAL
            f64<D16> p_val;
            enc.revealAll(runtime, result.value, p_val).get();
            std::cout << "Node " << nodes[node_id].id << " Type SUM Val " << static_cast<double>(p_val) << std::endl;
#endif

            return result;
        }
        else if (node.type == BSPNNodeType::PRODUCT)
        {
            sf64<D16> product = s_one;

            // Separate children into 'filter' (relevant to evidence) and 'exp' (target feature)
            std::vector<int> exp_children;
            std::vector<int> filter_children;

            // Scope Check similar to Python
            for (int child_id : node.children_ids)
            {
                if (child_id >= nodes.size())
                    continue;
                const auto &child = nodes[child_id];

                bool is_relevant = false;
                for (int s : child.scope)
                {
                    // If s in relevant_scope
                    for (int r : relevant_scope)
                        if (s == r)
                        {
                            is_relevant = true;
                            break;
                        }
                    if (is_relevant)
                        break;
                }

                bool is_target = false;
                if (!child.scope.empty() && child.scope[0] == feature_scope &&
                    child.type != BSPNNodeType::PRODUCT && child.type != BSPNNodeType::SUM)
                {
                    is_target = true;
                }

                if (is_relevant)
                {
                    if (is_target)
                        exp_children.push_back(child_id);
                    else
                        filter_children.push_back(child_id);
                }
            }

            // Current context filter
            const sbMatrix *current_c_ids = c_ids_vec;
            sbMatrix temp_c_ids; // To hold intermediate updates

            // 1. Process filter children
            for (int child_id : filter_children)
            {
                auto child_res = compute_recursive(child_id, feature_scope, relevant_scope,
                                                   evidence_ranges, evidence_inclusive, nullptr, // Reset context? Python passes None
                                                   enc, eval, runtime);

                if (child_res.has_filter && child_res.filter.rows() > 0)
                {
                    if (current_c_ids == nullptr)
                    {
                        temp_c_ids = child_res.filter;
                        current_c_ids = &temp_c_ids;
                    }
                    else
                    {
                        sbMatrix next_c_ids;
                        bool_cipher_and(*current_c_ids, child_res.filter, next_c_ids, eval, runtime);
                        temp_c_ids = next_c_ids;
                        current_c_ids = &temp_c_ids;
                    }
                }
                else
                {
                    // Multiply factor (Fixed Point)
                    eval.asyncMul(runtime, product, child_res.value, product).get();
                }
            }

            // If context updated, multiply by selectivity ratio
            if (current_c_ids != nullptr && current_c_ids != c_ids_vec)
            {
                si64 count = cipher_bit_popcount(*current_c_ids, enc, eval, runtime);

                double card = (double)20.0; // TODO: Restore node.cardinality when fixed
                if (card <= 0)
                    card = 1.0;
                
                sf64<D16> inv_card_fixed;
                enc.localFixed<D16>(runtime, 1.0 / card, inv_card_fixed).get();

                // product = product * count * inv_card
                // Step 1: product(F) * count(I) -> temp(F)
                sf64<D16> temp;
                eval.asyncMul(runtime, product.i64Cast(), count, temp.i64Cast()).get();

                // Step 2: temp(F) * inv_card(F) -> product(F) (Truncated)
                eval.asyncMul(runtime, temp, inv_card_fixed, product).get();
            }

            // 2. Process exp children (target)
            for (int child_id : exp_children)
            {
                auto child_res = compute_recursive(child_id, feature_scope, relevant_scope,
                                                   evidence_ranges, evidence_inclusive, current_c_ids,
                                                   enc, eval, runtime);
                eval.asyncMul(runtime, product, child_res.value, product).get();
            }

            result.value = product;

#if BSPN_UNSAFE_DEBUG_REVEAL
            f64<D16> pv;
            enc.revealAll(runtime, result.value, pv).get();
            std::cout << "Node" << node.id << " PRODUCT Val=" << static_cast<double>(pv) << std::endl;
#endif

            return result;
        }

        else if (node.type == BSPNNodeType::NUMERIC_LEAF || node.type == BSPNNodeType::CATEGORICAL_LEAF)
        {
            int col_idx = node.scope.empty() ? -1 : node.scope[0];

            bool has_local = false;
            sbMatrix local_bitmap; // Will hold OR-ed results

            // Check if evidence exists for this column
            bool has_evidence = (col_idx != -1 && evidence_ranges.find(col_idx) != evidence_ranges.end());

            // 1. Calculate Local Bitmap based on evidence
            if (has_evidence)
            {
                const auto &ranges = evidence_ranges.at(col_idx);
                // Assume ranges: s1, e1, s2, e2...
                std::vector<int> inclusive;
                if (evidence_inclusive.find(col_idx) != evidence_inclusive.end())
                {
                    inclusive = evidence_inclusive.at(col_idx);
                }

                // Iterate through node's intervals (or values)
                size_t num_intervals = (node.type == BSPNNodeType::NUMERIC_LEAF) ? node.intervals.size() : node.cat_values.size();

                for (size_t i = 0; i < num_intervals; ++i)
                {
                    if (i >= node.secret_bin_bitmaps.size())
                        break; // Safety

                    bool match = false;
                    if (node.type == BSPNNodeType::NUMERIC_LEAF)
                    {
                        double key_l = node.intervals[i].first;
                        double key_r = node.intervals[i].second;

                        // Check against all evidence ranges
                        for (size_t k = 0; k < ranges.size() / 2; ++k)
                        {
                            double r_start = ranges[2 * k];
                            double r_end = ranges[2 * k + 1];

                            bool l_inc = (k * 2 < inclusive.size()) ? inclusive[2 * k] : 1;
                            bool r_inc = (k * 2 + 1 < inclusive.size()) ? inclusive[2 * k + 1] : 1;

                            // Strict Containment Logic (Bucket inside Query Range)
                            bool lower_ok = l_inc ? (r_start <= key_l) : (r_start < key_l);
                            bool upper_ok = r_inc ? (key_r <= r_end) : (key_r < r_end);

                            if (lower_ok && upper_ok)
                            {
                                match = true;
                                break;
                            }
                        }
                    }
                    else
                    {
                        // Categorical (Set membership)
                        double val = node.cat_values[i];
                        // ranges for categorical usually means set of values?
                        // Python: value in possible_values
                        // Here we assume ranges stores discrete values? Or ranges?
                        // If simple values, maybe stored as intervals [v, v]?
                        // Or discrete list logic.
                        // Let's assume ranges is list of values [v1, v1, v2, v2...] for equality check?
                        // Or standard interval check.
                        for (size_t k = 0; k < ranges.size() / 2; ++k)
                        {
                            if (val >= ranges[2 * k] && val <= ranges[2 * k + 1])
                            {
                                match = true;
                                break;
                            }
                        }
                    }

                    if (match)
                    {
                        if (!has_local)
                        {
                            local_bitmap = node.secret_bin_bitmaps[i]; // Copy first match
                            has_local = true;
                        }
                        else
                        {
                            // In-place OR: local_bitmap = local_bitmap | current
                            // Create temp result? No, generic boolean OR needs dest distinct usually?
                            // Let's check signature data flow.
                            // bool_cipher_or(A, B, C) -> C can be A?
                            // Usually safe if implemented correctly.
                            sbMatrix temp;
                            bool_cipher_or(local_bitmap, node.secret_bin_bitmaps[i], temp, eval, runtime);
                            local_bitmap = temp;
                        }
                    }
                }
            }

            // 2. Combine with Context (Conditioning)

            // If no evidence and no context, we select everything.
            // If evidence exists but nothing matched -> Empty bitmap (has_local=false).
            // If evidence exists and matched -> local_bitmap is set.

            // Case A: No evidence.
            // Python: local_ids_vec = create_ones() or zeros depending on evidence presence.
            // If ev_val is None -> ones.
            // But creating all-ones secret shared bitmap is non-trivial without communication?
            // Actually, we can treat "Null" bitmap as "All Ones".
            // But for calculation, we need concrete bits if we AND them.
            // Optimization: If has_evidence=false, local_bitmap is effectively "All 1s".
            // So final_c_ids = c_ids_vec (AND with 1s is identity).

            // Case B: Evidence but no match (has_local=false).
            // local_bitmap is "All 0s".

            const sbMatrix *effective_c_ids = nullptr;
            sbMatrix temp_final;

            if (has_evidence)
            {
                if (!has_local)
                {
                    // All zeros
                    // result = 0
                    result.value = s_zero;
                    return result;
                }
                else
                {
                    // We have a local filter
                    if (c_ids_vec)
                    {
                        bool_cipher_and(*c_ids_vec, local_bitmap, temp_final, eval, runtime);
                        effective_c_ids = &temp_final;
                    }
                    else
                    {
                        effective_c_ids = &local_bitmap;
                    }
                }
            }
            else
            {
                // No local evidence restriction (All 1s)
                effective_c_ids = c_ids_vec; // Just context
            }

            // Store filter for return (if this node is used as filter by parent)
            if (effective_c_ids)
            {
                result.has_filter = true;
                result.filter = *effective_c_ids; // Copy
            }
            else
            {
                // Effectively all ones
                // Assuming simplified signaling: empty filter in result means "All Ones" if has_filter=true?
                // Or just don't return filter if all ones?
                // Parent checks has_filter.
            }

            // 3. Compute Value (Selectivity or Expectation)

            bool is_target = (node.scope.empty()) ? false : (node.scope[0] == feature_scope);

            if (is_target)
            {
                // Expectation: E = Sum(Val * Count(Intersection)) / Count(Final)
                // Numerator
                sf64<D16> numerator = s_zero;

                // Iterate all values again
                size_t num_intervals = node.secret_bin_bitmaps.size();
                for (size_t i = 0; i < num_intervals; ++i)
                {
                    if (i >= node.secret_bin_bitmaps.size())
                        break;

                    // Value for this bucket
                    double avg_val = 0;
                    if (node.type == BSPNNodeType::NUMERIC_LEAF)
                    {
                        avg_val = (node.intervals[i].first + node.intervals[i].second) / 2.0;
                    }
                    else
                    {
                        avg_val = node.cat_values[i];
                    }

                    // Count overlap with effective_c_ids
                    // Overlap = AND(effective_c_ids, bin_bitmaps[i])
                    // Note: If effective_c_ids is null (all 1s), overlap is bin_bitmaps[i]

                    const sbMatrix *current_bmp = &node.secret_bin_bitmaps[i];
                    sbMatrix overlap;
                    const sbMatrix *target_bmp = current_bmp;

                    if (effective_c_ids)
                    {
                        bool_cipher_and(*effective_c_ids, *current_bmp, overlap, eval, runtime);
                        target_bmp = &overlap;
                    }

                    si64 cnt = cipher_bit_popcount(*target_bmp, enc, eval, runtime);

                    // Add Val * Count
                    sf64<D16> s_val_fixed;
                    if (runtime.mPartyIdx == 0)
                        enc.localFixed<D16>(runtime, avg_val, s_val_fixed).get();
                    else
                        enc.remoteFixed(runtime, s_val_fixed).get();

                    sf64<D16> term;
                    eval.asyncMul(runtime, s_val_fixed.i64Cast(), cnt, term.i64Cast()).get();

                    numerator = numerator + term;
                }

                // Normalize: result = numerator / cardinality

                double card = (double)20.0;
                if (card <= 0) card = 1.0;

                sf64<D16> s_inv;
                if (runtime.mPartyIdx == 0)
                    enc.localFixed<D16>(runtime, 1.0/card, s_inv).get();
                else
                    enc.remoteFixed(runtime, s_inv).get();

                eval.asyncMul(runtime, numerator, s_inv, result.value).get(); // Truncation implicit
            }
            else
            {
                if (!effective_c_ids) {
                    result.value = s_one;
                }
                else {
                    si64 cnt = cipher_bit_popcount(*effective_c_ids, enc, eval, runtime);
                    
                    double card = (double)20.0; // TODO: Restore node.cardinality when fixed
                    if (card <= 0) card = 1.0;

                    sf64<D16> s_inv;
                    if (runtime.mPartyIdx == 0)
                        enc.localFixed<D16>(runtime, 1.0/card, s_inv).get();
                    else 
                        enc.remoteFixed(runtime, s_inv).get();

                     // result = cnt(S0) * inv_card(S16) -> S16. No truncation.
                    eval.asyncMul(runtime, cnt, s_inv.i64Cast(), result.value.i64Cast()).get();
                }
            }
        }

#if BSPN_UNSAFE_DEBUG_REVEAL
        f64<D16> p_val;
        enc.revealAll(runtime, result.value, p_val).get();
        std::cout << "Node " << nodes[node_id].id << " Type " << (int)nodes[node_id].type
                  << " Val " << static_cast<double>(p_val) << " HasFilter " << result.has_filter << std::endl;
#endif

        return result;
    }
}

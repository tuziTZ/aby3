#include "BSPN.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <map>

namespace aby3 {

// ==================== 辅助解析函数 ====================

// 分割字符串
std::vector<std::string> split_string(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

// 解析 double 列表 (逗号分隔)
std::vector<double> parse_doubles(const std::string& s) {
    std::vector<double> res;
    if (s.empty() || s == "none") return res;
    auto tokens = split_string(s, ',');
    for (const auto& t : tokens) {
        try {
            res.push_back(std::stod(t));
        } catch (const std::exception& e) {
            std::cerr << "Error parsing double: " << t << " - " << e.what() << std::endl;
        }
    }
    return res;
}

// 解析 int 列表 (逗号分隔)
std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> res;
    if (s.empty() || s == "none") return res;
    auto tokens = split_string(s, ',');
    for (const auto& t : tokens) {
        try {
            res.push_back(std::stoi(t));
        } catch (const std::exception& e) {
            std::cerr << "Error parsing int: " << t << " - " << e.what() << std::endl;
        }
    }
    return res;
}

// 解析位图列表 (分号分隔桶，竖线分隔 ID)
// 返回多个位图，每个位图是一个 ID 列表向量
std::vector<std::vector<int>> parse_bitmaps(const std::string& s) {
    std::vector<std::vector<int>> res;
    if (s.empty() || s == "none") return res;
    
    // 分号分隔各个桶
    std::stringstream ss(s);
    std::string segment;
    while (std::getline(ss, segment, ';')) {
        std::vector<int> ids;
        if (!segment.empty()) {
            // 竖线分隔 ID
            auto tokens = split_string(segment, '|');
            for (const auto& t : tokens) {
                try {
                    ids.push_back(std::stoi(t));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing bitmap id: " << t << std::endl;
                }
            }
        }
        res.push_back(ids);
    }
    return res;
}

// 解析 hex 字符串为 u64 向量
std::vector<u64> parse_bitmap_hex(const std::string& hex_str) {
    std::vector<u64> bitmap;
    if (hex_str == "none" || hex_str.empty()) {
        return bitmap;
    }

    // 如果字符串以 "0x" 或 "hex:" 开头，移除它
    size_t start = 0;
    if (hex_str.size() >= 2 && hex_str.substr(0, 2) == "0x") {
        start = 2;
    } else if (hex_str.size() >= 4 && hex_str.substr(0, 4) == "hex:") {
        start = 4;
    }
    
    // 每次处理 16 个 hex 字符 (64 bits)
    for (size_t i = start; i < hex_str.length(); i += 16) {
        std::string chunk = hex_str.substr(i, 16);
        try {
            u64 val = std::stoull(chunk, nullptr, 16);
            bitmap.push_back(val);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing bitmap hex chunk: " << chunk << " - " << e.what() << std::endl;
        }
    }
    return bitmap;
}

// ==================== PlainBSPN 加载函数 ====================

void PlainBSPN::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    nodes.clear();
    root_id = 0;
    
    while (std::getline(file, line)) {
        // 跳过空行和头部信息
        if (line.empty()) continue;
        if (line.find("Tree=") == 0 || line.find("format=") == 0 || 
            line.find("num_nodes=") == 0 || line.find("model_type=") == 0 ||
            line.find("num_trees=") == 0) {
            continue;
        }

        // 解析 key=value 对
        std::map<std::string, std::string> attrs;
        std::stringstream ss(line);
        std::string segment;
        
        while (ss >> segment) {
            size_t eq_pos = segment.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = segment.substr(0, eq_pos);
                std::string val = segment.substr(eq_pos + 1);
                attrs[key] = val;
            }
        }

        // 必须有 id 属性
        if (attrs.find("id") == attrs.end()) continue;

        BSPNNode node;
        try {
            node.id = std::stoi(attrs["id"]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing node id: " << e.what() << std::endl;
            continue;
        }

        // 解析节点类型
        if (attrs.find("type") != attrs.end()) {
            std::string type_str = attrs["type"];
            if (type_str == "Sum" || type_str == "sum") {
                node.type = BSPNNodeType::SUM;
            } else if (type_str == "Product" || type_str == "product") {
                node.type = BSPNNodeType::PRODUCT;
            } else if (type_str == "IdentityNumericLeaf") {
                node.type = BSPNNodeType::NUMERIC_LEAF;
            } else if (type_str == "Categorical") {
                node.type = BSPNNodeType::CATEGORICAL_LEAF;
            } else {
                node.type = BSPNNodeType::LEAF;
            }
        }

        // 解析 children (竖线分隔)
        if (attrs.find("children") != attrs.end()) {
            std::string children_str = attrs["children"];
            if (children_str != "none" && !children_str.empty()) {
                auto tokens = split_string(children_str, '|');
                for (const auto& t : tokens) {
                    try {
                        node.children_ids.push_back(std::stoi(t));
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing child id: " << t << std::endl;
                    }
                }
            }
        }

        // 解析权重 (竖线分隔，用于 Sum 节点)
        if (attrs.find("weights") != attrs.end()) {
            node.weights = parse_doubles(attrs["weights"]);
        }

        // 解析 scope (逗号分隔)
        if (attrs.find("scope") != attrs.end()) {
            node.scope = parse_ints(attrs["scope"]);
        }

        // 解析 NumericLeaf 属性
        if (node.type == BSPNNodeType::NUMERIC_LEAF) {
            if (attrs.find("buckets") != attrs.end()) {
                node.buckets = parse_doubles(attrs["buckets"]);
            }
            if (attrs.find("bin_values") != attrs.end()) {
                node.bin_values = parse_doubles(attrs["bin_values"]);
            }
            if (attrs.find("bin_bitmaps") != attrs.end()) {
                node.bin_bitmaps = parse_bitmaps(attrs["bin_bitmaps"]);
            }
        }

        // 解析 CategoricalLeaf 属性
        if (node.type == BSPNNodeType::CATEGORICAL_LEAF) {
            if (attrs.find("probabilities") != attrs.end()) {
                node.probabilities = parse_doubles(attrs["probabilities"]);
            }
            if (attrs.find("cat_values") != attrs.end()) {
                node.cat_values = parse_doubles(attrs["cat_values"]);
            }
            if (attrs.find("cat_bitmaps") != attrs.end()) {
                node.bin_bitmaps = parse_bitmaps(attrs["cat_bitmaps"]);
            }
        }

        // 解析 bitmap (hex 格式或其他格式) - 兼容旧格式
        if (attrs.find("bitmap") != attrs.end()) {
            std::string bitmap_str = attrs["bitmap"];
            if (bitmap_str != "none" && !bitmap_str.empty()) {
                // 尝试解析为 hex 格式
                if (bitmap_str.find("0x") == 0 || bitmap_str.find("hex") == 0) {
                    node.bitmap = parse_bitmap_hex(bitmap_str);
                } else {
                    // 备选：解析为十进制数值列表
                    auto tokens = split_string(bitmap_str, ',');
                    for (const auto& t : tokens) {
                        try {
                            node.bitmap.push_back(std::stoull(t));
                        } catch (const std::exception& e) {
                            std::cerr << "Error parsing bitmap value: " << t << std::endl;
                        }
                    }
                }
            }
        }

        // 调整向量大小以容纳该节点 ID
        if (nodes.size() <= (size_t)node.id) {
            nodes.resize(node.id + 1);
        }
        nodes[node.id] = node;
    }
    
    // 设置根节点 ID (假设为 0 或第一个非空节点)
    if (!nodes.empty()) {
        root_id = 0;
    }

    std::cout << "Loaded " << nodes.size() << " nodes from " << filepath << std::endl;
}

// ==================== SecureBSPN 加密函数 ====================

void SecureBSPN::encrypt(const PlainBSPN& plain, int pIdx, Sh3Encryptor& enc, Sh3Runtime& runtime) {
    nodes.resize(plain.nodes.size());
    root_id = plain.root_id;

    for (size_t i = 0; i < plain.nodes.size(); ++i) {
        const auto& p_node = plain.nodes[i];
        auto& s_node = nodes[i];

        s_node.id = p_node.id;
        s_node.type = p_node.type;
        s_node.children_ids = p_node.children_ids;

        if (s_node.type == BSPNNodeType::SUM) {
            // 加密权重
            // 将 double 权重转换为定点或整数表示
            // 这里我们缩放 10000 倍并存储为 int64
            if (!p_node.weights.empty()) {
                i64Matrix weight_mat(p_node.weights.size(), 1);
                for (size_t j = 0; j < p_node.weights.size(); ++j) {
                    weight_mat(j, 0) = static_cast<i64>(p_node.weights[j] * 10000); // 缩放
                }
                
                if (pIdx == 0) {
                    enc.localIntMatrix(runtime, weight_mat, s_node.secret_weights).get();
                } else {
                    enc.remoteIntMatrix(runtime, s_node.secret_weights).get();
                }
            }
        } else if (s_node.type == BSPNNodeType::LEAF) {
            // 加密位图
            // 将 u64 向量打包为矩阵
            if (!p_node.bitmap.empty()) {
                i64Matrix bitmap_mat(p_node.bitmap.size(), 1);
                for (size_t j = 0; j < p_node.bitmap.size(); ++j) {
                    bitmap_mat(j, 0) = static_cast<i64>(p_node.bitmap[j]);
                }
                
                // 使用二进制分享进行位图存储 (sbMatrix)
                if (pIdx == 0) {
                    enc.localBinMatrix(runtime, bitmap_mat, s_node.secret_bitmap).get();
                } else {
                    enc.remoteBinMatrix(runtime, s_node.secret_bitmap).get();
                }
            }
        }
    }
}

} // namespace aby3

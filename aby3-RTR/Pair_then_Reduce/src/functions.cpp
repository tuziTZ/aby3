#include "../include/datatype.h" 
#include "../include/tasks.h" 
#include "../include/functions.h"
#include "../include/gas_functions.h"
#include "cmath"
#include <memory>  


bool load_graph_data(const std::string& graph_folder, std::vector<node>* nodes, std::vector<edge>* edges) {

    // 构建节点数据文件路径
    std::string nodes_data_path = graph_folder + "nodes.txt";
    // 打开节点数据文件
    std::ifstream nodes_file(nodes_data_path);
    if (!nodes_file.is_open()) {
        std::cerr << "Failed to open nodes data file." << std::endl;
        return false;
    }

    // 构建边数据文件路径
    std::string edges_data_path = graph_folder + "edges.txt";
    // 打开边数据文件
    std::ifstream edges_file(edges_data_path);
    if (!edges_file.is_open()) {
        std::cerr << "Failed to open edges data file." << std::endl;
        return false;
    }

    // 读取节点数据
    int id, out_deg;
    double pr;
    while (nodes_file >> id >> pr >> out_deg) {
        node item = {id, pr, out_deg};
        nodes->push_back(item);
    }

    int start, end;
    double data;
    // 读取边数据
    while (edges_file >> id >> start >> end >> data) {
        edge item = {id, start, end, data};
        edges->push_back(item);
    }

    // 关闭文件
    nodes_file.close();
    edges_file.close();

    return true;
}


bool partial_node_load(const std::string& nodes_file, std::vector<node>& nodes, const int begin, const int end){

    // 打开节点数据文件
    std::ifstream nodes_stream(nodes_file);
    if (!nodes_stream.is_open()) {
        std::cerr << "Failed to open nodes data file." << std::endl;
        return false;
    }

    // distributed load data.
    std::string line;
    int currentLine = 0;

    // load nodes info.
    while(std::getline(nodes_stream, line)){
        currentLine ++;
        if(currentLine < begin) continue;
        else if(currentLine > end) break;

        std::istringstream iss(line);
        node item;
        if(iss >> item.id >> item.pr >> item.out_deg){
            nodes.push_back(item);
        }
        else{
            std::cerr << "Error loading nodes" << std::endl;
            return false;
        }
    }
    nodes_stream.close();

    return true;
}


bool partial_edge_load(const std::string& edges_file, std::vector<edge>& edges, const int begin, const int end){

    // 打开节点数据文件
    std::ifstream edges_stream(edges_file);
    if (!edges_stream.is_open()) {
        std::cerr << "Failed to open nodes data file." << std::endl;
        return false;
    }

    // distributed load data.
    std::string line;
    int currentLine = 0;

    // load nodes info.
    while(std::getline(edges_stream, line)){
        if(currentLine < begin) continue;
        else if(currentLine >= end) break;

        std::istringstream iss(line);
        edge item;
        if(iss >> item.id >> item.start >> item.end >> item.data){
            edges.push_back(item);
        }
        else{
            std::cerr << "Error loading edges" << std::endl;
            return false;
        }
        currentLine ++;
    }
    edges_stream.close();

    return true;
}


void oblivious_index(std::vector<float>& inputX, std::vector<int>& indices, std::vector<float>& res){
    auto ptrTask = new SecretIndex<int, int, float, float>((int)TASKS, (size_t)OPTIMAL_BLOCK);
    size_t n = indices.size(), m = inputX.size();
    int* range_index = new int[m];
    for(int i=0; i<m; i++) range_index[i] = i;
    ptrTask->set_default_value(0.0);
    ptrTask->circuit_construct({n}, {m});
    ptrTask->set_selective_value(inputX.data(), 0);
    ptrTask->circuit_evaluate(indices.data(), range_index, inputX.data(), res.data());
}


void oblivious_search(std::vector<float>& inputX, std::vector<float>& bins, std::vector<float>& tarVals, std::vector<float>& res){
    auto ptrTask = new Search<float, float, float, float>((int)TASKS, (size_t) OPTIMAL_BLOCK);
    ptrTask->circuit_construct({inputX.size()}, {bins.size()});
    ptrTask->set_default_value(0.0);
    std::adjacent_difference(tarVals.begin(), tarVals.end(), tarVals.begin());
    ptrTask->set_selective_value(tarVals.data(), 0);
    ptrTask->circuit_evaluate(inputX.data(), bins.data(), tarVals.data(), res.data());
}


int page_rank_single(std::vector<node>* vec_nodes, std::vector<edge>* vec_edges, int tasks, int block_size){

    size_t v = vec_nodes->size(); size_t e = vec_edges->size();
    for(int i=0; i<v; i++) cout << (*vec_nodes)[i].out_deg << " ";
    cout << endl;

    // func def
    auto prScatter = new PRScatter<edge, node, double, double>(1, block_size);
    auto prGather = new PRGather<node, edge, double, double>(1, block_size);
    // run page rank
    prScatter->circuit_construct({e}, {v});
    prGather->circuit_construct({v}, {e});

    vector<double> edges_data(e, 0.0);
    prScatter->circuit_evaluate((*vec_edges).data(), (*vec_nodes).data(), nullptr, edges_data.data());
    for(int i=0; i<e; i++) (*vec_edges)[i].data = edges_data[i];
    for(int i=0; i<e; i++) cout << edges_data[i] << " ";
    cout << endl;

    vector<double> nodes_data(v, 0.0);
    prGather->circuit_evaluate((*vec_nodes).data(), (*vec_edges).data(), nullptr, nodes_data.data()); 
    for(int i=0; i<v; i++) (*vec_nodes)[i].pr = nodes_data[i];

    // cout nodes pr value
    cout << "PR Value" << endl;
    for(int i=0; i<v; i++){
        cout << "id: " << (*vec_nodes)[i].id << " value: " << (*vec_nodes)[i].pr << endl;
    }

    return 0;
}
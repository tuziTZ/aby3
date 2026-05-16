#pragma once
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>
#include "GASFunction.h"
#include "debug.h"
#include "./Pair_then_Reduce/include/gas_functions.h"


// int environment_setup(oc::CLP& cmd);
int get_value(string key, oc::CLP& cmd, int& val);
int get_value(string key, oc::CLP& cmd, string& val);
int get_value(string key, oc::CLP& cmd, size_t& val);

int get_mean_and_std(const std::vector<double>& time_list, double& mean, double& std_var);
int dict_to_file(const std::map<string, double>& time_dict, std::string file_path);

template <aby3::Decimal D>
bool load_nodes_data(int pIdx, aby3::Sh3Encryptor &enc, aby3::Sh3Runtime &runtime, const std::string& nodes_file, std::vector<sfnode<D>>& nodes, const int begin, const int end){
    std::vector<node> plain_nodes;
    partial_node_load(nodes_file, plain_nodes, begin, end);
    // cout << "successfully loads data " << " nodes size = " << plain_nodes.size() << endl;
    int nodes_num = plain_nodes.size();
    std::vector<double> pr_list(nodes_num, 0.0); 
    std::vector<double> reci_out_deg_list(nodes_num, 0.0);

    // std::cout << "reci list: " << std::endl;
    for(int i=0; i<nodes_num; i++){
        pr_list[i] = plain_nodes[i].pr; reci_out_deg_list[i] = (1 / (double) (plain_nodes[i].out_deg + (plain_nodes[i].out_deg == 0)));
        // std::cout << pr_list[i] << " ";
        // std::cout << reci_out_deg_list[i] << " ";
    }
    // std::cout << std::endl;

    // encrypt all the data.
    std::vector<aby3::sf64<D>> spr_vec, sreci_out_deg_vec;
    vector_to_cipher_vector<D>(pIdx, enc, runtime, pr_list, spr_vec);
    vector_to_cipher_vector<D>(pIdx, enc, runtime, reci_out_deg_list, sreci_out_deg_vec);


    // debug_output_vector<D>(spr_vec, runtime, enc);
    // debug_output_vector<D>(sreci_out_deg_vec, runtime, enc);

    for(int i=0; i<nodes_num; i++){
        sfnode<D> item = {plain_nodes[i].id, spr_vec[i], sreci_out_deg_vec[i]};
        nodes.push_back(item);
    }

    return true;
}

template <aby3::Decimal D>
bool load_edges_data(int pIdx, aby3::Sh3Encryptor &enc, aby3::Sh3Runtime &runtime, const std::string& edges_file, std::vector<sfedge<D>>& edges, const int begin, const int end){
    std::vector<edge> plain_edges;
    partial_edge_load(edges_file, plain_edges, begin, end);
    int edges_num = plain_edges.size();
    // std::cout << "edges num: " << edges_num << std::endl;

    std::vector<int> starts_list(edges_num, 0);
    std::vector<int> ends_list(edges_num, 0);
    std::vector<double> data_list(edges_num, 0.0);

    for(int i=0; i<edges_num; i++){
        starts_list[i] = plain_edges[i].start; 
        ends_list[i] = plain_edges[i].end;
        data_list[i] = plain_edges[i].data;
    }

    // encrypt all the data.
    std::vector<aby3::si64> starts_vec, ends_vec;
    std::vector<aby3::sf64<D>> data_vec;
    vector_to_cipher_vector(pIdx, enc, runtime, starts_list, starts_vec);
    vector_to_cipher_vector(pIdx, enc, runtime, ends_list, ends_vec);
    vector_to_cipher_vector<D>(pIdx, enc, runtime, data_list, data_vec);

    for(int i=0; i<edges_num; i++){
        sfedge<D> item = {plain_edges[i].id, starts_vec[i], ends_vec[i], data_vec[i]};
        edges.push_back(item);
    }

    return true;
}

// test page rank using generated data in data_folder.
int test_page_rank(oc::CLP& cmd, const std::string& data_folder);
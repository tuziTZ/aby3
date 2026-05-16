#ifndef GAS_FUNCTIONS_H  
#define GAS_FUNCTIONS_H 

#include "datatype.h"
#include "tasks.h"

#define OPTIMAL_BLOCK 10
#define TASKS 8

struct node {
    int id; 
    double pr;
    int out_deg;
};

struct edge {
    int id, start, end;
    double data;
};



bool partial_edge_load(const std::string& edges_file, std::vector<edge>& edges, const int begin, const int end);

bool partial_node_load(const std::string& nodes_file, std::vector<node>& nodes, const int begin, const int end);

bool load_graph_data(const std::string& graph_folder, std::vector<node>* nodes, std::vector<edge>* edges);


// template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
// class SubPRScatter : public SubTask<NUMX, NUMY, NUMT, NUMR>{
//     public:
//         SubPRScatter(const size_t optimal_block, const int task_id):
//             SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
//             this->have_selective = false;
//         }
//     protected:
//         virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
//             for(int i=0; i < binfo->block_len; i++){
//                 local_table[i] = 0.85 * (expandX[i].start == expandY[i].id) * (expandY[i].pr / (expandY[i].out_deg + (double) (expandY[i].out_deg == 0)));
//             }
//             return;
//         }

//     public:
//         virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
//             for(int i=0; i<resLeft.size(); i++) update_res[i] = resLeft[i] + resRight[i];
//         return;
//     }
// };


// template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
// class SubPRGather : public SubTask<NUMX, NUMY, NUMT, NUMR>{
//     public:
//         SubPRGather(const size_t optimal_block, const int task_id):
//             SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
//             this->have_selective = false;
//         }
//     protected:
//         virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
//             for(int i=0; i<binfo->block_len; i++){
//                 local_table[i] = (expandX[i].id == expandY[i].end) * (expandY[i].data);
//             }
//             return;
//         }

//     public:
//         virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
//             for(int i=0; i<resLeft.size(); i++) update_res[i] = (resLeft[i] + resRight[i]);
//         return;
//     }
// };

// template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
// class PRScatter : public PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>{

// public:
//     PRScatter(int tasks, size_t optimal_block):
//         PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>(tasks, optimal_block){
//             this->default_value = 0.0;
//             this->have_selective = false;
//     }
// };

// template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
// class PRGather : public PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>{

// public:
//     // using PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>::PTRTask;
//     PRGather(int tasks, size_t optimal_block):
//         PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>(tasks, optimal_block){
//         this->default_value = 0.0;
//         this->have_selective = false;
//     }
// };


// template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
// class MPIPRGather : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>{

// public:
//     // using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>::MPIPTRTask;
//     MPIPRGather(int tasks, size_t optimal_block):
//         MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>(tasks, optimal_block){
//             this->default_value = 0.0;
//             this->have_selective = false;
//         }
// };


// template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
// class MPIPRScatter : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>{

// public:
//     MPIPRScatter(int tasks, size_t optimal_block):
//         MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>(tasks, optimal_block){
//             this->default_value = 0.0;
//             this->have_selective = false;
//         }
// };



int page_rank_single(std::vector<node>* vec_nodes, std::vector<edge>* vec_edges, int tasks, int block_size);

void test_page_rank(const std::string& graph_folder, int edge_size, int node_size, int task_num, int block_size);

#endif
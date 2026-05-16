#include <iostream>  
#include "../include/datatype.h" 
#include "../include/tasks.h" 
#include "../include/functions.h"
#include "../include/gas_functions.h"
#include <mpi.h>
#include <unistd.h>  
#include <netdb.h> 

using namespace std;
#define MPI
// #define MULTI_STEP
// #define HISTOGRAM
#define GRAPH_FOLDER "./data/page_rank/"


int main(int argc, char** argv){

    // data loading
    // vector<node> vec_nodes;
    // vector<edge> vec_edges;
    // load_graph_data(GRAPH_FOLDER, &vec_nodes, &vec_edges);
    // page_rank_single(&vec_nodes, &vec_edges, 1, 50);

    MPI_Init(&argc, &argv);

    int rank, size;  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    test_page_rank(GRAPH_FOLDER, 100, 25, size, 100);

    MPI_Finalize();

    return 0;
}


#ifdef HISTOGRAM
int main(int argc, char** argv) {

    #ifdef MPI
    // Initialize MPI environment
    MPI_Init(&argc, &argv);  
    int rank, size;  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. construct the data.
    size_t n = 10, m = 8888;

    // MPI task construction.
    auto mpiTask = new NewMPIHistogram<float, float, float, float>((int)size, (size_t)OPTIMAL_BLOCK);
    mpiTask->set_default_value(0.0);
    mpiTask->set_lookahead(1, 1);
    mpiTask->circuit_construct({(size_t) n}, {(size_t) m});

    // data construction
    vector<float> bins(n);
    vector<float> res(n);
    vector<float> real_res(n, 0);

    size_t gap = m / n;

    for(int i=0; i<n; i++) bins[i] = i*gap;
    for(int j=0; j<m; j++){
        for(int i=0; i<n-1; i++){
            if(j >= bins[i] && j < bins[i+1]) real_res[i] += 1;
        }
    }

    size_t expand_m = (mpiTask->m_end >= m-1) ? mpiTask->m_end - mpiTask->m_start + 1 : 
    mpiTask->m_end - mpiTask->m_start + 1;
    vector<float> tarVals(expand_m);
    for(int i=0; i<expand_m; i++){
        tarVals[i] = i+mpiTask->m_start;
    }

    mpiTask->circuit_evaluate(bins.data(), tarVals.data(), nullptr, res.data());

    if(rank == 0){
        cout << "run test HISTOGRAM" << endl;
        for(int i=0; i<n; i++){
            if(res[i] != real_res[i]){
                cout << "Error: res_" << i << " = " << res[i] << " != real_res_" << i << " = " << real_res[i] << endl;
            }
        }
        cout << "mean subtask time: " << mpiTask->mean_subTask << endl;
        cout << "var subtask time: " << mpiTask->var_subTask << endl;
        cout << "combine time: " << mpiTask->time_combine << endl;
    }

    #endif

    return 0;
}
#endif


// #ifndef HISTOGRAM
// #ifdef MULTI_STEP
// int main(int argc, char** argv) {

//     #ifdef MPI
//     // Initialize MPI environment
//     MPI_Init(&argc, &argv);  
//     int rank, size;  
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // size_t n = 5, m = 999;
//     // test_multi_data_sharing(n, m);

//     // test search.
//     // 1. construct the data.
//     size_t n = 20, m = 9991;

//     // MPI task construction.
//     auto mpiTask = new NewMPISearch<float, float, float, float>((int)size, (size_t)OPTIMAL_BLOCK);
//     mpiTask->set_default_value(0.0);
//     mpiTask->set_lookahead(1);
//     mpiTask->circuit_construct({(size_t) n}, {(size_t) m});

//     // data construction
//     vector<float> inputX(n);
//     vector<float> res(n);
//     vector<float> real_res(n);

//     for(int i=0; i<n; i++){
//         inputX[i] = i+2;
//     }
//     for(int i=0; i<n; i++){
//         real_res[i] = (i > (m - 3)) ? m-1+0.5 : i+2+0.5;
//     }
    
//     // cout << "before constructing data M, m_end = " << mpiTask->m_end << endl;
//     size_t expand_m = (mpiTask->m_end >= m-1) ? mpiTask->m_end - mpiTask->m_start + 1 : 
//     mpiTask->m_end - mpiTask->m_start + 1 + mpiTask->lookahead;
//     // cout << "expand_m: " << expand_m << endl;
//     vector<float> bins(expand_m);
//     vector<float> tarVals(expand_m);
//     for(int i=0; i<expand_m; i++){
//         bins[i] = i+mpiTask->m_start;
//         tarVals[i] = i+mpiTask->m_start + 0.5;
//     }

//     // cout << "before circuit evaluate" << endl;
//     mpiTask->set_selective_value(tarVals.data(), 0);
//     mpiTask->circuit_evaluate(inputX.data(), bins.data(), tarVals.data(), res.data());

//     if(rank == 0){
//         for(int i=0; i<n; i++){
//             if(res[i] != real_res[i]){
//                 cout << "Error: res_" << i << " = " << res[i] << " != real_res_" << i << " = " << real_res[i] << endl;
//             }
//         }
//     }
//     MPI_Finalize();
//     #endif

//     #ifndef MPI
//     int n = 100, m = 8;
//     // test_fake_repeat();
//     test_oblivious_index(n, m);
//     test_oblivious_search(n, m);
//     #endif
    
//     return 0;
// }
 
// #endif

// #ifndef MULTI_STEP
// int main(int argc, char** argv) {

//     #ifdef MPI
//     // Initialize MPI environment
//     MPI_Init(&argc, &argv);  

//     // Get current process rank and size  
//     int rank, size;  
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     cout << "rank - " << rank << endl;

//     // prepare data
//     int n = 1000, m = 1;

//     auto mpiTask = new MPISecretIndex<int, int, float, float>((int)TASKS, (size_t)OPTIMAL_BLOCK);
//     mpiTask->set_default_value(0.0);
//     mpiTask->circuit_construct({(size_t) m}, {(size_t) n});

//     size_t m_start = mpiTask->m_start, m_end = mpiTask->m_end;

//     // cout << "rank - " << rank << " start: " << m_start << " end: " << m_end << endl;
    
//     vector<float> inputX(m_end - m_start + 1);
//     vector<int> indices(m);
//     vector<float> res(m);

//     for(int i=m_start; i<m_end+1; i++) inputX[i-m_start] = i;
//     for(int i=0; i<m; i++) indices[i] = m-1-i;
//     for(int i=0; i<m; i++) res[i] = 0.0;
//     int* range_index = new int[m_end - m_start + 1];
//     for(int i=m_start; i<m_end+1; i++) range_index[i-m_start] = i;

//     mpiTask->set_selective_value(inputX.data(), 0);
//     mpiTask->circuit_evaluate(indices.data(), range_index, inputX.data(), res.data());

//     if(rank == 0){

//         vector<float> fullX(n);
//         for(int i=0; i<n; i++) fullX[i] = i;

//         // check function.
//         for(int i=0; i<m; i++){
//             if(res[i] != fullX[indices[i]]){
//                 cout << "Error: res_" << i << " = " << res[i] << " != inputX[indices_" << i << "] = " << fullX[indices[i]] << endl;
//             }
//         }
//         cout << "Oblivious index CHECK SUCCESS !" << endl;
//     }

//     MPI_Finalize();

//     #endif

//     #ifndef MPI
//     int n = 100, m = 8;
//     // test_fake_repeat();
//     test_oblivious_index(n, m);
//     test_oblivious_search(n, m);
//     #endif
    
//     return 0;
// }
// #endif
// #endif
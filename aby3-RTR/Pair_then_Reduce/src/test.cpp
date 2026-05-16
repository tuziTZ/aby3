#include "../include/datatype.h"  
#include "../include/tasks.h"
#include "../include/functions.h"
#include "../include/gas_functions.h"

std::chrono::duration<double, std::micro> fake_array_total_time(0);
std::chrono::duration<double, std::micro> other_time(0);
std::chrono::duration<double, std::micro> comp_time(0);
std::chrono::duration<double, std::micro> comp_process_time(0);
std::chrono::duration<double, std::micro> comp_comm_time(0);


using namespace std;

void test_page_rank(const std::string& graph_folder, int edge_size, int node_size, int task_num, int block_size){

    // Get current process rank and size  
    int rank, size;  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != task_num){
        std::cerr << "task_num " << task_num << " not equal with the MPI size " << size << " , exit!" << std::endl;
        return;
    }

    // data files.
    std::string edges_file = graph_folder + "/edges.txt";
    std::string nodes_file = graph_folder + "/nodes.txt";

    // init the tasks.
    auto mpiGather = new MPIPRGather<node, edge, double, double>(task_num, block_size);
    auto mpiScatter = new MPIPRScatter<edge, node, double, double>(task_num, block_size);

    // circuit construct.
    mpiGather->circuit_construct({(size_t) node_size}, {(size_t) edge_size});
    mpiScatter->circuit_construct({(size_t) edge_size}, {(size_t) node_size});

    // each task have to load the whole edges and nodes for the iterative functions.
    vector<edge> edges; vector<node> nodes;
    partial_node_load(nodes_file, nodes, 0, node_size);
    partial_edge_load(edges_file, edges, 0, edge_size);

    if(edges.size() != (edge_size)){
        std::cerr << "sizes of the loaded edges is " << edges.size() << " while the task requires " << (edge_size) << std::endl;
        return;
    }

    // run the scatter phase, result is the scattered edges data.
    vector<double> edges_data(edge_size, 0.0);
    mpiScatter->circuit_evaluate(edges.data(), nodes.data() + mpiScatter->m_start, nullptr, edges_data.data());

    // barrier in the gather phase and distribute the data.
    MPI_Barrier(MPI_COMM_WORLD);
    double* edges_data_ptr = edges_data.data();
    mpiScatter->data_sharing<double>(edges_data_ptr, edge_size, 0);
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0; i<edge_size; i++){
        edges[i].data = edges_data[i];
    }

    //run the gather phase.
    vector<double> nodes_data(node_size, 0.0);
    mpiGather->circuit_evaluate(nodes.data(), edges.data() + mpiGather->m_start, nullptr, nodes_data.data());
    for(int i=0; i<node_size; i++) nodes[i].pr = nodes_data[i];

    // check the pr value.
    if(rank == 0){
        for(int i=0; i<node_size; i++){
            cout << "id: " << nodes[i].id << " value: " <<  nodes[i].pr << endl;
        }
    }
    return;
}


void test_multi_data_sharing(int n, int m){

    // Get current process rank and size  
    int rank, size;  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> tarIndex(n);
    vector<float> numX;
    for(int i = 0; i<n; i++) 
        tarIndex[i] = i;

    if(rank == 0){
        numX.resize(m);
        for(int i=0; i<m; i++){
            numX[i] = i+5;
        }
    }

    auto mpiIndex = new MPISecretIndex<int, int, float, float>((int)size, (size_t)OPTIMAL_BLOCK);
    mpiIndex->circuit_construct({(size_t) n}, {(size_t) m});

    // construct the range_index
    vector<int> range_index(mpiIndex->m_end - mpiIndex->m_start + 1);
    for(int i=mpiIndex->m_start; i<mpiIndex->m_end + 1; i ++){
        range_index[i - mpiIndex->m_start] = i;
    }   

    // datasharing by rank0.
    MPI_Barrier(MPI_COMM_WORLD);
    size_t sharing_length;
    if(rank != 0){
        sharing_length = (mpiIndex->m_end - mpiIndex->m_start + 1);
        numX.resize(sharing_length);
    }
    else{
        sharing_length = m;
    }
    float* numX_ptr = numX.data();
    mpiIndex->data_sharing<float>(numX_ptr, sharing_length, 1);
    numX.resize(mpiIndex->m_end - mpiIndex->m_start + 1);
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        cout << "data_size: " << numX.size() << endl;
    }

    // check data sharing.
    for(int i=0; i<size; i++){
        if(rank == i){
            cout << "rank: " << rank << endl;
            MPI_Barrier(MPI_COMM_WORLD);
            for(int j=mpiIndex->m_start; j < mpiIndex->m_end; j++){
                if(numX[j-mpiIndex->m_start] != (j+5)){
                    cout << "rank" << i << " | error | target: " << j+5 << " | data: " << numX[j-mpiIndex->m_start] << endl;
                }
            }
        }
    }
    MPI_Finalize();
    return;
}


void test_oblivious_index(int n, int m){
    vector<float> inputX(n);
    vector<int> indices(m);
    vector<float> res(m);

    for(int i=0; i<n; i++) inputX[i] = i;
    for(int i=0; i<m; i++) indices[i] = m-1-i;
    for(int i=0; i<m; i++) res[i] = 0.0;

    oblivious_index(inputX, indices, res);
    
    // check function.
    for(int i=0; i<m; i++){
        if(res[i] != inputX[indices[i]]){
            cout << "Error: res_" << i << " = " << res[i] << " != inputX[indices_" << i << "] = " << inputX[indices[i]] << endl;
        }
    }
    cout << "Oblivious index CHECK SUCCESS !" << endl;
    
    return;   
}


void test_oblivious_search(int n, int m){
    vector<float> inputX(n);
    vector<float> bins(m);
    vector<float> tarVals(m);
    vector<float> res(n);
    vector<float> real_res(n);

    for(int i=0; i<n; i++){
        inputX[i] = i+2;
    }
    for(int i=0; i<m; i++){
        bins[i] = i;
        tarVals[i] = i+0.5;
    }
    for(int i=0; i<n; i++){
        real_res[i] = (i > (m - 3)) ? tarVals[m-1] : tarVals[i+2];
    }

    oblivious_search(inputX, bins, tarVals, res);
    
    // check function.
    for(int i=0; i<n; i++){
        if(res[i] != real_res[i]){
            cout << "Error: res_" << i << " = " << res[i] << " != real_res_" << i << " = " << real_res[i] << endl;
        }
    }
    cout << "Oblivious search CHECK SUCCESS !" << endl;
}


void test_fake_repeat(){
    const int size = 100;
    int* data = new int[size];
    for(int i=0; i<size; i++) data[i] = i;
    const vector<size_t> data_shape = {5, 20};
    FakeArray<int> data_expand(data, data_shape, 3, 1);

    // test the indexing
    for(int i=0; i<5; i++){
        for(int j=0; j<3; j++){
            for(int k=0; k<20; k++){
                if(data[i*20+k] != data_expand[i*(3*20) + j*20 + k]){
                    cout << "not equal! " << data[i*20+k] << " != " << data_expand[i*(3*20) + j*20 + k] << endl;
                }
            }
        }
    }
    cout << "CHECK SUCCESS!" << endl;


    const int psize = 10;
    int* pdata = new int[psize];
    int start_point = 20;
    for(int i=start_point; i<psize; i++) pdata[i] = i;

    FakeArray<int> pdata_expand(pdata, {100}, 5, 0, start_point, psize);
    for(int i=0; i<5; i++){
        for(int j=start_point; j<psize; j++){
            if(pdata_expand[i*100+j] != pdata[j]){
                cout << "not equal! " << pdata[j] << " != " << pdata_expand[i*100+j] << endl;
            }
        }
    }
    cout << "PARTIAL CHECK SUCCESS!" << endl;

}
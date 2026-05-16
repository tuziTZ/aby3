#include "./pGraph.h"

int main(int argc, char** argv){
    
    if(argc < 3){
        std::cout << "Usage: " << argv[0] << " <meta_data_file> <edge_block_file> (optional)<node_chunk_file>" << std::endl;
        return 1;
    }

    std::string meta_data_file = argv[1];
    std::string edge_block_file = argv[2];
    plainGraph2d graph;

    if(argc == 4){
        std::string node_chunk_file = argv[3];
        graph = plainGraph2d(meta_data_file, edge_block_file, node_chunk_file);
    }else{
        graph = plainGraph2d(meta_data_file, edge_block_file);
    }

    // print the graph.
    // graph.printGraphMeta();
    graph.printEdgeList();

    return 0;
}
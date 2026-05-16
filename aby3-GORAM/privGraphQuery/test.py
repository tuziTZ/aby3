from utils import *

def get_hashed_graph(graph):
    if(type(graph) == nx.Graph):
        return networkx_graph_hashing(graph)
    elif(type(graph) == Graph):
        return large_graph_hashing(graph)
    else:
        raise ValueError("Graph type %s not supported."%(type(graph)))
    

def test_graph_hashing(n = 10, tool = "nx", gtype = "random"):
    # graph = nx_graph_generation("complete", n)
    if(tool == "nx"):
        graph = nx_graph_generation(gtype, n)
        hash_list = [hash(str(node)) for node in graph.nodes()]
        arg_list = np.argsort(hash_list)
        mapping_list = {node: arg_list[i] for i, node in enumerate(graph.nodes())}
    elif(tool == "igraph"):
        graph = large_graph_generation(gtype, n)
        hash_list = [hash(str(node.index)) for node in graph.vs]
        arg_list = np.argsort(hash_list)
        mapping_list = {node.index: arg_list[i] for i, node in enumerate(graph.vs)}
    else:
        raise ValueError("Tool %s not supported."%(tool))
    
    print(mapping_list)
    if(type(graph) == Graph):
        for node in graph.vs:
            print(f"{node.index} - {graph.neighbors(node)}")
    elif(type(graph) == nx.Graph):
        for node in graph:
            print(f"{node} - {graph[node]}")
    
    print(" ==== hashed graph ===")
    HG = get_hashed_graph(graph)
    
    if(type(graph) == Graph):
        for node in HG.vs:
            print(f"{node.index} - {HG.neighbors(node)}")
    else:
        for node in HG:
            print(f"{node} - {HG[node]}")

    
    if(type(graph) == nx.Graph):
        for node in graph:
            hash_node = mapping_list[node]
            edge_num = len(graph[node])
            hash_edge_num = len(HG[hash_node])
            assert edge_num == hash_edge_num
            
    elif(type(graph) == Graph):
        
        for node in graph.vs:
            hash_node = mapping_list[node.index]
            edge_num = len(graph.neighbors(node))
            hash_edge_num = len(HG.neighbors(hash_node))
            assert edge_num == hash_edge_num
    
    print("\033[92mGraph %s hashing test passed!\033[0m"%(tool))
    

if __name__ == "__main__":
    print("networkx graph hashing test")
    test_graph_hashing(10, "nx")
    print("\n\nigraph graph hashing test")
    test_graph_hashing(10, "igraph")
import networkx as nx
import numpy as np 
import json
import pickle
import math
import random
from igraph import Graph

# random.seed(16)
PROB = 0.05

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def get_k(V, E, B=1024):
    B = max(B, V)
    k = int(np.ceil(B*V/E))
    k = 2**math.floor(math.log2(k))
    
    return k

def power_law_generation(n, p = PROB):
    m = int(n * p)    
    return nx.powerlaw_cluster_graph(n, m, p)

def random_graph(n, p = PROB):
    return nx.gnp_random_graph(n, p)

def bipartite_graph(n, m = -1, p = PROB):
    if(m < 0): m = n
    return nx.bipartite.random_graph(n, m, p)

def scale_free(n, m=-1, p=PROB):
    if(m<0): m = int(n*p)
    return nx.scale_free_graph(n, m, p) 

def star_graph(n):
    n -= 1
    return nx.star_graph(n)

def large_complete_graph(n):
    g = Graph.Full(n)
    return g

def large_star_graph(n):
    g = Graph.Star(n)
    return g

def large_powerlaw_graph(n):
    """
    https://www.geeksforgeeks.org/barabasi-albert-graph-scale-free-models/
    """
    m = int(n * PROB)
    g = Graph.Barabasi(n, m)
    return g

def large_random_graph(n):
    """
    Erdos_Renyi(n, p) will generate a graph from the so-called model where each edge between any two pair of nodes has an independent probability p of existing.
    """
    g = Graph.Erdos_Renyi(n, PROB)
    return g

def large_bipartite_graph(n):
    n1_nodes = n//2
    n2_nodes = n - n1_nodes
    g = Graph.Random_Bipartite(n1_nodes, n2_nodes, PROB)
    return g

def large_tree_graph(n):
    g = Graph.Tree_Game(n, directed=True)
    return g

def large_grg_graph(n):
    radius = (1 / PROB) * (1 / np.log(n)) * 0.1
    g = Graph.GRG(n, radius)
    return g.as_directed()

def large_k_regular(n):
    k = 15
    g = Graph.K_Regular(n, k)
    return g

graph_generation_configs = {
    "complete": nx.complete_graph, 
    "star": star_graph,
    "powerlaw": power_law_generation,
    "random": random_graph,
    "bipartite": bipartite_graph,
    "scale_free": scale_free,
}

large_graph_generation_configs = {
    "complete": large_complete_graph,
    "star": large_star_graph,
    "powerlaw": large_powerlaw_graph,
    "random": large_random_graph,
    "bipartite": large_bipartite_graph,
    "tree": large_tree_graph,
    "geometric": large_grg_graph,
    "k_regular": large_k_regular,
}

def nx_graph_generation(graph_type, n):
    graph = graph_generation_configs[graph_type](n)
    print("Graph nodes = %d edges = %d"%(graph.number_of_nodes(), graph.number_of_edges()))
    return graph


def large_graph_generation(graph_type, n):
    graph = large_graph_generation_configs[graph_type](n)
    print("Graph nodes = %d edges = %d"%(len(graph.vs), graph.ecount()))
    return graph


def graph_hashing(graph):
    """Hashing the graph into a unique string.
    Then map back to the successive integer field.
    """
    if(type(graph) == nx.Graph):
        return networkx_graph_hashing(graph)
    elif(type(graph) == Graph):
        return large_graph_hashing(graph)
    else:
        raise ValueError("Graph type %s not supported."%(type(graph)))


def networkx_graph_hashing(graph):
    random_seed = random.randint(0, 100)
    hash_list = [hash(str(node) + str(random_seed)) for node in graph.nodes()]
    arg_list = np.argsort(hash_list)
    mapping_list = {node: arg_list[i] for i, node in enumerate(graph.nodes())}
    HG = nx.relabel_nodes(graph, mapping_list)
    
    HGD = nx.to_dict_of_lists(HG)
    HGD = sorted(HGD.items())
    HGD = {k:sorted(v) for k,v in HGD}

    HG = nx.from_dict_of_lists(HGD) 

    return HG


def large_graph_hashing(graph):
    """The permutation is in inverse order: https://github.com/igraph/igraph/issues/1930
    """
    random_seed = random.randint(0, 100)
    hash_list = [hash(str(node.index) + str(random_seed)) for node in graph.vs]
    arg_list = np.argsort(hash_list)
    # mapping_list = {node.index: arg_list[i] for i, node in enumerate(graph.vs)}
    graph = graph.permute_vertices(arg_list)
    
    return graph
    

def trans2_2dpartition(graph, k):
    if(type(graph) == nx.Graph):
        return trans2_2dpartition_networkx(graph, k)
    elif(type(graph) == Graph):
        return trans2_2dpartition_igraph(graph, k)
    else:
        raise ValueError("Graph type %s not supported."%(type(graph)))


def trans2_2dpartition_networkx(graph, k):
    """_summary_

    Args:
        graph (list or nx.Graph): graph represented in list.
        k (int): partition size.
    """
    V = len(graph.nodes())
    b = math.ceil(V/k)
    
    partition = {i:[] for i in range(b**2)}
    for edge in graph.edges():
        u, v = edge
        
        # 50% trans the u->v edge to v->u.
        if(np.random.rand() > 0.5):
            u, v = v, u
        
        partition[(u//k) * b + (v//k)].append((u, v))
    
    return partition


def trans2_2dpartition_igraph(graph, k):
    """_summary_

    Args:
        graph (list or nx.Graph): graph represented in list.
        k (int): partition size.
    """
    V = len(graph.vs)
    b = math.ceil(V/k)
    
    partition = {i:[] for i in range(b**2)}
    for edge in graph.get_edgelist():
        u, v = edge
        # 50% trans the u->v edge to v->u.
        if(np.random.rand() > 0.5):
            u, v = v, u
        partition[(u//k) * b + (v//k)].append((u, v))
    
    return partition


def partition_format(partition):
    """fulfill the edges to the 2d partition, making all the edge blocks the same size.
    """
    max_size = max([len(partition[i]) for i in partition.keys()])
    min_size = min([len(partition[i]) for i in partition.keys()])
    mean_size = sum([len(partition[i]) for i in partition.keys()]) / len(partition.keys())
    
    for i in partition.keys():
        partition[i] = partition[i] + [(0, 0)]*(max_size - len(partition[i]))
    
    partition_list = [item for sublist in [partition[i] for i in partition.keys()] for item in sublist]
    
    utilization_dict = {
        "min": min_size / max_size,
        "mean": mean_size / max_size
    }
    
    return partition_list, max_size, utilization_dict


def get_meta_data(graph, partition_size, l):
    if(type(graph) == nx.Graph):
        return get_meta_data_networkx(graph, partition_size, l)
    elif(type(graph) == Graph):
        return get_meta_data_igraph(graph, partition_size, l)
    else:
        raise ValueError("Graph type %s not supported."%(type(graph)))
   
    
def get_meta_data_networkx(graph, partition_size, l):
    V = graph.number_of_nodes()
    E = graph.number_of_edges()
    b = math.ceil(V/partition_size)
    return {"v": V, "e": E, "b": b, "k": partition_size, "l": l}


def get_meta_data_igraph(graph, partition_size, l):
    V = len(graph.vs)
    E = graph.ecount()
    b = V//partition_size
    return {"v": V, "e": E, "b": b, "k": partition_size, "l": l}


def pad_partition_multi_barl(partition, bar_l):
    """Pad the partition into the multiple of bar_l.
    """
    max_size = max([len(partition[i]) for i in partition.keys()])
    min_size = min([len(partition[i]) for i in partition.keys()])
    mean_size = sum([len(partition[i]) for i in partition.keys()]) / len(partition.keys())
    
    # the target size now is the multiple of bar_l.
    target_size = ((max_size + bar_l - 1) // bar_l) * bar_l
    
    for i in partition.keys():
        partition[i] = partition[i] + [(0, 0)]*(target_size - len(partition[i]))
        
    partition_list = [item for sublist in [partition[i] for i in partition.keys()] for item in sublist]
    
    # update the utilization dict.
    utilization_dict = {
        "min": min_size / target_size,
        "mean": mean_size / target_size
    }
    
    return partition_list, target_size, utilization_dict


def graph_save(graph, file_prefix, saving_type = "edgelist", partition_size = -1, hashing_flag = False, bar_l=-1):
    """Saving the generate graph into the datafile.
    """
    supporting_type = ["edgelist", "2dpartition"]
    
    if(hashing_flag):
        print("Hashing the graph.")
        graph = graph_hashing(graph)
    
    if(saving_type not in supporting_type):
        saving_type = "edgelist"
    
    if(saving_type == "edgelist"):
        if(type(graph) == nx.Graph):
            gdf = nx.to_pandas_edgelist(graph)
            gdf.to_csv(file_prefix + "_edge_list.csv", index=False)
        else:
            edge_list = graph.get_edgelist()
            with open(file_prefix + "_edge_list.txt", "w") as f:
                for edge in edge_list:
                    f.write(str(edge[0]) + " " + str(edge[1]) + "\n")
        
        meta_file = file_prefix + "_edge_list_meta.txt"
        with open(meta_file, "w") as f:
            f.write(str(graph.vcount()) + " " + str(graph.ecount()) + "\n")

    print("Graph generated!")
    
    if(saving_type == "2dpartition"):
        # Save the graph into the 2d partition format
        if(partition_size < 0):
            # raise ValueError("Partition size should be positive.")
            V, E = graph.vcount(), graph.ecount()   
            partition_size = get_k(V, E)
            print(f"computed k = {partition_size} | E = {E}")
        
        partition = trans2_2dpartition(graph, partition_size)
        
        if(bar_l < 0):
            partition_list, l, utilization_dict = partition_format(partition)
        else:
            print("Pad the partition into the multiple of bar_l.")
            partition_list, l, utilization_dict = pad_partition_multi_barl(partition, bar_l)
        
        meta_data = get_meta_data(graph, partition_size, l)
        
        partition_file = file_prefix + "_2dpartition.txt"
        meta_file = file_prefix + "_meta.txt"
        utilization_file = file_prefix + "_utilization.txt"
        
        with open(partition_file, "w") as f:
            for i in partition_list:
                f.write(str(i[0]) + " " + str(i[1]) + "\n")
        
        # in meta file, we save the {node number}, {origional edge number}, {node block number}, {node block size} and the {edge block size} in sequence.
        with open(meta_file, "w") as f:
            for key in meta_data.keys():
                f.write(str(meta_data[key]) + " ")
        
        # in utilization file, we save the utilization_dict info.
        with open(utilization_file, "w") as f:
            f.write(json.dumps(utilization_dict))
        
    return


def split_list_into_sublists(edge_list, n):
    sub_lists = [[] for _ in range(n)]
    for edge in edge_list:
        random.choice(sub_lists).append(edge)
    return sub_lists
     
def split_partition_into_subpartitions(partition, n):
    partition_size = len(partition.keys())
    partition_dict_list = [{i:[] for i in range(partition_size)} for _ in range(n)]

    for i in partition.keys():
        for edge in partition[i]:
            random.choice(partition_dict_list)[i].append(edge)

    return partition_dict_list


def graph_save_multi(graph, file_prefix, data_providers, saving_type = "edgelist", partition_size = -1, hashing_flag = False, bar_l=-1):

    supporting_type = ["edgelist", "2dpartition"]

    print(f"saving_type = {saving_type}")

    # whole graph format transmission.
    if(hashing_flag):
        print("Hashing the graph.")
        graph = graph_hashing(graph)
    
    if(saving_type not in supporting_type):
        saving_type = "edgelist"
    
    if(saving_type == "edgelist"):
        if(type(graph) == nx.Graph):
            raise ValueError("Networkx graph does not support multi data provider.")
        else:
            edge_list = graph.get_edgelist()
            sub_lists = split_list_into_sublists(edge_list, data_providers)

            for i in range(data_providers):
                with open(file_prefix + "_edge_list_party-" + str(i) + ".txt", "w") as f:
                    for edge in sub_lists[i]:
                        f.write(str(edge[0]) + " " + str(edge[1]) + "\n")
                
                with open(file_prefix + "_edge_list_meta_party-" + str(i) + ".txt", "w") as f:
                    f.write(str(graph.vcount()) + " " + str(len(sub_lists[i])) + "\n")
            
            meta_file = file_prefix + "_edge_list_meta_multiparty.txt"
            with open(meta_file, "w") as f:
                f.write(str(graph.vcount()) + " " + str(data_providers) + " ")

                for i in range(data_providers):
                    f.write(str(len(sub_lists[i])) + " ")
                f.write("\n")
        
        print("Multi-party Edge-list Graph saved!")
        return

    if(saving_type == "2dpartition"):
        if(partition_size < 0):
            raise ValueError("Partition size should be positive.")
        
        partition = trans2_2dpartition(graph, partition_size)
        partition_dict_list = split_partition_into_subpartitions(partition, data_providers)
        # split the partition_list into multiple sublists.
        b = len(graph.vs) // partition_size
        k = partition_size

        print(f"configs b = {b} | k = {k}")
        
        total_data_list = []

        for i in range(data_providers):
            
            if(bar_l < 0):
                partition_list, l, utilization_dict = partition_format(partition_dict_list[i])
                meta_data = get_meta_data(graph, partition_size, l)
                assert meta_data["b"] == b
                assert meta_data["k"] == k
                total_data_list.append((partition_list, meta_data, utilization_dict))
            else:
                print("Pad the partition into the multiple of bar_l.")
                partition_list, l, utilization_dict = pad_partition_multi_barl(partition_dict_list[i], bar_l)
            
                meta_data = get_meta_data(graph, partition_size, l)
                assert meta_data["b"] == b
                assert meta_data["k"] == k
            
                num_sublists = l // bar_l
                 
                for j in range(num_sublists):
                    sub_partition_list = []
                    for par in range(b**2):
                        sub_par = partition_list[par*l + j*bar_l : par*l + (j+1)*bar_l]
                        sub_partition_list.extend(sub_par)
                    meta_data = get_meta_data(graph, partition_size, bar_l)
                    total_data_list.append((sub_partition_list, meta_data, utilization_dict))


        print(">>>>>>> total_data_list size = ", len(total_data_list))
        # save the partition data.
        for i in range(len(total_data_list)):
            partition_list, meta_data, utilization_dict = total_data_list[i]
            
            partition_file = file_prefix + "_2dpartition_party-" + str(i) + ".txt"
            meta_file = file_prefix + "_meta_party-" + str(i) + ".txt"
        
            with open(partition_file, "w") as f:
                for i in partition_list:
                    f.write(str(i[0]) + " " + str(i[1]) + "\n")
        
            # in meta file, we save the {node number}, {origional edge number}, {node block number}, {node block size} and the {edge block size} in sequence.
            with open(meta_file, "w") as f:
                for key in meta_data.keys():
                    f.write(str(meta_data[key]) + " ")
    
        meta_file = file_prefix + "_meta_multiparty.txt"
        with open(meta_file, "w") as f:
            f.write(str(graph.vcount()) + " " + str(len(total_data_list)) + " " + str(b) + " " + str(k) + " ")                
            f.write("\n")
        
        utilization_file = file_prefix + "_utilization.txt"
        with open(utilization_file, "w") as f:
            f.write(json.dumps(utilization_dict))
        
        print("Multi-party 2d-partition Graph saved!")

        return
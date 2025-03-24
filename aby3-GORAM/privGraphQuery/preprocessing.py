from igraph import Graph
import os
import argparse
from utils import *

ABY3_FOLDER = os.getcwd()
MAIN_FOLDER = ABY3_FOLDER + "/aby3-GORAM"
real_world_data_folder = MAIN_FOLDER + "/data/real_world/"
slash_dot_file = real_world_data_folder + "soc-Slashdot0902.txt"
dblp_file = real_world_data_folder + "com-dblp.all.cmty.txt"
twitter_file = real_world_data_folder + "twitter-2010.txt"

configurations = {
    "slashdot": {"k": -1},
    "dblp": {"k": -1},
    "twitter": {"k": -1}
}

# used for multi-servers.
# party1 = "aby31"
# party2 = "aby32"
# party_list = [party1, party2]

def edge_list2igraph(edge_list_file):
    """Convert the edge list file into the igraph object.
    """    
    g = Graph.Read_Edgelist(edge_list_file)
    V = 2**math.ceil(math.log(g.vcount(), 2))
    extra_vertices = V - g.vcount()

    if(extra_vertices > 0):
        g.add_vertices(extra_vertices)

    return g

def get_edge_list(edge_list_file):
    edge_list = np.loadtxt(edge_list_file, dtype=int)
    print("edge_list size = ", edge_list.shape)
    return edge_list

def generate_edge_list(dataset):
    
    edge_list = []
    
    if(dataset == "slashdot"):
        igraph_format_file = real_world_data_folder + "slash_dot_igraph.txt"
        if(os.path.exists(igraph_format_file)):
            return igraph_format_file
        
        with open(slash_dot_file, "r") as f:
            for line in f:
                if(line[0] != "#"):
                    edge_str = line.strip().split("\t")
                    start, end = int(edge_str[0]), int(edge_str[-1])
                    edge_list.append((start, end))
        
        with open(igraph_format_file, "w") as f:
            for e in edge_list:
                f.write(str(e[0]) + " " + str(e[1]) + "\n")
        
        return igraph_format_file
        
    elif(dataset == "dblp"):
        igraph_format_file = real_world_data_folder + "dblp_igraph.txt"
        if(os.path.exists(igraph_format_file)):
            return igraph_format_file
        with open(dblp_file, "r") as f:
            for line in f:
                nodes = line.split("\t")
                source = int(nodes[0])
                for i in range(1, len(nodes)):
                    end = int(nodes[i])
                    edge_list.append((source, end))
                end = int(nodes[-1][:-1])
                edge_list.append((source, end))
        
        with open(igraph_format_file, "w") as f:
            for e in edge_list:
                f.write(str(e[0]) + " " + str(e[1]) + "\n")
        return igraph_format_file
    elif(dataset == "twitter"):
        return twitter_file
        
    else:
        raise ValueError("Dataset %s not supported."%(dataset))


def get_vertex_set(edge_list):
    vertex_set = set()
    for edge in edge_list:
        vertex_set.add(edge[0])
        vertex_set.add(edge[1])
    return vertex_set

def get_vertex_sequence_id(vertex_set):
    vertex_seq_map = {}
    for i, v in enumerate(vertex_set):
        vertex_seq_map[v] = i
    return vertex_seq_map

def get_vertex_hash_map(vertex_seq_map):
    vertex_hash_map = {}
    vertex_set_size = len(vertex_seq_map)

    hash_list = [hash(str(i)) for i in range(vertex_set_size)]
    arg_list = np.argsort(hash_list)

    for v, i in vertex_seq_map.items():
        vertex_hash_map[v] = arg_list[i]
    
    return vertex_hash_map

def edge_list_hash(edge_list, vertex_hash_map):
    edge_hash_list = []
    for edge in edge_list:
        start, end = edge
        edge_hash_list.append((vertex_hash_map[start], vertex_hash_map[end]))
    return edge_hash_list

def generate_2d_partition(vectex_set, edge_list, k):
    V = len(vectex_set) 
    # V = int(2**math.ceil(math.log2(V)))
    
    if(k == -1):
        k = get_k(V, len(edge_list))

    # adjust b and k.
    # b = math.ceil(V/k)
    b = int(2**math.ceil(math.log2(V//k)))

    partition = {i:[] for i in range(b**2)}
    for edge in edge_list:
        u, v = edge
        # 50% trans the u->v edge to v->u.
        partition[(u//k)*b + v//k].append((u, v))
    
    return partition, b, k

def twitter_data_organization(twitter_edge_list, k):

    # Step 1: Get the vertex set.
    vertex_set = get_vertex_set(twitter_edge_list)

    # Step 2: Hash the vertex set.
    print("Vertex set size: %d\nHashing vertices"%(len(vertex_set)))
    vertex_seq_map = get_vertex_sequence_id(vertex_set)
    vertex_hash_map = get_vertex_hash_map(vertex_seq_map)

    # Step 3: Hash the edge list.
    print("Hashing edges.")
    edge_hash_list = edge_list_hash(twitter_edge_list, vertex_hash_map)
    vertex_set = list(vertex_hash_map.values())

    # Step 4: Generate 2d partition.
    print("Generating 2d partition.")
    partition_data, b, k = generate_2d_partition(vertex_set, edge_hash_list, k)
    partition_list, l, utilization_dict = partition_format(partition_data)
    
    V, E = len(vertex_set), len(edge_hash_list)
    meta_dict = {"v": V, "e": E, "b": b, "k": k, "l": l}  
    

    return partition_list, meta_dict, utilization_dict, edge_hash_list



if __name__ == "__main__":
    target_dataset = ["slashdot", "dblp", "twitter"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default="privGraph", help="target graph format")
    
    args = parser.parse_args()
    dataset = args.target

    # for dataset in target_dataset:
    #     print("Start dataset %s"%(dataset))
        # k = configurations[dataset]["k"]
    k = -1
    edge_list_file = generate_edge_list(dataset)
    
    if(dataset == "twitter"):
        print("Start graph loading.")
        twitter_edge_list = get_edge_list(edge_list_file)

        print("Generating 2d-partitioning...")
        partition_list, meta_dict, utilization_dict, edge_hash_list = twitter_data_organization(twitter_edge_list, k)
        
        # save edgelist.
        print("Saving edge list.")
        with open(real_world_data_folder + dataset + "_edge_list.txt", "w") as f:
            for e in edge_hash_list:
                f.write(str(e[0]) + " " + str(e[1]) + "\n")
        elist_meta_data_file = real_world_data_folder + dataset + "_edge_list_meta.txt"
        with open(elist_meta_data_file, "w") as f:
            f.write(str(meta_dict["v"]) + " " + str(meta_dict["e"]) + "\n")
        
        # save partition.
        print("Saving partition.")
        partition_file = real_world_data_folder + dataset + "_2dpartition.txt"
        meta_file = real_world_data_folder + dataset + "_meta.txt"
        utilization_file = real_world_data_folder + dataset + "_utilization.txt"
        with open(partition_file, "w") as f:
            for i in partition_list:
                f.write(str(i[0]) + " " + str(i[1]) + "\n")
        with open(meta_file, "w") as f:
            for k, v in meta_dict.items():
                f.write(str(v) + " ")
        with open(utilization_file, "w") as f:
            f.write(json.dumps(utilization_dict))

    else:        
        graph = edge_list2igraph(edge_list_file)
        print("Finish graph loading.")
        print("Start 2d-partationing...")

        graph_save(graph, real_world_data_folder + dataset, "2dpartition", k, True)
        graph_save(graph, real_world_data_folder + dataset, "edgelist", -1, False)
        print("Finish dataset %s"%(dataset))

    # data synchronize
    # for party in party_list:
    #     # generate data folders on the remote party machine, also, recursive mkdir for target real_world_data_folder
    #     os.system(f"ssh {party} 'rm -rf {real_world_data_folder}*'")
    #     os.system(f"ssh {party} 'mkdir -p {real_world_data_folder}'")
    #     os.system(f"scp -r {real_world_data_folder}* {party}:{real_world_data_folder}")
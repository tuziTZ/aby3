"""Sampling for GORAM representation.
"""

import time
import argparse
import json
import os
from get_k import *
import sys 
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'aby3-GORAM', 'privGraphQuery')))
print(sys.path)

from utils import *

ABY3_FOLDER = os.getcwd()
MAIN_FOLDER = ABY3_FOLDER + "/aby3/aby3-GORAM"
# gtype_list = ["random", "k_regular", "powerlaw", "bipartite", "geometric"]
n = 32768
SAMPLE_TIME = 100

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Microbenchmark data generation')
    parser.add_argument('--type', type=str, default= "complete", help='graph type')
    parser.add_argument('--filename', type=str, default= "complete", help='save file name')
    parser.add_argument('--samples', type=int, default=SAMPLE_TIME, help='sample times')
    args = parser.parse_args()
    
    data_folder = MAIN_FOLDER + "/data/sampling"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    record_folder = MAIN_FOLDER + "/record/sampling"
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)
        
    
    # sample for the graphs.
    # for gtype in gtype_list:
    gtype = args.type
    sample_time = args.samples
    for _ in range(sample_time):
        set_random_seed(random.randint(0, SAMPLE_TIME+10086))
        graph = large_graph_generation(gtype, n)
        graph = graph_hashing(graph)
        v, e = graph.vcount(), graph.ecount()
        k = get_k(v, e)
        partition = trans2_2dpartition(graph, k)
        
        partition_list, l, utilization_dict = partition_format(partition)
        
        #file_prefix = data_folder + "/tmp_graph_" + gtype
        #graph_save(graph, file_prefix, "2dpartition", k, True, l)
        
        record_file = record_folder + "/record_" + gtype + "-" + args.filename + ".txt"
        record_dict = {
            "v": v,
            "e": e,
            "k": k,
            "l": l,
            "min_util": utilization_dict["min"],
            "mean_util": utilization_dict["mean"]
        }
        with open(record_file, "a") as f:
            f.write(json.dumps(record_dict) + "\n")
    
    

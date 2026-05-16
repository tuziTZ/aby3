import numpy as np 
import pandas as pd
import igraph 
import argparse
import os
import time
import math
from multiprocessing import Pool


ABY3_FOLDER = os.getcwd()   
MAIN_FOLDER = ABY3_FOLDER + "/aby3-GORAM"
real_world_data_folder = MAIN_FOLDER + "/data/real_world/"
data_prefix = "twitter"
meta_data_file = f"{real_world_data_folder}/{data_prefix}_meta.txt"
data_file = f"{real_world_data_folder}/{data_prefix}_2dpartition.txt"
raw_edge_list_file = f"{real_world_data_folder}/{data_prefix}_edge_list.txt"


def process_single_provider(args):
    """处理单个provider的数据"""
    j, data_file, data_folder, partitions, l, sub_l, last_l = args
    
    # 获取该provider的分区数据
    partition_list = get_provider_partitions(data_file, j, partitions, l, sub_l, last_l)
    
    # 保存数据
    print(f"provider {j} has {len(partition_list)} partitions | partition edge size: {len(partition_list[0])}")
    provider_file = data_folder + f"provider_{j}.txt"
    with open(provider_file, "w") as f:
        for partition in partition_list:
            for edge in partition:
                f.write(f"{edge[0]} {edge[1]}\n")
    
    return j


def get_provider_partitions(file_path, provider_id, partitions, l, sub_l, last_l):
    """一次性获取一个provider的所有分区数据"""
    # 预计算该provider需要的所有行索引范围
    ranges = []
    for i in range(partitions):
        offset = i*l + provider_id*sub_l
        length = last_l if provider_id == args.providers-1 else sub_l
        ranges.append((offset, offset + length))
    
    # 存储所有分区数据
    all_partitions = [[] for _ in range(partitions)]
    partition_map = {ranges[i][0]: i for i in range(len(ranges))}
    
    # 一次性读取文件
    with open(file_path, "r") as f:
        current_range_idx = 0
        current_start, current_end = ranges[current_range_idx]
        
        for i, line in enumerate(f):
            if current_range_idx >= len(ranges):
                break
                
            if i < current_start:
                continue

            while i >= current_end and current_range_idx < len(ranges) - 1:
                current_range_idx += 1
                current_start, current_end = ranges[current_range_idx]

            if current_start <= i < current_end:
                nodes = line.strip().split()
                if len(nodes) == 2:
                    partition_idx = partition_map[current_start]
                    all_partitions[partition_idx].append((int(nodes[0]), int(nodes[1])))
    
    return all_partitions

                

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
    edge_array = np.array(edge_list)
    return list(zip(
        [vertex_hash_map[x] for x in edge_array[:, 0]],
        [vertex_hash_map[x] for x in edge_array[:, 1]]
    ))


def partition_organization(target_edge_list, k, vertex_hash_map):
    
    # target_edge_list = edge_list_hash(target_edge_list, vertex_hash_map)
    V = len(vertex_hash_map)
    print(f"V: {V} | k: {k}")
    b = int(2**math.ceil(math.log2(V//k)))
    
    edges = np.array(target_edge_list)
    u = edges[:, 0]
    v = edges[:, 1]
    
    # 一次性计算所有索引
    indices = (u//k)*b + v//k
    
    # 使用 NumPy 的 unique 和 split 进行分组
    unique_indices, inverse_indices, counts = np.unique(indices, return_inverse=True, return_counts=True)
    splits = np.split(np.arange(len(indices)), np.cumsum(counts)[:-1])
    
    # 构建结果字典
    partition = {idx: [target_edge_list[i] for i in split] 
                for idx, split in zip(unique_indices, splits)}
    
    
    return {i: partition.get(i, []) for i in range(b**2)}


def partition_organization_single(args):
    """处理单个边列表分区"""
    edge_list_part, k, vertex_map = args
    return partition_organization(edge_list_part, k, vertex_map)  


def parallel_2d_partition_process(edge_list_parts, k, vertex_set, threads_num):
    """process the edge_list_parts in parallel"""
    # vertex_seq_map = get_vertex_sequence_id(vertex_set)
    # vertex_hash_map = get_vertex_hash_map(vertex_seq_map)
    
    args_list = [(part, k, vertex_set) for part in edge_list_parts]
    
    # 使用进程池并行处理
    with Pool(processes=threads_num) as pool:
        results = pool.map(partition_organization_single, args_list)
    
    return results

def process_single_row(args):
    """for each row"""
    start_idx, end_idx, partitions_list = args
    chunk_results = []
    
    for row_idx in range(start_idx, end_idx):
        merged_row = []
        for partition in partitions_list:
            merged_row.extend(partition[row_idx])
        merged_row.sort()
        chunk_results.append((row_idx, merged_row))
    
    return chunk_results


def partition_merge(partitions_list, threads_num):
    
    if not partitions_list or not partitions_list[0]:
        return []
    
    total_rows = len(partitions_list[0])
    
    # 计算每个进程处理的行数
    chunk_size = (total_rows + threads_num - 1) // threads_num
    
    # 创建任务列表
    args_list = []
    for i in range(threads_num):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        if start_idx < total_rows:
            args_list.append((start_idx, end_idx, partitions_list))
    
    # 并行处理
    with Pool(processes=threads_num) as pool:
        chunk_results = pool.map(process_single_row, args_list)
    
    # 展平并排序结果
    merged_list = []
    for chunk in chunk_results:
        merged_list.extend([row for _, row in chunk])
    
    return merged_list


def partition_normalize(partition):
    """normalize the 2d-partition"""
    max_size = max([len(sublist) for sublist in partition])
    for sublist in partition:
        sublist[:0] = [(0, 0)] * (max_size-len(sublist))
    return partition


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--providers', type=int, default=2, help="the number of data providers")
    parser.add_argument('--threads', type=int, default=8, help="the number of threads")
    parser.add_argument('--preprocess', type=bool, default=True, help="whether to preprocess the data")
    args = parser.parse_args()
    
    # load the meta data.
    with open(meta_data_file, "r") as f:
        meta_data = f.readline().strip()
    
    v, e, b, k, l = map(int, meta_data.split(" "))
    print(f"v: {v} | e: {e} | b: {b} | k: {k} | l: {l}")
    data_folder = f"{real_world_data_folder}/{data_prefix}_{args.providers}/"
    flag_preprocess = args.preprocess
    if not os.path.exists(data_folder):
        flag_preprocess = True
    
    # obtain the vertex set.
    if not os.path.exists(f"{real_world_data_folder}/{data_prefix}_vertex_set.txt"):
        vertex_set = set()
        with open(raw_edge_list_file, "r") as f:
            for line in f:
                nodes = line.strip().split()
                if len(nodes) == 2:
                    # edge_list.append((int(nodes[0]), int(nodes[1])))
                    vertex_set.add(int(nodes[0]))
                    vertex_set.add(int(nodes[1]))
        # print("load the edge list.")
                    
        with open(f"{real_world_data_folder}/{data_prefix}_vertex_set.txt", "w") as f:
            f.write(' '.join(map(str, vertex_set)))
        del edge_list
    else:
        with open(f"{real_world_data_folder}/{data_prefix}_vertex_set.txt", "r") as f:
            vertex_set = list(map(int, f.readline().strip().split()))
    
    # save the hash map.
    print("load the vertex set.")
    if not os.path.exists(f"{real_world_data_folder}/{data_prefix}_vertex_hash_map.txt"):
        vertex_seq_map = get_vertex_sequence_id(vertex_set)
        vertex_hash_map = get_vertex_hash_map(vertex_seq_map)
        print("generate the vertex hash map.")
        with open(f"{real_world_data_folder}/{data_prefix}_vertex_hash_map.txt", "w") as f:
            for key, val in vertex_hash_map.items():
                f.write(f"{key} {val}\n")
        print("save the vertex hash map.")
    else:
        vertex_hash_map = {}
        with open(f"{real_world_data_folder}/{data_prefix}_vertex_hash_map.txt", "r") as f:
            for line in f.readlines():
                key, val = map(int, line.strip().split())
                vertex_hash_map[key] = val

    if(flag_preprocess):
        # split the data for each data provider.
        sub_l = l // args.providers
        last_l = l - sub_l*(args.providers-1)
        partitions = b*b # the number of partitions.
        print(f"the number of partitions: {partitions} | the size of each partition: {l}")
        
        # save the data, check whether the data folder (x) exists, if not, create it.
        data_folder = f"{real_world_data_folder}/{data_prefix}_{args.providers}/"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)   
        
        args_list = [(j, data_file, data_folder, partitions, l, sub_l, last_l) for j in range(args.providers)]
        
        # with Pool(processes=args.providers) as pool:
        #     results = pool.map(process_single_provider, args_list) 
        for _args in args_list:
            result = process_single_provider(_args)
            print(f"provider {result} has been processed.")
            
    
    # count the time for transferring the data into 2d-partitions.
    time_record = []
    for i in range(args.providers):
        time_start = time.time()
        
        provider_file = data_folder + f"provider_{i}.txt"
        if(not os.path.exists(provider_file)):
            raise ValueError(f"{provider_file} does not exist.")
        
        # in parallel process.
        threads_num = args.threads
        # split the list in privider_file into threads_num parts.
        total_edge_list = []
        with open(provider_file, "r") as f:
            edge_list = f.readlines()
            edge_list = [edge.strip().split() for edge in edge_list]
            total_edge_list = [(int(edge[0]), int(edge[1])) for edge in edge_list]
        
        # split the edge_list into threads_num parts
        edge_list_parts = [total_edge_list[j::threads_num] for j in range(threads_num)]
        
        # process the edge_list_parts in parallel
        results = parallel_2d_partition_process(edge_list_parts, k, vertex_set, threads_num)
        
        # merge the results
        merged_list = partition_merge(results, threads_num)
        
        # normalize the merged_list
        merged_list = partition_normalize(merged_list)
        print(f"provider {i} | the number of partitions: {len(merged_list)} | the size of each partition: {len(merged_list[0])}")
        
        time_end = time.time()
        
        time_record.append(time_end - time_start)
        print(f"provider {i} | time: {time_end - time_start}")
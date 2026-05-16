import argparse
import pandas as pd
import json

def get_record(target_folder, gtype, n, k, count):
    """fetch a result from the target file.
    """
    target_file = f"{target_folder}{gtype}_n-{n}_k-{k}-{count}.txt"
    result_dict =  {"gtype": gtype, "n": n, "k": k}
    with open(target_file, "r") as f:
        for line in f:
            if "Time taken" in line:
                parts = line.strip().split(" ")
                key = parts[3][:-1]
                value = float(parts[4])
                result_dict[key] = value
            elif " : " in line:
                parts = line.strip().split(" : ")
                key = parts[0]
                value = int(parts[1])
                result_dict[key] = value
    return result_dict


def get_utilization(target_folder, gtype, n, k):
    """_summary_
    """
    meta_file = f"{target_folder}meta/{gtype}_n-{n}_k-{k}_utilization.txt"
    meta_wo_hash_file = f"{target_folder}meta_wo_hash/{gtype}_n-{n}_k-{k}_utilization.txt"
    
    res_dict = {"min": -1, "mean": -1, "min_wo_hash": -1, "mean_wo_hash": -1}
    
    with open(meta_file, "r") as f:
        _dict = json.load(f)
        res_dict["min"] = _dict["min"]
        res_dict["mean"] = _dict["mean"]
    
    with open(meta_wo_hash_file, "r") as f:
        _dict = json.load(f)
        res_dict["min_wo_hash"] = _dict["min"]
        res_dict["mean_wo_hash"] = _dict["mean"]
    
    return res_dict


def construct_result_item(target_folder, gtype, n, k, counts):
    """construct a result item from the target file.
    """
    config_keys = ["v", "e", "b", "k", "l", "se", "pe", "sn", "pn"]
    time_keys = ["OuttingEdgesCountQuery", "NodeEdgesBlockFetch", "NodeOramInit", "GraphLoad", "EdgeOramInit", "EdgeExistQuery", "EdgeBlockFetch"]
    utilization_keys = ["min", "mean", "min_wo_hash", "mean_wo_hash"]
    
    target_record = {"gtype": gtype}
    for key in config_keys:
        target_record[key] = -1
    for key in time_keys:
        target_record[key] = []
    
    for count in range(1, counts+1):
        record_dict = get_record(target_folder, gtype, n, k, count)
        utilization_dict = get_utilization(target_folder, gtype, n, k)
        for key in config_keys:
            target_record[key] = record_dict[key]
        for key in utilization_keys:    
            target_record[key] = utilization_dict[key]
        for key in time_keys:
            target_record[key].append(record_dict[key])
    
    for key in time_keys:
        target_record[key] = sum(target_record[key])/len(target_record[key])
            
    return target_record


def get_pandas_record(target_folder, gtype_list, n_list, k_list, counts):
    """Get a series of record dict and construct a pandas table.
    """
    result_list = []
    for gtype in gtype_list:
        for i in range(len(n_list)):
            n = n_list[i]
            k = k_list[i]
            result_list.append(construct_result_item(target_folder, gtype, n, k, counts))
                
    return pd.DataFrame(result_list)    
    

# get the parameters.
parser = argparse.ArgumentParser(description='Microbenchmark performance analysis')


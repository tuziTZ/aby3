import random
import os
import argparse
random.seed(16)

def generate_graph_file(v, e, save_folder):
    nodes_ids = [i for i in range(v)]
    out_degrees = [0 for i in range(v)]
    
    edges_info = []
    nodes_info = []
    
    for i in range(e):
        start, end = random.sample(nodes_ids, 2)
        edge_data = 0
        tmp = (i, start, end, edge_data)
        out_degrees[start] += 1
        edges_info.append(tmp)
        
    for i in range(v):
        # pr_value = random.random()
        pr_value = 0.15 / v
        tmp = (i, pr_value, out_degrees[i])
        nodes_info.append(tmp)
    
    with open(save_folder + "/edges.txt", "w") as f:
        for item in edges_info:
            f.write("{} {} {} {}\n".format(item[0], item[1], item[2], item[3]))
    
    with open(save_folder + "/nodes.txt", "w") as f:
        for item in nodes_info:
            f.write("{} {} {}\n".format(item[0], item[1], item[2]))
    
    return edges_info, nodes_info


def gather(node, edges):
    gather_val = 0
    for edg_id in edges:
        edg = edges[edg_id]
        if(edg["end"] == node["id"]):
            gather_val += (0.85 * edg["data"])
    return gather_val


def scatter(edge_start, nodes_dict):
    e_data = nodes_dict[edge_start]["pr"] / nodes_dict[edge_start]["out_deg"]
    return e_data


def pta_gather(node, edges):
    tbl_list = []
    for edg_id in edges:
        edg = edges[edg_id]
        if(edg["end"] == node["id"]):
            tbl_list.append(0.85 * edg["data"])
        else:
            tbl_list.append(0)
            # gather_val += (0.85 * edg["data"])
    gather_val = 0
    for item in tbl_list:
        gather_val += item
    return gather_val, tbl_list


def pta_scatter(edge_start, nodes_dict):
    tbl_list = []
    pr_list = []
    reci_list = []
    for node_id in nodes_dict.keys():
        if(edge_start == node_id):
            tbl_list.append(nodes_dict[edge_start]["pr"] / nodes_dict[edge_start]["out_deg"])
            e_data = nodes_dict[edge_start]["pr"] / nodes_dict[edge_start]["out_deg"]
        else:
            tbl_list.append(0)
        pr_list.append(nodes_dict[node_id]["pr"])
        reci_list.append(1 / (nodes_dict[node_id]["out_deg"] + (nodes_dict[node_id]["out_deg"] == 0)))
    return e_data, (tbl_list, pr_list, reci_list)


def page_rank_test(edges_info, nodes_info, save_folder, iters=1):
    """Generate the test result.
    """
    nodes_dict = {item[0]: {"id": item[0], "pr": item[1], "out_deg": item[2]} for item in nodes_info}
    edges_dict = {item[0]: {"start": item[1], "end": item[2], "data": item[3]} for item in edges_info}
    
    # test iters rounds.
    i = 0
    while(i < iters):
        # scatter to edges
        for edge_id in edges_dict:
            edge = edges_dict[edge_id]
            edge["data"] = scatter(edge["start"], nodes_dict)
        
        # gather to nodes and apply
        for node_id in nodes_dict:
            # print(nodes_dict[node_id])
            node = nodes_dict[node_id]
            node["pr"] = gather(node, edges_dict)
        
        i += 1
    
    # save the nodes pr info.
    with open(save_folder + "/page_rank-iters" + str(iters) + ".txt", "w") as f:
        for node_id in nodes_dict:
            f.write("{} {}\n".format(nodes_dict[node_id]['id'], nodes_dict[node_id]['pr']))


def page_rank_pta_debug(edges_info, nodes_info, save_folder, iters=1):
    """Generate the test result for debugging.
    """
    nodes_dict = {item[0]: {"id": item[0], "pr": item[1], "out_deg": item[2]} for item in nodes_info}
    edges_dict = {item[0]: {"start": item[1], "end": item[2], "data": item[3]} for item in edges_info}
    
    # test iters rounds.
    scatter_tbl = []
    pr_tbl, reci_tbl = [], []
    gather_tbl = []
    i = 0
    while(i < iters):
        # scatter to edges
        for edge_id in edges_dict:
            edge = edges_dict[edge_id]
            edge["data"], debug_list = pta_scatter(edge["start"], nodes_dict)
            scatter_tbl.append(debug_list[0])
            pr_tbl.append(debug_list[1])
            reci_tbl.append(debug_list[2])
        
        # gather to nodes and apply
        for node_id in nodes_dict:
            # print(nodes_dict[node_id])
            node = nodes_dict[node_id]
            node["pr"], tbl_list = pta_gather(node, edges_dict)
            gather_tbl.append(tbl_list)
        
        i += 1
    
    # save the nodes pr info.
    with open(save_folder + "/page_rank-iters" + str(iters) + "-debug.txt", "w") as f:
        for node_id in nodes_dict:
            f.write("{} {}\n".format(nodes_dict[node_id]['id'], nodes_dict[node_id]['pr']))
        f.write("\nscatter tbl \n")
        for line in scatter_tbl:
            for item in line:
                f.write(str(item))
                f.write(" ")
            f.write("\n")
        f.write("\npr tbl \n")
        for line in pr_tbl:
            for item in line:
                f.write(str(item))
                f.write(" ")
            f.write("\n")
        f.write("\nreci tbl \n")
        for line in reci_tbl:
            for item in line:
                f.write(str(item))
                f.write(" ")
            f.write("\n")
        f.write("\ngather tbl \n")
        for line in gather_tbl:
            for item in line:
                f.write(str(item))
                f.write(" ")
            f.write("\n")
    # with open(save_folder + "/page_rank-iters" + str(iters) + ".txt", "w") as f:
    #     for node_id in nodes_dict:
    #         f.write("{} {}\n".format(nodes_dict[node_id]['id'], nodes_dict[node_id]['pr']))
    


if __name__ == "__main__":
    
    e, v = 10, 5
    data_folder = "./"
    iters = 1
    
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="这是一个示例脚本")

    # 添加命令行参数
    parser.add_argument('--e', type=int, help="edges num")
    parser.add_argument('--v', type=int, help="vertices num")
    parser.add_argument('--data_folder', type=str, help="data_folder")
    parser.add_argument('--iters', type=int, help="iterations")

    print("run this py")
    
    # 解析命令行参数
    args = parser.parse_args()

    if hasattr(args, "e"):
        e = args.e
    if hasattr(args, "v"):
        v = args.v
    if hasattr(args, "data_folder"):
        # data_folder = data_folder + args.data_folder
        data_folder = args.data_folder
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    edges_info, nodes_info = generate_graph_file(v, e, data_folder)
    # page_rank_test(edges_info, nodes_info, data_folder, iters)
    # page_rank_pta_debug(edges_info, nodes_info, data_folder, iters)
import os

unit_size = 1<<5
total_size_list = [1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20]
# total_size_list = [1<<10]
function_list = ["shuffMem", "permutation_net"]
root_record_folder = "/root/aby3/privGraph/shuffle_record/"

if not os.path.exists(root_record_folder):
    os.makedirs(root_record_folder)

if __name__ == "__main__":
    
    os.system("cp ./frontend/main.pgp ./frontend/main.cpp; python build.py")
    func = "shuffMem"
    for func in function_list:
        for size in total_size_list:
            # target_file = f"{root_record_folder}{func}-{size}.txt"
            file_prefix = f"{func}-{size}"
            eval_args = f" -{func} -length {size} -unit_length {unit_size} -record_folder {root_record_folder} -file_prefix {file_prefix}"
            os.system(f"./Eval/dis_exec.sh \"{eval_args}\"")
            
data_folder="./aby3-GORAM/data/micro_benchmark"
multiparty_folder="./aby3-GORAM/data/multiparty"

# mkdirs
if [ ! -d "$data_folder" ]; then
    mkdir -p "$data_folder"
else
    echo "Directory $data_folder already exists."
fi

if [ ! -d "$multiparty_folder" ]; then
    mkdir -p "$multiparty_folder"
else
    echo "Directory $multiparty_folder already exists."
fi

# star graph
python ./aby3-GORAM/privGraphQuery/micro_benchmark_generation.py --type "star" --n 32 --file_prefix ${data_folder}"/star"

# tmp_graph
python ./aby3-GORAM/privGraphQuery/micro_benchmark_generation.py --n 32 --file_prefix ${data_folder}"/tmp_graph"

# edgelist graphs.
python ./aby3-GORAM/privGraphQuery/micro_benchmark_generation.py --n 32 --file_prefix ${data_folder}"/adj_tmp" --saving_type "edgelist"

# multiparty graph
python ./aby3-GORAM/privGraphQuery/micro_benchmark_generation.py --n 32 --file_prefix ${multiparty_folder}"/random_n-16_k-2" --p 8

python ./aby3-GORAM/privGraphQuery/micro_benchmark_generation.py --n 32 --file_prefix ${multiparty_folder}"/random_n-16" --saving_type "edgelist" --p 8

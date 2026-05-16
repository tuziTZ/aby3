data_folder=~/aby3/aby3-RTR/Pair_then_Reduce/data/page_rank
record_folder=~/aby3/Record/
task="page_rank"
log_folder=${record_folder}Record_${task}/
taskN_list=(32 64 128)
edges_num=1048576
nodes_num=$((${edges_num} * 20 / 100))
echo ${nodes_num}
optB=1048576
iters=1

if [ ! -d ${log_folder} ]; then
    mkdir ${log_folder}
else
    rm -r ${log_folder};
    mkdir ${log_folder};
fi

python build.py

# synchronize with others
scp ./bin/frontend aby31:~/aby3/bin &
scp ./bin/frontend aby32:~/aby3/bin &
wait;

# test data generation.
python ./aby3-RTR/Pair_then_Reduce/data/graph_data_generation.py --e ${edges_num} --v ${nodes_num} --data_folder ${data_folder} --iters ${iters};

# synchronize with others
scp -r ./aby3-RTR/Pair_then_Reduce/data aby31:~/aby3/aby3-RTR/Pair_then_Reduce/ &
scp -r ./aby3-RTR/Pair_then_Reduce/data aby32:~/aby3/aby3-RTR/Pair_then_Reduce/ &
wait;


for taskN in ${taskN_list[@]}; do
    # log_file="log-config-E=${edges_num}-V=${nodes_num}-TASKS=${taskN}-OPT_B=${optB}-iters=${iters}";
    log_file="log-config-"
    # echo ${log_file};
    # echo "111" >> ${log_file};
    # # eval the task
    ./Eval/basic/mpi_subtask.sh taskN=${taskN} task=${task} data_folder=${data_folder} edges_num=${edges_num} nodes_num=${nodes_num} optB=${optB} iters=${iters} log_folder=${log_folder} logFile="${log_file}";
    wait;
done;
task="prof_task_setup"; log_folder=./Record/prof_deploy;
task_num_list=(1 4 8 16 32 64 128)

if [ ! -d ${log_folder} ]; then
    mkdir ${log_folder}
else
    rm -r ${log_folder};
    mkdir ${log_folder};
fi

# python build.py
# # synchronize with others.
# scp ./bin/frontend aby31:~/aby3/bin &
# scp ./bin/frontend aby32:~/aby3/bin &
# wait;

for (( k=0; k<${#task_num_list[@]}; k++ )); do
    taskN=${task_num_list[k]};
    timeout 20m ./Eval/basic/mpi_subtask.sh taskN=${taskN} log_folder=${log_folder}/ task=${task} n=100 m=100 optB=100;
done;

# sh ./Eval/network/network_clean.sh
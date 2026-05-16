# # profile at first.

# # compile and synchronize
# python build.py
# scp ./bin/frontend aby31:~/aby3/bin &
# scp ./bin/frontend aby32:~/aby3/bin &
# wait;

# # simulate WAN network
# bandwidth=1024
# latency=0.13
# ./Eval/network/network_set.sh $bandwidth $latency;
# wait;

# # profile for new_search functions.
# task_list=("prof_new_search_mpi")
# log_folder_list=("./Record/Prof_mpi_new_search_test")

# # cleanup the old folder and create a clean one.
# for log_folder in ${log_folder_list[@]}; do
#   if [ ! -d ${log_folder} ]; then
#     mkdir ${log_folder}
#   else
#     mkdir ${log_folder};
#   fi
# done

# available_cores_list=(256);
# retry_threshold=5;
# repeat_list=(10);

# for (( i=0; i<${#task_list[@]}; i++ )); do
#   task=${task_list[i]}; log_folder=${log_folder_list[i]};

#   # for available_cores in ${available_cores_list[@]}; do
#   for (( k=0; k<${#available_cores_list[@]}; k++ )); do
#     available_cores=${available_cores_list[k]}; repeat=${repeat_list[k]}

#     # add a new, clean log folder.
#     day=$(date +%m-%d);
#     timeStamp=$(date +"%H%M%s");

#     # evaluate on aby3
#     n=1; optB=16384; m=1000000000; epsilon=0.1; gap=16384; K=1;

#     j=0;
#     while [ $j -lt $retry_threshold ]; do
#       start_time=$(date +%s)
#       timeout 100m ./Eval/basic/mpi_subtask.sh taskN=${available_cores} n=${n} m=${m} repeat=${repeat} task=${task} optB=${optB} log_folder=${log_folder}/ epsilon=${epsilon} gap=${gap} vec_start=${optB} k=${K};
#       if [ $? -eq 0 ]; then
#         break; 
#       fi
#       j=$(expr $j + 1);
#       if [ $j -eq $retry_threshold ]; then
#         echo "Max retry: "${n}-${taskN} >> ${log_folder}/error.log;
#       fi
#     done;

#     sh ./Eval/basic/kill_all.sh frontend;

#     cp ${log_folder}/probe.log ${res_folder}/probe-task_${task}_c=${available_cores}.log
#     cp ${log_folder}/probe.res ${res_folder}/probe-task_${task}_c=${available_cores}.res

#     rm -r ${log_folder}/*
#   done;
# done;


# test the performance
task_list=(new_search new_search new_search new_search new_search new_search)
record_list=("./Record/new_search_test" "./Record/new_search_test" "./Record/new_search_test" "./Record/new_search_test" "./Record/new_search_test" "./Record/new_search_test")
# m_list=(16 512 16384 524288 16777216 67108864)
# n_list=(1 1 1 1 1 1)
# c_list=(1 1 6 28 171 256)
# B_list=(180224 180224 180224 180224 180224 180224)

m_list=(524288)
n_list=(1)
c_list=(28)
B_list=(180224)

test_times=3;outLimit=30;retry_threshold=5;K=1;

./Eval/basic/seq_test.sh \
  "$(IFS=":"; echo "${task_list[*]}")" \
  "$(IFS=":"; echo "${record_list[*]}")" \
  "$(IFS=":"; echo "${n_list[*]}")" \
  "$(IFS=":"; echo "${m_list[*]}")" \
  "$(IFS=":"; echo "${c_list[*]}")" \
  "$(IFS=":"; echo "${B_list[*]}")" \
  ${test_times} ${outLimit} ${retry_threshold} ${K}


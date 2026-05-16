# task_list=("prof_metric_mpi" "prof_cipher_index_mpi" "prof_average_mpi" "prof_max_mpi" "prof_new_search_mpi")
# log_folder_list=(./Record/Prof_mpi_metric ./Record/Prof_mpi_cipher_index ./Record/Prof_mpi_average ./Record/Prof_mpi_max ./Record/Prof_mpi_new_search)
task_list=("prof_cipher_index_mpi")
log_folder_list=(./Record/Prof_mpi_cipher_index)


# cleanup the old folder and create a clean one.
for log_folder in ${log_folder_list[@]}; do
  if [ ! -d ${log_folder} ]; then
    mkdir ${log_folder}
  else
    rm -r ${log_folder};
    mkdir ${log_folder};
  fi
done

# construct the res folder
res_folder=./Record/probe_info_wan
if [ ! -d ${res_folder} ]; then
  mkdir ${res_folder}
else
  # rm -r ${res_folder};
  mkdir ${res_folder};
fi

available_cores_list=(16 128 256)
repeat_list=(1 1)

# # compile.
# python build.py

# # synchronize with others.
# scp ./bin/frontend aby31:~/aby3/bin &
# scp ./bin/frontend aby32:~/aby3/bin &
# wait;

retry_threshold=5;

# # simulate WAN network
# bandwidth=100
# latency=50
# ./Eval/network/network_set.sh $bandwidth $latency;
# wait;

for (( i=0; i<${#task_list[@]}; i++ )); do
  task=${task_list[i]}; log_folder=${log_folder_list[i]};

  # for available_cores in ${available_cores_list[@]}; do
  for (( k=0; k<${#available_cores_list[@]}; k++ )); do
    available_cores=${available_cores_list[k]}; repeat=${repeat_list[k]}

    # add a new, clean log folder.
    day=$(date +%m-%d);
    timeStamp=$(date +"%H%M%s");

    # evaluate on aby3
    n=1; optB=16384; m=1000000000; epsilon=0.1; gap=16384; K=1;

    j=0;
    while [ $j -lt $retry_threshold ]; do
      start_time=$(date +%s)
      timeout 100m ./Eval/basic/mpi_subtask.sh taskN=${available_cores} n=${n} m=${m} repeat=${repeat} task=${task} optB=${optB} log_folder=${log_folder}/ epsilon=${epsilon} gap=${gap} vec_start=${optB} k=${K};
      if [ $? -eq 0 ]; then
        break; 
      fi
      j=$(expr $j + 1);
      if [ $j -eq $retry_threshold ]; then
        echo "Max retry: "${n}-${taskN} >> ${log_folder}/error.log;
      fi
    done;

    sh ./Eval/basic/kill_all.sh frontend;

    cp ${log_folder}/probe.log ${res_folder}/probe-task_${task}_c=${available_cores}.log
    cp ${log_folder}/probe.res ${res_folder}/probe-task_${task}_c=${available_cores}.res

    rm -r ${log_folder}/*
  done;
done;

# sh ./Eval/network/network_clean.sh
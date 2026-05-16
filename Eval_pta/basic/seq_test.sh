#!/bin/bash

# Set IFS to colon
IFS=":"

# Access the passed arguments and split them into arrays
task_list=($1)
log_folder_list=($2)
M_list=($3)
N_list=($4)
c_list=($5)
optB_list=($6)

# Reset IFS to its default value (space, tab, newline) or your preferred delimiter
IFS=$' \t\n'  # This sets IFS back to the default whitespace delimiters

test_times=$7; outLimit=$8; retry_threshold=$9; K=${10};

for log_folder in ${log_folder_list[@]}; do
  if [ ! -d ${log_folder} ]; then
    mkdir ${log_folder}
  else
    # rm -r ${log_folder};
    mkdir ${log_folder};
  fi
done

# compile
./Eval/monitor/utilization_monitor.sh ./Monitor/compile.log &
monitor_id=$!
python build.py
kill ${monitor_id}

# synchroonize with others
scp ./bin/frontend aby31:~/aby3/bin &
scp ./bin/frontend aby32:~/aby3/bin &
wait;

for (( i=0; i<${#task_list[@]}; i++ )); do
  task=${task_list[i]}; log_folder=${log_folder_list[i]}; 
  N=${N_list[i]}; M=${M_list[i]};
  taskN=${c_list[i]}; optB=${optB_list[i]};

  for (( z=0; z<${test_times}; z++ )); do
    j=0;
    while [ $j -lt $retry_threshold ]; do
      monitor_log=./Monitor/log-${task}-N=${N}-M=${M}-K=${K}-c=${task}-B=${optB}
      ./Eval/monitor/utilization_monitor.sh ${monitor_log} &
      exec_monitor_id=$!
      timeout ${outLimit}m ./Eval/basic/mpi_subtask.sh taskN=${taskN} n=$N m=$M k=$K task=$task optB=$optB log_folder=$log_folder/;
      kill ${exec_monitor_id};
      if [ $? -eq 0 ]; then
        break; 
      fi
      ./Eval/basic/kill_all.sh frontend; sleep 10;
      j=$(expr $j + 1);
      if [ $j -eq $retry_threshold ]; then
        echo "Max retry: "${n}-${taskN} >> ${log_folder}/error.log;
      fi
    done;
  done;
done;

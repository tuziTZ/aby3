#!/bin/bash

taskN=1
repeat=100

# 循环遍历所有参数
for param in "$@"
do
  # 解析参数，提取键值对
  IFS="=" read -ra params <<< "$param"
  key="${params[0]}"
  value="${params[1]}"

  echo ${key}"="${value}

  # 使用键作为变量名，并赋值
  eval "$key='$value'"
done

# evaluate the task
mpirun -np ${taskN} ./bin/frontend -prog -1 -role 0 -N ${n} -TASK_NUM ${taskN} -FUNC ${task} -M ${m} -K ${k} -OPT_BLOCK ${optB} -repeats ${repeat} -logFolder ${log_folder} -VEC_START ${vec_start} -EPSILON ${epsilon} -GAP ${gap} -data_folder ${data_folder} -nodes_num ${nodes_num} -edges_num ${edges_num} -logFile ${logFile} -iters ${iters} &
ssh aby31 "cd ./aby3/; ulimit -n 65536; mpirun -np ${taskN} ./bin/frontend -prog -1 -role 1 -N ${n} -TASK_NUM ${taskN} -FUNC ${task} -M ${m} -K ${k} -OPT_BLOCK ${optB} -repeats ${repeat} -logFolder ${log_folder} -VEC_START ${vec_start} -EPSILON ${epsilon} -GAP ${gap}" -data_folder ${data_folder} -nodes_num ${nodes_num} -edges_num ${edges_num} -logFile ${logFile} -iters ${iters} &
ssh aby32 "cd ./aby3/; ulimit -n 65536; mpirun -np ${taskN} ./bin/frontend -prog -1 -role 2 -N ${n} -TASK_NUM ${taskN} -FUNC ${task} -M ${m} -K ${k} -OPT_BLOCK ${optB} -repeats ${repeat} -logFolder ${log_folder} -VEC_START ${vec_start} -EPSILON ${epsilon} -GAP ${gap}" -data_folder ${data_folder} -nodes_num ${nodes_num} -edges_num ${edges_num} -logFile ${logFile} -iters ${iters} &
wait;
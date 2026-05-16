#!/bin/bash

log_folder_list=(./Record/Record_cipher_index)
task_list=("cipher_index")
M_list=(1)
N_list=(1073741824)
c_list=(256)
optB_list=(737280)

test_times=1;outLimit=10;retry_threshold=5;K=1;

./Eval/basic/seq_test.sh \
  "$(IFS=":"; echo "${task_list[*]}")" \
  "$(IFS=":"; echo "${log_folder_list[*]}")" \
  "$(IFS=":"; echo "${M_list[*]}")" \
  "$(IFS=":"; echo "${N_list[*]}")" \
  "$(IFS=":"; echo "${c_list[*]}")" \
  "$(IFS=":"; echo "${optB_list[*]}")" \
  ${test_times} ${outLimit} ${retry_threshold} ${K}
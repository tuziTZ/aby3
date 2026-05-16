record_list=(./Record/Record_new_search)
task_list=(new_search)
n_list=(1)
m_list=(1048576)
c_list=(10)
B_list=(32768)
# c_list=(27)
# B_list=(38837)

test_times=1;outLimit=10;retry_threshold=1;K=1;

./Eval/basic/seq_test.sh \
  "$(IFS=":"; echo "${task_list[*]}")" \
  "$(IFS=":"; echo "${record_list[*]}")" \
  "$(IFS=":"; echo "${n_list[*]}")" \
  "$(IFS=":"; echo "${m_list[*]}")" \
  "$(IFS=":"; echo "${c_list[*]}")" \
  "$(IFS=":"; echo "${B_list[*]}")" \
  ${test_times} ${outLimit} ${retry_threshold} ${K}
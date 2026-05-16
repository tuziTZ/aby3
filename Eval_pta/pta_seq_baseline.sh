# record_list=(./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_metric ./Record/Record_metric ./Record/Record_metric ./Record/Record_metric ./Record/Record_metric ./Record/Record_metric ./Record/Record_average ./Record/Record_average ./Record/Record_average ./Record/Record_average ./Record/Record_average ./Record/Record_average ./Record/Record_cipher_index ./Record/Record_cipher_index ./Record/Record_cipher_index ./Record/Record_cipher_index ./Record/Record_cipher_index ./Record/Record_cipher_index ./Record/Record_max ./Record/Record_max ./Record/Record_max ./Record/Record_max ./Record/Record_max ./Record/Record_max)
# task_list=(new_search new_search new_search new_search new_search new_search metric metric metric metric metric metric average average average average average average cipher_index cipher_index cipher_index cipher_index cipher_index cipher_index max max max max max max)
# n_list=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1024 2048 4096 8192 16384 32768)
# m_list=(1048576 4194304 16777216 67108864 268435456 1073741824 1048576 4194304 16777216 67108864 268435456 1073741824 1048576 4194304 16777216 67108864 268435456 1073741824 1048576 4194304 16777216 67108864 268435456 1073741824 1024 2048 4096 8192 16384 32768)
# c_list=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
# B_list=(892928 892928 892928 892928 892928 892928 2247671 2247671 2247671 2247671 2247671 2247671 360448 360448 360448 360448 360448 360448 737280 737280 737280 737280 737280 737280 647168 647168 647168 647168 647168 647168)


record_list=(./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search)
task_list=(new_search new_search new_search new_search new_search new_search)
n_list=(1 1 1 1 1 1)
m_list=(1048576 4194304 16777216 67108864 268435456 1073741824)
c_list=(1 1 1 1 1 1)
B_list=(892928 892928 892928 892928 892928 892928)

# record_list=(./Record/Record_new_search)
# task_list=(new_search)
# n_list=(1)
# m_list=(1073741824)
# c_list=(1)
# B_list=(892928)


test_times=3;outLimit=40;retry_threshold=5;K=1;

./Eval/basic/seq_test.sh \
  "$(IFS=":"; echo "${task_list[*]}")" \
  "$(IFS=":"; echo "${record_list[*]}")" \
  "$(IFS=":"; echo "${n_list[*]}")" \
  "$(IFS=":"; echo "${m_list[*]}")" \
  "$(IFS=":"; echo "${c_list[*]}")" \
  "$(IFS=":"; echo "${B_list[*]}")" \
  ${test_times} ${outLimit} ${retry_threshold} ${K}
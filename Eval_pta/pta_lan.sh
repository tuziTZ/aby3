
# record_list=(./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search ./Record/Record_new_search)
# task_list=(new_search new_search new_search new_search new_search new_search)
# n_list=(1 1 1 1 1 1)
# m_list=(1048576 4194304 16777216 67108864 268435456 1073741824)
# c_list=(27 54 115 235 256 256)
# B_list=(38837 77673 145889 285570 892928 892928)

# record_list=(./Record/Record_cipher_index ./Record/Record_cipher_index )
# task_list=(cipher_index cipher_index )
# n_list=(1 1 )
# m_list=(1048576 1073741824)
# c_list=(37 256)
# B_list=(28340 98304)


test_times=10;outLimit=20;retry_threshold=5;K=1;

./Eval/basic/seq_test.sh \
  "$(IFS=":"; echo "${task_list[*]}")" \
  "$(IFS=":"; echo "${record_list[*]}")" \
  "$(IFS=":"; echo "${n_list[*]}")" \
  "$(IFS=":"; echo "${m_list[*]}")" \
  "$(IFS=":"; echo "${c_list[*]}")" \
  "$(IFS=":"; echo "${B_list[*]}")" \
  ${test_times} ${outLimit} ${retry_threshold} ${K}
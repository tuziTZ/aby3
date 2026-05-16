# task_list=("new_search" "average" "metric")
# log_folder_list=(./Record/Record_new_search ./Record/Record_average ./Record/Record_metric)
# optB_list=(892928 360448 2247671)

# task_list=("metric" "average")
# log_folder_list=(./Record/Record_metric ./Record/Record_average)
# optB_list=(2247671 360448)

# task_list=("metric")
# log_folder_list=(./Record/Record_metric)
# optB_list=(2247671)

# N_list=(1048576 16777216)
# M_list=(1 1 1 1)
# c_list=(1 4 16 32 64)

# record_list=(./Record/Record_cipher_index ./Record/Record_cipher_index )
# task_list=(cipher_index cipher_index )
# n_list=(1 1 )
# m_list=(1048576 1073741824)
# c_list=(37 256)
# B_list=(28340 98304)

# B_list=(16384 65536 262144 1048576 4194304)

# cleanup the old folder and create a clean one.
for log_folder in ${log_folder_list[@]}; do
  if [ ! -d ${log_folder} ]; then
    mkdir ${log_folder}
  else
    mkdir ${log_folder};
  fi
done

# compile
python build.py

# synchroonize with others
scp ./bin/frontend aby31:~/aby3/bin &
scp ./bin/frontend aby32:~/aby3/bin &
wait;

test_times=3;
retry_threshold=5;
outLimit=30;
K=1;


for (( i=0; i<${#task_list[@]}; i++ )); do

  #   # set the task the correspondng log folder
  task=${task_list[i]}; log_folder=${log_folder_list[i]}; optB=${optB_list[i]};

  for (( t=0; t<${#N_list[@]}; t++ )); do
    N=${N_list[t]}; M=${M_list[t]};

    for (( c=0; c<${#c_list[@]}; c++ )); do
      taskN=${c_list[c]};

      # for (( k=0; k<${#optB_list[@]}; k++ )); do
      #   optB=${optB_list[k]};

        for (( z=0; z<${test_times}; z++  )); do
          j=0;
          while [ $j -lt $retry_threshold ]; do
            timeout ${outLimit}m ./Eval/basic/mpi_subtask.sh taskN=${taskN} n=$N m=$M k=$K task=$task optB=$optB log_folder=$log_folder/;
            if [ $? -eq 0 ]; then
              break; 
            fi
            ./Eval/basic/kill_all.sh frontend;
            j=$(expr $j + 1);
            if [ $j -eq $retry_threshold ]; then
              echo "Max retry: "${n}-${taskN} >> ${log_folder}/error.log;
            fi
          done;
        done;
      # done;
    done;
  done;
done;



# task_list=("max")
# log_folder_list=(./Record/Record_max)
# optB_list=(647168)
# N_list=(1024 4096 16384 32768)
# M_list=(1024 4096 16384 32768)
# c_list=(1 4 16 32 64 128 256)

# # cleanup the old folder and create a clean one.
# for log_folder in ${log_folder_list[@]}; do
#   if [ ! -d ${log_folder} ]; then
#     mkdir ${log_folder}
#   else
#     # rm -r ${log_folder};
#     mkdir ${log_folder};
#   fi
# done

# # compile
# python build.py

# # synchroonize with others
# scp ./bin/frontend aby31:~/aby3/bin &
# scp ./bin/frontend aby32:~/aby3/bin &
# wait;

# test_times=3;
# retry_threshold=5;


# for (( i=0; i<${#task_list[@]}; i++ )); do

#   #   # set the task the correspondng log folder
#   task=${task_list[i]}; log_folder=${log_folder_list[i]}; optB=${optB_list[i]};

#   for (( t=0; t<${#N_list[@]}; t++ )); do
#     N=${N_list[t]}; M=${M_list[t]};

#     for (( c=0; c<${#c_list[@]}; c++ )); do
#       taskN=${c_list[c]};

#       # for (( k=0; k<${#optB_list[@]}; k++ )); do
#       #   optB=${optB_list[k]};

#         for (( z=0; z<${test_times}; z++  )); do
#           j=0;
#           while [ $j -lt $retry_threshold ]; do
#             timeout ${outLimit}m ./Eval/basic/mpi_subtask.sh taskN=${taskN} n=$N m=$M k=$K task=$task optB=$optB log_folder=$log_folder/;
#             if [ $? -eq 0 ]; then
#               break; 
#             fi
#             ./Eval/basic/kill_all.sh frontend;
#             j=$(expr $j + 1);
#             if [ $j -eq $retry_threshold ]; then
#               echo "Max retry: "${n}-${taskN} >> ${log_folder}/error.log;
#             fi
#           done;
#         done;
#       # done;
#     done;
#   done;
# done;
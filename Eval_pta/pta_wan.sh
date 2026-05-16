record_list=(./Record/Record_cipher_index ./Record/Record_cipher_index)
task_list=(cipher_index cipher_index)
n_list=(1 )
m_list=(16777216)
c_list=(128)
B_list=(98304)

test_times=10;outLimit=100;retry_threshold=5;K=1;

./Eval/network/network_clean.sh;

# simulate WAN network
bandwidth=100
latency=50
./Eval/network/network_set.sh $bandwidth $latency;
wait;

./Eval/basic/seq_test.sh \
  "$(IFS=":"; echo "${task_list[*]}")" \
  "$(IFS=":"; echo "${record_list[*]}")" \
  "$(IFS=":"; echo "${n_list[*]}")" \
  "$(IFS=":"; echo "${m_list[*]}")" \
  "$(IFS=":"; echo "${c_list[*]}")" \
  "$(IFS=":"; echo "${B_list[*]}")" \
  ${test_times} ${outLimit} ${retry_threshold} ${K}
wait;


# task_list=("cipher_index" "new_search" "average" "metric")
task_list=("cipher_index")
log_folder_list=(./Record/Record_cipher_index_all)
N_list=(1048576 16777216)
M_list=(1 1)
c_list=(4 16 32 64 128)
optB_list=(16384 65536 262144 1048576)
K=1

# cleanup the old folder and create a clean one.
for log_folder in ${log_folder_list[@]}; do
  if [ ! -d ${log_folder} ]; then
    mkdir ${log_folder}
  else
    # rm -r ${log_folder};
    mkdir ${log_folder};
  fi
done

# compile
python build.py

# synchroonize with others
scp ./bin/frontend aby31:~/aby3/bin &
scp ./bin/frontend aby32:~/aby3/bin &
wait;

test_times=2;
retry_threshold=5;
outLimit=200;

for (( i=0; i<${#task_list[@]}; i++ )); do

  #   # set the task the correspondng log folder
  task=${task_list[i]}; log_folder=${log_folder_list[i]};

  for (( t=0; t<${#N_list[@]}; t++ )); do
    N=${N_list[t]}; M=${M_list[t]};

    for (( c=0; c<${#c_list[@]}; c++ )); do
      taskN=${c_list[c]}; 

      for (( k=0; k<${#optB_list[@]}; k++ )); do
        optB=${optB_list[k]};

        for (( z=0; z<${test_times}; z++  )); do
          j=0;
          while [ $j -lt $retry_threshold ]; do

            timeout ${outLimit}m ./Eval/basic/mpi_subtask.sh taskN=${taskN} n=$N m=$M k=$K task=$task optB=$optB log_folder=$log_folder/;

            if [ $? -eq 0 ]; then
              break; 
            fi

            ./Eval/kill_all.sh frontend;
            j=$(expr $j + 1);
            if [ $j -eq $retry_threshold ]; then
              echo "Max retry: "${N}-${taskN} >> ${log_folder}/error.log;
            fi

          done;
        done;
      done;
    done;
  done;
done;

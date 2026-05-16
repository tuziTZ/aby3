# generate the parallel data.
MAIN_FOLDER=/root/aby3/aby3-GORAM/data
MAIN_RECORD=/root/aby3/aby3-GORAM/record

mpi_task_list=(16)

for mpi_task in ${mpi_task_list[@]}; do
    python ./GORAM/mpi_data_organization.py --target twitter --MPI_TASK $mpi_task --origional_folder $MAIN_FOLDER/realworld/ --mpi_folder $MAIN_FOLDER/realworld_mpi/
done

for mpi_task in ${mpi_task_list[@]}; do
    python ./GORAM/real_world_graph_analysis.py --target twitter --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/
done

# # run the advanced tasks for real_world data.
for mpi_task in ${mpi_task_list[@]}; do
    # python ./privGraph/advanced_benchmark.py --target cycle_detect --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/ --record_folder $MAIN_RECORD/realworld_adv/privGraph/

    python ./privGraph/advanced_benchmark.py --target two_hop --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/ --record_folder $MAIN_RECORD/realworld_adv/privGraph/

    # python ./privGraph/advanced_benchmark.py --target neighbor_stats --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/ --record_folder $MAIN_RECORD/realworld_adv/privGraph/

#     python ./privGraph/advanced_benchmark.py --target cycle_detect_edgelist --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/ --record_folder $MAIN_RECORD/realworld_adv/edgelist/

#     python ./privGraph/advanced_benchmark.py --target two_hop_edgelist --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/ --record_folder $MAIN_RECORD/realworld_adv/edgelist/

#     python ./privGraph/advanced_benchmark.py --target neighbor_stats_edgelist --MPI_TASK $mpi_task --data_folder $MAIN_FOLDER/realworld_mpi/ --record_folder $MAIN_RECORD/realworld_adv/edgelist/
done
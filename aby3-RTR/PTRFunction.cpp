#include "PTRFunction.h"
static std::string LOG_FOLDER="/root/aby3/Record/Record_index/";

int ptr_secret_index(int pIdx, std::vector<aby3::si64>& sharedM,
                 std::vector<aby3::si64>& secretIndex, std::vector<aby3::si64>& res,
                 aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                 aby3::Sh3Encryptor &enc){
  auto ptrTask = new SecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(TASKS, OPTIMAL_BLOCK, pIdx, enc, runtime, eval);
  size_t n = secretIndex.size(), m = sharedM.size();
  int* range_index = new int[m];
  for(int i=0; i<m; i++) range_index[i] = i;
  aby3::si64 dval;
  dval.mData[0] = 0, dval.mData[1] = 0;
  ptrTask->set_default_value(dval);
  ptrTask->circuit_construct({n}, {m});
  ptrTask->set_selective_value(sharedM.data(), 0);
  ptrTask->circuit_evaluate(secretIndex.data(), range_index, sharedM.data(), res.data());
}

int mpi_ptr_secret_index(int pIdx, std::vector<aby3::si64>& sharedM,
                 std::vector<aby3::si64>& secretIndex, std::vector<aby3::si64>& res,
                 aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                 aby3::Sh3Encryptor &enc, int task_num, int opt_B){

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &(rank));
  clock_t start, end;

  start = clock();
  auto mpiPtrTask = new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(task_num, opt_B, pIdx, enc, runtime, eval);
  size_t n = secretIndex.size(), m = sharedM.size();

  aby3::si64 dval;
  dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);
  mpiPtrTask->circuit_construct({n}, {m});

  size_t m_start = mpiPtrTask->m_start;
  size_t m_end = mpiPtrTask->m_end;

  std::string logging_file = LOG_FOLDER + "log-config-N=" + std::to_string(m) + "-M=" + std::to_string(n) + "-TASKS=" + std::to_string(task_num) + "-OPT_B=" + std::to_string(opt_B) + "-" + std::to_string(rank);

  int* range_index = new int[m_end - m_start + 1];
  for(int i=m_start; i<m_end + 1; i++) range_index[i-m_start] = i;
  end = clock();
  double time_data_prepare = double((end - start)*1000)/(CLOCKS_PER_SEC);

  std::vector<aby3::si64> partialM(m_end - m_start + 1);
  std::memcpy(partialM.data(), sharedM.data() + m_start, (m_end - m_start +1 )*sizeof(aby3::si64));

  // distributed task.
  start = clock();
  // mpiPtrTask->set_default_value(dval);
  // mpiPtrTask->circuit_construct({n}, {m});
  mpiPtrTask->set_selective_value(partialM.data(), 0);
  // mpiPtrTask->set_selective_value(sharedM.data(), 0);
  end = clock();
  double time_task_init = double((end - start)*1000/(CLOCKS_PER_SEC));

  start = clock();
  mpiPtrTask->circuit_evaluate(secretIndex.data(), range_index, partialM.data(), res.data());
  // mpiPtrTask->circuit_evaluate(secretIndex.data(), range_index, sharedM.data(), res.data());
  end = clock();
  double time_evaluate = double((end - start)*1000/(CLOCKS_PER_SEC));

  // cout << logging_file << endl;
  std::ofstream ofs(logging_file, std::ios_base::app);
  ofs << "time_data_prepare: " << std::setprecision(5) << time_data_prepare << "\ntime_task_init: " << std::setprecision(5) << time_task_init << "\ntime_task_evaluate: " << std::setprecision(5) << time_task_init << "\n" << std::endl;
  ofs.close();

  return 0;
}
#include "PTRProfile.h"
#include <chrono>
#include <thread>
#include "./Pair_then_Reduce/include/datatype.h"
#include "BuildingBlocks.h"
#include "PTRFunction.h"


int profile_task_setup(oc::CLP& cmd){
  clock_t start, end;
  // cout << "in function testing" << endl;
  int role = -1;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  start = clock();
  // setup communications.
  IOService ios;
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  end = clock();

  double time_task_setup = double((end - start) * 1000) / (CLOCKS_PER_SEC);

  static std::string LOG_FOLDER = "/root/aby3/Record/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  std::string logging_file = LOG_FOLDER + "task_setup";
  if(rank == 0){
    std::ofstream ofs(logging_file, std::ios_base::app);
    ofs << "task: " << size << "\n"
      << "time: " << std::setprecision(5) << time_task_setup << "\n"
      << std::endl;
  }
  return 0;
}


int profile_cipher_index(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){
  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_index/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";


  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask =
      new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
          1, vector_size_start, role, enc, runtime, eval);
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)vector_size_start});

  // data construct.
  m = vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<int> range_index(partial_len); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  mpiPtrTask->set_selective_value(vecM.data(), 0);
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<int> dataY = fake_repeat(
      range_index.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(start_time) +
                            " | ratio = " + to_string(last_ratio));
  write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask =
        new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
            1, vector_size, role, enc, runtime, eval);
    testMpiTask->circuit_construct({n}, {vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<int> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

    testMpiTask->set_default_value(dval);
    testMpiTask->set_selective_value(vecM_.data(), 0);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

    // cout << "last_ratio: " << last_ratio << " ratio: " << ratio << endl;
#ifdef LOGING
    // write_log(logging_file, "role: " + to_string(role) + "last_ratio: " + to_string(last_ratio) + " ratio: " + to_string(ratio));
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto testMpiTask =
        new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
            1, vector_size, role, enc, runtime, eval);
    testMpiTask->circuit_construct({n}, {vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<int> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

    testMpiTask->set_default_value(dval);
    testMpiTask->set_selective_value(vecM_.data(), 0);

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      // cout << "in test: " << i << endl;
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;
}


int profile_cipher_index_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){

  // set the config info.
  clock_t start, end;
  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_index/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 1;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask =
      new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
          size, vector_size_start, role, enc, runtime, eval);
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)size * vector_size_start});

  // data construct.
  m = vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<int> range_index(partial_len); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  mpiPtrTask->set_selective_value(vecM.data(), 0);
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<int> dataY = fake_repeat(
      range_index.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  if(rank == 0){
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(start_time) +
                            " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask =
        new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
            size, vector_size, role, enc, runtime, eval);
    testMpiTask->circuit_construct({n}, {size * vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<int> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

    testMpiTask->set_default_value(dval);
    testMpiTask->set_selective_value(vecM_.data(), 0);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = ;
    if (vector_size > 100000) repeats_ = 1;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

    // cout << "last_ratio: " << last_ratio << " ratio: " << ratio << endl;
#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(rank == 0) write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto testMpiTask =
        new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
            size, vector_size, role, enc, runtime, eval);
    testMpiTask->circuit_construct({n}, {size * vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<int> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

    testMpiTask->set_default_value(dval);
    testMpiTask->set_selective_value(vecM_.data(), 0);

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);
  // measure time
  auto testMpiTask =
      new MPISecretIndex<aby3::si64, int, aby3::si64, aby3::si64, SubIndex>(
          size, vector_size, role, enc, runtime, eval);
  testMpiTask->circuit_construct({n}, {size * vector_size});
  m = vector_size;

  // data construct
  size_t m_start_ = testMpiTask->m_start;
  size_t m_end_ = testMpiTask->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
  vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
  vector<int> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

  FakeArray<aby3::si64> dataX_ =
      fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
  FakeArray<int> dataY_ = fake_repeat(
      range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
      testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

  testMpiTask->set_default_value(dval);
  testMpiTask->set_selective_value(vecM_.data(), 0);

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                          testMpiTask->selectV);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;
}


int profile_average(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){
  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_average/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

   // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask =
      new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
          1, vector_size_start, role, enc, runtime, eval);
  mpiPtrTask->circuit_construct({(size_t) n}, {(size_t)vector_size_start});

  // data construct
  m = vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<int> x(1); x[0] = 0;
  vector<si64> res(n); vector_generation(role, enc, runtime, res);

  // construct the fake data
  FakeArray<int> dataX =
      fake_repeat(x.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<si64> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(start_time) +
                            " | ratio = " + to_string(last_ratio));
  write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask =
        new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
          1, vector_size, role, enc, runtime, eval);

    testMpiTask->circuit_construct({n}, {vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<int> x_(n); vector_generation(role, enc, runtime, x_);

    FakeArray<int> dataX_ =
        fake_repeat(x.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        vecM_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 20;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto testMpiTask =
        new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
          1, vector_size, role, enc, runtime, eval);

    testMpiTask->circuit_construct({n}, {vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<int> x_(n); vector_generation(role, enc, runtime, x_);

    FakeArray<int> dataX_ =
        fake_repeat(x.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        vecM_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      // cout << "in test: " << i << endl;
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;
}


int profile_average_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){

  // set the config info.
  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_average/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

   // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask =
      new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
          size, vector_size_start, role, enc, runtime, eval);
  mpiPtrTask->circuit_construct({(size_t) n}, {(size_t)size*vector_size_start});

  // data construct
  m = size*vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<int> x(1); x[0] = 0;
  vector<si64> res(n); vector_generation(role, enc, runtime, res);

  // construct the fake data
  FakeArray<int> dataX =
      fake_repeat(x.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<si64> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  if(rank == 0){
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(start_time) +
                            " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask =
        new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
          size, vector_size, role, enc, runtime, eval);

    testMpiTask->circuit_construct({n}, {size * vector_size});
    m = size*vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<int> x_(n); vector_generation(role, enc, runtime, x_);

    FakeArray<int> dataX_ =
        fake_repeat(x.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        vecM_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 100;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(rank == 0) write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto testMpiTask =
        new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
          size, vector_size, role, enc, runtime, eval);

    testMpiTask->circuit_construct({n}, {size * vector_size});
    m = size * vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<int> x_(n); vector_generation(role, enc, runtime, x_);

    FakeArray<int> dataX_ =
        fake_repeat(x.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        vecM_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      // cout << "in test: " << i << endl;
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);
  // further evaluate the optimal_b time.
  // measure time
  auto testMpiTask =
      new MPIAverage<int, aby3::si64, aby3::si64, aby3::si64, SubAvg>(
        size, vector_size, role, enc, runtime, eval);

  testMpiTask->circuit_construct({n}, {size * vector_size});
  m = vector_size;

  // data construct
  size_t m_start_ = testMpiTask->m_start;
  size_t m_end_ = testMpiTask->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
  vector<int> x_(n); vector_generation(role, enc, runtime, x_);

  FakeArray<int> dataX_ =
      fake_repeat(x.data(), testMpiTask->shapeX, testMpiTask->m, 0);
  FakeArray<si64> dataY_ = fake_repeat(
      vecM_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
      testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
  testMpiTask->set_default_value(dval);

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    // cout << "in test: " << i << endl;
    testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                          testMpiTask->selectV);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;
}


int profile_rank(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){
  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_rank/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          1, vector_size_start, role, enc, runtime, eval);
  
  if(vector_size_start > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size_start;
    m = n;
  }

  aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)n});

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<si64> range_index(partial_len); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<aby3::si64> dataY = fake_repeat(
      range_index.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  // double last_ratio = start_time / vector_size; double ratio = last_ratio / 2;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          1, vector_size, role, enc, runtime, eval);
    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }
    testMpiTask->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<si64> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto testMpiTask =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          1, vector_size, role, enc, runtime, eval);
    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }
    testMpiTask->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<si64> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      // cout << "in test: " << i << endl;
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;
}


int profile_rank_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_rank/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }

  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          size, vector_size_start, role, enc, runtime, eval);
  
  if(vector_size_start > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size_start;
    m = n;
  }

  aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)n});

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<si64> range_index(partial_len); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<aby3::si64> dataY = fake_repeat(
      range_index.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  // double last_ratio = start_time / vector_size; double ratio = last_ratio / 2;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  if(rank == 0) write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          size, vector_size, role, enc, runtime, eval);
    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }
    testMpiTask->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<si64> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 100000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto testMpiTask =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          size, vector_size, role, enc, runtime, eval);
    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }
    testMpiTask->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<si64> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<si64> dataY_ = fake_repeat(
        range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
    testMpiTask->set_default_value(dval);

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      // cout << "in test: " << i << endl;
      testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                            testMpiTask->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);
  // measure time
  auto testMpiTask =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        size, vector_size, role, enc, runtime, eval);
  if(vector_size > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size;
    m = n;
  }
  testMpiTask->circuit_construct({(size_t)n}, {(size_t)n});

  // data construct
  size_t m_start_ = testMpiTask->m_start;
  size_t m_end_ = testMpiTask->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<si64> vecM_(partial_len_); vector_generation(role, enc, runtime, vecM_);
  vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
  vector<si64> range_index_(partial_len_); vector_generation(role, enc, runtime, range_index_);

  FakeArray<aby3::si64> dataX_ =
      fake_repeat(vecIndex_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
  FakeArray<si64> dataY_ = fake_repeat(
      range_index_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
      testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);
  testMpiTask->set_default_value(dval);

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    // cout << "in test: " << i << endl;
    testMpiTask->subTask->circuit_profile(dataX_, dataY_,
                                          testMpiTask->selectV);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;
}


int profile_sort(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){
    // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_sort/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask_step1 =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          1, vector_size_start, role, enc, runtime, eval);
  auto mpiPtrTask_step2 =
      new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
          1, vector_size_start, role, enc, runtime, eval);
  
  if(vector_size_start > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size_start;
    m = n;
  }

  aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask_step1->set_default_value(dval);
  mpiPtrTask_step1->circuit_construct({(size_t)n}, {(size_t)n});
  mpiPtrTask_step2->set_default_value(dval);
  mpiPtrTask_step2->circuit_construct({(size_t)n}, {(size_t)n});

  size_t m_start = mpiPtrTask_step1->m_start; size_t m_end = mpiPtrTask_step1->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<int> range_index(n); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask_step1->shapeX, mpiPtrTask_step1->m, 0);
  FakeArray<aby3::si64> dataY = fake_repeat(
      vecM.data(), mpiPtrTask_step1->shapeY, mpiPtrTask_step1->n, 1, mpiPtrTask_step1->m_start,
      mpiPtrTask_step1->m_end - mpiPtrTask_step1->m_start + 1);
  
  mpiPtrTask_step2->set_selective_value(vecM.data(), 0);
  FakeArray<int> dataX2 =
      fake_repeat(range_index.data(), mpiPtrTask_step2->shapeX, mpiPtrTask_step2->m, 0);
  FakeArray<aby3::si64> dataY2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);
  FakeArray<aby3::si64> dataV2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++){
    mpiPtrTask_step1->subTask->circuit_profile(dataX, dataY, mpiPtrTask_step1->selectV);
    mpiPtrTask_step2->subTask->circuit_profile(dataX2, dataY2, dataV2);
  }
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  // double last_ratio = start_time / vector_size; double ratio = last_ratio / 2;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    if(vector_size == 4096) vector_size *= 2;
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        1, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        1, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(n); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 200;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        1, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        1, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(n); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;
}


int profile_sort_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_sort/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask_step1 =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          size, vector_size_start, role, enc, runtime, eval);
  auto mpiPtrTask_step2 =
      new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
          size, vector_size_start, role, enc, runtime, eval);
  
  if(vector_size_start > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size_start;
    m = n;
  }

  aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask_step1->set_default_value(dval);
  mpiPtrTask_step1->circuit_construct({(size_t)n}, {(size_t)n});
  mpiPtrTask_step2->set_default_value(dval);
  mpiPtrTask_step2->circuit_construct({(size_t)n}, {(size_t)n});

  size_t m_start = mpiPtrTask_step1->m_start; size_t m_end = mpiPtrTask_step1->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<int> range_index(n); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask_step1->shapeX, mpiPtrTask_step1->m, 0);
  FakeArray<aby3::si64> dataY = fake_repeat(
      vecM.data(), mpiPtrTask_step1->shapeY, mpiPtrTask_step1->n, 1, mpiPtrTask_step1->m_start,
      mpiPtrTask_step1->m_end - mpiPtrTask_step1->m_start + 1);
  
  mpiPtrTask_step2->set_selective_value(vecM.data(), 0);
  FakeArray<int> dataX2 =
      fake_repeat(range_index.data(), mpiPtrTask_step2->shapeX, mpiPtrTask_step2->m, 0);
  FakeArray<aby3::si64> dataY2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);
  FakeArray<aby3::si64> dataV2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++){
    mpiPtrTask_step1->subTask->circuit_profile(dataX, dataY, mpiPtrTask_step1->selectV);
    mpiPtrTask_step2->subTask->circuit_profile(dataX2, dataY2, dataV2);
  }
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  // double last_ratio = start_time / vector_size; double ratio = last_ratio / 2;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  if(rank == 0) write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    if(vector_size == 4096) vector_size *= 2;
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        size, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        size, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(n); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 200;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(rank == 0) write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        size, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        size, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)n}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(n); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);

  // measure time
  auto mpiPtrTask_step1_ =
  new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
      size, vector_size, role, enc, runtime, eval);
  auto mpiPtrTask_step2_ =
  new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
      size, vector_size, role, enc, runtime, eval);

  if(vector_size > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size;
    m = n;
  }

  aby3::si64 dval_; dval_.mData[0] = 0, dval_.mData[1] = 0;
  mpiPtrTask_step1_->set_default_value(dval_);
  mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
  mpiPtrTask_step2_->set_default_value(dval_);
  mpiPtrTask_step2_->circuit_construct({(size_t)n}, {(size_t)n});

  // data construct
  size_t m_start_ = mpiPtrTask_step1_->m_start;
  size_t m_end_ = mpiPtrTask_step1_->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
  vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
  vector<int> range_index_(n); vector_generation(role, enc, runtime, range_index_);

  // construct the fake data
  FakeArray<aby3::si64> dataX_ =
      fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
  FakeArray<aby3::si64> dataY_ = fake_repeat(
      vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
      mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
  
  mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
  FakeArray<int> dataX2_ =
      fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
  FakeArray<aby3::si64> dataY2_ = fake_repeat(
      vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
      mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
  FakeArray<aby3::si64> dataV2_ = fake_repeat(
      vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
      mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

  // adjusting the repeat time.
  // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
  if (vector_size > 50000) repeats_ = 10;

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                          mpiPtrTask_step1_->selectV);
    mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  
  return 0;
}


int profile_max(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){
    // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_max/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask_step1 =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          1, vector_size_start, role, enc, runtime, eval);
  auto mpiPtrTask_step2 =
      new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
          1, vector_size_start, role, enc, runtime, eval);
  
  if(vector_size_start > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size_start;
    m = n;
  }

  aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask_step1->set_default_value(dval);
  mpiPtrTask_step1->circuit_construct({(size_t)n}, {(size_t)n});
  mpiPtrTask_step2->set_default_value(dval);
  mpiPtrTask_step2->circuit_construct({(size_t)1}, {(size_t)n});

  size_t m_start = mpiPtrTask_step1->m_start; size_t m_end = mpiPtrTask_step1->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<int> range_index(1); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask_step1->shapeX, mpiPtrTask_step1->m, 0);
  FakeArray<aby3::si64> dataY = fake_repeat(
      vecM.data(), mpiPtrTask_step1->shapeY, mpiPtrTask_step1->n, 1, mpiPtrTask_step1->m_start,
      mpiPtrTask_step1->m_end - mpiPtrTask_step1->m_start + 1);
  
  mpiPtrTask_step2->set_selective_value(vecM.data(), 0);
  FakeArray<int> dataX2 =
      fake_repeat(range_index.data(), mpiPtrTask_step2->shapeX, mpiPtrTask_step2->m, 0);
  FakeArray<aby3::si64> dataY2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);
  FakeArray<aby3::si64> dataV2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++){
    mpiPtrTask_step1->subTask->circuit_profile(dataX, dataY, mpiPtrTask_step1->selectV);
    mpiPtrTask_step2->subTask->circuit_profile(dataX2, dataY2, dataV2);
  }
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  // double last_ratio = start_time / vector_size; double ratio = last_ratio / 2;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  if(role == 0)
    write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    if(vector_size == 4096) vector_size *= 2;
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        1, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        1, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)1}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(1); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 200;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(role == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(role == 0)
    write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        1, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        1, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)1}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(1); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(role == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;
}


int profile_max_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_max/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask_step1 =
      new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
          size, vector_size_start, role, enc, runtime, eval);
  auto mpiPtrTask_step2 =
      new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
          size, vector_size_start, role, enc, runtime, eval);
  
  if(vector_size_start > n*n){
    // std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size_start * size;
    m = n * size;
  }

  aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
  mpiPtrTask_step1->set_default_value(dval);
  mpiPtrTask_step1->circuit_construct({(size_t)n}, {(size_t)n});
  mpiPtrTask_step2->set_default_value(dval);
  mpiPtrTask_step2->circuit_construct({(size_t)1}, {(size_t)n});

  size_t m_start = mpiPtrTask_step1->m_start; size_t m_end = mpiPtrTask_step1->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<si64> vecM(partial_len); vector_generation(role, enc, runtime, vecM);
  vector<si64> vecIndex(n); vector_generation(role, enc, runtime, vecIndex);
  vector<int> range_index(1); vector_generation(role, enc, runtime, range_index);

  // construct the fake data
  FakeArray<aby3::si64> dataX =
      fake_repeat(vecIndex.data(), mpiPtrTask_step1->shapeX, mpiPtrTask_step1->m, 0);
  FakeArray<aby3::si64> dataY = fake_repeat(
      vecM.data(), mpiPtrTask_step1->shapeY, mpiPtrTask_step1->n, 1, mpiPtrTask_step1->m_start,
      mpiPtrTask_step1->m_end - mpiPtrTask_step1->m_start + 1);
  
  mpiPtrTask_step2->set_selective_value(vecM.data(), 0);
  FakeArray<int> dataX2 =
      fake_repeat(range_index.data(), mpiPtrTask_step2->shapeX, mpiPtrTask_step2->m, 0);
  FakeArray<aby3::si64> dataY2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);
  FakeArray<aby3::si64> dataV2 = fake_repeat(
      vecM.data(), mpiPtrTask_step2->shapeY, mpiPtrTask_step2->n, 1, mpiPtrTask_step2->m_start,
      mpiPtrTask_step2->m_end - mpiPtrTask_step2->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++){
    mpiPtrTask_step1->subTask->circuit_profile(dataX, dataY, mpiPtrTask_step1->selectV);
    mpiPtrTask_step2->subTask->circuit_profile(dataX2, dataY2, dataV2);
  }
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  // double last_ratio = start_time / vector_size; double ratio = last_ratio / 2;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;

  // first: 1) exp
  #ifdef LOGING
  if(rank == 0)
    write_log(logging_file, "EXP start");
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;
    // if(vector_size == 4096) vector_size *= 2;
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        size, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        size, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      // std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)1}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(1); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 200;
    if (vector_size > 50000) repeats_ = 1;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(rank == 0)
    write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // measure time
    auto mpiPtrTask_step1_ =
    new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
        size, vector_size, role, enc, runtime, eval);
    auto mpiPtrTask_step2_ =
    new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
        size, vector_size, role, enc, runtime, eval);

    if(vector_size > n*n){
      // std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
      n = vector_size;
      m = n;
    }

    aby3::si64 dval; dval.mData[0] = 0, dval.mData[1] = 0;
    mpiPtrTask_step1_->set_default_value(dval);
    mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
    mpiPtrTask_step2_->set_default_value(dval);
    mpiPtrTask_step2_->circuit_construct({(size_t)1}, {(size_t)n});

    // data construct
    size_t m_start_ = mpiPtrTask_step1_->m_start;
    size_t m_end_ = mpiPtrTask_step1_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
    vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
    vector<int> range_index_(1); vector_generation(role, enc, runtime, range_index_);

    // construct the fake data
    FakeArray<aby3::si64> dataX_ =
        fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
    FakeArray<aby3::si64> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
        mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
    
    mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
    FakeArray<int> dataX2_ =
        fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
    FakeArray<aby3::si64> dataY2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
    FakeArray<aby3::si64> dataV2_ = fake_repeat(
        vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
        mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 10;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_step1_->selectV);
      mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size;

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);

  // measure time
  auto mpiPtrTask_step1_ =
  new MPIRank<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubRank>(
      size, vector_size, role, enc, runtime, eval);
  auto mpiPtrTask_step2_ =
  new MPISecretIndex<int, aby3::si64, aby3::si64, aby3::si64, SubIndex>(
      size, vector_size, role, enc, runtime, eval);

  if(vector_size > n*n){
    std::cerr << "Warnning: profiling vector size is large than the table size, set the table size to corresponidng value." << std::endl;
    n = vector_size;
    m = n;
  }

  aby3::si64 dval_; dval_.mData[0] = 0, dval_.mData[1] = 0;
  mpiPtrTask_step1_->set_default_value(dval_);
  mpiPtrTask_step1_->circuit_construct({(size_t)n}, {(size_t)n});
  mpiPtrTask_step2_->set_default_value(dval);
  mpiPtrTask_step2_->circuit_construct({(size_t)1}, {(size_t)n});

  // data construct
  size_t m_start_ = mpiPtrTask_step1_->m_start;
  size_t m_end_ = mpiPtrTask_step1_->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<si64> vecM_(partial_len); vector_generation(role, enc, runtime, vecM_);
  vector<si64> vecIndex_(n); vector_generation(role, enc, runtime, vecIndex_);
  vector<int> range_index_(1); vector_generation(role, enc, runtime, range_index_);

  // construct the fake data
  FakeArray<aby3::si64> dataX_ =
      fake_repeat(vecIndex_.data(), mpiPtrTask_step1_->shapeX, mpiPtrTask_step1_->m, 0);
  FakeArray<aby3::si64> dataY_ = fake_repeat(
      vecM_.data(), mpiPtrTask_step1_->shapeY, mpiPtrTask_step1_->n, 1, mpiPtrTask_step1_->m_start,
      mpiPtrTask_step1_->m_end - mpiPtrTask_step1_->m_start + 1);
  
  mpiPtrTask_step2_->set_selective_value(vecM_.data(), 0);
  FakeArray<int> dataX2_ =
      fake_repeat(range_index_.data(), mpiPtrTask_step2_->shapeX, mpiPtrTask_step2_->m, 0);
  FakeArray<aby3::si64> dataY2_ = fake_repeat(
      vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
      mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);
  FakeArray<aby3::si64> dataV2_ = fake_repeat(
      vecM_.data(), mpiPtrTask_step2_->shapeY, mpiPtrTask_step2_->n, 1, mpiPtrTask_step2_->m_start,
      mpiPtrTask_step2_->m_end - mpiPtrTask_step2_->m_start + 1);

  // adjusting the repeat time.
  // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
  if (vector_size > 50000) repeats_ = 1;

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    mpiPtrTask_step1_->subTask->circuit_profile(dataX_, dataY_,
                                          mpiPtrTask_step1_->selectV);
    mpiPtrTask_step2->subTask->circuit_profile(dataX2_, dataY2_, dataV2_);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;
}


int profile_bio_metric(oc::CLP& cmd, size_t n, size_t m, size_t k, int vector_size_start, double epsilon, size_t gap){
  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_bio_metric/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                   aby3::si64, aby3::si64, SubBioMetric>(
      1, vector_size_start, role, enc, runtime, eval);
  // cout << "n: " << n << "m: " << m << endl; 
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)vector_size_start});

  // data construct.
  m = vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<vector<si64>> vecM(partial_len, vector<si64>(k)); vector_generation(role, enc, runtime, vecM);
  vector<vector<si64>> vecTarget(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget);

  FakeArray<vector<si64>> dataX =
      fake_repeat(vecTarget.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<vector<si64>> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);
    
  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  if(role == 0){
    write_log(logging_file,  "vector_size = " + to_string(vector_size) +
                              " | time = " + to_string(start_time) +
                              " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;

    // task setup.
    // write_log(logging_file, "role: " + to_string(role) + " before circuit_construct");
    auto mpiPtrTask_ = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubBioMetric>(
        1, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});
    // write_log(logging_file, "role: " + to_string(role) + " after circuit_construct");
    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
    mpiPtrTask_->set_default_value(dval);

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);
    // write_log(logging_file, "role: " + to_string(role) + " after vector construct");
    
    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    // write_log(logging_file, "role: " + to_string(role) + " after fake repeat");
    
    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

  #ifdef LOGING
    // write_log(logging_file, "role: " + to_string(role) + "last_ratio: " + to_string(last_ratio) + " ratio: " + to_string(ratio));
    if(role == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
  #endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(role == 0)
    write_log(logging_file, "BINARY start");
#endif

  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // task setup.
    auto mpiPtrTask_ = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubBioMetric>(
        1, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});

    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    
    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
    if (vector_size > 50000) repeats_ = 2;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

#ifdef LOGING
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;

}


int profile_bio_metric_mpi(oc::CLP& cmd, size_t n, size_t m, size_t k, int vector_size_start, double epsilon, size_t gap){

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_bio_metric/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                   aby3::si64, aby3::si64, SubBioMetric>(
      size, vector_size_start, role, enc, runtime, eval);
  // cout << "n: " << n << "m: " << m << endl; 
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)size * vector_size_start});

  // data construct.
  m = size * vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<vector<si64>> vecM(partial_len, vector<si64>(k)); vector_generation(role, enc, runtime, vecM);
  vector<vector<si64>> vecTarget(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget);

  FakeArray<vector<si64>> dataX =
      fake_repeat(vecTarget.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<vector<si64>> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);
    
  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  if(rank == 0){
    write_log(logging_file,  "vector_size = " + to_string(vector_size) +
                              " | time = " + to_string(start_time) +
                              " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;

    // task setup.
    // write_log(logging_file, "role: " + to_string(role) + " before circuit_construct");
    auto mpiPtrTask_ = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubBioMetric>(
        size, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});
    // write_log(logging_file, "role: " + to_string(role) + " after circuit_construct");
    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
    mpiPtrTask_->set_default_value(dval);

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);
    // write_log(logging_file, "role: " + to_string(role) + " after vector construct");
    
    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

  #ifdef LOGING
    // write_log(logging_file, "role: " + to_string(role) + "last_ratio: " + to_string(last_ratio) + " ratio: " + to_string(ratio));
    if(rank == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
  #endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(rank == 0)
    write_log(logging_file, "BINARY start");
#endif

  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // task setup.
    auto mpiPtrTask_ = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubBioMetric>(
        size, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});

    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    
    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
    if (vector_size > 50000) repeats_ = 2;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

#ifdef LOGING
    if(rank == 0){
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);

  // task setup.
  auto mpiPtrTask_ = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
                                  aby3::si64, aby3::si64, SubBioMetric>(
      size, vector_size, role, enc, runtime, eval);
  mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});

  // data construct.
  m = size * vector_size;
  // aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
  vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

  FakeArray<vector<si64>> dataX_ =
      fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
  FakeArray<vector<si64>> dataY_ = fake_repeat(
      vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
      mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
  
  // adjusting the repeat time.
  // if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
  if (vector_size > 50000) repeats_ = 2;

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                          mpiPtrTask_->selectV);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;

}


int profile_metric_mpi(oc::CLP& cmd, size_t n, size_t m, size_t k, int vector_size_start, double epsilon, size_t gap){

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_metric/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 5;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask = new MPIMetric<vector<aby3::si64>, vector<aby3::si64>,
                                   indData<aby3::si64>, indData<aby3::si64>, SubMetric>(
     size, vector_size_start, role, enc, runtime, eval);
  // auto mpiPtrTask = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
  //                                  aby3::si64, aby3::si64, SubBioMetric>(
  //     size, vector_size_start, role, enc, runtime, eval);
  // cout << "n: " << n << "m: " << m << endl; 
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)size * vector_size_start});

  // data construct.
  m = size * vector_size_start;
  indData<aby3::si64> dData;
  aby3::si64 dval;
  if(role == 0){
    dval.mData[0] = 2<<32, dval.mData[1] = 0;
  }
  else if(role == 1){
    dval.mData[1] = 2<<32, dval.mData[0] = 0;
  }
  else{
    dval.mData[1] = 0, dval.mData[0] = 0;
  }

  dData.value = dval;
  dData.index = dval;
  mpiPtrTask->set_default_value(dData);

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<vector<si64>> vecM(partial_len, vector<si64>(k)); vector_generation(role, enc, runtime, vecM);
  vector<vector<si64>> vecTarget(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget);

  FakeArray<vector<si64>> dataX =
      fake_repeat(vecTarget.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<vector<si64>> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);
    
  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  if(rank == 0){
    write_log(logging_file,  "vector_size = " + to_string(vector_size) +
                              " | time = " + to_string(start_time) +
                              " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;

    // task setup.
    // write_log(logging_file, "role: " + to_string(role) + " before circuit_construct");
    // auto mpiPtrTask = new MPIMetric<vector<aby3::si64>, vector<aby3::si64>,
    //                                indData<aby3::si64>, indData<aby3::si64>, SubMetric>(
    //  size, vector_size_start, role, enc, runtime, eval);
    auto mpiPtrTask_ = new MPIMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    indData<aby3::si64>, indData<aby3::si64>, SubMetric>(
        size, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});
    // data construct.
    m = vector_size;
    // aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
    mpiPtrTask_->set_default_value(dData);

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);
    // write_log(logging_file, "role: " + to_string(role) + " after vector construct");
    
    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 1;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

  #ifdef LOGING
    // write_log(logging_file, "role: " + to_string(role) + "last_ratio: " + to_string(last_ratio) + " ratio: " + to_string(ratio));
    if(rank == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
  #endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(rank == 0)
    write_log(logging_file, "BINARY start");
#endif

  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // task setup.
    auto mpiPtrTask_ = new MPIMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    indData<aby3::si64>, indData<aby3::si64>, SubMetric>(
        size, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});
    mpiPtrTask_->set_default_value(dData);
    // data construct.
    // m = vector_size;
    // aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    
    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
    if (vector_size > 50000) repeats_ = 1;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

#ifdef LOGING
  if(rank == 0){
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
  }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
    // else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);

  // task setup.
  auto mpiPtrTask_ = new MPIMetric<vector<aby3::si64>, vector<aby3::si64>,
                                    indData<aby3::si64>, indData<aby3::si64>, SubMetric>(
        size, vector_size, role, enc, runtime, eval);
  mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});
  mpiPtrTask_->set_default_value(dData);
  // auto mpiPtrTask_ = new MPIBioMetric<vector<aby3::si64>, vector<aby3::si64>,
  //                                 aby3::si64, aby3::si64, SubBioMetric>(
  //     size, vector_size, role, enc, runtime, eval);
  // mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)size * vector_size});

  // data construct.
  // m = size * vector_size;
  // aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  // vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
  vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

  FakeArray<vector<si64>> dataX_ =
      fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
  FakeArray<vector<si64>> dataY_ = fake_repeat(
      vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
      mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
  
  // adjusting the repeat time.
  // if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
  if (vector_size > 50000) repeats_ = 1;

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                          mpiPtrTask_->selectV);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;

}


int profile_mean_distance(oc::CLP& cmd, size_t n, size_t m, size_t k, int vector_size_start, double epsilon, size_t gap){
  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mean_distance/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, 0, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                   aby3::si64, aby3::si64, SubMeanDis>(
      1, vector_size_start, role, enc, runtime, eval);
  // cout << "n: " << n << "m: " << m << endl; 
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)vector_size_start});

  // data construct.
  m = vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<vector<si64>> vecM(partial_len, vector<si64>(k)); vector_generation(role, enc, runtime, vecM);
  vector<vector<si64>> vecTarget(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget);

  FakeArray<vector<si64>> dataX =
      fake_repeat(vecTarget.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<vector<si64>> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);
    
  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  if(role == 0){
    write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(start_time) +
                            " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;

    // task setup.
    write_log(logging_file, "role: " + to_string(role) + " before circuit_construct");
    auto mpiPtrTask_ = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubMeanDis>(
        1, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});
    // write_log(logging_file, "role: " + to_string(role) + " after circuit_construct");
    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
    mpiPtrTask_->set_default_value(dval);

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);
    // write_log(logging_file, "role: " + to_string(role) + " after vector construct");
    
    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    // write_log(logging_file, "role: " + to_string(role) + " after fake repeat");
    
    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

  #ifdef LOGING
    // write_log(logging_file, "role: " + to_string(role) + "last_ratio: " + to_string(last_ratio) + " ratio: " + to_string(ratio));
    if(role == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
  #endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(role == 0)
    write_log(logging_file, "BINARY start");
#endif

  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // task setup.
    auto mpiPtrTask_ = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubMeanDis>(
        1, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});

    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    
    // adjusting the repeat time.
    if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
    if (vector_size > 50000) repeats_ = 2;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

#ifdef LOGING
    if(role == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
    // else break;
  }

  size_t optimal_b = (size_t)((vector_start + vector_end) / 2);

  // save the optimalB to config file.
  std::ofstream resfs(profiler_file, std::ios_base::app);
  resfs << "optimal_B: " << optimal_b << std::endl;
  resfs.close();
  
  return 0;

}


int profile_mean_distance_mpi(oc::CLP& cmd, size_t n, size_t m, size_t k, int vector_size_start, double epsilon, size_t gap){

  int rank, size;  
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER = "/root/aby3/Record/Prof_mpi_mean_distance/";
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  }
  // cout << "LOG_FOLDER: " << LOG_FOLDER << endl;
  std::string logging_file = LOG_FOLDER + "probe.log";
  std::string profiler_file = LOG_FOLDER + "probe.res";

  // environment setup.
  int role = -1;
  int repeats_ = 100;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }
  if (cmd.isSet("repeats")) {
    auto keys = cmd.getMany<int>("repeats");
    repeats_ = keys[0];
  }

  // setup communications.
  IOService ios; Sh3Encryptor enc; Sh3Evaluator eval; Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                   aby3::si64, aby3::si64, SubMeanDis>(
      size, vector_size_start, role, enc, runtime, eval);
  // cout << "n: " << n << "m: " << m << endl; 
  mpiPtrTask->circuit_construct({(size_t)n}, {(size_t)vector_size_start});

  // data construct.
  m = vector_size_start;
  aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
  mpiPtrTask->set_default_value(dval);

  size_t m_start = mpiPtrTask->m_start; size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<si64> res(n); vector_generation(role, enc, runtime, res);
  vector<vector<si64>> vecM(partial_len, vector<si64>(k)); vector_generation(role, enc, runtime, vecM);
  vector<vector<si64>> vecTarget(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget);

  FakeArray<vector<si64>> dataX =
      fake_repeat(vecTarget.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<vector<si64>> dataY = fake_repeat(
      vecM.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);
    
  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++) mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();

  // time synchronized.
  double start_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size; double ratio = last_ratio;
  bool start_flag = true;
  // first: 1) exp
  #ifdef LOGING
  if(role == 0){
    write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(start_time) +
                            " | ratio = " + to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
  #endif

  while((ratio < last_ratio) || start_flag){
    start_flag = false;
    vector_size *= 2;

    // task setup.
    // write_log(logging_file, "role: " + to_string(role) + " before circuit_construct");
    auto mpiPtrTask_ = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubMeanDis>(
        size, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});
    // write_log(logging_file, "role: " + to_string(role) + " after circuit_construct");
    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;
    mpiPtrTask_->set_default_value(dval);

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);
    // write_log(logging_file, "role: " + to_string(role) + " after vector construct");
    
    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    // write_log(logging_file, "role: " + to_string(role) + " after fake repeat");
    
    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 500;
    if (vector_size > 50000) repeats_ = 5;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

  #ifdef LOGING
    // write_log(logging_file, "role: " + to_string(role) + "last_ratio: " + to_string(last_ratio) + " ratio: " + to_string(ratio));
    if(role == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
  #endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if(role == 0)
    write_log(logging_file, "BINARY start");
#endif

  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    // task setup.
    auto mpiPtrTask_ = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                    aby3::si64, aby3::si64, SubMeanDis>(
        size, vector_size, role, enc, runtime, eval);
    mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});

    // data construct.
    m = vector_size;
    aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

    size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
    vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
    vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

    FakeArray<vector<si64>> dataX_ =
        fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
    FakeArray<vector<si64>> dataY_ = fake_repeat(
        vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
        mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
    
    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
    if (vector_size > 50000) repeats_ = 2;

    start = clock();
    for (int i = 0; i < repeats_; i++) {
      mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                            mpiPtrTask_->selectV);
    }
    end = clock();

    double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
    synchronized_time(role, double_time, runtime);
    MPI_Bcast(&double_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    last_ratio = ratio;
    ratio = double_time / vector_size; 

#ifdef LOGING
    if(role == 0){
      write_log(logging_file, "vector_size = " + to_string(vector_size) +
                            " | time = " + to_string(double_time) +
                            " | ratio = " + to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio) vector_end = vector_size;
    else break;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);

  // task setup.
  auto mpiPtrTask_ = new MPIMeanDis<vector<aby3::si64>, vector<aby3::si64>,
                                  aby3::si64, aby3::si64, SubMeanDis>(
      size, vector_size, role, enc, runtime, eval);
  mpiPtrTask_->circuit_construct({(size_t)n}, {(size_t)vector_size});

  // data construct.
  m = vector_size;
  // aby3::si64 dval; dval.mData[0] = 0; dval.mData[1] = 0;

  size_t m_start_ = mpiPtrTask_->m_start; size_t m_end_ = mpiPtrTask_->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<si64> res_(n); vector_generation(role, enc, runtime, res_);
  vector<vector<si64>> vecM_(partial_len_, vector<si64>(k)); vector_generation(role, enc, runtime, vecM_);
  vector<vector<si64>> vecTarget_(n, vector<si64>(k)); vector_generation(role, enc, runtime, vecTarget_);

  FakeArray<vector<si64>> dataX_ =
      fake_repeat(vecTarget_.data(), mpiPtrTask_->shapeX, mpiPtrTask_->m, 0);
  FakeArray<vector<si64>> dataY_ = fake_repeat(
      vecM_.data(), mpiPtrTask_->shapeY, mpiPtrTask_->n, 1, mpiPtrTask_->m_start,
      mpiPtrTask_->m_end - mpiPtrTask_->m_start + 1);
  
  // adjusting the repeat time.
  // if (vector_size > 5000 && vector_size < 50000) repeats_ = 100;
  if (vector_size > 50000) repeats_ = 10;

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    mpiPtrTask_->subTask->circuit_profile(dataX_, dataY_,
                                          mpiPtrTask_->selectV);
  }
  end = clock();

  double_time = double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, double_time, runtime);  

  // save the optimalB to config file.
  if(rank == 0){
    std::ofstream resfs(profiler_file, std::ios_base::app);
    resfs << "optimal_B: " << vector_size << std::endl;
    resfs << "exec time: " << double_time << std::endl;
    resfs << "ratio: " << (double_time / vector_size) << std::endl;
    resfs.close();
  }
  
  return 0;

}


int profile_new_search_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start, double epsilon, size_t gap){
  // cout << 'in this func' << endl;
  return abs_profile<aby3::si64, aby3::si64, aby3::si64, aby3::si64, SubNewSearch, MPINewSearch>(cmd, n, m, 1, vector_size_start, epsilon, gap);
}
#pragma once
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>
#include <cmath>
#include "PTRFunction.h"

#define LOGING

using namespace oc;
using namespace aby3;
using namespace std;

int profile_task_setup(oc::CLP& cmd);

int profile_cipher_index(oc::CLP& cmd, size_t n, size_t m,
                         int vector_size_start, double epsilon, size_t gap);

int profile_cipher_index_mpi(oc::CLP& cmd, size_t n, size_t m,
                             int vector_size_start, double epsilon, size_t gap);

int profile_average(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                    double epsilon, size_t gap);

int profile_average_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                        double epsilon, size_t gap);

int profile_rank(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                 double epsilon, size_t gap);

int profile_rank_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                     double epsilon, size_t gap);

int profile_sort(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                 double epsilon, size_t gap);

int profile_sort_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                     double epsilon, size_t gap);

int profile_max(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                double epsilon, size_t gap);

int profile_max_mpi(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                    double epsilon, size_t gap);

int profile_bio_metric(oc::CLP& cmd, size_t n, size_t m, size_t k,
                       int vector_size_start, double epsilon, size_t gap);

int profile_bio_metric_mpi(oc::CLP& cmd, size_t n, size_t m, size_t k,
                           int vector_size_start, double epsilon, size_t gap);

int profile_mean_distance(oc::CLP& cmd, size_t n, size_t m, size_t k,
                          int vector_size_start, double epsilon, size_t gap);

int profile_mean_distance_mpi(oc::CLP& cmd, size_t n, size_t m, size_t k,
                              int vector_size_start, double epsilon,
                              size_t gap);

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK,
          template <typename, typename, typename, typename,
                    template <typename, typename, typename, typename> class>
          class MPIType>
int abs_profile(oc::CLP& cmd, size_t n, size_t m, size_t k,
                int vector_size_start, double epsilon, size_t gap) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // set the config info.
  clock_t start, end;
  // set the log file.
  std::string LOG_FOLDER;
  if (cmd.isSet("logFolder")) {
    auto keys = cmd.getMany<std::string>("logFolder");
    LOG_FOLDER = keys[0];
  } else {
    throw std::runtime_error("logFolder not defined. ");
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
  IOService ios;
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;
  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
  // task setup.
  auto mpiPtrTask = new MPIType<NUMX, NUMY, NUMT, NUMR, TASK>(
      size, vector_size_start, role, enc, runtime, eval);
  if (std::is_same<TASK<NUMX, NUMY, NUMT, NUMR>,
                   SubNewSearch<NUMX, NUMY, NUMT, NUMR>>::value) {
    mpiPtrTask->set_lookahead(1);
  }
  // cout << "in this funct" << endl;
  // data construct.
  m = vector_size_start;
  NUMR dval;
  if (std::is_same<NUMR, aby3::si64>::value) {
    dval.mData[0] = 0;
    dval.mData[1] = 0;
  } else {
    throw std::runtime_error(
        "Currently do not support NUMR types except aby3::si64");
  }
  mpiPtrTask->set_default_value(dval);
  mpiPtrTask->circuit_construct({(size_t)n},
                                {(size_t)(size * vector_size_start)});

  size_t m_start = mpiPtrTask->m_start;
  size_t m_end = mpiPtrTask->m_end;
  size_t partial_len = m_end - m_start + 1;

  vector<NUMR> res(n);
  vector_generation(role, enc, runtime, res);
  vector<NUMY> vecY;
  vector<NUMX> vecX;
  if (std::is_same<NUMY, std::vector<typename NUMY::value_type>>::value) {
    throw std::runtime_error("high dimensional not supported yet...");
  } else {
    vecY.resize(partial_len);
    vector_generation(role, enc, runtime, vecY);
  }
  if (std::is_same<NUMX, std::vector<typename NUMX::value_type>>::value) {
    throw std::runtime_error("high dimensional not supported yet...");
  } else {
    vecX.resize(n);
    vector_generation(role, enc, runtime, vecX);
  }
  mpiPtrTask->set_selective_value(vecY.data(), 0);
  FakeArray<NUMX> dataX =
      fake_repeat(vecX.data(), mpiPtrTask->shapeX, mpiPtrTask->m, 0);
  FakeArray<NUMY> dataY = fake_repeat(
      vecY.data(), mpiPtrTask->shapeY, mpiPtrTask->n, 1, mpiPtrTask->m_start,
      mpiPtrTask->m_end - mpiPtrTask->m_start + 1);

  // begin the profiler: 0) set the initial
  start = clock();
  for (int i = 0; i < repeats_; i++)
    mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  // mpiPtrTask->subTask->circuit_profile(dataX, dataY, mpiPtrTask->selectV);
  end = clock();
  // time synchronized.
  double start_time =
      double((end - start) * 1000) / (repeats_ * CLOCKS_PER_SEC);
  synchronized_time(role, start_time, runtime);
  MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double double_time = -1;
  size_t vector_size = vector_size_start;
  double last_ratio = start_time / vector_size;
  double ratio = last_ratio;
  bool start_flag = true;

// first: 1) exp
#ifdef LOGING
  if (rank == 0) {
    write_log(logging_file, "role: " + to_string(role) + "vector_size = " +
                                to_string(vector_size) + " | time = " +
                                to_string(start_time) + " | ratio = " +
                                to_string(last_ratio));
    write_log(logging_file, "EXP start");
  }
#endif

  while ((ratio < last_ratio) || start_flag) {
    start_flag = false;
    vector_size *= 2;
    auto testMpiTask = new MPIType<NUMX, NUMY, NUMT, NUMR, TASK>(
        size, vector_size, role, enc, runtime, eval);
    if (std::is_same<TASK<NUMX, NUMY, NUMT, NUMR>,
                     SubNewSearch<NUMX, NUMY, NUMT, NUMR>>::value) {
      testMpiTask->set_lookahead(1);
    }
    testMpiTask->circuit_construct({n}, {size * vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<NUMR> res_(n);
    vector_generation(role, enc, runtime, res);
    vector<NUMY> vecY_;
    vector<NUMX> vecX_;
    if (std::is_same<NUMY, std::vector<typename NUMY::value_type>>::value) {
      throw std::runtime_error("high dimensional not supported yet...");
    } else {
      vecY_.resize(partial_len_);
      vector_generation(role, enc, runtime, vecY_);
    }
    if (std::is_same<NUMX, std::vector<typename NUMX::value_type>>::value) {
      throw std::runtime_error("high dimensional not supported yet...");
    } else {
      vecX_.resize(n);
      vector_generation(role, enc, runtime, vecX_);
    }
    testMpiTask->set_selective_value(vecY_.data(), 0);
    FakeArray<NUMX> dataX_ =
        fake_repeat(vecX_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<NUMY> dataY_ = fake_repeat(
        vecY_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = ;
    if (vector_size > 100000) repeats_ = 20;

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
    if (rank == 0) {
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " +
                                  to_string(vector_size) + " | time = " +
                                  to_string(double_time) + " | ratio = " +
                                  to_string(double_time / vector_size));
    }
#endif
  }

  // second: 2) binary search
  size_t vector_start = vector_size / 2;
  size_t vector_end = vector_size;

#ifdef LOGING
  if (rank == 0) write_log(logging_file, "BINARY start");
#endif
  while ((vector_end - vector_start) > gap) {
    vector_size = (size_t)(vector_start + vector_end) / 2;

    auto testMpiTask = new MPIType<NUMX, NUMY, NUMT, NUMR, TASK>(
        size, vector_size, role, enc, runtime, eval);
    if (std::is_same<TASK<NUMX, NUMY, NUMT, NUMR>,
                     SubNewSearch<NUMX, NUMY, NUMT, NUMR>>::value) {
      testMpiTask->set_lookahead(1);
    }
    testMpiTask->circuit_construct({n}, {size * vector_size});
    m = vector_size;

    // data construct
    size_t m_start_ = testMpiTask->m_start;
    size_t m_end_ = testMpiTask->m_end;
    size_t partial_len_ = m_end_ - m_start_ + 1;

    vector<NUMR> res_(n);
    vector_generation(role, enc, runtime, res);
    vector<NUMY> vecY_;
    vector<NUMX> vecX_;
    if (std::is_same<NUMY, std::vector<typename NUMY::value_type>>::value) {
      throw std::runtime_error("high dimensional not supported yet...");
    } else {
      vecY_.resize(partial_len_);
      vector_generation(role, enc, runtime, vecY_);
    }
    if (std::is_same<NUMX, std::vector<typename NUMX::value_type>>::value) {
      throw std::runtime_error("high dimensional not supported yet...");
    } else {
      vecX_.resize(n);
      vector_generation(role, enc, runtime, vecX_);
    }
    testMpiTask->set_selective_value(vecY_.data(), 0);
    FakeArray<NUMX> dataX_ =
        fake_repeat(vecX_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
    FakeArray<NUMY> dataY_ = fake_repeat(
        vecY_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
        testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

    // adjusting the repeat time.
    // if (vector_size > 5000 && vector_size < 50000) repeats_ = ;
    if (vector_size > 100000) repeats_ = 20;

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
    if (rank == 0) {
      write_log(logging_file, "role: " + to_string(role) + "vector_size = " +
                                  to_string(vector_size) + " | time = " +
                                  to_string(double_time) + " | ratio = " +
                                  to_string(double_time / vector_size));
    }
#endif

    if (ratio > last_ratio)
      vector_end = vector_size;
    else
      vector_start = vector_size;
  }

  vector_size = (size_t)((vector_start + vector_end) / 2);
  auto testMpiTask = new MPIType<NUMX, NUMY, NUMT, NUMR, TASK>(
      size, vector_size, role, enc, runtime, eval);
  if (std::is_same<TASK<NUMX, NUMY, NUMT, NUMR>,
                   SubNewSearch<NUMX, NUMY, NUMT, NUMR>>::value) {
    testMpiTask->set_lookahead(1);
  }
  testMpiTask->circuit_construct({n}, {size * vector_size});
  m = vector_size;

  // data construct
  size_t m_start_ = testMpiTask->m_start;
  size_t m_end_ = testMpiTask->m_end;
  size_t partial_len_ = m_end_ - m_start_ + 1;

  vector<NUMR> res_(n);
  vector_generation(role, enc, runtime, res);
  vector<NUMY> vecY_;
  vector<NUMX> vecX_;
  if (std::is_same<NUMY, std::vector<typename NUMY::value_type>>::value) {
    throw std::runtime_error("high dimensional not supported yet...");
  } else {
    vecY_.resize(partial_len);
    vector_generation(role, enc, runtime, vecY_);
  }
  if (std::is_same<NUMX, std::vector<typename NUMX::value_type>>::value) {
    throw std::runtime_error("high dimensional not supported yet...");
  } else {
    vecX_.resize(n);
    vector_generation(role, enc, runtime, vecX_);
  }
  testMpiTask->set_selective_value(vecY_.data(), 0);
  FakeArray<NUMX> dataX_ =
      fake_repeat(vecX_.data(), testMpiTask->shapeX, testMpiTask->m, 0);
  FakeArray<NUMY> dataY_ = fake_repeat(
      vecY_.data(), testMpiTask->shapeY, testMpiTask->n, 1,
      testMpiTask->m_start, testMpiTask->m_end - testMpiTask->m_start + 1);

  // adjusting the repeat time.
  // if (vector_size > 5000 && vector_size < 50000) repeats_ = ;
  if (vector_size > 100000) repeats_ = 20;

  start = clock();
  for (int i = 0; i < repeats_; i++) {
    testMpiTask->subTask->circuit_profile(dataX_, dataY_, testMpiTask->selectV);
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

int profile_new_search_mpi(oc::CLP& cmd, size_t n, size_t m,
                           int vector_size_start, double epsilon, size_t gap);

int profile_metric_mpi(oc::CLP& cmd, size_t n, size_t m, size_t k,
                           int vector_size_start, double epsilon, size_t gap);
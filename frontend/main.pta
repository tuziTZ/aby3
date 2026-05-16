
#include <cryptoTools/Common/CLP.h>
#include <map>

#include <aby3-ML/main-linear.h>
#include <aby3-ML/main-logistic.h>
#include <mpi.h>
#include "aby3-RTR/DistributeRTRTest.h"
#include "aby3-RTR/PTRTest.h"
#include "aby3-RTR/RTRTest.h"
#include "aby3-RTR/PTRProfile.h"
#include "aby3-RTR/GASTest.h"
#include "aby3-RTR/Test.h"
#include "eric.h"

#define MPI
// #define MPIDEBUG

using namespace oc;
using namespace aby3;

#ifdef MPI
int main(int argc, char** argv) {
  oc::CLP cmd(argc, argv);
  // reinit the environment and then finalize the environment.
  MPI_Init(&argc, &argv);


  if (cmd.isSet("Test")){
    basic_test(cmd);
    return 0;
  }

  size_t N, M, TASK_NUM, OPT_BLOCK;
  std::string FUNC;
  if (cmd.isSet("N")) {
    auto keys = cmd.getMany<size_t>("N");
    if(keys.size() != 0) N = keys[0];
  } else {
    throw std::runtime_error("No N defined");
  }
  if (cmd.isSet("M")) {
    auto keys = cmd.getMany<size_t>("M");
    if(keys.size() != 0) M = keys[0];
  } else {
    throw std::runtime_error("No M defined");
  }
  if (cmd.isSet("TASK_NUM")) {
    auto keys = cmd.getMany<size_t>("TASK_NUM");
    if(keys.size() != 0) TASK_NUM = keys[0];
  } else {
    throw std::runtime_error("No TASK_NUM defined");
  }
  if (cmd.isSet("OPT_BLOCK")) {
    auto keys = cmd.getMany<size_t>("OPT_BLOCK");
    if(keys.size() != 0) OPT_BLOCK = keys[0];
  } else {
    throw std::runtime_error("No OPT_BLOCK defined");
  }

  if(cmd.isSet("FUNC")) {
    auto keys = cmd.getMany<std::string>("FUNC");
    if(keys.size() != 0) FUNC = keys[0];
    else throw std::runtime_error("No FUNC defined");
  } else {
    FUNC = "index";
  }

  // cout << "in the main " << endl;
  // std::cout << "FUNC: " << FUNC << std::endl;
  // check the config task_num is the same as the MPI task num
  int task_size;
  MPI_Comm_size(MPI_COMM_WORLD, &task_size);
  if (TASK_NUM != task_size) {
    throw std::runtime_error("MPI task number: " + std::to_string(task_size) +
                             " != config task size: " +
                             std::to_string(TASK_NUM));
  }

  if (FUNC == "cipher_index")
    test_cipher_index_ptr_mpi(cmd, N, M, TASK_NUM, OPT_BLOCK);
  
  if (FUNC == "average")
    test_cipher_average_ptr_mpi(cmd, N, M, TASK_NUM, OPT_BLOCK);

  if (FUNC == "rank") 
		test_cipher_rank_ptr_mpi(cmd, N, TASK_NUM, OPT_BLOCK);

	if (FUNC == "search") 
		test_cipher_search_ptr_mpi(cmd, N, M, TASK_NUM, OPT_BLOCK);
  
  if (FUNC == "new_search") 
		test_cipher_search_new_ptr_mpi(cmd, N, M, TASK_NUM, OPT_BLOCK);
  
  if (FUNC == "select")
    test_cipher_select_ptr_mpi(cmd, N, M, TASK_NUM, OPT_BLOCK);

	if(FUNC == "vector")
    test_vectorization(cmd, N, TASK_NUM);
  
  if(FUNC == "profile_index"){
    if(task_size != 1){
      std::runtime_error("For profiling, the task_size can only be 1, instead of : " + to_string(task_size));
    }
    double epsilon = 5;
    size_t gap = 100;
    size_t vec_start = 1 << 5;
    if(cmd.isSet("EPSILON")) {
      auto keys = cmd.getMany<double>("EPSILON");
      epsilon = keys[0];
    }
    if(cmd.isSet("GAP")) {
      auto keys = cmd.getMany<size_t>("GAP");
      gap = keys[0];
    }
    if(cmd.isSet("VEC_START")){
      auto keys = cmd.getMany<size_t>("VEC_START");
      vec_start = keys[0];
    }
    probe_profile_index(cmd, N, M, vec_start, epsilon, gap);
  }

  if(FUNC == "mean_distance"){
    int k = -1;
    if(cmd.isSet("K")){
      auto keys = cmd.getMany<int>("K");
      k = keys[0];
    }
    if(k < 0){
      throw std::runtime_error("For high-dimensional test case: " + FUNC +
                             " K must be setted, while K = " +
                             std::to_string(k));
    }
    test_cipher_mean_distance(cmd, N, M, k, TASK_NUM, OPT_BLOCK);
  }

  if(FUNC == "bio_metric"){
    int k = -1;
    if(cmd.isSet("K")){
      auto keys = cmd.getMany<int>("K");
      k = keys[0];
    }
    if(k < 0){
      throw std::runtime_error("For high-dimensional test case: " + FUNC +
                             " K must be setted, while K = " +
                             std::to_string(k));
    }
    // cout << "can in this block!" << endl;
    test_cipher_bio_metric(cmd, N, M, k, TASK_NUM, OPT_BLOCK);
  }

  if(FUNC == "metric"){
    int k = -1;
    if(cmd.isSet("K")){
      auto keys = cmd.getMany<int>("K");
      k = keys[0];
    }
    if(k < 0){
      throw std::runtime_error("For high-dimensional test case: " + FUNC +
                             " K must be setted, while K = " +
                             std::to_string(k));
    }
    // cout << "can in this block!" << endl;
    test_cipher_metric(cmd, N, M, k, TASK_NUM, OPT_BLOCK);
  }

  // multi-step functions.
  if(FUNC == "sort"){
    test_cipher_sort_ptr_mpi(cmd, N, TASK_NUM, OPT_BLOCK);
  }

  if(FUNC == "max"){
    test_cipher_max_ptr_mpi(cmd, N, TASK_NUM, OPT_BLOCK);
  }

  if(FUNC == "min"){
    test_cipher_min_ptr_mpi(cmd, N, TASK_NUM, OPT_BLOCK);
  }

  if(FUNC == "medium"){
    test_cipher_medium_ptr_mpi(cmd, N, TASK_NUM, OPT_BLOCK);
  }

  // gas test functions.
  if(FUNC == "page_rank"){
    std::string data_folder; get_value("data_folder", cmd, data_folder);
    test_page_rank(cmd, data_folder);
  }

  // profile...
  std::string start_prefix = FUNC.substr(0, 4);
  // cout << "func: " << FUNC << endl;
  if(start_prefix == "prof"){

    double epsilon = 5;
    size_t gap = 1000;
    size_t vec_start = 1 << 5;
    if(cmd.isSet("EPSILON")) {
      auto keys = cmd.getMany<double>("EPSILON");
      if(keys.size() != 0) epsilon = keys[0];
    }
    if(cmd.isSet("GAP")) {
      auto keys = cmd.getMany<size_t>("GAP");
      if(keys.size() != 0) gap = keys[0];
    }
    if(cmd.isSet("VEC_START")){
      auto keys = cmd.getMany<size_t>("VEC_START");
      if(keys.size() != 0) vec_start = keys[0];
    }

    if(FUNC == "prof_cipher_index")
      profile_cipher_index(cmd, N, M, vec_start, epsilon, gap);    

    if(FUNC == "prof_cipher_index_mpi")
      profile_cipher_index_mpi(cmd, N, M, vec_start, epsilon, gap);  
    
    if(FUNC == "prof_average")
      profile_average(cmd, N, M, vec_start, epsilon, gap);

    if(FUNC == "prof_average_mpi")
      profile_average_mpi(cmd, N, M, vec_start, epsilon, gap);
    
    if(FUNC == "prof_rank")
      profile_rank(cmd, N, M, vec_start, epsilon, gap);

    if(FUNC == "prof_rank_mpi")
      profile_rank_mpi(cmd, N, M, vec_start, epsilon, gap);
    
    if(FUNC == "prof_sort")
      profile_sort(cmd, N, M, vec_start, epsilon, gap);

    if(FUNC == "prof_sort_mpi")
      profile_sort_mpi(cmd, N, M, vec_start, epsilon, gap);
    
    if(FUNC == "prof_max")
      profile_max(cmd, N, M, vec_start, epsilon, gap);
    
    if(FUNC == "prof_max_mpi")
      profile_max_mpi(cmd, N, M, vec_start, epsilon, gap);
    
    if(FUNC == "prof_new_search_mpi")
      profile_new_search_mpi(cmd, N, M, vec_start, epsilon, gap);
    
    if(FUNC == "prof_bio_metric"){
      int k = -1;
      if(cmd.isSet("K")){
        // cout << "in check" << endl;
        auto keys = cmd.getMany<int>("K");
        k = keys[0];
        // cout << "k = " << k << endl;
      }
      if(k < 0){
        throw std::runtime_error("For high-dimensional test case: " + FUNC +
                              " K must be setted, while K = " +
                              std::to_string(k));
      }
      profile_bio_metric(cmd, N, M, k, vec_start, epsilon, gap);
    }

    if(FUNC == "prof_bio_metric_mpi"){
      int k = -1;
      if(cmd.isSet("K")){
        // cout << "in check" << endl;
        auto keys = cmd.getMany<int>("K");
        k = keys[0];
        // cout << "k = " << k << endl;
      }
      if(k < 0){
        throw std::runtime_error("For high-dimensional test case: " + FUNC +
                              " K must be setted, while K = " +
                              std::to_string(k));
      }
      profile_bio_metric_mpi(cmd, N, M, k, vec_start, epsilon, gap);
    }

    if(FUNC == "prof_metric_mpi"){
      int k = -1;
      if(cmd.isSet("K")){
        // cout << "in check" << endl;
        auto keys = cmd.getMany<int>("K");
        k = keys[0];
        // cout << "k = " << k << endl;
      }
      if(k < 0){
        throw std::runtime_error("For high-dimensional test case: " + FUNC +
                              " K must be setted, while K = " +
                              std::to_string(k));
      }
      profile_metric_mpi(cmd, N, M, k, vec_start, epsilon, gap);
    }

    if(FUNC == "prof_mean_distance"){
      int k = -1;
      if(cmd.isSet("K")){
        // cout << "in check" << endl;
        auto keys = cmd.getMany<int>("K");
        k = keys[0];
        // cout << "k = " << k << endl;
      }
      if(k < 0){
        throw std::runtime_error("For high-dimensional test case: " + FUNC +
                              " K must be setted, while K = " +
                              std::to_string(k));
      }
      profile_mean_distance(cmd, N, M, k, vec_start, epsilon, gap);
    }

    if(FUNC == "prof_mean_distance_mpi"){
      int k = -1;
      if(cmd.isSet("K")){
        // cout << "in check" << endl;
        auto keys = cmd.getMany<int>("K");
        k = keys[0];
        // cout << "k = " << k << endl;
      }
      if(k < 0){
        throw std::runtime_error("For high-dimensional test case: " + FUNC +
                              " K must be setted, while K = " +
                              std::to_string(k));
      }
      profile_mean_distance_mpi(cmd, N, M, k, vec_start, epsilon, gap);
    }

    if(FUNC == "prof_task_setup"){
      // cout << "in func" << endl;
      profile_task_setup(cmd);
    }

  }


  MPI_Finalize();

#ifdef MPIDEBUG
  int role = -1;
  if (cmd.isSet("role")) {
    auto keys = cmd.getMany<int>("role");
    role = keys[0];
  }
  if (role == -1) {
    throw std::runtime_error(LOCATION);
  }

  // Get current process rank and size
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  IOService ios;
  Sh3Encryptor enc;
  Sh3Evaluator eval;
  Sh3Runtime runtime;

  multi_processor_setup((u64)role, rank, ios, enc, eval, runtime);
#endif

  return 0;
}
#endif

#ifndef MPI
int main(int argc, char** argv) {
  oc::CLP cmd(argc, argv);

  int prog = -1;
  if (cmd.isSet("prog")) {
    auto progs = cmd.getMany<int>("prog");
    prog = progs[0];
  }
  if (prog == -1) {
    std::cout << "Set prog to 0 (basic performance test) by default "
              << std::endl;
  }

  if (prog == 0) {  // test the vectorization for basic ops (mul) and (gt).
    int repeats = int(100);

    std::vector<int> n_list = {
        10,      13,      17,      23,      30,      40,      54,      71,
        95,      126,     167,     222,     294,     390,     517,     686,
        910,     1206,    1599,    2120,    2811,    3727,    4941,    6551,
        8685,    11513,   15264,   20235,   26826,   35564,   47148,   62505,
        82864,   109854,  145634,  193069,  255954,  339322,  449843,  596362,
        790604,  1048113, 1389495, 1842069, 2442053, 3237457, 4291934, 5689866,
        7543120, 10000000};

    std::map<int, std::map<std::string, std::vector<double>>> performance_dict;
    for (int i = 0; i < n_list.size(); i++) {
      int n = n_list[i];
      std::map<std::string, std::vector<double>> tmp_map;
      basic_performance(cmd, n, repeats, tmp_map);
      // dis_basic_performance(cmd, n, repeats, tmp_map);
      performance_dict[n] = tmp_map;

      // execute one evaluation and record one.
      std::cout << "\nvector size = " << n_list[i] << std::endl;
      std::cout << "mul" << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << performance_dict[n_list[i]]["mul"][j] << " ";
      }
      std::cout << "\ngt" << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << performance_dict[n_list[i]]["gt"][j] << " ";
      }
      // std::cout << std::endl;
      std::cout << "\nadd" << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << performance_dict[n_list[i]]["add"][j] << " ";
      }
      std::cout << std::endl;
    }

    // cout the result.
    for (int i = 0; i < n_list.size(); i++) {
      std::cout << "\nvector size = " << n_list[i] << std::endl;
      std::cout << "mul" << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << performance_dict[n_list[i]]["mul"][j] << " ";
      }
      std::cout << "\ngt" << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << performance_dict[n_list[i]]["gt"][j] << " ";
      }
      // std::cout << std::endl;
      std::cout << "\nadd" << std::endl;
      for (int j = 0; j < 3; j++) {
        std::cout << performance_dict[n_list[i]]["add"][j] << " ";
      }
      std::cout << std::endl;
    }

    return 0;
  }

  if (prog == 1) {  // test the performance in the distribued setting.
    int repeats;
    std::vector<int> n_list = {
        10,      13,      17,      23,      30,      40,      54,      71,
        95,      126,     167,     222,     294,     390,     517,     686,
        910,     1206,    1599,    2120,    2811,    3727,    4941,    6551,
        8685,    11513,   15264,   20235,   26826,   35564,   47148,   62505,
        82864,   109854,  145634,  193069,  255954,  339322,  449843,  596362,
        790604,  1048113, 1389495, 1842069, 2442053, 3237457, 4291934, 5689866,
        7543120, 10000000};

    std::map<int, std::map<std::string, double>> performance_dict;
    for (int i = 0; i < n_list.size(); i++) {
      int n = n_list[i];

      // set the repeat times.
      if (i < 20)
        repeats = int(1e4);
      else if (i < 40)
        repeats = int(1e3);
      else
        repeats = int(100);

      std::map<std::string, double> tmp_map;
      dis_basic_performance(cmd, n, repeats, tmp_map);
      performance_dict[n] = tmp_map;

      // execute one evaluation and record one.
      std::cout << "\nvector size = " << n_list[i] << std::endl;
      std::map<std::string, double>::iterator iter;
      iter = tmp_map.begin();
      while (iter != tmp_map.end()) {
        std::cout << iter->first << std::endl;
        std::cout << iter->second << std::endl;
        iter++;
      }
    }
    return 0;
  }

  if (prog == 2) {  // basic test.
    // test gt
    test_gt(cmd);

    // test eq - has problems.
    test_eq(cmd);

    // test multiplication between bits and ints.
    test_mul(cmd);

    // test cipher_argsort
    test_argsort(cmd, 1);
    test_argsort(cmd, 0);

    // test cipher_index
    test_cipher_index(cmd, 0);
    test_cipher_index(cmd, 1);

    // test binning.
    test_cipher_binning(cmd, 0);
    test_cipher_binning(cmd, 1);

    return 0;
  }

  if (prog == 3) {  // basic test in the distributed setting.
    // test_mul(cmd);
    // dis_test_mul(cmd);
    int repeats = int(10);

    std::vector<int> n_list = {
        5,      10,     20,      40,      80,      160,      320,     640,
        1280,   2560,   5120,    10240,   20480,   40960,    81920,   163840,
        327680, 655360, 1310720, 2621440, 5242880, 10485760, 20971520};
    std::vector<int> m_list;

    if (cmd.isSet("mRatio")) {
      auto m_ratios = cmd.getMany<double>("mRatio");
      double m_ratio = m_ratios[0];
      for (int i = 0; i < n_list.size(); i++) {
        // m_list.push_back(int(m_ratio * n_list[i]));
        m_list.push_back(n_list[i] * m_ratio);
      }
    } else {
      int m_value = 10;
      if (cmd.isSet("mValue")) {
        auto m_values = cmd.getMany<double>("mValue");
        m_value = m_values[0];
      }
      for (int i = 0; i < n_list.size(); i++) {
        // m_list.push_back(int(m_ratio * n_list[i]));
        m_list.push_back(m_value);
      }
    }

    int testFlag = -1;
    if (cmd.isSet("testFlag")) {
      auto testFlags = cmd.getMany<int>("testFlag");
      testFlag = testFlags[0];
      if (testFlag > 2) {
        std::cout << "testFlag should be within 0-2" << std::endl;
        testFlag = -1;
      }
    }
    // test and output the time.
    for (int i = 0; i < n_list.size(); i++) {
      std::map<std::string, double> dict;
      dis_cipher_index_performance(cmd, n_list[i], m_list[i], repeats, dict,
                                   testFlag);
      std::cout << "n = " << n_list[i] << " m = " << m_list[i]
                << " normal_time = " << dict["normal"]
                << " plain_time = " << dict["plain"]
                << " rtr_time = " << dict["rtr"] << std::endl;
    }
    return 0;
  }

  if (prog == 4) {  // test the functions in the new API.
    test_argsort(cmd, 2);
    test_cipher_index(cmd, 2);
    test_cipher_binning(cmd, 2);
    test_sort(cmd, 2);
    test_max(cmd, 2);
  }

  if (prog == 5) {
    test_cipher_index_ptr(cmd, 100, 50);
  }

  std::cout << "prog only support 0 - 4" << std::endl;
  return 0;
}

#endif

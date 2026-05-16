#pragma once
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>
#include "PTRFunction.h"

int test_cipher_index_ptr(oc::CLP& cmd, size_t n, size_t m);

int test_cipher_index_ptr_mpi(oc::CLP& cmd, size_t n, size_t m, int task_num,
                              int opt_B);

int test_cipher_search_ptr_mpi(oc::CLP& cmd, size_t n, size_t m, int task_num,
                               int opt_B);

int test_cipher_search_new_ptr_mpi(oc::CLP& cmd, size_t n, size_t m, int task_num, int opt_B);

int test_cipher_rank_ptr_mpi(oc::CLP& cmd, size_t n, int task_num, int opt_B);

int test_cipher_sort_ptr_mpi(oc::CLP& cmd, size_t n, int task_num, int opt_B);

int test_cipher_max_ptr_mpi(oc::CLP& cmd, size_t n, int task_num, int opt_B);

int test_cipher_min_ptr_mpi(oc::CLP& cmd, size_t n, int task_num, int opt_B);

int test_cipher_medium_ptr_mpi(oc::CLP& cmd, size_t n, int task_num, int opt_B);

int test_cipher_select_ptr_mpi(oc::CLP& cmd, size_t n, size_t m, int task_num,
                               int opt_B);

int test_cipher_average_ptr_mpi(oc::CLP& cmd, size_t n, size_t m, int task_num,
                                int opt_B);

int test_cipher_mean_distance(oc::CLP& cmd, size_t n, size_t m, size_t k, int task_num,
                              int opt_B);

int test_cipher_bio_metric(oc::CLP& cmd, size_t n, size_t m, size_t k, int task_num,
                              int opt_B);

int test_cipher_metric(oc::CLP& cmd, size_t n, size_t m, size_t k, int task_num,
                              int opt_B);

int profile_index(oc::CLP& cmd, size_t n, size_t m, int vector_size, int task_num);

int probe_profile_index(oc::CLP& cmd, size_t n, size_t m, int vector_size_start,
                        double epsilon = 5, size_t gap = 100);

int test_vectorization(oc::CLP& cmd, size_t n, int task_num);
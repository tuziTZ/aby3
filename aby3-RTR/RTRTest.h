#pragma once
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>
#include <map>

// funtion test with real test cases.
int test_mul(oc::CLP& cmd);

int test_gt(oc::CLP& cmd);

int test_eq(oc::CLP& cmd);

int test_argsort(oc::CLP& cmd, int rtrFlag);

int test_cipher_index(oc::CLP& cmd, int rtrFlag);

int test_cipher_binning(oc::CLP& cmd, int rtrFlag);

int test_sort(oc::CLP& cmd, int rtrFlag);

int test_max(oc::CLP& cmd, int rtrFlag);

// performance test.
int basic_performance(oc::CLP& cmd, int n, int repeats,
                      std::map<std::string, std::vector<double>>& dict);
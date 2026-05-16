#pragma once
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>
#include <map>

int dis_test_mul(oc::CLP& cmd);

// performance test.
int dis_basic_performance(oc::CLP& cmd, int n, int repeats, std::map<std::string, double>& dict);

int dis_cipher_index_performance(oc::CLP& cmd, int n, int m, int repeats, std::map<std::string, double>& dict, int testFlag);
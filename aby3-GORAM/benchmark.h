#pragma once
#include <chrono>
#include <random>
#include <thread>
#include <tuple>
#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>
#include <cryptoTools/Common/CLP.h>
#include <cryptoTools/Network/IOService.h>

#include "../aby3-RTR/debug.h"
#include "../aby3-GORAM-Core/Basics.h"
#include "../aby3-GORAM-Core/timer.h"
#include "../aby3-RTR/BuildingBlocks.h"
#include "Graph.h"

void communication_synchronize(aby3Info &party_info);

int privGraph_performance_profiling(oc::CLP& cmd);
int adj_performance_profiling(oc::CLP& cmd);
int list_performance_profiling(oc::CLP& cmd);

int privGraph_integration_profiling(oc::CLP& cmd);
int adj_integration_profiling(oc::CLP& cmd);
int list_integration_profiling(oc::CLP& cmd);

int cycle_detection_profiling(oc::CLP& cmd);
int twohop_neighbor_profiling(oc::CLP& cmd);
int neighbor_statistics_profiling(oc::CLP& cmd);

// advanced applications for edgelist.
int cycle_detection_profiling_edgelist(oc::CLP& cmd);
int twohop_neighbor_profiling_edgelist(oc::CLP& cmd);
int neighbor_statistics_profiling_edgelist(oc::CLP& cmd);

// microbenchmarks.
int permutation_network_profiling(oc::CLP& cmd);
int shuffMem_profiling(oc::CLP& cmd);

// preprocessing.
int data_preparation(oc::CLP& cmd);
int partition_initialization_profiling(oc::CLP& cmd);
int partition_transmission_profiling(oc::CLP& cmd);
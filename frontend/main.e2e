#include <cryptoTools/Common/CLP.h>
#include <tests_cryptoTools/UnitTests.h>
#include <map>
#include <mpi.h>
#include "aby3-GORAM/benchmark.h"

using namespace oc;
using namespace aby3;

int main(int argc, char** argv) {
  oc::CLP cmd(argc, argv);
  debug_info("in main");
  if(cmd.isSet("prepare")){
    debug_info("in prepare");
	  data_preparation(cmd);
  }
  if(cmd.isSet("init")){
	  partition_initialization_profiling(cmd);
  }
  if(cmd.isSet("transfer")){
	  partition_transmission_profiling(cmd);
  }
  if(cmd.isSet("getShare")){
    random_share_generation(cmd);
  }
  return 0;
}
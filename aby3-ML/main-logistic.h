#pragma once
#include <cryptoTools/Common/CLP.h>
namespace aby3
{
	int logistic_main_3pc_sh(oc::CLP& cmd);
	int logistic_plain_main(oc::CLP& cmd);
	int logistic_main_3pc_sh(int N, int dim, int B, int IT, int testN, int pIdx, bool print, oc::CLP& cmd, oc::Session& chlPrev, oc::Session& chlNext);
}

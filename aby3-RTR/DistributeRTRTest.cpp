#include "DistributeRTRTest.h"

#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <map>

#include "BuildingBlocks.h"
#include "CipherIndex.h"

using namespace oc;
using namespace aby3;
using namespace std;

int dis_test_mul(CLP& cmd){

    // set the role for this process.
    int role = -1;
    if(cmd.isSet("role")){
        auto keys = cmd.getMany<int>("role");
        role = keys[0];
    }
    if(role == -1){
        throw std::runtime_error(LOCATION);
    }

    // setup the corresponding communications.
    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    distribute_setup((u64)role, ios, enc, eval, runtime);

    // run test functions
    // 1. set the test data.
    u64 rows = 10, cols = 1;
    i64Matrix plainInt(rows, cols);
    i64Matrix plainBits(rows, cols);

    for(int i=0; i<rows; i++){
        plainInt(i, 0) = i;
        plainBits(i, 0) = (i%2 == 0) * -1;
    }
    si64Matrix ea(plainInt.rows(), plainInt.cols()), eb(plainBits.rows(), plainBits.cols()), res(plainBits.rows(), plainBits.cols());

    i64Matrix zeros(plainInt.rows(), plainInt.cols());
    for(int j=0; j<plainInt.rows(); j++) zeros(j, 0) = 0;
    si64Matrix izeros(zeros.rows(), zeros.cols());

    if (role == 0) {
        enc.localIntMatrix(runtime, plainInt, ea).get();
        enc.localIntMatrix(runtime, plainBits, eb).get();
    } else {
        enc.remoteIntMatrix(runtime, ea).get();
        enc.remoteIntMatrix(runtime, eb).get();
    }
    sbMatrix bitsM;

    // 2. run function.
    Sh3Task task = runtime.noDependencies();
    fetch_msb(role, eb, bitsM, eval, runtime, task);

    // eval.asyncMul(runtime, ea, bitsM, res).get();
    cipher_mul_seq(role, ea, bitsM, res, eval, enc, runtime);
    // cout << "after mul seq" << endl;
    i64Matrix pres;
    enc.revealAll(runtime, res, pres).get();
    if(role == 0){
        cout << "Expected res = [0, 0, 2, 0, 4, 0, 6, ...]" << endl;
        cout << "Real compute = ";
        for(int j=0; j<ea.rows(); j++) cout << pres(j, 0) << " ";
        cout << "\n" << endl;
    }

    return 0;
}


int dis_basic_performance(CLP& cmd, int n, int repeats, map<std::string, double>& dict){
    // set the role for this process.
    int role = -1;
    if(cmd.isSet("role")){
        auto keys = cmd.getMany<int>("role");
        role = keys[0];
    }
    if(role == -1){
        throw std::runtime_error(LOCATION);
    }

    // setup the corresponding communications.
    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    distribute_setup((u64)role, ios, enc, eval, runtime);

    // set the test data
    u64 rows = n, cols = 1;
    f64Matrix<D8> plainA(rows, cols);
    f64Matrix<D8> plainB(rows, cols);
    i64Matrix iplainA(rows, cols);
    i64Matrix iplainB(rows, cols);

    for(u64 j=0; j<rows; j++){
        for (u64 i = 0; i<cols; i++){
            plainA(j, i) = (double) j;
            plainB(j, i) = (double) j + 1;
            iplainA(j, i) = j;
            iplainB(j, i) = j + 1;
        }
    }

    clock_t start, end;

    sf64Matrix<D8> sharedA(plainA.rows(), plainA.cols());
    sf64Matrix<D8> sharedB(plainB.rows(), plainB.cols());
    si64Matrix isharedA(iplainA.rows(), iplainA.cols());
    si64Matrix isharedB(iplainB.rows(), iplainB.cols());

    if(role == 0){
        enc.localFixedMatrix(runtime, plainA, sharedA).get();
        enc.localFixedMatrix(runtime, plainB, sharedB).get();
        enc.localIntMatrix(runtime, iplainA, isharedA).get();
        enc.localIntMatrix(runtime, iplainB, isharedB).get();
    }
    else{
        enc.remoteFixedMatrix(runtime, sharedA).get();
        enc.remoteFixedMatrix(runtime, sharedB).get();
        enc.remoteIntMatrix(runtime, isharedA).get();
        enc.remoteIntMatrix(runtime, isharedB).get();
    }

    // test the performance of basic ops.
    // 1. sfixed multiplication.
    sf64Matrix<D8> mul_res(plainA.rows(), plainA.cols());
    start = clock();
    for(int k=0; k<repeats; k++)
        eval.asyncMul(runtime, sharedA, sharedB, mul_res).get();
    end = clock();
    dict["fxp-mul"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // 2. sfixed addition
    sf64Matrix<D8> add_res(plainA.rows(), plainA.cols());
    start = clock();
    for(int k=0; k<repeats; k++)
        add_res = sharedA + sharedB;
    end = clock();
    dict["fxp-add"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // 3. sfixed greater-then
    sbMatrix lt_res;
    start = clock();
    for(int k=0; k<repeats; k++)
        cipher_gt(role, sharedB, sharedA, lt_res, eval, runtime);
    end = clock();
    dict["fxp-gt"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    sbMatrix eq_res;
    start = clock();
    for(int k=0; k<repeats; k++)
        cipher_eq(role, sharedB, sharedA, eq_res, eval, runtime);
    end = clock();
    dict["fxp-eq"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // 4. sint multiplication.
    si64Matrix imul_res(iplainA.rows(), iplainB.cols());
    start = clock();
    for(int k=0; k<repeats; k++)
        eval.asyncMul(runtime, isharedA, isharedB, imul_res).get();
    end = clock();
    dict["int-mul"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // 5. sint addition
    start = clock();
    for(int k=0; k<repeats; k++)
        imul_res = isharedA + isharedB;
    end = clock();
    dict["int-add"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // sint gt
    start = clock();
    for(int k=0; k<repeats; k++)
        cipher_gt(role, isharedB, isharedA, lt_res, eval, runtime);
    end = clock();
    dict["int-gt"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // sint eq
    start = clock();
    for(int k=0; k<repeats; k++)
        cipher_eq(role, isharedB, isharedA, eq_res, eval, runtime);
    end = clock();
    dict["int-eq"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    // multiplications
    // 5. sb & si multiplication.
    si64Matrix ibmul_res(iplainA.rows(), iplainA.cols());
    start = clock();
    for(int k=0; k<repeats; k++)
        eval.asyncMul(runtime, isharedA, lt_res, ibmul_res).get();
    end = clock();
    dict["mul-ib"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    sf64Matrix<D8> fbmul_res(plainA.rows(), plainA.cols());
    start = clock();
    for(int k=0; k<repeats; k++){
        cipher_mul_seq(role, sharedA, lt_res, fbmul_res, eval, enc, runtime);
    }
    end = clock();
    dict["mul-fb"] = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);


    return 0;
}


// n: data volumn; m: index volumn.
int dis_cipher_index_performance(CLP& cmd, int n, int m, int repeats, map<std::string, double>& dict, int testFlag){
    int role = -1;
    if(cmd.isSet("role")){
        auto keys = cmd.getMany<int>("role");
        role = keys[0];
    }
    if(role == -1){
        throw std::runtime_error(LOCATION);
    }
    
    // setup the corresponding communications.
    IOService ios;
    Sh3Encryptor enc;
    Sh3Evaluator eval;
    Sh3Runtime runtime;
    distribute_setup((u64)role, ios, enc, eval, runtime);

    // generate the test data.
    u64 rows = n, cols = 1;
    u64 ilen = m;
    i64Matrix plainTest(rows, cols);
    i64Matrix plainIndex(ilen, cols);
    for(int i=0; i<rows; i++){
        plainTest(i, 0) = i;
    }
    for(int i=0; i<ilen; i++){
        // plainIndex(i, 0) = max(rows - 1 - i, 0);
        plainIndex(i, 0) = (rows - 1 - i > 0) ? rows - 1 - i : 0;
    }

    clock_t start, end;
    // generate the cipher test data.
    si64Matrix sharedM(rows, cols);
    si64Matrix sharedIndex(plainIndex.rows(), plainIndex.cols());
    if(role == 0){
        enc.localIntMatrix(runtime, plainTest, sharedM).get();
        enc.localIntMatrix(runtime, plainIndex, sharedIndex).get();
    }
    else{
        enc.remoteIntMatrix(runtime, sharedM).get();
        enc.remoteIntMatrix(runtime, sharedIndex).get();
    }
    
    si64Matrix res;
    // test cipher_index using different strategies.
    start = clock();
    for(int k=0; k<repeats; k++){
        if(testFlag == 0 || testFlag == -1) normal_cipher_index(role, sharedM, sharedIndex, res, eval, runtime, enc);
        // normal_cipher_index(role, sharedM, sharedIndex, res, eval, runtime, enc);
    }
    end = clock();
    double time_normal = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    start = clock();
    for(int k=0; k<repeats; k++){
        if (testFlag == 1 || testFlag == -1) cipher_index(role, sharedM, sharedIndex, res, eval, runtime, enc);
    }
    end = clock();
    double time_plain = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);

    start = clock();
    for(int k=0; k<repeats; k++){
        if(testFlag == 2 || testFlag == -1) rtr_cipher_index(role, sharedM, sharedIndex, res, eval, runtime, enc);
    }
    end = clock();
    double time_rtr = double((end - start)*1000)/(CLOCKS_PER_SEC * repeats);


    dict["normal"] = time_normal;
    dict["plain"] = time_plain;
    dict["rtr"] = time_rtr;

  return 0;
}
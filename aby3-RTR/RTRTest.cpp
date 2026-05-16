#include "RTRTest.h"

#include <fstream>
#include <iostream>

#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Runtime.h>

#include "BuildingBlocks.h"
#include "CipherIndex.h"
#include "RepeatThenReduce.h"

using namespace oc;
using namespace aby3;
using namespace std;

// sint and sbit matrix multiplication test.
int test_mul(CLP &cmd) {
  IOService ios;
  u64 rows = 10, cols = 1;
  i64Matrix plainInt(rows, cols);
  i64Matrix plainBits(rows, cols);

  for (int i = 0; i < rows; i++) {
    plainInt(i, 0) = i;
    plainBits(i, 0) = (i % 2 == 0) * -1;
  }

  vector<thread> thrds;
  for (int i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, plainInt, plainBits, &ios]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup((u64)i, ios, enc, eval,
                  runtime);  // same setup function in the tutorial.
      si64Matrix ea(plainInt.rows(), plainInt.cols()),
          eb(plainBits.rows(), plainBits.cols()),
          res(plainBits.rows(), plainBits.cols());

      i64Matrix zeros(plainInt.rows(), plainInt.cols());
      for (int j = 0; j < plainInt.rows(); j++) zeros(j, 0) = 0;
      si64Matrix izeros(zeros.rows(), zeros.cols());

      if (i == 0) {
        enc.localIntMatrix(runtime, plainInt, ea).get();
        enc.localIntMatrix(runtime, plainBits, eb).get();
      } else {
        enc.remoteIntMatrix(runtime, ea).get();
        enc.remoteIntMatrix(runtime, eb).get();
      }
      sbMatrix bitsM;
      Sh3Task task = runtime.noDependencies();
      fetch_msb(i, eb, bitsM, eval, runtime, task);

      // eval.asyncMul(runtime, ea, bitsM, res).get();
      cipher_mul_seq(i, ea, bitsM, res, eval, enc, runtime);
      // cout << "after mul seq" << endl;
      i64Matrix pres;
      enc.revealAll(runtime, res, pres).get();
      if (i == 0) {
        cout << "Expected res = [0, 0, 2, 0, 4, 0, 6, ...]" << endl;
        cout << "Real compute = ";
        for (int j = 0; j < ea.rows(); j++) cout << pres(j, 0) << " ";
        cout << "\n" << endl;
      }
      // cout << "\n" << endl;
    }));
  }
  for (auto &t : thrds) t.join();
  return 0;
}

// sfixed greater-than test.
int test_gt(CLP &cmd) {
  IOService ios;
  u64 rows = 10, cols = 1;
  f64Matrix<D8> plainA(rows, cols);
  f64Matrix<D8> plainB(rows, cols);

  for (u64 j = 0; j < rows; j++) {
    for (u64 i = 0; i < cols; i++) {
      plainA(j, i) = (double)j;
      plainB(j, i) = (double)j + 1;
    }
  }

  vector<thread> thrds;
  for (int i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, plainA, plainB, &ios]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup((u64)i, ios, enc, eval,
                  runtime);  // same setup function in the tutorial.

      sf64Matrix<D8> sharedA(plainA.rows(), plainA.cols());
      sf64Matrix<D8> sharedB(plainB.rows(), plainB.cols());

      if (i == 0) {
        enc.localFixedMatrix(runtime, plainA, sharedA).get();
        enc.localFixedMatrix(runtime, plainB, sharedB).get();
      } else {
        enc.remoteFixedMatrix(runtime, sharedA).get();
        enc.remoteFixedMatrix(runtime, sharedB).get();
      }

      sbMatrix lt_res;
      cipher_gt(i, sharedB, sharedA, lt_res, eval, runtime);

      // si64Matrix larger_res;
      sf64Matrix<D8> res;
      res.resize(sharedA.rows(), sharedA.cols());
      eval.asyncMul(runtime, sharedA.i64Cast(), lt_res, res.i64Cast()).get();

      f64Matrix<D8> pres;
      enc.revealAll(runtime, res, pres).get();

      if (i == 0) {
        cout << "Expected res = [0.0, 1.0, 2.0, ..., 9.0]" << endl;
        cout << "Real compute = ";
        for (int j = 0; j < sharedA.rows(); j++) cout << pres(j, 0) << ", ";
        cout << "\n" << endl;
      }
    }));
  }

  for (auto &t : thrds) t.join();
  return 0;
}

// sint eq test.
int test_eq(CLP &cmd) {
  IOService ios;
  u64 rows = 10, cols = 1;

  // test for i64 datatype.
  i64Matrix plainA(rows, cols);
  i64Matrix plainB(rows, cols);
  i64Matrix ones(rows, cols);

  for (int j = 0; j < rows; j++) {
    plainA(j, 0) = j;
    if (j % 2 == 0)
      plainB(j, 0) = j;
    else
      plainB(j, 0) = j - 1;
  }

  for (u64 j = 0; j < rows; j++) {
    ones(j, 0) = 1;
  }

  vector<thread> thrds;
  for (int i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, plainA, plainB, ones, &ios]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup((u64)i, ios, enc, eval,
                  runtime);  // same setup function in the tutorial.

      si64Matrix sharedA(plainA.rows(), plainA.cols());
      si64Matrix sharedB(plainA.rows(), plainA.cols());
      si64Matrix sharedOnes(plainA.rows(), plainA.cols());

      if (i == 0) {
        enc.localIntMatrix(runtime, plainA, sharedA).get();
        enc.localIntMatrix(runtime, plainB, sharedB).get();
        enc.localIntMatrix(runtime, ones, sharedOnes).get();
      } else {
        enc.remoteIntMatrix(runtime, sharedA).get();
        enc.remoteIntMatrix(runtime, sharedB).get();
        enc.remoteIntMatrix(runtime, sharedOnes).get();
      }

      sbMatrix eq_res;
      // fanxy: test eq function.
      // cipher_eq(i, sharedA, sharedB, eq_res, eval, runtime);
      circuit_cipher_eq(i, sharedA, sharedB, eq_res, eval, runtime);

      si64Matrix res;
      res.resize(sharedA.rows(), sharedA.cols());
      eval.asyncMul(runtime, sharedOnes, eq_res, res).get();
      i64Matrix pres;
      // cout << "here" << endl;
      enc.revealAll(runtime, res, pres).get();

      if (i == 0) {
        cout << "Expected res = [1, 0, 1, ...]" << endl;
        cout << "Real compute = ";
        for (int j = 0; j < sharedA.rows(); j++) cout << pres(j, 0) << " ";
        cout << "\n" << endl;
      }
    }));
  }

  for (auto &t : thrds) t.join();
  return 0;
}

int test_argsort(CLP &cmd, int rtrFlag) {
  IOService ios;

  // generate the test data.
  u64 rows = 100, cols = 1;
  f64Matrix<D8> plainTest(rows, cols);
  for (int i = 0; i < rows; i++) {
    plainTest(i, 0) = i;
  }

  // start the three parties computation.
  vector<thread> thrds;
  for (u64 i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, &ios, plainTest, rows, cols, rtrFlag]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup(i, ios, enc, eval, runtime);

      // generate the cipher test data.
      sf64Matrix<D8> sharedM(rows, cols);
      if (i == 0) {
        enc.localFixedMatrix(runtime, plainTest, sharedM).get();
      } else {
        enc.remoteFixedMatrix(runtime, sharedM).get();
      }

      si64Matrix argres;
      // test argsort using different strategies.
      if (rtrFlag == 1)
        rtr_cipher_argsort(i, sharedM, argres, eval, runtime, enc);
      else if (rtrFlag == 0)
        cipher_argsort(i, sharedM, argres, eval, runtime, enc);
      else if (rtrFlag == 2)
        argsort(i, sharedM, argres, eval, runtime, enc);
      else
        rtr_cipher_argsort(i, sharedM, argres, eval, runtime, enc);

      i64Matrix plain_argres;
      enc.revealAll(runtime, argres, plain_argres).get();

      if (i == 0) {
        cout << "Expected res = [0 1 2 3 4 ...]" << endl;
        // cout << plain_argres << endl;
        cout << "Real compute = ";
        for (int j = 0; j < 7; j++) {
          cout << plain_argres(j, 0) << " ";
        }
        cout << "\n" << endl;
      }
    }));
  }
  for (auto &t : thrds) t.join();
  return 0;
}

int test_cipher_index(CLP &cmd, int rtrFlag) {
  IOService ios;

  // generate the test data.
  u64 rows = 50, cols = 1;
  u64 m = 20;
  i64Matrix plainTest(rows, cols);
  i64Matrix plainIndex(m, cols);
  for (int i = 0; i < rows; i++) {
    plainTest(i, 0) = i;
  }
  // inverse sequence.
  for (int i = 0; i < m; i++) {
    plainIndex(i, 0) = rows - 1 - i;
  }

  // start the three parties computation.
  vector<thread> thrds;
  for (u64 i = 0; i < 3; i++) {
    thrds.emplace_back(
        thread([i, &ios, plainTest, plainIndex, rows, cols, rtrFlag]() {
          Sh3Encryptor enc;
          Sh3Evaluator eval;
          Sh3Runtime runtime;
          basic_setup(i, ios, enc, eval, runtime);

          // generate the cipher test data.
          si64Matrix sharedM(rows, cols);
          si64Matrix sharedIndex(plainIndex.rows(), plainIndex.cols());
          if (i == 0) {
            enc.localIntMatrix(runtime, plainTest, sharedM).get();
            enc.localIntMatrix(runtime, plainIndex, sharedIndex).get();
          } else {
            enc.remoteIntMatrix(runtime, sharedM).get();
            enc.remoteIntMatrix(runtime, sharedIndex).get();
          }

          si64Matrix res;
          // test argsort using different strategies.
          if (rtrFlag == 1)
            rtr_cipher_index(i, sharedM, sharedIndex, res, eval, runtime, enc);
          else if (rtrFlag == 0)
            cipher_index(i, sharedM, sharedIndex, res, eval, runtime, enc);
          else if (rtrFlag == 2)
            secret_index(i, sharedM, sharedIndex, res, eval, runtime, enc);
          else
            rtr_cipher_index(i, sharedM, sharedIndex, res, eval, runtime, enc);

          i64Matrix plain_res;
          enc.revealAll(runtime, res, plain_res).get();

          if (i == 0) {
            cout << "Expected res = [49, 48, 47...]" << endl;
            cout << "Real compute = ";
            for (int j = 0; j < res.rows(); j++) {
              cout << plain_res(j, 0) << " ";
            }
            cout << "\n" << endl;
          }
        }));
  }
  for (auto &t : thrds) t.join();
  return 0;
}

int test_cipher_binning(CLP &cmd, int rtrFlag) {
  IOService ios;

  // generate the test data
  // u64 n = 100, m = 50, cols = 1;
  u64 n = 10, m = 5, cols = 1;
  i64Matrix testM(n, cols);
  i64Matrix bins(m, cols);
  i64Matrix tarVals(m, cols);

  for (int i = 0; i < n; i++) {
    testM(i, 0) = i * int(m * 10 / n);
  }
  // inverse sequence.
  for (int i = 0; i < m; i++) {
    bins(i, 0) = 10 * i;
    tarVals(i, 0) = i;
  }

  // start the three parties computation.
  vector<thread> thrds;
  for (int i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, &ios, testM, &bins, &tarVals, cols,
                               rtrFlag]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup((u64)i, ios, enc, eval, runtime);

      u64 n = testM.rows(), m = bins.rows();

      // generate the cipher test data.
      si64Matrix sharedM(n, cols);
      if (i == 0) {
        enc.localIntMatrix(runtime, testM, sharedM).get();
      } else {
        enc.remoteIntMatrix(runtime, sharedM).get();
      }
      si64Matrix res;
      // test cipher_binning with different strategries.
      if (rtrFlag == 1) {
        rtr_cipher_binning(i, sharedM, bins, tarVals, res, eval, runtime, enc);
      } else if (rtrFlag == 0) {
        cipher_binning(i, sharedM, bins, tarVals, res, eval, runtime, enc);
      } else if (rtrFlag == 2) {
        get_binning_value(i, sharedM, bins, tarVals, res, eval, runtime, enc);
      } else {
        rtr_cipher_binning(i, sharedM, bins, tarVals, res, eval, runtime, enc);
      }

      i64Matrix plain_res;
      enc.revealAll(runtime, res, plain_res).get();

      if (i == 0) {
        cout << "Expected res = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]" << endl;
        // cout << plain_argres << endl;
        cout << "Real compute = ";
        for (int j = 0; j < res.rows(); j++) {
          cout << plain_res(j, 0) << " ";
        }
        cout << "\n" << endl;
      }
    }));
  }
  for (auto &t : thrds) t.join();
  return 0;
}

int test_sort(CLP &cmd, int rtrFlag) {
  IOService ios;

  // generate the test data.
  u64 rows = 100, cols = 1;
  f64Matrix<D8> plainTest(rows, cols);
  for (u64 i = 0; i < rows; i++) {
    plainTest(i, 0) = rows - i;
  }

  // start the three parties computation.
  vector<thread> thrds;
  for (u64 i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, &ios, plainTest, rows, cols, rtrFlag]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup(i, ios, enc, eval, runtime);

      // generate the cipher test data.
      sf64Matrix<D8> sharedM(rows, cols);
      if (i == 0) {
        enc.localFixedMatrix(runtime, plainTest, sharedM).get();
      } else {
        enc.remoteFixedMatrix(runtime, sharedM).get();
      }

      sf64Matrix<D8> res;
      // test argsort using different strategies.
      sort(i, sharedM.i64Cast(), res.i64Cast(), eval, runtime, enc);

      // if (rtrFlag == 1)
      //   rtr_cipher_argsort(i, sharedM, argres, eval, runtime, enc);
      // else if (rtrFlag == 0)
      //   cipher_argsort(i, sharedM, argres, eval, runtime, enc);
      // else if (rtrFlag == 2)
      //   argsort(i, sharedM, argres, eval, runtime, enc);
      // else
      //   rtr_cipher_argsort(i, sharedM, argres, eval, runtime, enc);

      f64Matrix<D8> plain_res;
      enc.revealAll(runtime, res, plain_res).get();

      if (i == 0) {
        cout << "Expected res = [0 1 2 3 4 ...]" << endl;
        // cout << plain_argres << endl;
        cout << "Real compute = ";
        for (int j = 0; j < 7; j++) {
          cout << plain_res(j, 0) << " ";
        }
        cout << "\n" << endl;
      }
    }));
  }
  for (auto &t : thrds) t.join();
  return 0;
}


int test_max(CLP &cmd, int rtrFlag){
  IOService ios;

  // generate the test data
  u64 n = 10, cols = 1;
  i64Matrix testM(n, cols);

  // inverse sequence
  for (int i = 0; i < n; i++) {
    testM(i, 0) = n - i;
  }

  // start the three parties computation.
  vector<thread> thrds;
  for (int i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, &ios, testM, cols,
                               rtrFlag]() {
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup((u64)i, ios, enc, eval, runtime);

      u64 n = testM.rows();

      // generate the cipher test data.
      si64Matrix sharedM(n, cols);
      if (i == 0) {
        enc.localIntMatrix(runtime, testM, sharedM).get();
      } else {
        enc.remoteIntMatrix(runtime, sharedM).get();
      }
      si64Matrix res;
      
      // test max
      max_rtr(i, sharedM, res, eval, runtime, enc);
      
      i64Matrix plain_res;
      enc.revealAll(runtime, res, plain_res).get();

      if (i == 0) {
        cout << "Expected res = " << n << endl;
        // cout << plain_argres << endl;
        cout << "Real compute = ";
        for (int j = 0; j < res.rows(); j++) {
          cout << plain_res(j, 0) << " ";
        }
        cout << "\n" << endl;
      }
    }));
  }
  for (auto &t : thrds) t.join();
  return 0;
}


int basic_performance(CLP &cmd, int n, int repeats,
                      map<string, vector<double>> &dict) {
  IOService ios;
  u64 rows = n, cols = 1;
  f64Matrix<D8> plainA(rows, cols);
  f64Matrix<D8> plainB(rows, cols);

  for (u64 j = 0; j < rows; j++) {
    for (u64 i = 0; i < cols; i++) {
      plainA(j, i) = (double)j;
      plainB(j, i) = (double)j + 1;
    }
  }

  vector<double> time_mul_array(3);
  vector<double> time_gt_array(3);
  vector<double> time_add_array(3);

  vector<thread> thrds;
  for (u64 i = 0; i < 3; i++) {
    thrds.emplace_back(thread([i, plainA, plainB, repeats, &time_mul_array,
                               &time_gt_array, &time_add_array, &ios]() {

      // setup the environment.
      clock_t start, end;
      Sh3Encryptor enc;
      Sh3Evaluator eval;
      Sh3Runtime runtime;
      basic_setup(i, ios, enc, eval, runtime);

      // generate the test matrix.
      sf64Matrix<D8> sharedA(plainA.rows(), plainA.cols());
      sf64Matrix<D8> sharedB(plainB.rows(), plainB.cols());

      if (i == 0) {
        enc.localFixedMatrix(runtime, plainA, sharedA).get();
        enc.localFixedMatrix(runtime, plainB, sharedB).get();
      } else {
        enc.remoteFixedMatrix(runtime, sharedA).get();
        enc.remoteFixedMatrix(runtime, sharedB).get();
      }

      // test the performance of basic ops.
      sbMatrix lt_res;
      start = clock();
      for (int k = 0; k < repeats; k++)
        cipher_gt(i, sharedB, sharedA, lt_res, eval, runtime);
      end = clock();

      double time_gt =
          double((end - start) * 1000) / (CLOCKS_PER_SEC * repeats);
      time_gt_array[i] = time_gt;

      sf64Matrix<D8> mul_res(plainA.rows(), plainA.cols());
      start = clock();
      for (int k = 0; k < repeats; k++)
        eval.asyncMul(runtime, sharedA, sharedB, mul_res).get();
      end = clock();

      double time_mul =
          double((end - start) * 1000) / (CLOCKS_PER_SEC * repeats);
      time_mul_array[i] = time_mul;

      sf64Matrix<D8> add_res(plainA.rows(), plainA.cols());
      start = clock();
      for (int k = 0; k < repeats; k++) add_res = sharedA + sharedB;
      end = clock();

      double time_add =
          double((end - start) * 1000) / (CLOCKS_PER_SEC * repeats);
      time_add_array[i] = time_add;

    }));
  }

  for (auto &t : thrds) t.join();

  // map<string, double(*)[3]> dict;
  dict["mul"] = time_mul_array;
  dict["gt"] = time_gt_array;
  dict["add"] = time_add_array;

  return 0;
}
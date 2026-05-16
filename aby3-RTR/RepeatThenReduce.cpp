#include "RepeatThenReduce.h"

#include <bitset>
#include <fstream>
#include <iostream>

#include <aby3/Circuit/CircuitLibrary.h>
#include <aby3/sh3/Sh3BinaryEvaluator.h>
using namespace aby3;
using namespace std;
using namespace oc;

static const int FLAG = 2;

int argsort(int pIdx, si64Matrix &sharedM, si64Matrix &res, Sh3Evaluator &eval,
            Sh3Runtime &runtime, Sh3Encryptor &enc) {
  int n = sharedM.size();

  if (FLAG == 0) {
    repeat_then_reduce_pure<si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, sharedM, n, sharedM, n, 1, relation_gt,
        reduce_count, res);
  } else if (FLAG == 1) {
    repeat_then_reduce<si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, sharedM, n, sharedM, n, 1, relation_gt_offset,
        reduce_count_offset, res);
  } else {
    return 1;
  }
  return 0;
}

int secret_index(int pIdx, si64Matrix &sharedM, si64Matrix &secretIndex,
                 si64Matrix &res, Sh3Evaluator &eval, Sh3Runtime &runtime,
                 Sh3Encryptor &enc) {
  int n = sharedM.size();
  int m = secretIndex.size();
  vector<int> plainIndex(n);
  for (int i = 0; i < n; i++) plainIndex[i] = i;

  if (FLAG == 0) {
    repeat_then_reduce_pure<si64Matrix, vector<int>, si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, secretIndex, m, plainIndex, n, 1, relation_eq,
        reduce_select, res, sharedM);
  } else if (FLAG == 1) {
    repeat_then_reduce<si64Matrix, vector<int>, si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, secretIndex, m, plainIndex, n, 1,
        relation_eq_offset, reduce_select_offset, res, sharedM);
  }
  return 0;
}

int get_binning_value(int pIdx, si64Matrix &sharedM, i64Matrix &bins,
                      i64Matrix &targetVals, si64Matrix &res,
                      Sh3Evaluator &eval, Sh3Runtime &runtime,
                      Sh3Encryptor &enc) {
  int n = sharedM.size();
  int m = bins.size();

  if (FLAG == 0) {
    repeat_then_reduce_pure<si64Matrix, i64Matrix, si64Matrix, i64Matrix>(
        pIdx, eval, runtime, enc, sharedM, n, bins, m, 1, relation_gt,
        reduce_lb_select, res, targetVals);
  } else if (FLAG == 1) {
    repeat_then_reduce<si64Matrix, i64Matrix, si64Matrix, i64Matrix>(
        pIdx, eval, runtime, enc, sharedM, n, bins, m, 1, relation_gt_offset,
        reduce_lb_select_offset, res, targetVals);
  }
  return 0;
}

int sort(int pIdx, si64Matrix &sharedM, si64Matrix &res, Sh3Evaluator &eval,
         Sh3Runtime &runtime, Sh3Encryptor &enc) {
  int n = sharedM.size();
  si64Matrix argIndex;
  vector<int> plainIndex(n);
  for (int i = 0; i < n; i++) plainIndex[i] = i;
  if (FLAG == 0) {
    repeat_then_reduce_pure<si64Matrix, si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, sharedM, n, sharedM, n, 1, relation_gt,
        reduce_count, argIndex);
    repeat_then_reduce_pure<si64Matrix, vector<int>, si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, argIndex, n, plainIndex, n, 1, relation_eq,
        reduce_select, res, sharedM);
  } else if (FLAG == 1) {
    repeat_then_reduce<si64Matrix, si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, sharedM, n, sharedM, n, 1, relation_gt_offset,
        reduce_count_offset, argIndex);
    repeat_then_reduce<si64Matrix, vector<int>, si64Matrix, si64Matrix>(
        pIdx, eval, runtime, enc, argIndex, n, plainIndex, n, 1,
        relation_eq_offset, reduce_select_offset, res, sharedM);
  }
  return 0;
}

int max_rtr(int pIdx, si64Matrix &sharedM, si64Matrix &res, Sh3Evaluator &eval,
            Sh3Runtime &runtime, Sh3Encryptor &enc) {
  int n = sharedM.size();
  si64Matrix argIndex;

  repeat_then_reduce<si64Matrix, si64Matrix, si64Matrix>(
      pIdx, eval, runtime, enc, sharedM, n, sharedM, n, 1, relation_gt_offset,
      reduce_count_offset, argIndex);

  vector<int> targetIndex(1);
  targetIndex[0] = (n - 1);

  repeat_then_reduce<si64Matrix, vector<int>, si64Matrix, si64Matrix>(
      pIdx, eval, runtime, enc, argIndex, n, targetIndex, 1, 0,
      relation_eq_offset, reduce_select_offset, res, sharedM);

  return 0;
}


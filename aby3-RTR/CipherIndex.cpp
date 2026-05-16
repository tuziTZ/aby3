#include "CipherIndex.h"
#include "BuildingBlocks.h"

#include <fstream>
#include <iostream>

using namespace std;
using namespace oc;
using namespace aby3;

#define SEPARATE_BY_BLOCK_SIZE true
#define COUT_LOG false

static const int BLOCK_SIZE = 50000;
static const int THREADS_SIZE = 10000000;

// normal version cipher_index.
int normal_cipher_index(int pIdx, si64Matrix &sharedM, si64Matrix &indexMatrix,
                        si64Matrix &res, Sh3Evaluator &eval,
                        Sh3Runtime &runtime, Sh3Encryptor &enc) {
  u64 m = indexMatrix.rows(), n = sharedM.rows();

  // initialize res to all zero si64Matrix.
  i64Matrix zeros(m, 1);
  for (int i = 0; i < m; i++) zeros(i, 0) = 0;
  res.resize(m, 1);

  // fetch elements one by one.
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      // construct the `is-i-element` mask
      sbMatrix maskA(1, sizeof(i64) * 8), maskB(1, sizeof(i64) * 8);
      sbMatrix mask;

      switch (pIdx) {
        case 0: {
          maskA.mShares[0](0) =
              j - indexMatrix.mShares[0](i, 0) - indexMatrix.mShares[1](i, 0);
          maskB.mShares[0].setZero();
          maskB.mShares[1].setZero();
          break;
        }
        case 1: {
          maskB.mShares[0](0) = indexMatrix.mShares[0](i, 0);
          maskB.mShares[1].setZero();
          maskA.mShares[0].setZero();
          break;
        }
        case 2: {
          maskB.mShares[1](0) = indexMatrix.mShares[1](i, 0);
          maskB.mShares[0].setZero();
          maskA.mShares[0].setZero();
          break;
        }
      }
      runtime.mComm.mNext.asyncSend(maskA.mShares[0].data(),
                                    maskA.mShares[0].size());
      auto fu = runtime.mComm.mPrev.asyncRecv(maskA.mShares[1].data(),
                                              maskA.mShares[1].size());
      fu.get();
      fetch_eq_res(pIdx, maskA, maskB, mask, eval, runtime);

      // multiply the element with mask.
      si64Matrix element(1, 1);
      element(0, 0) = sharedM(j, 0);
      cipher_mul_seq(pIdx, element, mask, element, eval, enc, runtime);

      // compute the res[i]
      res.mShares[0](i, 0) += element.mShares[0](0, 0);
      res.mShares[1](i, 0) += element.mShares[1](0, 0);
    }
  }
  return 0;
}

// pure repeat-version cipher_index.
int cipher_index(int pIdx, si64Matrix &sharedM, si64Matrix &indexMatrix,
                 si64Matrix &res, Sh3Evaluator &eval, Sh3Runtime &runtime,
                 Sh3Encryptor &enc) {
  Sh3Task task = runtime.noDependencies();
  u64 n = sharedM.rows(), m = indexMatrix.rows();
  i64Matrix zeros(m, 1);
  for (int i = 0; i < m; i++) zeros(i, 0) = 0;
  res.resize(m, 1);
  if (pIdx == 0) {
    enc.localIntMatrix(runtime, zeros, res).get();
  } else {
    enc.remoteIntMatrix(runtime, res).get();
  }
  return cipher_index_offset(pIdx, sharedM, indexMatrix, res, eval, runtime,
                             enc, task, 0, (int)m * n);
}

// isolated cipher_index offset, computing between offsetLeft and offsetRight.
int cipher_index_offset(int pIdx, si64Matrix &sharedM, si64Matrix &indexMatrix,
                        si64Matrix &res, Sh3Evaluator &eval,
                        Sh3Runtime &runtime, Sh3Encryptor &enc, Sh3Task &task,
                        int offsetLeft, int offsetRight) {
  // 0. initialize the config values.
  u64 block_length = offsetRight - offsetLeft;
#ifdef COUT_LOG
  cout << "block size: " << block_length << endl;
#endif

  u64 n = sharedM.rows(), m = indexMatrix.rows();
  u64 start_i = offsetLeft / n, start_j = offsetLeft % n;
  u64 end_i = offsetRight / n, end_j = offsetRight % n;
  if (COUT_LOG) cout << "before computation" << endl;

  // 1. construct the required one-hot encoding between offsetLeft and
  // offsetRight.
  sbMatrix expandCircuitA(block_length, sizeof(i64) * 8),
      expandCircuitB(block_length, sizeof(i64) * 8);
  u64 p = 0, i = start_i, j = start_j;
  while (i < end_i) {
    for (j; j < n; j++) {
      switch (pIdx) {
        case 0: {
          expandCircuitA.mShares[0](p) =
              j - indexMatrix.mShares[0](i, 0) - indexMatrix.mShares[1](i, 0);
          break;
        }
        case 1: {
          expandCircuitB.mShares[0](p) = indexMatrix.mShares[0](i, 0);
          break;
        }
        case 2: {
          expandCircuitB.mShares[1](p) = indexMatrix.mShares[1](i, 0);
          break;
        }
      }
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    switch (pIdx) {
      case 0: {
        expandCircuitA.mShares[0](p) =
            j - indexMatrix.mShares[0](i, 0) - indexMatrix.mShares[1](i, 0);
        break;
      }
      case 1: {
        expandCircuitB.mShares[0](p) = indexMatrix.mShares[0](i, 0);
        break;
      }
      case 2: {
        expandCircuitB.mShares[1](p) = indexMatrix.mShares[1](i, 0);
      }
    }
    p++;
  }

  switch (pIdx) {
    case 0: {
      expandCircuitB.mShares[0].setZero();
      expandCircuitB.mShares[1].setZero();
      break;
    }
    case 1: {
      expandCircuitB.mShares[1].setZero();
      expandCircuitA.mShares[0].setZero();
      break;
    }
    case 2: {
      expandCircuitB.mShares[0].setZero();
      expandCircuitA.mShares[0].setZero();
    }
  }

  if (COUT_LOG) cout << "start to communicate" << endl;
  runtime.mComm.mNext.asyncSend(expandCircuitA.mShares[0].data(),
                                expandCircuitA.mShares[0].size());
  auto fu = runtime.mComm.mPrev.asyncRecv(expandCircuitA.mShares[1].data(),
                                          expandCircuitA.mShares[1].size());
  fu.get();
  if (COUT_LOG) cout << "success communication" << endl;

  sbMatrix pairwise_eq_res;
  fetch_eq_res(pIdx, expandCircuitA, expandCircuitB, pairwise_eq_res, eval,
               runtime);
  if (COUT_LOG) cout << "success eq" << endl;

  // 2. construct the required elements.
  si64Matrix expandSharedM(block_length, sharedM.cols());
  p = 0, i = start_i, j = start_j;
  while (i < end_i) {
    for (j; j < n; j++) {
      expandSharedM.mShares[0](p, 0) = sharedM.mShares[0](j, 0);
      expandSharedM.mShares[1](p, 0) = sharedM.mShares[1](j, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    expandSharedM.mShares[0](p, 0) = sharedM.mShares[0](j, 0);
    expandSharedM.mShares[1](p, 0) = sharedM.mShares[1](j, 0);
    p++;
  }
  // 3. Multiply the required elements with the required one-hot encodings.
  cipher_mul_seq(pIdx, expandSharedM, pairwise_eq_res, expandSharedM, eval, enc,
                 runtime);

  // cout << "success mul" << endl;

  // 4. reduce to the final result.
  i = start_i, j = start_j;
  p = 0;
  while (i < end_i) {
    for (j; j < n; j++) {
      res.mShares[0](i, 0) += expandSharedM.mShares[0](p, 0);
      res.mShares[1](i, 0) += expandSharedM.mShares[1](p, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    res.mShares[0](i, 0) += expandSharedM.mShares[0](p, 0);
    res.mShares[1](i, 0) += expandSharedM.mShares[1](p, 0);
    p++;
  }
  // cout << "about to success" << endl;

  return 0;
}

// repeat-then-reduce version cipher_index.
int rtr_cipher_index(int pIdx, si64Matrix &sharedM, si64Matrix &indexMatrix,
                     si64Matrix &res, Sh3Evaluator &eval, Sh3Runtime &runtime,
                     Sh3Encryptor &enc) {
  // 0. define the config value.
  u64 n = sharedM.rows(), m = indexMatrix.rows();
  u64 expand_length = n * m;

  // 1. initialize the res to zeros.
  i64Matrix zeros(m, 1);
  for (int i = 0; i < m; i++) zeros(i, 0) = 0;
  res.resize(m, 1);
  if (pIdx == 0) {
    enc.localIntMatrix(runtime, zeros, res).get();
  } else {
    enc.remoteIntMatrix(runtime, res).get();
  }

  // 2. separate the task by expand_length.
  int numBlocks = (expand_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int parallelTimes =
      (numBlocks / THREADS_SIZE > 1) ? (numBlocks / THREADS_SIZE) : 1;
  int curBlock = 0;

  vector<Sh3Task> vecTasks;
  for (int i = 0; i < numBlocks; i++) {
    int offsetLeft = curBlock * BLOCK_SIZE;
    int offsetRight = (curBlock + 1) * BLOCK_SIZE;
    if (curBlock == numBlocks - 1) {
      offsetRight = (int)expand_length;
    }
    // execute computation of each block in parallel.
    Sh3Task taskCur = runtime.noDependencies();
    string task_id = "task-" + to_string(curBlock) + "-" + to_string(pIdx);
    Sh3Task dep = taskCur.then(
        [offsetLeft, offsetRight, curBlock, pIdx, task_id, &sharedM,
         &indexMatrix, &res, &eval, &runtime, &enc](Sh3Task &self) {
          cipher_index_offset(pIdx, sharedM, indexMatrix, res, eval, runtime,
                              enc, self, offsetLeft, offsetRight);
        },
        task_id);
    vecTasks.push_back(dep);
    // compute the next block.
    curBlock++;
  }
  // only after all the tasks in the current parallel finished, start the next
  // group.
  for (auto &dep : vecTasks) dep.get();

  if (runtime.mIsActive) {
    cout << "party - " << pIdx << " INDEED EXISTS ACTIVE TASKS" << endl;
  }
  return 0;
}

// isolated argsort offset, computing between offsetLeft and offsetRight.
int cipher_argsort_offset(int pIdx, si64Matrix &isharedM, si64Matrix &res,
                          Sh3Evaluator &eval, Sh3Runtime &runtime,
                          Sh3Encryptor &enc, Sh3Task &task, int offsetLeft,
                          int offsetRight) {
  // 1. compute the diff between target elements from offsetLeft to offsetRight.
  u64 block_length = offsetRight - offsetLeft;
  u64 n = isharedM.rows();
  si64Matrix diff(block_length, 1);

  u64 start_i = offsetLeft / n, start_j = offsetLeft % n;
  u64 end_i = offsetRight / n, end_j = offsetRight % n;
  u64 p = 0, i = start_i, j = start_j;

  while (i < end_i) {
    for (j; j < n; j++) {
      diff.mShares[0](p, 0) =
          isharedM.mShares[0](j, 0) - isharedM.mShares[0](i, 0);
      diff.mShares[1](p, 0) =
          isharedM.mShares[1](j, 0) - isharedM.mShares[1](i, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    diff.mShares[0](p, 0) =
        isharedM.mShares[0](j, 0) - isharedM.mShares[0](end_i, 0);
    diff.mShares[1](p, 0) =
        isharedM.mShares[1](j, 0) - isharedM.mShares[1](end_i, 0);
    p++;
  }

  // 2. get the msb (comparision result) of the diff computed above.
  sbMatrix pairwise_comp;
  fetch_msb(pIdx, diff, pairwise_comp, eval, runtime, task);

  // 3. convert the comparision results to secret ints.
  i64Matrix ones(block_length, 1);
  for (int i = 0; i < block_length; i++) ones(i, 0) = 1;
  si64Matrix ipairwise(block_length, 1);

  // synchronzed initialize the secret ipairwise matrix to ones.
  if (pIdx == 0) {
    for (int i = 0; i < ipairwise.mShares[0].size(); ++i) {
      ipairwise.mShares[0](i) = enc.mShareGen.getShare() + ones(i);
    }
    runtime.mComm.mNext.asyncSendCopy(ipairwise.mShares[0].data(),
                                      ipairwise.mShares[0].size());
    auto fu = runtime.mComm.mPrev.asyncRecv(ipairwise.mShares[1].data(),
                                            ipairwise.mShares[1].size());
    fu.get();
  } else {
    for (int i = 0; i < ipairwise.mShares[0].size(); ++i) {
      ipairwise.mShares[0](i) = enc.mShareGen.getShare();
    }
    runtime.mComm.mNext.asyncSendCopy(ipairwise.mShares[0].data(),
                                      ipairwise.mShares[0].size());
    auto fu = runtime.mComm.mPrev.asyncRecv(ipairwise.mShares[1].data(),
                                            ipairwise.mShares[1].size());
    fu.get();
  }
  // multiply ones with sbMatrix pairwise_comp, convert to secret ints.
  cipher_mul_seq(pIdx, ipairwise, pairwise_comp, ipairwise, eval, enc, runtime);

  // 4. reduce the result to the global res.
  i = start_i, j = start_j;
  p = 0;
  while (i < end_i) {
    for (j; j < n; j++) {
      res.mShares[0](i, 0) += ipairwise.mShares[0](p, 0);
      res.mShares[1](i, 0) += ipairwise.mShares[1](p, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    res.mShares[0](i, 0) += ipairwise.mShares[0](p, 0);
    res.mShares[1](i, 0) += ipairwise.mShares[1](p, 0);
    p++;
  }
  return 0;
}

// repeat-then-reduce version cipher_argsort
int rtr_cipher_argsort(int pIdx, si64Matrix &sharedM, si64Matrix &res,
                       Sh3Evaluator &eval, Sh3Runtime &runtime,
                       Sh3Encryptor &enc) {
  // 0. define the separate volumn.
  u64 n = sharedM.rows();
  u64 expand_length = n * n;

  // 1. initialize res to all zeros matrix.
  i64Matrix zeros(n, 1);
  for (int i = 0; i < n; i++) zeros(i, 0) = 0;
  res.resize(n, 1);
  if (pIdx == 0) {
    enc.localIntMatrix(runtime, zeros, res).get();
  } else {
    enc.remoteIntMatrix(runtime, res).get();
  }

  // 2. separate the task by the largest volumn.
  int numBlocks = (expand_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
  vector<Sh3Task> vecTasks;

  for (int curBlock = 0; curBlock < numBlocks; curBlock++) {
    // compute the offsets.
    int offsetLeft = curBlock * BLOCK_SIZE;
    int offsetRight = (curBlock + 1) * BLOCK_SIZE;
    if (curBlock == numBlocks - 1) {
      offsetRight = (int)expand_length;
    }
    // execute computation of each block in parallel.
    Sh3Task taskCur = runtime.noDependencies();
    string task_id = "task-" + to_string(curBlock) + "-" + to_string(pIdx);
    Sh3Task dep =
        taskCur
            .then(
                [&, offsetLeft, offsetRight](Sh3Task &self) {
                  cipher_argsort_offset(pIdx, sharedM, res, eval, runtime, enc,
                                        self, offsetLeft, offsetRight);
                },
                task_id)
            .getClosure();
    vecTasks.push_back(dep);
  }

  for (auto &dep : vecTasks) dep.get();

  if (runtime.mIsActive) {
    cout << "party - " << pIdx << " INDEED EXISTS ACTIVE TASKS" << endl;
  }
  return 0;
}

// pure repeat-version cipher_argsort.
int cipher_argsort(int pIdx, si64Matrix &sharedM, si64Matrix &res,
                   Sh3Evaluator &eval, Sh3Runtime &runtime, Sh3Encryptor &enc) {
  Sh3Task task = runtime.noDependencies();
  u64 n = sharedM.rows();
  i64Matrix zeros(n, 1);
  for (int i = 0; i < n; i++) zeros(i, 0) = 0;
  res.resize(n, 1);
  if (pIdx == 0) {
    enc.localIntMatrix(runtime, zeros, res).get();
  } else {
    enc.remoteIntMatrix(runtime, res).get();
  }
  return cipher_argsort_offset(pIdx, sharedM, res, eval, runtime, enc, task, 0,
                               (int)n * n);
}

// pure repeat-version cipher_binning algorithm.
int cipher_binning(int pIdx, si64Matrix &sharedM, i64Matrix &bins,
                   i64Matrix &targetVals, si64Matrix &res, Sh3Evaluator &eval,
                   Sh3Runtime &runtime, Sh3Encryptor &enc) {
  Sh3Task task = runtime.noDependencies();
  u64 n = sharedM.rows(), m = bins.rows();
  i64Matrix zeros(n, 1);
  for (int i = 0; i < n; i++) zeros(i, 0) = 0;
  res.resize(n, 1);
  if (pIdx == 0) {
    enc.localIntMatrix(runtime, zeros, res).get();
  } else {
    enc.remoteIntMatrix(runtime, res).get();
  }
  return cipher_binning_offset(pIdx, sharedM, bins, targetVals, res, eval,
                               runtime, enc, task, 0, (int)n * m);
}

// isolated cipher binning, computeing between offsetLeft and offsetRight.
int cipher_binning_offset(int pIdx, si64Matrix &sharedM, i64Matrix &bins,
                          i64Matrix &targetVals, si64Matrix &res,
                          Sh3Evaluator &eval, Sh3Runtime &runtime,
                          Sh3Encryptor &enc, Sh3Task &task, int offsetLeft,
                          int offsetRight) {
  // 0. set the configs.
  u64 block_length = offsetRight - offsetLeft;
  u64 n = sharedM.rows(), m = bins.rows();
  u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  u64 end_i = offsetRight / m, end_j = offsetRight % m;
  u64 extend_end_j = end_j;

  if (end_j != 0) {
    block_length += 1;
    extend_end_j += 1;
  }

  // 1. compare values with bins. 1-1: compute the pairwise difference.
  u64 p = 0, i = start_i, j = start_j;
  si64Matrix diffMB(block_length, 1);
  while (i < end_i) {
    for (j; j < m; j++) {
      switch (pIdx) {
        case 0: {
          diffMB.mShares[0](p) = -sharedM.mShares[0](i, 0) + bins(j, 0);
          diffMB.mShares[1](p) = -sharedM.mShares[1](i, 0);
          break;
        }
        case 1: {
          diffMB.mShares[0](p) = -sharedM.mShares[0](i, 0);
          diffMB.mShares[1](p) = -sharedM.mShares[1](i, 0) + bins(j, 0);
          break;
        }
        case 2: {
          diffMB.mShares[0](p) = -sharedM.mShares[0](i, 0);
          diffMB.mShares[1](p) = -sharedM.mShares[1](i, 0);
          break;
        }
      }
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < extend_end_j; j++) {
    switch (pIdx) {
      case 0: {
        diffMB.mShares[0](p) = -sharedM.mShares[0](i, 0) + bins(j, 0);
        diffMB.mShares[1](p) = -sharedM.mShares[1](i, 0);
        break;
      }
      case 1: {
        diffMB.mShares[0](p) = -sharedM.mShares[0](i, 0);
        diffMB.mShares[1](p) = -sharedM.mShares[1](i, 0) + bins(j, 0);
        break;
      }
      case 2: {
        diffMB.mShares[0](p) = -sharedM.mShares[0](i, 0);
        diffMB.mShares[1](p) = -sharedM.mShares[1](i, 0);
      }
    }
    p++;
  }
  sbMatrix oneHots;
  fetch_msb(pIdx, diffMB, oneHots, eval, runtime, task);
  // 2. sequentially compute the XOR to fetch the last one in each binning.
  p = 0, i = start_i, j = start_j;
  while (i < end_i) {
    for (j; j < m - 1; j++) {
      oneHots.mShares[0](i * m + j - offsetLeft) ^=
          oneHots.mShares[0](i * m + j + 1 - offsetLeft);
      oneHots.mShares[1](i * m + j - offsetLeft) ^=
          oneHots.mShares[1](i * m + j + 1 - offsetLeft);
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    oneHots.mShares[0](end_i * m + j - offsetLeft) ^=
        oneHots.mShares[0](end_i * m + j + 1 - offsetLeft);
    oneHots.mShares[1](end_i * m + j - offsetLeft) ^=
        oneHots.mShares[1](end_i * m + j + 1 - offsetLeft);
    p++;
  }

  // 3. expand the tarVals to compute target elements.
  i64Matrix expandTarVals(block_length, 1);
  p = 0, i = start_i, j = start_j;
  while (i < end_i) {
    for (j; j < m; j++) {
      expandTarVals(p) = targetVals(j);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    expandTarVals(p) = targetVals(j);
    p++;
  }
  if (extend_end_j > end_j) expandTarVals(i * m + extend_end_j) = 0;

  si64Matrix maskedTarVals(block_length, 1);
  pi_cb_mul(pIdx, expandTarVals, oneHots, maskedTarVals, eval, enc, runtime);

  // 4. reduce the mastedTarVals to the fin(l result.
  p = 0, i = start_i, j = start_j;
  while (i < end_i) {
    for (j; j < m; j++) {
      res.mShares[0](i) += maskedTarVals.mShares[0](p);
      res.mShares[1](i) += maskedTarVals.mShares[1](p);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    res.mShares[0](i) += maskedTarVals.mShares[0](p);
    res.mShares[1](i) += maskedTarVals.mShares[1](p);
    p++;
  }
  return 0;
}

// repeat-then-reduce version cipher_binning algorithm.
int rtr_cipher_binning(int pIdx, si64Matrix &sharedM, i64Matrix &bins,
                       i64Matrix &targetVals, si64Matrix &res,
                       Sh3Evaluator &eval, Sh3Runtime &runtime,
                       Sh3Encryptor &enc) {
  // 0. define the separate volumn.
  u64 n = sharedM.rows(), m = bins.rows();
  u64 expand_length = n * m;
  // 1. initialize res to all zeros matrix.
  i64Matrix zeros(n, 1);
  for (int i = 0; i < n; i++) zeros(i, 0) = 0;
  res.resize(n, 1);
  if (pIdx == 0) {
    enc.localIntMatrix(runtime, zeros, res).get();
  } else {
    enc.remoteIntMatrix(runtime, res).get();
  }

  // 2. separate the task by the largest volumn.
  int numBlocks = (expand_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
  vector<Sh3Task> vecTasks;

  for (int curBlock = 0; curBlock < numBlocks; curBlock++) {
    // compute the offsets.
    int offsetLeft = curBlock * BLOCK_SIZE;
    int offsetRight = (curBlock + 1) * BLOCK_SIZE;
    if (curBlock == numBlocks - 1) {
      offsetRight = (int)expand_length;
    }
    // execute computation of each block in parallel.
    Sh3Task taskCur = runtime.noDependencies();
    string task_id = "task-" + to_string(curBlock) + "-" + to_string(pIdx);
    Sh3Task dep = taskCur
                      .then(
                          [&, offsetLeft, offsetRight](Sh3Task &self) {
                            cipher_binning_offset(
                                pIdx, sharedM, bins, targetVals, res, eval,
                                runtime, enc, self, offsetLeft, offsetRight);
                          },
                          task_id)
                      .getClosure();
    vecTasks.push_back(dep);
  }

  for (auto &dep : vecTasks) dep.get();

  if (runtime.mIsActive) {
    cout << "party - " << pIdx << " INDEED EXISTS ACTIVE TASKS" << endl;
  }
  return 0;
}

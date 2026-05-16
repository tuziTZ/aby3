#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>
#include <cryptoTools/Network/IOService.h>
#include <mpi.h>

#include "BuildingBlocks.h"

static const int BLOCK_SIZE = 50000;
  
extern MPI_Comm COMM_WORLD;  

#ifndef _RTR_H_
#define _RTR_H_

template <typename T>
struct tensor {
  T &data;
  int dim = 1;
  int size = 1;
  std::vector<int> shape;

  tensor(T &val, std::vector<int> &val_shape) {
    data = val;
    dim = shape.size();
    shape = val_shape;
    for (int i = 0; i < dim; i++) {
      size *= shape[i];
    }
  }
};

template <typename T, typename U, typename Y = T, typename V = void *>
int repeat_then_reduce_pure(
    int pIdx, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
    aby3::Sh3Encryptor &enc, T &sharedX, int n, U &vectorY, int m, int axis,
    int (*relation)(int, T &, int, U &, int, aby3::Sh3Evaluator &,
                    aby3::Sh3Runtime &, aby3::sbMatrix &),
    int (*reduce)(int, int, int, int, aby3::sbMatrix &, aby3::Sh3Evaluator &,
                  aby3::Sh3Runtime &, aby3::Sh3Encryptor &, Y &, V &),
    Y &res, V v = V()) {
  // define the expand length.
  aby3::sbMatrix pairwise_relationship;

  // initiate the result with all zeros.
  int res_length = (axis == 0) ? m : n;
  init_zeros(pIdx, enc, runtime, res, res_length);

  // repeat for pairwise relationship.
  relation(pIdx, sharedX, n, vectorY, m, eval, runtime, pairwise_relationship);

  // reduce for the final result.
  reduce(pIdx, axis, n, m, pairwise_relationship, eval, runtime, enc, res, v);

  return 0;
}

template <typename T, typename U, typename Y = T, typename V = void *>
int repeat_then_reduce(
    int pIdx, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
    aby3::Sh3Encryptor &enc, T &sharedX, int n, U &vectorY, int m, int axis,
    int (*relation)(int, T &, int, U &, int, aby3::Sh3Evaluator &,
                    aby3::Sh3Runtime &, aby3::sbMatrix &, int, int),
    int (*reduce)(int, int, int, int, aby3::sbMatrix &, aby3::Sh3Evaluator &,
                  aby3::Sh3Runtime &, aby3::Sh3Encryptor &, Y &, int, int, V &),
    Y &res, V v = V()) {
  // define the expand length.
  aby3::u64 expand_length = n * m;

  // initiate the result with all zeros.
  int res_length = (axis == 0) ? m : n;
  init_zeros(pIdx, enc, runtime, res, res_length);

  int curBlock = 0;
  int numBlocks = (expand_length + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int i = 0; i < numBlocks; i++) {
    int offsetLeft = curBlock * BLOCK_SIZE;
    int offsetRight = (curBlock + 1) * BLOCK_SIZE;
    if (curBlock == numBlocks - 1) {
      offsetRight = (int)expand_length;
    }
    aby3::sbMatrix pairwise_relationship;
    // compute the pairwise relationship.
    relation(pIdx, sharedX, n, vectorY, m, eval, runtime, pairwise_relationship,
             offsetLeft, offsetRight);

    // reduce for the final result.
    reduce(pIdx, axis, n, m, pairwise_relationship, eval, runtime, enc, res,
           offsetLeft, offsetRight, v);
    curBlock++;
  }

  return 0;
}

inline int relation_gt(int pIdx, aby3::si64Matrix &sharedX, int n,
                       aby3::si64Matrix &sharedY, int m,
                       aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                       aby3::sbMatrix &res) {
  aby3::u64 pairwise_length = n * m;
  aby3::si64Matrix diff(pairwise_length, 1);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      diff.mShares[0](i * m + j, 0) =
          sharedX.mShares[0](j, 0) - sharedY.mShares[0](i, 0);
      diff.mShares[1](i * m + j, 0) =
          sharedX.mShares[1](j, 0) - sharedY.mShares[1](i, 0);
    }
  }
  fetch_msb(pIdx, diff, res, eval, runtime);
  return 0;
}

inline int relation_gt_offset(int pIdx, aby3::si64Matrix &sharedX, int n,
                              aby3::si64Matrix &sharedY, int m,
                              aby3::Sh3Evaluator &eval,
                              aby3::Sh3Runtime &runtime, aby3::sbMatrix &res,
                              int offsetLeft, int offsetRight) {
  aby3::u64 block_length = offsetRight - offsetLeft;

  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 p = 0, i = start_i, j = start_j;
  aby3::u64 extend_end_j = end_j;

  if (end_j != 0) {
    block_length += 1;
    extend_end_j += 1;
  }

  aby3::si64Matrix diff(block_length, 1);

  while (i < end_i) {
    for (j; j < m; j++) {
      diff.mShares[0](p, 0) =
          sharedX.mShares[0](j, 0) - sharedY.mShares[0](i, 0);
      diff.mShares[1](p, 0) =
          sharedX.mShares[1](j, 0) - sharedY.mShares[1](i, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < extend_end_j; j++) {
    diff.mShares[0](p, 0) =
        sharedX.mShares[0](j, 0) - sharedY.mShares[0](end_i, 0);
    diff.mShares[1](p, 0) =
        sharedX.mShares[1](j, 0) - sharedY.mShares[1](end_i, 0);
    p++;
  }
  fetch_msb(pIdx, diff, res, eval, runtime);
  return 0;
}

inline int relation_gt(int pIdx, aby3::si64Matrix &sharedX, int n,
                       aby3::i64Matrix &sharedY, int m,
                       aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                       aby3::sbMatrix &res) {
  aby3::u64 pairwise_length = n * m;
  aby3::si64Matrix diff(pairwise_length, 1);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      switch (pIdx) {
        case 0: {
          diff.mShares[0](i * m + j) =
              -sharedX.mShares[0](i, 0) + sharedY(j, 0);
          diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0);
          break;
        }
        case 1: {
          diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0);
          diff.mShares[1](i * m + j) =
              -sharedX.mShares[1](i, 0) + sharedY(j, 0);
          break;
        }
        case 2: {
          diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0);
          diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0);
          break;
        }
      }
    }
  }
  fetch_msb(pIdx, diff, res, eval, runtime);
  return 0;
}

inline int relation_gt_offset(int pIdx, aby3::si64Matrix &sharedX, int n,
                              aby3::i64Matrix &sharedY, int m,
                              aby3::Sh3Evaluator &eval,
                              aby3::Sh3Runtime &runtime, aby3::sbMatrix &res,
                              int offsetLeft, int offsetRight) {
  aby3::u64 block_length = offsetRight - offsetLeft;

  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 p = 0, i = start_i, j = start_j;
  aby3::u64 extend_end_j = end_j;

  if (end_j != 0) {
    block_length += 1;
    extend_end_j += 1;
  }

  aby3::si64Matrix diff(block_length, 1);

  while (i < end_i) {
    for (j; j < m; j++) {
      switch (pIdx) {
        case 0: {
          diff.mShares[0](i * m + j) =
              -sharedX.mShares[0](i, 0) + sharedY(j, 0);
          diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0);
          break;
        }
        case 1: {
          diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0);
          diff.mShares[1](i * m + j) =
              -sharedX.mShares[1](i, 0) + sharedY(j, 0);
          break;
        }
        case 2: {
          diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0);
          diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0);
          break;
        }
      }
    }
    j = 0;
    i++;
  }
  for (j = 0; j < extend_end_j; j++) {
    switch (pIdx) {
      case 0: {
        diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0) + sharedY(j, 0);
        diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0);
        break;
      }
      case 1: {
        diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0);
        diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0) + sharedY(j, 0);
        break;
      }
      case 2: {
        diff.mShares[0](i * m + j) = -sharedX.mShares[0](i, 0);
        diff.mShares[1](i * m + j) = -sharedX.mShares[1](i, 0);
        break;
      }
    }
  }

  fetch_msb(pIdx, diff, res, eval, runtime);
  return 0;
}

inline int relation_eq(int pIdx, aby3::si64Matrix &sharedX, int n,
                       std::vector<int> &vectorY, int m,
                       aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                       aby3::sbMatrix &res) {
  aby3::u64 pairwise_length = n * m;
  aby3::sbMatrix expandCircuitA(pairwise_length, sizeof(aby3::i64) * 8),
      expandCircuitB(pairwise_length, sizeof(aby3::i64) * 8);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      switch (pIdx) {
        case 0: {
          expandCircuitA.mShares[0](i * m + j) =
              vectorY[j] - sharedX.mShares[0](i, 0) - sharedX.mShares[1](i, 0);
          break;
        }
        case 1: {
          expandCircuitB.mShares[0](i * m + j) = sharedX.mShares[0](i, 0);
          break;
        }
        case 2: {
          expandCircuitB.mShares[1](i * m + j) = sharedX.mShares[1](i, 0);
        }
      }
    }
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

  runtime.mComm.mNext.asyncSend(expandCircuitA.mShares[0].data(),
                                expandCircuitA.mShares[0].size());
  auto fu = runtime.mComm.mPrev.asyncRecv(expandCircuitA.mShares[1].data(),
                                          expandCircuitA.mShares[1].size());
  fu.get();

  fetch_eq_res(pIdx, expandCircuitA, expandCircuitB, res, eval, runtime);

  return 0;
}

inline int relation_eq_offset(int pIdx, aby3::si64Matrix &sharedX, int n,
                              std::vector<int> &vectorY, int m,
                              aby3::Sh3Evaluator &eval,
                              aby3::Sh3Runtime &runtime, aby3::sbMatrix &res,
                              int offsetLeft, int offsetRight) {
  aby3::u64 pairwise_length = offsetRight - offsetLeft;
  aby3::sbMatrix expandCircuitA(pairwise_length, sizeof(aby3::i64) * 8),
      expandCircuitB(pairwise_length, sizeof(aby3::i64) * 8);

  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 p = 0, i = start_i, j = start_j;

  while (i < end_i) {
    for (j; j < m; j++) {
      switch (pIdx) {
        case 0: {
          expandCircuitA.mShares[0](p) =
              vectorY[j] - sharedX.mShares[0](i, 0) - sharedX.mShares[1](i, 0);
          break;
        }
        case 1: {
          expandCircuitB.mShares[0](p) = sharedX.mShares[0](i, 0);
          break;
        }
        case 2: {
          expandCircuitB.mShares[1](p) = sharedX.mShares[1](i, 0);
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
        expandCircuitA.mShares[0](p) = vectorY[j] -
                                       sharedX.mShares[0](end_i, 0) -
                                       sharedX.mShares[1](end_i, 0);
        break;
      }
      case 1: {
        expandCircuitB.mShares[0](p) = sharedX.mShares[0](end_i, 0);
        break;
      }
      case 2: {
        expandCircuitB.mShares[1](p) = sharedX.mShares[1](end_i, 0);
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

  runtime.mComm.mNext.asyncSend(expandCircuitA.mShares[0].data(),
                                expandCircuitA.mShares[0].size());
  auto fu = runtime.mComm.mPrev.asyncRecv(expandCircuitA.mShares[1].data(),
                                          expandCircuitA.mShares[1].size());
  fu.get();

  fetch_eq_res(pIdx, expandCircuitA, expandCircuitB, res, eval, runtime);

  return 0;
}

template <typename V = void *>
inline int reduce_count(int pIdx, int axis, int n, int m,
                        aby3::sbMatrix &pairwise_relationship,
                        aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                        aby3::Sh3Encryptor &enc, aby3::si64Matrix &res, V &v) {
  // for count, first convert the sbMatrix into siMatrix.
  aby3::si64Matrix si_pairwise_relationship(n * m, 1);
  init_ones(pIdx, enc, runtime, si_pairwise_relationship, n * m);

  // multiply ones with sbMatrix pairwise_comp, convert to secret ints.
  cipher_mul_seq(pIdx, si_pairwise_relationship, pairwise_relationship,
                 si_pairwise_relationship, eval, enc, runtime);

  int i, j;
  int &index = (axis == 0) ? j : i;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      res.mShares[0](index, 0) +=
          si_pairwise_relationship.mShares[0](i * n + j, 0);
      res.mShares[1](index, 0) +=
          si_pairwise_relationship.mShares[1](i * n + j, 0);
    }
  }

  return 0;
}

template <typename V = void *>
inline int reduce_count_offset(int pIdx, int axis, int n, int m,
                               aby3::sbMatrix &pairwise_relationship,
                               aby3::Sh3Evaluator &eval,
                               aby3::Sh3Runtime &runtime,
                               aby3::Sh3Encryptor &enc, aby3::si64Matrix &res,
                               int offsetLeft, int offsetRight, V &v) {
  // for count, first convert the sbMatrix into siMatrix.
  aby3::si64Matrix si_pairwise_relationship((offsetRight - offsetLeft), 1);

  init_ones(pIdx, enc, runtime, si_pairwise_relationship,
            (offsetRight - offsetLeft));
  // multiply ones with sbMatrix pairwise_comp, convert to secret ints.
  cipher_mul_seq(pIdx, si_pairwise_relationship, pairwise_relationship,
                 si_pairwise_relationship, eval, enc, runtime);

  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 p = 0, i = start_i, j = start_j;

  i = start_i, j = start_j;
  aby3::u64 &index = (axis == 0) ? j : i;
  p = 0;
  while (i < end_i) {
    for (j; j < m; j++) {
      res.mShares[0](index, 0) += si_pairwise_relationship.mShares[0](p, 0);
      res.mShares[1](index, 0) += si_pairwise_relationship.mShares[1](p, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    res.mShares[0](index, 0) += si_pairwise_relationship.mShares[0](p, 0);
    res.mShares[1](index, 0) += si_pairwise_relationship.mShares[1](p, 0);
    p++;
  }

  return 0;
}

inline int reduce_select(int pIdx, int axis, int n, int m,
                         aby3::sbMatrix &pairwise_relationship,
                         aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                         aby3::Sh3Encryptor &enc, aby3::si64Matrix &res,
                         aby3::si64Matrix &selectValue) {
  // for select, repeat the selectValue to the pairwise relation length, in
  // order to vectorized multiply.
  aby3::si64Matrix expandValue(n * m, 1);
  int i, j;
  int &index = (axis == 0) ? j : i;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      expandValue.mShares[0](i * m + j, 0) = selectValue.mShares[0](j, 0);
      expandValue.mShares[1](i * m + j, 0) = selectValue.mShares[1](j, 0);
    }
  }

  // multiply to select the corresponding elements.
  cipher_mul_seq(pIdx, expandValue, pairwise_relationship, expandValue, eval,
                 enc, runtime);

  // reduce to the final result.
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      res.mShares[0](index, 0) += expandValue.mShares[0](i * m + j, 0);
      res.mShares[1](index, 0) += expandValue.mShares[1](i * m + j, 0);
    }
  }

  return 0;
}

inline int reduce_select(int pIdx, int axis, int n, int m,
                         aby3::sbMatrix &pairwise_relationship,
                         aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                         aby3::Sh3Encryptor &enc, aby3::si64Matrix &res,
                         aby3::i64Matrix &selectValue) {
  // for select, repeat the selectValue to the pairwise relation length, in
  // order to vectorized multiply.
  aby3::i64Matrix expandValue(n * m, 1);
  int i, j;
  int &index = (axis == 0) ? j : i;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      expandValue(i * m + j, 0) = selectValue(j, 0);
    }
  }
  aby3::si64Matrix si_expandValue(n * m, 1);
  // multiply to select the corresponding elements.
  cipher_mul_seq(pIdx, expandValue, pairwise_relationship, si_expandValue, eval,
                 enc, runtime);

  // reduce to the final result.
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      res.mShares[0](index, 0) += si_expandValue.mShares[0](i * m + j, 0);
      res.mShares[1](index, 0) += si_expandValue.mShares[1](i * m + j, 0);
    }
  }

  return 0;
}

inline int reduce_select_offset(int pIdx, int axis, int n, int m,
                                aby3::sbMatrix &pairwise_relationship,
                                aby3::Sh3Evaluator &eval,
                                aby3::Sh3Runtime &runtime,
                                aby3::Sh3Encryptor &enc, aby3::si64Matrix &res,
                                int offsetLeft, int offsetRight,
                                aby3::si64Matrix &selectValue) {
  // for select, repeat the selectValue to the pairwise relation length, in
  // order to vectorized multiply.
  aby3::u64 block_length = offsetRight - offsetLeft;
  aby3::si64Matrix expandValue(block_length, 1);

  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 p = 0, i = start_i, j = start_j;

  i = start_i, j = start_j;
  p = 0;

  while (i < end_i) {
    for (j; j < m; j++) {
      expandValue.mShares[0](p, 0) = selectValue.mShares[0](j, 0);
      expandValue.mShares[1](p, 0) = selectValue.mShares[1](j, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    expandValue.mShares[0](p, 0) = selectValue.mShares[0](j, 0);
    expandValue.mShares[1](p, 0) = selectValue.mShares[1](j, 0);
    p++;
  }

  // multiply to select the corresponding elements.
  cipher_mul_seq(pIdx, expandValue, pairwise_relationship, expandValue, eval,
                 enc, runtime);

  i = start_i, j = start_j;
  aby3::u64 &index = (axis == 0) ? j : i;
  p = 0;
  while (i < end_i) {
    for (j; j < m; j++) {
      res.mShares[0](index, 0) += expandValue.mShares[0](p, 0);
      res.mShares[1](index, 0) += expandValue.mShares[1](p, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    res.mShares[0](index, 0) += expandValue.mShares[0](p, 0);
    res.mShares[1](index, 0) += expandValue.mShares[1](p, 0);
    p++;
  }

  return 0;
}

inline int reduce_select_offset(int pIdx, int axis, int n, int m,
                                aby3::sbMatrix &pairwise_relationship,
                                aby3::Sh3Evaluator &eval,
                                aby3::Sh3Runtime &runtime,
                                aby3::Sh3Encryptor &enc, aby3::si64Matrix &res,
                                int offsetLeft, int offsetRight,
                                aby3::i64Matrix &selectValue) {
  // for select, repeat the selectValue to the pairwise relation length, in
  // order to vectorized multiply.
  aby3::u64 block_length = offsetRight - offsetLeft;
  aby3::i64Matrix expandValue(block_length, 1);

  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 p = 0, i = start_i, j = start_j;

  i = start_i, j = start_j;
  p = 0;

  while (i < end_i) {
    for (j; j < m; j++) {
      expandValue(p, 0) = selectValue(j, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    expandValue(p, 0) = selectValue(j, 0);
    p++;
  }

  aby3::si64Matrix si_expandValue(block_length, 1);
  // multiply to select the corresponding elements.
  cipher_mul_seq(pIdx, expandValue, pairwise_relationship, si_expandValue, eval,
                 enc, runtime);

  i = start_i, j = start_j;
  aby3::u64 &index = (axis == 0) ? j : i;
  p = 0;
  while (i < end_i) {
    for (j; j < m; j++) {
      res.mShares[0](index, 0) += si_expandValue.mShares[0](p, 0);
      res.mShares[1](index, 0) += si_expandValue.mShares[1](p, 0);
      p++;
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    res.mShares[0](index, 0) += si_expandValue.mShares[0](p, 0);
    res.mShares[1](index, 0) += si_expandValue.mShares[1](p, 0);
    p++;
  }

  return 0;
}
template <typename T>
inline int reduce_lb_select(int pIdx, int axis, int n, int m,
                            aby3::sbMatrix &pairwise_relationship,
                            aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                            aby3::Sh3Encryptor &enc, aby3::si64Matrix &res,
                            T &selectValue) {
  // firstly construct the one-hot vector.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m - 1; j++) {
      pairwise_relationship.mShares[0](i * m + j) ^=
          pairwise_relationship.mShares[0](i * m + j + 1);
      pairwise_relationship.mShares[1](i * m + j) ^=
          pairwise_relationship.mShares[1](i * m + j + 1);
    }
  }

  return reduce_select(pIdx, axis, n, m, pairwise_relationship, eval, runtime,
                       enc, res, selectValue);
}

template <typename T>
inline int reduce_lb_select_offset(int pIdx, int axis, int n, int m,
                                   aby3::sbMatrix &pairwise_relationship,
                                   aby3::Sh3Evaluator &eval,
                                   aby3::Sh3Runtime &runtime,
                                   aby3::Sh3Encryptor &enc,
                                   aby3::si64Matrix &res, int offsetLeft,
                                   int offsetRight, T &selectValue) {
  aby3::u64 block_length = offsetRight - offsetLeft;
  aby3::u64 start_i = offsetLeft / m, start_j = offsetLeft % m;
  aby3::u64 end_i = offsetRight / m, end_j = offsetRight % m;
  aby3::u64 extend_end_j = end_j;

  if (end_j != 0) {
    block_length += 1;
    extend_end_j += 1;
  }

  aby3::u64 i = start_i, j = start_j;
  while (i < end_i) {
    for (j; j < m - 1; j++) {
      pairwise_relationship.mShares[0](i * m + j - offsetLeft) ^=
          pairwise_relationship.mShares[0](i * m + j + 1 - offsetLeft);
      pairwise_relationship.mShares[1](i * m + j - offsetLeft) ^=
          pairwise_relationship.mShares[1](i * m + j + 1 - offsetLeft);
    }
    j = 0;
    i++;
  }
  for (j = 0; j < end_j; j++) {
    pairwise_relationship.mShares[0](end_i * m + j - offsetLeft) ^=
        pairwise_relationship.mShares[0](end_i * m + j + 1 - offsetLeft);
    pairwise_relationship.mShares[1](end_i * m + j - offsetLeft) ^=
        pairwise_relationship.mShares[1](end_i * m + j + 1 - offsetLeft);
  }

  return reduce_select_offset(pIdx, axis, n, m, pairwise_relationship, eval,
                              runtime, enc, res, offsetLeft, offsetRight,
                              selectValue);
}

int argsort(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &res,
            aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
            aby3::Sh3Encryptor &enc);

template <aby3::Decimal D>
int argsort(int pIdx, aby3::sf64Matrix<D> &sharedM, aby3::si64Matrix &res,
            aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
            aby3::Sh3Encryptor &enc) {
  return argsort(pIdx, sharedM.i64Cast(), res, eval, runtime, enc);
}

int secret_index(int pIdx, aby3::si64Matrix &sharedM,
                 aby3::si64Matrix &secretIndex, aby3::si64Matrix &res,
                 aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
                 aby3::Sh3Encryptor &enc);

int get_binning_value(int pIdx, aby3::si64Matrix &sharedM,
                      aby3::i64Matrix &bins, aby3::i64Matrix &targetVals,
                      aby3::si64Matrix &res, aby3::Sh3Evaluator &eval,
                      aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc);

int sort(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &res,
         aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
         aby3::Sh3Encryptor &enc);

int max_rtr(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &res,
        aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime,
        aby3::Sh3Encryptor &enc);
#endif
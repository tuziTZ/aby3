#pragma once
#include <cryptoTools/Network/IOService.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>


int normal_cipher_index(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &indexMatrix, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc);

int cipher_argsort_offset(int pIdx, aby3::si64Matrix& sharedM, aby3::si64Matrix& res, aby3::Sh3Evaluator& eval, aby3::Sh3Runtime& runtime, aby3::Sh3Encryptor& enc, aby3::Sh3Task& task, int offsetLeft, int offsetRight);

int cipher_argsort(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc);

template <aby3::Decimal D>
int cipher_argsort(int pIdx, aby3::sf64Matrix<D> &sharedM, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc){
    return cipher_argsort(pIdx, sharedM.i64Cast(), res, eval, runtime, enc);
}

int rtr_cipher_argsort(int pIdx, aby3::si64Matrix& sharedM, aby3::si64Matrix& res, aby3::Sh3Evaluator& eval, aby3::Sh3Runtime& runtime, aby3::Sh3Encryptor& enc);

template <aby3::Decimal D>
int rtr_cipher_argsort(int pIdx, aby3::sf64Matrix<D>& sharedM, aby3::si64Matrix& res, aby3::Sh3Evaluator& eval, aby3::Sh3Runtime& runtime, aby3::Sh3Encryptor& enc){
    return rtr_cipher_argsort(pIdx, sharedM.i64Cast(), res, eval, runtime, enc);
}

int cipher_index(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &cipherIndex, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor& enc);

int cipher_index_offset(int pIdx, aby3::si64Matrix &sharedM, aby3::si64Matrix &indexMatrix, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor& enc, aby3::Sh3Task& task, int offsetLeft, int offsetRight);

int rtr_cipher_index(int pIdx, aby3::si64Matrix& sharedM, aby3::si64Matrix& indexMatrix, aby3::si64Matrix& res, aby3::Sh3Evaluator& eval, aby3::Sh3Runtime&runtime, aby3::Sh3Encryptor& enc);

// return the cipher index indicating the bin that each cipher variable belongs to.
template <aby3::Decimal D>
int cipher_binning(int pIdx, aby3::sf64Matrix<D> &sharedM, aby3::f64Matrix<D> &bins,aby3::f64Matrix<D> &targetVals, aby3::sf64Matrix<D> &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc){
    return cipher_binning(pIdx, sharedM.i64Cast(), bins.i64Cast(), targetVals.i64Cast(), res.i64Cast(), eval, runtime, enc);
}

int cipher_binning(int pIdx, aby3::si64Matrix &sharedM, aby3::i64Matrix &bins, aby3::i64Matrix &targetVals, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc);

int cipher_binning_offset(int pIdx, aby3::si64Matrix &sharedM, aby3::i64Matrix &bins, aby3::i64Matrix &targetVals, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc, aby3::Sh3Task& task, int offsetLeft, int offsetRight);

int rtr_cipher_binning(int pIdx, aby3::si64Matrix &sharedM, aby3::i64Matrix &bins, aby3::i64Matrix &targetVals, aby3::si64Matrix &res, aby3::Sh3Evaluator &eval, aby3::Sh3Runtime &runtime, aby3::Sh3Encryptor &enc);
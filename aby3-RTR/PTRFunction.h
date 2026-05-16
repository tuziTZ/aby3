#pragma once
#include <aby3/sh3/Sh3Encryptor.h>
#include <aby3/sh3/Sh3Evaluator.h>
#include <aby3/sh3/Sh3FixedPoint.h>
#include <aby3/sh3/Sh3Runtime.h>
#include <aby3/sh3/Sh3Types.h>
#include <cryptoTools/Network/IOService.h>
#include <mpi.h>
#include <iomanip>

#include "./Pair_then_Reduce/include/datatype.h"
#include "./Pair_then_Reduce/include/tasks.h"
#include "BuildingBlocks.h"
#include "debug.h"

#define OPTIMAL_BLOCK 100
#define TASKS 5
#define TEST
// #define OPENMP

template<typename T>
struct indData{
  T value;
  aby3::si64 index;
};

#ifndef OPENMP

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubIndex : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubIndex(const size_t optimal_block, const int task_id, const int pIdx,
           aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
           aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = true;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "resLeft: " << std::endl;
      ofs.close();
      debug_output_vector(resLeft, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "resRight: " << std::endl;
      ofs.close();
      debug_output_vector(resRight, *(this->runtime), *(this->enc));
    }
#endif
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "local_res: " << std::endl;
      ofs.close();
      debug_output_vector(local_res, *(this->runtime), *(this->enc));
    }
#endif
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    aby3::u64 block_length = binfo->block_len;
    // pairwise vector equal test.
    aby3::sbMatrix partTable;

    vector_cipher_eq(this->pIdx, expandX, expandY, partTable, *(this->eval),
                     *(this->runtime));


    // pairwise vector abmul.
    aby3::si64Matrix expandV;
    expandV.resize(block_length, 1);
    for (size_t i = 0; i < block_length; i++) {
      expandV(i, 0, this->selectV[binfo->t_start + i]);
    }
    cipher_mul_seq(this->pIdx, expandV, partTable, expandV, *(this->eval),
                   *(this->enc), *(this->runtime));


    // transfor to the local_table
    for (size_t i = 0; i < block_length; i++) local_table[i] = expandV(i, 0);
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubRank : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubRank(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {

    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    aby3::u64 block_length = binfo->block_len;
    // debug_mpi(this->task_id, this->pIdx, "block start: " + to_string(binfo->t_start) + " end: " + to_string(binfo->t_end) + " comp size: " + to_string(binfo->t_end - binfo->t_start) + " block len: " + to_string(binfo->block_len) + "X size: " + to_string(expandX.size()));

    vector_cipher_gt(this->pIdx, expandX, expandY, local_table, *(this->eval),
                     *(this->enc), *(this->runtime));
    // debug_mpi(this->task_id, this->pIdx, "success single round");
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubSearch : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubSearch(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = true;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "resLeft: " << std::endl;
      ofs.close();
      debug_output_vector(resLeft, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "resRight: " << std::endl;
      ofs.close();
      debug_output_vector(resRight, *(this->runtime), *(this->enc));
    }
#endif
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "local_res: " << std::endl;
      ofs.close();
      debug_output_vector(local_res, *(this->runtime), *(this->enc));
    }
#endif
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    aby3::u64 block_length = binfo->block_len;
    aby3::sbMatrix partTable;

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "expandX: " << binfo->t_start << std::endl;
      ofs.close();
      debug_output_vector(expandX, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "expandY: " << binfo->t_start << std::endl;
      for (int i = 0; i < expandY.size(); i++) ofs << expandY[i] << " ";
      ofs << std::endl;
      ofs.close();
      // debug_output_vector(expandY, *(this->runtime), *(this->enc));
      // aby3::si64Matrix expandXM(block_length);
      // for(int i=0; i<block_length; i++) expandXM(i, 0, expandX[i]);
    }
#endif

    clock_t start, end;
    vector_cipher_ge(this->pIdx, expandX, expandY, partTable, *(this->eval),
                     *(this->enc), *(this->runtime));
    
    // pairwise vector abmul.
    aby3::si64Matrix expandV;
    expandV.resize(block_length, 1);
    for (size_t i = 0; i < block_length; i++) expandV(i, 0, this->selectV[binfo->t_start + i]);

    cipher_mul_seq(this->pIdx, expandV, partTable, expandV, *(this->eval),
                   *(this->enc), *(this->runtime));
    
    for (size_t i = 0; i < block_length; i++) local_table[i] = expandV(i, 0);
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubNewSearch : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubNewSearch(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = true;
    this->lookahead=1;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {

    aby3::u64 block_length = binfo->block_len;
    aby3::u64 expand_length = binfo->block_len + this->lookahead * this->n;
    if(binfo->t_start + expand_length >= (this->n * this->m)) expand_length = (this->n * this->m) - binfo->t_start;
    aby3::sbMatrix partTable(binfo->block_len, 1); 
    aby3::sbMatrix expandTable(expand_length, 1);

    // if(expandY.size() != expand_length) cout << "Y PROBLEM!!!" << " Y size = " << expandY.size() << " != " << expand_length << endl;
    // if(expandX.size() != expand_length) cout << "X PROBLEM!!!" << " X size = " << expandX.size() << " != " << expand_length <<endl;

    // debug_mpi(this->task_id, pIdx, "block start: " + to_string(binfo->t_start) + " end: " + to_string(binfo->t_end) + " comp size: " + to_string(binfo->t_end - binfo->t_start) + " block len: " + to_string(binfo->block_len) + " expand len: " + to_string(expand_length) + "X size: " + to_string(expandX.size()));

    vector_cipher_ge(this->pIdx, expandX, expandY, expandTable, *(this->eval),
                     *(this->enc), *(this->runtime));
    // debug_mpi(this->task_id, pIdx, "finish GE");

    // shift & substraction.
    if(binfo->t_start < (this->m - 1) * this->n){ // only when the table is not started in the last column, perform the shift & substraction. 
      for(int i=0; i<expand_length - this->n; i++){
        partTable.mShares[0](i) = expandTable.mShares[0](i) ^ expandTable.mShares[0](i + this->n);
        partTable.mShares[1](i) = expandTable.mShares[1](i) ^ expandTable.mShares[1](i + this->n);
      }
      for(int i=expand_length-this->n; i<binfo->block_len; i++){
        partTable.mShares[0](i) = expandTable.mShares[0](i);
        partTable.mShares[1](i) = expandTable.mShares[1](i);
      }
    }
    else{
      for(int i=0; i<binfo->block_len; i++){
        partTable.mShares[0](i) = expandTable.mShares[0](i);
        partTable.mShares[1](i) = expandTable.mShares[1](i);
      }
    }
    // pairwise vector abmul.
    aby3::si64Matrix expandV(block_length, 1);
    for (size_t i = 0; i < block_length; i++) expandV((int)i, (int)0, this->selectV[binfo->t_start + i]);

    cipher_mul_seq(this->pIdx, expandV, partTable, expandV, *(this->eval),
                   *(this->enc), *(this->runtime));
    
    for (size_t i = 0; i < block_length; i++) local_table[i] = expandV(i, 0);

    // debug_mpi(this->task_id, pIdx, "block start: " + to_string(binfo->t_start) + " end: " + to_string(binfo->t_end) + " comp size: " + to_string(binfo->t_end - binfo->t_start) + " block len: " + to_string(binfo->block_len));
    // debug_mpi(this->task_id, pIdx, "finish one round");
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class SecretIndex : public PTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  using PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::PTRTask;
  SecretIndex(int tasks, size_t optimal_block, const int pIdx,
              aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
              aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  SecretIndex(const int pIdx, aby3::Sh3Encryptor& enc,
              aby3::Sh3Runtime& runtime, aby3::Sh3Evaluator& eval,
              std::string deployment_profile_name = "./Config/profile.json")
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(deployment_profile_name) {}

  using PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::task_split;
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTask(new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id,
                                                  this->pIdx, this->enc,
                                                  this->runtime, this->eval));
    subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                               table_end);
    subTask->initial_value = this->default_value;
    for (int j = 0; j < this->n; j++) subTask->res[j] = this->default_value;
    this->subTasks.emplace_back(subTask);
  }

};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPISecretIndex : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPISecretIndex(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIRank : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPIRank(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPISearch : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPISearch(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPINewSearch : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPINewSearch(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {
          // cout << "in creation" << endl;
          this->lookahead = 1;
        }

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubAvg : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubAvg(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {

    for (int i = 0; i < resLeft.size(); i++){
      local_res[i] = resLeft[i] + resRight[i];
      local_res[i].mData[0] = (aby3::i64) (local_res[i].mData[0] * double (1 / this->m));
      // cout << sizeof(local_res[i]) << endl;
      local_res[i].mData[1] = (aby3::i64) (local_res[i].mData[1] * double (1 / this->m));
    }
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    local_table = expandY;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIAverage : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPIAverage(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubSum : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubSum(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {

    for (int i = 0; i < resLeft.size(); i++){
      local_res[i] = resLeft[i] + resRight[i];
    }
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    local_table = expandY;
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPISum : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPISum(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubMeanDis : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubMeanDis(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
    
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    // expand the selectV
    size_t k = expandX[0].size();

    // flat the two-dimensional inputs.
    size_t exp_len = expandX.size()*k;
    std::vector<typename NUMX::value_type> flatX, flatY;
    for (const auto& innerVec : expandX){
        for (const auto& element : innerVec) flatX.push_back(element);
    }
    for (const auto& innerVec : expandY){
        for (const auto& element : innerVec) flatY.push_back(element);
    }
    // vector mul
    if(std::is_same<typename NUMX::value_type, aby3::si64>::value){
      vector_mean_square(this->pIdx, flatX, flatY, flatX, *(this->eval), *(this->enc), *(this->runtime));
    }
    
    // reduce to local table
    for(int i=0; i<expandX.size(); i++){
        local_table[i] = this->initial_value;
        for (int j=0; j<k; j++) local_table[i] = local_table[i] + flatX[j];
    }
    return;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIMeanDis : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPIMeanDis(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubBioMetric : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubBioMetric(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
    aby3::sbMatrix comp_res;
    vector_cipher_gt(this->pIdx, resLeft, resRight, comp_res, *(this->eval), *(this->enc), *(this->runtime));
    // cout << "after new gt" << endl;
    // multiply for value extraction.
    aby3::si64Matrix matSub;
    matSub.resize(resLeft.size(), 1);
    for(int i=0; i<resLeft.size(); i++) matSub(i, 0, resRight[i] - resLeft[i]);
    cipher_mul_seq(this->pIdx, matSub, comp_res, matSub, *(this->eval), *(this->enc), *(this->runtime));
    // compute the final result
    for(int i=0; i<resLeft.size(); i++){
      local_res[i] = resRight[i] - matSub(i, 0);
    }
    // cout << "before return reduce? " << endl;
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    // expand the selectV
    size_t k = expandX[0].size();

    // flat the two-dimensional inputs.
    size_t exp_len = expandX.size()*k;
    std::vector<typename NUMX::value_type> flatX, flatY;
    for (const auto& innerVec : expandX){
        for (const auto& element : innerVec) flatX.push_back(element);
    }
    for (const auto& innerVec : expandY){
        for (const auto& element : innerVec) flatY.push_back(element);
    }
    // vector mul
    if(std::is_same<typename NUMX::value_type, aby3::si64>::value){
      vector_mean_square(this->pIdx, flatX, flatY, flatX, *(this->eval), *(this->enc), *(this->runtime));
    }
    
    // reduce to local table
    for(int i=0; i<expandX.size(); i++){
        local_table[i] = this->initial_value;
        for (int j=0; j<k; j++) local_table[i] = local_table[i] + flatX[j];
    }
    // cout << "can finished local table" << endl;
    return;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIBioMetric : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPIBioMetric(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubMetric : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubMetric(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
    aby3::sbMatrix comp_res;
    std::vector<aby3::si64> resLeft_value(resLeft.size());
    std::vector<aby3::si64> resRight_value(resRight.size());
    // initiate the value
    for(int i=0; i<resLeft.size(); i++){
      resLeft_value[i] = resLeft[i].value; resRight_value[i] = resRight[i].value;
    }
    vector_cipher_gt(this->pIdx, resLeft_value, resRight_value, comp_res, *(this->eval), *(this->enc), *(this->runtime));
    // cout << "after new gt" << endl;
    // multiply for value extraction.
    aby3::si64Matrix matSub;
    matSub.resize(resLeft.size(), 1);
    for(int i=0; i<resLeft.size(); i++) matSub(static_cast<aby3::u64>(i), static_cast<aby3::u64>(0), resRight[i].value - resLeft[i].value);
    cipher_mul_seq(this->pIdx, matSub, comp_res, matSub, *(this->eval), *(this->enc), *(this->runtime));

    aby3::si64Matrix matIndex;
    matIndex.resize(resLeft.size(), 1);
    for(int i=0; i<matIndex.size(); i++) matIndex(i, 0, resRight[i].index - resLeft[i].index);
    cipher_mul_seq(this->pIdx, matIndex, comp_res, matIndex, *(this->eval), *(this->enc), *(this->runtime));

    // compute the final result
    for(int i=0; i<resLeft.size(); i++){
      local_res[i].value = resRight[i].value - matSub(i, 0);
      local_res[i].index = resRight[i].index - matIndex(i, 0);
    }
    // cout << "before return reduce? " << endl;
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    // expand the selectV
    size_t k = expandX[0].size();

    // flat the two-dimensional inputs.
    size_t exp_len = expandX.size()*k;
    std::vector<typename NUMX::value_type> flatX, flatY;
    for (const auto& innerVec : expandX){
        for (const auto& element : innerVec) flatX.push_back(element);
    }
    for (const auto& innerVec : expandY){
        for (const auto& element : innerVec) flatY.push_back(element);
    }
    // vector mul
    if(std::is_same<typename NUMX::value_type, aby3::si64>::value){
      vector_mean_square(this->pIdx, flatX, flatY, flatX, *(this->eval), *(this->enc), *(this->runtime));
    }
    
    // enc index.
    aby3::i64Matrix indexMat(expandX.size(), 1);
    aby3::si64Matrix sindex(expandX.size(), 1);
    for(int i=0; i<expandX.size(); i++) indexMat(i, 0) = binfo->t_start + i;
    if(this->pIdx == 0){
      this->enc->localIntMatrix(*(this->runtime), indexMat, sindex).get();
    }
    else{
      this->enc->remoteIntMatrix(*(this->runtime), sindex).get();
    }

    // reduce to local table
    for(int i=0; i<expandX.size(); i++){
        local_table[i].value = this->initial_value.value;
        local_table[i].index = sindex(i, 0);
        for (int j=0; j<k; j++){
          local_table[i].value = local_table[i].value + flatX[j];
        }
    }
    // cout << "can finished local table" << endl;
    return;
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIMetric : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  MPIMetric(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubHistogram : public SubTask<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

  SubHistogram(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = true;
    this->lookahead=1;
    this->lookahead_axis=1;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {

    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {

    aby3::u64 block_length = binfo->block_len;
    aby3::u64 expand_length = binfo->block_len + this->lookahead;
    if(binfo->t_start + expand_length >= (this->n * this->m)) expand_length = (this->n * this->m) - binfo->t_start;

    aby3::sbMatrix partTable(binfo->block_len, 1); aby3::sbMatrix expandTable(expand_length, 1);

    vector_cipher_ge(this->pIdx, expandX, expandY, expandTable, *(this->eval),
                     *(this->enc), *(this->runtime));
    
    // shift & substraction.
    for(int i=0; i<block_length; i++){
      int index_ = binfo->t_start + i;
      if(index_ % this->n != (this->n - 1)){
        partTable.mShares[0](i) = expandTable.mShares[0](i) ^ expandTable.mShares[0](i + 1);
        partTable.mShares[1](i) = expandTable.mShares[1](i) ^ expandTable.mShares[1](i + 1);
      }
    }

    // trans to secret int.
    boolean_to_arith(pIdx, partTable, local_table, *(this->eval), *(this->enc), *(this->runtime));

    return;
  }
};

#endif

#ifdef OPENMP
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubIndex : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>::SubTask_OpenMP;

  SubIndex(const size_t optimal_block, const int task_id, const int pIdx,
           aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
           aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = true;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "resLeft: " << std::endl;
      ofs.close();
      debug_output_vector(resLeft, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "resRight: " << std::endl;
      ofs.close();
      debug_output_vector(resRight, *(this->runtime), *(this->enc));
    }
#endif
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "local_res: " << std::endl;
      ofs.close();
      debug_output_vector(local_res, *(this->runtime), *(this->enc));
    }
#endif
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    aby3::u64 block_length = binfo->block_len;
    // pairwise vector equal test.
    aby3::sbMatrix partTable;

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "expandX: " << binfo->t_start << std::endl;
      ofs.close();
      debug_output_vector(expandX, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "expandY: " << binfo->t_start << std::endl;
      for (int i = 0; i < expandY.size(); i++) ofs << expandY[i] << " ";
      ofs << std::endl;
      ofs.close();
      // debug_output_vector(expandY, *(this->runtime), *(this->enc));

      // aby3::si64Matrix expandXM(block_length);
      // for(int i=0; i<block_length; i++) expandXM(i, 0, expandX[i]);
    }
#endif

    // clock_t start, end;
    // start = clock();
    vector_cipher_eq(this->pIdx, expandX, expandY, partTable, *(this->eval),
                     *(this->runtime));
    // end = clock();
    // cout << "eq time: " << ((end-start)*1000) / CLOCKS_PER_SEC;

#ifdef DEBUG
    std::ofstream ofs(debugFile, std::ios_base::app);
    ofs << "partTable: " << binfo->t_start << std::endl;
    ofs.close();
    debug_output_matrix(partTable, *(this->runtime), *(this->enc), this->pIdx,
                        *(this->eval));
#endif

    // pairwise vector abmul.
    aby3::si64Matrix expandV;
    expandV.resize(block_length, 1);
    for (size_t i = 0; i < block_length; i++) {
      expandV(i, 0, this->selectV[binfo->t_start + i]);
    }
    cipher_mul_seq(this->pIdx, expandV, partTable, expandV, *(this->eval),
                   *(this->enc), *(this->runtime));

#ifdef DEBUG
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "table_start: " << binfo->t_start << std::endl;
      ofs.close();
      debug_output_matrix(expandV, *(this->runtime), *(this->enc));
    }
#endif

    // transfor to the local_table
    for (size_t i = 0; i < block_length; i++) local_table[i] = expandV(i, 0);
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubRank : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>::SubTask_OpenMP;

  SubRank(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {

    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    aby3::u64 block_length = binfo->block_len;
    vector_cipher_gt(this->pIdx, expandX, expandY, local_table, *(this->eval),
                     *(this->enc), *(this->runtime));
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubSearch : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>::SubTask_OpenMP;

  SubSearch(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = true;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "resLeft: " << std::endl;
      ofs.close();
      debug_output_vector(resLeft, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "resRight: " << std::endl;
      ofs.close();
      debug_output_vector(resRight, *(this->runtime), *(this->enc));
    }
#endif
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "local_res: " << std::endl;
      ofs.close();
      debug_output_vector(local_res, *(this->runtime), *(this->enc));
    }
#endif
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    aby3::u64 block_length = binfo->block_len;
    aby3::sbMatrix partTable;

#ifdef DEBUG
    // debug -> aby3 eq has sometimes errorness
    if (std::is_same<NUMR, aby3::si64>::value) {
      std::ofstream ofs(debugFile, std::ios_base::app);
      ofs << "expandX: " << binfo->t_start << std::endl;
      ofs.close();
      debug_output_vector(expandX, *(this->runtime), *(this->enc));

      ofs.open(debugFile, std::ios_base::app);
      ofs << "expandY: " << binfo->t_start << std::endl;
      for (int i = 0; i < expandY.size(); i++) ofs << expandY[i] << " ";
      ofs << std::endl;
      ofs.close();
      // debug_output_vector(expandY, *(this->runtime), *(this->enc));
      // aby3::si64Matrix expandXM(block_length);
      // for(int i=0; i<block_length; i++) expandXM(i, 0, expandX[i]);
    }
#endif

    clock_t start, end;
    vector_cipher_ge(this->pIdx, expandX, expandY, partTable, *(this->eval),
                     *(this->enc), *(this->runtime));
    
    // pairwise vector abmul.
    aby3::si64Matrix expandV;
    expandV.resize(block_length, 1);
    for (size_t i = 0; i < block_length; i++) expandV(i, 0, this->selectV[binfo->t_start + i]);

    cipher_mul_seq(this->pIdx, expandV, partTable, expandV, *(this->eval),
                   *(this->enc), *(this->runtime));
    
    for (size_t i = 0; i < block_length; i++) local_table[i] = expandV(i, 0);
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class SecretIndex : public PTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  using PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::PTRTask;
  SecretIndex(int tasks, size_t optimal_block, const int pIdx,
              aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
              aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  SecretIndex(const int pIdx, aby3::Sh3Encryptor& enc,
              aby3::Sh3Runtime& runtime, aby3::Sh3Evaluator& eval,
              std::string deployment_profile_name = "./Config/profile.json")
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(deployment_profile_name) {}

  using PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::task_split;
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTask(new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id,
                                                  this->pIdx, this->enc,
                                                  this->runtime, this->eval));
    subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                               table_end);
    subTask->initial_value = this->default_value;
    for (int j = 0; j < this->n; j++) subTask->res[j] = this->default_value;
    this->subTasks.emplace_back(subTask);
  }

  // void circuit_evaluate(NUMX* dataX, NUMY* dataY, NUMR* selectV, NUMR* res){

  //     // prepare data structures.
  //     this->inputX = fake_repeat(dataX, this->shapeX, this->m, 0);
  //     this->inputY = fake_repeat(dataY, this->shapeY, this->n, 1);
  //     this->res = res;

  //     // compute functions => currently, using sequential.
  //     for(int i=0; i<this->total_tasks; i++){
  //         // call the corresponding functions on different machines.
  //         this->subTasks[i]->circuit_evaluate(this->inputX, this->inputY,
  //         this->selectV);
  //     }

  //     #ifdef DEBUG
  //     // debug -> aby3 eq has sometimes errorness
  //     for(int i=0; i<this->total_tasks; i++){
  //       if(std::is_same<NUMR, aby3::si64>::value){
  //         std::ofstream ofs(debugFile, std::ios_base::app);
  //         ofs << "subTask-" << i << "res: " << std::endl;
  //         ofs.close();
  //         debug_output_vector(this->subTasks[i]->res, this->runtime,
  //         this->enc);

  //         // ofs.open(debugFile, std::ios_base::app);
  //         // ofs << "expandY: " << binfo->t_start << std::endl;
  //         // for(int i=0; i<expandY.size(); i++) ofs << expandY[i] << " ";
  //         // ofs << std::endl;
  //         // ofs.close();
  //         // debug_output_vector(expandY, *(this->runtime), *(this->enc));

  //         // aby3::si64Matrix expandXM(block_length);
  //         // for(int i=0; i<block_length; i++) expandXM(i, 0, expandX[i]);
  //       }
  //     }
  //     #endif

  //     // simulate
  //     for(int i=this->total_tasks-1; i>=0; i--){
  //         size_t left_tasks = this->total_tasks;
  //         size_t send_start = (left_tasks + 1) / 2;
  //         while(i < send_start && left_tasks > 1){
  //             size_t receive_target = i + send_start;
  //             if(receive_target < left_tasks){
  //                 std::vector<NUMR> receive_res =
  //                 this->subTasks[receive_target]->res;
  //                 this->subTasks[i]->partical_reduction(receive_res,
  //                 this->subTasks[i]->res, this->subTasks[i]->res, nullptr);
  //             }

  //             left_tasks = send_start;
  //             send_start = (left_tasks + 1) / 2;
  //         }
  //         if(i >= send_start) size_t end_target = i - send_start;
  //     }
  //     std::copy(this->subTasks[0]->res.begin(), this->subTasks[0]->res.end(),
  //     res);
  // }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPISecretIndex : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask_OpenMP;
  MPISecretIndex(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    // cout << "in this func" << endl;
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIRank : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask_OpenMP;
  MPIRank(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPISearch : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask_OpenMP;
  MPISearch(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubAvg : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>::SubTask_OpenMP;

  SubAvg(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {

    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    local_table = expandY;
  }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIAverage : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask_OpenMP;
  MPIAverage(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubMeanDis : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>::SubTask_OpenMP;

  SubMeanDis(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
    
    for (int i = 0; i < resLeft.size(); i++)
      local_res[i] = resLeft[i] + resRight[i];
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    // expand the selectV
    size_t k = expandX[0].size();

    // flat the two-dimensional inputs.
    size_t exp_len = expandX.size()*k;
    std::vector<typename NUMX::value_type> flatX, flatY;
    for (const auto& innerVec : expandX){
        for (const auto& element : innerVec) flatX.push_back(element);
    }
    for (const auto& innerVec : expandY){
        for (const auto& element : innerVec) flatY.push_back(element);
    }
    // vector mul
    if(std::is_same<typename NUMX::value_type, aby3::si64>::value){
      vector_mean_square(this->pIdx, flatX, flatY, flatX, *(this->eval), *(this->enc), *(this->runtime));
    }
    
    // reduce to local table
    for(int i=0; i<expandX.size(); i++){
        local_table[i] = this->initial_value;
        for (int j=0; j<k; j++) local_table[i] = local_table[i] + flatX[j];
    }
    return;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIMeanDis : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask_OpenMP;
  MPIMeanDis(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubBioMetric : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor* enc;
  aby3::Sh3Runtime* runtime;
  aby3::Sh3Evaluator* eval;

  using SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>::SubTask_OpenMP;

  SubBioMetric(const size_t optimal_block, const int task_id, const int pIdx,
          aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
          aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(&enc),
        runtime(&runtime),
        eval(&eval),
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
    this->have_selective = false;
  }

  virtual void partical_reduction(std::vector<NUMR>& resLeft,
                                  std::vector<NUMR>& resRight,
                                  std::vector<NUMR>& local_res,
                                  BlockInfo* binfo) override {
    aby3::sbMatrix comp_res;
    vector_cipher_gt(this->pIdx, resLeft, resRight, comp_res, *(this->eval), *(this->enc), *(this->runtime));
    // cout << "after new gt" << endl;
    // multiply for value extraction.
    aby3::si64Matrix matSub;
    matSub.resize(resLeft.size(), 1);
    for(int i=0; i<resLeft.size(); i++) matSub(i, 0, resRight[i] - resLeft[i]);
    cipher_mul_seq(this->pIdx, matSub, comp_res, matSub, *(this->eval), *(this->enc), *(this->runtime));
    // compute the final result
    for(int i=0; i<resLeft.size(); i++){
      local_res[i] = resRight[i] - matSub(i, 0);
    }
    // cout << "before return reduce? " << endl;
    return;
  }

 protected:
  virtual void compute_local_table(std::vector<NUMX>& expandX,
                                   std::vector<NUMY>& expandY,
                                   std::vector<NUMT>& local_table,
                                   BlockInfo* binfo) {
    // expand the selectV
    size_t k = expandX[0].size();

    // flat the two-dimensional inputs.
    size_t exp_len = expandX.size()*k;
    std::vector<typename NUMX::value_type> flatX, flatY;
    for (const auto& innerVec : expandX){
        for (const auto& element : innerVec) flatX.push_back(element);
    }
    for (const auto& innerVec : expandY){
        for (const auto& element : innerVec) flatY.push_back(element);
    }
    // vector mul
    if(std::is_same<typename NUMX::value_type, aby3::si64>::value){
      vector_mean_square(this->pIdx, flatX, flatY, flatX, *(this->eval), *(this->enc), *(this->runtime));
    }
    
    // reduce to local table
    for(int i=0; i<expandX.size(); i++){
        local_table[i] = this->initial_value;
        for (int j=0; j<k; j++) local_table[i] = local_table[i] + flatX[j];
    }
    // cout << "can finished local table" << endl;
    return;
  }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIBioMetric : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask_OpenMP;
  MPIBioMetric(int tasks, size_t optimal_block, const int pIdx,
                 aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                 aby3::Sh3Evaluator& eval)
      : pIdx(pIdx),
        enc(enc),
        runtime(runtime),
        eval(eval),
        MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block) {}

  // override sub_task create.
  void create_sub_task(size_t optimal_block, int task_id, size_t table_start,
                       size_t table_end) override {
    auto subTaskPtr(
        new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id, this->pIdx,
                                         this->enc, this->runtime, this->eval));
    this->subTask.reset(subTaskPtr);
    this->subTask->initial_value = this->default_value;
    this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start,
                                     table_end);
    for (int j = 0; j < this->n; j++)
      this->subTask->res[j] = this->default_value;
  }
};

#endif
int ptr_secret_index(int pIdx, std::vector<aby3::si64>& sharedM,
                     std::vector<aby3::si64>& secretIndex,
                     std::vector<aby3::si64>& res, aby3::Sh3Evaluator& eval,
                     aby3::Sh3Runtime& runtime, aby3::Sh3Encryptor& enc);

int mpi_ptr_secret_index(int pIdx, std::vector<aby3::si64>& sharedM,
                         std::vector<aby3::si64>& secretIndex,
                         std::vector<aby3::si64>& res, aby3::Sh3Evaluator& eval,
                         aby3::Sh3Runtime& runtime, aby3::Sh3Encryptor& enc,
                         int task_num, int opt_B);
  
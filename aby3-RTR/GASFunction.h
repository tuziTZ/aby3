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

template <aby3::Decimal D>
struct sfnode {
    int id; 
    aby3::sf64<D> pr;
    aby3::sf64<D> reci_out_deg;
};

template <aby3::Decimal D>
struct sfedge {
    int id;
    aby3::si64 start, end;
    aby3::sf64<D> data;
};

// functional class override.
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubPRScatterABY3 : public SubTask<NUMX, NUMY, NUMT, NUMR>{
    public:
        // aby3 info
        int pIdx;
        aby3::Sh3Encryptor* enc;
        aby3::Sh3Runtime* runtime;
        aby3::Sh3Evaluator* eval;

        using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

        SubPRScatterABY3(const size_t optimal_block, const int task_id, const int pIdx,
                aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                aby3::Sh3Evaluator& eval)
            : pIdx(pIdx),
                enc(&enc),
                runtime(&runtime),
                eval(&eval),
                SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
            this->have_selective = false;
        }

    protected:
        virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
            aby3::u64 block_length = binfo->block_len;
            // aby3::si64Matrix matrixX, matrixY;
            // matrixX.resize(block_length, 1); matrixY.resize(block_length, 1);
            std::vector<aby3::si64> matrixX;
            std::vector<int> matrixY;
            aby3::sbMatrix indicatorTable; indicatorTable.resize(block_length, 1);

            aby3::sf64Matrix<aby3::Decimal::D8> matrixPR, matrixDeg;
            matrixPR.resize(block_length, 1); matrixDeg.resize(block_length, 1);
            aby3::sf64Matrix<aby3::Decimal::D8> mulTable;

            // set vectorized computation data.
            for(int i=0; i < block_length; i++){
                // matrixX(i, 0, expandX[i].start); matrixY(i, 0, expandY[i].id);
                matrixX.push_back(expandX[i].start); matrixY.push_back(expandY[i].id);
                matrixPR(i, 0, expandY[i].pr); matrixDeg(i, 0, expandY[i].reci_out_deg);
            }

            // evaluate the vectorized computation.
            vector_cipher_eq(this->pIdx, matrixX, matrixY, indicatorTable, *(this->eval), *(this->runtime));

            // the mul in aby3 have some correctness problem.
            cipher_mul_seq(this->pIdx, matrixPR, matrixDeg, mulTable, *(this->eval), *(this->enc), *(this->runtime));

            // compute the local table.
            cipher_mul_seq(this->pIdx, mulTable, indicatorTable, mulTable, *(this->eval), *(this->enc), *(this->runtime));

            // trans to the final result.
            for(int i=0; i<block_length; i++) local_table[i] = mulTable(i, 0);

            return;
        }

    public:
        virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
            for(int i=0; i<resLeft.size(); i++) update_res[i] = resLeft[i] + resRight[i];
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubPRGatherABY3 : public SubTask<NUMX, NUMY, NUMT, NUMR>{
    public:
        // aby3 info
        int pIdx;
        aby3::Sh3Encryptor* enc;
        aby3::Sh3Runtime* runtime;
        aby3::Sh3Evaluator* eval;

        using SubTask<NUMX, NUMY, NUMT, NUMR>::SubTask;

        SubPRGatherABY3(const size_t optimal_block, const int task_id, const int pIdx,
                aby3::Sh3Encryptor& enc, aby3::Sh3Runtime& runtime,
                aby3::Sh3Evaluator& eval)
            : pIdx(pIdx),
                enc(&enc),
                runtime(&runtime),
                eval(&eval),
                SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id) {
            this->have_selective = false;
        }

    protected:
        virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
            aby3::u64 block_length = binfo->block_len;
            aby3::sbMatrix indicatorTable; indicatorTable.resize(block_length, 1);
            std::vector<aby3::si64> vec_y; std::vector<int> vec_x;
            aby3::sf64Matrix<aby3::Decimal::D8> mulTable; mulTable.resize(block_length, 1);
            std::vector<double> const_val_plain(block_length, 0.85);
            aby3::sf64Matrix<aby3::Decimal::D8> const_val; 
            vector_to_matrix(this->pIdx, *(this->enc), *(this->runtime), const_val_plain, const_val);

            // data preparation.
            for(int i=0; i<block_length; i++){
                vec_x.push_back(expandX[i].id); vec_y.push_back(expandY[i].end);
                mulTable(i, 0, expandY[i].data);
            }
            // vectorized computation
            vector_cipher_eq(this->pIdx, vec_y, vec_x, indicatorTable, *(this->eval), *(this->runtime));
            cipher_mul_seq(this->pIdx, mulTable, indicatorTable, mulTable, *(this->eval), *(this->enc), *(this->runtime));
            cipher_mul_seq(this->pIdx, mulTable, const_val, mulTable, *(this->eval), *(this->enc), *(this->runtime));

            for(int i=0; i<block_length; i++){
                local_table[i] = mulTable(i, 0);
            }
            return;
        }

    public:
        virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
            for(int i=0; i<resLeft.size(); i++) update_res[i] = (resLeft[i] + resRight[i]);
        return;
    }
};

// mpi function definition.
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class ABY3MPITask : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // aby3 info
  int pIdx;
  aby3::Sh3Encryptor& enc;
  aby3::Sh3Runtime& runtime;
  aby3::Sh3Evaluator& eval;

  // setup all the aby3 environment variables, pIdx and rank.
  using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::MPIPTRTask;
  ABY3MPITask(int tasks, size_t optimal_block, const int pIdx,
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
class MPIPRScatter : public ABY3MPITask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // setup all the aby3 environment variables, pIdx and rank.
  using ABY3MPITask<NUMX, NUMY, NUMT, NUMR, TASK>::ABY3MPITask;
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR,
          template <typename, typename, typename, typename> class TASK>
class MPIPRGather : public ABY3MPITask<NUMX, NUMY, NUMT, NUMR, TASK> {
 public:
  // setup all the aby3 environment variables, pIdx and rank.
  using ABY3MPITask<NUMX, NUMY, NUMT, NUMR, TASK>::ABY3MPITask;
};

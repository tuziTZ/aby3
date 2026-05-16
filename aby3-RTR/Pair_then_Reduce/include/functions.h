#ifndef FUNCTIONS_H  
#define FUNCTIONS_H 

#include "datatype.h"
#include "gas_functions.h"

#define OPTIMAL_BLOCK 10
#define TASKS 10
// #define OPENMP
// #define DEBUG


#ifndef OPENMP
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubSecretIndex : public SubTask<NUMX, NUMY, NUMT, NUMR>{

public:
    SubSecretIndex(const size_t optimal_block, const int task_id):  
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = true;
        }
    // SubSecretIndex(const size_t optimal_block, const int task_id):  
    //     SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
    //         this->have_selective = true;
    //     }
#endif

#ifdef OPENMP
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubSecretIndex : public SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>{

public:
    SubSecretIndex(const size_t optimal_block, const int task_id):  
        SubTask_OpenMP<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = true;
        }
#endif

protected:
    virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
        // expand the selectV
        std::vector<NUMR> expandV(binfo->block_len);
        for(int i=0; i<binfo->block_len; i++){
            expandV[i] = this->selectV[i+binfo->t_start];
        }

        for(int i=0; i< binfo->block_len; i++){
            local_table[i] = (expandX[i] == expandY[i]);
            local_table[i] *= expandV[i];
        }

        return;
    }

public:

    virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
        for(int i=0; i<resLeft.size(); i++){
            update_res[i] = resLeft[i] + resRight[i];
        }
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubSearch : public SubTask<NUMX, NUMY, NUMT, NUMR>{

public:
    SubSearch(const size_t optimal_block, const int task_id):
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = true;
        }

protected:
    virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
        std::vector<NUMR> expandV(binfo->block_len);
        for(int i=0; i<binfo->block_len; i++){
            expandV[i] = this->selectV[i+binfo->t_start];
        }
        for(int i=0; i<binfo->block_len; i++){
            local_table[i] = (expandX[i] >= expandY[i]) * expandV[i];
        }
        return;
    }

public:
    virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
        for(int i=0; i<resLeft.size(); i++) update_res[i] = resLeft[i] + resRight[i];
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class NewSubSearch : public SubTask<NUMX, NUMY, NUMT, NUMR>{

public:
    NewSubSearch(const size_t optimal_block, const int task_id):
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = true;
        }

protected:
    virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
        
        std::vector<NUMR> expandV(binfo->block_len);
        for(int i=0; i<binfo->block_len; i++) expandV[i] = this->selectV[i+binfo->t_start];

        size_t valid_length = binfo->block_len + this->lookahead * this->n;
        if(binfo->t_start + valid_length >= (this->n * this->m)) valid_length = (this->n * this->m) - binfo->t_start;

        // firstly compute the expanded table
        NUMT* expandTable = new NUMT[valid_length];
        for(int i=0; i<valid_length; i++) expandTable[i] = (expandX[i] >= expandY[i]);
        #ifdef DEBUG
        cout << "start point: " << binfo->t_start << " | end point: " << binfo->t_end << endl;
        // debug
        cout << "valid_length: " << valid_length << endl;
        // for(int i=0; i<valid_length; i++) cout << expandTable[i] << " ";
        // cout << endl;
        #endif

        // shift & substractive to construct the boolean local_table.
        if(binfo->t_start < (this->m - 1) * this->n){ // only when the table is not started in the last column, perform the shift & substraction. 
            for(int i=0; i<valid_length - this->n; i++) local_table[i] = expandTable[i] - expandTable[i+this->n];
            for(int i=valid_length-this->n; i<binfo->block_len; i++) local_table[i] = expandTable[i];
        }
        #ifdef DEBUG
        cout << "block len: " << binfo->block_len << endl;
        // for(int i=0; i<binfo->block_len; i++) cout << local_table[i] << " ";
        // cout << endl;
        #endif

        // finally multiply with the selective value.
        for(int i=0; i< binfo->block_len; i++) local_table[i] = local_table[i] * expandV[i];
        #ifdef DEBUG
        cout << "selection" << endl;
        for(int i=0; i<binfo->block_len; i++) cout << local_table[i] << " ";
        cout << endl;
        #endif
        return;
    }

public:
    virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
        for(int i=0; i<resLeft.size(); i++) update_res[i] = resLeft[i] + resRight[i];
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubHistogram : public SubTask<NUMX, NUMY, NUMT, NUMR>{
public:
    SubHistogram(const size_t optimal_block, const int task_id):
        SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = false;
        }
 
    virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
        
        size_t valid_length = binfo->block_len + this->lookahead;
        if(binfo->t_start + valid_length >= (this->n * this->m)) valid_length = (this->n * this->m) - binfo->t_start;

        // firstly compute the expanded table
        NUMT* expandTable = new NUMT[valid_length];
        for(int i=0; i<valid_length; i++) expandTable[i] = (expandX[i] <= expandY[i]);
        #ifdef DEBUG
        cout << "start point: " << binfo->t_start << " | end point: " << binfo->t_end << endl;
        // debug
        cout << "valid_length: " << valid_length << endl;
        #endif

        // shift & substractive to construct the uniry local_table.
        for(int i=0; i<binfo->block_len; i++){
            int index_ = i + binfo->t_start;
            if(index_ % this->n != (this->n - 1)) local_table[i] = expandTable[i] - expandTable[i+1];
        }
        return;
    }

    virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
        for(int i=0; i<resLeft.size(); i++) update_res[i] = resLeft[i] + resRight[i];
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class Search : public PTRTask<NUMX, NUMY, NUMT, NUMR, SubSearch>{

public:
    using PTRTask<NUMX, NUMY, NUMT, NUMR, SubSearch>::PTRTask;
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SecretIndex : public PTRTask<NUMX, NUMY, NUMT, NUMR, SubSecretIndex>{

public:
    using PTRTask<NUMX, NUMY, NUMT, NUMR, SubSecretIndex>::PTRTask;
};


#ifndef OPENMP
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class MPISecretIndex : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubSecretIndex>{

public:
    using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubSecretIndex>::MPIPTRTask;
};
#endif
#ifdef OPENMP
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class MPISecretIndex : public MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, SubSecretIndex>{

public:
    using MPIPTRTask_OpenMP<NUMX, NUMY, NUMT, NUMR, SubSecretIndex>::MPIPTRTask_OpenMP;
};
#endif

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class NewMPISearch : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, NewSubSearch>{

public:
    using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, NewSubSearch>::MPIPTRTask;
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class NewMPIHistogram : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubHistogram>{

public:
    using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubHistogram>::MPIPTRTask;
};



template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubPRScatter : public SubTask<NUMX, NUMY, NUMT, NUMR>{
    public:
        SubPRScatter(const size_t optimal_block, const int task_id):
            SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = false;
        }
    protected:
        virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
            for(int i=0; i < binfo->block_len; i++){
                local_table[i] = 0.85 * (expandX[i].start == expandY[i].id) * (expandY[i].pr / (expandY[i].out_deg + (double) (expandY[i].out_deg == 0)));
            }
            return;
        }

    public:
        virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
            for(int i=0; i<resLeft.size(); i++) update_res[i] = resLeft[i] + resRight[i];
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubPRGather : public SubTask<NUMX, NUMY, NUMT, NUMR>{
    public:
        SubPRGather(const size_t optimal_block, const int task_id):
            SubTask<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id){
            this->have_selective = false;
        }
    protected:
        virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) override {
            for(int i=0; i<binfo->block_len; i++){
                local_table[i] = (expandX[i].id == expandY[i].end) * (expandY[i].data);
            }
            return;
        }

    public:
        virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) override {
            for(int i=0; i<resLeft.size(); i++) update_res[i] = (resLeft[i] + resRight[i]);
        return;
    }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class PRScatter : public PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>{

public:
    PRScatter(int tasks, size_t optimal_block):
        PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>(tasks, optimal_block){
            this->default_value = 0.0;
            this->have_selective = false;
    }
};

template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class PRGather : public PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>{

public:
    // using PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>::PTRTask;
    PRGather(int tasks, size_t optimal_block):
        PTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>(tasks, optimal_block){
        this->default_value = 0.0;
        this->have_selective = false;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class MPIPRGather : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>{

public:
    // using MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>::MPIPTRTask;
    MPIPRGather(int tasks, size_t optimal_block):
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRGather>(tasks, optimal_block){
            this->default_value = 0.0;
            this->have_selective = false;
        }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class MPIPRScatter : public MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>{

public:
    MPIPRScatter(int tasks, size_t optimal_block):
        MPIPTRTask<NUMX, NUMY, NUMT, NUMR, SubPRScatter>(tasks, optimal_block){
            this->default_value = 0.0;
            this->have_selective = false;
        }
};




void oblivious_index(std::vector<float>& inputX, std::vector<int>& indices, std::vector<float>& res);
void oblivious_search(std::vector<float>& inputX, std::vector<float>& bins, std::vector<float>& tarVals, std::vector<float>& res);


void test_oblivious_index(int n, int m);
void test_oblivious_search(int n, int m);
void test_multi_data_sharing(int n, int m);
void test_fake_repeat();


#endif

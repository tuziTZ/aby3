#ifndef TASKS_H
#define TASKS_H

#include "datatype.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <json/json.h>
#include <cmath>
#include <mpi.h>
#include <iomanip>
#include <chrono>

/**
 * @brief 
 * 
 * @tparam NUMX 
 * @tparam NUMY 
 * @tparam NUMT 
 * @tparam NUMR 
 */
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubTask{

public:
    // functional logic parameters.
    const int task_id;
    const size_t optimal_block;
    bool have_selective=true;
    size_t lookahead = 0;
    int lookahead_axis = 0;

    // task specific parameters.
    size_t n, m;
    size_t table_start, table_end;
    size_t block_num;
    NUMR initial_value;
    std::vector<NUMR> res;
    size_t m_start = 0, m_end = 0;

    // timer_list;
    std::vector<double> round_time_list;
    // double preprocessing_time = 0;
    // std::vector<double> preprocessing_time_list;

public:

    void set_lookahead(size_t lookahead, int axis){
        this->lookahead = lookahead;
        this->lookahead_axis = axis;
        return;
    }

    virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) = 0;

    virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) = 0;

    SubTask(const size_t optimal_block, const int task_id):
        task_id(task_id),
        optimal_block(optimal_block){}


    void circuit_construct(const std::vector<size_t> shapeX, const std::vector<size_t> shapeY, const size_t table_start, size_t table_end){
        this->n = shapeX[0], this->m = shapeY[0];
        size_t table_size = table_end - table_start;
        this->table_start = table_start, this->table_end = table_end;
        // how many blocks we need to compute, each block is in optimal block size.
        block_num = (table_size + this->optimal_block - 1) / this->optimal_block;
        for(int i=0; i<block_num; i++){
            size_t local_table_start = table_start + i*optimal_block;
            size_t local_table_end = (i == block_num - 1) ? table_end : local_table_start + optimal_block;
            BlockInfo* blockinfo = new BlockInfo(local_table_start, local_table_end);
            block_info_list.emplace_back(blockinfo);
        }
        this->res.resize(this->n, this->initial_value);
    }


    virtual void circuit_profile(const FakeArray<NUMX>& dataX, const FakeArray<NUMY>& dataY, const FakeArray<NUMR>& selectV){
        // it will generate data at first and then profile?
        if(this->have_selective) this->selectV = selectV;

        // using vectorization to construct the pairwise vector expandX and expandY.
        auto binfo = block_info_list[0];
        std::vector<NUMX> expandX(binfo->block_len + this->lookahead * this->n);
        std::vector<NUMY> expandY(binfo->block_len + this->lookahead * this->n);
        std::vector<NUMR> local_table(binfo->block_len);

        for(int p=0; p<(binfo->block_len + this->lookahead * this->n); p++){
            if(p + binfo->t_start < n*m){
                expandX[p] = dataX[p+binfo->t_start];
                expandY[p] = dataY[p+binfo->t_start];
            }
        }
        // cout << "before compute local table, rank = " << this->task_id  << endl;
        compute_local_table(expandX, expandY, local_table, binfo);
        // cout << "after compute local table, rank = " << this->task_id  << endl;

        // log l rounds partical reduction
        size_t padding_lens = binfo->block_len % this->n != 0 ? this->n - (binfo->block_len % this->n) : 0;
        std::vector<NUMR> current_table = local_table;
        current_table.insert(current_table.end(), padding_lens, NUMR(this->initial_value));

        size_t current_len = binfo->block_len + padding_lens;
        size_t current_rows = current_len / this->n;
        int res_flag = 1;
        int start_point = this->which_column(binfo->t_start);

        while(current_rows + res_flag > 1){

            std::vector<NUMR> resLeft;
            std::vector<NUMR> resRight;
            size_t mid_point;
            if(current_rows % 2 == 0){
                mid_point = current_len / 2;
            }
            else{
                if(res_flag == 1) std::rotate(this->res.begin(), this->res.begin() + start_point, this->res.end());  
                current_table.insert(current_table.end(), this->res.begin(), this->res.end());
                mid_point = (current_len + this->n) / 2;
                if(res_flag == 1){
                    this->res = std::vector<NUMR>(this->n, NUMR(this->initial_value));
                    res_flag = 0;
                }
            }
            resLeft = std::vector<NUMR>(current_table.begin(), current_table.begin()+mid_point);
            resRight = std::vector<NUMR>(current_table.begin() + mid_point, current_table.end());

            current_table.resize(mid_point);
            partical_reduction(resLeft, resRight, current_table, binfo);

            current_len = mid_point;
            current_rows = current_len / this->n;
        }

        if(start_point == 0) this->res = current_table;
        else{
            this->res = std::vector<NUMR>(current_table.end() - start_point, current_table.end());
            this->res.insert(this->res.end(), current_table.begin(), current_table.begin() + this->n - start_point);
        }
        return;
    }


    virtual void circuit_evaluate(const FakeArray<NUMX>& dataX, const FakeArray<NUMY>& dataY, const FakeArray<NUMR>& selectV){
        if(this->have_selective) this->selectV = selectV;

        // using vectorization to construct the pairwise vector expandX and expandY.
        for(int i=0; i<block_num; i++){

            auto start = std::chrono::high_resolution_clock::now(); 

            auto binfo = block_info_list[i];
            size_t slice_length = (this->lookahead_axis == 0) ? (binfo->block_len + this->lookahead * this->n) : (binfo->block_len + this->lookahead);

            // set the expand length.
            std::vector<NUMX> expandX(slice_length); std::vector<NUMY> expandY(slice_length);
            std::vector<NUMR> local_table(binfo->block_len);

            // note that even with non-zero lookahead, the size of the local_table is still block_size.
            for(int p=0; p<slice_length; p++){
                if(p + binfo->t_start < n*m){
                    expandX[p] = dataX[p+binfo->t_start];
                    expandY[p] = dataY[p+binfo->t_start];
                }
            }

            auto mid = std::chrono::high_resolution_clock::now();

            compute_local_table(expandX, expandY, local_table, binfo);

            auto start2 = std::chrono::high_resolution_clock::now();

            // log l rounds partical reduction
            size_t padding_lens = binfo->block_len % this->n != 0 ? this->n - (binfo->block_len % this->n) : 0;
            std::vector<NUMR> current_table = local_table;
            current_table.insert(current_table.end(), padding_lens, NUMR(this->initial_value));

            size_t current_len = binfo->block_len + padding_lens;
            size_t current_rows = current_len / this->n;
            int res_flag = 1;
            int start_point = this->which_column(binfo->t_start);

            auto mid2 = std::chrono::high_resolution_clock::now();

            other_time += (std::chrono::duration_cast<std::chrono::microseconds>(mid2 - start2) + std::chrono::duration_cast<std::chrono::microseconds>(mid - start));

            while(current_rows + res_flag > 1){

                auto start3 = std::chrono::high_resolution_clock::now();

                std::vector<NUMR> resLeft;
                std::vector<NUMR> resRight;
                size_t mid_point;
                if(current_rows % 2 == 0){
                    mid_point = current_len / 2;
                }
                else{
                    if(res_flag == 1) std::rotate(this->res.begin(), this->res.begin() + start_point, this->res.end());  
                    current_table.insert(current_table.end(), this->res.begin(), this->res.end());
                    mid_point = (current_len + this->n) / 2;
                    if(res_flag == 1){
                        this->res = std::vector<NUMR>(this->n, NUMR(this->initial_value));
                        res_flag = 0;
                    }
                }
                resLeft = std::vector<NUMR>(current_table.begin(), current_table.begin()+mid_point);
                resRight = std::vector<NUMR>(current_table.begin() + mid_point, current_table.end());

                current_table.resize(mid_point);

                auto mid3 = std::chrono::high_resolution_clock::now();

                partical_reduction(resLeft, resRight, current_table, binfo);

                current_len = mid_point;
                current_rows = current_len / this->n;

                other_time += (std::chrono::duration_cast<std::chrono::microseconds>(mid3 - start3));
            }

            if(start_point == 0) this->res = current_table;
            else{
                this->res = std::vector<NUMR>(current_table.end() - start_point, current_table.end());
                this->res.insert(this->res.end(), current_table.begin(), current_table.begin() + this->n - start_point);
            }


            auto all_end = std::chrono::high_resolution_clock::now();

            double round_time = std::chrono::duration_cast<std::chrono::milliseconds>(all_end - start).count();
            double pair_time = std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start).count();
            double agg_time = std::chrono::duration_cast<std::chrono::milliseconds>(all_end - start2).count();
            double preprocessing_time = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count();

            round_time_list.emplace_back(round_time);
            round_time_list.emplace_back(pair_time);
            round_time_list.emplace_back(agg_time);
            round_time_list.emplace_back(preprocessing_time);

            // MPI_Barrier(MPI_COMM_WORLD);
        }
        return;
    }


    FakeArray<NUMR> selectV;

    std::vector<BlockInfo*> block_info_list;

    size_t which_column(size_t table_loc){
        return table_loc % this->n;
    }

    size_t which_row(size_t table_loc){
        return table_loc % this->m;
    }
    
    size_t get_partial_m_lens(){
        this->m_start = this->table_start / this->n;
        this->m_end = (this->table_end - 1) / this->n;
        size_t partial_len = this->m_end - this->m_start + 1;
        if(this->m_end >= this->m - 1){
            return partial_len;
        }
        else{
            partial_len += this->lookahead;
        }
        return partial_len;
    }

    void print_timers(std::string logging_file){
        std::ofstream ofs(logging_file, ios::app);
        ofs << "time per round: " << std::endl;
        for(size_t i=0; i<block_num; i++){
            ofs << "round " << i << ": " << std::endl;
            ofs << "round-time: " << round_time_list[i*3] << std::endl;
            ofs << "pair-time: " << round_time_list[i*3+1] << std::endl;
            ofs << "agg-time: " << round_time_list[i*3+2] << std::endl;
            ofs << "preprocessing-time: " << round_time_list[i*3+3] << std::endl;
        }
        ofs.close();
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR, template<typename, typename, typename, typename> class TASK>
class PTRTask{

public:
    
    // logical parameters.
    int total_tasks;
    size_t optimal_block;
    std::vector<std::shared_ptr<TASK<NUMX, NUMY, NUMT, NUMR>>> subTasks;
    size_t lookahead = 0;
    int lookahead_axis = 0;

    // task specific parameters.
    std::vector<size_t> shapeX, shapeY;
    size_t n, m;
    NUMR* res;
    NUMR default_value;

    // data parameters.
    FakeArray<NUMX> inputX;
    FakeArray<NUMY> inputY;

    // some optional arguments
    FakeArray<NUMR> selectV;
    bool have_selective;

    PTRTask(int tasks, size_t optimal_block):
        total_tasks(tasks), optimal_block(optimal_block){}

    PTRTask(std::string deployment_profile_name="./Config/profile.json"){
        load_profile(deployment_profile_name);
        return;
    }

    void circuit_construct(const std::vector<size_t>& shapeX, const std::vector<size_t>& shapeY){
        this->shapeX = shapeX, this->shapeY = shapeY;
        this->n = shapeX[0], this->m = shapeY[0];
        size_t table_size = n*m;

        size_t task_length = table_size / total_tasks;
        task_split(table_size, task_length);
        return;
    }


    virtual void task_split(size_t table_size, size_t task_length){
        size_t left_table_size = table_size - task_length*total_tasks;
        for(int i=0; i<total_tasks; i++){
            size_t table_start, table_end;
            if (i < left_table_size){
                table_start = i*(task_length+1);
                table_end = (i+1)*(task_length+1);
            }
            else {
                table_start = i*task_length + left_table_size;
                table_end = (i+1)*task_length + left_table_size;
            }
            // let each process create this subTask object by their own.
            create_sub_task(optimal_block, i, table_start, table_end);
        }
    }


    virtual void create_sub_task(size_t optimal_block, int task_id, size_t table_start, size_t table_end){
        auto subTask(new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id));
        subTask->initial_value = this->default_value;
        subTask->circuit_construct(shapeX, shapeY, table_start, table_end);
        subTasks.emplace_back(subTask);
        return;
    }


    virtual void circuit_evaluate(NUMX* dataX, NUMY* dataY, NUMR* selectV, NUMR* res){
        
        // prepare data structures.
        this->inputX = fake_repeat(dataX, shapeX, this->m, 0);
        // this->inputY = fake_repeat(dataY, shapeY, this->n, 1);
        this->res = res;

        // compute functions => currently, using sequential.
        for(int i=0; i<total_tasks; i++){
            // size_t m_start, m_end = subTasks[i]->table_start / this->n, ((subTasks[i])->table_end / this->n)+1;
            size_t m_start = subTasks[i]->table_start / this->n;
            size_t m_end = (subTasks[i]->table_end / this->n)+ 1;
            NUMY partialY_data[m_end - m_start + 1];
            std::memcpy(partialY_data, dataY + m_start, sizeof(NUMY)*(m_end - m_start+1));
            FakeArray<NUMY> partialY = fake_repeat(partialY_data, shapeY, this->n, 1, m_start, m_end - m_start + 1);

            NUMR partialV_data[m_end - m_start + 1];
            FakeArray<NUMR> partialV;
            
            if(this->have_selective){
                std::memcpy(partialV_data, this->selectV.data + m_start, sizeof(NUMR)*(m_end - m_start+1));
                partialV = fake_repeat(partialV_data, this->selectV.dim, this->selectV.fake_dim[this->selectV.fake_axis], this->selectV.fake_axis, m_start, m_end - m_start + 1);
            }

            // call the corresponding functions on different machines.
            subTasks[i]->circuit_evaluate(this->inputX, partialY, partialV);
            // subTasks[i]->circuit_evaluate(this->inputX, this->inputY, this->selectV);
        }

        // simulate
        for(int i=total_tasks-1; i>=0; i--){
            size_t left_tasks = this->total_tasks;
            size_t send_start = (left_tasks + 1) / 2;
            while(i < send_start && left_tasks > 1){
                size_t receive_target = i + send_start;
                if(receive_target < left_tasks){
                    std::vector<NUMR> receive_res = subTasks[receive_target]->res;
                    subTasks[i]->partical_reduction(receive_res, subTasks[i]->res, subTasks[i]->res, nullptr);
                }

                left_tasks = send_start;
                send_start = (left_tasks + 1) / 2;
            }
            if(i >= send_start) size_t end_target = i - send_start;
        }
        std::copy(subTasks[0]->res.begin(), subTasks[0]->res.end(), res);
    }


    void circuit_evaluate(NUMX* dataX, NUMY* dataY, NUMR* res){
        return this->circuit_evaluate(dataX, dataY, nullptr, res);
    }


    void set_default_value(NUMR default_value){
        this->default_value = default_value;
        return;
    }


    virtual void set_selective_value(NUMR* selectV, int axis = 0){
        this->have_selective = true;
        if(axis == 0) this->selectV = fake_repeat(selectV, shapeY, this->n, 1);
        else this->selectV = fake_repeat(selectV, shapeX, this->m, 0);
        return;
    }


    void set_lookahead(size_t lookahead, int axis = 0){
        this->lookahead = lookahead;
        this->lookahead_axis = axis;
    }
// protected:
//     FakeArray<NUMR> selectV;
//     bool have_selective;

private:
    void load_profile(std::string deployment_profile_name){
        std::ifstream ifs(deployment_profile_name);
        Json::Reader reader;
        Json::Value obj;
        reader.parse(ifs, obj);

        this->total_tasks = obj["tasks"].asInt();
        this->optimal_block = obj["optimal_block"].asUInt64();
        return;
    }
};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR, template<typename, typename, typename, typename> class TASK>
class MPIPTRTask : public PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>{

public:
    int rank;
    std::shared_ptr<TASK<NUMX, NUMY, NUMT, NUMR>> subTask;
    size_t m_start, m_end;
    double time_subTask, time_combine, time_barrier;
    double mean_subTask, var_subTask, mean_combine, var_combine, max_subTask, min_subTask, total_idling;;
    std::vector<size_t> partial_m_lens;
    std::vector<size_t> partial_m_offsets;
    // size_t lookahead = 0;

    using PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::PTRTask;
    MPIPTRTask(int tasks, size_t optimal_block):
        PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block){
            int task_size;
            MPI_Comm_size(MPI_COMM_WORLD, &task_size);
            if(task_size != tasks){
                throw std::runtime_error("MPI task number: " + std::to_string(task_size) + " != config task size: " + std::to_string(tasks));
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &(this->rank));
            // cout << "my rank is: " << this->rank << endl;
            m_start = 0;
            m_end = 0;
        }
    

    void task_split(size_t table_size, size_t task_length) override {

        size_t left_table_size = table_size - task_length * this->total_tasks;

        // each task compute its table_start and table_end;
        size_t table_start, table_end;
        if(this->rank < left_table_size){
            table_start = this->rank * (task_length + 1);
            table_end = (this->rank + 1) * (task_length + 1);
        }
        else{
            table_start = this->rank * task_length + left_table_size;
            table_end = (this->rank+1) * task_length + left_table_size;
        }

        this->m_start = table_start / this->n;
        this->m_end = (table_end - 1) / this->n;

        if(this->rank == 0){
            this->partial_m_lens.resize(this->total_tasks);
            this->partial_m_offsets.resize(this->total_tasks);

            int tmp_table_start, tmp_table_end, tmp_m_start, tmp_m_end;
            for (int i = 0; i < this->total_tasks; i++) {
                if (i < left_table_size) {
                    tmp_table_start = i * (task_length + 1);
                    tmp_table_end = (i + 1) * (task_length + 1);
                }
                else {
                    tmp_table_start = i * task_length + left_table_size;
                    tmp_table_end = (i + 1) * task_length + left_table_size;
                }

                tmp_m_start = tmp_table_start / this->n;
                tmp_m_end = (tmp_table_end - 1) / this->n;

                // consider this->lookahead;
                if (this->lookahead > 0 && this->lookahead_axis == 0)
                    tmp_m_end = min(tmp_m_end + this->lookahead, this->m);

                this->partial_m_lens[i] = (tmp_m_end - tmp_m_start + 1);
                this->partial_m_offsets[i] = tmp_m_start;
            }
        }

        // then each task locally construct the sub-circuit.
        create_sub_task(this->optimal_block, this->rank, table_start, table_end);
    }
    

    void create_sub_task(size_t optimal_block, int task_id, size_t table_start, size_t table_end) override {
        if(optimal_block < this->n)
            std::cerr << "Warnning: Error possible when n is less than the block size." << std::endl;
        auto subTaskPtr(new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id));
        this->subTask.reset(subTaskPtr);
        this->subTask->initial_value = this->default_value;
        // this->subTask->lookahead = this->lookahead;
        this->subTask->set_lookahead(this->lookahead, this->lookahead_axis);
        this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start, table_end);
        return;
    }

    // for data sharing
    template<typename D>
    void data_sharing(D* &total_data, size_t length, int axis){

        if(axis == 0){ // if axis == 0, means broadcast.
            if(length != this->n){ // if axis == 0, the length must be n.
                throw std::runtime_error("If axis is 0, the data length should be the same as n");
            }
            if(this->rank != 0){
                // save the data.
                std::vector<char> received_data_bytes(length * sizeof(D));
                MPI_Bcast(received_data_bytes.data(), received_data_bytes.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
                // interprate to the D type array.
                std::memcpy(total_data, received_data_bytes.data(), received_data_bytes.size());
            }
            else{
                MPI_Bcast(total_data, length*sizeof(D), MPI_BYTE, 0, MPI_COMM_WORLD);
            }
            return;
        }
        else if(axis == 1){ // if axis == 1, means scatter.

            size_t reci_len = this->m_end - this->m_start + this->lookahead + 1;
            std::vector<char> received_data_bytes(reci_len * sizeof(D));
            // cout << "in this function " << this->rank << endl;
            if(this->rank != 0){
                if(length != reci_len) throw std::runtime_error("If axis = 1, the length must be set to the receive length, while current length = " + to_string(length) + " reci length = " + to_string(reci_len));
                std::vector<int> sendcounts(this->total_tasks, 0);
                std::vector<int> displs(this->total_tasks, 0);
                MPI_Scatterv(nullptr, sendcounts.data(), displs.data(), MPI_BYTE, received_data_bytes.data(), received_data_bytes.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
                // transfer to template datatype.
                std::memcpy(total_data, received_data_bytes.data(), received_data_bytes.size());
            }
            else{
                if(length != this->m) throw std::runtime_error("If axis = 1, the length of rank 0 must be equal to m, while current length = " + to_string(length) + " m = " + to_string(this->m));

                std::vector<int> sendcounts(this->total_tasks);
                std::vector<int> displs(this->total_tasks);
                for(int j=0; j<this->total_tasks; j++){
                    sendcounts[j] = this->partial_m_lens[j] * sizeof(D);
                    displs[j] = this->partial_m_offsets[j] * sizeof(D);
                }

                MPI_Scatterv(total_data, sendcounts.data(), displs.data(), MPI_BYTE, received_data_bytes.data(), received_data_bytes.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
                D* partial_data = new D[reci_len];
                std::memcpy(partial_data, received_data_bytes.data(), received_data_bytes.size());
                total_data = partial_data;
            }
            return;
        }
        else{
            throw std::runtime_error("The axis must be 0 or 1.");
        }
    }


    void set_selective_value(NUMR* selectV, int axis = 0){
        this->have_selective = true;
        if(axis == 0){
            this->selectV = fake_repeat(selectV, this->shapeY, this->n, 1, this->m_start, this->m_end - this->m_start + 1);
        }
        else{
            this->selectV = fake_repeat(selectV, this->shapeX, this->m, 0);
        }
        return;
    }

    FakeArray<NUMX> fake_repeatX(NUMX* dataX){
        return fake_repeat(dataX, this->shapeX, this->m, 0);
    }

    FakeArray<NUMY> fake_repeatY(NUMY* dataY){
        size_t valid_m_length = this->m_end - this->m_start + 1 + this->lookahead;
        if(this->m_end + this->lookahead >= this->m) valid_m_length -= this->lookahead;
        return fake_repeat(dataY, this->shapeY, this->n, 1, this->m_start, valid_m_length);
    }


    void circuit_evaluate(NUMX* dataX, NUMY* dataY, NUMR* selectV, NUMR* res){
        
        std::chrono::high_resolution_clock::time_point start, end;
        start = std::chrono::high_resolution_clock::now();

        // prepare data structures.
        this->inputX = fake_repeat(dataX, this->shapeX, this->m, 0);
        size_t valid_m_length = this->m_end - this->m_start + 1 + this->lookahead;
        if(this->m_end + this->lookahead >= this->m) valid_m_length -= this->lookahead;
        this->inputY = fake_repeat(dataY, this->shapeY, this->n, 1, this->m_start, valid_m_length);
        this->res = res;

        // each task evaluate the circuit locally
        this->subTask->circuit_evaluate(this->inputX, this->inputY, this->selectV);
        end = std::chrono::high_resolution_clock::now();
        time_subTask = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        // MPI_Barrier(MPI_COMM_WORLD);
        end = std::chrono::high_resolution_clock::now();
        time_barrier = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        size_t left_tasks = this->total_tasks;
        int send_start = (left_tasks + 1) / 2;
        while(this->rank < send_start && left_tasks > 1){
            int receive_target = this->rank + send_start;
            if(receive_target < left_tasks){
                std::vector<char> res_recv_buf(this->n * sizeof(NUMR));
                MPI_Recv(res_recv_buf.data(), this->n * sizeof(NUMR), MPI_BYTE, receive_target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // transform the received result to partical result and reduce with my own.
                std::vector<NUMR> tmp(this->n);
                std::memcpy(tmp.data(), res_recv_buf.data(), res_recv_buf.size());

                // partial reduce
                this->subTask->partical_reduction(tmp, this->subTask->res, this->subTask->res, nullptr);
            }

            left_tasks = send_start;
            send_start = (left_tasks+1)/2;
        }

        if(this->rank >= send_start){
            int send_target = this->rank - send_start;
            MPI_Send(this->subTask->res.data(), this->n * sizeof(NUMR), MPI_BYTE, send_target, 0, MPI_COMM_WORLD);
        }

        if(this->rank == 0){
            std::copy(this->subTask->res.begin(), this->subTask->res.end(), res);
        }
        end = std::chrono::high_resolution_clock::now();
        time_combine = (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


        double time_subtask_lists[this->total_tasks];
        double time_idling_list[this->total_tasks];
        // double time_combine_lists[this->total_tasks];
        MPI_Gather(&time_subTask, 1, MPI_DOUBLE, time_subtask_lists, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&time_barrier, 1, MPI_DOUBLE, time_idling_list, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(this->rank == 0){

            // compute the mean time among all the tasks.
            mean_subTask = 0.0, mean_combine = 0.0, var_subTask = 0.0, var_combine = 0.0, max_subTask = 0.0;
            min_subTask = time_subtask_lists[0];
            for(int i=0; i<this->total_tasks; i++){
                mean_subTask += time_subtask_lists[i];
                max_subTask = (time_subtask_lists[i] > max_subTask) ? time_subtask_lists[i] : max_subTask;
                min_subTask = (time_subtask_lists[i] < min_subTask) ? time_subtask_lists[i] : min_subTask;
                total_idling += time_idling_list[i];
            }
            mean_subTask /= this->total_tasks;

            // compute the variance among all the tasks.
            for(int i=0; i<this->total_tasks; i++){
                var_subTask += (time_subtask_lists[i] - mean_subTask) * (time_subtask_lists[i] - mean_subTask);
            }
            var_subTask = sqrt(var_subTask / this->total_tasks);
        }
        else{
            mean_subTask = -1, var_subTask = -1, mean_combine = -1, var_combine = -1;
            max_subTask = -1, min_subTask = -1;
        }
    }

    void print_time_profile(std::ostream& os = std::cout){
        os << "subTask: " << std::setprecision(5) << max_subTask / 1000 << " ms" << std::endl;
        os << "meanSubTask: " << std::setprecision(5) << mean_subTask / 1000 << " ms" << std::endl;
        os << "combine: " << std::setprecision(5) << time_combine / 1000 << " ms" << std::endl;
        os << "barrier: " << std::setprecision(5) << time_barrier / 1000 << " ms" << std::endl;
        os << "total idling" << std::setprecision(5) << total_idling / 1000 << " ms" << std::endl;

        os << "fake array accessing time: " << std::setprecision(5) << fake_array_total_time.count() << " micros" << std::endl;
        // os << "processing before pair: "<< std::setprecision(5) << before_pair_time.count() << " micros" << std::endl;
        os << "other processing time: " << std::setprecision(5) << other_time.count() / 1000 << " ms" << std::endl;
        os << "eq comp time: " << std::setprecision(5) << comp_time.count() / 1000 << " ms" << std::endl;
    }
};


// openMP style parallelization baseline.
template <typename NUMX, typename NUMY, typename NUMT, typename NUMR>
class SubTask_OpenMP{

public:
    // functional logic parameters.
    const int task_id;
    const size_t optimal_block;
    bool have_selective=true;

    // task specific parameters.
    size_t n, m;
    size_t table_start, table_end;
    size_t block_num;
    NUMR initial_value;
    std::vector<NUMR> res;

protected:

    FakeArray<NUMR> selectV;

    std::vector<BlockInfo*> block_info_list;

    size_t which_column(size_t table_loc){
        return table_loc % this->n;
    }

    size_t which_row(size_t table_loc){
        return table_loc % this->m;
    }

    virtual void compute_local_table(std::vector<NUMX>& expandX, std::vector<NUMY>& expandY, std::vector<NUMT>& local_table, BlockInfo* binfo) = 0;

public:

    virtual void partical_reduction(std::vector<NUMR>& resLeft, std::vector<NUMR>& resRight, std::vector<NUMR>& update_res, BlockInfo* binfo) = 0;

    SubTask_OpenMP(const size_t optimal_block, const int task_id):
        task_id(task_id),
        optimal_block(optimal_block){}


    void circuit_construct(const std::vector<size_t> shapeX, const std::vector<size_t> shapeY, const size_t table_start, size_t table_end){
        this->n = shapeX[0], this->m = shapeY[0];
        size_t table_size = table_end - table_start;
        this->table_start = table_start, this->table_end = table_end;
        // how many blocks we need to compute, each block is in optimal block size.
        block_num = (table_size + this->optimal_block - 1) / this->optimal_block;
        for(int i=0; i<block_num; i++){
            size_t local_table_start = table_start + i*optimal_block;
            size_t local_table_end = (i == block_num - 1) ? table_end : local_table_start + optimal_block;
            BlockInfo* blockinfo = new BlockInfo(local_table_start, local_table_end);
            block_info_list.emplace_back(blockinfo);
        }
        this->res.resize(this->n, this->initial_value);
    }


    virtual void circuit_profile(const FakeArray<NUMX>& dataX, const FakeArray<NUMY>& dataY, const FakeArray<NUMR>& selectV){
        // it will generate data at first and then profile?
        if(this->have_selective) this->selectV = selectV;

        // using vectorization to construct the pairwise vector expandX and expandY.
        // std::cout << ">>>>" << block_num << std::endl;
        auto binfo = block_info_list[0];
        std::vector<NUMX> expandX(binfo->block_len);
        std::vector<NUMY> expandY(binfo->block_len);
        std::vector<NUMR> local_table(binfo->block_len);

        for(int p=0; p<binfo->block_len; p++){
            if(p + binfo->t_start < n*m){
                expandX[p] = dataX[p+binfo->t_start];
                expandY[p] = dataY[p+binfo->t_start];
            }
        }

        compute_local_table(expandX, expandY, local_table, binfo);


        // log l rounds partical reduction
        size_t padding_lens = binfo->block_len % this->n != 0 ? this->n - (binfo->block_len % this->n) : 0;
        std::vector<NUMR> current_table = local_table;
        current_table.insert(current_table.end(), padding_lens, NUMR(this->initial_value));

        size_t current_len = binfo->block_len + padding_lens;
        size_t current_rows = current_len / this->n;
        int res_flag = 1;
        int start_point = this->which_column(binfo->t_start);

        while(current_rows + res_flag > 1){

            std::vector<NUMR> resLeft;
            std::vector<NUMR> resRight;
            size_t mid_point;
            if(current_rows % 2 == 0){
                mid_point = current_len / 2;
            }
            else{
                if(res_flag == 1) std::rotate(this->res.begin(), this->res.begin() + start_point, this->res.end());  
                current_table.insert(current_table.end(), this->res.begin(), this->res.end());
                mid_point = (current_len + this->n) / 2;
                if(res_flag == 1){
                    this->res = std::vector<NUMR>(this->n, NUMR(this->initial_value));
                    res_flag = 0;
                }
            }
            resLeft = std::vector<NUMR>(current_table.begin(), current_table.begin()+mid_point);
            resRight = std::vector<NUMR>(current_table.begin() + mid_point, current_table.end());

            current_table.resize(mid_point);
            partical_reduction(resLeft, resRight, current_table, binfo);

            current_len = mid_point;
            current_rows = current_len / this->n;
        }

        if(start_point == 0) this->res = current_table;
        else{
            this->res = std::vector<NUMR>(current_table.end() - start_point, current_table.end());
            this->res.insert(this->res.end(), current_table.begin(), current_table.begin() + this->n - start_point);
        }
        return;
    }


    virtual void circuit_evaluate(const FakeArray<NUMX>& dataX, const FakeArray<NUMY>& dataY, const FakeArray<NUMR>& selectV){
        if(this->have_selective) this->selectV = selectV;

        // using vectorization to construct the pairwise vector expandX and expandY.

        // for openmp
        this->res.clear(); // clear all elems, and set size = 0;

        for(int i=0; i<block_num; i++){
            auto binfo = block_info_list[i];
            std::vector<NUMX> expandX(binfo->block_len);
            std::vector<NUMY> expandY(binfo->block_len);
            std::vector<NUMR> local_table(binfo->block_len);

            for(int p=0; p<binfo->block_len; p++){
                if(p + binfo->t_start < n*m){
                    expandX[p] = dataX[p+binfo->t_start];
                    expandY[p] = dataY[p+binfo->t_start];
                }
            }

            compute_local_table(expandX, expandY, local_table, binfo);

            this->res.insert(this->res.end(), local_table.begin(), local_table.end());
        }
        return;
    }

};


template <typename NUMX, typename NUMY, typename NUMT, typename NUMR, template<typename, typename, typename, typename> class TASK>
class MPIPTRTask_OpenMP : public PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>{

public:
    int rank;
    std::shared_ptr<TASK<NUMX, NUMY, NUMT, NUMR>> subTask;
    size_t m_start, m_end;
    double time_subTask, time_combine;
    std::vector<int> res_lengths, res_offsets;

    using PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>::PTRTask;
    MPIPTRTask_OpenMP(int tasks, size_t optimal_block):
        PTRTask<NUMX, NUMY, NUMT, NUMR, TASK>(tasks, optimal_block){
            int task_size;
            MPI_Comm_size(MPI_COMM_WORLD, &task_size);
            if(task_size != tasks){
                throw std::runtime_error("MPI task number: " + std::to_string(task_size) + " != config task size: " + std::to_string(tasks));
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &(this->rank));
            // cout << "my rank is: " << this->rank << endl;
            m_start = 0;
            m_end = 0;
            res_lengths.resize(task_size);
            res_offsets.resize(task_size);
        }
    

    void task_split(size_t table_size, size_t task_length) override {

        size_t left_table_size = table_size - task_length * this->total_tasks;

        // each task compute its table_start and table_end;
        size_t table_start, table_end;
        if(this->rank < left_table_size){
            table_start = this->rank * (task_length + 1);
            table_end = (this->rank + 1) * (task_length + 1);
        }
        else{
            table_start = this->rank * task_length + left_table_size;
            table_end = (this->rank+1) * task_length + left_table_size;
        }

        this->m_start = table_start / this->n;
        this->m_end = (table_end - 1) / this->n;

        // for rank0, construct the results lengthes and offsets.
        // this->res_lengths.resize(this->total_tasks);
        // this->res_offsets.resize(this->total_tasks);

        int tmp_table_start, tmp_table_end;
        for(int i=0; i < this->total_tasks; i++){
            if(i < left_table_size){
                tmp_table_start = i * (task_length + 1);
                tmp_table_end = (i + 1) * (task_length + 1);
            }
            else{
                tmp_table_start = i * task_length + left_table_size;
                tmp_table_end = (i+1) * task_length + left_table_size;
            }
            this->res_lengths[i] = (tmp_table_end - tmp_table_start) * sizeof(NUMR);
            this->res_offsets[i] = tmp_table_start * sizeof(NUMR);
        }

        // then each task locally construct the sub-circuit.
        create_sub_task(this->optimal_block, this->rank, table_start, table_end);
    }

    void create_sub_task(size_t optimal_block, int task_id, size_t table_start, size_t table_end) override {
        auto subTaskPtr(new TASK<NUMX, NUMY, NUMT, NUMR>(optimal_block, task_id));
        this->subTask.reset(subTaskPtr);
        this->subTask->initial_value = this->default_value;
        this->subTask->circuit_construct(this->shapeX, this->shapeY, table_start, table_end);

        // openmp
        // params for mpi gatherv
        

        return;
    }

    void set_selective_value(NUMR* selectV, int axis = 0){
        this->have_selective = true;
        if(axis == 0){
            this->selectV = fake_repeat(selectV, this->shapeY, this->n, 1, this->m_start, this->m_end - this->m_start + 1);
        }
        else{
            this->selectV = fake_repeat(selectV, this->shapeX, this->m, 0);
        }
        return;
    }

    void circuit_evaluate(NUMX* dataX, NUMY* dataY, NUMR* selectV, NUMR* res){
        
        clock_t start, end;
        start = clock();
        // prepare data structures.
        this->inputX = fake_repeat(dataX, this->shapeX, this->m, 0);
        this->inputY = fake_repeat(dataY, this->shapeY, this->n, 1, this->m_start, this->m_end - this->m_start + 1);
        this->res = res;

        // each task evaluate the circuit locally
        this->subTask->circuit_evaluate(this->inputX, this->inputY, this->selectV);
        end = clock();
        time_subTask = double((end - start)*1000)/(CLOCKS_PER_SEC);

        start = clock();

        // for openmp
        // Notice: for openmp version, subtask.res is used to store local_table 
        // Todo: receive and send
        std::vector<char> res_recv_buf(this->n * this->m * sizeof(NUMR)); // n should be 1;

        // todo: need to define
        // this->result_lengths;
        // this->offsets;

        MPI_Gatherv(this->subTask->res.data(), this->subTask->res.size() * sizeof(NUMR), MPI_BYTE, res_recv_buf.data(), this->res_lengths.data(), this->res_offsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (this->rank == 0){
            // todo: get all others res and concat into one vector
            //std::vector<NUMR> current_table = res_recv_buf;

            std::vector<NUMR> current_table(this->n * this->m);
            std::memcpy(current_table.data(), res_recv_buf.data(), res_recv_buf.size());

            size_t current_len = current_table.size();
            int start_point = 0;

            while(current_len  > 1){

                std::vector<NUMR> resLeft;
                std::vector<NUMR> resRight;
                size_t mid_point;
                if(current_len % 2 == 0){
                    mid_point = current_len / 2;
                }
                else{
                    current_table.push_back(this->default_value); // push_back: insert into the tail of the vector
                    mid_point = (current_len + 1) / 2;
                }

                resLeft = std::vector<NUMR>(current_table.begin(), current_table.begin()+mid_point);
                resRight = std::vector<NUMR>(current_table.begin() + mid_point, current_table.end());

                current_table.resize(mid_point);
                this->subTask->partical_reduction(resLeft, resRight, current_table, nullptr);

                current_len = mid_point;
            }

            std::copy(current_table.begin(), current_table.end(), res);
        }
        end = clock();
        time_combine = double((end - start)*1000)/(CLOCKS_PER_SEC);
    }
};

#endif
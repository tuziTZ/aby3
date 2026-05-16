#ifndef DATATYPE_H  
#define DATATYPE_H  

#include <array>
#include <vector>
#include <numeric>  
#include <algorithm>  
#include <sstream>  
#include <stdexcept> 
#include <iostream>
#include <unordered_map>
#include <chrono>

using namespace std;

extern std::chrono::duration<double, std::micro> fake_array_total_time;
extern std::chrono::duration<double, std::micro> other_time;
extern std::chrono::duration<double, std::micro> comp_time;
extern std::chrono::duration<double, std::micro> comp_process_time;
extern std::chrono::duration<double, std::micro> comp_comm_time;


template <typename T>
class FakeArray{

public:
    std::vector<size_t> dim;  
    std::vector<size_t> fake_dim;
    T* data;
    size_t data_len;
    size_t start_point;
    size_t fake_axis; 

private:
    std::vector<size_t> real_dim_prod; 
    size_t real_size;
    std::vector<size_t> dim_prod;  

public:
    FakeArray(T* data_value, const std::vector<size_t>& shape, size_t reps, size_t axis=0):
        dim(shape),  
        fake_dim(shape),  
        data(data_value),  
        data_len(0),
        start_point(0),
        fake_axis(axis),
        real_dim_prod(cumprod(dim)),
        real_size(real_dim_prod[0])
    {
        fake_dim.insert(fake_dim.begin() + axis, reps);
        dim_prod = cumprod(fake_dim);
        data_len = real_size;
    }

    FakeArray() = default;

    FakeArray(T* data_value, const std::vector<size_t>& shape, size_t reps, size_t axis, size_t start_point, size_t data_len):
        dim(shape),
        fake_dim(shape),
        data(data_value),
        data_len(data_len),
        start_point(start_point),
        fake_axis(axis),
        real_dim_prod(cumprod(dim)),
        real_size(real_dim_prod[0])
    {
        fake_dim.insert(fake_dim.begin() + axis, reps);
        dim_prod = cumprod(fake_dim);
    }


    T& operator[](size_t index) const {
        auto start = std::chrono::high_resolution_clock::now();  // 获取当前时间
        std::vector<size_t> indices = index2tuple(index);  
        indices.erase(indices.begin() + fake_axis);  
        size_t new_index = tuple2index(indices);
        if(new_index > real_size){
            std::ostringstream oss;  
            oss << "FakeArray index out of range (index " << new_index << " < total length " << real_size << ")";  
            throw std::out_of_range(oss.str());  
        }
        if(new_index - start_point > data_len){
            std::ostringstream oss;  
            oss << " Partial FakeArray index out of range (index " << new_index + start_point << " < total real length " << data_len << ")";  
            throw std::out_of_range(oss.str());
        }
        T& res = data[new_index - start_point];
        auto end = std::chrono::high_resolution_clock::now();  // 获取当前时间
        fake_array_total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return res;  
    }

private:  
    std::vector<size_t> cumprod(const std::vector<size_t>& lst) {  
        std::vector<size_t> result(lst.size());  
        std::partial_sum(lst.rbegin(), lst.rend(), result.rbegin(), std::multiplies<size_t>());  
        return result;  
    }  

    std::vector<size_t> index2tuple(size_t i) const {  
        std::vector<size_t> indices(fake_dim.size());  
        size_t offset = i;  
        for (size_t j = 1; j < dim_prod.size(); j++) {  
            indices[j-1] = offset / dim_prod[j];  
            offset -= indices[j-1] * dim_prod[j];  
        }  
        indices.back() = offset % dim_prod.back();  
        return indices;  
    }

    size_t tuple2index(const std::vector<size_t>& lst) const {  
        size_t index = lst.back();
        for (size_t j = dim.size() - 1; j > 0; j--) {  
            index += lst[j-1] * real_dim_prod[j];  
        }  
        return index;  
    }  
};


template <typename T>  
FakeArray<T> fake_repeat(T* data, const std::vector<size_t>& shape, const size_t reps, const size_t axis) {  
    return FakeArray<T>(data, shape, reps, axis);  
}


template <typename T>  
FakeArray<T> fake_repeat(T* partial_data, const std::vector<size_t>& shape, const size_t reps, const size_t axis, size_t start_point, size_t data_len) {  
    return FakeArray<T>(partial_data, shape, reps, axis, start_point, data_len);  
}


class BlockInfo{
public:
    size_t t_start, t_end;
    size_t r_start, r_end;
    size_t block_len;

public:
    BlockInfo(size_t t_start, size_t t_end):
        t_start(t_start),
        t_end(t_end)
    {
        block_len = t_end - t_start;
    }
};


#endif
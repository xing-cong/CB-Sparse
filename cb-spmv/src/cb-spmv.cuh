#pragma once

#include "format.h"
#include "utils.h"

namespace cb_spmv_detail {

inline char *get_addr_with_coo_comp(const COO_BLOCK_COMP& blk_coo_comp)
{
    int blk_nnz = blk_coo_comp.blk_nnz;
    const IndexTypeComp *h_coo_idx = blk_coo_comp.idx.data();
    const ValType *h_val = blk_coo_comp.val.data();

    int padding = (blk_nnz * sizeof(IndexTypeComp)) % sizeof(ValType);
    if (padding != 0)
    {
        padding = sizeof(ValType) - padding;
    }

    char *d_blk_head_index;
    CHECK_CUDA(cudaMalloc((void **)(&d_blk_head_index), blk_nnz * sizeof(IndexTypeComp) + blk_nnz * sizeof(ValType) + padding));

    IndexTypeComp *d_coo_idx = (IndexTypeComp *)d_blk_head_index;
    ValType *d_val = (ValType *)((char *)d_coo_idx + blk_nnz * sizeof(IndexTypeComp) + padding);

    CHECK_CUDA(cudaMemcpy(d_coo_idx, h_coo_idx, blk_nnz * sizeof(IndexTypeComp), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_val, blk_nnz * sizeof(ValType), cudaMemcpyHostToDevice));

    return d_blk_head_index;
}

inline char* get_addr_with_csr(const CSR_BLOCK& blk_csr){
    int blk_nnz = blk_csr.blk_nnz;
    const IndexTypeOrig *h_row_ptr = blk_csr.row_ptr.data();
    const IndexTypeOrig *h_col_idx = blk_csr.col_idx.data();
    const ValType *h_val = blk_csr.val.data();
    int padding = ((BLOCK_SIZE+1+blk_nnz) * sizeof(IndexTypeOrig)) % sizeof(ValType);
    if(padding != 0){
        padding = sizeof(ValType) - padding;
    }

    char *d_blk_head_index;
    CHECK_CUDA(cudaMalloc((void **)(&d_blk_head_index), (BLOCK_SIZE  + 1 + blk_nnz) * sizeof(IndexTypeOrig) + blk_nnz * sizeof(ValType) + padding));

    IndexTypeOrig *d_row_ptr = (IndexTypeOrig *)d_blk_head_index;
    IndexTypeOrig *d_col_idx = (IndexTypeOrig *)((char *)d_row_ptr + (BLOCK_SIZE+1) * sizeof(IndexTypeOrig));
    ValType *d_val = (ValType *)((char *)d_col_idx + blk_nnz * sizeof(IndexTypeOrig) + padding);

    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr, (BLOCK_SIZE+1) * sizeof(IndexTypeOrig), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, h_col_idx, blk_nnz * sizeof(IndexTypeOrig), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_val, blk_nnz * sizeof(ValType), cudaMemcpyHostToDevice));

    return d_blk_head_index;
}

inline char *get_addr_with_dense(const DENSE_BLOCK& blk_dense){
    const ValType *h_val = blk_dense.val.data();
    char *d_blk_head_index;
    CHECK_CUDA(cudaMalloc((void **)(&d_blk_head_index), BLOCK_SIZE * BLOCK_SIZE * sizeof(ValType)));
    CHECK_CUDA(cudaMemcpy(d_blk_head_index, h_val, BLOCK_SIZE * BLOCK_SIZE * sizeof(ValType), cudaMemcpyHostToDevice));
    return d_blk_head_index;
}

__global__ void spmv_cuda_kernel_comp(int gb_nnzb, int *nnz_per_blk, BlockType *type_per_blk, int *gb_row_idx, int *gb_col_idx, char **blk_head_index_array, ValType *d_x, ValType *d_y)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_id = global_id >> 5;
    int warp_id_in_tblock = threadIdx.x >> 5;
    int tid_in_warp = threadIdx.x & 0b11111;

    if (blk_id < gb_nnzb && nnz_per_blk[blk_id] > 0)
    {
        int blk_nnz = nnz_per_blk[blk_id];
        BlockType blk_type = type_per_blk[blk_id];
        char *blk_head_index = blk_head_index_array[blk_id];
        int blk_row_idx = gb_row_idx[blk_id];
        int blk_col_idx = gb_col_idx[blk_id];

        int gb_row_offset = blk_row_idx*BLOCK_SIZE;

        __shared__ ValType s_x[WARP_PER_BLOCK*BLOCK_SIZE];

        ValType *s_x_warp = &s_x[warp_id_in_tblock*BLOCK_SIZE];

        if (tid_in_warp < BLOCK_SIZE)
        {
            s_x_warp[tid_in_warp] = d_x[blk_col_idx * BLOCK_SIZE + tid_in_warp];
        }

        if (blk_type == COO)
        {
            int padding = (blk_nnz * sizeof(IndexTypeComp)) % sizeof(ValType);
            if (padding != 0)
            {
                padding = sizeof(ValType) - padding;
            }
            IndexTypeComp *d_coo_idx = (IndexTypeComp *)blk_head_index;
            ValType *d_val = (ValType *)((char *)d_coo_idx + blk_nnz * sizeof(IndexTypeComp) + padding);
            
            for (int k = tid_in_warp; k < blk_nnz; k = k + WARP_SIZE)
            {
                ValType d_val_k = d_val[k];
                IndexTypeComp coo_idx = d_coo_idx[k];
                IndexTypeComp d_col_index_k = coo_idx & mask_0_3;
                IndexTypeComp d_row_index_k = (coo_idx & mask_4_7) >> 4;
                ValType d_x_d_col_index_k = s_x_warp[d_col_index_k];
                atomicAdd(&d_y[gb_row_offset + d_row_index_k], d_val_k * d_x_d_col_index_k);
            }
        }
        else if (blk_type == CSR)
        {
            int padding = ((BLOCK_SIZE + 1 + blk_nnz) * sizeof(IndexTypeOrig)) % sizeof(ValType);
            if (padding != 0)
            {
                padding = sizeof(ValType) - padding;
            }
            IndexTypeOrig *d_row_ptr = (IndexTypeOrig *)blk_head_index;
            IndexTypeOrig *d_col_idx = (IndexTypeOrig *)((char *)d_row_ptr + (BLOCK_SIZE+1) * sizeof(IndexTypeOrig));
            ValType *d_val = (ValType *)((char *)d_col_idx + blk_nnz * sizeof(IndexTypeOrig) + padding);
            ValType sum = 0;
            ValType sumsum = 0;
            int row_id = tid_in_warp>>1;
            int offset = tid_in_warp&0x1;
            for(int k=d_row_ptr[row_id]+offset; k < d_row_ptr[row_id+1]; k = k+2){
                ValType d_val_k = d_val[k];
                IndexTypeOrig d_col_idx_k = d_col_idx[k];
                sum += s_x_warp[d_col_idx_k] * d_val_k;
            }
            sum += __shfl_down_sync(0xffffffff, sum, 1);
            sumsum += __shfl_down_sync(0xffffffff, sum, tid_in_warp);

            if(tid_in_warp < BLOCK_SIZE && sumsum !=0){
                atomicAdd(&d_y[gb_row_offset+tid_in_warp], sumsum);
            }
        }
        else if (blk_type == DENSE)
        {
            ValType* d_val = (ValType*)blk_head_index;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, nvcuda::wmma::col_major> x_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> y_frag;
            nvcuda::wmma::fill_fragment(y_frag, 0.0f);
            __shared__ __align__(16) __half d_val_half[BLOCK_SIZE*BLOCK_SIZE*8];
            __shared__ __align__(16) __half d_x_half[BLOCK_SIZE*BLOCK_SIZE*8];
            __half *d_x_half_warp = &d_x_half[warp_id_in_tblock*BLOCK_SIZE*BLOCK_SIZE];
            __half *d_val_half_warp = &d_val_half[warp_id_in_tblock*BLOCK_SIZE*BLOCK_SIZE];
            for(int i=tid_in_warp; i < BLOCK_SIZE*BLOCK_SIZE; i=i+WARP_SIZE){
                float d_val_i = static_cast<float>(d_val[i]);
                float d_x_i = static_cast<float>(s_x_warp[i & 0b1111]);
                d_val_half_warp[i] = __float2half(d_val_i);
                d_x_half_warp[i] = __float2half(d_x_i);
            }
            nvcuda::wmma::load_matrix_sync(a_frag, d_val_half_warp, BLOCK_SIZE);
            nvcuda::wmma::load_matrix_sync(x_frag, d_x_half_warp, BLOCK_SIZE);
            nvcuda::wmma::mma_sync(y_frag, a_frag, x_frag, y_frag);
            __shared__ float d_y_float[BLOCK_SIZE*BLOCK_SIZE*8];
            float *d_y_float_warp = &d_y_float[warp_id_in_tblock*BLOCK_SIZE*BLOCK_SIZE];
            nvcuda::wmma::store_matrix_sync(d_y_float_warp, y_frag, BLOCK_SIZE, nvcuda::wmma::mem_row_major);
            if(tid_in_warp < BLOCK_SIZE){
                atomicAdd(&d_y[gb_row_offset+tid_in_warp], (double)d_y_float_warp[tid_in_warp*BLOCK_SIZE]);
            }
        }
        else
        {
        }
    }
}

__global__ void spmv_cuda_kernel_gather(int gb_nnzb, int *nnz_per_blk, BlockType *type_per_blk, int *gb_row_idx, int *gb_col_idx, char **blk_head_index_array, int* d_block_cols, int* d_block_cols_offset, ValType *d_x, ValType *d_y)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int blk_id = global_id >> 5;
    int tid_in_warp = threadIdx.x & 0b11111;

    if (blk_id < gb_nnzb && nnz_per_blk[blk_id] > 0)
    {
        int blk_nnz = nnz_per_blk[blk_id];
        BlockType blk_type = type_per_blk[blk_id];
        char *blk_head_index = blk_head_index_array[blk_id];
        int blk_row_idx = gb_row_idx[blk_id];
        int blk_col_idx = gb_col_idx[blk_id];

        int gb_row_offset = blk_row_idx*BLOCK_SIZE;

        if (blk_type == COO)
        {
            int padding = (blk_nnz * sizeof(IndexTypeComp)) % sizeof(ValType);
            if (padding != 0)
            {
                padding = sizeof(ValType) - padding;
            }
            IndexTypeComp *d_coo_idx = (IndexTypeComp *)blk_head_index;
            ValType *d_val = (ValType *)((char *)d_coo_idx + blk_nnz * sizeof(IndexTypeComp) + padding);
            
            for (int k = tid_in_warp; k < blk_nnz; k = k + WARP_SIZE)
            {
                ValType d_val_k = d_val[k];
                IndexTypeComp coo_idx = d_coo_idx[k];
                IndexTypeComp d_col_index_k = coo_idx & mask_0_3;
                int d_gather_col_idx = blk_col_idx * BLOCK_SIZE + (int)d_col_index_k;
                int d_col_idx_origin =  d_block_cols[d_block_cols_offset[blk_row_idx]+d_gather_col_idx];
                IndexTypeComp d_row_index_k = (coo_idx & mask_4_7) >> 4;
                ValType d_x_d_col_index_k = d_x[d_col_idx_origin];
                atomicAdd(&d_y[gb_row_offset + d_row_index_k], d_val_k * d_x_d_col_index_k);
            }
        }
        else if (blk_type == CSR)
        {
            int padding = ((BLOCK_SIZE + 1 + blk_nnz) * sizeof(IndexTypeOrig)) % sizeof(ValType);
            if (padding != 0)
            {
                padding = sizeof(ValType) - padding;
            }
            IndexTypeOrig *d_row_ptr = (IndexTypeOrig *)blk_head_index;
            IndexTypeOrig *d_col_idx = (IndexTypeOrig *)((char *)d_row_ptr + (BLOCK_SIZE+1) * sizeof(IndexTypeOrig));
            ValType *d_val = (ValType *)((char *)d_col_idx + blk_nnz * sizeof(IndexTypeOrig) + padding);
            ValType sum = 0;
            ValType sumsum = 0;
            int row_id = tid_in_warp>>1;
            int offset = tid_in_warp&0x1;
            for(int k=d_row_ptr[row_id]+offset; k < d_row_ptr[row_id+1]; k = k+2){
                ValType d_val_k = d_val[k];
                IndexTypeOrig d_col_idx_k = d_col_idx[k];
                int d_gather_col_idx = blk_col_idx * BLOCK_SIZE + d_col_idx_k;
                int d_col_idx_origin =  d_block_cols[d_block_cols_offset[blk_row_idx]+d_gather_col_idx];
                ValType d_x_d_col_idx_k = d_x[d_col_idx_origin];
                sum += d_x_d_col_idx_k * d_val_k;
            }
            sum += __shfl_down_sync(0xffffffff, sum, 1);
            sumsum += __shfl_down_sync(0xffffffff, sum, tid_in_warp);

            if(tid_in_warp < BLOCK_SIZE && sumsum !=0){
                atomicAdd(&d_y[gb_row_offset+tid_in_warp], sumsum);
            }
        }
        else if (blk_type == DENSE)
        {
            ValType* d_val = (ValType*)blk_head_index;
            ValType sum = 0.0f;
            if (tid_in_warp < 16) {
                for (int col = 0; col < 8; col++) {
                    float val = __ldg(&d_val[tid_in_warp * BLOCK_SIZE + col]);
                    int d_col_idx_origin =  d_block_cols[d_block_cols_offset[blk_row_idx]+col];
                    sum += val * d_x[d_col_idx_origin];
                }
            } else {
                int row_id2 = tid_in_warp - 16;
                if (row_id2 < 16) {
                    for (int col = 8; col < 16; col++) {
                        float val = __ldg(&d_val[row_id2 * BLOCK_SIZE + col]);
                        int d_col_idx_origin =  d_block_cols[d_block_cols_offset[blk_row_idx]+col];
                        sum += val * d_x[d_col_idx_origin];
                    }
                }
            }
            float sum_other = __shfl_xor_sync(0xffffffff, sum, 16);
            float final_sum = 0.0f;
            if (tid_in_warp < 16) {
                final_sum = sum + sum_other;
            }
            if (tid_in_warp < 16) {
                atomicAdd(&d_y[gb_row_offset + tid_in_warp], (double)final_sum);
            }
        }
    }
}

}

inline PROGRAM_METRICS cb_spmv(const GLOBAL_BLOCK& gb_block, const ValType *x, ValType *y)
{
    const size_t x_elements = static_cast<size_t>(gb_block.gb_n) * BLOCK_SIZE;
    const size_t y_elements = static_cast<size_t>(gb_block.gb_m) * BLOCK_SIZE;
    std::vector<ValType> h_x(x_elements, 0.0);
    memcpy(h_x.data(), x, gb_block.mtx_n * sizeof(ValType));

    std::vector<char *> block_addresses;
    block_addresses.reserve(gb_block.gb_nnzb);
    for (int i = 0; i < gb_block.gb_nnzb; i++)
    {
        char *block_address = nullptr;
        if (gb_block.nnz_per_blk[i] > 0)
        {
            int old_idx = gb_block.block_idx_array[i].origin_idx;
            BlockType block_type = gb_block.type_per_blk[i];
            if (block_type == COO)
            {
                block_address = cb_spmv_detail::get_addr_with_coo_comp(gb_block.data_per_coo_blk_comp.at(old_idx));
            }
            else if (block_type == CSR)
            {
                block_address = cb_spmv_detail::get_addr_with_csr(gb_block.data_per_csr_blk.at(old_idx));
            }
            else if (block_type == DENSE)
            {
                block_address = cb_spmv_detail::get_addr_with_dense(gb_block.data_per_dense_blk.at(old_idx));
            }
        }
        block_addresses.push_back(block_address);
    }

    int *d_gb_row_idx = nullptr;
    int *d_gb_col_idx = nullptr;
    int *d_nnz_per_blk = nullptr;
    BlockType *d_type_per_blk = nullptr;
    int *d_block_cols = nullptr;
    int *d_block_cols_offset = nullptr;
    char **d_block_addresses = nullptr;
    ValType *d_x = nullptr;
    ValType *d_y = nullptr;

    CHECK_CUDA(cudaMalloc((void **)(&d_gb_row_idx), gb_block.gb_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)(&d_gb_col_idx), gb_block.gb_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)(&d_nnz_per_blk), gb_block.gb_nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)(&d_type_per_blk), gb_block.gb_nnzb * sizeof(BlockType)));
    CHECK_CUDA(cudaMalloc((void **)(&d_block_addresses), gb_block.gb_nnzb * sizeof(char *)));
    CHECK_CUDA(cudaMalloc((void **)(&d_x), x_elements * sizeof(ValType)));
    CHECK_CUDA(cudaMalloc((void **)(&d_y), y_elements * sizeof(ValType)));

    if (gb_block.gather_flag)
    {
        CHECK_CUDA(cudaMalloc((void **)(&d_block_cols), gb_block.block_cols.size() * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void **)(&d_block_cols_offset), gb_block.block_cols_offset.size() * sizeof(int)));
    }

    CHECK_CUDA(cudaMemcpy(d_gb_row_idx, gb_block.gb_row_idx.data(), gb_block.gb_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gb_col_idx, gb_block.gb_col_idx.data(), gb_block.gb_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_nnz_per_blk, gb_block.nnz_per_blk.data(), gb_block.gb_nnzb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_type_per_blk, gb_block.type_per_blk.data(), gb_block.gb_nnzb * sizeof(BlockType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_block_addresses, block_addresses.data(), gb_block.gb_nnzb * sizeof(char *), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), x_elements * sizeof(ValType), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, y_elements * sizeof(ValType)));

    if (gb_block.gather_flag)
    {
        CHECK_CUDA(cudaMemcpy(d_block_cols, gb_block.block_cols.data(), gb_block.block_cols.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_cols_offset, gb_block.block_cols_offset.data(), gb_block.block_cols_offset.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = (gb_block.gb_nnzb + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;

    auto launch_kernel = [&]()
    {
        if (gb_block.gather_flag)
        {
            cb_spmv_detail::spmv_cuda_kernel_gather<<<num_blocks, num_threads>>>(
                gb_block.gb_nnzb,
                d_nnz_per_blk,
                d_type_per_blk,
                d_gb_row_idx,
                d_gb_col_idx,
                d_block_addresses,
                d_block_cols,
                d_block_cols_offset,
                d_x,
                d_y);
        }
        else
        {
            cb_spmv_detail::spmv_cuda_kernel_comp<<<num_blocks, num_threads>>>(
                gb_block.gb_nnzb,
                d_nnz_per_blk,
                d_type_per_blk,
                d_gb_row_idx,
                d_gb_col_idx,
                d_block_addresses,
                d_x,
                d_y);
        }
    };

    launch_kernel();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y, d_y, gb_block.mtx_m * sizeof(ValType), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemset(d_y, 0, y_elements * sizeof(ValType)));

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < ITER; i++)
    {
        launch_kernel();
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    PROGRAM_METRICS metrics;
    metrics.runtime = static_cast<double>(elapsed_time) / ITER;
    if (metrics.runtime > 0.0)
    {
        metrics.gflops = 2.0 * static_cast<double>(gb_block.mtx_nnz) * 1.0e-6 / metrics.runtime;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_gb_row_idx));
    CHECK_CUDA(cudaFree(d_gb_col_idx));
    CHECK_CUDA(cudaFree(d_nnz_per_blk));
    CHECK_CUDA(cudaFree(d_type_per_blk));
    for (char *block_address : block_addresses)
    {
        if (block_address != nullptr)
        {
            CHECK_CUDA(cudaFree(block_address));
        }
    }
    CHECK_CUDA(cudaFree(d_block_addresses));
    if (d_block_cols != nullptr)
    {
        CHECK_CUDA(cudaFree(d_block_cols));
    }
    if (d_block_cols_offset != nullptr)
    {
        CHECK_CUDA(cudaFree(d_block_cols_offset));
    }
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return metrics;
}

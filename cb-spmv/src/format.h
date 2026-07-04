#pragma once

#include "macros.h"

struct COO_BLOCK_COMP{
    int blk_nnz;
    std::vector<IndexTypeComp> idx;
    std::vector<ValType> val;
    COO_BLOCK_COMP() : blk_nnz(0) {}
};

struct COO_BLOCK_ORIG{
    int blk_nnz;
    std::vector<IndexTypeOrig> row_idx;
    std::vector<IndexTypeOrig> col_idx;
    std::vector<ValType> val;
    COO_BLOCK_ORIG() : blk_nnz(0) {}
};

struct CSR_BLOCK{
    int blk_nnz;
    std::vector<IndexTypeOrig> row_ptr;
    std::vector<IndexTypeOrig> col_idx;
    std::vector<ValType> val;
    CSR_BLOCK() : blk_nnz(0) {}
};

struct DENSE_BLOCK{
    int blk_nnz;
    std::vector<ValType> val;
    DENSE_BLOCK() : blk_nnz(0) {}
};

struct BLOCK_IDX{
    int blk_nnz;
    int origin_idx;
    int end_idx;

    BLOCK_IDX& operator=(const BLOCK_IDX& other) {
        if (this == &other) {
            return *this;
        }
        this->blk_nnz = other.blk_nnz;
        this->origin_idx = other.origin_idx;
        this->end_idx = other.end_idx;
        return *this;
    }
};

bool cmp_nnz(BLOCK_IDX a, BLOCK_IDX b){
    return a.blk_nnz > b.blk_nnz;
};

bool cmp_origin_idx(BLOCK_IDX a, BLOCK_IDX b){
    return a.origin_idx < b.origin_idx;
};

bool cmp_end_idx(BLOCK_IDX a, BLOCK_IDX b){
    return a.end_idx < b.end_idx;
};

struct GLOBAL_BLOCK{
    int mtx_m;
    int mtx_n;
    int mtx_nnz;
    bool gather_flag;
    int super_sparse_blocks;
    int analyzed_blocks;
    double super_sparse_ratio;
    bool load_balanced;
    double load_stddev_before;
    double load_stddev_after;
    int gb_m;
    int gb_n;
    int gb_nnzb;

    std::vector<int> gb_row_idx;
    std::vector<int> gb_col_idx;
    std::vector<int> nnz_per_blk;
    std::vector<BlockType> type_per_blk;

    std::vector<int> block_cols_offset;
    std::vector<int> block_cols;

    std::vector<BLOCK_IDX> block_idx_array;

    std::vector<COO_BLOCK_ORIG> data_per_coo_blk_orig;
    std::unordered_map<int, COO_BLOCK_COMP> data_per_coo_blk_comp;
    std::unordered_map<int, CSR_BLOCK> data_per_csr_blk;
    std::unordered_map<int, DENSE_BLOCK> data_per_dense_blk;

    GLOBAL_BLOCK(int mtx_m, int mtx_n, int mtx_nnz) : mtx_m(mtx_m), mtx_n(mtx_n), mtx_nnz(mtx_nnz) {
        gb_m = (mtx_m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gb_n = (mtx_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gb_nnzb = 0;
        gather_flag = false;
        super_sparse_blocks = 0;
        analyzed_blocks = 0;
        super_sparse_ratio = 0.0;
        load_balanced = false;
        load_stddev_before = 0.0;
        load_stddev_after = 0.0;
    }
    void reset(int mtx_m, int mtx_n, int mtx_nnz) {
        this->mtx_m = mtx_m;
        this->mtx_n = mtx_n;
        this->mtx_nnz = mtx_nnz;
        gb_m = (mtx_m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gb_n = (mtx_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gb_nnzb = 0;
        gather_flag = false;
        super_sparse_blocks = 0;
        analyzed_blocks = 0;
        super_sparse_ratio = 0.0;
        load_balanced = false;
        load_stddev_before = 0.0;
        load_stddev_after = 0.0;
        gb_row_idx.clear();
        gb_col_idx.clear();
        nnz_per_blk.clear();
        type_per_blk.clear();
        block_cols_offset.clear();
        block_cols.clear();
        block_idx_array.clear();
        data_per_coo_blk_orig.clear();
        data_per_coo_blk_comp.clear();
        data_per_csr_blk.clear();
        data_per_dense_blk.clear();

        gb_row_idx.shrink_to_fit();
        gb_col_idx.shrink_to_fit();
        nnz_per_blk.shrink_to_fit();
        type_per_blk.shrink_to_fit();
        block_cols_offset.shrink_to_fit();
        block_cols.shrink_to_fit();
        block_idx_array.shrink_to_fit();
        data_per_coo_blk_orig.shrink_to_fit();
    }
};

struct PROGRAM_METRICS{
    double runtime{0.0};
    double gflops{0.0};
};

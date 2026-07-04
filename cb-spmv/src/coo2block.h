#include "format.h"
#include <algorithm>
#include <cmath>

using namespace std;

void gather(const int* row_idx, const int* col_idx, std::vector<int>& gather_col_idx, GLOBAL_BLOCK &gb_block){
    vector<vector<int>> block_cols(gb_block.gb_m);

    #pragma omp parallel
    {
        vector<vector<int>> local_block_cols(gb_block.gb_m);
        #pragma omp for nowait
        for (int i=0; i<gb_block.mtx_nnz; i++){
            int blk_idx = row_idx[i] / BLOCK_SIZE;
            local_block_cols[blk_idx].push_back(col_idx[i]);
        }

        #pragma omp critical
        {
            for (int i=0; i<gb_block.gb_m; i++) {
                block_cols[i].insert(block_cols[i].end(), local_block_cols[i].begin(), local_block_cols[i].end());
            }
        }
    }

    #pragma omp parallel for
    for (int i=0; i<gb_block.gb_m; i++){
        sort(block_cols[i].begin(), block_cols[i].end());
        block_cols[i].erase(unique(block_cols[i].begin(), block_cols[i].end()), block_cols[i].end());
    }

    #pragma omp parallel for
    for (int i=0; i<gb_block.mtx_nnz; i++){
        int blk_idx = row_idx[i] / BLOCK_SIZE;
        int pos = (int)(lower_bound(block_cols[blk_idx].begin(), block_cols[blk_idx].end(), col_idx[i]) - block_cols[blk_idx].begin());
        gather_col_idx[i] = pos;
    }

    vector<int> block_cols_offset(gb_block.gb_m+1, 0);

    for(int i=1; i<=gb_block.gb_m; i++){
        block_cols_offset[i] = block_cols_offset[i-1] + (int)block_cols[i-1].size();
    }

    vector<int> flat_block_cols(block_cols_offset.back());
    #pragma omp parallel for
    for (int i = 0; i < gb_block.gb_m; i++) {
        int base = block_cols_offset[i];
        for (size_t j = 0; j < block_cols[i].size(); j++) {
            flat_block_cols[base + j] = block_cols[i][j];
        }
    }
    gb_block.block_cols_offset = move(block_cols_offset);
    gb_block.block_cols = move(flat_block_cols);
}

bool gather_flag(GLOBAL_BLOCK &gb_block){
    vector<int> elenumToblkcnt_0_255(BLOCK_SIZE*BLOCK_SIZE, 0);
    for(int i=0; i<gb_block.gb_nnzb; i++){
        elenumToblkcnt_0_255[gb_block.nnz_per_blk[i]]++;
    }
    int blk_0_8 = 0;
    int blk_sum = 0;
    for(int i=0; i<elenumToblkcnt_0_255.size(); i++){
        if(i<=8){
            blk_0_8 += elenumToblkcnt_0_255[i];
        }
        blk_sum += elenumToblkcnt_0_255[i];
    }
    gb_block.super_sparse_blocks = blk_0_8;
    gb_block.analyzed_blocks = blk_sum;
    gb_block.super_sparse_ratio = blk_sum > 0 ? blk_0_8 * 1.0 / blk_sum : 0.0;
    return gb_block.mtx_nnz >= 10 && gb_block.super_sparse_ratio > 0.15;
}

void coo2block_pre(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    int nnz = gb_block.mtx_nnz;
    std::unordered_map<int,bool> blk_nnz_cnt;
    std::unordered_map<long long, long long> blk_map;
    for(int i = 0; i < nnz; i++){
        int blk_idx = row_idx[i] / BLOCK_SIZE;
        int blk_idy = col_idx[i] / BLOCK_SIZE;
        long long blk_id = blk_idx * gb_block.gb_n + blk_idy;
        if(blk_nnz_cnt.find(blk_id)==blk_nnz_cnt.end()){
            gb_block.gb_nnzb++;
            blk_map[blk_id] = gb_block.gb_nnzb - 1;
            gb_block.gb_row_idx.push_back(blk_idx);
            gb_block.gb_col_idx.push_back(blk_idy);
            gb_block.nnz_per_blk.push_back(1);
            COO_BLOCK_ORIG coo_block_orig;
            coo_block_orig.blk_nnz++;
            coo_block_orig.row_idx.push_back(IndexTypeOrig(row_idx[i] - blk_idx * BLOCK_SIZE));
            coo_block_orig.col_idx.push_back(IndexTypeOrig(col_idx[i] - blk_idy * BLOCK_SIZE));
            coo_block_orig.val.push_back(ValType(val[i]));
            gb_block.data_per_coo_blk_orig.push_back(coo_block_orig);
            gb_block.type_per_blk.push_back(COO);
        }
        else{
            int nnz_blk_id = blk_map[blk_id];
            gb_block.nnz_per_blk[nnz_blk_id]++;
            gb_block.data_per_coo_blk_orig[nnz_blk_id].blk_nnz++;
            gb_block.data_per_coo_blk_orig[nnz_blk_id].row_idx.push_back(IndexTypeOrig(row_idx[i] - blk_idx * BLOCK_SIZE));
            gb_block.data_per_coo_blk_orig[nnz_blk_id].col_idx.push_back(IndexTypeOrig(col_idx[i] - blk_idy * BLOCK_SIZE));
            gb_block.data_per_coo_blk_orig[nnz_blk_id].val.push_back(ValType(val[i]));
        }
        blk_nnz_cnt[blk_id] = true;
    }
}

void coo2block_post(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    for(int i = 0; i < gb_block.gb_nnzb; i++){
        COO_BLOCK_ORIG coo_block_orig_tmp = gb_block.data_per_coo_blk_orig[i];
        if(coo_block_orig_tmp.blk_nnz < 33 ){
            COO_BLOCK_COMP coo_block_comp_tmp;
            coo_block_comp_tmp.blk_nnz = coo_block_orig_tmp.blk_nnz;
            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                IndexTypeComp idx_row = coo_block_orig_tmp.row_idx[j];
                IndexTypeComp idx_col = coo_block_orig_tmp.col_idx[j];
                IndexTypeComp idx = (idx_row << 4) | idx_col;
                coo_block_comp_tmp.idx.push_back(idx);
                coo_block_comp_tmp.val.push_back(coo_block_orig_tmp.val[j]);
            }
            gb_block.data_per_coo_blk_comp[i] = coo_block_comp_tmp;

        }else if (coo_block_orig_tmp.blk_nnz >= 240){
            gb_block.type_per_blk[i] = DENSE;
            DENSE_BLOCK dense_block_tmp;
            dense_block_tmp.blk_nnz = coo_block_orig_tmp.blk_nnz;
            dense_block_tmp.val.resize(BLOCK_SIZE*BLOCK_SIZE);
            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                IndexTypeOrig idx_row = coo_block_orig_tmp.row_idx[j];
                IndexTypeOrig idx_col = coo_block_orig_tmp.col_idx[j];
                dense_block_tmp.val[idx_row*BLOCK_SIZE + idx_col] = coo_block_orig_tmp.val[j];
            }
            gb_block.data_per_dense_blk[i] = dense_block_tmp;
        }else{
            gb_block.type_per_blk[i] = CSR;
            CSR_BLOCK csr_block_tmp;
            csr_block_tmp.blk_nnz = coo_block_orig_tmp.blk_nnz;
            csr_block_tmp.row_ptr.resize(BLOCK_SIZE+1, 0);
            csr_block_tmp.col_idx.resize(coo_block_orig_tmp.blk_nnz);
            csr_block_tmp.val.resize(coo_block_orig_tmp.blk_nnz);

            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                csr_block_tmp.row_ptr[coo_block_orig_tmp.row_idx[j]]++;
            }

            for(int j = 0, sum = 0; j < BLOCK_SIZE; j++){
                int temp = csr_block_tmp.row_ptr[j];
                csr_block_tmp.row_ptr[j] = sum;
                sum += temp;
            }
            csr_block_tmp.row_ptr[BLOCK_SIZE] = csr_block_tmp.blk_nnz;

            std::vector<IndexTypeOrig> row_start = csr_block_tmp.row_ptr;

            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                int row = coo_block_orig_tmp.row_idx[j];
                int dest = row_start[row]++;
                csr_block_tmp.col_idx[dest] = coo_block_orig_tmp.col_idx[j];
                csr_block_tmp.val[dest] = coo_block_orig_tmp.val[j];
            }
            gb_block.data_per_csr_blk[i] = csr_block_tmp;
        }
    }
}

void coo2block_banlance(GLOBAL_BLOCK &gb_block){
    vector<BLOCK_IDX> blk_idx_array(gb_block.gb_nnzb,{0,0,0});
    #pragma omp parallel for
    for(int i=0; i<gb_block.gb_nnzb; i++){
        blk_idx_array[i].blk_nnz = gb_block.nnz_per_blk[i];
        blk_idx_array[i].origin_idx = i;
    }

    int num_warps_per_block = WARP_PER_BLOCK;
    int num_blocks_before = (gb_block.gb_nnzb + num_warps_per_block - 1) / num_warps_per_block;
    vector<int> block_loads_before(num_blocks_before, 0);
    for(int i=0; i<gb_block.gb_nnzb; i++){
        block_loads_before[i/num_warps_per_block] += gb_block.nnz_per_blk[i];
    }
    double load_sum_before = 0.0;
    for(int load : block_loads_before){
        load_sum_before += load;
    }
    double load_mean_before = load_sum_before / block_loads_before.size();
    double load_variance_before = 0.0;
    for(int load : block_loads_before){
        double difference = load - load_mean_before;
        load_variance_before += difference * difference;
    }
    load_variance_before /= block_loads_before.size();
    double load_stddev_before = std::sqrt(load_variance_before);

    std::sort(std::execution::par, blk_idx_array.begin(), blk_idx_array.end(), cmp_nnz);

    int num_blocks = 0;
    if(gb_block.gb_nnzb % 8 != 0) {
        num_blocks = gb_block.gb_nnzb / 8 + 1;
        int pad_count = 8 - gb_block.gb_nnzb % 8;
        for(int i=0; i<pad_count; i++){
            blk_idx_array.push_back({0,INT_MAX,-1});
        }
        gb_block.gb_nnzb += pad_count;
    } else {
        num_blocks = gb_block.gb_nnzb / 8;
    }

    struct PQItem {
        int total;
        int index;
        int assigned_count;
        bool operator>(const PQItem &other) const {
            if (total == other.total) return index > other.index;
            return total > other.total;
        }
    };

    priority_queue<PQItem, vector<PQItem>, greater<PQItem>> pq;
    for (int j = 0; j < num_blocks; j++) {
        pq.push({0, j, 0});
    }

    for (int i = 0; i < gb_block.gb_nnzb; i++) {
        PQItem top_item = pq.top();
        pq.pop();
        blk_idx_array[i].end_idx = top_item.index * num_warps_per_block + top_item.assigned_count;
        top_item.total += blk_idx_array[i].blk_nnz;
        top_item.assigned_count += 1;
        if (top_item.assigned_count < num_warps_per_block) {
            pq.push(top_item);
        }
    }

    std::sort(std::execution::par, blk_idx_array.begin(), blk_idx_array.end(), cmp_end_idx);

    vector<int> gb_row_idx_tmp(gb_block.gb_nnzb);
    vector<int> gb_col_idx_tmp(gb_block.gb_nnzb);
    vector<int> nnz_per_blk_tmp(gb_block.gb_nnzb);
    vector<BlockType> type_per_blk_tmp(gb_block.gb_nnzb);

    #pragma omp parallel for
    for(int i=0; i<gb_block.gb_nnzb; i++){
        int origin_idx = blk_idx_array[i].origin_idx;
        if(origin_idx != INT_MAX){
            gb_row_idx_tmp[i] = gb_block.gb_row_idx[origin_idx];
            gb_col_idx_tmp[i] = gb_block.gb_col_idx[origin_idx];
            nnz_per_blk_tmp[i] = gb_block.nnz_per_blk[origin_idx];
            type_per_blk_tmp[i] = gb_block.type_per_blk[origin_idx];
        } else {
            gb_row_idx_tmp[i] = -1;
            gb_col_idx_tmp[i] = -1;
            nnz_per_blk_tmp[i] = 0;
            type_per_blk_tmp[i] = COO;
        }
    }

    vector<int> block_loads_after(num_blocks, 0);
    for(int i=0; i<gb_block.gb_nnzb; i++){
        block_loads_after[i/num_warps_per_block] += blk_idx_array[i].blk_nnz;
    }
    double load_sum_after = 0.0;
    for(int load : block_loads_after){
        load_sum_after += load;
    }
    double load_mean_after = load_sum_after / block_loads_after.size();
    double load_variance_after = 0.0;
    for(int load : block_loads_after){
        double difference = load - load_mean_after;
        load_variance_after += difference * difference;
    }
    load_variance_after /= block_loads_after.size();
    double load_stddev_after = std::sqrt(load_variance_after);
    gb_block.load_balanced = true;
    gb_block.load_stddev_before = load_stddev_before;
    gb_block.load_stddev_after = load_stddev_after;

    gb_block.gb_row_idx = move(gb_row_idx_tmp);
    gb_block.gb_col_idx = move(gb_col_idx_tmp);
    gb_block.nnz_per_blk = move(nnz_per_blk_tmp);
    gb_block.type_per_blk = move(type_per_blk_tmp);
    gb_block.block_idx_array = move(blk_idx_array);

}


void coo2block_comp(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    coo2block_pre(row_idx, col_idx, val, gb_block);
    coo2block_post(row_idx, col_idx, val, gb_block);
    if(gb_block.mtx_nnz >= 10000){
        coo2block_banlance(gb_block);
    }else{
        vector<BLOCK_IDX> blk_idx_array(gb_block.gb_nnzb,{0,0,0});
        #pragma omp parallel for
        for(int i=0; i<gb_block.gb_nnzb; i++){
            blk_idx_array[i].blk_nnz = gb_block.nnz_per_blk[i];
            blk_idx_array[i].origin_idx = i;
            blk_idx_array[i].end_idx = i;
        }
        gb_block.block_idx_array = move(blk_idx_array);
    }
}

void coo2block_gather(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    coo2block_pre(row_idx, col_idx, val, gb_block);
    if(gather_flag(gb_block)){
        int super_sparse_blocks = gb_block.super_sparse_blocks;
        int analyzed_blocks = gb_block.analyzed_blocks;
        double super_sparse_ratio = gb_block.super_sparse_ratio;
        vector<int> gather_col_idx(gb_block.mtx_nnz);
        gather(row_idx, col_idx, gather_col_idx, gb_block);
        vector<int> block_cols_offset = move(gb_block.block_cols_offset);
        vector<int> block_cols = move(gb_block.block_cols);
        gb_block.reset(gb_block.mtx_m, gb_block.mtx_n, gb_block.mtx_nnz);
        gb_block.gather_flag = true;
        gb_block.super_sparse_blocks = super_sparse_blocks;
        gb_block.analyzed_blocks = analyzed_blocks;
        gb_block.super_sparse_ratio = super_sparse_ratio;
        gb_block.block_cols_offset = move(block_cols_offset);
        gb_block.block_cols = move(block_cols);
        coo2block_pre(row_idx, gather_col_idx.data(), val, gb_block);
        coo2block_post(row_idx, gather_col_idx.data(), val, gb_block);
    }else{
        coo2block_post(row_idx, col_idx, val, gb_block);
    }
    if(gb_block.mtx_nnz >= 10000){
        coo2block_banlance(gb_block);
    }else{
        vector<BLOCK_IDX> blk_idx_array(gb_block.gb_nnzb,{0,0,0});
        #pragma omp parallel for
        for(int i=0; i<gb_block.gb_nnzb; i++){
            blk_idx_array[i].blk_nnz = gb_block.nnz_per_blk[i];
            blk_idx_array[i].origin_idx = i;
            blk_idx_array[i].end_idx = i;
        }
        gb_block.block_idx_array = move(blk_idx_array);
    }
}

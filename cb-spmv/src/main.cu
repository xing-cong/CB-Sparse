#include <random>

#include "utils.h"
#include "io/mmio_highlevel.h"
#include "coo2block.h"
#include "cb-spmv.cuh"

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <absolute_matrix_path>" << endl;
        return 1;
    }

    const string mtx_path = argv[1];
    if (mtx_path.empty() || mtx_path.front() != '/')
    {
        Error("Matrix path must be absolute: " + mtx_path);
        return 1;
    }

    const size_t separator = mtx_path.find_last_of('/');
    const string mtx_name = mtx_path.substr(separator + 1);

    CHECK_CUDA(cudaSetDevice(0));

    string mm_type;
    int mtx_m, mtx_n, mtx_nnz;
    if (read_matrix_info(mtx_path, mm_type, mtx_m, mtx_n, mtx_nnz) != 0)
    {
        Error("Error in reading matrix info!");
        return 1;
    }

    vector<int> mtx_row_idx;
    vector<int> mtx_col_idx;
    vector<ValType> mtx_val;

    if (read_matrix_coo_data(mtx_path, mtx_row_idx, mtx_col_idx, mtx_val) != 0)
    {
        Error("Error in reading matrix data!");
        return 1;
    }
    mtx_nnz = mtx_row_idx.size();

    constexpr unsigned int random_seed = 42;
    mt19937 random_engine(random_seed);
    uniform_real_distribution<ValType> distribution(0.0, 1.0);
    vector<ValType> x(mtx_n);
    for (ValType &value : x)
    {
        value = distribution(random_engine);
    }

    vector<ValType> y(mtx_m, 0.0);
    GLOBAL_BLOCK gb_block_gather(mtx_m, mtx_n, mtx_nnz);
    coo2block_gather(mtx_row_idx.data(), mtx_col_idx.data(), mtx_val.data(), gb_block_gather);
    PROGRAM_METRICS spmv_metrics = cb_spmv(gb_block_gather, x.data(), y.data());
    bool result_is_correct = check_results_consistency(
        mtx_row_idx,
        mtx_col_idx,
        mtx_val,
        x,
        y);
    print_benchmark_report(
        mtx_name,
        mtx_m,
        mtx_n,
        mtx_nnz,
        gb_block_gather,
        spmv_metrics,
        result_is_correct);
    // writeCSV("cb_spmv.csv", spmv_metrics, mtx_name, mtx_nnz);
    CHECK_CUDA(cudaDeviceReset());
    return result_is_correct ? 0 : 1;
}

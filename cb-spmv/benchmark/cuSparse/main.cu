#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "io/mmio_highlevel.h"
#include "macros.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <absolute_matrix_path>\n";
        return EXIT_FAILURE;
    }

    const std::string matrix_path = argv[1];
    if (matrix_path.empty() || matrix_path.front() != '/') {
        Error("Matrix path must be absolute: " + matrix_path);
        return EXIT_FAILURE;
    }

    const std::size_t separator = matrix_path.find_last_of('/');
    const std::string matrix_name = matrix_path.substr(separator + 1);

    CsrMatrix matrix;
    std::string read_error;
    if (!read_csr_matrix(matrix_path, matrix, read_error)) {
        Error(read_error);
        return EXIT_FAILURE;
    }

    if (matrix.rows < BSR_BLOCK_SIZES[0]
        || matrix.columns < BSR_BLOCK_SIZES[0]) {
        Info("Matrix size is too small, skipping");
        return EXIT_SUCCESS;
    }

    if (matrix.nonzeros() == 0) {
        Info("Matrix has no nonzero entries, skipping");
        return EXIT_SUCCESS;
    }

    int* device_csr_row_offsets = nullptr;
    int* device_csr_column_indices = nullptr;
    double* device_csr_values = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_csr_row_offsets),
        matrix.row_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_csr_column_indices),
        matrix.column_indices.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&device_csr_values),
        matrix.values.size() * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(
        device_csr_row_offsets, matrix.row_offsets.data(),
        matrix.row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        device_csr_column_indices, matrix.column_indices.data(),
        matrix.column_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        device_csr_values, matrix.values.data(),
        matrix.values.size() * sizeof(double), cudaMemcpyHostToDevice));

    cusparseHandle_t handle = nullptr;
    cusparseMatDescr_t csr_descriptor = nullptr;
    cusparseMatDescr_t bsr_descriptor = nullptr;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descriptor));
    CHECK_CUSPARSE(cusparseSetMatType(
        csr_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(
        csr_descriptor, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descriptor));
    CHECK_CUSPARSE(cusparseSetMatType(
        bsr_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(
        bsr_descriptor, CUSPARSE_INDEX_BASE_ZERO));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    const cusparseDirection_t direction = CUSPARSE_DIRECTION_ROW;

    PROGRAM_METRICS best_metrics;
    std::vector<BSR_BENCHMARK_RESULT> benchmark_results;
    int best_block_size = 0;
    std::vector<double> best_result(matrix.rows, 0.0);

    constexpr unsigned int random_seed = 42;
    std::mt19937 random_engine(random_seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::vector<double> input_x(matrix.columns);
    for (double& value : input_x) {
        value = distribution(random_engine);
    }

    for (const int block_size : BSR_BLOCK_SIZES) {
        if (block_size > matrix.rows || block_size > matrix.columns) {
            continue;
        }

        const int block_rows =
            (matrix.rows + block_size - 1) / block_size;
        const int block_columns =
            (matrix.columns + block_size - 1) / block_size;
        const int padded_columns = block_columns * block_size;
        const int padded_rows = block_rows * block_size;

        std::vector<double> host_x(padded_columns, 0.0);
        std::copy(input_x.begin(), input_x.end(), host_x.begin());

        double* device_x = nullptr;
        double* device_y = nullptr;
        int* device_bsr_row_offsets = nullptr;
        int* device_bsr_column_indices = nullptr;
        double* device_bsr_values = nullptr;

        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&device_x),
            host_x.size() * sizeof(double)));
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&device_y),
            padded_rows * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(
            device_x, host_x.data(),
            host_x.size() * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(
            device_y, 0, padded_rows * sizeof(double)));

        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&device_bsr_row_offsets),
            (block_rows + 1) * sizeof(int)));

        int nonzero_blocks = 0;
        CHECK_CUSPARSE(cusparseXcsr2bsrNnz(
            handle, direction, matrix.rows, matrix.columns,
            csr_descriptor, device_csr_row_offsets,
            device_csr_column_indices, block_size,
            bsr_descriptor, device_bsr_row_offsets,
            &nonzero_blocks));

        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&device_bsr_column_indices),
            nonzero_blocks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void**>(&device_bsr_values),
            static_cast<std::size_t>(nonzero_blocks)
                * block_size * block_size * sizeof(double)));

        CHECK_CUSPARSE(cusparseDcsr2bsr(
            handle, direction, matrix.rows, matrix.columns,
            csr_descriptor, device_csr_values, device_csr_row_offsets,
            device_csr_column_indices, block_size, bsr_descriptor,
            device_bsr_values, device_bsr_row_offsets,
            device_bsr_column_indices));

        for (int iteration = 0;
             iteration < WARMUP_ITERATIONS;
             ++iteration) {
            CHECK_CUSPARSE(cusparseDbsrmv(
                handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
                block_rows, block_columns, nonzero_blocks, &alpha,
                bsr_descriptor, device_bsr_values,
                device_bsr_row_offsets, device_bsr_column_indices,
                block_size, device_x, &beta, device_y));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            CHECK_CUSPARSE(cusparseDbsrmv(
                handle, direction, CUSPARSE_OPERATION_NON_TRANSPOSE,
                block_rows, block_columns, nonzero_blocks, &alpha,
                bsr_descriptor, device_bsr_values,
                device_bsr_row_offsets, device_bsr_column_indices,
                block_size, device_x, &beta, device_y));
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_time = 0.0F;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        const double computation_time = elapsed_time / ITERATIONS;
        const double gflops =
            2.0 * matrix.nonzeros() * 1.0e-6 / computation_time;

        BSR_BENCHMARK_RESULT benchmark_result;
        benchmark_result.block_size = block_size;
        benchmark_result.metrics.comp_time = computation_time;
        benchmark_result.metrics.gflops = gflops;
        benchmark_results.push_back(benchmark_result);

        if (best_block_size == 0 || gflops > best_metrics.gflops) {
            best_block_size = block_size;
            best_metrics.comp_time = computation_time;
            best_metrics.gflops = gflops;
            CHECK_CUDA(cudaMemcpy(
                best_result.data(), device_y,
                best_result.size() * sizeof(double),
                cudaMemcpyDeviceToHost));
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(device_bsr_values));
        CHECK_CUDA(cudaFree(device_bsr_row_offsets));
        CHECK_CUDA(cudaFree(device_bsr_column_indices));
        CHECK_CUDA(cudaFree(device_x));
        CHECK_CUDA(cudaFree(device_y));
    }

    CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descriptor));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descriptor));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA(cudaFree(device_csr_row_offsets));
    CHECK_CUDA(cudaFree(device_csr_column_indices));
    CHECK_CUDA(cudaFree(device_csr_values));

    if (best_block_size == 0) {
        Error("No valid BSR block size for this matrix");
        return EXIT_FAILURE;
    }

    const bool result_is_correct = check_results_consistency(
        matrix.rows, matrix.row_offsets, matrix.column_indices,
        matrix.values, input_x, best_result);

    print_benchmark_report(
        matrix_name, matrix.rows, matrix.columns, matrix.nonzeros(),
        benchmark_results, best_block_size, result_is_correct);

    // writeCSV("cusparse_spmv.csv", best_metrics, matrix_name, matrix.nonzeros());

    return result_is_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

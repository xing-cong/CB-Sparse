#pragma once

#include <cmath>
#include <iomanip>

#include "format.h"

using namespace std;

inline void Info(const string& message)
{
    cout << BLUE << "[Info]" << RESET << ": " << message << '\n';
}

inline void Error(const string& message)
{
    cerr << RED << "[Error]" << RESET << ": " << message << '\n';
}

inline void Time(const string& message)
{
    cout << GREEN << "[Time]" << RESET << ": " << message << '\n';
}

void coo_to_csr(int m, int n, int nnz, int *coo_row, int *coo_col, double *coo_val, int *&csr_row_ptr, int *&csr_col_ind, double *&csr_val)
{
    csr_row_ptr = new int[m + 1]();
    csr_col_ind = new int[nnz];
    csr_val = new double[nnz];

    for (int i = 0; i < nnz; ++i)
    {
        csr_row_ptr[coo_row[i]]++;
    }

    for (int i = 0, sum = 0; i < m; ++i)
    {
        int temp = csr_row_ptr[i];
        csr_row_ptr[i] = sum;
        sum += temp;
    }
    csr_row_ptr[m] = nnz;

    for (int i = 0; i < nnz; ++i)
    {
        int row = coo_row[i];
        int dest = csr_row_ptr[row];

        csr_col_ind[dest] = coo_col[i];
        csr_val[dest] = coo_val[i];

        csr_row_ptr[row]++;
    }

    for (int i = m; i > 0; --i)
    {
        csr_row_ptr[i] = csr_row_ptr[i - 1];
    }
    csr_row_ptr[0] = 0;
}

void coo_to_ell(int m, int n, int nnz, int *coo_row, int *coo_col, double *coo_val, int *&ell_col_ind, double *&ell_val)
{
    
    ell_col_ind = new int[m]();
    ell_val = new double[m]();

    for (int i = 0; i < m; ++i)
    {
        ell_col_ind[i] = 0;
        ell_val[i] = 0.0;
    }

    for (int i = 0; i < nnz; ++i)
    {
        int row = coo_row[i];
        int col = coo_col[i];
        double value = coo_val[i];
        ell_col_ind[row] = col;
        ell_val[row] = value;
    }

}

inline bool check_results_consistency(
    const vector<int>& row_index,
    const vector<int>& col_index,
    const vector<ValType>& val,
    const vector<ValType>& x,
    const vector<ValType>& y_test)
{
    vector<ValType> y(y_test.size(), 0.0);
    for (size_t i = 0; i < val.size(); i++)
    {
        int row = row_index[i];
        int col = col_index[i];
        ValType value = val[i];
        y[row] += value * x[col];
    }

    constexpr ValType absolute_tolerance = 1e-3;
    constexpr ValType relative_tolerance = 1e-4;
    for (size_t i = 0; i < y_test.size(); i++)
    {
        const ValType absolute_error = std::abs(y[i] - y_test[i]);
        const ValType allowed_error =
            absolute_tolerance + relative_tolerance * std::abs(y[i]);
        if (absolute_error > allowed_error)
        {
            Error("Result differs at index " + to_string(i)
                  + ": expected " + to_string(y[i])
                  + ", got " + to_string(y_test[i])
                  + ", absolute error " + to_string(absolute_error)
                  + ", tolerance " + to_string(allowed_error));
            return false;
        }
    }
    return true;
}

size_t get_memory_size(size_t size, size_t bmab){
    if(size % bmab == 0){   
        return size;
    }else{
        return (size / bmab + 1) * bmab;
    }
}


inline void print_benchmark_report(
    const string& matrix_name,
    int rows,
    int columns,
    int nonzeros,
    const GLOBAL_BLOCK& block,
    const PROGRAM_METRICS& metrics,
    bool result_is_correct)
{
    const string divider(64, '=');
    const string separator(64, '-');

    cout << '\n'
         << divider << '\n'
         << "                       CB-SpMV Benchmark\n"
         << divider << '\n'
         << left
         << setw(24) << "Matrix" << ": " << matrix_name << '\n'
         << setw(24) << "Dimensions" << ": "
         << rows << " x " << columns << '\n'
         << setw(24) << "Nonzeros" << ": " << nonzeros << '\n'
         << setw(24) << "Sparse blocks (nnz<8)" << ": "
         << block.super_sparse_blocks << " / " << block.analyzed_blocks
         << " (" << fixed << setprecision(6)
         << block.super_sparse_ratio * 100.0 << "%)\n"
         << setw(24) << "Column gather" << ": "
         << (block.gather_flag ? "YES" : "NO") << '\n'
         << setw(24) << "Load balance" << ": "
         << (block.load_balanced ? "YES" : "NO") << '\n'
         << setw(24) << "Load stddev" << ": ";

    if (block.load_balanced)
    {
        cout << block.load_stddev_before << " -> " << block.load_stddev_after << '\n';
    }
    else
    {
        cout << "N/A -> N/A\n";
    }

    cout << separator << '\n'
         << left << setw(14) << "Method"
         << right << setw(25) << "Time (ms)"
         << setw(25) << "GFLOPS" << '\n'
         << separator << '\n'
         << left << setw(14) << "CB-SpMV"
         << right << fixed << setprecision(6)
         << setw(25) << metrics.runtime
         << setw(25) << metrics.gflops << '\n'
         << separator << '\n'
         << left << setw(14) << "Validation" << ": "
         << (result_is_correct ? GREEN : RED)
         << (result_is_correct ? "PASS" : "FAIL")
         << RESET << '\n'
         << divider << '\n';
}

void print_cuda_device_properties(int device_id = 0) {

    CHECK_CUDA(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %s\n", "Property", "Value");
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d\n", "Device id", device_id);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %s\n", "Device name", prop.name);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d.%d\n", "Compute capability", prop.major, prop.minor);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %g GB\n", "Amount of global memory", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %g KB\n", "Amount of constant memory", prop.totalConstMem / 1024.0);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d %d %d\n", "Maximum grid size", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d %d %d\n", "Maximum block size", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d\n", "Number of SMs", prop.multiProcessorCount);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %g KB\n", "Maximum amount of shared memory per block", prop.sharedMemPerBlock / 1024.0);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %g KB\n", "Maximum amount of shared memory per SM", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d K\n", "Maximum number of registers per block", prop.regsPerBlock / 1024);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d K\n", "Maximum number of registers per SM", prop.regsPerMultiprocessor / 1024);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d\n", "Maximum number of threads per block", prop.maxThreadsPerBlock);
    printf("+--------------------------------------------+-------------------------+ \n");
    printf("  %-42s |  %d\n", "Maximum number of threads per SM", prop.maxThreadsPerMultiProcessor);
    printf("+--------------------------------------------+-------------------------+ \n");

}

template <typename T>
void printHostArray(const string& name, const vector<T>& data) {
    cout<<"name: "<<name<<endl;
    for (const auto& val : data) {
        cout << val << " ";
    }
    cout << endl;
}

template <typename T>
void printDeviceArray(const string& name, T* d_data, int size) {
    vector<T> data(size);
    cudaMemcpy(data.data(), d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
    printHostArray(name, data);
}

void writeCSV(const string& filename, const PROGRAM_METRICS& metrics, const string& mtx_name, const int nnz) {
    
    ofstream file(filename, ios::app);

    if (file.is_open()) {
        file << mtx_name << "," << nnz << "," << metrics.runtime << "," << metrics.gflops << endl;
        file.close();
        Info("Data written to " + filename + " successfully.");
    } else {
        Error("Failed to open " + filename);
    }

}

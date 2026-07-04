#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <cusparse.h>

constexpr int BSR_BLOCK_SIZES[] = {2, 4, 8, 16};
constexpr int ITERATIONS = 1000;
constexpr int WARMUP_ITERATIONS = 100;

constexpr char COLOR_RESET[] = "\033[0m";
constexpr char COLOR_RED[] = "\033[31m";
constexpr char COLOR_GREEN[] = "\033[32m";
constexpr char COLOR_BLUE[] = "\033[34m";

inline void check_cuda(cudaError_t status, const char* expression,
                       const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::cerr << "CUDA error at " << file << ':' << line
              << " for " << expression << ": "
              << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
}

inline void check_cusparse(cusparseStatus_t status, const char* expression,
                           const char* file, int line) {
    if (status == CUSPARSE_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "cuSPARSE error at " << file << ':' << line
              << " for " << expression << ": "
              << static_cast<int>(status) << '\n';
    std::exit(EXIT_FAILURE);
}

#define CHECK_CUDA(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

#define CHECK_CUSPARSE(expression) \
    check_cusparse((expression), #expression, __FILE__, __LINE__)

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <assert.h>
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <execution>
#include <queue>

#define CHECK_CUDA(call)                                                    \
do                                                                          \
{                                                                           \
    const cudaError_t error_code = call;                                    \
    if (error_code != cudaSuccess)                                          \
    {                                                                       \
        printf("CUDA Error:\n");                                            \
        printf("     - File: %s\n", __FILE__);                              \
        printf("     - Line: %d\n", __LINE__);                              \
        printf("     - Error code: %d\n", error_code);                      \
        printf("     - Error text: %s\n", cudaGetErrorString(error_code));  \
        cudaDeviceReset();                                                  \
        exit(1);                                                            \
    }                                                                       \
} while (0)

#define CHECK_CUSPARSE(call)                                                   \
    {                                                                          \
        cusparseStatus_t err = call;                                           \
        if (err != CUSPARSE_STATUS_SUCCESS) {                                  \
            cerr << "cuSPARSE error: " << err << endl;                         \
            exit(1);                                                           \
        }                                                                      \
    }

#define ITER 10

typedef uint8_t IndexTypeComp;
typedef uint8_t IndexTypeOrig;

typedef double ValType;

#define BLOCK_SIZE 16
#define WARP_PER_BLOCK 8
#define WARP_SIZE 32

typedef uint8_t BlockType; 
#define COO 0
#define CSR 1
#define ELL 2
#define DENSE 3

#ifndef mask_0_3
#define mask_0_3 0b1111
#endif

#ifndef mask_4_7
#define mask_4_7 0b11110000
#endif

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"

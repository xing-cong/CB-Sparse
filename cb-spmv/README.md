# CB-SpMV

CUDA implementation and benchmark driver for **CB-SpMV**. The repository contains the CB-SpMV implementation and a cuSPARSE BSR SpMV baseline.

## Repository contents

| Component | Source | Run script | Executable |
| --- | --- | --- | --- |
| CB-SpMV | [`src/`](src/) | [`run/run_cbspmv.sh`](run/run_cbspmv.sh) | `src/cb-spmv` |
| cuSPARSE BSR baseline | [`benchmark/cuSparse/`](benchmark/cuSparse/) | [`run/run_cusparse.sh`](run/run_cusparse.sh) | `benchmark/cuSparse/cusparse_spmv` |

The cuSPARSE baseline converts each input matrix from CSR to BSR, benchmarks block sizes 2, 4, 8, and 16, and reports the best-performing block size.

## Requirements

- An NVIDIA GPU
- CUDA Toolkit, including `nvcc`, the CUDA runtime, and cuSPARSE
- GNU Make
- Bash and `realpath`
- A host C++ compiler with OpenMP support

The code has been tested with the following environment:

- NVIDIA RTX 4090 (Ada, compute capability `sm_89`)
- CUDA Toolkit 12.4
- NVIDIA driver 550.144.03

Both Makefiles target `sm_89` by default. For an A100, change `-arch=sm_89` to `-arch=sm_80` in:

- [`src/Makefile`](src/Makefile)
- [`benchmark/cuSparse/Makefile`](benchmark/cuSparse/Makefile)

For other NVIDIA GPUs, use the architecture flag matching the GPU's compute capability.

## Quick start

Run the following commands from the repository root:

```bash
# Build CB-SpMV and run the included example matrices
./run/run_cbspmv.sh

# Build the cuSPARSE baseline and run the included example matrices
./run/run_cusparse.sh
```

The scripts build their corresponding executables and run the three matrices included in [`data/`](data/) serially:

- `1138_bus.mtx`
- `heart2.mtx`
- `sme3Da.mtx`

## Running another matrix

To run another Matrix Market (`.mtx`) matrix, enter the corresponding source directory, build the executable, and pass the matrix's **absolute path**.

CB-SpMV:

```bash
cd src
make
./cb-spmv /absolute/path/to/example.mtx
```

cuSPARSE:

```bash
cd benchmark/cuSparse
make
./cusparse_spmv /absolute/path/to/example.mtx
```

To remove the generated executables from the repository root:

```bash
make -C src clean
make -C benchmark/cuSparse clean
```

Each program prints the matrix dimensions, number of nonzeros, average kernel time, throughput in GFLOPS, and a validation result. CB-SpMV also reports its block sparsity, column-gather, and load-balancing statistics.

Example cuSPARSE output:

```text
================================================================
                 cuSPARSE BSR SpMV Benchmark
================================================================
Matrix        : sme3Da.mtx
Dimensions    : 12504 x 12504
Nonzeros      : 874887
----------------------------------------------------------------
Block size             Time (ms)            GFLOPS     Selection
----------------------------------------------------------------
2                       0.106770         16.388189          BEST
4                       0.235286          7.436789
8                       0.395750          4.421408
16                      0.670189          2.610865
----------------------------------------------------------------
Best result   : block 2, 0.106770 ms, 16.388189 GFLOPS
Validation    : PASS
================================================================
```

Example CB-SpMV output:

```text
================================================================
                       CB-SpMV Benchmark
================================================================
Matrix                  : sme3Da.mtx
Dimensions              : 12504 x 12504
Nonzeros                : 874887
Sparse blocks (nnz<8)   : 263568 / 279551 (94.282617%)
Column gather           : YES
Load balance            : YES
Load stddev             : 32.504051 -> 1.585464
----------------------------------------------------------------
Method                        Time (ms)                   GFLOPS
----------------------------------------------------------------
CB-SpMV                        0.018432                94.931314
----------------------------------------------------------------
Validation    : PASS
================================================================
```

## Project structure

```text
cb-spmv-tmp/
├── src/                  # CB-SpMV implementation and Makefile
├── benchmark/
│   └── cuSparse/         # cuSPARSE BSR SpMV baseline
├── data/                 # Matrix Market input matrices
├── run/
│   ├── run_cbspmv.sh
│   └── run_cusparse.sh
├── .gitignore
└── README.md
```

## Acknowledgements

We gratefully acknowledge:

- [NVIDIA cuSPARSE and the CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuSPARSE) for the benchmark comparison.
- The [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) for providing the sparse matrices used in the benchmarks.

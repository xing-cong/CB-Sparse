# Code Guide for CB-Sparse

Each method in CB-Sparse is maintained in its own directory. Select the operation you want to evaluate and follow the instructions in the corresponding directory.

| Method | Status | Source and instructions |
| --- | --- | --- |
| CB-SpMV | Available | [`cb-spmv/`](cb-spmv/) · [`README`](cb-spmv/README.md) |
| CB-SpMM | Coming soon | — |
| CB-SpGEMM | Coming soon | — |

## CB-SpMV

The [`cb-spmv/`](cb-spmv/) directory contains the CUDA implementation of CB-SpMV, example matrices, runnable scripts, and a cuSPARSE BSR baseline. See the [CB-SpMV README](cb-spmv/README.md) for the complete environment and usage documentation.

### Quick start

From the root of this repository, run:

```bash
cd cb-spmv

# Build CB-SpMV and run all example matrices in data/
./run/run_cbspmv.sh

# Build the cuSPARSE BSR baseline and run the same matrices
./run/run_cusparse.sh
```

The scripts compile the corresponding implementation and run the included Matrix Market (`.mtx`) matrices serially.

### Run a custom matrix

CB-SpMV accepts an absolute path to a Matrix Market file:

```bash
cd cb-spmv/src
make
./cb-spmv /absolute/path/to/matrix.mtx
```

To run the cuSPARSE BSR baseline on the same input:

```bash
cd cb-spmv/benchmark/cuSparse
make
./cusparse_spmv /absolute/path/to/matrix.mtx
```

Both implementations target NVIDIA `sm_89` GPUs by default. If you use a different GPU architecture, update the `-arch` option in [`cb-spmv/src/Makefile`](cb-spmv/src/Makefile) and [`cb-spmv/benchmark/cuSparse/Makefile`](cb-spmv/benchmark/cuSparse/Makefile) before compiling.

## CB-SpMM

The source code and execution guide will be released after the corresponding work is available.

## CB-SpGEMM

The source code and execution guide will be released after the corresponding work is available.

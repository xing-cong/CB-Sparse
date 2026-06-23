# CB-Sparse

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/CB-Sparse.png" width="300"></img>
</div>

**CB-Sparse** is a cache-friendly sparse computing framework for GPUs that unifies high-performance implementations of **SpMV**, **SpMM**, and **SpGEMM**, designed for irregular sparse workloads in HPC.

---

## OverView

![OverView](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/202605172325225.png)

**CB-Sparse** is a unified GPU sparse computing framework that focuses on improving cache locality, hardware utilization, and workload balance for irregular sparse matrix operations. Built upon a novel blocked sparse representation and adaptive execution strategy, CB-Sparse efficiently supports sparse matrix-vector multiplication (SpMV), sparse matrix-matrix multiplication (SpMM), and sparse matrix-sparse matrix multiplication (SpGEMM). By leveraging intra-block data aggregation, adaptive sparse formats, and GPU-aware scheduling, CB-Sparse significantly improves memory efficiency and parallel execution on modern NVIDIA GPUs.

---

## Motivation: Cache-Friendly Sparse Computing

![Motivation](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/202605172323625.png)

Traditional sparse formats such as CSR and COO introduce irregular memory accesses that reduce GPU cache efficiency and underutilize warp-level parallelism. CB-Sparse addresses these limitations through a cache-friendly blocked sparse structure and adaptive sparse execution framework optimized for modern GPU architectures.

---

## CB-SpMV

### Paper (ICS'25)

<div align="center">
    <h3>
      CB-SpMV:A Data Aggregating and Balance Algorithm for Cache-Friendly Block-Based SpMV on GPUs
    </h3>
    <p>
        Xing Cong, Fukai Sun, Yifan Chen, Chenhao Xie, Yi Liu, Depei Qian
    </p>
    <p></p>
</div>

<div align="center">
  <a href="https://dl.acm.org/doi/10.1145/3721145.3725746"><b>[Paper]</b></a> | 
  <a href="https://dl.acm.org/doi/10.1145/3721145.3725746"><b>[Slides]</b></a> | 
  <a href="https://github.com/xing-cong/CB-Sparse/blob/master/Code-Guide.md"><b>[Code Guide]</b></a>
</div>


### Method

**CB-SpMV** is a cache-friendly block-based sparse matrix-vector multiplication framework designed for irregular sparse workloads on GPUs. The method partitions sparse matrices into independent 16×16 sub-blocks and organizes them using a unified 2D sparse structure to improve memory locality and warp-level parallelism.

![Method](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/202605172328823.png)

To reduce scattered memory accesses, CB-SpMV introduces an intra-block data aggregation mechanism that tightly packs sparse metadata and values into contiguous memory regions through coordinate compression and virtual-pointer-based layouts. The framework further applies adaptive sparse execution strategies by dynamically selecting COO, CSR, or Dense block formats according to sub-block sparsity characteristics. For highly sparse blocks, a block-aware column aggregation strategy improves GPU thread utilization, while dense blocks leverage warp-level reductions for efficient computation.

To address irregular workload distribution, CB-SpMV also introduces an inter-thread-block load balancing strategy that redistributes sparse blocks across GPU thread blocks, improving SM utilization and execution efficiency.

### Performance

CB-SpMV achieves significant acceleration across large-scale sparse matrices on modern NVIDIA GPUs including A100 and RTX 4090. Compared with state-of-the-art sparse computing methods such as cuSPARSE-BSR, TileSpMV, and DASP, CB-SpMV substantially improves cache locality, hardware utilization, and workload balance.

![Performance](https://cdn.jsdelivr.net/gh/NEUQer-xing/Markdown_images@master/images-2/202605172329283.png)

Experimental results demonstrate that CB-SpMV achieves higher L1/L2 cache hit rates and delivers strong end-to-end speedups across diverse sparsity patterns and matrix structures. The framework particularly excels on highly irregular sparse matrices where traditional sparse formats suffer from severe memory divergence and GPU underutilization.

---

## CB-SpMM and CB-SpGEMM

Will coming soon! (Under review)

---


## Citation

If you find `CB-Sparse(CB-SpMV)` useful or relevant to your research, please cite our paper:

```bibtex
@inproceedings{cong2025cb-spmv,
  title={CB-SpMV: A Data Aggregating and Balance Algorithm for for Cache-Friendly Block-Based SpMV on GPUs},
  author={Cong, Xing and Sun, FuKai and Chen, YiFan and Xie, Chenhao and Liu, Yi and Qian, Depei},
  booktitle={Proceedings of the 39th ACM International Conference on Supercomputing},
  pages={149--160},
  year={2025}
}
```

---

<!-- # Star History -->

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=xxx/CB-Sparse)](https://www.star-history.com/#xxx/CB-Sparse) -->


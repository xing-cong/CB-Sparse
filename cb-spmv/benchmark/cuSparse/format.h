#pragma once

struct PROGRAM_METRICS {
    double comp_time = 0.0;
    double gflops = 0.0;
};

struct BSR_BENCHMARK_RESULT {
    int block_size = 0;
    PROGRAM_METRICS metrics;
};

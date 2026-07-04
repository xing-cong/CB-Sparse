#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "format.h"
#include "macros.h"

inline void Info(const std::string& message) {
    std::cout << COLOR_BLUE << "[Info]" << COLOR_RESET << ": " << message << '\n';
}

inline void Error(const std::string& message) {
    std::cerr << COLOR_RED << "[Error]" << COLOR_RESET << ": " << message << '\n';
}

inline void Time(const std::string& message) {
    std::cout << COLOR_GREEN << "[Time]" << COLOR_RESET << ": " << message << '\n';
}

inline bool check_results_consistency(
    int rows,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_indices,
    const std::vector<double>& values,
    const std::vector<double>& x,
    const std::vector<double>& actual) {
    std::vector<double> expected(rows, 0.0);

    for (int row = 0; row < rows; ++row) {
        for (int index = row_offsets[row]; index < row_offsets[row + 1]; ++index) {
            expected[row] += values[index] * x[column_indices[index]];
        }
    }

    for (int row = 0; row < rows; ++row) {
        if (std::abs(expected[row] - actual[row]) > 1e-3) {
            Error("Result differs at index " + std::to_string(row)
                  + ": expected " + std::to_string(expected[row])
                  + ", got " + std::to_string(actual[row]));
            return false;
        }
    }

    return true;
}

inline void print_benchmark_report(
    const std::string& matrix_name,
    int rows,
    int columns,
    int nonzeros,
    const std::vector<BSR_BENCHMARK_RESULT>& results,
    int best_block_size,
    bool result_is_correct) {
    const std::string divider(64, '=');
    const std::string separator(64, '-');

    std::cout << '\n'
              << divider << '\n'
              << "                 cuSPARSE BSR SpMV Benchmark\n"
              << divider << '\n'
              << std::left
              << std::setw(14) << "Matrix" << ": " << matrix_name << '\n'
              << std::setw(14) << "Dimensions" << ": "
              << rows << " x " << columns << '\n'
              << std::setw(14) << "Nonzeros" << ": " << nonzeros << '\n'
              << separator << '\n'
              << std::left << std::setw(14) << "Block size"
              << std::right << std::setw(18) << "Time (ms)"
              << std::setw(18) << "GFLOPS"
              << std::setw(14) << "Selection" << '\n'
              << separator << '\n';

    const BSR_BENCHMARK_RESULT* best_result = nullptr;
    for (const BSR_BENCHMARK_RESULT& result : results) {
        const bool is_best = result.block_size == best_block_size;
        if (is_best) {
            best_result = &result;
        }

        std::cout << std::left << std::setw(14) << result.block_size
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(18) << result.metrics.comp_time
                  << std::setw(18) << result.metrics.gflops
                  << std::setw(14)
                  << (is_best ? "BEST" : "") << '\n';
    }

    std::cout << separator << '\n';
    if (best_result != nullptr) {
        std::cout << std::left << std::setw(14) << "Best result" << ": "
                  << "block " << best_result->block_size
                  << ", " << best_result->metrics.comp_time << " ms"
                  << ", " << best_result->metrics.gflops << " GFLOPS\n";
    }

    std::cout << std::left << std::setw(14) << "Validation" << ": "
              << (result_is_correct ? COLOR_GREEN : COLOR_RED)
              << (result_is_correct ? "PASS" : "FAIL")
              << COLOR_RESET << '\n'
              << divider << '\n';
}

inline bool writeCSV(
    const std::string& filename,
    const PROGRAM_METRICS& metrics,
    const std::string& matrix_name,
    int nonzeros) {
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        Error("Failed to open " + filename);
        return false;
    }

    file << matrix_name << ','
         << nonzeros << ','
         << metrics.comp_time << ','
         << metrics.gflops << '\n';
    Info("Data written to " + filename);
    return true;
}

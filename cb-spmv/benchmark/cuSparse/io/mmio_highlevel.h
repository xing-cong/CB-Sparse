#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct CsrMatrix {
    int rows = 0;
    int columns = 0;
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<double> values;

    int nonzeros() const {
        return static_cast<int>(values.size());
    }
};

namespace matrix_market_detail {

struct Entry {
    int row;
    int column;
};

inline std::string lowercase(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

inline bool next_data_line(std::ifstream& input, std::string& line) {
    while (std::getline(input, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos && line[first] != '%') {
            return true;
        }
    }
    return false;
}

}

inline bool read_csr_matrix(
    const std::string& path,
    CsrMatrix& matrix,
    std::string& error_message) {
    std::ifstream input(path);
    if (!input) {
        error_message = "Could not open matrix file: " + path;
        return false;
    }

    std::string line;
    if (!std::getline(input, line)) {
        error_message = "Matrix file is empty: " + path;
        return false;
    }

    std::string banner;
    std::string object;
    std::string storage;
    std::string field;
    std::string symmetry;
    std::istringstream header(line);
    if (!(header >> banner >> object >> storage >> field >> symmetry)) {
        error_message = "Invalid Matrix Market header";
        return false;
    }

    banner = matrix_market_detail::lowercase(banner);
    object = matrix_market_detail::lowercase(object);
    storage = matrix_market_detail::lowercase(storage);
    field = matrix_market_detail::lowercase(field);
    symmetry = matrix_market_detail::lowercase(symmetry);

    if (banner != "%%matrixmarket" || object != "matrix"
        || storage != "coordinate") {
        error_message = "Only Matrix Market coordinate matrices are supported";
        return false;
    }

    if (field != "real" && field != "integer" && field != "pattern") {
        error_message = "Only real, integer, and pattern matrix values are supported";
        return false;
    }

    if (symmetry != "general" && symmetry != "symmetric") {
        error_message = "Only general and symmetric matrices are supported";
        return false;
    }

    if (!matrix_market_detail::next_data_line(input, line)) {
        error_message = "Matrix dimensions are missing";
        return false;
    }

    long long rows = 0;
    long long columns = 0;
    long long reported_nonzeros = 0;
    std::istringstream dimensions(line);
    if (!(dimensions >> rows >> columns >> reported_nonzeros)
        || rows < 0 || columns < 0 || reported_nonzeros < 0
        || rows > std::numeric_limits<int>::max()
        || columns > std::numeric_limits<int>::max()
        || reported_nonzeros > std::numeric_limits<int>::max()) {
        error_message = "Invalid matrix dimensions";
        return false;
    }

    if (symmetry == "symmetric" && rows != columns) {
        error_message = "A symmetric matrix must be square";
        return false;
    }

    std::vector<matrix_market_detail::Entry> entries;
    entries.reserve(
        static_cast<std::size_t>(reported_nonzeros)
        * (symmetry == "symmetric" ? 2 : 1));

    for (long long index = 0; index < reported_nonzeros; ++index) {
        if (!matrix_market_detail::next_data_line(input, line)) {
            error_message = "Matrix contains fewer entries than reported";
            return false;
        }

        long long row = 0;
        long long column = 0;
        double ignored_value = 1.0;
        std::istringstream entry_stream(line);
        if (!(entry_stream >> row >> column)
            || (field != "pattern" && !(entry_stream >> ignored_value))) {
            error_message = "Invalid matrix entry at index " + std::to_string(index);
            return false;
        }

        if (row < 1 || row > rows || column < 1 || column > columns) {
            error_message = "Matrix entry index is out of bounds";
            return false;
        }

        const int zero_based_row = static_cast<int>(row - 1);
        const int zero_based_column = static_cast<int>(column - 1);
        entries.push_back({zero_based_row, zero_based_column});

        if (symmetry == "symmetric" && zero_based_row != zero_based_column) {
            if (entries.size() == static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
                error_message = "Expanded matrix has too many nonzero entries";
                return false;
            }
            entries.push_back({zero_based_column, zero_based_row});
        }
    }

    matrix.rows = static_cast<int>(rows);
    matrix.columns = static_cast<int>(columns);
    matrix.row_offsets.assign(matrix.rows + 1, 0);
    matrix.column_indices.resize(entries.size());
    matrix.values.assign(entries.size(), 1.0);

    for (const matrix_market_detail::Entry& entry : entries) {
        ++matrix.row_offsets[entry.row + 1];
    }

    for (int row = 0; row < matrix.rows; ++row) {
        matrix.row_offsets[row + 1] += matrix.row_offsets[row];
    }

    std::vector<int> next_offset = matrix.row_offsets;
    for (const matrix_market_detail::Entry& entry : entries) {
        const int offset = next_offset[entry.row]++;
        matrix.column_indices[offset] = entry.column;
    }

    return true;
}

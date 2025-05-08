#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <omp.h>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

vector<int> generate_matrix(int rows, int cols, int seed = 42) {
    if (rows <= 0 || cols <= 0) {
        throw runtime_error("Size of the matrix must be positive");
    }

    vector<int> matrix(rows * cols);
    mt19937 generator(seed);
    uniform_int_distribution<> distrib(-1000, 1000);

#pragma omp parallel for
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = distrib(generator);
    }

    return matrix;
}

void save_matrix_to_file(const vector<int>& matrix, int rows, int cols, const string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        throw runtime_error("Failed to open file for writing: " + filename);
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << matrix[i * cols + j] << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

vector<int> transpose_matrix(const vector<int>& matrix, int rows, int cols) {
    vector<int> transposed(cols * rows);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j * rows + i] = matrix[i * cols + j];
        }
    }

    return transposed;
}


vector<int> multiply_only(const vector<int>& A, const vector<int>& B,
    int rowsA, int colsA, int rowsB, int colsB) {
    if (colsA != rowsB) {
        throw runtime_error("Matrix dimensions do not match for multiplication");
    }

    vector<int> C(rowsA * colsB, 0);
    vector<int> Bt = transpose_matrix(B, rowsB, colsB);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            int sum = 0;
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * Bt[j * rowsB + k];
            }
            C[i * colsB + j] = sum;
        }
    }

    return C;
}

void run_tests_for_threads(int num_threads, const vector<int>& sizes, int trials) {
    string results_dir = "results_" + to_string(num_threads) + "_threads";
    string reports_dir = "reports";
    fs::create_directories(results_dir);
    fs::create_directories(reports_dir);

    string report_filename = reports_dir + "/report_" + to_string(num_threads) + "_threads.txt";
    ofstream report(report_filename);

    if (!report) {
        cerr << "Error opening report file for writing: " << report_filename << endl;
        return;
    }

    report << "Matrix Size | Average Time (ms) | Min Time | Max Time | Speedup\n";
    report << "----------------------------------------------------------------\n";

    cout << "Testing with " << num_threads << " threads...\n";

    for (int size : sizes) {
        vector<long long> execution_times;

        cout << "  Processing size: " << size << "x" << size << "... ";

        
        vector<vector<int>> matricesA, matricesB;
        for (int trial = 1; trial <= trials; ++trial) {
            matricesA.push_back(generate_matrix(size, size, trial));
            matricesB.push_back(generate_matrix(size, size, trial + 1000));
        }

        for (int trial = 1; trial <= trials; ++trial) {
            try {
                omp_set_num_threads(num_threads);

                
                auto start = high_resolution_clock::now();
                auto C = multiply_only(matricesA[trial - 1], matricesB[trial - 1], size, size, size, size);
                auto finish = high_resolution_clock::now();

                auto duration = duration_cast<milliseconds>(finish - start).count();
                execution_times.push_back(duration);

                
                string size_dir = results_dir + "/size_" + to_string(size);
                fs::create_directories(size_dir);

                save_matrix_to_file(matricesA[trial - 1], size, size, size_dir + "/A_" + to_string(trial) + ".txt");
                save_matrix_to_file(matricesB[trial - 1], size, size, size_dir + "/B_" + to_string(trial) + ".txt");
                save_matrix_to_file(C, size, size, size_dir + "/C_" + to_string(trial) + ".txt");
            }
            catch (const exception& e) {
                cerr << "Error in trial " << trial << " for size " << size << ": " << e.what() << endl;
            }
        }

        if (!execution_times.empty()) {
            long long total_time = accumulate(execution_times.begin(), execution_times.end(), 0LL);
            double avg_time = static_cast<double>(total_time) / trials;
            auto min_time = *min_element(execution_times.begin(), execution_times.end());
            auto max_time = *max_element(execution_times.begin(), execution_times.end());

            double speedup = 1.0;
            if (num_threads > 1) {
                ifstream base_report("reports/report_1_threads.txt");
                string line;
                while (getline(base_report, line)) {
                    if (line.find(to_string(size)) != string::npos) {
                        size_t pos = line.find("|");
                        pos = line.find("|", pos + 1);
                        double base_time = stod(line.substr(pos + 1, 12));
                        speedup = base_time / avg_time;
                        break;
                    }
                }
                base_report.close();
            }

            report << setw(10) << size << " | "
                << setw(12) << fixed << setprecision(2) << avg_time << " | "
                << setw(8) << min_time << " | "
                << setw(8) << max_time << " | "
                << setw(7) << fixed << setprecision(2) << speedup << "\n";
        }

        cout << "done\n";
    }

    report.close();
    cout << "Testing with " << num_threads << " threads completed.\n";
}

int main() {
    vector<int> matrix_sizes = { 100, 500, 1000 };
    vector<int> thread_counts = { 1, 2, 4, 8 };
    int num_trials = 5;

    for (int threads : thread_counts) {
        run_tests_for_threads(threads, matrix_sizes, num_trials);
    }

    ofstream summary_report("reports/summary_report.txt");
    if (summary_report) {
        summary_report << "Performance Comparison (Average Time in ms)\n";
        summary_report << "Matrix Size | 1 thread | 2 threads | 4 threads | 8 threads | Speedup (8 vs 1)\n";
        summary_report << "--------------------------------------------------------------------------------\n";

        for (int size : matrix_sizes) {
            summary_report << setw(10) << size;
            vector<double> times;

            for (int threads : thread_counts) {
                string report_filename = "reports/report_" + to_string(threads) + "_threads.txt";
                ifstream in_report(report_filename);
                string line;

                while (getline(in_report, line)) {
                    if (line.find(to_string(size)) != string::npos) {
                        size_t pos = line.find("|");
                        pos = line.find("|", pos + 1);
                        string avg_time = line.substr(pos + 1, 12);
                        summary_report << " | " << setw(9) << avg_time;
                        times.push_back(stod(avg_time));
                        break;
                    }
                }
                in_report.close();
            }

            if (!times.empty()) {
                double speedup = times[0] / times.back();
                summary_report << " | " << setw(10) << fixed << setprecision(2) << speedup << "\n";
            }
        }
        summary_report.close();
    }

    cout << "All tests completed. Results are available in the results_*_threads directories.\n";
    return 0;
}
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;


vector<vector<int>> generate_matrix(int rows, int cols) {
	if (rows <= 0 || cols <= 0) {
		throw runtime_error("Size of the matrix must be positive");
	}

	vector<vector<int>> matrix(rows, vector<int>(cols));
	random_device rd;
	mt19937 generator(rd());
	uniform_int_distribution<> distrib(-1000, 1000);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = distrib(generator);
		}
	}

	return matrix;
}


void save_matrix_to_file(const vector<vector<int>>& matrix, const string& filename) {
	ofstream outFile(filename);
	if (!outFile) {
		throw runtime_error("Failed to open file for writing: " + filename);
	}

	for (const auto& row : matrix) {
		for (const auto& elem : row) {
			outFile << elem << " ";
		}
		outFile << "\n";
	}

	outFile.close();
}


vector<vector<int>> multiply_matrices(const vector<vector<int>>& A, const vector<vector<int>>& B) {
	int rowsA = A.size(), colsA = A[0].size();
	int rowsB = B.size(), colsB = B[0].size();

	if (colsA != rowsB) {
		throw runtime_error("Matrix dimensions do not match for multiplication");
	}

	vector<vector<int>> C(rowsA, vector<int>(colsB, 0));

	for (int i = 0; i < rowsA; ++i) {
		for (int j = 0; j < colsB; ++j) {
			for (int m = 0; m < colsA; ++m) {
				C[i][j] += A[i][m] * B[m][j];
			}
		}
	}

	return C;
}

int main() {
	vector<int> sizes = { 10, 50, 100, 500, 1000, 1500, 2000, 2500 };
	ofstream report("report.txt");

	if (!report) {
		cerr << "Error opening report.txt for writing" << endl;
		return 1;
	}

	report << "Matrix Size | Execution Time (ms)\n";

	for (int size : sizes) {
		try {
			string filenameA = "matrixA" + to_string(size) + ".txt";
			string filenameB = "matrixB" + to_string(size) + ".txt";
			string filenameC = "result_matrix" + to_string(size) + ".txt";

			cout << "Processing size: " << size << "x" << size << "...\n";

			vector<vector<int>> A = generate_matrix(size, size);
			vector<vector<int>> B = generate_matrix(size, size);

			save_matrix_to_file(A, filenameA);
			save_matrix_to_file(B, filenameB);

			auto start = high_resolution_clock::now();
			vector<vector<int>> C = multiply_matrices(A, B);
			auto finish = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(finish - start);

			save_matrix_to_file(C, filenameC);

			report << setw(6) << size << " | " << setw(6) << duration.count() << "\n";

		}
		catch (const exception& e) {
			cerr << "Error: " << e.what() << endl;
		}
	}

	cout << "Processing completed.";
	report.close();
	return 0;
}
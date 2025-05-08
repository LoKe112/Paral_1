import numpy as np


def load_matrix(filename):
    try:
        return np.loadtxt(filename, dtype=int)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def verify_multiplication(size):
    try:
        print(f"Checking matrix of size {size}x{size}...")

        A = load_matrix(f"output/matrixA{size}.txt")
        B = load_matrix(f"output/matrixB{size}.txt")
        C_cpp = load_matrix(f"output/result_matrix{size}.txt")

        if A is None or B is None or C_cpp is None:
            print(f"Skipping size {size} due to missing files")
            return

        C_py = np.dot(A, B)

        match = np.array_equal(C_cpp, C_py)

        with open("output\\report.txt", "a") as report:
            report.write(
                f"Matrix Size: {size}x{size} - {'The values matched' if match else 'the values did not match'}\n")

    except Exception as e:
        print(f"Error processing size {size}: {e}")


sizes = [10, 50, 100, 500, 1000, 1500, 2000, 2500]

with open("\\output\\report.txt", "a") as report:
    report.write(
        f"\n\n Comparing the results of matrix multiplication: \n\n")


for s in sizes:
    verify_multiplication(s)
print("That is all")
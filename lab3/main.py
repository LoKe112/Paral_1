import numpy as np


def check() -> None:
    counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for i in counts:
        matrix1 = np.loadtxt(f"data/1_{i}.txt", dtype=int)
        matrix2 = np.loadtxt(f"data/2_{i}.txt", dtype=int)
        result = np.dot(matrix1, matrix2)
        cpp_result = np.loadtxt(f"data/result_{i}.txt")
        if np.array_equal(cpp_result, result):
            print(
                f"Проверка для матрицы {i} успешна: результаты совпадают"
            )
        else:
            print(" Ошибка: результаты не совпадают")



if __name__ == "__main__":
    check()


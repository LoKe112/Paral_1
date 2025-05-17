import numpy as np


def check() -> None:
    counts = [100,600,1100,1600,2100,2600,3100,3600,4100]
    for i in counts:
        matrix1 = np.loadtxt(f"results/{i}_1.txt", dtype=int)
        matrix2 = np.loadtxt(f"results/{i}_2.txt", dtype=int)
        result = np.dot(matrix1, matrix2)
        cpp_result = np.loadtxt(f"results/result_cuda_{i}.txt")
        if np.array_equal(cpp_result, result):
            print(
                f"Проверка для матрицы {i} успешна: результаты совпадают"
            )
        else:
            print(" Ошибка: результаты не совпадают")



if __name__ == "__main__":
    check()


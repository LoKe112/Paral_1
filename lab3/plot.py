import matplotlib.pyplot as plt

def read_stats(file_path='data/stats.txt'):
    sizes = []
    times_ms = []
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                size_part, time_part = line.split(':')
                size = int(size_part.strip())
                time_str = time_part.strip().replace(' ms', '').replace('ms', '').strip()
                time = float(time_str)
                sizes.append(size)
                times_ms.append(time)
    return sizes, times_ms

def plot_performance(sizes, times_ms):
    plt.figure(figsize=(10, 6))
    
    # Конвертируем время в секунды для лучшего отображения
    times_sec = [t/1000 for t in times_ms]
    
    plt.plot(sizes, times_sec, 'o-', markersize=5, linewidth=2)
    plt.xlabel('Размер матрицы (N × N)', fontsize=12)
    plt.ylabel('Время умножения (секунды)', fontsize=12)
    plt.title('Зависимость времени умножения матриц от их размера', fontsize=14)
    
    # Добавляем сетку и логарифмическую шкалу для Y
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('linear')
    
    # Форматируем оси
    plt.xticks(sizes, rotation=45)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('matrix_multiplication_performance.png', dpi=300)
    plt.show()

# Основная программа
if __name__ == '__main__':
    try:
        sizes, times_ms = read_stats()
        plot_performance(sizes, times_ms)
        print("График успешно построен и сохранен как 'matrix_multiplication_performance.png'")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
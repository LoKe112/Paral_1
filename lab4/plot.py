import matplotlib.pyplot as plt
import numpy as np
import os

def read_stats(file_path='results/statistics_cuda.txt'):
    """Чтение данных из файла статистики"""
    sizes = []
    times_ms = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Пропускаем пустые строки
            if not line.strip():
                continue
                
            # Разделяем строку по табуляции или пробелам
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    size = int(parts[0])
                    time_str = parts[1].replace('ms', '').replace('ms', '').strip()
                    time = float(time_str)
                    sizes.append(size)
                    times_ms.append(time)
                except ValueError:
                    print(f"Пропуск строки с некорректными данными: {line.strip()}")
    
    return sizes, times_ms

def plot_performance(sizes, times_ms):
    """Построение графика производительности"""
    plt.figure(figsize=(12, 7))
    
    # Основной график
    plt.plot(sizes, times_ms, 'o-', 
             color='#1f77b4', 
             markersize=8, 
             linewidth=2,
             markerfacecolor='white',
             markeredgewidth=2)
    
    # Настройка осей
    plt.xlabel('Размер матрицы (N × N)', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.title('Зависимость времени умножения матриц от размера (CUDA)', fontsize=14, pad=20)
    
    # Настройка сетки
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    
    # Форматирование оси X
    plt.xticks(np.arange(min(sizes), max(sizes)+1, 500), rotation=45)
    plt.xlim(min(sizes)-100, max(sizes)+100)
    
    # Форматирование оси Y
    y_max = max(times_ms) * 1.1
    plt.ylim(0, y_max)
    
    # Подписи точек данных
    for size, time in zip(sizes, times_ms):
        plt.annotate(f"{int(time)} ms", 
                     xy=(size, time), 
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', 
                              fc='white', 
                              alpha=0.7))
    
    # Оптимизация расположения
    plt.tight_layout()
    
    # Сохранение графика
    output_file = 'cuda_matrix_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен как: {output_file}")

if __name__ == '__main__':
    try:
        # Чтение данных
        sizes, times_ms = read_stats()
        
        if not sizes:
            print("Ошибка: не найдены данные для построения графика")
            exit(1)
            
        # Построение графика
        plot_performance(sizes, times_ms)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
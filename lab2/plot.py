import matplotlib.pyplot as plt
import numpy as np

def plot_performance():
    
    with open('output/reports/summary_report.txt', 'r') as f:
        lines = f.readlines()
    
    
    data_lines = [line.strip() for line in lines[3:] if line.strip()]
    
    
    sizes = []
    times_1 = []
    times_2 = []
    times_4 = []
    times_8 = []
    
    for line in data_lines:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5:
            try:
                sizes.append(int(parts[0]))
                times_1.append(float(parts[1])/1000 if parts[1] else 0)  
                times_2.append(float(parts[2])/1000 if parts[2] else 0)
                times_4.append(float(parts[3])/1000 if parts[3] else 0)
                times_8.append(float(parts[4])/1000 if parts[4] else 0)
            except ValueError as e:
                print(f"Пропуск строки из-за ошибки: {line}\nОшибка: {e}")
                continue
    
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Линии с разными стилями и цветами
    plt.plot(sizes, times_1, 'b-o', linewidth=2, markersize=8, label='1 поток')
    plt.plot(sizes, times_2, 'g--s', linewidth=2, markersize=8, label='2 потока')
    plt.plot(sizes, times_4, 'r-.^', linewidth=2, markersize=8, label='4 потока')
    plt.plot(sizes, times_8, 'm:d', linewidth=2, markersize=8, label='8 потоков')
    
    
    plt.title('Зависимость среднего времени выполнения перемножения матриц от размера', fontsize=14, pad=20)
    plt.xlabel('Размер матрицы', fontsize=12)
    plt.ylabel('Среднее время (сек)', fontsize=12)
    
    
    plt.ylim(0, 5)
    
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, framealpha=1)
    
    
    plt.xticks(sizes, fontsize=10)
    plt.yticks(np.arange(0, 5.5, 0.5), fontsize=10)
    
    
    plt.tight_layout()
    
    
    plt.savefig('matrix_performance.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_performance()
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_files(root_directory='results'):
    data = []
    size_re = re.compile(r"Matrix size: (\d+)x\d+")
    proc_re = re.compile(r"Processes count: (\d+)")
    time_re = re.compile(r"Calculation time: ([\d\.]+) sec")

    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith(".out"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        size_match = size_re.search(content)
                        proc_match = proc_re.search(content)
                        time_match = time_re.search(content)
                        
                        if size_match and proc_match and time_match:
                            data.append({
                                'Size': int(size_match.group(1)),
                                'Processes': int(proc_match.group(1)),
                                'Time': float(time_match.group(1))
                            })
                except Exception as e:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
                    
    return pd.DataFrame(data)

df = parse_files('results')
if df.empty:
    print("Данные не найдены. Убедитесь, что файлы .out находятся в папках внутри 'results'.")
    exit()

df = df.sort_values(by=['Size', 'Processes'])

plt.figure(figsize=(10, 6))

for size in sorted(df['Size'].unique()):
    subset = df[df['Size'] == size].dropna()
    plt.plot(subset['Processes'], subset['Time'], marker='o', label=f'{size}x{size}')

plt.title('Зависимость времени выполнения от числа ядер')
plt.xlabel('Количество ядер')
plt.ylabel('Время (секунды)')

plt.xscale('log', base=2)
all_procs = sorted(df['Processes'].unique())
plt.xticks(all_procs, all_procs)

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(title="Размер матрицы")
plt.tight_layout()

plt.savefig('lab5_plots.png')
print("График сохранен в lab5_plots.png")
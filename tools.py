import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS_TEST = [1, 2, 4]
CORES_TEST = [1, 2, 4] 
FIXED_THREADS = 4
DATA_DIR = "matrix_data"

def prepare_data(n):
    """Генерирует матрицы, если их нет"""
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    p_a, p_b = f"{DATA_DIR}/A_{n}.txt", f"{DATA_DIR}/B_{n}.txt"
    if not os.path.exists(p_a) or not os.path.exists(p_b):
        print(f"[*] Генерирую данные для N={n}...")
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        np.savetxt(p_a, a, fmt='%.5f')
        np.savetxt(p_b, b, fmt='%.5f')
    return p_a, p_b

def set_symlinks(p_a, p_b):
    """Обновляет ссылки A.txt и B.txt на конкретные файлы"""
    for f in ["A.txt", "B.txt"]:
        if os.path.exists(f): os.remove(f)
    os.symlink(p_a, "A.txt")
    os.symlink(p_b, "B.txt")

def run_cmd(args):
    """Запускает команду и вытягивает время выполнения"""
    proc = subprocess.run(args, capture_output=True, text=True)
    match = re.search(r"(\d+\.\d+)", proc.stdout)
    return float(match.group(1)) * 1000 if match else 0.0

def start_benchmarks():
    thread_results = {n: {} for n in SIZES}
    print("\n[!] Эксперимент 1: Зависимость от количества потоков...")
    for n in SIZES:
        p_a, p_b = prepare_data(n)
        set_symlinks(p_a, p_b)
        for t in THREADS_TEST:
            time_ms = run_cmd(["./lab2", str(n), str(t)])
            thread_results[n][t] = time_ms
            print(f"  N={n}, T={t} -> {time_ms:.2f} ms")

    core_results = {n: {} for n in SIZES}
    print("\n[!] Эксперимент 2: Зависимость от количества ядер (Фикс. 4 потока)...")
    for n in SIZES:
        p_a, p_b = prepare_data(n)
        set_symlinks(p_a, p_b)
        for c in CORES_TEST:
            mask = f"0-{c-1}"
            time_ms = run_cmd(["taskset", "-c", mask, "./lab2", str(n), str(FIXED_THREADS)])
            core_results[n][c] = time_ms
            print(f"  N={n}, Cores={c} -> {time_ms:.2f} ms")

    return thread_results, core_results

def print_markdown_tables(t_res, c_res):
    header1 = "| N / Потоки | " + " | ".join([f"{t} T" for t in THREADS_TEST]) + " |"
    print(header1)
    print("| " + "--- | " * (len(THREADS_TEST) + 1))
    for n in SIZES:
        print(f"| **{n}** | " + " | ".join([f"{t_res[n][t]:.2f}" for t in THREADS_TEST]) + " |")

    header2 = "| N / Ядра | " + " | ".join([f"{c} Core(s)" for c in CORES_TEST]) + " |"
    print(header2)
    print("| " + "--- | " * (len(CORES_TEST) + 1))
    for n in SIZES:
        print(f"| **{n}** | " + " | ".join([f"{c_res[n][c]:.2f}" for c in CORES_TEST]) + " |")

def plot_all(t_data, c_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for n in SIZES:
        ax1.plot(THREADS_TEST, [t_data[n][t] for t in THREADS_TEST], marker='o', label=f'N={n}')
    ax1.set_title('Зависимость от потоков')
    ax1.set_xlabel('Threads')
    ax1.set_ylabel('ms')
    ax1.grid(True)
    ax1.legend(title="Размеры", bbox_to_anchor=(1.05, 1), loc='upper left')

    for n in SIZES:
        ax2.plot(CORES_TEST, [c_data[n][c] for c in CORES_TEST], marker='s', linestyle='--', label=f'N={n}')
    ax2.set_title('Зависимость от ядер')
    ax2.set_xlabel('Cores')
    ax2.set_ylabel('ms')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('lab2_plots.png', dpi=300)
    print("\n[!] Графики со всеми размерами сохранены в lab2_plots.png")

if __name__ == "__main__":
    print("[*] Компиляция lab2_openmp.cpp...")
    subprocess.run(["g++", "-O3", "-fopenmp", "lab2_openmp.cpp", "-o", "lab2"])
    
    t_data, c_data = start_benchmarks()
    print_markdown_tables(t_data, c_data)
    plot_all(t_data, c_data)

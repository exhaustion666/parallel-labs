import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
CORES_TEST = [1, 2, 4] 
DATA_DIR = "matrix_data"

def prepare_data(n):
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
    for f in ["A.txt", "B.txt"]:
        if os.path.exists(f): os.remove(f)
    os.symlink(p_a, "A.txt")
    os.symlink(p_b, "B.txt")

def run_mpi_cmd(n, cores):
    cmd = ["mpirun", "--oversubscribe", "-np", str(cores), "./lab3", str(n)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"(\d+\.\d+)", proc.stdout)
    return float(match.group(1)) * 1000 if match else 0.0

def start_benchmarks():
    mpi_results = {n: {} for n in SIZES}
    print("\n[!] Эксперимент: Зависимость от количества процессов MPI...")
    for n in SIZES:
        p_a, p_b = prepare_data(n)
        set_symlinks(p_a, p_b)
        for c in CORES_TEST:
            time_ms = run_mpi_cmd(n, c)
            mpi_results[n][c] = time_ms
            print(f"  N={n}, Ядра={c} -> {time_ms:.2f} ms")
    return mpi_results

def print_markdown_table(res):
    header = "| N / Ядра | " + " | ".join([f"{c} Core(s)" for c in CORES_TEST]) + " |"
    print(header)
    print("| " + "--- | " * (len(CORES_TEST) + 1))
    for n in SIZES:
        print(f"| **{n}** | " + " | ".join([f"{res[n][c]:.2f}" for c in CORES_TEST]) + " |")

def plot_mpi(data):
    plt.style.use('seaborn-v0_8-muted') 
    plt.figure(figsize=(10, 6))

    for n in SIZES:
        plt.plot(CORES_TEST, [data[n][c] for c in CORES_TEST], marker='o', linewidth=2, label=f'N={n}')
    
    plt.title('Зависимость времени от количества процессов (MPI)', fontsize=12, fontweight='bold')
    plt.xlabel('Количество процессов (mpirun -np)')
    plt.ylabel('Время выполнения (мс)')
    plt.xticks(CORES_TEST)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Размеры", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('lab3_plots.png', dpi=300)
    print("\n[!] График сохранен в lab3_plots.png")

if __name__ == "__main__":
    print("[*] Компиляция lab3_mpi.cpp...")
    subprocess.run(["mpic++", "-O3", "lab3_mpi.cpp", "-o", "lab3"])
    
    mpi_data = start_benchmarks()
    print_markdown_table(mpi_data)
    plot_mpi(mpi_data)

import os, subprocess, re
import matplotlib.pyplot as plt

SIZES = [400, 800, 1600, 2000]
BLOCK_SIZES = [8, 16, 32]

def run_cuda_tests():
    results = {bs: {} for bs in BLOCK_SIZES}
    executable = "./lab4.exe" if os.name == 'nt' else "./lab4"
    
    for bs in BLOCK_SIZES:
        print(f"\n[!] Тестирование сетки: {bs}x{bs}")
        for n in SIZES:
            path_a = f"matrix_data/A_{n}.txt"
            path_b = f"matrix_data/B_{n}.txt"
            
            proc = subprocess.run([executable, str(n), str(bs), path_a, path_b], 
                                  capture_output=True, text=True)
            
            output = proc.stdout
            match = re.search(r"RESULT_MS:\s*([\d\.]+)", output)
            if match:
                time_ms = float(match.group(1))
                results[bs][n] = time_ms
                print(f"  N={n} -> {time_ms:.2f} ms")
            else:
                print(f"  [!] Ошибка для N={n}. Проверьте пути к файлам.")
    return results

def generate_markdown_table(data):
    header = "| Размер матрицы (N) | " + " | ".join([f"Блок {bs}x{bs} (ms)" for bs in BLOCK_SIZES]) + " |"
    separator = "| " + " | ".join(["---"] * (len(BLOCK_SIZES) + 1)) + " |"
    
    rows = []
    for n in SIZES:
        row = f"| **{n}** | "
        times = []
        for bs in BLOCK_SIZES:
            val = data[bs].get(n, "N/A")
            times.append(f"{val:.2f}" if isinstance(val, float) else val)
        row += " | ".join(times) + " |"
        rows.append(row)
    
    table = "\n".join([header, separator] + rows)
    print(table)

def plot_cuda_grids(data):
    plt.figure(figsize=(10, 6))
    for bs in BLOCK_SIZES:
        if not data[bs]: continue
        valid_sizes = sorted(data[bs].keys())
        times = [data[bs][n] for n in valid_sizes]
        plt.plot(valid_sizes, times, marker='o', label=f'Block {bs}x{bs}')
    
    plt.xlabel('Размер матрицы (N)')
    plt.ylabel('Время (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('lab4_plots.png')
    print("\n[!] График сохранен: lab4_plots.png")

if __name__ == "__main__":
    print("[*] Компиляция lab4_cuda.cu...")
    subprocess.run(["nvcc", "-O3", "lab4_cuda.cu", "-o", "lab4"])
    
    res_data = run_cuda_tests()
    generate_markdown_table(res_data)
    plot_cuda_grids(res_data)
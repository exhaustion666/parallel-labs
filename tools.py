import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

SIZES = [100, 200, 400, 800, 1600, 2000]
DATA_DIR = "matrix_data"
RESULT_FILE = "C.txt"

def prepare_data(n):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    p_a, p_b = f"{DATA_DIR}/A_{n}.txt", f"{DATA_DIR}/B_{n}.txt"
    if not os.path.exists(p_a) or not os.path.exists(p_b):
        print(f"[*] Генерация матриц для N={n}...")
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        np.savetxt(p_a, a, fmt='%.5f')
        np.savetxt(p_b, b, fmt='%.5f')
    return p_a, p_b

def verify_result(n, p_a, p_b):
    """Сверяет C.txt с результатом NumPy"""
    if not os.path.exists(RESULT_FILE):
        return False
    try:
        a = np.loadtxt(p_a)
        b = np.loadtxt(p_b)
        c_cpp = np.loadtxt(RESULT_FILE)
        c_target = a @ b
        return np.allclose(c_cpp, c_target, atol=1e-2)
    except Exception as e:
        print(f"  [!] Ошибка верификации: {e}")
        return False

def run_benchmarks():
    results = []

    for n in SIZES:
        p_a, p_b = prepare_data(n)
        
        for f in ["A.txt", "B.txt"]:
            if os.path.exists(f): os.remove(f)
        os.symlink(p_a, "A.txt")
        os.symlink(p_b, "B.txt")

        try:
            proc = subprocess.run(["./lab1", str(n)], capture_output=True, text=True, timeout=600)
            output = proc.stdout + proc.stderr
            
            match = re.search(r"(\d+\.\d+)", output)
            time_ms = float(match.group(1)) * 1000 if match else 0.0
            
            is_valid = verify_result(n, p_a, p_b)
            status = "True" if is_valid else "False"
            
        except Exception as e:
            print(f"[!] Ошибка на N={n}: {e}")
            time_ms, status = 0.0, "ERROR"

        ops = 2 * (n**3)
        results.append({"n": n, "ms": time_ms, "ops": ops, "valid": status})
        print(f"[N={n}] Time: {time_ms:>10.2f} ms | Valid: {status}")

    print("| Размер матриц | Время, мс | Кол-во операций | Верификация |")
    print("| :--- | :--- | :--- | :--- |")
    for r in results:
        ops_fmt = f"{r['ops']:.2e}" if r['ops'] >= 1e9 else f"{r['ops']:,}".replace(",", " ")
        print(f"| {r['n']:<13} | {r['ms']:<9.2f} | {ops_fmt:<15} | {r['valid']:<11} |")
    print("—"*70 + "\n")

    return results

if __name__ == "__main__":
    if not os.path.exists("./lab1"):
        print("[!] Компилирую lab1_seq.cpp...")
        subprocess.run(["g++", "-O3", "lab1_seq.cpp", "-o", "lab1"])

    data = run_benchmarks()

    ns = [r['n'] for r in data]
    ms = [r['ms'] for r in data]
    ops = [r['ops'] for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ns, ms, 'r-o', label='Время (мс)')
    ax1.set_title('Зависимость времени от размера N')
    ax1.set_xlabel('N')
    ax1.set_ylabel('ms')
    ax1.grid(True)

    ax2.plot(ms, ops, 'b-s', label='Операции')
    ax2.set_title('Зависимость кол-ва операций от времени')
    ax2.set_xlabel('ms')
    ax2.set_ylabel('V (ops)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('lab1_plots.png')
    print("[!] Графики сохранены в lab1_plots.png")

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cout << "Использование: mpirun -np <кол-во_процессов> ./lab3 <N>" << endl;
        MPI_Finalize();
        return 1;
    }

    int N = stoi(argv[1]);

    if (N % size != 0) {
        if (rank == 0) {
            cout << "Ошибка: N (" << N << ") должно нацело делиться на количество процессов (" << size << ")!" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;

    vector<double> A, B(N * N), C;
    vector<double> local_A(rows_per_proc * N);
    vector<double> local_C(rows_per_proc * N, 0.0);

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        
        ifstream fa("A.txt"), fb("B.txt");
        if (!fa.is_open() || !fb.is_open()) {
            cout << "Ошибка: Не удалось открыть файлы A.txt или B.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N * N; ++i) fa >> A[i];
        for (int i = 0; i < N * N; ++i) fb >> B[i];
        
        fa.close();
        fb.close();
        cout << "[*] Данные загружены. Начинаю вычисления для N=" << N << " на " << size << " процессах..." << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE,
                local_A.data(), rows_per_proc * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; ++i) {
        for (int k = 0; k < N; ++k) {
            double temp = local_A[i * N + k];
            for (int j = 0; j < N; ++j) {
                local_C[i * N + j] += temp * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C.data(), rows_per_proc * N, MPI_DOUBLE,
               C.data(), rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        cout << "Time: " << (end_time - start_time) << " s" << endl;

        ofstream fc("C.txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                fc << C[i * N + j] << (j == N - 1 ? "" : " ");
            }
            fc << "\n";
        }
        fc.close();
        cout << "[*] Результат сохранен в C.txt" << endl;
    }

    MPI_Finalize();
    return 0;
}

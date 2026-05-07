#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <ctime>

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

        mt19937 gen(static_cast<unsigned>(time(nullptr)));
        uniform_real_distribution<double> dist(-10.0, 10.0);

        for (int i = 0; i < N * N; ++i) {
            A[i] = dist(gen);
        }

        for (int i = 0; i < N * N; ++i) {
            B[i] = dist(gen);
        }

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
        double execution_time = end_time - start_time;

        cout << "Matrix size: " << N << "x" << N << endl;
            cout << "Processes count: " << size << endl;
        cout << "Calculation time: " << execution_time << " sec." << endl;
    }

    MPI_Finalize();
    return 0;
}

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    int N = stoi(argv[1]);

    vector<double> A(N * N), B(N * N), C(N * N, 0.0);

    ifstream fa("A.txt"), fb("B.txt");
    for (int i = 0; i < N * N; ++i) fa >> A[i];
    for (int i = 0; i < N * N; ++i) fb >> B[i];

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double temp = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    ofstream fc("C.txt");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fc << C[i * N + j] << (j == N - 1 ? "" : " ");
        }
        fc << "\n";
    }

    cout << "Time: " << diff.count() << " s" << endl;

    return 0;
}

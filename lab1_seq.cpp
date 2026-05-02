#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

using namespace std;

void readMatrix(const string& filename, vector<double>& M, int N) {
    ifstream file(filename);
    for (int i = 0; i < N * N; ++i) file >> M[i];
}

void writeMatrix(const string& filename, const vector<double>& M, int N) {
    ofstream file(filename);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) file << M[i * N + j] << " ";
        file << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) return 1;
    int N = stoi(argv[1]);
    
    vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    readMatrix("A.txt", A, N);
    readMatrix("B.txt", B, N);

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    
    long long operations = 2LL * N * N * N;
    cout << "Seq | N: " << N << " | Time: " << diff.count() << " s | Ops: " << operations << endl;

    writeMatrix("C.txt", C, N);
    return 0;
}

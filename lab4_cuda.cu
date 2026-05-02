#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMulCUDA(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float temp = 0;
        for (int k = 0; k < N; ++k) temp += A[row * N + k] * B[k * N + col];
        C[row * N + col] = temp;
    }
}

int main(int argc, char* argv[]) {
    // Теперь принимаем: N, blockSize, путь_к_A, путь_к_B
    if (argc < 5) return 1;
    int N = stoi(argv[1]);
    int blockSize = stoi(argv[2]);
    string pathA = argv[3];
    string pathB = argv[4];

    ifstream fa(pathA), fb(pathB);
    if (!fa.is_open() || !fb.is_open()) {
        cout << "ERROR_FILES_NOT_FOUND" << endl;
        return 1;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector<float> h_A(N * N), h_B(N * N), h_C(N * N);
    for (int i = 0; i < N * N; ++i) fa >> h_A[i];
    for (int i = 0; i < N * N; ++i) fb >> h_B[i];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(blockSize, blockSize);
    dim3 blocks((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    

    matrixMulCUDA<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cout << "RESULT_MS: " << ms << endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
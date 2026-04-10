#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int m = 4096, n = 4096, k = 4096;

    // Load pre-generated inputs
    float *A = new float[m * k];
    float *B = new float[k * n];
    float *C = new float[m * n]();

    FILE *fa = fopen("../inputs/a.bin", "rb");
    FILE *fb = fopen("../inputs/b.bin", "rb");
    fread(A, sizeof(float), m * k, fa);
    fread(B, sizeof(float), k * n, fb);
    fclose(fa);
    fclose(fb);

    float alpha = 1.0f;
    float beta = 0.0f;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result of A * B (top-left 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", C[i + j * m]);
        }
        printf("\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

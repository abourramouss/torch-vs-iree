#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    hipblasHandle_t handle;
    hipblasCreate(&handle);

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
    hipMalloc(&d_A, m * k * sizeof(float));
    hipMalloc(&d_B, k * n * sizeof(float));
    hipMalloc(&d_C, m * n * sizeof(float));
    hipMemcpy(d_A, A, m * k * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, k * n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_C, C, m * n * sizeof(float), hipMemcpyHostToDevice);
    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);
    printf("Result of A * B (top-left 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", C[i + j * m]);
        }
        printf("\n");
    }
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipblasDestroy(handle);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

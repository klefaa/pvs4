#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void matrix_ops(float* a, float* b, float* add, float* sub, float* mul, float* divv, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N) {
        int idx = i * N + j;
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        divv[idx] = b[idx] != 0 ? a[idx] / b[idx] : 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Использование: %s <размерность матрицы NxN> <размер блока>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int size = N * N;
    int blockSize = atoi(argv[2]);

    float *a, *b;
    a = (float*)malloc(size * sizeof(float));
    b = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_add, size * sizeof(float));
    cudaMalloc(&d_sub, size * sizeof(float));
    cudaMalloc(&d_mul, size * sizeof(float));
    cudaMalloc(&d_div, size * sizeof(float));

    cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(blockSize, blockSize);
    dim3 blocks((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

    clock_t start = clock();
    matrix_ops<<<blocks, threads>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, N);
    cudaDeviceSynchronize();
    clock_t end = clock();

    printf("Операции с матрицами завершены.\n");
    printf("Время выполнения: %.4f секунд\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(a); free(b);
    return 0;
}
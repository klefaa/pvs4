#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

// Ядро CUDA: операции с матрицами
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
        printf("Использование: %s <размерность матрицы NxN> <количество потоков на блок>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int size = N * N;

    // Выбор "квадратной" формы блока
    int blockDimX = (int)sqrt((double)threadsPerBlock);
    while (threadsPerBlock % blockDimX != 0) blockDimX--;
    int blockDimY = threadsPerBlock / blockDimX;

    dim3 threads(blockDimX, blockDimY);
    dim3 blocks((N + blockDimX - 1) / blockDimX, (N + blockDimY - 1) / blockDimY);

    printf("Конфигурация CUDA:\n");
    printf("  blockDim = (%d, %d)\n", blockDimX, blockDimY);
    printf("  gridDim  = (%d, %d)\n", blocks.x, blocks.y);

    float *a = (float*)malloc(size * sizeof(float));
    float *b = (float*)malloc(size * sizeof(float));
    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;

    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_add, size * sizeof(float));
    cudaMalloc(&d_sub, size * sizeof(float));
    cudaMalloc(&d_mul, size * sizeof(float));
    cudaMalloc(&d_div, size * sizeof(float));

    srand(time(NULL));
    float totalTime = 0.0f;

    for (int run = 0; run < 100; run++) {
        for (int i = 0; i < size; i++) {
            a[i] = rand() / (float)RAND_MAX;
            b[i] = rand() / (float)RAND_MAX;
        }

        cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        matrix_ops<<<blocks, threads>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Среднее время выполнения за 100 запусков: %.4f мс\n", totalTime / 100.0f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(a); free(b);
    return 0;
}

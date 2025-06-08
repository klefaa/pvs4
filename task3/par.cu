#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void array_ops(float* a, float* b, float* add, float* sub, float* mul, float* divv, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        divv[idx] = b[idx] != 0 ? a[idx] / b[idx] : 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Использование: %s <размер массива> <кол-во потоков>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int threads = atoi(argv[2]);
    int blocks = (size + threads - 1) / threads;

    float *h_a = (float*)malloc(size * sizeof(float));
    float *h_b = (float*)malloc(size * sizeof(float));
    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;

    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_add, size * sizeof(float));
    cudaMalloc(&d_sub, size * sizeof(float));
    cudaMalloc(&d_mul, size * sizeof(float));
    cudaMalloc(&d_div, size * sizeof(float));

    float totalTime = 0.0f;
    srand(time(NULL));

    for (int iter = 0; iter < 100; iter++) {
        // Генерация случайных данных
        for (int i = 0; i < size; i++) {
            h_a[i] = rand() / (float)RAND_MAX;
            h_b[i] = rand() / (float)RAND_MAX;
        }

        cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

        // Таймеры CUDA
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        array_ops<<<blocks, threads>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, size);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        totalTime += elapsed;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Среднее время выполнения на GPU за 100 запусков: %.4f мс\n", totalTime / 100.0f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    free(h_a); free(h_b);

    return 0;
}

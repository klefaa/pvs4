#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void sum_reduce(float* input, float* output, int size) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Инициализация shared memory
    temp[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    // Редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            temp[tid] += temp[tid + stride];
        __syncthreads();
    }

    // Сохранение частичной суммы
    if (tid == 0)
        output[blockIdx.x] = temp[0];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <num_elements> <threads_per_block>\n", argv[0]);
        return 1;
    }

    // Инициализация генератора случайных чисел
    srand(time(NULL));

    const int size = atoi(argv[1]);
    const int threadsPerBlock = atoi(argv[2]);
    const int numRuns = 100; // Количество прогонов
    
    // Расчет количества блоков (не более 256)
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf("Using %d blocks, %d threads per block, %d runs\n", blocks, threadsPerBlock, numRuns);

    // Выделение и инициализация памяти на хосте
    float* h_data = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        h_data[i] = (float)rand() / RAND_MAX;  // Числа от 0 до 1
    }

    // Выделение памяти на устройстве
    float *d_data, *d_partial;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Создание событий CUDA для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalTime = 0;
    float finalSum = 0;

    for (int run = 0; run < numRuns; ++run) {
        // Замер времени выполнения
        cudaEventRecord(start);
        sum_reduce<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_data, d_partial, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Копирование результатов
        float* h_partial = (float*)malloc(blocks * sizeof(float));
        cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

        // Финальное суммирование
        float sum = 0;
        for (int i = 0; i < blocks; ++i) {
            sum += h_partial[i];
        }
        finalSum = sum; // Сохраняем последнее значение суммы

        // Вычисление времени выполнения
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;

        free(h_partial);
    }

    // Вычисление среднего времени
    float avgTime = totalTime / numRuns;

    printf("Final sum: %.2f\n", finalSum);
    printf("Average execution time over %d runs: %.4f ms\n", numRuns, avgTime);

    // Освобождение ресурсов
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

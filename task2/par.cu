#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__device__ void swap(float &a, float &b, bool dir) {
    if ((a > b) == dir) {
        float temp = a;
        a = b;
        b = temp;
    }
}

__global__ void bitonic_sort(float* data, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0)
            swap(data[i], data[ixj], true);
        else
            swap(data[i], data[ixj], false);
    }
}

void fillArray(float* a, int size) {
    srand(time(NULL)); // Инициализация генератора случайных чисел
    for (int i = 0; i < size; i++)
        a[i] = (float)rand() / RAND_MAX; // Генерация чисел от 0 до 1
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Использование: %s <кол-во элементов (степень 2)> <потоки>\n", argv[0]);
        return 1;
    }

    const int size = atoi(argv[1]);
    const int threads = atoi(argv[2]);
    const int blocks = size / threads;
    const int num_runs = 20;

    // Проверка, что размер является степенью двойки
    if ((size & (size - 1)) != 0) {
        printf("Ошибка: размер должен быть степенью 2\n");
        return 1;
    }

    // Выделение и инициализация памяти на хосте
    float* h_data = (float*)malloc(size * sizeof(float));
    fillArray(h_data, size);

    // Выделение памяти на устройстве
    float* d_data;
    cudaError_t err = cudaMalloc(&d_data, size * sizeof(float));
    checkCudaError(err, "cudaMalloc");
    
    // Создание событий CUDA для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0.0f;

    for (int run = 0; run < num_runs; ++run) {
        // Копирование данных на устройство
        err = cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);
        checkCudaError(err, "cudaMemcpy H2D");
        
        // Замер времени выполнения
        cudaEventRecord(start);
        
        // Bitonic сортировка
        for (int k = 2; k <= size; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_sort<<<blocks, threads>>>(d_data, j, k);
                err = cudaGetLastError();
                checkCudaError(err, "bitonic_sort kernel");
            }
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Вычисление времени выполнения
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }

    // Копирование результатов обратно
    err = cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy D2H");

    // Вычисление среднего времени
    float avg_time = total_time / num_runs;
    
    printf("Bitonic Sort (параллельная)\n");
    printf("Размер массива: %d элементов\n", size);
    printf("Блоков: %d, Потоков на блок: %d\n", blocks, threads);
    printf("Среднее время выполнения за %d запусков: %.4f мс\n", num_runs, avg_time);

    // Освобождение ресурсов
    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

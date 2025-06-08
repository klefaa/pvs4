#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
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
    for (int i = 0; i < size; i++)
        a[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Использование: %s <кол-во элементов (степень 2)> <потоки>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int threads = atoi(argv[2]);
    int blocks = size / threads;

    float* h_data = (float*)malloc(size * sizeof(float));
    fillArray(h_data, size);

    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start = clock();
    for (int k = 2; k <= size; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            bitonic_sort<<<blocks, threads>>>(d_data, j, k);

    cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end = clock();

    printf("Параллельная сортировка (Bitonic) завершена.\n");
    printf("Время выполнения: %.4f секунд\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(h_data);
    cudaFree(d_data);
    return 0;
}
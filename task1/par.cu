#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

__global__ void sum_reduce(float* input, float* output, int size) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp[tid] = (i < size) ? input[i] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            temp[tid] += temp[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = temp[0];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Использование: %s <кол-во элементов> <кол-во потоков>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    float* h_data = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
        h_data[i] = rand() / (float)RAND_MAX;

    float* d_data;
    float* d_partial;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start = clock();
    sum_reduce<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_data, d_partial, size);

    float* h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < blocks; i++)
        sum += h_partial[i];
    clock_t end = clock();

    printf("Сумма элементов: %f\n", sum);
    printf("Время выполнения: %.4f секунд\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(h_data);
    free(h_partial);
    cudaFree(d_data);
    cudaFree(d_partial);
    return 0;
}
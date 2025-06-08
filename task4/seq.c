#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_ops(float* a, float* b, float* add, float* sub, float* mul, float* divv, int N) {
    int size = N * N;
    for (int i = 0; i < size; i++) {
        add[i] = a[i] + b[i];
        sub[i] = a[i] - b[i];
        mul[i] = a[i] * b[i];
        divv[i] = b[i] != 0 ? a[i] / b[i] : 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Использование: %s <размерность матрицы NxN>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int size = N * N;

    float *a = (float*)malloc(size * sizeof(float));
    float *b = (float*)malloc(size * sizeof(float));
    float *add = (float*)malloc(size * sizeof(float));
    float *sub = (float*)malloc(size * sizeof(float));
    float *mul = (float*)malloc(size * sizeof(float));
    float *divv = (float*)malloc(size * sizeof(float));

    srand(time(NULL));
    double totalTime = 0.0;

    for (int run = 0; run < 100; run++) {
        for (int i = 0; i < size; i++) {
            a[i] = rand() / (float)RAND_MAX;
            b[i] = rand() / (float)RAND_MAX;
        }

        clock_t start = clock();
        matrix_ops(a, b, add, sub, mul, divv, N);
        clock_t end = clock();

        totalTime += (double)(end - start) * 1000 / CLOCKS_PER_SEC; // в мс
    }

    printf("Среднее время выполнения на CPU за 100 запусков: %.4f мс\n", totalTime / 100.0);

    free(a); free(b); free(add); free(sub); free(mul); free(divv);
    return 0;
}

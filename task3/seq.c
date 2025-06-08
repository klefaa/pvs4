#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_RUNS 100
#define PRINT_SAMPLES 5

void elementwise_operations_seq(double *a, double *b, double *add, double *sub, 
                              double *mul, double *div, long n) {
    for (long i = 0; i < n; i++) {
        add[i] = a[i] + b[i];
        sub[i] = a[i] - b[i];
        mul[i] = a[i] * b[i];
        div[i] = (fabs(b[i]) > 1e-10) ? a[i] / b[i] : 0.0;
    }
}

void fill_random(double *arr, long n) {
    for (long i = 0; i < n; i++) {
        arr[i] = (double)rand() / RAND_MAX * 100.0; // Генерация чисел [0, 100)
    }
}

void print_samples(double *arr, long n, const char *name) {
    printf("%s (первые %d): ", name, PRINT_SAMPLES);
    for (int i = 0; i < PRINT_SAMPLES && i < n; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Использование: %s <размер_массивов>\n", argv[0]);
        return 1;
    }

    long n = atol(argv[1]);
    if (n < 100000) {
        printf("Размер массивов должен быть >= 100000\n");
        return 1;
    }

    // Выделение памяти
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *add = malloc(n * sizeof(double));
    double *sub = malloc(n * sizeof(double));
    double *mul = malloc(n * sizeof(double));
    double *div = malloc(n * sizeof(double));

    if (!a || !b || !add || !sub || !mul || !div) {
        printf("Ошибка выделения памяти\n");
        return 1;
    }

    // Инициализация генератора случайных чисел
    srand(time(NULL));

    // Заполнение массивов случайными значениями
    fill_random(a, n);
    fill_random(b, n);

    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        clock_t start = clock();
        elementwise_operations_seq(a, b, add, sub, mul, div, n);
        clock_t end = clock();

        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    // Вывод результатов
    printf("\n=== Последовательная версия ===\n");
    printf("Размер массивов: %ld элементов\n", n);
    printf("Количество прогонов: %d\n", NUM_RUNS);
    printf("Среднее время выполнения: %.6f секунд\n", total_time / NUM_RUNS);
    printf("Общее время выполнения: %.6f секунд\n", total_time);

    // Вывод первых элементов для проверки
    print_samples(a, n, "Массив A");
    print_samples(b, n, "Массив B");
    print_samples(add, n, "Сложение");
    print_samples(sub, n, "Вычитание");
    print_samples(mul, n, "Умножение");
    print_samples(div, n, "Деление");

    // Освобождение памяти
    free(a); free(b); free(add); free(sub); free(mul); free(div);
    
    return 0;
}

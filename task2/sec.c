#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge(float* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    float* L = malloc(n1 * sizeof(float));
    float* R = malloc(n2 * sizeof(float));
    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
    free(L); free(R);
}

void mergeSort(float* arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Использование: %s <кол-во элементов>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    float* data = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
        data[i] = rand() / (float)RAND_MAX;

    clock_t start = clock();
    mergeSort(data, 0, size - 1);
    clock_t end = clock();

    printf("Последовательная сортировка завершена.\n");
    printf("Время выполнения: %.4f секунд\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(data);
    return 0;
}
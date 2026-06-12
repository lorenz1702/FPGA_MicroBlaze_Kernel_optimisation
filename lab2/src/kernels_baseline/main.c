#include <xil_printf.h>
#include <stdlib.h>
#include <stdio.h>
#include "kernels.h"
#include "timer.h"

#define KERNEL1_SIZE 10000
#define KERNEL1_OFFSET 50

#define KERNEL2_SIZE 10000

#define KERNEL3_SIZE 10000
#define KERNEL3_H_SIZE 20000

#define KERNEL4_SIZE 10000

void fill_random_int(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

void fill_random_float(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / (float)(RAND_MAX)) * 100.0f;
    }
}

void fill_random_indices(int *arr, int size, int max_index) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % max_index;
    }
}

int main() {
    srand(42);
    xil_printf("Starting Baseline Benchmark for Kernels 1-4\r\n");

    int *A1 = (int *)malloc(KERNEL1_SIZE * sizeof(int));
    int *A2 = (int *)malloc(KERNEL2_SIZE * sizeof(int));
    
    float *h3 = (float *)malloc(KERNEL3_H_SIZE * sizeof(float));
    float *w3 = (float *)malloc(KERNEL3_SIZE * sizeof(float));
    int *idx3 = (int *)malloc(KERNEL3_SIZE * sizeof(int));
    
    float *A4 = (float *)malloc(KERNEL4_SIZE * sizeof(float));
    float *B4 = (float *)malloc(KERNEL4_SIZE * sizeof(float));

    if (!A1 || !A2 || !h3 || !w3 || !idx3 || !A4 || !B4) {
        xil_printf("Memory allocation failed!\r\n");
        return -1;
    }

    fill_random_int(A1, KERNEL1_SIZE);
    fill_random_int(A2, KERNEL2_SIZE);
    
    fill_random_float(h3, KERNEL3_H_SIZE);
    fill_random_float(w3, KERNEL3_SIZE);
    fill_random_indices(idx3, KERNEL3_SIZE, KERNEL3_H_SIZE);
    
    fill_random_float(A4, KERNEL4_SIZE);
    fill_random_float(B4, KERNEL4_SIZE);

    double t;
    char s[128] = {};

    start_timer();
    kernel1(A1, KERNEL1_SIZE, KERNEL1_OFFSET);
    t = stop_timer();
    sprintf(s, "Kernel 1 Time: %.6fs\r\n", t);
    xil_printf("%s", s);

    start_timer();
    kernel2(A2, KERNEL2_SIZE);
    t = stop_timer();
    sprintf(s, "Kernel 2 Time: %.6fs\r\n", t);
    xil_printf("%s", s);

    start_timer();
    kernel3(h3, w3, idx3, KERNEL3_SIZE);
    t = stop_timer();
    sprintf(s, "Kernel 3 Time: %.6fs\r\n", t);
    xil_printf("%s", s);

    start_timer();
    float result4 = kernel4(A4, B4, KERNEL4_SIZE);
    t = stop_timer();
    
    int res4_int = (int)result4;
    sprintf(s, "Kernel 4 Time: %.6fs, Result: %d\r\n", t, res4_int);
    xil_printf("%s", s);

    xil_printf("Benchmark Complete.\r\n");

    free(A1);
    free(A2);
    free(h3);
    free(w3);
    free(idx3);
    free(A4);
    free(B4);

    return 0;
}

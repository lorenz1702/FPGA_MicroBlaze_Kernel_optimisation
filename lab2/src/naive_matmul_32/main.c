#include <xil_printf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "kernels.h"
#include "kernels_opt.h"
#include "timer.h"
#include "xil_cache.h"

#define KERNEL1_SIZE 1000000
#define KERNEL1_OFFSET 50

#define KERNEL2_SIZE 1000000

#define KERNEL3_SIZE 1000000
#define KERNEL3_H_SIZE 2000000

#define KERNEL4_SIZE 1000000

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
	microblaze_enable_dcache();
    xil_printf("Starting Benchmark for Kernels 1-4 (Baseline vs Optimized)\r\n");

    int *A1 = (int *)malloc(KERNEL1_SIZE * sizeof(int));

    int *A2 = (int *)malloc(KERNEL2_SIZE * sizeof(int));

    float *h3 = (float *)malloc(KERNEL3_H_SIZE * sizeof(float));
    float *w3 = (float *)malloc(KERNEL3_SIZE * sizeof(float));
    int *idx3 = (int *)malloc(KERNEL3_SIZE * sizeof(int));

    float *A4 = (float *)malloc(KERNEL4_SIZE * sizeof(float));
    float *B4 = (float *)malloc(KERNEL4_SIZE * sizeof(float));

    xil_printf("allocated memory \r\n");

    if (!A1 || !A2 || !h3 || !w3 || !idx3 || !A4 || !B4 || !h3 || !w3 || !idx3) {
        xil_printf("Memory allocation failed!\r\n");
        return -1;
    }

    srand(42);
    fill_random_int(A1, KERNEL1_SIZE);

    fill_random_int(A2, KERNEL2_SIZE);

    fill_random_float(h3, KERNEL3_H_SIZE);
    fill_random_float(w3, KERNEL3_SIZE);
    fill_random_indices(idx3, KERNEL3_SIZE, KERNEL3_H_SIZE);

    fill_random_float(A4, KERNEL4_SIZE);
    fill_random_float(B4, KERNEL4_SIZE);

    xil_printf("filled with random valoues \r\n");

    double t_base, t_opt;
    char s[128] = {};

    // --- Benchmark Kernel 1 --
    microblaze_invalidate_dcache();
    start_timer();
    kernel1(A1, KERNEL1_SIZE, KERNEL1_OFFSET);
    t_base = stop_timer();
    microblaze_invalidate_dcache();
    xil_printf("kernel1 \r\n");


    start_timer();
    kernel1_opt(A1, KERNEL1_SIZE, KERNEL1_OFFSET);
    t_opt = stop_timer();
    microblaze_invalidate_dcache();
    xil_printf("kernel1 opt \r\n");

    sprintf(s, "Kernel 1 -> Base: %.6fs | Opt: %.6fs\r\n", t_base, t_opt);
    xil_printf("%s", s);

    // --- Benchmark Kernel 2 ---
    start_timer();
    kernel2(A2, KERNEL2_SIZE);
    t_base = stop_timer();
    microblaze_invalidate_dcache();

    start_timer();
    kernel2_opt(A2, KERNEL2_SIZE);
    t_opt = stop_timer();
    microblaze_invalidate_dcache();

    sprintf(s, "Kernel 2 -> Base: %.6fs | Opt: %.6fs\r\n", t_base, t_opt);
    xil_printf("%s", s);

    // --- Benchmark Kernel 3 ---
    start_timer();
    kernel3(h3, w3, idx3, KERNEL3_SIZE);
    t_base = stop_timer();
    microblaze_invalidate_dcache();

    start_timer();
    kernel3_opt(h3, w3, idx3, KERNEL3_SIZE);
    t_opt = stop_timer();
    microblaze_invalidate_dcache();

    sprintf(s, "Kernel 3 -> Base: %.6fs | Opt: %.6fs\r\n", t_base, t_opt);
    xil_printf("%s", s);

    // --- Benchmark Kernel 4 ---
    start_timer();
    float result4_base = kernel4(A4, B4, KERNEL4_SIZE);
    t_base = stop_timer();
    microblaze_invalidate_dcache();

    start_timer();
    float result4_opt = kernel4_opt(A4, B4, KERNEL4_SIZE);
    t_opt = stop_timer();
    microblaze_invalidate_dcache();

    int r_base = (int)result4_base;
    int r_opt = (int)result4_opt;
    sprintf(s, "Kernel 4 -> Base: %.6fs (Res: %d) | Opt: %.6fs (Res: %d)\r\n", t_base, r_base, t_opt, r_opt);
    xil_printf("%s", s);

    xil_printf("Benchmark Complete.\r\n");

    free(A1); ;
    free(A2); ;
    free(h3);
    free(w3);
    free(idx3);
    free(A4);
    free(B4);

    return 0;
}

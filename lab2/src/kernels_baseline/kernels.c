#include "kernels.h"

void kernel1 (int *A, int size, int offset) {
    int i;
    for(i = 0; i < size-offset; i++)
        A[i] += A[i +offset];
}

void kernel2 (int *A, int size) {
    int i;
    for(i = 3; i < size; i++)
        A[i] = A[i - 1] + A[i - 2] * A[i - 3];
}

void kernel3 (float *h, float *w, int *idx, int size) {
    for (int i = 0; i < size; ++i)
        h[idx[i]] = h[idx[i]] + w[i];
}

float kernel4 (float *A, float *B, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        float diff = A[i] - B[i];
        if (diff > 0) sum = (sum + diff);
    }
    return sum;
}

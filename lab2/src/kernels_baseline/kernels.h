#ifndef KERNELS_H
#define KERNELS_H

void kernel1 (int *A, int size, int offset);
void kernel2 (int *A, int size);
void kernel3 (float *h, float *w, int *idx, int size);
float kernel4 (float *A, float *B, int size);

#endif // KERNELS_H

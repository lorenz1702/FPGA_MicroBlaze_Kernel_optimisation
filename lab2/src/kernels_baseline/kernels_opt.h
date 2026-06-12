#ifndef KERNELS_OPT_H
#define KERNELS_OPT_H

void kernel1_opt (int * restrict A, int size, int offset);
void kernel2_opt (int * restrict A, int size);
void kernel3_opt (float * restrict h, const float * restrict w, const int * restrict idx, int size);
float kernel4_opt (const float * restrict A, const float * restrict B, int size);

#endif // KERNELS_OPT_H

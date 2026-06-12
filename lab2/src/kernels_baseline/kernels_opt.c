#include "kernels_opt.h"

// Kernel 1: Loop unrolling. 
// Reduziert den Loop-Overhead und erlaubt es dem Compiler, Instruktionen besser zu schedulen.
void kernel1_opt (int * restrict A, int size, int offset) {
    int i;
    int limit = size - offset;
    
    // Unroll by 4
    for(i = 0; i <= limit - 4; i+=4) {
        A[i]   += A[i + offset];
        A[i+1] += A[i+1 + offset];
        A[i+2] += A[i+2 + offset];
        A[i+3] += A[i+3 + offset];
    }
    
    // Handle remaining elements
    for(; i < limit; i++) {
        A[i] += A[i + offset];
    }
}

// Kernel 2: Register Caching. 
// Im Original muss A[i-1], A[i-2], A[i-3] jedes mal aus dem Speicher gelesen werden.
// Wir cachen diese in lokalen Variablen (Registern), was Speicherzugriffe drastisch reduziert.
void kernel2_opt (int * restrict A, int size) {
    if (size <= 3) return;
    
    int a_minus_3 = A[0];
    int a_minus_2 = A[1];
    int a_minus_1 = A[2];
    
    for(int i = 3; i < size; i++) {
        int a_i = a_minus_1 + a_minus_2 * a_minus_3;
        A[i] = a_i;
        
        // Update registers for next iteration
        a_minus_3 = a_minus_2;
        a_minus_2 = a_minus_1;
        a_minus_1 = a_i;
    }
}

// Kernel 3: Loop unrolling & Pointer Aliasing. 
// "restrict" versichert dem Compiler, dass h, w und idx nicht im Speicher überlappen.
// Unrolling beschleunigt die Ausführung.
void kernel3_opt (float * restrict h, const float * restrict w, const int * restrict idx, int size) {
    int i;
    // Unroll by 4
    for (i = 0; i <= size - 4; i+=4) {
        h[idx[i]]   += w[i];
        h[idx[i+1]] += w[i+1];
        h[idx[i+2]] += w[i+2];
        h[idx[i+3]] += w[i+3];
    }
    for (; i < size; ++i) {
        h[idx[i]] += w[i];
    }
}

// Kernel 4: Branch Elimination & Accumulator Splitting.
// Die if(diff > 0) Bedingung stört die Branch Prediction und blockiert Pipelining.
// Wir ersetzen sie durch einen Ternary Operator, der vom Compiler oft als Conditional Move kompiliert wird.
// Zudem splitten wir "sum" in sum1, sum2, sum3, sum4 um Data Dependencies aufzubrechen.
float kernel4_opt (const float * restrict A, const float * restrict B, int size) {
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    int i;
    for (i = 0; i <= size - 4; i+=4) {
        float d1 = A[i] - B[i];
        float d2 = A[i+1] - B[i+1];
        float d3 = A[i+2] - B[i+2];
        float d4 = A[i+3] - B[i+3];
        
        sum1 += (d1 > 0.0f) ? d1 : 0.0f;
        sum2 += (d2 > 0.0f) ? d2 : 0.0f;
        sum3 += (d3 > 0.0f) ? d3 : 0.0f;
        sum4 += (d4 > 0.0f) ? d4 : 0.0f;
    }
    
    for (; i < size; i++) {
        float d = A[i] - B[i];
        sum1 += (d > 0.0f) ? d : 0.0f;
    }
    
    return sum1 + sum2 + sum3 + sum4;
}

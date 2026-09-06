// matmul.h  shared interface for all matmul stages (CS683 PA-1, Task 2)
//
// Every stage implements the SAME signature so the harness can call them uniformly and
// compare results and timings  and so the SAME kernel drops straight into llama.cpp.
//
// Layout (chosen to match ggml's ggml_mul_mat, the "NT" form). The contraction dim K is
// the INNER / contiguous dimension of BOTH operands:
//
//   A : M x K, row-major, row stride = lda (elements). A[i][p] = A[i*lda + p]
//   B : N x K, row-major, row stride = ldb (elements). B[j][p] = B[j*ldb + p]  (B is "transposed")
//   C : M x N, row-major, row stride = ldc (elements). C[i][j] = C[i*ldc + j]
//
//   C[i][j] = sum_{p=0..K-1}  A[i][p] * B[j][p]
//
// In the microbenchmark the matrices are contiguous, so lda = ldb = K and ldc = N. The
// leading-dimension arguments exist so the identical function can be injected into
// ggml's CPU matmul, where the strides come from the tensor's `nb` fields.
//
// The inner K-loop is therefore a unit-stride dot product over both operands  the shape
// AVX2/FMA and register tiling want.

#ifndef CS683_PA1_MATMUL_H
#define CS683_PA1_MATMUL_H

// Function-pointer type used by the harness to iterate over the stages.
using MatMulFn = void (*)(const float* A, const float* B, float* C,
                          int M, int N, int K, int lda, int ldb, int ldc);

// ---- The stages -----------------------------------------------------------
// matmul_naive is PROVIDED and serves as the correctness reference.
// The rest are student TODOs (see src/*.cpp). matmul_optimized is graded AND is the
// kernel injected into llama.cpp.

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc);
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc);
void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc);
void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc);

#endif  // CS683_PA1_MATMUL_H
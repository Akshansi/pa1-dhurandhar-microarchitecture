// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    const int m = 4;

    for (int i = 0; i < M; i++) {

        // Process N columns in batches of m
        for (int j = 0; j < N; j += m) {

            int batch = m< (N-j)? m : (N-j);

            // AVX2 accumulators
            __m256 acc[4];

            // Scalar accumulators for the K leftovers
            float scalar_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int k = 0; k < batch; k++) {
                acc[k] = _mm256_setzero_ps();
            }

            // -------------------------
            // SIMD part
            // -------------------------
            int p = 0;

            for (; p + 7 < K; p += 8) {

                // Load 8 values from A
                __m256 a = _mm256_loadu_ps(
                    &A[i * lda + p]
                );

                for (int k = 0; k < batch; k++) {

                    // Load 8 values from B
                    __m256 b = _mm256_loadu_ps(
                        &B[(j + k) * ldb + p]
                    );

                    // acc[k] += a * b
                    acc[k] = _mm256_fmadd_ps(a, b, acc[k]);
                }
            }

            // -------------------------
            // Scalar cleanup
            // -------------------------
            for (; p < K; p++) {

                for (int k = 0; k < batch; k++) {

                    scalar_acc[k] +=
                        A[i * lda + p] *
                        B[(j + k) * ldb + p];
                }
            }

            // -------------------------
            // Reduce and store
            // -------------------------
            for (int k = 0; k < batch; k++) {

                alignas(32) float temp[8];

                // Convert the AVX2 accumulator
                // into 8 scalar values
                _mm256_store_ps(temp, acc[k]);

                float sum = 0.0f;

                for (int x = 0; x < 8; x++) {
                    sum += temp[x];
                }

                // Add the leftover K elements
                sum += scalar_acc[k];

                // Store final result
                C[i * ldc + (j + k)] = sum;
            }
        }
    }
}

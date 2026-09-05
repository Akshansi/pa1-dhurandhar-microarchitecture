// matmul_simd.cpp  STAGE 1: SIMD with 128-bit intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    const int m = 4;

    for (int i = 0; i < M; i++) {

        // Process N columns in batches of m
        for (int j = 0; j < N; j += m) {

            int batch = m < (N - j) ? m : (N - j);

            // 128-bit SIMD accumulators
            __m128 acc[4];

            // Scalar accumulators for the K leftovers
            float scalar_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int k = 0; k < batch; k++) {
                acc[k] = _mm_setzero_ps();
            }

            // -------------------------
            // SIMD part
            // -------------------------
            int p = 0;

            // 128 bits = 4 floats
            for (; p + 3 < K; p += 4) {

                // Load 4 values from A
                __m128 a = _mm_loadu_ps(
                    &A[i * lda + p]
                );

                for (int k = 0; k < batch; k++) {

                    // Load 4 values from B
                    __m128 b = _mm_loadu_ps(
                        &B[(j + k) * ldb + p]
                    );

                    // acc[k] += a * b
                    acc[k] = _mm_add_ps(
                        acc[k],
                        _mm_mul_ps(a, b)
                    );
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

                alignas(16) float temp[4];

                // Convert the 128-bit accumulator
                // into 4 scalar values
                _mm_store_ps(temp, acc[k]);

                float sum = 0.0f;

                for (int x = 0; x < 4; x++) {
                    sum += temp[x];
                }

                // Add leftover K elements
                sum += scalar_acc[k];

                // Store final result
                C[i * ldc + (j + k)] = sum;
            }
        }
    }
}
// matmul_simd.cpp  STAGE 1: SIMD with AVX-512 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    const int m = 4;

    for (int i = 0; i < M; i++) {

        // Process N columns in batches of m
        for (int j = 0; j < N; j += m) {

            int batch = m < (N - j) ? m : (N - j);

            // AVX-512 accumulators
            __m512 acc[4];

            // Scalar accumulators for K leftovers
            float scalar_acc[4] = {
                0.0f, 0.0f, 0.0f, 0.0f
            };

            for (int k = 0; k < batch; k++) {
                acc[k] = _mm512_setzero_ps();
            }

            // -------------------------
            // SIMD part
            // -------------------------
            int p = 0;

            // 512 bits = 16 floats
            for (; p + 15 < K; p += 16) {

                // Load 16 values from A
                __m512 a = _mm512_loadu_ps(
                    &A[i * lda + p]
                );

                for (int k = 0; k < batch; k++) {

                    // Load 16 values from B
                    __m512 b = _mm512_loadu_ps(
                        &B[(j + k) * ldb + p]
                    );

                    // acc[k] += a * b
                    acc[k] = _mm512_fmadd_ps(
                        a,
                        b,
                        acc[k]
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

                alignas(64) float temp[16];

                // Convert AVX-512 accumulator
                // into 16 scalar values
                _mm512_store_ps(temp, acc[k]);

                float sum = 0.0f;

                for (int x = 0; x < 16; x++) {
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
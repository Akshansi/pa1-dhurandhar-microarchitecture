// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int m = 4;
    // Cache tile sizes
    const int BM = 32;
    const int BN = 32;
    const int PREFETCH_DISTANCE = 64;
    for (int i0 = 0; i0 < M; i0 += BM) {

        int i_end = (i0 + BM < M) ? i0 + BM : M;

        for (int j0 = 0; j0 < N; j0 += BN) {

            int j_end = (j0 + BN < N) ? j0 + BN : N;
            for (int i = i0; i < i_end; i++) {

                
                for (int j = j0; j < j_end; j += m) {

                    int batch = m < (j_end - j)
                              ? m
                              : (j_end - j);

                    // AVX2 accumulators
                    __m256 acc[4];

                    // Scalar accumulators for K leftovers
                    float scalar_acc[4] = {
                        0.0f, 0.0f, 0.0f, 0.0f
                    };

                    for (int k = 0; k < batch; k++) {
                        acc[k] = _mm256_setzero_ps();
                    }

                   
                    int p = 0;

                    for (; p + 7 < K; p += 8) {
                        if (p + PREFETCH_DISTANCE < K) {

                            _mm_prefetch(
                                (const char*)&A[
                                    i * lda +
                                    p + PREFETCH_DISTANCE
                                ],
                                _MM_HINT_T0
                            );
                        }
                        if (p + PREFETCH_DISTANCE < K) {

                            for (int k = 0; k < batch; k++) {

                                _mm_prefetch(
                                    (const char*)&B[
                                        (j + k) * ldb +
                                        p + PREFETCH_DISTANCE
                                    ],
                                    _MM_HINT_T0
                                );
                            }
                        }

                        __m256 a = _mm256_loadu_ps(
                            &A[i * lda + p]
                        );
                        for (int k = 0; k < batch; k++) {

                            __m256 b = _mm256_loadu_ps(
                                &B[(j + k) * ldb + p]
                            );

                            // Fused multiply-add
                            acc[k] = _mm256_fmadd_ps(
                                a,
                                b,
                                acc[k]
                            );
                        }
                    }

                    // -----------------------------------------
                    // Scalar K cleanup
                    // -----------------------------------------
                    for (; p < K; p++) {

                        for (int k = 0; k < batch; k++) {

                            scalar_acc[k] +=
                                A[i * lda + p] *
                                B[(j + k) * ldb + p];
                        }
                    }

                    // -----------------------------------------
                    // Reduce and store
                    // -----------------------------------------
                    for (int k = 0; k < batch; k++) {

                        alignas(32) float temp[8];

                        _mm256_store_ps(
                            temp,
                            acc[k]
                        );

                        float sum = 0.0f;

                        for (int x = 0; x < 8; x++) {
                            sum += temp[x];
                        }

                        sum += scalar_acc[k];

                        C[i * ldc + (j + k)] = sum;
                    }
                }
            }
        }
    }
}


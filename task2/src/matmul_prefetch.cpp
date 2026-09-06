// matmul_prefetch.cpp
// STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

// 0 = OFF
// 1 = ON
#define SOFTWARE_PREFETCH 1

#define PREFETCH_DISTANCE 16

#define PREFETCH_LEVEL 3

static inline void software_prefetch(const float* ptr)
{
#if SOFTWARE_PREFETCH

#if PREFETCH_LEVEL == 3

    _mm_prefetch(
        reinterpret_cast<const char*>(ptr),
        _MM_HINT_T0
    );

#elif PREFETCH_LEVEL == 2

    _mm_prefetch(
        reinterpret_cast<const char*>(ptr),
        _MM_HINT_T1
    );

#elif PREFETCH_LEVEL == 1

    _mm_prefetch(
        reinterpret_cast<const char*>(ptr),
        _MM_HINT_T2
    );

#elif PREFETCH_LEVEL == 0

    _mm_prefetch(
        reinterpret_cast<const char*>(ptr),
        _MM_HINT_NTA
    );

#endif

#else
    (void)ptr;

#endif
}

void matmul_prefetch(const float* A,
                     const float* B,
                     float* C,
                     int M,
                     int N,
                     int K,
                     int lda,
                     int ldb,
                     int ldc)
{
    for (int i = 0; i < M; ++i) {

        for (int j = 0; j < N; ++j) {

            float acc = 0.0f;

            const float* a =
                A + static_cast<long>(i) * lda;

            const float* b =
                B + static_cast<long>(j) * ldb;


            
            for (int p = 0; p < K; ++p) {

#if SOFTWARE_PREFETCH

            
                const int pf =
                    p + PREFETCH_DISTANCE;

                if (pf < K) {
                    software_prefetch(&a[pf]);
                    software_prefetch(&b[pf]);
                }

#endif

                acc += a[p] * b[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
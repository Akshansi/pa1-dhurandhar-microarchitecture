// matmul_prefetch.cpp
// STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

// 0 = Software prefetch OFF
// 1 = Software prefetch ON
#define SOFTWARE_PREFETCH 1

#define PREFETCH_DISTANCE 16

// Cache fill:
// 0 = NTA
// 1 = L3
// 2 = L2
// 3 = L1
#define PREFETCH_LOCALITY 3


static inline void software_prefetch(const float* ptr)
{
#if SOFTWARE_PREFETCH

    __builtin_prefetch(
        ptr,
        0,                 
        PREFETCH_LOCALITY  
    );

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

                const int pf = p + PREFETCH_DISTANCE;

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
// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

#define SIMD_WIDTH 128

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    // conv_naive(in, out, ker, H, W, K);

    const int p = K / 2;            //padding for kernel
    const int stride = W + 2 * p;  //stride for input matrix

#if SIMD_WIDTH == 128   

    //processing 4 floats at a time
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j += 4) {

            __m128 sum = _mm_setzero_ps();

            //applying evry kernel elem to curr tile
            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {


                    __m128 input = _mm_loadu_ps(                
                        &in[(i + ki) * stride + (j + kj)]
                    );

                    __m128 weight = _mm_set1_ps(
                        ker[ki * K + kj]
                    );

                    sum = _mm_fmadd_ps(input, weight, sum); // fused multiply-add
                }
            }

            _mm_storeu_ps(&out[i * W + j], sum); 
        }
    }


//same for 256 and 512 bit SIMD widths

#elif SIMD_WIDTH == 256

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j += 8) {

            __m256 sum = _mm256_setzero_ps();

            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {

                    __m256 input = _mm256_loadu_ps(
                        &in[(i + ki) * stride + (j + kj)]
                    );

                    __m256 weight = _mm256_set1_ps(
                        ker[ki * K + kj]
                    );

                    sum = _mm256_fmadd_ps(input, weight, sum);
                }
            }

            _mm256_storeu_ps(&out[i * W + j], sum);
        }
    }

#elif SIMD_WIDTH == 512

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j += 16) {

            __m512 sum = _mm512_setzero_ps();

            for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {

                    __m512 input = _mm512_loadu_ps(
                        &in[(i + ki) * stride + (j + kj)]
                    );

                    __m512 weight = _mm512_set1_ps(
                        ker[ki * K + kj]
                    );

                    sum = _mm512_fmadd_ps(input, weight, sum);
                }
            }

            _mm512_storeu_ps(&out[i * W + j], sum);
        }
    }

#endif
}

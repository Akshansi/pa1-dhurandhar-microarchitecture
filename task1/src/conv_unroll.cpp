// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include <cstddef>
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    constexpr int UNROLL = 8;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + static_cast<std::size_t>(oy) * W;
        for (int ox = 0; ox < W; ++ox) out_row[ox] = 0.0f;
    }

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + static_cast<std::size_t>(oy) * W;

        for (int ky = 0; ky < K; ++ky) {
            const float* in_row = in + static_cast<std::size_t>(oy + ky) * in_stride;

            for (int kx = 0; kx < K; ++kx) {
                const float w = ker[ky * K + kx];
                const float* in_row_k = in_row + kx;

                int ox = 0;
                for (; ox + UNROLL <= W; ox += UNROLL) {
                    float t0 = out_row[ox + 0];
                    float t1 = out_row[ox + 1];
                    float t2 = out_row[ox + 2];
                    float t3 = out_row[ox + 3];
                    float t4 = out_row[ox + 4];
                    float t5 = out_row[ox + 5];
                    float t6 = out_row[ox + 6];
                    float t7 = out_row[ox + 7];

                    t0 += in_row_k[ox + 0] * w;
                    t1 += in_row_k[ox + 1] * w;
                    t2 += in_row_k[ox + 2] * w;
                    t3 += in_row_k[ox + 3] * w;
                    t4 += in_row_k[ox + 4] * w;
                    t5 += in_row_k[ox + 5] * w;
                    t6 += in_row_k[ox + 6] * w;
                    t7 += in_row_k[ox + 7] * w;

                    out_row[ox + 0] = t0;
                    out_row[ox + 1] = t1;
                    out_row[ox + 2] = t2;
                    out_row[ox + 3] = t3;
                    out_row[ox + 4] = t4;
                    out_row[ox + 5] = t5;
                    out_row[ox + 6] = t6;
                    out_row[ox + 7] = t7;
                }
                for (; ox < W; ++ox) out_row[ox] += in_row_k[ox] * w;
            }
        }
    }
}

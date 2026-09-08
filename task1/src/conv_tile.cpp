// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);

    constexpr int TILE = 128;
    const int p = K/2;
    const int in_stride = W + 2*p;
  
    //processing output tile by tile
    for (int ii = 0; ii < H; ii+=TILE) {
        for (int jj = 0; jj < W; jj +=TILE) {

            const int i_end = std::min(ii + TILE, H);
            const int j_end = std::min(jj + TILE, W);
    // initializing curr output tile.
            for (int i = ii; i < i_end; i++) {
                for (int j = jj; j < j_end;j++) {
                    out[i * W + j] = 0.0f;
                }
            }

    // apply every kernel element to the current tile.
            for (int ki = 0;ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {

                    const float w = ker[ki * K + kj];
                    for(int i =ii; i< i_end;i++) {
                        for(int j =jj; j< j_end;j++) {

                            out[i * W + j] += in[(i + ki) * in_stride + (j + kj)]* w;
                        }
                    }
                }
            }
        }
    }
}

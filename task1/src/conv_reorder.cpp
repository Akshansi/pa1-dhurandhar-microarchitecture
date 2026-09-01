// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // We want ox and kx to be the innermost but then we cannot use acc. Instead, we will use out directly.
    // Initialize the output array first for now (shall see improvements later)
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy * W + ox] = 0.0f;
        }
    }

    // We should keep ox and kx as innermost and oy outermost (for output locality)
    for (int oy = 0; oy < H; ++oy) {
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                // Precompute values used repeatedly
                float ker_val = ker[ky * K + kx];
                int in_row_offset = (oy + ky) * in_stride;
                int out_row_offset = oy * W;
                
                for (int ox = 0; ox < W; ++ox) {
                    out[out_row_offset + ox] += in[in_row_offset + (ox + kx)] * ker_val;
                }
            }
        }
    }
}
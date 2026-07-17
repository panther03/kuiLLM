// Probe the WMMA 16x16x16 fp32 accumulator fragment layout: which (row,col) of
// the stored 16x16 tile does each (lane, element-index) map to? We need this to
// rescale a register-resident O accumulator by a per-row softmax correction
// without a shared-memory round-trip. Encode value = row*100 + col into each
// element, store_matrix_sync row-major, then read back to recover the mapping.
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

__global__ void probe() {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    int lane = threadIdx.x % 32;
    // Set each element to a unique per-(lane,i) tag so we can find it post-store.
    for (int i = 0; i < acc.num_elements; ++i)
        acc.x[i] = lane * 10 + i;   // tag
    __shared__ float S[16 * 16];
    wmma::store_matrix_sync(S, acc, 16, wmma::mem_row_major);
    __syncthreads();
    if (lane == 0 && threadIdx.x < 32) {
        printf("num_elements=%d\n", acc.num_elements);
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                int tag = (int)S[r * 16 + c];
                int ln = tag / 10, i = tag % 10;
                printf("row=%2d col=%2d  <- lane=%2d elem=%d\n", r, c, ln, i);
            }
    }
}

int main() {
    probe<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}

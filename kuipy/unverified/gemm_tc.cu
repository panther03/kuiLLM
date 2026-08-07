// Instantiation table and launcher for the templated tensor-core GEMM in
// gemm_tc.cuh.
//
// Every tiling in GEMM_TC_CONFIG_LIST is instantiated for both element types
// and registered in a table indexed by a small integer, so a tiling can be
// picked at run time with no compilation. That is what lets kuipy's autotuner
// -- which selects a spec per (shape, dtype, GPU) -- drive an unverified kernel
// the same way it drives a verified one, without a JIT step per candidate.
//
// Which tilings are legal for a call is a property of the config alone
// (divisibility of M/N/K and the shared-memory budget), so the table is
// published to Python via gemm_tc_config_info and filtered there.
#include "gemm_tc.cuh"
#include "kernels.h"

namespace {

using namespace gemm_tc;

//    BM   BN  BK   WM  WN  ST SKEW
#define GEMM_TC_CONFIG_LIST(X)             \
    X(128, 128, 32,  64, 64,  3,  8)       \
    X(128, 128, 32,  64, 64,  4,  8)       \
    X(128, 128, 32,  64, 64,  2,  8)       \
    X(128, 128, 64,  64, 64,  2,  8)       \
    X(128, 128, 32,  32, 64,  3,  8)       \
    X(128, 128, 32,  64, 32,  3,  8)       \
    X(128, 128, 32,  32, 32,  3,  8)       \
    X(128, 128, 32,  32, 64,  2,  8)       \
    X(128, 128, 32,  64, 32,  2,  8)       \
    X(128, 128, 32,  32, 32,  2,  8)       \
    X(128, 256, 32,  64, 64,  2,  8)       \
    X(256, 128, 32,  64, 64,  2,  8)       \
    X(128, 256, 64,  64, 64,  2,  8)       \
    X(128, 256, 32,  64, 64,  3,  8)       \
    X(256, 128, 32,  64, 64,  3,  8)       \
    X(256, 128, 32,  64, 32,  3,  8)       \
    X(256,  64, 32,  64, 32,  3,  8)       \
    X(128,  64, 32,  64, 32,  3,  8)       \
    X(128,  64, 32,  32, 32,  3,  8)       \
    X( 64, 128, 32,  32, 64,  3,  8)       \
    X( 64, 128, 32,  32, 32,  3,  8)       \
    X( 64, 128, 64,  32, 64,  3,  8)       \
    X( 64,  64, 32,  32, 32,  4,  8)       \
    X( 64,  64, 32,  32, 32,  3,  8)       \
    X( 64,  64, 64,  32, 32,  3,  8)       \
    X( 64,  64, 64,  32, 32,  4,  8)       \
    X( 64, 128, 64,  32, 64,  2,  8)       \
    X(128,  64, 64,  64, 32,  2,  8)       \
    X( 32, 128, 32,  32, 64,  4,  8)       \
    X( 32, 128, 64,  32, 64,  3,  8)       \
    X( 64,  32, 32,  32, 32,  4,  8)       \
    X( 32,  64, 32,  32, 32,  4,  8)

template <typename T, int BM, int BN, int BK, int WM, int WN, int ST, int SKEW>
void launch_cfg(const Params& p, cudaStream_t stream) {
    using Tl = Tile<T, BM, BN, BK, WM, WN, ST, SKEW>;
    auto kernel = gemm_tc_kernel<T, BM, BN, BK, WM, WN, ST, SKEW>;
    // Past 48KB the opt-in has to be requested per kernel; doing it once here
    // keeps it off the launch path.
    static bool opted_in = false;
    if (!opted_in) {
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Tl::SMEM_BYTES);
        opted_in = true;
    }
    const dim3 grid((p.M / BM) * (p.N / BN), 1, p.splits);
    kernel<<<grid, Tl::THREADS, Tl::SMEM_BYTES, stream>>>(p);
}

struct Desc {
    int bm, bn, bk, wm, wn, stages, skew, warps, smem;
    void (*half_fn)(const Params&, cudaStream_t);
    void (*bf16_fn)(const Params&, cudaStream_t);
};

#define GEMM_TC_DESC(BM, BN, BK, WM, WN, ST, SKEW)                            \
    Desc{BM, BN, BK, WM, WN, ST, SKEW,                                        \
         Tile<half, BM, BN, BK, WM, WN, ST, SKEW>::WARPS,                     \
         Tile<half, BM, BN, BK, WM, WN, ST, SKEW>::SMEM_BYTES,                \
         &launch_cfg<half, BM, BN, BK, WM, WN, ST, SKEW>,                     \
         &launch_cfg<__nv_bfloat16, BM, BN, BK, WM, WN, ST, SKEW>},

const Desc kDescs[] = {GEMM_TC_CONFIG_LIST(GEMM_TC_DESC)};
constexpr int kNumConfigs = (int)(sizeof(kDescs) / sizeof(kDescs[0]));

}  // namespace

int gemm_tc_num_configs() { return kNumConfigs; }

void gemm_tc_config_info(int index, int* out) {
    const Desc& d = kDescs[index];
    out[0] = d.bm; out[1] = d.bn; out[2] = d.bk;
    out[3] = d.wm; out[4] = d.wn; out[5] = d.stages;
    out[6] = d.skew; out[7] = d.warps; out[8] = d.smem;
}

void gemm_tc_launch(bool bf16, int config, const void* A, const void* B,
                    const void* C, void* D, float* workspace, int M, int N,
                    int K, float alpha, float beta, int splits, int group,
                    int epi, cudaStream_t stream) {
    const Desc& d = kDescs[config];
    // Every split must get at least one k-tile: an empty one would contribute
    // an uninitialised workspace slice to the reduction.
    splits = max(1, min(splits, K / d.bk));
    Params p{};
    p.A = A; p.B = B; p.C = C; p.D = D; p.workspace = workspace;
    p.M = M; p.N = N; p.K = K;
    p.alpha = alpha; p.beta = beta;
    p.splits = splits; p.group = group; p.epi = epi;
    (bf16 ? d.bf16_fn : d.half_fn)(p, stream);
    if (splits > 1) {
        const int threads = 256;
        const int vec = bf16 ? 8 : 8;
        const int elems = (int)(((size_t)M * N + vec - 1) / vec);
        const int blocks = min(4096, (elems + threads - 1) / threads);
        if (bf16)
            gemm_tc_reduce_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                workspace, C, D, M, N, alpha, beta, splits, epi);
        else
            gemm_tc_reduce_kernel<half><<<blocks, threads, 0, stream>>>(
                workspace, C, D, M, N, alpha, beta, splits, epi);
    }
}

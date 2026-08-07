// Standalone driver for gemm_tc.cuh: correctness against a cuBLAS reference and
// a sweep of every registered tiling over the bench_ops.ipynb shapes. Not part
// of the extension; it exists so the kernel can be iterated on without a torch
// import, and it is what to reach for when adding a tiling to the table.
//
//   nvcc -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr \
//        -gencode=arch=compute_86,code=sm_86 -I kuipy/unverified \
//        -o /tmp/gemm_tc_bench etc/gemm_tc_bench.cu kuipy/unverified/gemm_tc.cu -lcublas
//
// argv is the epilogue mode (0 none, 1 matrix, 2 broadcast vector) followed by
// case indices; SPLITS, GROUPS, TOP, F16ONLY and BF16ONLY narrow the sweep.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "gemm_tc.cuh"
#include "kernels.h"

#define CK(x)                                                             \
    do {                                                                  \
        cudaError_t e = (x);                                              \
        if (e != cudaSuccess) {                                           \
            printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

template <typename T>
__global__ void fill(T* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        unsigned h = (unsigned)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = gemm_tc::from_float(((float)(h & 0xffff) / 65535.0f - 0.5f) * 0.2f,
                                   T());
    }
}

template <typename T>
__global__ void diff(const T* a, const T* b, size_t n, double* num, double* den) {
    __shared__ double sn[256], sd[256];
    double ln = 0, ld = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        double x = gemm_tc::to_float(a[i]), y = gemm_tc::to_float(b[i]);
        ln += (x - y) * (x - y);
        ld += y * y;
    }
    sn[threadIdx.x] = ln; sd[threadIdx.x] = ld;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sn[threadIdx.x] += sn[threadIdx.x + s];
            sd[threadIdx.x] += sd[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { num[blockIdx.x] = sn[0]; den[blockIdx.x] = sd[0]; }
}

struct Case { const char* name; int M, K, N; };

static std::vector<int> kSplits = {1, 2, 4, 8};
static std::vector<int> kGroups = {1, 4, 8, 16};
static int kTop = 5;
static std::vector<int> kOnly;

static const Case kCases[] = {
    {"o_proj", 256, 896, 896},     {"gate_proj", 256, 896, 4864},
    {"down_proj", 256, 4864, 896}, {"lm_head", 256, 896, 151936},
    {"square_4096", 4096, 4096, 4096},
    {"square_1024", 1024, 1024, 1024},
    {"square_2048", 2048, 2048, 2048},
    {"square_8192", 8192, 8192, 8192},
};

template <typename T>
struct Traits;
template <>
struct Traits<half> {
    static constexpr bool bf16 = false;
    static constexpr cudaDataType dt = CUDA_R_16F;
};
template <>
struct Traits<__nv_bfloat16> {
    static constexpr bool bf16 = true;
    static constexpr cudaDataType dt = CUDA_R_16BF;
};

static cublasHandle_t g_cublas;

template <typename T>
static void cublas_ref(const T* A, const T* B, T* D, int M, int N, int K,
                       float alpha, float beta) {
    // Row-major (M,K)x(K,N) == column-major (N,K)x(K,M).
    cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                 Traits<T>::dt, N, A, Traits<T>::dt, K, &beta, D,
                 Traits<T>::dt, N, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static bool legal(const int* info, int M, int N, int K, int splits, int smem_cap) {
    const int bm = info[0], bn = info[1], bk = info[2];
    if (M % bm || N % bn || K % bk) return false;
    if (info[8] > smem_cap) return false;
    const int ktiles = K / bk;
    if (splits > ktiles) return false;
    return true;
}

template <typename T>
static void run(int epi) {
    const int ncfg = gemm_tc_num_configs();
    int smem_cap = 0;
    CK(cudaDeviceGetAttribute(&smem_cap,
                              cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    int ci = -1;
    for (const Case& c : kCases) {
        ci++;
        if (!kOnly.empty() &&
            std::find(kOnly.begin(), kOnly.end(), ci) == kOnly.end())
            continue;
        const size_t na = (size_t)c.M * c.K, nb = (size_t)c.K * c.N,
                     nd = (size_t)c.M * c.N;
        T *A, *B, *C, *D, *R;
        CK(cudaMalloc(&A, na * sizeof(T)));
        CK(cudaMalloc(&B, nb * sizeof(T)));
        CK(cudaMalloc(&C, nd * sizeof(T)));
        CK(cudaMalloc(&D, nd * sizeof(T)));
        CK(cudaMalloc(&R, nd * sizeof(T)));
        fill<<<256, 256>>>(A, na, 1);
        fill<<<256, 256>>>(B, nb, 2);
        fill<<<256, 256>>>(C, nd, 3);
        float* ws = nullptr;
        const int max_splits = 16;
        CK(cudaMalloc(&ws, nd * max_splits * sizeof(float)));

        const float alpha = (epi == gemm_tc::EPI_NONE) ? 1.0f : 0.75f;
        const float beta = (epi == gemm_tc::EPI_NONE) ? 0.0f : 1.5f;
        CK(cudaMemcpy(R, C, nd * sizeof(T), cudaMemcpyDeviceToDevice));
        cublas_ref<T>(A, B, R, c.M, c.N, c.K, alpha, beta);
        CK(cudaDeviceSynchronize());

        // cuBLAS timing.
        double cub_ms = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            cudaEvent_t ev0, ev1;
            CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
            for (int i = 0; i < 5; i++)
                cublas_ref<T>(A, B, D, c.M, c.N, c.K, alpha, beta);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(ev0));
            for (int i = 0; i < 30; i++)
                cublas_ref<T>(A, B, D, c.M, c.N, c.K, alpha, beta);
            CK(cudaEventRecord(ev1));
            CK(cudaEventSynchronize(ev1));
            float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
            CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
            if (ms / 30 < cub_ms) cub_ms = ms / 30;
        }
        const double flops = 2.0 * c.M * c.N * c.K;
        printf("\n== %s %dx%dx%d  epi=%d  %s   cublas %8.1f us  %6.2f TF/s\n",
               c.name, c.M, c.K, c.N, epi, Traits<T>::bf16 ? "bf16" : "f16",
               cub_ms * 1e3, flops / (cub_ms * 1e-3) / 1e12);

        std::vector<std::pair<double, std::string>> results;
        for (int cfg = 0; cfg < ncfg; cfg++) {
            int info[9];
            gemm_tc_config_info(cfg, info);
            for (int splits : kSplits)
            for (int group : kGroups) {
                if (!legal(info, c.M, c.N, c.K, splits, smem_cap)) continue;
                const long blocks =
                    (long)(c.M / info[0]) * (c.N / info[1]) * splits;
                if (blocks > 400000) continue;
                CK(cudaMemset(D, 0, nd * sizeof(T)));
                gemm_tc_launch(Traits<T>::bf16, cfg, A, B, C, D, ws, c.M, c.N,
                               c.K, alpha, beta, splits, group, epi, 0);
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    printf("  cfg%-2d s%-2d LAUNCH FAIL %s\n", cfg, splits,
                           cudaGetErrorString(err));
                    cudaGetLastError();
                    continue;
                }
                double *dn, *dd;
                CK(cudaMalloc(&dn, 256 * sizeof(double)));
                CK(cudaMalloc(&dd, 256 * sizeof(double)));
                diff<<<256, 256>>>(D, R, nd, dn, dd);
                std::vector<double> hn(256), hd(256);
                CK(cudaMemcpy(hn.data(), dn, 256 * sizeof(double),
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hd.data(), dd, 256 * sizeof(double),
                              cudaMemcpyDeviceToHost));
                double sn = 0, sd = 0;
                for (int i = 0; i < 256; i++) { sn += hn[i]; sd += hd[i]; }
                const double rel = sqrt(sn) / (sqrt(sd) + 1e-30);
                CK(cudaFree(dn)); CK(cudaFree(dd));

                double ms = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    cudaEvent_t ev0, ev1;
                    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
                    for (int i = 0; i < 5; i++)
                        gemm_tc_launch(Traits<T>::bf16, cfg, A, B, C, D, ws,
                                       c.M, c.N, c.K, alpha, beta, splits,
                                       group, epi, 0);
                    CK(cudaDeviceSynchronize());
                    CK(cudaEventRecord(ev0));
                    for (int i = 0; i < 30; i++)
                        gemm_tc_launch(Traits<T>::bf16, cfg, A, B, C, D, ws,
                                       c.M, c.N, c.K, alpha, beta, splits,
                                       group, epi, 0);
                    CK(cudaEventRecord(ev1));
                    CK(cudaEventSynchronize(ev1));
                    float t; CK(cudaEventElapsedTime(&t, ev0, ev1));
                    CK(cudaEventDestroy(ev0)); CK(cudaEventDestroy(ev1));
                    if (t / 30 < ms) ms = t / 30;
                }
                char tag[160];
                snprintf(tag, sizeof(tag),
                         "cfg%-2d %3dx%3dx%2d w%2dx%2d st%d split%d grp%-2d "
                         "smem%5d", cfg, info[0], info[1], info[2], info[3],
                         info[4], info[5], splits, group, info[8]);
                if (rel > 5e-2) {
                    printf("  %s  WRONG rel %.2e\n", tag, rel);
                    continue;
                }
                results.push_back({ms, tag});
            }
        }
        std::sort(results.begin(), results.end());
        for (int i = 0; i < (int)results.size() && i < kTop; i++)
            printf("  %s  %8.1f us  %6.2f TF/s  (%.2fx cublas)\n",
                   results[i].second.c_str(), results[i].first * 1e3,
                   flops / (results[i].first * 1e-3) / 1e12,
                   cub_ms / results[i].first);
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CK(cudaFree(D));
        CK(cudaFree(R)); CK(cudaFree(ws));
    }
}

int main(int argc, char** argv) {
    cublasCreate(&g_cublas);
    cublasSetMathMode(g_cublas, CUBLAS_TENSOR_OP_MATH);
    const int epi = argc > 1 ? atoi(argv[1]) : 0;
    if (argc > 2) { kOnly.clear(); for (int i = 2; i < argc; i++) kOnly.push_back(atoi(argv[i])); }
    if (getenv("SPLITS")) { kSplits.clear(); kSplits.push_back(atoi(getenv("SPLITS"))); }
    if (getenv("GROUPS")) { kGroups.clear(); kGroups.push_back(atoi(getenv("GROUPS"))); }
    if (getenv("TOP")) kTop = atoi(getenv("TOP"));
    if (!getenv("F16ONLY")) run<__nv_bfloat16>(epi);
    if (!getenv("BF16ONLY")) run<half>(epi);
    return 0;
}

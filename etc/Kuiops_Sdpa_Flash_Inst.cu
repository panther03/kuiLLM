
#include "Kuiops_Sdpa_Flash_Inst.h"

__device__ static uint32_t szlt16(uint32_t i)
{
    return i;
}

__global__
/**
  hoisted when extracting sdpa_flash_bf16_f32
*/
static void
__hoisted_sdpa_flash_bf16_f32_0(uint32_t nw,
                                uint32_t nthr,
                                uint32_t hq,
                                uint32_t hkv,
                                uint32_t group,
                                uint32_t sq,
                                uint32_t rows,
                                uint32_t tiles,
                                uint32_t sk,
                                uint32_t d,
                                __nv_bfloat16 *gQ,
                                __nv_bfloat16 *gK,
                                __nv_bfloat16 *gV,
                                __nv_bfloat16 *gmask,
                                __nv_bfloat16 *gout, bool causal, float scale)
{
    uint32_t bh = blockIdx.x / tiles;
    uint32_t kvh = bh % hkv;
    uint32_t bi = bh / hkv;
    uint32_t r0 = blockIdx.x % tiles * 16U;
    __nv_bfloat16 *gKkv = gK;
    __nv_bfloat16 *gVkv = gV;
    uint32_t ncells = 16U * d;
    uint32_t idx0 = threadIdx.x;
    uint32_t iter0 = 0U;
    for (; idx0 < ncells; iter0++) {
        uint32_t flat = idx0;
        uint32_t i = flat / d;
        uint32_t dd = flat % d;
        uint32_t r = r0 + i;
        uint32_t rr = r < rows ? r : 0U;
        uint32_t qh0 = kvh * group + rr / sq;
        __nv_bfloat16 qread =
            gQ[bi * hq * sq * d + (qh0 < hq ? qh0 : 0U) * sq * d + rr % sq * d +
               dd];
        __nv_bfloat16 qv = r < rows ? qread : __float2bfloat16(0.0f);
        uint32_t ni = i * d + dd;
        ((__nv_bfloat16 *) KPR_SHMEM_AT(0U))[ni] = qv;
        idx0 += nthr;
    }
    if (threadIdx.x % 32U < 16U) {
        ((float *)
         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                      4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                      4U * nw * 16U * 16U +
                      4U * nw * 16U))[threadIdx.x / 32U * 16U +
                                      (threadIdx.x % 32U <
                                       16U ? threadIdx.x % 32U : 0U)]
            = 0.0f - INFINITY;
        ((float *)
         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                      4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                      4U * nw * 16U * 16U + 4U * nw * 16U +
                      4U * nw * 16U))[threadIdx.x / 32U * 16U +
                                      (threadIdx.x % 32U <
                                       16U ? threadIdx.x % 32U : 0U)]
            = 0.0f;
    }
    idx0 = threadIdx.x % 32U;
    iter0 = 0U;
    for (; idx0 < ncells; iter0++) {
        uint32_t flat = idx0;
        uint32_t ni = (16U * (threadIdx.x / 32U) + flat / d) * d + flat % d;
        ((float *)
         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                      4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                      4U * nw * 16U * 16U + 4U * nw * 16U + 4U * nw * 16U +
                      4U * nw * 16U + 4U * nw * 16U))[ni]
            = 0.0f;
        idx0 += 32U;
    }
    __syncthreads();
    uint32_t r = r0 + (threadIdx.x % 32U < 16U ? threadIdx.x % 32U : 0U);
    uint32_t rr = r < rows ? r : 0U;
    uint32_t qh0 = kvh * group + rr / sq;
    uint32_t qh = qh0 < hq ? qh0 : 0U;
    uint32_t qpos = rr % sq;
    bool row_active = r < rows;
    uint32_t cbound = qpos + (sk - sq);
    uint32_t kmax;
    if (causal) {
        uint32_t maxpos = 0U;
        bool found = false;
        uint32_t i = 0U;
        for (; i < 16U; i++) {
            uint32_t r1 = r0 + i;
            bool valid = r1 < rows;
            uint32_t pos = r1 % sq;
            maxpos = valid && (!found || pos > maxpos) ? pos : maxpos;
            found = found || valid;
        }
        uint32_t base = sk - sq;
        uint32_t __anf0 = maxpos;
        uint32_t extra = found ? __anf0 + 1U : 0U;
        kmax = sk < base + extra ? sk : base + extra;
    } else
        kmax = sk;
    uint32_t nkt = kmax / 16U + (uint32_t) (kmax % 16U != 0U);
    uint32_t jt = threadIdx.x / 32U;
    uint32_t iter = 0U;
    for (; jt < nkt; iter++) {
        uint32_t k0 = jt * 16U;
        uint32_t ncol = d / 16U;
        uint32_t a = 0U;
        for (; a < 8U; a++) {
            uint32_t arow = a;
            uint32_t trow = 2U * arow + threadIdx.x % 32U / 16U;
            uint32_t kr = k0 + trow < sk ? k0 + trow : 0U;
            uint32_t b1 = 0U;
            for (; b1 < ncol; b1++) {
                uint32_t bcol = b1;
                uint32_t dd = 16U * bcol + threadIdx.x % 32U % 16U;
                __nv_bfloat16 vk =
                    gKkv[bi * hkv * sk * d + kvh * sk * d + kr * d + dd];
                uint32_t ni0 =
                    (16U * (threadIdx.x / 32U) + threadIdx.x % 32U / 16U +
                     arow * 2U) * d + threadIdx.x % 32U % 16U + bcol * 16U;
                ((__nv_bfloat16 *) KPR_SHMEM_AT(2U * 16U * d))[ni0] = vk;
                __nv_bfloat16 vv =
                    gVkv[bi * hkv * sk * d + kvh * sk * d + kr * d + dd];
                uint32_t ni =
                    (16U * (threadIdx.x / 32U) + threadIdx.x % 32U / 16U +
                     arow * 2U) * d + threadIdx.x % 32U % 16U + bcol * 16U;
                ((__nv_bfloat16 *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d))[ni] = vv;
            }
        }
        __syncwarp();
        auto &
            qFrag =
            KPR_INIT(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major));
        auto & kFrag =
            KPR_INIT(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::col_major));
        auto & sFrag =
            KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float));
        wmma::fill_fragment(sFrag, 0.0f);
        uint32_t nchunks = d / 16U;
        uint32_t chunk = 0U;
        for (; chunk < nchunks; chunk++) {
            uint32_t __anf0 = chunk;
            __nv_bfloat16 *qtile = (__nv_bfloat16 *) KPR_SHMEM_AT(0U);
            uint32_t __anf01 = chunk;
            __nv_bfloat16 *ktile = (__nv_bfloat16 *) KPR_SHMEM_AT(2U * 16U * d);
            wmma::load_matrix_sync(qFrag, qtile + __anf0 * 16U, d);
            wmma::load_matrix_sync(kFrag,
                                   ktile + (d * (threadIdx.x / 32U) * 16U +
                                            __anf01 * 16U), d);
            wmma::mma_sync(sFrag, qFrag, kFrag, sFrag);
        }
        wmma::
            store_matrix_sync((float *)
                              KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                           2U * nw * 16U * d)
                              + 16U * (threadIdx.x / 32U) * 16U, sFrag, 16U,
                              wmma::mem_row_major);
        __syncwarp();
        __syncwarp();
        if (threadIdx.x % 32U < 16U) {
            float
            *rm =
                (float *)KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                      2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                      2U * nw * 16U * 16U +
                                      4U * nw * 16U * 16U + 4U * nw * 16U)
                + (threadIdx.x / 32U * 16U + szlt16(threadIdx.x % 32U));
            float
            *rl =
                (float *)KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                      2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                      2U * nw * 16U * 16U +
                                      4U * nw * 16U * 16U + 4U * nw * 16U +
                                      4U * nw * 16U)
                + (threadIdx.x / 32U * 16U + szlt16(threadIdx.x % 32U));
            float
            *rcw =
                (float *)KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                      2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                      2U * nw * 16U * 16U + 4U * nw * 16U * 16U)
                + (threadIdx.x / 32U * 16U + szlt16(threadIdx.x % 32U));
            float rowmax = 0.0f - INFINITY;
            uint32_t j = 0U;
            for (; j < 16U; j++) {
                uint32_t vj = j;
                uint32_t jj = vj;
                uint32_t kj = k0 + vj;
                uint32_t ni0 =
                    (16U * (threadIdx.x / 32U) +
                     szlt16(threadIdx.x % 32U)) * 16U + jj;
                float
                 sc =
                    ((float *)
                     KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                  2U * nw * 16U * d))[ni0];
                __nv_bfloat16 mv =
                    gmask[bi * hq * sq * sk + qh * sq * sk + qpos * sk +
                          (kj < sk ? kj : 0U)];
                float s1;
                if (row_active && !(causal && kj > cbound) && kj < sk)
                    s1 = sc * scale + __bfloat162float(mv);
                else
                    s1 = 0.0f - INFINITY;
                uint32_t ni =
                    (16U * (threadIdx.x / 32U) +
                     szlt16(threadIdx.x % 32U)) * 16U + jj;
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d))[ni] = s1;
                rowmax = fmaxf(rowmax, s1);
            }
            float m_old = *rm;
            float mnew = fmaxf(m_old, rowmax);
            float corr = expf(m_old - mnew);
            float rowsum = 0.0f;
            j = 0U;
            for (; j < 16U; j++) {
                uint32_t jj = j;
                uint32_t ni0 =
                    (16U * (threadIdx.x / 32U) +
                     szlt16(threadIdx.x % 32U)) * 16U + jj;
                float
                 sv =
                    ((float *)
                     KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                  2U * nw * 16U * d))[ni0];
                float p;
                if (sv == 0.0f - INFINITY)
                    p = 0.0f;
                else
                    p = expf(sv - mnew);
                uint32_t ni =
                    (16U * (threadIdx.x / 32U) +
                     szlt16(threadIdx.x % 32U)) * 16U + jj;
                ((__nv_bfloat16 *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U))[ni]
                    = __float2bfloat16(p);
                rowsum += p;
            }
            *rl = *rl * corr + rowsum;
            *rcw = corr;
            *rm = mnew;
        }
        __syncwarp();
        uint32_t ncol0 = d / 16U;
        uint32_t orow0 = 0U;
        for (; orow0 < 8U; orow0++) {
            uint32_t vor = orow0;
            uint32_t ni0 =
                threadIdx.x / 32U * 16U + 2U * vor + threadIdx.x % 32U / 16U;
            float
             cwv =
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U +
                              2U * nw * 16U * 16U + 4U * nw * 16U * 16U))[ni0];
            uint32_t ocol = 0U;
            for (; ocol < ncol0; ocol++) {
                uint32_t oc = ocol;
                uint32_t
                    ni0 =
                    (16U * (threadIdx.x / 32U) + threadIdx.x % 32U / 16U +
                     vor * 2U) * d + threadIdx.x % 32U % 16U + oc * 16U;
                float
                 ov =
                    ((float *)
                     KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                  2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                  2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                                  4U * nw * 16U + 4U * nw * 16U +
                                  4U * nw * 16U + 4U * nw * 16U))[ni0];
                uint32_t ni =
                    (16U * (threadIdx.x / 32U) + threadIdx.x % 32U / 16U +
                     vor * 2U) * d + threadIdx.x % 32U % 16U + oc * 16U;
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U +
                              2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                              4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U +
                              4U * nw * 16U))[ni]
                    = ov * cwv;
            }
        }
        __syncwarp();
        auto &
            pf =
            KPR_INIT(kpr_fragment
                     (wmma::matrix_a, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major));
        auto & vf =
            KPR_INIT(kpr_fragment
                     (wmma::matrix_b, 16U, 16U, 16U, __nv_bfloat16,
                      wmma::row_major));
        auto & pvacc =
            KPR_INIT(kpr_fragment(wmma::accumulator, 16U, 16U, 16U, float));
        uint32_t njcol = d / 16U;
        uint32_t jcol = 0U;
        for (; jcol < njcol; jcol++) {
            uint32_t vjcol = jcol;
            uint32_t ocol = vjcol;
            wmma::fill_fragment(pvacc, 0.0f);
            __nv_bfloat16 *vtile =
                (__nv_bfloat16 *) KPR_SHMEM_AT(2U * 16U * d +
                                               2U * nw * 16U * d);
            wmma::load_matrix_sync(pf,
                                   (__nv_bfloat16 *) KPR_SHMEM_AT(2U * 16U * d +
                                                                  2U * nw *
                                                                  16U * d +
                                                                  2U * nw *
                                                                  16U * d +
                                                                  4U * nw *
                                                                  16U * 16U)
                                   + 16U * (threadIdx.x / 32U) * 16U, 16U);
            wmma::load_matrix_sync(vf,
                                   vtile + (d * (threadIdx.x / 32U) * 16U +
                                            vjcol * 16U), d);
            wmma::mma_sync(pvacc, pf, vf, pvacc);
            wmma::
                store_matrix_sync((float *)
                                  KPR_SHMEM_AT(2U * 16U * d +
                                               2U * nw * 16U * d +
                                               2U * nw * 16U * d +
                                               4U * nw * 16U * 16U +
                                               2U * nw * 16U * 16U)
                                  + 16U * (threadIdx.x / 32U) * 16U, pvacc, 16U,
                                  wmma::mem_row_major);
            __syncwarp();
            __syncwarp();
            uint32_t k = 0U;
            for (; k < 8U; k++) {
                uint32_t vk = k;
                uint32_t orow = vk;
                uint32_t
                    ni0 =
                    (16U * (threadIdx.x / 32U) + 2U * vk +
                     threadIdx.x % 32U / 16U) * 16U + threadIdx.x % 32U % 16U;
                float
                 pv =
                    ((float *)
                     KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                  2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                  2U * nw * 16U * 16U))[ni0];
                uint32_t ni1 =
                    (16U * (threadIdx.x / 32U) + threadIdx.x % 32U / 16U +
                     orow * 2U) * d + threadIdx.x % 32U % 16U + ocol * 16U;
                float
                 old =
                    ((float *)
                     KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                  2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                  2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                                  4U * nw * 16U + 4U * nw * 16U +
                                  4U * nw * 16U + 4U * nw * 16U))[ni1];
                uint32_t ni =
                    (16U * (threadIdx.x / 32U) + threadIdx.x % 32U / 16U +
                     orow * 2U) * d + threadIdx.x % 32U % 16U + ocol * 16U;
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U +
                              2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                              4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U +
                              4U * nw * 16U))[ni]
                    = old + pv;
            }
        }
        __syncwarp();
        jt += nw;
    }
    __syncthreads();
    if (threadIdx.x / 32U == 0U && threadIdx.x % 32U < 16U) {
        float gm = 0.0f - INFINITY;
        uint32_t ww = 0U;
        for (; ww < nw; ww++) {
            uint32_t iw = ww;
            uint32_t ni =
                iw * 16U + (threadIdx.x % 32U < 16U ? threadIdx.x % 32U : 0U);
            float
             mv =
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U +
                              2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                              4U * nw * 16U))[ni];
            gm = fmaxf(gm, mv);
        }
        float gl = 0.0f;
        ww = 0U;
        for (; ww < nw; ww++) {
            uint32_t iw = ww;
            uint32_t ni0 =
                iw * 16U + (threadIdx.x % 32U < 16U ? threadIdx.x % 32U : 0U);
            float
             mv =
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U +
                              2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                              4U * nw * 16U))[ni0];
            uint32_t ni1 =
                iw * 16U + (threadIdx.x % 32U < 16U ? threadIdx.x % 32U : 0U);
            float
             lv =
                ((float *)
                 KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                              2U * nw * 16U * d + 4U * nw * 16U * 16U +
                              2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                              4U * nw * 16U + 4U * nw * 16U))[ni1];
            float sc = expf(mv - gm);
            uint32_t ni =
                iw * 16U + (threadIdx.x % 32U < 16U ? threadIdx.x % 32U : 0U);
            ((float *)
             KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                          4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                          4U * nw * 16U * 16U + 4U * nw * 16U + 4U * nw * 16U +
                          4U * nw * 16U))[ni]
                = sc;
            gl += sc * lv;
        }
        float __anf0 = gm;
        ((float *)
         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                      4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                      4U * nw * 16U * 16U + 4U * nw * 16U + 4U * nw * 16U +
                      4U * nw * 16U + 4U * nw * 16U +
                      4U * nw * 16U * d))[threadIdx.x % 32U <
                                          16U ? threadIdx.x % 32U : 0U]
            = __anf0;
        float __anf01 = gl;
        ((float *)
         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                      4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                      4U * nw * 16U * 16U + 4U * nw * 16U + 4U * nw * 16U +
                      4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U * d +
                      64U))[threadIdx.x % 32U < 16U ? threadIdx.x % 32U : 0U]
            = __anf01;
    }
    __syncthreads();
    if (threadIdx.x / 32U == 0U) {
        uint32_t ncells = 16U * d;
        uint32_t idx = threadIdx.x % 32U;
        uint32_t iter1 = 0U;
        for (; idx < ncells; iter1++) {
            uint32_t flat = idx;
            uint32_t i = flat / d;
            uint32_t dd = flat % d;
            uint32_t r1 = r0 + i;
            if (r1 < rows) {
                uint32_t rr1 = r1 < rows ? r1 : 0U;
                float acc = 0.0f;
                uint32_t ww = 0U;
                for (; ww < nw; ww++) {
                    uint32_t iw = ww;
                    uint32_t orow = 16U * iw + i;
                    uint32_t ni0 = iw * 16U + i;
                    float
                     sv =
                        ((float *)
                         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                      2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                      2U * nw * 16U * 16U +
                                      4U * nw * 16U * 16U + 4U * nw * 16U +
                                      4U * nw * 16U + 4U * nw * 16U))[ni0];
                    uint32_t ni = orow * d + dd;
                    float
                     ov =
                        ((float *)
                         KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                      2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                      2U * nw * 16U * 16U +
                                      4U * nw * 16U * 16U + 4U * nw * 16U +
                                      4U * nw * 16U + 4U * nw * 16U +
                                      4U * nw * 16U))[ni];
                    acc += sv * ov;
                }
                uint32_t ni = i;
                float
                 lv =
                    ((float *)
                     KPR_SHMEM_AT(2U * 16U * d + 2U * nw * 16U * d +
                                  2U * nw * 16U * d + 4U * nw * 16U * 16U +
                                  2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
                                  4U * nw * 16U + 4U * nw * 16U +
                                  4U * nw * 16U + 4U * nw * 16U +
                                  4U * nw * 16U * d + 64U))[ni];
                float inv = 0.0f < lv ? 1.0f / lv : 0.0f;
                uint32_t qh01 = kvh * group + rr1 / sq;
                uint32_t ni0 =
                    bi * hq * sq * d + (qh01 <
                                        hq ? qh01 : 0U) * sq * d +
                    rr1 % sq * d + dd;
                gout[ni0] = __float2bfloat16(acc * inv);
            }
            idx += 32U;
        }
    }
}

void
Kuiops_Sdpa_Flash_Inst_sdpa_flash_bf16_f32(uint32_t nblk,
                                           uint32_t nw,
                                           uint32_t nthr,
                                           uint32_t b,
                                           uint32_t hq,
                                           uint32_t hkv,
                                           uint32_t group,
                                           uint32_t sq,
                                           uint32_t rows,
                                           uint32_t tiles,
                                           uint32_t sk,
                                           uint32_t d,
                                           __nv_bfloat16 *gQ,
                                           __nv_bfloat16 *gK,
                                           __nv_bfloat16 *gV,
                                           __nv_bfloat16 *gmask,
                                           __nv_bfloat16 *gout,
                                           bool causal,
                                           float scale, cudaStream_t s)
{
    KRML_MAYBE_UNUSED_VAR(b);
    KPR_SHMEM_FITS(2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
                   4U * nw * 16U * 16U + 2U * nw * 16U * 16U +
                   4U * nw * 16U * 16U + 4U * nw * 16U + 4U * nw * 16U +
                   4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U * d + 64U +
                   64U);
    MUST(cudaFuncSetAttribute
         (__hoisted_sdpa_flash_bf16_f32_0,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
          4U * nw * 16U * 16U + 2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
          4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U +
          4U * nw * 16U * d + 64U + 64U));
    KPR_KCALL(__hoisted_sdpa_flash_bf16_f32_0, nblk, nthr,
              2U * 16U * d + 2U * nw * 16U * d + 2U * nw * 16U * d +
              4U * nw * 16U * 16U + 2U * nw * 16U * 16U + 4U * nw * 16U * 16U +
              4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U + 4U * nw * 16U +
              4U * nw * 16U * d + 64U + 64U, s, nw, nthr, hq, hkv, group, sq,
              rows, tiles, sk, d, gQ, gK, gV, gmask, gout, causal, scale);
}

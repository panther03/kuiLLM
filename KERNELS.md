# Kernel implementation checklist

| Op | Input dtypes | Output dtype | Calls | kuiper impl. | connected ? | 
| --- | --- | --- | ---: | :---: | --- |
| `aten::scaled_dot_product_attention` | `bf16,bf16,bf16` | `bf16` | 24 |  |
| `aten::is_nonzero` | `b8` | `?` | 1 | |
| `aten::add`        | `bf16,bf16` | `bf16` | 96 |  |
| `aten::add`        | `f32,f64` | `?` | 49 | |
| `aten::add`        | `long int,long int` | `?` | 1 | |
| `aten::addmm`      | `bf16,bf16,bf16` | `bf16` | 72 | Klas_GEMM_BlockTiling2D (need to add extraction + ext. output cast) |
| `aten::all`        | `b8` | `b8` | 1 |  |
| `aten::arange`     | `long int` | `?` | 1 | |
| `aten::bmm`        | `f32,f32` | `f32` | 1 | Klas_GEMM_Batched | yes |
| `aten::cat`        | `TensorList` (dont know arity..) | `?` | 97 | |
| `aten::to`         | `bf16,f32` | `?` | 51 | |
| `aten::to`         | `f32,bf16` | `?` | 49 | |
| `aten::to`         | `bf16,bf16` | `?` | 48 | |
| `aten::to`         | `b8,long int` | `?` | 1 | |
| `aten::to`         | `f32,long int` | `?` | 1 | |
| `aten::cos`        | `f32` | `f32` | 1 | |
| `aten::gather`     | `bf16,long int,bf16` | `?` | 1 | |
| `aten::mean`       | `f32` | `f32` | 49 | |
| `aten::mm`         | `bf16,bf16` | `bf16` | 97 | Klas_GEMM_TensorCore2D (output cast req.) | yes |
| `aten::mul`        | `bf16,bf16` | `bf16` | 169 | |
| `aten::mul`        | `f32,f32` | `f32` | 49 | |
| `aten::mul`        | `f32,f64` | `?` | 2 | |
| `aten::neg`        | `bf16` | `bf16` | 48 | |
| `aten::pow`        | `f32` | `f32` | 49 | |
| `aten::rsqrt`      | `f32` | `f32` | 49 | |
| `aten::silu`       | `bf16` | `bf16` | 24 | |
| `aten::sin`        | `f32` | `f32` | 1 | |


source:

```
Profile of Qwen/Qwen2.5-0.5B-Instruct  (CUDA time)
Kuiper: off
Total inclusive CUDA time at roots: 39.983 ms

NODE                                                                                               TOTAL          SELF   CALLS      %
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
├─ model.forward                                                                               39.983 ms        0.0 µs       1  100.0%
│  ├─ aten::linear [bf16,bf16 ×97 | bf16,bf16,bf16 ×72] ×169                                   38.342 ms        0.0 µs     169  95.9%
│  │  ├─ aten::matmul [bf16,bf16] ×97                                                          35.919 ms        0.0 µs      97  89.8%
│  │  │  ├─ aten::mm [bf16,bf16 → bf16] ×97                                                    35.919 ms        0.0 µs      97  89.8%
│  │  │  │  ├─ ⟦cuda⟧ nvjet_sm120_tst_mma_64x64x128_3_32x32x128_tmaAB_alignCD4_bz_TNNN ×48     15.470 ms     15.470 ms      48  38.7%
│  │  │  │  ├─ ⟦cuda⟧ cutlass[cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8]      10.316 ms     10.316 ms       1  25.8%
│  │  │  │  ├─ ⟦cuda⟧ cutlass[cutlass_80_tensorop_s16816gemm_bf16_64x64_32x6_tn_align8] ×24      8.515 ms      8.515 ms      24  21.3%
│  │  │  │  ├─ ⟦cuda⟧ cutlass[cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8] ×24      1.577 ms      1.577 ms      24   3.9%
│  │  │  │  ├─ ⟦cuda⟧ cublasLt::splitKreduce_kernel(…) ×24                                       41.6 µs       41.6 µs      24   0.1%
│  │  │  │  ├─ cudaDeviceGetAttribute ×49                                                         0.0 µs        0.0 µs      49   0.0%
│  │  │  │  ├─ cuLaunchKernel ×49                                                                 0.0 µs        0.0 µs      49   0.0%
│  │  │  │  ├─ cuLaunchKernelEx ×48                                                               0.0 µs        0.0 µs      48   0.0%
│  │  │  │  ├─ cudaLaunchKernelExC ×24                                                            0.0 µs        0.0 µs      24   0.0%
│  │  │  │  ├─ cudaStreamIsCapturing                                                              0.0 µs        0.0 µs       1   0.0%
│  │  │  │  └─ cudaMalloc                                                                         0.0 µs        0.0 µs       1   0.0%
│  │  │  ├─ aten::reshape [bf16] ×97                                                              0.0 µs        0.0 µs      97   0.0%
│  │  │  │  └─ aten::view [bf16 → bf16 ×96 | bf16 ×1] ×97                                         0.0 µs        0.0 µs      97   0.0%
│  │  │  └─ aten::_unsafe_view [bf16 → bf16] ×97                                                  0.0 µs        0.0 µs      97   0.0%
│  │  ├─ aten::addmm [bf16,bf16,bf16 → bf16] ×72                                                2.423 ms        0.0 µs      72   6.1%
│  │  │  ├─ ⟦cuda⟧ cutlass[cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8] ×72      2.268 ms      2.268 ms      72   5.7%
│  │  │  ├─ ⟦cuda⟧ cublasLt::splitKreduce_kernel(…) ×48                                         154.5 µs      154.5 µs      48   0.4%
│  │  │  ├─ cudaDeviceGetAttribute ×72                                                            0.0 µs        0.0 µs      72   0.0%
│  │  │  ├─ cudaMemsetAsync ×24                                                                   0.0 µs        0.0 µs      24   0.0%
│  │  │  ├─ cuLaunchKernel ×72                                                                    0.0 µs        0.0 µs      72   0.0%
│  │  │  └─ cudaLaunchKernelExC ×48                                                               0.0 µs        0.0 µs      48   0.0%
│  │  ├─ aten::view [bf16 → bf16] ×144                                                            0.0 µs        0.0 µs     144   0.0%
│  │  └─ aten::t [bf16 → bf16] ×169                                                               0.0 µs        0.0 µs     169   0.0%
│  │     └─ aten::transpose [bf16 ×124 | bf16 → bf16 ×45] ×169                                    0.0 µs        0.0 µs     169   0.0%
│  │        └─ aten::as_strided [bf16] ×169                                                       0.0 µs        0.0 µs     169   0.0%
│  ├─ aten::mul [bf16,bf16 → bf16 ×169 | f32,f32 → f32 ×49 | f32,f64 ×2] ×220                   493.2 µs        0.0 µs     220   1.2%
│  │  ├─ ⟦cuda⟧ elementwise<Binary> >(…)::{lambda(int)#1}>(int, gpu_kernel_impl_nocast<Binary… ×145      379.1 µs      379.1 µs     145   0.9%
│  │  ├─ ⟦cuda⟧ elementwise<Binary> >(…)::{lambda(int)#1}>(int, gpu_kernel_impl_nocast<Binary… ×49       61.2 µs       61.2 µs      49   0.2%
│  │  ├─ ⟦cuda⟧ elementwise<Binary>, std::array<char*, 3ul> >(…) ×24                             49.1 µs       49.1 µs      24   0.1%
│  │  ├─ ⟦cuda⟧ elementwise<AUnary>, std::array<char*, 2ul> >(…) ×2                               3.9 µs        3.9 µs       2   0.0%
│  │  └─ cudaLaunchKernel ×220                                                                    0.0 µs        0.0 µs     220   0.0%
│  ├─ aten::scaled_dot_product_attention [bf16,bf16,bf16] ×24                                   205.8 µs        0.0 µs      24   0.5%
│  │  └─ aten::_scaled_dot_product_flash_attention [bf16,bf16,bf16 → bf16,f32,torch.uint64,torch.uint64,bf16] ×24      205.8 µs        0.0 µs      24   0.5%
│  │     ├─ aten::_flash_attention_forward [bf16,bf16,bf16] ×24                                 205.8 µs        0.0 µs      24   0.5%
│  │     │  ├─ ⟦cuda⟧ pytorch_flash::flash_fwd_kernel >, false, true, false, false, false, true, fa… ×24      205.8 µs      205.8 µs      24   0.5%
│  │     │  ├─ aten::empty_like [bf16] ×24                                                        0.0 µs        0.0 µs      24   0.0%
│  │     │  │  └─ aten::empty_strided ×24                                                         0.0 µs        0.0 µs      24   0.0%
│  │     │  ├─ aten::empty ×96                                                                    0.0 µs        0.0 µs      96   0.0%
│  │     │  ├─ cudaFuncSetAttribute ×24                                                           0.0 µs        0.0 µs      24   0.0%
│  │     │  └─ cudaLaunchKernel ×24                                                               0.0 µs        0.0 µs      24   0.0%
│  │     └─ aten::transpose [bf16 ×72 | bf16 → bf16 ×24] ×96                                      0.0 µs        0.0 µs      96   0.0%
│  │        └─ aten::as_strided [bf16] ×96                                                        0.0 µs        0.0 µs      96   0.0%
│  ├─ aten::add [bf16,bf16 → bf16 ×96 | f32,f64 ×49 | long int,long int ×1] ×146                186.2 µs        0.0 µs     146   0.5%
│  │  ├─ ⟦cuda⟧ elementwise<add> >(…)::{lambda(int)#1}>(int, gpu_kernel_impl_nocast<CUDAFunct… ×48       90.1 µs       90.1 µs      48   0.2%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel, std::array<char*, 2ul> >(…) ×49                   47.6 µs       47.6 µs      49   0.1%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel, std::array<char*, 3ul> >(…) ×48                   45.4 µs       45.4 µs      48   0.1%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel, std::array<char*, 2ul> >(…)                        3.1 µs        3.1 µs       1   0.0%
│  │  └─ cudaLaunchKernel ×146                                                                    0.0 µs        0.0 µs     146   0.0%
│  ├─ aten::cat [TensorList] ×97                                                                175.5 µs        0.0 µs      97   0.4%
│  │  ├─ ⟦cuda⟧ CatArrayBatchedCopy<OpaqueType<2u>, unsigned int, 4, 64, 64>(…) ×96             171.4 µs      171.4 µs      96   0.4%
│  │  ├─ ⟦cuda⟧ CatArrayBatchedCopy<OpaqueType<4u>, unsigned int, 3, 64, 64>(…)                   4.0 µs        4.0 µs       1   0.0%
│  │  └─ cudaLaunchKernel ×97                                                                     0.0 µs        0.0 µs      97   0.0%
│  ├─ aten::to [bf16 ×97 | f32 ×55 | long int ×2] ×154                                          157.0 µs        0.0 µs     154   0.4%
│  │  └─ aten::_to_copy [f32 → bf16 ×51 | bf16 → f32 ×49 | bf16 ×48 | long int ×2] ×150         157.0 µs        0.0 µs     150   0.4%
│  │     ├─ aten::copy_ [bf16,f32 ×51 | f32,bf16 ×49 | bf16,bf16 ×48 | b8,long int ×1 | f32,long int ×1] ×150      157.0 µs        0.0 µs     150   0.4%
│  │     │  ├─ ⟦cuda⟧ unrolled_elementwise_kernel, 4, TrivialOffsetCalculator<1, unsigned int>, Tri… ×50      101.3 µs      101.3 µs      50   0.3%
│  │     │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel >(int, bfloat16_copy_kernel_cuda(at::TensorIter… ×51       50.1 µs       50.1 µs      51   0.1%
│  │     │  ├─ ⟦cuda⟧ unrolled_elementwise_kernel, 4, TrivialOffsetCalculator<1, unsigned int>, Tri…         5.7 µs        5.7 µs       1   0.0%
│  │     │  └─ cudaLaunchKernel ×102                                                              0.0 µs        0.0 µs     102   0.0%
│  │     └─ aten::empty_strided ×150                                                              0.0 µs        0.0 µs     150   0.0%
│  ├─ aten::mean [f32 → f32] ×49                                                                128.1 µs        0.0 µs      49   0.3%
│  │  ├─ ⟦cuda⟧ reduce_kernel, unsigned int, float, 4, 4> >(…) ×49                              128.1 µs      128.1 µs      49   0.3%
│  │  └─ cudaLaunchKernel ×49                                                                     0.0 µs        0.0 µs      49   0.0%
│  ├─ aten::neg [bf16 → bf16] ×48                                                                82.5 µs        0.0 µs      48   0.2%
│  │  ├─ ⟦cuda⟧ elementwise_kernel(at::TensorIteratorBase&, neg_kernel_cuda(at::TensorIterato… ×48       82.5 µs       82.5 µs      48   0.2%
│  │  └─ cudaLaunchKernel ×48                                                                     0.0 µs        0.0 µs      48   0.0%
│  ├─ aten::rsqrt [f32 → f32] ×49                                                                52.4 µs        0.0 µs      49   0.1%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel >(int, rsqrt_kernel_cuda(at::TensorIteratorBase… ×49       52.4 µs       52.4 µs      49   0.1%
│  │  └─ cudaLaunchKernel ×49                                                                     0.0 µs        0.0 µs      49   0.0%
│  ├─ aten::pow [f32 → f32] ×49                                                                  46.3 µs        0.0 µs      49   0.1%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel(…)::{lambda(float)#1}, std::array<char*, 2ul> >… ×49       46.3 µs       46.3 µs      49   0.1%
│  │  ├─ aten::result_type [f32] ×49                                                              0.0 µs        0.0 µs      49   0.0%
│  │  ├─ aten::to [f32] ×49                                                                       0.0 µs        0.0 µs      49   0.0%
│  │  └─ cudaLaunchKernel ×49                                                                     0.0 µs        0.0 µs      49   0.0%
│  ├─ aten::silu [bf16 → bf16] ×24                                                               35.5 µs        0.0 µs      24   0.1%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel >(int, silu_kernel(at::TensorIteratorBase&)::{l… ×24       35.5 µs       35.5 µs      24   0.1%
│  │  └─ cudaLaunchKernel ×24                                                                     0.0 µs        0.0 µs      24   0.0%
│  ├─ aten::embedding [bf16,long int]                                                            31.4 µs        0.0 µs       1   0.1%
│  │  ├─ aten::index_select [bf16,long int]                                                      31.4 µs        0.0 µs       1   0.1%
│  │  │  ├─ aten::gather [bf16,long int,bf16]                                                    31.4 µs        0.0 µs       1   0.1%
│  │  │  │  ├─ ⟦cuda⟧ vectorized_gather_kernel(…)                                                15.7 µs       15.7 µs       1   0.0%
│  │  │  │  ├─ Activity Buffer Request                                                           15.7 µs        0.0 µs       1   0.0%
│  │  │  │  │  └─ ⟦cuda⟧ vectorized_gather_kernel(…)                                             15.7 µs       15.7 µs       1   0.0%
│  │  │  │  ├─ aten::as_strided [bf16] ×2                                                         0.0 µs        0.0 µs       2   0.0%
│  │  │  │  └─ cudaLaunchKernel                                                                   0.0 µs        0.0 µs       1   0.0%
│  │  │  ├─ aten::empty                                                                           0.0 µs        0.0 µs       1   0.0%
│  │  │  ├─ aten::resize_ [bf16]                                                                  0.0 µs        0.0 µs       1   0.0%
│  │  │  ├─ aten::view [long int]                                                                 0.0 µs        0.0 µs       1   0.0%
│  │  │  └─ aten::expand [long int]                                                               0.0 µs        0.0 µs       1   0.0%
│  │  │     └─ aten::as_strided [long int]                                                        0.0 µs        0.0 µs       1   0.0%
│  │  ├─ aten::reshape [long int]                                                                 0.0 µs        0.0 µs       1   0.0%
│  │  │  └─ aten::view [long int]                                                                 0.0 µs        0.0 µs       1   0.0%
│  │  └─ aten::view [bf16 → bf16]                                                                 0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::all [b8 → b8]                                                                        15.6 µs        0.0 µs       1   0.0%
│  │  ├─ ⟦cuda⟧ reduce_kernel, unsigned int, bool, 4, 4> >(ReduceOp<bool, func_wrapper_t<…>, …        15.6 µs       15.6 µs       1   0.0%
│  │  ├─ aten::as_strided [b8]                                                                    0.0 µs        0.0 µs       1   0.0%
│  │  └─ cudaLaunchKernel                                                                         0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::sin [f32 → f32]                                                                       9.4 µs        0.0 µs       1   0.0%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel >(int, sin_kernel_cuda(at::TensorIteratorBase&)…         9.4 µs        9.4 µs       1   0.0%
│  │  └─ cudaLaunchKernel                                                                         0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::cos [f32 → f32]                                                                       8.5 µs        0.0 µs       1   0.0%
│  │  ├─ ⟦cuda⟧ vectorized_elementwise_kernel >(int, cos_kernel_cuda(at::TensorIteratorBase&)…         8.5 µs        8.5 µs       1   0.0%
│  │  └─ cudaLaunchKernel                                                                         0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::is_nonzero [b8]                                                                       6.5 µs        0.0 µs       1   0.0%
│  │  └─ aten::item [b8]                                                                          6.5 µs        0.0 µs       1   0.0%
│  │     └─ aten::_local_scalar_dense [b8]                                                        6.5 µs        0.0 µs       1   0.0%
│  │        ├─ ⟦cuda⟧ Memcpy DtoH (Device -> Pinned)                                              6.5 µs        6.5 µs       1   0.0%
│  │        ├─ cudaMemcpyAsync                                                                    0.0 µs        0.0 µs       1   0.0%
│  │        └─ cudaStreamSynchronize                                                              0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::matmul [f32,f32]                                                                      5.0 µs        0.0 µs       1   0.0%
│  │  ├─ aten::bmm [f32,f32 → f32]                                                                5.0 µs        0.0 µs       1   0.0%
│  │  │  ├─ ⟦cuda⟧ gemmk1_kernel, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorS…         5.0 µs        5.0 µs       1   0.0%
│  │  │  └─ cudaLaunchKernel                                                                      0.0 µs        0.0 µs       1   0.0%
│  │  ├─ aten::expand [f32 → f32] ×2                                                              0.0 µs        0.0 µs       2   0.0%
│  │  │  └─ aten::as_strided [f32] ×2                                                             0.0 µs        0.0 µs       2   0.0%
│  │  ├─ aten::reshape [f32] ×2                                                                   0.0 µs        0.0 µs       2   0.0%
│  │  │  └─ aten::view [f32 → f32] ×2                                                             0.0 µs        0.0 µs       2   0.0%
│  │  └─ aten::_unsafe_view [f32 → f32]                                                           0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::arange [→ i64]                                                                        2.7 µs        0.0 µs       1   0.0%
│  │  ├─ aten::arange [long int]                                                                  2.7 µs        0.0 µs       1   0.0%
│  │  │  ├─ ⟦cuda⟧ (anonymous namespace)::elementwise_kernel_with_index<…>(int, arange_cuda_out(…         2.7 µs        2.7 µs       1   0.0%
│  │  │  ├─ aten::resize_ [long int]                                                              0.0 µs        0.0 µs       1   0.0%
│  │  │  └─ cudaLaunchKernel                                                                      0.0 µs        0.0 µs       1   0.0%
│  │  └─ aten::empty                                                                              0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::unsqueeze [bf16 → bf16 ×48 | long int ×2 | f32 → f32 ×2] ×52                          0.0 µs        0.0 µs      52   0.0%
│  │  └─ aten::as_strided [bf16 ×48 | long int ×2 | f32 ×2] ×52                                   0.0 µs        0.0 µs      52   0.0%
│  ├─ cudaStreamIsCapturing                                                                       0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::expand [f32 → f32]                                                                    0.0 µs        0.0 µs       1   0.0%
│  │  └─ aten::as_strided [f32]                                                                   0.0 µs        0.0 µs       1   0.0%
│  ├─ aten::transpose [bf16 ×69 | bf16 → bf16 ×27 | f32 → f32 ×1] ×97                             0.0 µs        0.0 µs      97   0.0%
│  │  └─ aten::as_strided [bf16 ×96 | f32 ×1] ×97                                                 0.0 µs        0.0 µs      97   0.0%
│  ├─ aten::view [bf16 → bf16] ×72                                                                0.0 µs        0.0 µs      72   0.0%
│  ├─ aten::slice [bf16 → bf16] ×96                                                               0.0 µs        0.0 µs      96   0.0%
│  │  └─ aten::as_strided [bf16] ×96                                                              0.0 µs        0.0 µs      96   0.0%
│  ├─ aten::empty ×48                                                                             0.0 µs        0.0 µs      48   0.0%
│  ├─ aten::lift_fresh [bf16 → bf16] ×48                                                          0.0 µs        0.0 µs      48   0.0%
│  ├─ aten::detach_ [bf16] ×48                                                                    0.0 µs        0.0 µs      48   0.0%
│  │  └─ detach_ [bf16] ×48                                                                       0.0 µs        0.0 µs      48   0.0%
│  ├─ aten::reshape [bf16] ×24                                                                    0.0 µs        0.0 µs      24   0.0%
│  │  └─ aten::view [bf16 → bf16] ×24                                                             0.0 µs        0.0 µs      24   0.0%
│  └─ aten::alias [bf16 → bf16]                                                                   0.0 µs        0.0 µs       1   0.0%
└─ cudaDeviceSynchronize ×2                                                                       0.0 µs        0.0 µs       2   0.0%
```
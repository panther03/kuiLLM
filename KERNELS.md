# Kernel implementation checklist

Auto-generated from the compiled pipeline by `infer.py --dump-kernels`.
Each row is an ATen op the Inductor graph executed; **Kuiper?** marks the
ops served by a verified `kuiperjit::*` kernel (the rest fall back to
Triton / cuBLAS / cuDNN).

## Kernel dependency graph

Each node is a unique operator signature; repeated calls are collapsed.
Edges show observed data dependencies and are labelled when repeated.
Green nodes use Kuiper kernels and gray nodes use the fallback backend.

```mermaid
flowchart LR
  k0["aten::_scaled_dot_product_cudnn_attention<br/>[T(4,bf16,c), T(4,f32,c), NoneType, NoneType, int, int, T(0,int64,c), T(0,int64,c), NoneType]<br/>calls: 24"]
  k1["aten::addmm<br/>T(2,bf16,c)<br/>calls: 24"]
  k2["aten::mm<br/>T(2,bf16,c)<br/>calls: 97"]
  k3["kuiperjit::hreduce_poly<br/>T(3,f32,c)<br/>calls: 49"]
  k4["aten::add.Tensor<br/>T(4,bf16,c)<br/>calls: 48"]
  k5["aten::add.Tensor<br/>T(3,bf16,c)<br/>calls: 48"]
  k6["aten::add.Tensor<br/>T(3,f32,c)<br/>calls: 24"]
  k7["aten::add.Tensor<br/>T(1,int64,c)<br/>calls: 2"]
  k8["aten::argmax<br/>T(2,int64,c)<br/>calls: 1"]
  k9["aten::cat<br/>T(4,bf16,c)<br/>calls: 48"]
  k10["aten::copy_<br/>T(1,int64,c)<br/>calls: 2"]
  k11["aten::copy_<br/>T(2,int64,c)<br/>calls: 2"]
  k12["aten::copy_<br/>T(4,bf16,c)<br/>calls: 48"]
  k13["aten::div.Tensor<br/>T(3,f32,c)<br/>calls: 24"]
  k14["aten::embedding<br/>T(3,bf16,c)<br/>calls: 1"]
  k15["aten::exp<br/>T(3,f32,c)<br/>calls: 24"]
  k16["aten::index.Tensor<br/>T(2,bf16,c)<br/>calls: 3"]
  k17["aten::index_put<br/>T(4,bf16,c)<br/>calls: 48"]
  k18["aten::index_put<br/>T(2,int64,c)<br/>calls: 1"]
  k19["aten::mul.Tensor<br/>T(3,f32,c)<br/>calls: 49"]
  k20["aten::mul.Tensor<br/>T(3,f32,c)<br/>calls: 49"]
  k21["aten::mul.Tensor<br/>T(4,bf16,c)<br/>calls: 96"]
  k22["aten::mul.Tensor<br/>T(3,bf16,c)<br/>calls: 24"]
  k23["aten::neg<br/>T(4,bf16,c)<br/>calls: 48"]
  k24["aten::neg<br/>T(3,f32,c)<br/>calls: 24"]
  k25["aten::permute<br/>T(2,bf16,c)<br/>calls: 121"]
  k26["aten::permute<br/>T(4,bf16,c)<br/>calls: 96"]
  k27["aten::reshape<br/>T(2,bf16,c)<br/>calls: 121"]
  k28["aten::reshape<br/>T(3,bf16,c)<br/>calls: 121"]
  k29["aten::reshape<br/>T(4,bf16,c)<br/>calls: 72"]
  k30["aten::reshape<br/>T(4,bf16,c)<br/>calls: 3"]
  k31["aten::reshape<br/>T(3,bf16,c)<br/>calls: 24"]
  k32["aten::slice.Tensor<br/>T(4,bf16,c)<br/>calls: 96"]
  k33["aten::split_with_sizes<br/>[T(3,bf16,c), T(3,bf16,c), T(3,bf16,c)]<br/>calls: 24"]
  k34["prims::convert_element_type<br/>T(3,f32,c)<br/>calls: 73"]
  k35["prims::convert_element_type<br/>T(3,bf16,c)<br/>calls: 73"]
  k0 -->|24| k26
  k1 -->|24| k28
  k13 -->|24| k35
  k14 --> k34
  k14 --> k5
  k15 -->|24| k6
  k16 -->|3| k30
  k17 -->|24| k0
  k17 -->|48| k12
  k18 --> k11
  k19 -->|49| k20
  k2 -->|97| k28
  k20 -->|49| k35
  k21 -->|48| k4
  k22 -->|24| k27
  k23 -->|48| k9
  k24 -->|24| k15
  k25 -->|24| k1
  k25 -->|97| k2
  k26 -->|24| k17
  k26 -->|48| k21
  k26 -->|24| k31
  k26 -->|96| k32
  k27 -->|24| k1
  k27 -->|97| k2
  k28 -->|24| k22
  k28 -->|24| k33
  k28 -->|24| k34
  k28 -->|48| k5
  k28 --> k8
  k29 -->|72| k26
  k3 -->|49| k19
  k30 -->|24| k0
  k30 -->|96| k21
  k31 -->|24| k27
  k32 -->|48| k23
  k32 -->|48| k9
  k33 -->|72| k29
  k34 -->|24| k13
  k34 -->|49| k19
  k34 -->|24| k24
  k34 -->|49| k3
  k35 -->|24| k22
  k35 -->|73| k27
  k4 -->|24| k0
  k4 -->|24| k17
  k5 -->|48| k34
  k5 -->|47| k5
  k6 -->|24| k13
  k7 -->|2| k10
  k8 --> k11
  k8 --> k18
  k9 -->|48| k21
  classDef kuiper fill:#d5f5e3,stroke:#1e8449,color:#17202a
  classDef fallback fill:#e5e7e9,stroke:#626567,color:#17202a
  class k0,k1,k2,k3 kuiper
  class k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32,k33,k34,k35 fallback
```

## Kernel inventory

| Op | Args | Out | Kuiper? |
| -- | ---- | --- | ------- |
| aten::_scaled_dot_product_cudnn_attention | T(4,bf16,c), T(4,bf16,c), T(4,bf16,c), T(4,bf16,c), False | scale=float | [T(4,bf16,c), T(4,f32,c), NoneType, NoneType, int, int, T(0,int64,c), T(0,int64,c), NoneType] | yes |
| aten::addmm | T(1,bf16,c), T(2,bf16,c), T(2,bf16,c) | T(2,bf16,c) | yes |
| aten::mm | T(2,bf16,c), T(2,bf16,c) | T(2,bf16,c) | yes |
| kuiperjit::hreduce_poly | T(3,f32,c), str, int, True, NoneType, str, str | T(3,f32,c) | yes |
| aten::add.Tensor | T(4,bf16,c), T(4,bf16,c) | T(4,bf16,c) |  |
| aten::add.Tensor | T(3,bf16,c), T(3,bf16,c) | T(3,bf16,c) |  |
| aten::add.Tensor | T(3,f32,c), int | T(3,f32,c) |  |
| aten::add.Tensor | T(1,int64,c), int | T(1,int64,c) |  |
| aten::argmax | T(3,bf16,c), int | T(2,int64,c) |  |
| aten::cat | [T(4,bf16,c), T(4,bf16,c)], int | T(4,bf16,c) |  |
| aten::copy_ | T(1,int64,c), T(1,int64,c) | T(1,int64,c) |  |
| aten::copy_ | T(2,int64,c), T(2,int64,c) | T(2,int64,c) |  |
| aten::copy_ | T(4,bf16,c), T(4,bf16,c) | T(4,bf16,c) |  |
| aten::div.Tensor | T(3,f32,c), T(3,f32,c) | T(3,f32,c) |  |
| aten::embedding | T(2,bf16,c), T(2,int64,c) | T(3,bf16,c) |  |
| aten::exp | T(3,f32,c) | T(3,f32,c) |  |
| aten::index.Tensor | T(2,bf16,c), [T(1,int64,c)] | T(2,bf16,c) |  |
| aten::index_put | T(4,bf16,c), [NoneType, NoneType, T(1,int64,c)], T(4,bf16,c) | T(4,bf16,c) |  |
| aten::index_put | T(2,int64,c), [NoneType, T(1,int64,c)], T(2,int64,c) | T(2,int64,c) |  |
| aten::mul.Tensor | T(3,f32,c), T(3,f32,c) | T(3,f32,c) |  |
| aten::mul.Tensor | T(3,f32,c), T(1,bf16,c) | T(3,f32,c) |  |
| aten::mul.Tensor | T(4,bf16,c), T(4,bf16,c) | T(4,bf16,c) |  |
| aten::mul.Tensor | T(3,bf16,c), T(3,bf16,c) | T(3,bf16,c) |  |
| aten::neg | T(4,bf16,c) | T(4,bf16,c) |  |
| aten::neg | T(3,f32,c) | T(3,f32,c) |  |
| aten::permute | T(2,bf16,c), [int, int] | T(2,bf16,c) |  |
| aten::permute | T(4,bf16,c), [int, int, int, int] | T(4,bf16,c) |  |
| aten::reshape | T(3,bf16,c), [int, int] | T(2,bf16,c) |  |
| aten::reshape | T(2,bf16,c), [int, int, int] | T(3,bf16,c) |  |
| aten::reshape | T(3,bf16,c), [int, int, int, int] | T(4,bf16,c) |  |
| aten::reshape | T(2,bf16,c), [int, int, int, int] | T(4,bf16,c) |  |
| aten::reshape | T(4,bf16,c), [int, int, int] | T(3,bf16,c) |  |
| aten::slice.Tensor | T(4,bf16,c), int, int, int | T(4,bf16,c) |  |
| aten::split_with_sizes | T(3,bf16,c), [int, int, int], int | [T(3,bf16,c), T(3,bf16,c), T(3,bf16,c)] |  |
| prims::convert_element_type | T(3,bf16,c), f32 | T(3,f32,c) |  |
| prims::convert_element_type | T(3,f32,c), bf16 | T(3,bf16,c) |  |

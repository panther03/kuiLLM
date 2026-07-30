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
  k0["aten::mm<br/>T(2,bf16,c)<br/>calls: 97"]
  k1["aten::_scaled_dot_product_cudnn_attention<br/>[T(4,bf16,c), T(4,f32,c), NoneType, NoneType, int, int, T(0,int64,c), T(0,int64,c), NoneType]<br/>calls: 24"]
  k2["aten::add.Scalar<br/>T(3,f32,c)<br/>calls: 49"]
  k3["aten::add.Tensor<br/>T(4,bf16,c)<br/>calls: 48"]
  k4["aten::add.Tensor<br/>T(3,bf16,c)<br/>calls: 48"]
  k5["aten::add.Tensor<br/>T(3,f32,c)<br/>calls: 24"]
  k6["aten::add.Tensor<br/>T(1,int64,c)<br/>calls: 2"]
  k7["aten::addmm<br/>T(2,bf16,c)<br/>calls: 24"]
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
  k19["aten::mean.dim<br/>T(3,f32,c)<br/>calls: 49"]
  k20["aten::mul.Tensor<br/>T(3,f32,c)<br/>calls: 49"]
  k21["aten::mul.Tensor<br/>T(3,f32,c)<br/>calls: 49"]
  k22["aten::mul.Tensor<br/>T(4,bf16,c)<br/>calls: 96"]
  k23["aten::mul.Tensor<br/>T(3,bf16,c)<br/>calls: 24"]
  k24["aten::neg<br/>T(4,bf16,c)<br/>calls: 48"]
  k25["aten::neg<br/>T(3,f32,c)<br/>calls: 24"]
  k26["aten::permute<br/>T(2,bf16,c)<br/>calls: 121"]
  k27["aten::permute<br/>T(4,bf16,c)<br/>calls: 96"]
  k28["aten::pow.Tensor_Scalar<br/>T(3,f32,c)<br/>calls: 49"]
  k29["aten::reshape<br/>T(2,bf16,c)<br/>calls: 121"]
  k30["aten::reshape<br/>T(3,bf16,c)<br/>calls: 121"]
  k31["aten::reshape<br/>T(4,bf16,c)<br/>calls: 72"]
  k32["aten::reshape<br/>T(4,bf16,c)<br/>calls: 3"]
  k33["aten::reshape<br/>T(3,bf16,c)<br/>calls: 24"]
  k34["aten::rsqrt<br/>T(3,f32,c)<br/>calls: 49"]
  k35["aten::slice.Tensor<br/>T(4,bf16,c)<br/>calls: 96"]
  k36["aten::split_with_sizes<br/>[T(3,bf16,c), T(3,bf16,c), T(3,bf16,c)]<br/>calls: 24"]
  k37["prims::convert_element_type<br/>T(3,f32,c)<br/>calls: 73"]
  k38["prims::convert_element_type<br/>T(3,bf16,c)<br/>calls: 73"]
  k0 -->|97| k30
  k1 -->|24| k27
  k13 -->|24| k38
  k14 --> k37
  k14 --> k4
  k15 -->|24| k5
  k16 -->|3| k32
  k17 -->|24| k1
  k17 -->|48| k12
  k18 --> k11
  k19 -->|49| k2
  k2 -->|49| k34
  k20 -->|49| k21
  k21 -->|49| k38
  k22 -->|48| k3
  k23 -->|24| k29
  k24 -->|48| k9
  k25 -->|24| k15
  k26 -->|97| k0
  k26 -->|24| k7
  k27 -->|24| k17
  k27 -->|48| k22
  k27 -->|24| k33
  k27 -->|96| k35
  k28 -->|49| k19
  k29 -->|97| k0
  k29 -->|24| k7
  k3 -->|24| k1
  k3 -->|24| k17
  k30 -->|24| k23
  k30 -->|24| k36
  k30 -->|24| k37
  k30 -->|48| k4
  k30 --> k8
  k31 -->|72| k27
  k32 -->|24| k1
  k32 -->|96| k22
  k33 -->|24| k29
  k34 -->|49| k20
  k35 -->|48| k24
  k35 -->|48| k9
  k36 -->|72| k31
  k37 -->|24| k13
  k37 -->|49| k20
  k37 -->|24| k25
  k37 -->|49| k28
  k38 -->|24| k23
  k38 -->|73| k29
  k4 -->|48| k37
  k4 -->|47| k4
  k5 -->|24| k13
  k6 -->|2| k10
  k7 -->|24| k30
  k8 --> k11
  k8 --> k18
  k9 -->|48| k22
  classDef kuiper fill:#d5f5e3,stroke:#1e8449,color:#17202a
  classDef fallback fill:#e5e7e9,stroke:#626567,color:#17202a
  class k0 kuiper
  class k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32,k33,k34,k35,k36,k37,k38 fallback
```

## Kernel inventory

| Op | Args | Out | Kuiper? |
| -- | ---- | --- | ------- |
| aten::mm | T(2,bf16,c), T(2,bf16,c) | T(2,bf16,c) | yes |
| aten::_scaled_dot_product_cudnn_attention | T(4,bf16,c), T(4,bf16,c), T(4,bf16,c), T(4,bf16,c), False | scale=float | [T(4,bf16,c), T(4,f32,c), NoneType, NoneType, int, int, T(0,int64,c), T(0,int64,c), NoneType] |  |
| aten::add.Scalar | T(3,f32,c), float | T(3,f32,c) |  |
| aten::add.Tensor | T(4,bf16,c), T(4,bf16,c) | T(4,bf16,c) |  |
| aten::add.Tensor | T(3,bf16,c), T(3,bf16,c) | T(3,bf16,c) |  |
| aten::add.Tensor | T(3,f32,c), int | T(3,f32,c) |  |
| aten::add.Tensor | T(1,int64,c), int | T(1,int64,c) |  |
| aten::addmm | T(1,bf16,c), T(2,bf16,c), T(2,bf16,c) | T(2,bf16,c) |  |
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
| aten::mean.dim | T(3,f32,c), [int], True | T(3,f32,c) |  |
| aten::mul.Tensor | T(3,f32,c), T(3,f32,c) | T(3,f32,c) |  |
| aten::mul.Tensor | T(3,f32,c), T(1,bf16,c) | T(3,f32,c) |  |
| aten::mul.Tensor | T(4,bf16,c), T(4,bf16,c) | T(4,bf16,c) |  |
| aten::mul.Tensor | T(3,bf16,c), T(3,bf16,c) | T(3,bf16,c) |  |
| aten::neg | T(4,bf16,c) | T(4,bf16,c) |  |
| aten::neg | T(3,f32,c) | T(3,f32,c) |  |
| aten::permute | T(2,bf16,c), [int, int] | T(2,bf16,c) |  |
| aten::permute | T(4,bf16,c), [int, int, int, int] | T(4,bf16,c) |  |
| aten::pow.Tensor_Scalar | T(3,f32,c), int | T(3,f32,c) |  |
| aten::reshape | T(3,bf16,c), [int, int] | T(2,bf16,c) |  |
| aten::reshape | T(2,bf16,c), [int, int, int] | T(3,bf16,c) |  |
| aten::reshape | T(3,bf16,c), [int, int, int, int] | T(4,bf16,c) |  |
| aten::reshape | T(2,bf16,c), [int, int, int, int] | T(4,bf16,c) |  |
| aten::reshape | T(4,bf16,c), [int, int, int] | T(3,bf16,c) |  |
| aten::rsqrt | T(3,f32,c) | T(3,f32,c) |  |
| aten::slice.Tensor | T(4,bf16,c), int, int, int | T(4,bf16,c) |  |
| aten::split_with_sizes | T(3,bf16,c), [int, int, int], int | [T(3,bf16,c), T(3,bf16,c), T(3,bf16,c)] |  |
| prims::convert_element_type | T(3,bf16,c), f32 | T(3,f32,c) |  |
| prims::convert_element_type | T(3,f32,c), bf16 | T(3,bf16,c) |  |

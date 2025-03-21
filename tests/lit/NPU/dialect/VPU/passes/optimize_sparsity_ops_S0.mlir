//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --optimize-sparsity-ops="sparsity-profile=S0" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @RemoveDuplicatedSparsify(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Sparsify
    // CHECK-NOT:   VPU.Sparsify

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg0)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @RemoveSparsifyOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.Sparsify(%arg1) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Sparsify
    // CHECK-NOT:   VPU.Sparsify

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @RemoveExtraDesparsify(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.Desparsify(%0) : !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.MaxPool(%1) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
      } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise(%arg0, %arg0)
    // CHECK-NOT:   !VPU.SparseTensor
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL1:%.+]] = VPU.MaxPool([[VAL0]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL1]]
}

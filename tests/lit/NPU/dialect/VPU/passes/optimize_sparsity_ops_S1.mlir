//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --optimize-sparsity-ops="sparsity-profile=S1" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveDuplicatedSparsify
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>
func.func @RemoveDuplicatedSparsify(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.NCE.Eltwise(%0, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
    // CHECK-NOT:   VPU.Sparsify

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL0]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL1]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @KeepSparsifyOps
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>
func.func @KeepSparsifyOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %arg1: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %1 = VPU.Sparsify(%arg1) : tensor<1x16x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x16x16x16xf16, {order = #NHWC}>>
    %2 = VPU.NCE.Eltwise(%0, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
    // CHECK:       [[VAL1:%.+]] = VPU.Sparsify([[ARG_1]])

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Eltwise([[VAL0]], [[VAL1]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL2]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveExtraDesparsify
// CHECK-SAME: [[ARG_0:%[^:]+]]:  tensor<1x16x16x16xf16, {order = #NHWC}>
func.func @RemoveExtraDesparsify(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
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

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Eltwise([[ARG_0]], [[ARG_0]])
    // CHECK-NOT:   !VPU.SparseTensor
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL1:%.+]] = VPU.MaxPool([[VAL0]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL1]]
}

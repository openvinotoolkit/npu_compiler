//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="enable-activation-sparsity-mode=true sparsity-profile=S0" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapSingleOp
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @WrapSingleOp(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
    // CHECK-NOT:   VPU.Desparsify
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]
    // CHECK:       return [[VAL3]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapChainedMixedOps
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @WrapChainedMixedOps(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[POOL:%.+]] = VPU.MaxPool([[ARG_0]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[POOL]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.Desparsify
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

    // CHECK:       return [[VAL3]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapMultipleConsumers
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @WrapMultipleConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%1, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

    // CHECK:       [[VAL4:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Convolution([[VAL4]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL6:%.+]] = VPU.Sparsify([[VAL5]])
    // CHECK:       [[VAL7:%.+]] = VPU.Desparsify([[VAL6]]

    // CHECK:       [[VAL8:%.+]] = VPU.Sparsify([[VAL3]])
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL9:%.+]] = VPU.NCE.Convolution([[VAL8]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL10:%.+]] = VPU.Sparsify([[VAL9]])
    // CHECK:       [[VAL11:%.+]] = VPU.Desparsify([[VAL10]]

    // CHECK:       return [[VAL7]], [[VAL11]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapMultipleMixedConsumers
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @WrapMultipleMixedConsumers(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %1 = VPU.NCE.Convolution(%arg0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %3 = VPU.MaxPool(%1) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %2, %3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify([[ARG_0]])
    // CHECK-NOT:   VPU.Desparsify

    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL2:%.+]] = VPU.Sparsify([[VAL1]])
    // CHECK:       [[VAL3:%.+]] = VPU.Desparsify([[VAL2]]

    // CHECK:       [[VAL6:%.+]] = VPU.NCE.Eltwise([[VAL3]], [[VAL3]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL7:%.+]] = VPU.Sparsify([[VAL6]])
    // CHECK:       [[VAL8:%.+]] = VPU.Desparsify([[VAL7]]

    // CHECK:       [[VAL9:%.+]] = VPU.MaxPool([[VAL3]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>


    // CHECK:       return [[VAL8]], [[VAL9]]
}

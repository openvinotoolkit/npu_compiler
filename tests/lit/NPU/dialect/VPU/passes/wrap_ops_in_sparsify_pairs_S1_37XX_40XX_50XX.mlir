//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --wrap-ops-in-sparsify-pairs="enable-activation-sparsity-mode=true sparsity-profile=S1" %s | FileCheck %s
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
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]
    // CHECK:       return [[VAL4]]
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
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]

    // CHECK:       return [[VAL4]]
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
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]

    // CHECK:       [[VAL5:%.+]] = VPU.Sparsify([[VAL4]])
    // CHECK:       [[VAL6:%.+]] = VPU.Desparsify([[VAL5]]

    // CHECK:       [[VAL9:%.+]] = VPU.NCE.Convolution([[VAL6]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL10:%.+]] = VPU.Sparsify([[VAL9]])
    // CHECK:       [[VAL11:%.+]] = VPU.Desparsify([[VAL10]]

    // CHECK:       [[VAL12:%.+]] = VPU.Sparsify([[VAL4]])
    // CHECK:       [[VAL13:%.+]] = VPU.Desparsify([[VAL12]]

    // CHECK:       [[VAL14:%.+]] = VPU.NCE.Convolution([[VAL13]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL15:%.+]] = VPU.Sparsify([[VAL14]])
    // CHECK:       [[VAL16:%.+]] = VPU.Desparsify([[VAL15]]

    // CHECK:       return [[VAL11]], [[VAL16]]
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
                ppe = #VPU.PPEStub<>,
                op_type = #VPU.eltwise_type<ADD>
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
    // CHECK:       [[VAL1:%.+]] = VPU.Desparsify([[VAL0]]

    // CHECK:       [[VAL2:%.+]] = VPU.NCE.Convolution([[VAL1]], [[ARG_2]], [[ARG_1]])
    // CHECK-NOT:       -> !VPU.SparseTensor
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL3:%.+]] = VPU.Sparsify([[VAL2]])
    // CHECK:       [[VAL4:%.+]] = VPU.Desparsify([[VAL3]]

    // CHECK:       [[VAL5:%.+]] = VPU.NCE.Eltwise([[VAL4]], [[VAL4]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL6:%.+]] = VPU.Sparsify([[VAL5]])
    // CHECK:       [[VAL7:%.+]] = VPU.Desparsify([[VAL6]]

    // CHECK:       [[VAL8:%.+]] = VPU.MaxPool([[VAL4]])
    // CHECK-NOT:       !VPU.SparseTensor
    // CHECK-SAME:      tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL7]], [[VAL8]]
}

//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --lower-sparsity-ops="fake-sparsify=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16xf16, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerSparsifyOpF16
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerSparsifyOpF16(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : !sparseType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !defaultType

    return %1 : !defaultType

    // CHECK-DAG:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x16x16x16xi1, {order = #NHWC}> = dense<true> : tensor<1x16x16x16xi1, {order = #NHWC}>
    // CHECK:       [[VAL0:%.+]] = VPU.GroupSparseTensor([[ARG_0]], [[SPARSITY_MAP]])
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL1]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!defaultType = tensor<1x16x16x16x!quant.uniform<u8:f16, 2.0>, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerSparsifyOpUniformQuant
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerSparsifyOpUniformQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : !sparseType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !defaultType

    return %1 : !defaultType

    // CHECK-DAG:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x16x16x16xi1, {order = #NHWC}> = dense<true> : tensor<1x16x16x16xi1, {order = #NHWC}>
    // CHECK:       [[VAL0:%.+]] = VPU.GroupSparseTensor([[ARG_0]], [[SPARSITY_MAP]])
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL1]]
}

//
// -----
//

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16:1, {1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}>
!defaultType = tensor<1x16x16x16x!qElemType, {order = #NHWC}>
!smType = tensor<1x16x16x16xi1, {order = #NHWC}>
!sparseType = !VPU.SparseTensor<data=!defaultType, sparsity_map=!smType>

// CHECK-LABEL: @LowerSparsifyOpPerAxisQuant
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<16x1x1x4xsi32>, [[ARG_2:%[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @LowerSparsifyOpPerAxisQuant(%arg0: !defaultType, %wt: tensor<16x1x1x4xsi32>, %weights: tensor<16x16x1x1xf16, {order = #NHWC}>) -> !defaultType {
    %0 = VPU.Sparsify(%arg0) : !defaultType -> !sparseType
    %1 = VPU.NCE.Convolution(%0, %weights, %wt) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            
            strides = [1, 1]
        } : !sparseType, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> !defaultType

    return %1 : !defaultType

    // CHECK-DAG:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x16x16x16xi1, {order = #NHWC}> = dense<true> : tensor<1x16x16x16xi1, {order = #NHWC}>
    // CHECK:       [[VAL0:%.+]] = VPU.GroupSparseTensor([[ARG_0]], [[SPARSITY_MAP]])
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[ARG_2]], [[ARG_1]])
    // CHECK:       return [[VAL1]]
}

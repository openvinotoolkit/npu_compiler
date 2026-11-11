//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --recompute-sparsity-ptrs %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvNoChangeWithSplitWeightTableFormat
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @NCEConvNoChangeWithSplitWeightTableFormat(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %scale_table_cst = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst = const.Declare tensor<16x1x1x1xf32> = dense<2.0> : tensor<16x1x1x1xf32>
    %1 = VPU.NCE.Convolution(%arg0, %weights_cst, %scale_table_cst, %bias_table_cst) rawFilterShape [16, 16, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            
            strides = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x4x4xf16, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK-DAG:       [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       {{.+}} = VPU.NCE.Convolution([[ARG_0]], {{%.+}}, [[SCALE_TABLE]], [[BIAS_TABLE]])
    // CHECK-SAME:       -> tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvNoChangeWithoutAnyWeightTable
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @NCEConvNoChangeWithoutAnyWeightTable(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights_cst = const.Declare tensor<16x16x4x4xf16, {order = #NHWC}> = dense<0.0> : tensor<16x16x4x4xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%arg0, %weights_cst) rawFilterShape [16, 16, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>,
            
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x4x4xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       {{.+}} = VPU.NCE.Convolution([[ARG_0]], {{%.+}})
    // CHECK-SAME:       -> tensor<1x16x16x16xf16, {order = #NHWC}>
}

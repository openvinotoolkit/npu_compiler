//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-expand-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.01>
// CHECK:   !qElemType = !quant.uniform<f8E4M3FN:f16, 1.000000e-02>
// CHECK:   !qElemType1 = !quant.uniform<f8E4M3FN:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertExpandToConv16ChannelsF8E4M3FN
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x3x64x224x!qElemType, {order = #NHWC}
func.func @ConvertExpandToConv16ChannelsF8E4M3FN(%arg0: tensor<1x3x64x224x!qElemType, {order = #NHWC}>)
    -> tensor<1x16x64x224x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224x!qElemType, {order = #NHWC}> -> tensor<1x16x64x224x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224x!qElemType, {order = #NHWC}>


    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224x!qElemType, {order = #NHWC}> -> tensor<1x48x64x14x!qElemType, {order = #NHWC}>
    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<256x48x1x1x!qElemType1, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000{{([0]{180})}}
    // CHECK-SAME:      0000003C0000{{([0]{180})}}
    // CHECK-SAME:      00000000003C{{([0]{180})}}
    // CHECK-SAME:      000000000000{{([0]{180})}}

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x64x14x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 64, 224]
    // CHECK-SAME:  } : tensor<1x256x64x14x!qElemType, {order = #NHWC}> -> tensor<1x16x64x224x!qElemType, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x64x224x!qElemType, {order = #NHWC}>
}

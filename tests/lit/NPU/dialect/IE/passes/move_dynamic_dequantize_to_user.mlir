//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --move-dynamic-dequantize-to-user %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.0:8>

// CHECK-LABEL: @ReorderReshapePermuteOps
// CHECK-SAME:     [[INPUT:%.+]]: tensor<1x4096x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:     [[WEIGHTS:%.+]]: tensor<4096x4096xui4>,
// CHECK-SAME:     [[SCALE:%.+]]: tensor<4096x1xf16>)
func.func @ReorderReshapePermuteOps(%input: tensor<1x4096x1x1xf16, {order = #NHWC}>, %weights: tensor<4096x4096xui4>, %scale: tensor<4096x1xf16>) -> tensor<1x4096x1x1xf16, {order = #NHWC}> {
    %quant_cast = IE.QuantizeCast(%weights) {dstElemType = !qElemType} : tensor<4096x4096xui4> -> tensor<4096x4096x!qElemType>
    %dyn_quant = IE.DynamicDequantize(%quant_cast, %scale) {dstElemType = f16} : tensor<4096x4096x!qElemType>, tensor<4096x1xf16> -> tensor<4096x4096xf16>
    %reshaped_w = IE.AffineReshape(%dyn_quant) {dim_mapping = [[0], [1, 2, 3]], shape_value = [4096, 4096, 1, 1]} : tensor<4096x4096xf16> -> tensor<4096x4096x1x1xf16>
    %permuted_w = IE.PermuteCast(%reshaped_w) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4096x4096x1x1xf16> -> tensor<4096x4096x1x1xf16, {order = #NHWC}>

    %conv = IE.Convolution(%input, %permuted_w)
        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
        : tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}>
    -> tensor<1x4096x1x1xf16, {order = #NHWC}>

    return %conv : tensor<1x4096x1x1xf16, {order = #NHWC}>

    // CHECK:   [[QUANT_CAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} :
    // CHECK-SAME:  tensor<4096x4096xui4> -> tensor<4096x4096x!qElemType>

    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[QUANT_CAST]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2, 3]], shape_value = [4096, 4096, 1, 1]} :
    // CHECK-SAME:  tensor<4096x4096x!qElemType> -> tensor<4096x4096x1x1x!qElemType>

    // CHECK:   [[PERMUTE:%.+]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} :
    // CHECK-SAME:  tensor<4096x4096x1x1x!qElemType> -> tensor<4096x4096x1x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SCALE_RESHAPE:%.+]] = IE.AffineReshape([[SCALE]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2, 3]], shape_value = [4096, 1, 1, 1]} :
    // CHECK-SAME:  tensor<4096x1xf16> -> tensor<4096x1x1x1xf16>

    // CHECK:   [[SCALE_PERMUTE:%.+]] = IE.PermuteCast([[SCALE_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NHWC} :
    // CHECK-SAME:  tensor<4096x1x1x1xf16> -> tensor<4096x1x1x1xf16, {order = #NHWC}>

    // CHECK:   [[DYN_QUANT:%.+]] = IE.DynamicDequantize([[PERMUTE]], [[SCALE_PERMUTE]]) {dstElemType = f16} :
    // CHECK-SAME:  tensor<4096x4096x1x1x!qElemType, {order = #NHWC}>, tensor<4096x1x1x1xf16, {order = #NHWC}> ->
    // CHECK-SAME:  tensor<4096x4096x1x1xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[DYN_QUANT]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<4096x4096x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x4096x1x1xf16, {order = #NHWC}>

    // CHECK:   return [[CONV]] : tensor<1x4096x1x1xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @NotReorderReshapePermuteOpsDueToIncompatibleShapes
// CHECK-SAME:    [[INPUT:%.+]]: tensor<28x4608x128x!qElemType>
// CHECK-SAME:    [[SCALE:%.+]]: tensor<28x4608x1xf16>
func.func @NotReorderReshapePermuteOpsDueToIncompatibleShapes(%arg0: tensor<28x4608x128x!qElemType>, %arg1: tensor<28x4608x1xf16>) -> tensor<1x128x28x4608xf16, {order = #NWHC}> {
    %0 = IE.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<28x4608x128x!qElemType>, tensor<28x4608x1xf16> -> tensor<28x4608x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 28, 4608, 128]} : tensor<28x4608x128xf16> -> tensor<1x28x4608x128xf16>
    %2 = IE.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x4608x128xf16> -> tensor<1x128x28x4608xf16, {order = #NHWC}>
    %3 = IE.MaxPool(%2) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x28x4608xf16, {order = #NHWC}> -> tensor<1x128x28x4608xf16, {order = #NWHC}>

    return %3 : tensor<1x128x28x4608xf16, {order = #NWHC}>

    // CHECK:   [[DYNAMICDQ:%.+]] = IE.DynamicDequantize([[INPUT]], [[SCALE]]) {dstElemType = f16} : tensor<28x4608x128x!qElemType>, tensor<28x4608x1xf16> -> tensor<28x4608x128xf16>
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[DYNAMICDQ]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 28, 4608, 128]} : tensor<28x4608x128xf16> -> tensor<1x28x4608x128xf16>
    // CHECK:   [[PERMUTE:%.+]] = IE.PermuteCast([[RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x28x4608x128xf16> -> tensor<1x128x28x4608xf16, {order = #NHWC}>
    // CHECK:   [[POOL:%.+]] = IE.MaxPool([[PERMUTE]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x28x4608xf16, {order = #NHWC}> -> tensor<1x128x28x4608xf16, {order = #NWHC}>

    // CHECK:   return [[POOL]] : tensor<1x128x28x4608xf16, {order = #NWHC}>
}

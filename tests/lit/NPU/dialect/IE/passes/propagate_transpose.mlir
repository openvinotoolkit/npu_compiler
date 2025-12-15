//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-transpose %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithSoftmax
func.func @SwapWithSoftmax(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.SoftMax(%1) {axisInd = -1 : i64} : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK:        [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK:        [[TRANSPOSE:%.*]] = IE.Transpose([[SOFTMAX]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK:        return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithGelu
func.func @SwapWithGelu(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.Gelu(%1) : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK: [[GELU:%.*]] = IE.Gelu(%arg0) : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[GELU]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapWithSlice
func.func @SwapWithSlice(%arg0 : tensor<1x320x4096x1xf16>) -> tensor<4096x1280x1x1xf16> {
    %cst = const.Declare tensor<2560x320x1x1xf16> = dense<1.0> : tensor<320x2560xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 320, 2560]>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0055789117719612872:140>>, #const.Reshape<[1, 320, 1, 2560]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.Reshape<[2560, 320, 1, 1]>, #const.Dequantize]
    %cst_0 = const.Declare tensor<1x2560x1x1xf16> = dense<1.0> : tensor<1x1x1x2560xf32>, [#const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reshape<[1, 2560, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<2560x320x1x1xf16>, tensor<1x2560x1x1xf16> -> tensor<1x2560x4096x1xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : tensor<1x2560x4096x1xf16> -> tensor<4096x2560x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [4096, 1280, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<4096x1280x1x1xf16>
    return %2 : tensor<4096x1280x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-NEXT:  [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 1280, 4096, 1] : tensor<1x2560x4096x1xf16> to tensor<1x1280x4096x1xf16>
    // CHECK-NEXT:  [[TRANSPOSE:%.+]] = IE.Transpose([[SLICE]]) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    // CHECK:       return [[TRANSPOSE]] : tensor<4096x1280x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapWithMultipleSlice
func.func @SwapWithMultipleSlice(%arg0 : tensor<1x320x4096x1xf16>) -> tensor<4096x1280x1x1xf16> {
    %cst = const.Declare tensor<2560x320x1x1xf16> = dense<1.0> : tensor<320x2560xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 320, 2560]>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0055789117719612872:140>>, #const.Reshape<[1, 320, 1, 2560]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.Reshape<[2560, 320, 1, 1]>, #const.Dequantize]
    %cst_0 = const.Declare tensor<1x2560x1x1xf16> = dense<1.0> : tensor<1x1x1x2560xf32>, [#const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reshape<[1, 2560, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<2560x320x1x1xf16>, tensor<1x2560x1x1xf16> -> tensor<1x2560x4096x1xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : tensor<1x2560x4096x1xf16> -> tensor<4096x2560x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [4096, 1280, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<4096x1280x1x1xf16>
    %3 = IE.Slice %1 [0, 1280, 0, 0] [4096, 1280, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<4096x1280x1x1xf16>
    %4 = IE.Multiply(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x1280x1x1xf16>, tensor<4096x1280x1x1xf16> -> tensor<4096x1280x1x1xf16>
    return %4 : tensor<4096x1280x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-NEXT:  [[SLICE1:%.+]] = IE.Slice [[CONV]] [0, 1280, 0, 0] [1, 1280, 4096, 1] : tensor<1x2560x4096x1xf16> to tensor<1x1280x4096x1xf16>
    // CHECK-NEXT:  [[SLICE2:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 1280, 4096, 1] : tensor<1x2560x4096x1xf16> to tensor<1x1280x4096x1xf16>
    // CHECK-NEXT:  [[MUL:%.+]] = IE.Multiply([[SLICE2]], [[SLICE1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1280x4096x1xf16>, tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16>
    // CHECK-NEXT:  [[TRANSPOSE:%.+]] = IE.Transpose([[MUL]]) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    // CHECK:       return [[TRANSPOSE]] : tensor<4096x1280x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @NotSwapWithFakeChannelSlice
func.func @NotSwapWithFakeChannelSlice(%arg0 : tensor<1x320x2560x1xf16>) -> tensor<4096x1280x1x1xf16> {
    %cst = const.Declare tensor<4096x320x1x1xf16> = dense<1.0> : tensor<320x4096xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 320, 4096]>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0055789117719612872:140>>, #const.Reshape<[1, 320, 1, 4096]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.Reshape<[4096, 320, 1, 1]>, #const.Dequantize]
    %cst_0 = const.Declare tensor<1x4096x1x1xf16> = dense<1.0> : tensor<1x1x1x4096xf32>, [#const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reshape<[1, 4096, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x2560x1xf16>, tensor<4096x320x1x1xf16>, tensor<1x4096x1x1xf16> -> tensor<1x4096x2560x1xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d1, d2, d0, d3)>} : tensor<1x4096x2560x1xf16> -> tensor<4096x2560x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [4096, 1280, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<4096x1280x1x1xf16>
    return %2 : tensor<4096x1280x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-NEXT:  [[TRANSPOSE:%.+]] = IE.Transpose
    // CHECK-NEXT:  [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<4096x1280x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @NotSwapWithHeightSlice
func.func @NotSwapWithHeightSlice(%arg0 : tensor<1x320x4096x1xf16>) -> tensor<2048x2560x1x1xf16> {
    %cst = const.Declare tensor<2560x320x1x1xf16> = dense<1.0> : tensor<320x2560xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 320, 2560]>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0055789117719612872:140>>, #const.Reshape<[1, 320, 1, 2560]>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>>, #const.Reshape<[2560, 320, 1, 1]>, #const.Dequantize]
    %cst_0 = const.Declare tensor<1x2560x1x1xf16> = dense<1.0> : tensor<1x1x1x2560xf32>, [#const.CastElemType<f16>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>>, #const.Transpose<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reshape<[1, 2560, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x320x4096x1xf16>, tensor<2560x320x1x1xf16>, tensor<1x2560x1x1xf16> -> tensor<1x2560x4096x1xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : tensor<1x2560x4096x1xf16> -> tensor<4096x2560x1x1xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [2048, 2560, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<2048x2560x1x1xf16>
    return %2 : tensor<2048x2560x1x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-NEXT:  [[TRANSPOSE:%.+]] = IE.Transpose
    // CHECK-NEXT:  [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]] : tensor<2048x2560x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @SwapWithAvgPool
func.func @SwapWithAvgPool(%arg0: tensor<1x512x768x1xf16>) -> tensor<512x768x1x1xf16> {
    %0 = IE.Add(%arg0, %arg0)  { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x512x768x1xf16>, tensor<1x512x768x1xf16> -> tensor<1x512x768x1xf16>
    %1 = IE.Transpose(%0) {order_value = #map} : tensor<1x512x768x1xf16> -> tensor<512x768x1x1xf16>
    %2 = IE.AvgPool(%1) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<512x768x1x1xf16> -> tensor<512x768x1x1xf16>
    return %2: tensor<512x768x1x1xf16>

    // CHECK:        [[ADD:%.*]] = IE.Add
    // CHECK:        [[AVGPOOL:%.*]] = IE.AvgPool([[ADD]]) {exclude_pads, kernel_size = [1, 1],
    // CHECK-SAME:     tensor<1x512x768x1xf16> -> tensor<1x512x768x1xf16>
    // CHECK-NEXT:   [[TRANSPOSE:%.*]] = IE.Transpose([[AVGPOOL]])
    // CHECK-SAME{LITERAL}: {order_value = #map} : tensor<1x512x768x1xf16> -> tensor<512x768x1x1xf16>
    // CHECK-NEXT:   return [[TRANSPOSE]] : tensor<512x768x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @NotSwapWithAvgPoolConvertInput
func.func @NotSwapWithAvgPoolConvertInput(%arg0: tensor<1x512x768x1xf32>) -> tensor<512x768x1x1xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x512x768x1xf32> -> tensor<1x512x768x1xf16>
    %1 = IE.Transpose(%0) {order_value = #map} : tensor<1x512x768x1xf16> -> tensor<512x768x1x1xf16>
    %2 = IE.AvgPool(%1) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<512x768x1x1xf16> -> tensor<512x768x1x1xf16>
    return %2: tensor<512x768x1x1xf16>

    // CHECK:        [[CONVERT:%.+]] = IE.Convert
    // CHECK-NEXT:   [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]])
    // CHECK-NEXT:   [[AVGPOOL:%.+]] = IE.AvgPool([[TRANSPOSE]])
    // CHECK:   return [[AVGPOOL]] : tensor<512x768x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @NotSwapWithAvgPoolAsChannelAlign
func.func @NotSwapWithAvgPoolAsChannelAlign(%arg0: tensor<1x24x768x1xf32>) -> tensor<24x768x1x1xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x24x768x1xf32> -> tensor<1x24x768x1xf16>
    %1 = IE.Transpose(%0) {order_value = #map} : tensor<1x24x768x1xf16> -> tensor<24x768x1x1xf16>
    %2 = IE.AvgPool(%1) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<24x768x1x1xf16> -> tensor<24x768x1x1xf16>
    return %2: tensor<24x768x1x1xf16>

    // CHECK:        [[CONVERT:%.+]] = IE.Convert
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]])
    // CHECK-SAME{LITERAL}: {order_value = #map} : tensor<1x24x768x1xf16> -> tensor<24x768x1x1xf16>
    // CHECK-NEXT:   [[AVGPOOL:%.+]] = IE.AvgPool([[TRANSPOSE]]) {exclude_pads, kernel_size = [1, 1],
    // CHECK-SAME:     tensor<24x768x1x1xf16> -> tensor<24x768x1x1xf16>
    // CHECK-NEXT:   return [[AVGPOOL]] : tensor<24x768x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @NotSwapWithAvgPoolAsModelInputToTranspose
func.func @NotSwapWithAvgPoolAsModelInputToTranspose(%arg0: tensor<1x512x768x1xf16>) -> tensor<512x768x1x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x512x768x1xf16> -> tensor<512x768x1x1xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<512x768x1x1xf16> -> tensor<512x768x1x1xf16>
    return %1: tensor<512x768x1x1xf16>

    // CHECK:        [[TRANSPOSE:%.*]] = IE.Transpose(%arg0)
    // CHECK-SAME{LITERAL}: {order_value = #map} : tensor<1x512x768x1xf16> -> tensor<512x768x1x1xf16>
    // CHECK-NEXT:   [[AVGPOOL:%.*]] = IE.AvgPool([[TRANSPOSE]]) {exclude_pads, kernel_size = [1, 1],
    // CHECK-SAME:     tensor<512x768x1x1xf16> -> tensor<512x768x1x1xf16>
    // CHECK-NEXT:   return [[AVGPOOL]] : tensor<512x768x1x1xf16>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithSwish
func.func @SwapWithSwish(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.Swish(%1) {beta_value = 1.000000e+00 : f64} : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>

    return %2 : tensor<1x1x16x24xf32>

    // CHECK: [[LAYER:%.*]] = IE.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK: [[TRANSPOSE:%.*]] = IE.Transpose([[LAYER]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithConvert
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x24x16x1xf32>
func.func @SwapWithConvert(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf16> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %2 = IE.Convert(%1) {dstElemType = f16} : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf16>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x16x24xf16>, tensor<1x1x16x24xf16> -> tensor<1x1x16x24xf16>
    return %3 : tensor<1x1x16x24xf16>

    // CHECK: [[LAYER:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[LAYER]]) {order_value = #NWHC} : tensor<1x24x16x1xf16> -> tensor<1x1x16x24xf16>
    // CHECK: [[ADD:%.+]] = IE.Add([[TRANSPOSE]], [[TRANSPOSE]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x16x24xf16>, tensor<1x1x16x24xf16> -> tensor<1x1x16x24xf16>
    // CHECK: return [[ADD]] : tensor<1x1x16x24xf16>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @NotSwapWithConvert
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x24x16x1xf16>
func.func @NotSwapWithConvert(%arg0 : tensor<1x24x16x1xf16>) -> tensor<1x1x16x24xf32> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf16> -> tensor<1x1x16x24xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x1x16x24xf16> -> tensor<1x1x16x24xf32>
    return %2 : tensor<1x1x16x24xf32>

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWHC} : tensor<1x24x16x1xf16> -> tensor<1x1x16x24xf16>
    // CHECK: [[LAYER:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32} : tensor<1x1x16x24xf16> -> tensor<1x1x16x24xf32>
    // CHECK: return [[LAYER]] : tensor<1x1x16x24xf32>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithConvertSubByte
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x24x16x1xi4>
func.func @SwapWithConvertSubByte(%arg0 : tensor<1x24x16x1xi4>) -> tensor<1x1x16x24xi1> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xi4> -> tensor<1x1x16x24xi4>
    %2 = IE.Convert(%1) {dstElemType = i1} : tensor<1x1x16x24xi4> -> tensor<1x1x16x24xi1>
    return %2 : tensor<1x1x16x24xi1>

    // CHECK: [[LAYER:%.+]] = IE.Convert([[INPUT]]) {dstElemType = i1} : tensor<1x24x16x1xi4> -> tensor<1x24x16x1xi1>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[LAYER]]) {order_value = #NWHC} : tensor<1x24x16x1xi1> -> tensor<1x1x16x24xi1>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xi1>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @NotSwapWithConvertSubByte
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x24x16x1xi1>
func.func @NotSwapWithConvertSubByte(%arg0 : tensor<1x24x16x1xi1>) -> tensor<1x1x16x24xi4> {
    %1 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xi1> -> tensor<1x1x16x24xi1>
    %2 = IE.Convert(%1) {dstElemType = i4} : tensor<1x1x16x24xi1> -> tensor<1x1x16x24xi4>
    return %2 : tensor<1x1x16x24xi4>

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWHC} : tensor<1x24x16x1xi1> -> tensor<1x1x16x24xi1>
    // CHECK: [[LAYER:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = i4} : tensor<1x1x16x24xi1> -> tensor<1x1x16x24xi4>
    // CHECK: return [[LAYER]] : tensor<1x1x16x24xi4>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d2, d0)>

// CHECK-LABEL: @SwapWithConstAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x16x128x1xf16>
func.func @SwapWithConstAdd(%arg0: tensor<1x16x128x1xf16>) -> tensor<1x16x1x128xf16> {
    %cst = const.Declare tensor<1x16x1x128xf16> = dense<1.0> : tensor<1x16x1x128xf16>
    %0 = IE.Transpose(%arg0) {order_value = #map1} : tensor<1x16x128x1xf16> -> tensor<16x1x128x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2], [3, 3]], shape_value = [1, 16, 1, 128]} : tensor<16x1x128x1xf16> -> tensor<1x16x1x128xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x128xf16>, tensor<1x16x1x128xf16> -> tensor<1x16x1x128xf16>

    return %2 : tensor<1x16x1x128xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x16x128x1xf16> = dense<1.000000e+00> : tensor<1x16x1x128xf16>, [#const.Reshape<[16, 1, 128, 1]>, #const.Transpose<#map>]
    // CHECK:           [[ADD:%.+]] = IE.Add([[INPUT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x128x1xf16>, tensor<1x16x128x1xf16> -> tensor<1x16x128x1xf16>
    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #map1} : tensor<1x16x128x1xf16> -> tensor<16x1x128x1xf16>
    // CHECK:           [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1], [2], [3, 3]], shape_value = [1, 16, 1, 128]} : tensor<16x1x128x1xf16> -> tensor<1x16x1x128xf16>
    // CHECK:           return [[AFFINE_RESHAPE]] : tensor<1x16x1x128xf16>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotSwapWithConstAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x9x128x16xf16>
func.func @NotSwapWithConstAdd(%arg0: tensor<1x9x128x16xf16>) -> tensor<1x144x128x1xf16> {
    %cst = const.Declare tensor<1x144x128x1xf16> = dense<1.0> : tensor<1x144x128x1xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x9x128x16xf16> -> tensor<1x16x9x128xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 144, 128, 1]} : tensor<1x16x9x128xf16> -> tensor<1x144x128x1xf16>
    %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x128x1xf16>, tensor<1x144x128x1xf16> -> tensor<1x144x128x1xf16>

    return %2 : tensor<1x144x128x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x144x128x1xf16> = dense<1.000000e+00> : tensor<1x144x128x1xf16>
    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x9x128x16xf16> -> tensor<1x16x9x128xf16>
    // CHECK:           [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 144, 128, 1]} : tensor<1x16x9x128xf16> -> tensor<1x144x128x1xf16>
    // CHECK:           [[ADD:%.+]] = IE.Add([[AFFINE_RESHAPE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x128x1xf16>, tensor<1x144x128x1xf16> -> tensor<1x144x128x1xf16>
    // CHECK:           return [[ADD]] : tensor<1x144x128x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0031202451855528589>
!qElemType1 = !quant.uniform<u8:f16, 0.0062404903711057178>
#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapWithQuantizeAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x144x2304x1xf16>
func.func @SwapWithQuantizeAdd(%arg0: tensor<1x144x2304x1xf16>) -> tensor<1x1x2304x144x!qElemType> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>} : tensor<1x144x2304x1xf16> -> tensor<2304x144x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 2304, 144]} : tensor<2304x144x1x1xf16> -> tensor<1x1x2304x144xf16>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1x2304x144xf16>, tensor<1x1x2304x144xf16> -> tensor<1x1x2304x144x!qElemType1>
    %3 = IE.QuantizeCast(%2) {dstElemType = !qElemType} : tensor<1x1x2304x144x!qElemType1> -> tensor<1x1x2304x144x!qElemType>

    return %3 : tensor<1x1x2304x144x!qElemType>

    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x144x2304x1xf16>, tensor<1x144x2304x1xf16> -> tensor<1x144x2304x1x!qElemType1>
    // CHECK:   [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x144x2304x1x!qElemType1> -> tensor<1x144x2304x1x!qElemType>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[QUANTIZE_CAST]]) {order_value = #map} : tensor<1x144x2304x1x!qElemType> -> tensor<2304x144x1x1x!qElemType>
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 2304, 144]} : tensor<2304x144x1x1x!qElemType> -> tensor<1x1x2304x144x!qElemType>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x1x2304x144x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0037092456630631989:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithDeQuantizeAdd
// CHECK-SAME:     [[INPUT:%arg[0-9]]]: tensor<1x512x2x1x!qElemType>
func.func @SwapWithDeQuantizeAdd(%arg0: tensor<1x512x2x1x!qElemType>) -> tensor<1x2x256x2xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x512x2x1x!qElemType> -> tensor<1x2x1x512x!qElemType>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 256, 2]} : tensor<1x2x1x512x!qElemType> -> tensor<1x2x256x2x!qElemType>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x2x256x2x!qElemType>, tensor<1x2x256x2x!qElemType> -> tensor<1x2x256x2xf16>

    return %2 : tensor<1x2x256x2xf16>

    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x2x1x!qElemType>, tensor<1x512x2x1x!qElemType> -> tensor<1x512x2x1xf16>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[ADD]]) {order_value = #NHWC} : tensor<1x512x2x1xf16> -> tensor<1x2x1x512xf16>
    // CHECK:   [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME{LITERAL}:    {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 256, 2]} : tensor<1x2x1x512xf16> -> tensor<1x2x256x2xf16>
    // CHECK:   return [[AFFINE_RESHAPE]] : tensor<1x2x256x2xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapWithMultiply
// CHECK-SAME:     ([[INPUT0:%arg[0-9]]]: tensor<1x1280x4096x1xf16>, [[INPUT1:%arg[0-9]]]: tensor<1x1280x4096x1xf16>)
func.func @SwapWithMultiply(%arg0: tensor<1x1280x4096x1xf16>, %arg1: tensor<1x1280x4096x1xf16>) -> tensor<4096x1280x1x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    %1 = IE.Transpose(%arg1) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x1280x1x1xf16>, tensor<4096x1280x1x1xf16> -> tensor<4096x1280x1x1xf16>
    return %2 : tensor<4096x1280x1x1xf16>

    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1280x4096x1xf16>, tensor<1x1280x4096x1xf16> -> tensor<1x1280x4096x1xf16>
    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[MULTIPLY]]) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    // CHECK:           return [[TRANSPOSE]] : tensor<4096x1280x1x1xf16>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
// CHECK: [[HCNW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK: @SwapWithBroadcastableMultiply
// CHECK-SAME:     ([[INPUT0:%arg[0-9]]]: tensor<1x11008x1024x1xf16>, [[INPUT1:%arg[0-9]]]: tensor<1x11008x1x1xf16>)
func.func @SwapWithBroadcastableMultiply(%arg0: tensor<1x11008x1024x1xf16>, %arg1: tensor<1x11008x1x1xf16>) -> tensor<1024x11008x1x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #HCNW} : tensor<1x11008x1024x1xf16> -> tensor<1024x11008x1x1xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1024x11008x1x1xf16>, tensor<1x11008x1x1xf16> -> tensor<1024x11008x1x1xf16>
    return %1 : tensor<1024x11008x1x1xf16>

    // CHECK:        [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT1]]) {order_value = [[HCNW]]}
    // CHECK-SAME:      : tensor<1x11008x1x1xf16> -> tensor<1x11008x1x1xf16>
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[INPUT0]], [[TRANSPOSE_IN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x11008x1024x1xf16>, tensor<1x11008x1x1xf16> -> tensor<1x11008x1024x1xf16>
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[MULTIPLY]]) {order_value = [[HCNW]]}
    // CHECK-SAME:      : tensor<1x11008x1024x1xf16> -> tensor<1024x11008x1x1xf16>
    // CHECK:        return [[TRANSPOSE]] : tensor<1024x11008x1x1xf16>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>
// CHECK: [[HCNW:#.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK: @NotSwapWithBroadcastableMultiplyMultipleUses
// CHECK-SAME:     ([[INPUT0:%arg[0-9]]]: tensor<1x11008x1024x1xf16>, [[INPUT1:%arg[0-9]]]: tensor<1x11008x1x1xf16>)
func.func @NotSwapWithBroadcastableMultiplyMultipleUses(
        %arg0: tensor<1x11008x1024x1xf16>, %arg1: tensor<1x11008x1x1xf16>)
            -> (tensor<1024x11008x1x1xf16>, tensor<1024x11008x1x1xf16>) {
    %0 = IE.Transpose(%arg0) {order_value = #HCNW} : tensor<1x11008x1024x1xf16> -> tensor<1024x11008x1x1xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
        : tensor<1024x11008x1x1xf16>, tensor<1x11008x1x1xf16> -> tensor<1024x11008x1x1xf16>
    %2 = IE.Swish(%0) {beta_value = 1.000000e+00 : f64}
    : tensor<1024x11008x1x1xf16> -> tensor<1024x11008x1x1xf16>
    return %1, %2 : tensor<1024x11008x1x1xf16>, tensor<1024x11008x1x1xf16>

    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT0]]) {order_value = [[HCNW]]}
    // CHECK-SAME:      : tensor<1x11008x1024x1xf16> -> tensor<1024x11008x1x1xf16>
    // CHECK:        [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1024x11008x1x1xf16>, tensor<1x11008x1x1xf16> -> tensor<1024x11008x1x1xf16>

}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @NotSwapWithMultiply
// CHECK-SAME:     ([[INPUT0:%arg[0-9]]]: tensor<1x1280x4096x1xf16>, [[INPUT1:%arg[0-9]]]: tensor<4096x1280x1x1xf16>)
func.func @NotSwapWithMultiply(%arg0: tensor<1x1280x4096x1xf16>, %arg1: tensor<4096x1280x1x1xf16>) -> tensor<4096x1280x1x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x1280x1x1xf16>, tensor<4096x1280x1x1xf16> -> tensor<4096x1280x1x1xf16>
    return %1 : tensor<4096x1280x1x1xf16>

    // CHECK:           [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT0]]) {order_value = #map} : tensor<1x1280x4096x1xf16> -> tensor<4096x1280x1x1xf16>
    // CHECK:           [[MULTIPLY:%.+]] = IE.Multiply([[TRANSPOSE]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x1280x1x1xf16>, tensor<4096x1280x1x1xf16> -> tensor<4096x1280x1x1xf16>
    // CHECK:           return [[MULTIPLY]] : tensor<4096x1280x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

// CHECK-LABEL: @SwapTwoSlices
// CHECK-SAME:     ([[IN:%arg[0-9]]]: tensor<1x320x4096x1xf16>, [[WEIGHTS:%arg[0-9]]]: tensor<2560x320x1x1xf16>)
func.func @SwapTwoSlices(
    %IN: tensor<1x320x4096x1xf16>,
    %WEIGHTS: tensor<2560x320x1x1xf16>
) -> (tensor<4096x1280x1x1xf16>, tensor<4096x1280x1x1xf16>) {
    %BIAS = const.Declare tensor<1x2560x1x1xf16> = dense<1.0> : tensor<1x2560x1x1xf16>
    // CHECK:           [[BIAS:%.+]] = const.Declare tensor<1x2560x1x1xf16>

    %CONV = IE.Convolution(%IN, %WEIGHTS, %BIAS) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x320x4096x1xf16>,
        tensor<2560x320x1x1xf16>,
        tensor<1x2560x1x1xf16>
            -> tensor<1x2560x4096x1xf16>
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[IN]], [[WEIGHTS]], [[BIAS]])

    %TRANSPOSE = IE.Transpose(%CONV) {
        order_value = #map
    } : tensor<1x2560x4096x1xf16> -> tensor<4096x2560x1x1xf16>

    %SLICE_0 = IE.Slice %TRANSPOSE [0, 0, 0, 0] [4096, 1280, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<4096x1280x1x1xf16>
    // CHECK-DAG:           [[SLICE_0:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 1280, 4096, 1]
    // CHECK-DAG:           [[TRANSPOSE_0:%.+]] = IE.Transpose([[SLICE_0]]) {order_value = #map}

    %SLICE_1 = IE.Slice %TRANSPOSE [0, 1280, 0, 0] [4096, 1280, 1, 1] : tensor<4096x2560x1x1xf16> to tensor<4096x1280x1x1xf16>
    // CHECK-DAG:           [[SLICE_1:%.+]] = IE.Slice [[CONV]] [0, 1280, 0, 0] [1, 1280, 4096, 1]
    // CHECK-DAG:           [[TRANSPOSE_1:%.+]] = IE.Transpose([[SLICE_1]]) {order_value = #map}

    return %SLICE_0, %SLICE_1 : tensor<4096x1280x1x1xf16>, tensor<4096x1280x1x1xf16>
    // CHECK:   return [[TRANSPOSE_0]], [[TRANSPOSE_1]]
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>

// CHECK-LABEL: @SwapWithTanh
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x121x2xf16>
func.func @SwapWithTanh(%arg0 : tensor<1x32x121x2xf16>) -> tensor<1x32x121x2xf16> {
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x32x121x2xf16> -> tensor<2x32x121x1xf16>
    %1 = IE.Tanh(%0) : tensor<2x32x121x1xf16> -> tensor<2x32x121x1xf16>
    %2 = IE.Transpose(%1) {order_value = #map} : tensor<2x32x121x1xf16> -> tensor<1x32x121x2xf16>

    return %2 : tensor<1x32x121x2xf16>

    // CHECK: [[LAYER:%.+]] = IE.Tanh([[INPUT]]) : tensor<1x32x121x2xf16> -> tensor<1x32x121x2xf16>
    // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[LAYER]]) {order_value = #map} : tensor<1x32x121x2xf16> -> tensor<2x32x121x1xf16>
    // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[TRANSPOSE_0]]) {order_value = #map} : tensor<2x32x121x1xf16> -> tensor<1x32x121x2xf16>
    // CHECK: return [[TRANSPOSE_1]] : tensor<1x32x121x2xf16>
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SwapWithAbs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x24x16x1xf32>
func.func @SwapWithAbs(%arg0 : tensor<1x24x16x1xf32>) -> tensor<1x1x16x24xf32> {
    %0 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    %1 = IE.Abs(%0) : tensor<1x1x16x24xf32> -> tensor<1x1x16x24xf32>
    return %1 : tensor<1x1x16x24xf32>

    // CHECK: [[LAYER:%.+]] = IE.Abs([[INPUT]]) : tensor<1x24x16x1xf32> -> tensor<1x24x16x1xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[LAYER]]) {order_value = #NWHC} : tensor<1x24x16x1xf32> -> tensor<1x1x16x24xf32>
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x16x24xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithFixedOrderSlice
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x192x9x16xf16>
func.func @SwapWithFixedOrderSlice(%arg0 : tensor<1x192x9x16xf16>) -> tensor<1x9x16x3xf16> {
    %cst = const.Declare tensor<1439x192x1x1xf16> = dense<1.0> : tensor<1439x192x1x1xf16>
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x192x9x16xf16>, tensor<1439x192x1x1xf16> -> tensor<1x1439x9x16xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x1439x9x16xf16> -> tensor<1x9x16x1439xf16>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 9, 16, 1] : tensor<1x9x16x1439xf16> to tensor<1x9x16x1xf16>
    %3 = IE.Sigmoid(%2) : tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    %4 = IE.Slice %1 [0, 0, 0, 1] [1, 9, 16, 1] : tensor<1x9x16x1439xf16> to tensor<1x9x16x1xf16>
    %5 = IE.Slice %1 [0, 0, 0, 2] [1, 9, 16, 1] : tensor<1x9x16x1439xf16> to tensor<1x9x16x1xf16>
    %6 = IE.Sigmoid(%5) : tensor<1x9x16x1xf16> -> tensor<1x9x16x1xf16>
    %7 = IE.Concat(%3, %4, %6) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<1x9x16x3xf16>

    return %7 : tensor<1x9x16x3xf16>

    // CHECK:  [[CONV:%.+]] = IE.Convolution
    // CHECK:  [[SLICE_0:%.+]] = IE.Slice [[CONV]] [0, 2, 0, 0] [1, 1, 9, 16] : tensor<1x1439x9x16xf16> to tensor<1x1x9x16xf16>
    // CHECK:  [[SLICE_1:%.+]] = IE.Slice [[CONV]] [0, 1, 0, 0] [1, 1, 9, 16] : tensor<1x1439x9x16xf16> to tensor<1x1x9x16xf16>
    // CHECK:  [[TRANSPOSE_0:%.+]] = IE.Transpose([[SLICE_1]]) {order_value = #NHWC} : tensor<1x1x9x16xf16> -> tensor<1x9x16x1xf16>
    // CHECK:  [[SLICE_2:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 1, 9, 16] : tensor<1x1439x9x16xf16> to tensor<1x1x9x16xf16>
    // CHECK:  [[SIGMOID_0:%.+]] = IE.Sigmoid([[SLICE_2]]) : tensor<1x1x9x16xf16> -> tensor<1x1x9x16xf16>
    // CHECK:  [[TRANSPOSE_1:%.+]] = IE.Transpose([[SIGMOID_0]]) {order_value = #NHWC} : tensor<1x1x9x16xf16> -> tensor<1x9x16x1xf16>
    // CHECK:  [[SIGMOID_1:%.+]] = IE.Sigmoid([[SLICE_0]]) : tensor<1x1x9x16xf16> -> tensor<1x1x9x16xf16>
    // CHECK:  [[TRANSPOSE_2:%.+]] = IE.Transpose([[SIGMOID_1]]) {order_value = #NHWC} : tensor<1x1x9x16xf16> -> tensor<1x9x16x1xf16>
    // CHECK:  [[CONCAT:%.+]] = IE.Concat([[TRANSPOSE_1]], [[TRANSPOSE_0]], [[TRANSPOSE_2]])
    // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16>, tensor<1x9x16x1xf16> -> tensor<1x9x16x3xf16>

    // CHECK: return [[CONCAT]] : tensor<1x9x16x3xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithRMS
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x256x24x64xf16>
func.func @SwapWithRMS(%arg0 : tensor<1x256x24x64xf16>) -> tensor<1x24x256x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<2.0> : tensor<1x1x1x64xf16>, [#const.Rescale<1.250000e-01 : f64>]

    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x256x24x64xf16> -> tensor<1x24x256x64xf16>
    %1 = IE.RMS(%0, %cst) {eps = 1.0132789611816406E-6 : f64} : tensor<1x24x256x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x24x256x64xf16>

    return %1 : tensor<1x24x256x64xf16>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<2.000000e+00> : tensor<1x1x1x64xf16>, [#const.Rescale<1.250000e-01 : f64>]
    // CHECK:        [[RMS:%.+]] = IE.RMS([[INPUT]], [[CST]]) {eps = 1.0132789611816406E-6 : f64} : tensor<1x256x24x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x256x24x64xf16>
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[RMS]]) {order_value = #NHCW} : tensor<1x256x24x64xf16> -> tensor<1x24x256x64xf16>
    // CHECK:        return [[TRANSPOSE]] : tensor<1x24x256x64xf16>
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @NotSwapWithRMS
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x256x24x64xf16>
func.func @NotSwapWithRMS(%arg0 : tensor<1x256x24x64xf16>) -> tensor<1x256x64x24xf16> {
    %cst = const.Declare tensor<1x1x1x24xf16> = dense<2.0> : tensor<1x1x1x24xf16>, [#const.Rescale<1.250000e-01 : f64>]

    %0 = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x256x24x64xf16> -> tensor<1x256x64x24xf16>
    %1 = IE.RMS(%0, %cst) {eps = 1.0132789611816406E-6 : f64} : tensor<1x256x64x24xf16>, tensor<1x1x1x24xf16> -> tensor<1x256x64x24xf16>

    return %1 : tensor<1x256x64x24xf16>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x24xf16> = dense<2.000000e+00> : tensor<1x1x1x24xf16>, [#const.Rescale<1.250000e-01 : f64>]
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NCWH} : tensor<1x256x24x64xf16> -> tensor<1x256x64x24xf16>
    // CHECK:        [[RMS:%.+]] = IE.RMS([[TRANSPOSE]], [[CST]]) {eps = 1.0132789611816406E-6 : f64} : tensor<1x256x64x24xf16>, tensor<1x1x1x24xf16> -> tensor<1x256x64x24xf16>
    // CHECK:        return [[RMS]] : tensor<1x256x64x24xf16>
}

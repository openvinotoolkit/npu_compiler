//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --optimize-mempermute-and-activation-channels-expand %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MemPermuteProcessingWithFusing
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x32x64xf16, {order = #NHWC}>
func.func @MemPermuteProcessingWithFusing(%arg0: tensor<1x3x32x64xf16, {order = #NHWC}>) -> tensor<1x3x64x32xf16> {
    %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<3x3x1x1xf16, {order = #NHWC}>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x3x32x64xf16, {order = #NHWC}>,
        tensor<3x3x1x1xf16, {order = #NHWC}>
            -> tensor<1x3x32x64xf16, {order = #NHWC}>

    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x3x32x64xf16, {order = #NHWC}> -> tensor<1x3x32x64xf16>
    %2 = IE.Transpose(%1) {order_value = #NCWH} : tensor<1x3x32x64xf16> -> tensor<1x3x64x32xf16>

    return %2 : tensor<1x3x64x32xf16>

    // CHECK:    [[CST:%.*]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x3x1x1xf16, {order = #NHWC}>, [#const.PadWithZero<[0, 0, 0, 0], [13, 13, 0, 0]>]
    // CHECK:    [[EXPAND:%.*]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK:    [[CONV:%.*]] = IE.Convolution([[EXPAND]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x64xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NCWH}>
    // CHECK:    [[SLICE:%.*]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 3, 32, 64] : tensor<1x16x32x64xf16, {order = #NCWH}> to tensor<1x3x32x64xf16, {order = #NCWH}>
    // CHECK:    [[PERMUTECAST:%.*]] = IE.PermuteCast([[SLICE]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x3x32x64xf16, {order = #NCWH}> -> tensor<1x3x64x32xf16>
    // CHECK:    return [[PERMUTECAST]] : tensor<1x3x64x32xf16>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @MemPermuteProcessingConvertPass
func.func @MemPermuteProcessingConvertPass(%arg0: tensor<1x2x3x4xf32>,
                            %arg1: tensor<1x2x3x4xf32, {order = #NHWC}>,
                            %arg2: tensor<1x2x3x4xf32>) ->
                        (tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>) {
    %0 = IE.Transpose(%arg0) {order_value = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>

    %1 = IE.Reorder(%arg1) {dstOrder = #NCHW} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>

    %2 = IE.Reorder(%arg2) {dstOrder = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    return %0, %1, %2 : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>

    // CHECK-NOT: IE.Transpose
    // CHECK-NOT: IE.Reorder
    // CHECK-NOT: IE.Reorder
    // CHECK:     %[[VAL0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32> -> tensor<1x4x2x3xf32>
    // CHECK:     %[[VAL1:.*]] = IE.MemPermute(%arg1) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x2x3x4xf32, {order = #NHWC}> -> tensor<1x2x3x4xf32>
    // CHECK:     %[[VAL2:.*]] = IE.MemPermute(%arg2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x2x3x4xf32> -> tensor<1x2x3x4xf32, {order = #NHWC}>
    // CHECK:     return %[[VAL0]], %[[VAL1:.*]], %[[VAL2:.*]] : tensor<1x4x2x3xf32>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0038768677150501927>

// CHECK-LABEL: @MemPermuteProcessingWithReorderSoftMax
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x16x512x512xf16, {order = #NHWC}>
func.func @MemPermuteProcessingWithReorderSoftMax(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>) ->  tensor<1x16x512x512x!qElemType, {order = #NHWC}> {
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHWC}>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    // CHECK:  [[MAXPOOL:%.*]] = IE.MaxPool([[INPUT0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x512x512xf16, {order = #NHWC}> -> tensor<1x16x512x512xf16, {order = #NHCW}>
    // CHECK:  [[SOFTMAX:%.*]] = IE.SoftMax([[MAXPOOL]]) {axisInd = 3 : i64} : tensor<1x16x512x512xf16, {order = #NHCW}> -> tensor<1x16x512x512xf16, {order = #NHCW}>
    // CHECK:  [[PERMUTECAST0:%.*]] = IE.PermuteCast([[SOFTMAX]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x512x512xf16, {order = #NHCW}> -> tensor<1x512x512x16xf16, {order = #NHWC}>
    // CHECK:  [[AVGPOOL:%.*]] = IE.AvgPool([[PERMUTECAST0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x512x512x16xf16, {order = #NHWC}> -> tensor<1x512x512x16x!qElemType, {order = #NHCW}>
    // CHECK:  [[PERMUTECAST1:%.*]] = IE.PermuteCast([[AVGPOOL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x512x16x!qElemType, {order = #NHCW}> -> tensor<1x16x512x512x!qElemType, {order = #NHWC}>
    // CHECK:  return [[PERMUTECAST1]] : tensor<1x16x512x512x!qElemType, {order = #NHWC}>
}

// -----
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @MemPermuteProcessingWithNDReorder
func.func @MemPermuteProcessingWithNDReorder(%arg0: tensor<6x10x10x4x1xf16, {order = #map}>) -> tensor<6x10x10x4x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<6x10x10x4x1xf16, {order = #map}> -> tensor<6x10x10x4x1xf16>
    return %0 : tensor<6x10x10x4x1xf16>

    // CHECK-NOT: IE.Reorder
    // CHECK: [[VAL0:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<6x10x10x4x1xf16, {order = #map}> -> tensor<1x10x10x4x6xf16>
    // CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 100, 4, 6]} : tensor<1x10x10x4x6xf16> -> tensor<1x100x4x6xf16>
    // CHECK: [[VAL2:%.*]] = IE.MemPermute([[VAL1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x100x4x6xf16> -> tensor<1x6x100x4xf16>
    // CHECK: [[VAL3:%.*]] = IE.AffineReshape([[VAL2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [6, 10, 10, 4, 1]} : tensor<1x6x100x4xf16> -> tensor<6x10x10x4x1xf16>

    // CHECK: return [[VAL3]] : tensor<6x10x10x4x1xf16>
}

// -----
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @MemPermuteProcessingForNHCWMemPermute
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x380x720x1xf16>
func.func @MemPermuteProcessingForNHCWMemPermute(%arg0: tensor<1x380x720x1xf16>) -> tensor<1x720x380x1xf16> {
    %0 = IE.MemPermute(%arg0) {dstElemType = f16, dst_order = #NCHW, mem_perm = #NHCW} :
        tensor<1x380x720x1xf16> -> tensor<1x720x380x1xf16>

    return %0 : tensor<1x720x380x1xf16>

    // CHECK: [[PERMUTECAST_IN:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x380x720x1xf16> -> tensor<1x720x380x1xf16, {order = #NHWC}>
    // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x720x380x1xf16, {order = #NHWC}> -> tensor<1x720x380x1xf16, {order = #NCWH}>
    // CHECK: [[PERMUTECAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x720x380x1xf16, {order = #NCWH}> -> tensor<1x720x380x1xf16>

    // CHECK: return [[PERMUTECAST_OUT]] : tensor<1x720x380x1xf16>
}

// -----

// CHECK-LABEL: @MemPermuteProcessingForNHWCMemPermute
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x75x16x64xf16>
func.func @MemPermuteProcessingForNHWCMemPermute(%arg0: tensor<4x75x16x64xf16>) -> tensor<4x16x64x75xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<4x75x16x64xf16> -> tensor<4x16x64x75xf16>

    return %0 : tensor<4x16x64x75xf16>

    // CHECK:  [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 4, 75, 1024]} inputs([[INPUT]] : tensor<4x75x16x64xf16>) -> tensor<1x4x75x1024xf16>
    // CHECK:  [[PERMUTECAST1:%.+]] = IE.PermuteCast([[SHAPECAST0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x75x1024xf16> -> tensor<1x1024x4x75xf16, {order = #NHWC}>
    // CHECK:  [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1024x4x75xf16, {order = #NHWC}> -> tensor<1x1024x4x75xf16, {order = #NHCW}>
    // CHECK:  [[PERMUTECAST3:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1024x4x75xf16, {order = #NHCW}> -> tensor<1x4x1024x75xf16>
    // CHECK:  [[SHAPECAST4:%.+]] = IE.ShapeCast {shape = [4, 16, 64, 75]} inputs([[PERMUTECAST3]] : tensor<1x4x1024x75xf16>) -> tensor<4x16x64x75xf16>

    // CHECK: return [[SHAPECAST4]] : tensor<4x16x64x75xf16>
}

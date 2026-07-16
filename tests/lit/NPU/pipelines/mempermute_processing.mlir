//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --mempermute-processing %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

 // CHECK-LABEL: func @MemPermuteInBetweenShapeCast
 // CHECK-SAME:  ([[ARG:%.+]]: tensor<2x3x96x96xf16>) -> tensor<2x1x94x94xf16>
func.func @MemPermuteInBetweenShapeCast(%arg0: tensor<2x3x96x96xf16>) -> tensor<2x1x94x94xf16> {
  %cst = const.Declare tensor<1x3x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<0.0> : tensor<1x3x3x3xf16>, [#const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
  %0 = IE.ShapeCast {shape = [1, 2, 3, 9216]} inputs(%arg0 : tensor<2x3x96x96xf16>) -> tensor<1x2x3x9216xf16>
  %1 = IE.MemPermute(%0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x2x3x9216xf16> -> tensor<1x3x2x9216xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %2 = IE.ShapeCast {shape = [2, 3, 96, 96]} inputs(%1 : tensor<1x3x2x9216xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) -> tensor<2x3x96x96xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %3 = IE.Convolution(%2, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x3x96x96xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<1x3x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<2x1x94x94xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  %4 = IE.PermuteCast(%3) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<2x1x94x94xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<2x1x94x94xf16>
  return %4 : tensor<2x1x94x94xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x3x3x3xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x3x3x3xf16>, [#const.Reorder<#NHWC>]
  // CHECK: [[SHAPCAST:%.+]] = IE.ShapeCast {shape = [1, 2, 3, 9216]} inputs([[ARG]] : tensor<2x3x96x96xf16>) -> tensor<1x2x3x9216xf16>
  // CHECK: [[PERMUTE_CAST_IN:%.+]] = IE.PermuteCast([[SHAPCAST]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x2x3x9216xf16> -> tensor<1x9216x2x3xf16, {order = #NHWC}>
  // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTE_CAST_IN]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x9216x2x3xf16, {order = #NHWC}> -> tensor<1x9216x2x3xf16, {order = #NHCW}>
  // CHECK: [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x9216x2x3xf16, {order = #NHCW}> -> tensor<1x3x2x9216xf16, {order = #NHWC}>
  // CHECK: [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [2, 3, 96, 96]} inputs([[PERMUTE_CAST_OUT]] : tensor<1x3x2x9216xf16, {order = #NHWC}>) -> tensor<2x3x96x96xf16, {order = #NHWC}>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[SHAPECAST]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<2x3x96x96xf16, {order = #NHWC}>, tensor<1x3x3x3xf16, {order = #NHWC}> -> tensor<2x1x94x94xf16, {order = #NHWC}>
  // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[CONV]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<2x1x94x94xf16, {order = #NHWC}> -> tensor<2x1x94x94xf16>
  // CHECK: return [[PERMUTECAST]] : tensor<2x1x94x94xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @MemPermuteInBetweenAffineReshape
// CHECK-SAME:  ([[ARG_0:%.+]]: tensor<1x112x4x128xf16, {order = #NHWC}>, [[ARG_1:%.+]]: tensor<112x1x1x1xf16, {order = #NHWC}>) -> tensor<1x512x120x1xf16>
func.func @MemPermuteInBetweenAffineReshape(%arg0: tensor<1x112x4x128xf16, {order = #NHWC}>, %arg1: tensor<112x1x1x1xf16, {order = #NHWC}>) -> tensor<1x512x120x1xf16> {
  %0 = const.Declare tensor<1x512x10x1xf16> = dense<0.000000e+00> : tensor<1x512x10x1xf16>
  %1 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 112 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x112x4x128xf16, {order = #NHWC}>, tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<1x112x4x128xf16, {order = #NHWC}>
  %2 = IE.Slice %1 [0, 0, 0, 0] [1, 100, 4, 128] : tensor<1x112x4x128xf16, {order = #NHWC}> to tensor<1x100x4x128xf16, {order = #NHWC}>
  %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 100, 1, 512]} : tensor<1x100x4x128xf16, {order = #NHWC}> -> tensor<1x100x1x512xf16, {order = #NHWC}>
  %4 = IE.MemPermute(%3) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x100x1x512xf16, {order = #NHWC}> -> tensor<1x100x1x512xf16>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 100, 512]} : tensor<1x100x1x512xf16> -> tensor<1x1x100x512xf16>
  %6 = IE.MemPermute(%4) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x100x1x512xf16> -> tensor<1x512x1x100xf16>
  %7 = IE.AffineReshape(%6) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 100, 1]} : tensor<1x512x1x100xf16> -> tensor<1x512x100x1xf16>
  %8 = IE.Concat(%0, %7, %0) {static_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 110, 0]]} : tensor<1x512x10x1xf16>, tensor<1x512x100x1xf16>, tensor<1x512x10x1xf16> -> tensor<1x512x120x1xf16>

  return %8 : tensor<1x512x120x1xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x512x10x1xf16> = dense<0.000000e+00> : tensor<1x512x10x1xf16>
  // CHECK: [[GROUPCONV:%.+]] = IE.GroupConvolution([[ARG_0]], [[ARG_1]]) {dilations = [1, 1], groups = 112 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x112x4x128xf16, {order = #NHWC}>, tensor<112x1x1x1xf16, {order = #NHWC}> -> tensor<1x112x4x128xf16, {order = #NHWC}>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[GROUPCONV]] [0, 0, 0, 0] [1, 100, 4, 128] : tensor<1x112x4x128xf16, {order = #NHWC}> to tensor<1x100x4x128xf16, {order = #NHWC}>
  // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x100x4x128xf16, {order = #NHWC}> -> tensor<1x4x128x100xf16>
  // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 512, 100, 1]} : tensor<1x4x128x100xf16> -> tensor<1x512x100x1xf16>
  // CHECK: [[CONCAT:%.+]] = IE.Concat([[CST]], [[AFFINERESHAPE]], [[CST]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 10, 0], [0, 0, 110, 0]]} : tensor<1x512x10x1xf16>, tensor<1x512x100x1xf16>, tensor<1x512x10x1xf16> -> tensor<1x512x120x1xf16>
  // CHECK: return [[CONCAT]] : tensor<1x512x120x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK: func @UniquifyOpsPipeline([[ARG:%.+]]: tensor<1x2x3xf32>)
// CHECK-SAME: -> (tensor<1x2x3x1xf32>, tensor<1x3x2x1xf32, {order = #NHCW}>)
func.func @UniquifyOpsPipeline(%arg: tensor<1x2x3xf32>)
        -> (tensor<1x2x3x1xf32>, tensor<1x3x2x1xf32, {order = #NHCW}>) {
    // optimized by regular -uniquify-ops / -cse:
    %reshape0 = IE.Reshape(%arg) {shape_value = [1, 1, 2, 3]} : tensor<1x2x3xf32> -> tensor<1x1x2x3xf32>
    %reshape1 = IE.Reshape(%arg) {shape_value = [1, 1, 2, 3]} : tensor<1x2x3xf32> -> tensor<1x1x2x3xf32>

    // optimized by -uniquify-similar-ops:
    %permute0 = IE.MemPermute(%reshape0) {dst_order = #NCHW, mem_perm = #NHWC} :
        tensor<1x1x2x3xf32> -> tensor<1x2x3x1xf32>
    %permute1 = IE.MemPermute(%reshape1) {dst_order = #NHCW, mem_perm = #NHWC} :
        tensor<1x1x2x3xf32> -> tensor<1x3x2x1xf32, {order = #NHCW}>

    return %permute0, %permute1 : tensor<1x2x3x1xf32>, tensor<1x3x2x1xf32, {order = #NHCW}>

    // CHECK: [[SINGLE_RESHAPE:%.+]] = IE.AffineReshape([[ARG]]) {{.+}} shape_value = [1, 1, 2, 3]
    // CHECK: [[MEM_PERM0:%.+]] = IE.PermuteCast([[SINGLE_RESHAPE]]) {dst_order = #NCHW, mem_perm = #NHWC}
    // CHECK: [[MEM_PERM1:%.+]] = IE.PermuteCast([[SINGLE_RESHAPE]]) {dst_order = #NHCW, mem_perm = #NHWC}
    // CHECK: return [[MEM_PERM0]], [[MEM_PERM1]]
}

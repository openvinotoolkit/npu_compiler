//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --convert-extract-image-patches %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertExtractImagePatchesTransposeAffineReshapeToSliceConcat
func.func @ConvertExtractImagePatchesTransposeAffineReshapeToSliceConcat(%arg0: tensor<1x1x6x5xf16>) -> tensor<1x4x3x5xf16> {
  %0 = IE.ExtractImagePatches(%arg0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 5], strides = [1, 1]} : tensor<1x1x6x5xf16> -> tensor<1x15x4x1xf16>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x15x4x1xf16> -> tensor<1x4x1x15xf16>
  %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 4, 3, 5]} : tensor<1x4x1x15xf16> -> tensor<1x4x3x5xf16>
  return %2 : tensor<1x4x3x5xf16>

  //CHECK: [[VAL0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 3, 5] : tensor<1x1x6x5xf16> to tensor<1x1x3x5xf16>
  //CHECK: [[VAL1:%.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 1, 3, 5] : tensor<1x1x6x5xf16> to tensor<1x1x3x5xf16>
  //CHECK: [[VAL2:%.*]] = IE.Slice %arg0 [0, 0, 2, 0] [1, 1, 3, 5] : tensor<1x1x6x5xf16> to tensor<1x1x3x5xf16>
  //CHECK: [[VAL3:%.*]] = IE.Slice %arg0 [0, 0, 3, 0] [1, 1, 3, 5] : tensor<1x1x6x5xf16> to tensor<1x1x3x5xf16>
  //CHECK: [[VAL4:%.*]] = IE.Concat([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x3x5xf16>, tensor<1x1x3x5xf16>, tensor<1x1x3x5xf16>, tensor<1x1x3x5xf16> -> tensor<1x4x3x5xf16>
  //CHECK: return [[VAL4]] : tensor<1x4x3x5xf16>
}

// -----

// CHECK-LABEL: @ConvertExtractImagePatchesTransposeToSliceConcatAffineReshape
func.func @ConvertExtractImagePatchesTransposeToSliceConcatAffineReshape(%arg0: tensor<1x1x7x4xf16>) -> tensor<1x5x1x12xf16> {
  %0 = IE.ExtractImagePatches(%arg0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 4], strides = [1, 1]} : tensor<1x1x7x4xf16> -> tensor<1x12x5x1xf16>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x12x5x1xf16> -> tensor<1x5x1x12xf16>
  return %1 : tensor<1x5x1x12xf16>

  //CHECK: [[VAL0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 3, 4] : tensor<1x1x7x4xf16> to tensor<1x1x3x4xf16>
  //CHECK: [[VAL1:%.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 1, 3, 4] : tensor<1x1x7x4xf16> to tensor<1x1x3x4xf16>
  //CHECK: [[VAL2:%.*]] = IE.Slice %arg0 [0, 0, 2, 0] [1, 1, 3, 4] : tensor<1x1x7x4xf16> to tensor<1x1x3x4xf16>
  //CHECK: [[VAL3:%.*]] = IE.Slice %arg0 [0, 0, 3, 0] [1, 1, 3, 4] : tensor<1x1x7x4xf16> to tensor<1x1x3x4xf16>
  //CHECK: [[VAL4:%.*]] = IE.Slice %arg0 [0, 0, 4, 0] [1, 1, 3, 4] : tensor<1x1x7x4xf16> to tensor<1x1x3x4xf16>
  //CHECK: [[VAL5:%.*]] = IE.Concat([[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]], [[VAL4]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16>, tensor<1x1x3x4xf16> -> tensor<1x5x3x4xf16>
  //CHECK: [[VAL6:%.*]] = IE.AffineReshape([[VAL5]]) {
  //CHECK-SAME{LITERAL}:     dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 5, 1, 12]
  //CHECK-same: } : tensor<1x5x3x4xf16> -> tensor<1x5x1x12xf16>
  //CHECK: return [[VAL6]] : tensor<1x5x1x12xf16>
}
// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NotConvertExtractImagePatchesTransposeToSliceConcatAffineReshape
func.func @NotConvertExtractImagePatchesTransposeToSliceConcatAffineReshape(%arg0: tensor<1x1x7x4xf16>) -> tensor<1x5x12x1xf16> {
  %0 = IE.ExtractImagePatches(%arg0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 4], strides = [1, 1]} : tensor<1x1x7x4xf16> -> tensor<1x12x5x1xf16>
  %1 = IE.Transpose(%0) {order_value = #NHCW} : tensor<1x12x5x1xf16> -> tensor<1x5x12x1xf16>
  return %1 : tensor<1x5x12x1xf16>

  //CHECK: [[VAL0:%.*]] = IE.ExtractImagePatches(%arg0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [3, 4], strides = [1, 1]} : tensor<1x1x7x4xf16> -> tensor<1x12x5x1xf16>
  //CHECK: [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NHCW} : tensor<1x12x5x1xf16> -> tensor<1x5x12x1xf16>
  //CHECK: return [[VAL1]] : tensor<1x5x12x1xf16>
}

// -----

// CHECK-LABEL: @FuseReduceSumExtractImagePatchesReduceSumToReduceSumSqueeze
func.func @FuseReduceSumExtractImagePatchesReduceSumToReduceSumSqueeze(%arg0: tensor<1x64x51x40xf16>) -> tensor<1x51x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x64x51x40xf16> -> tensor<1x1x51x40xf16>
  %1 = IE.ExtractImagePatches(%0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [1, 40], strides = [1, 1]} : tensor<1x1x51x40xf16> -> tensor<1x40x51x1xf16>
  %2 = IE.ReduceSum(%1) {axes_value = [1]}: tensor<1x40x51x1xf16> -> tensor<1x51x1xf16>
  return %2 : tensor<1x51x1xf16>

  //CHECK: [[VAL0:%.*]] = IE.ReduceSum(%arg0) {axes_value = [1, 3]} : tensor<1x64x51x40xf16> -> tensor<1x51xf16>
  //CHECK: [[VAL1:%.*]] = IE.Unsqueeze([[VAL0]]) {axes_value = [2]} : tensor<1x51xf16> -> tensor<1x51x1xf16>
  //CHECK: return [[VAL1]] : tensor<1x51x1xf16>
}

// -----

// CHECK-LABEL: @FuseReduceSumExtractImagePatchesTransposeReduceSumToReduceSumSqueeze
func.func @FuseReduceSumExtractImagePatchesTransposeReduceSumToReduceSumSqueeze(%arg0: tensor<1x8x10x5xf16>) -> tensor<1x10x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x8x10x5xf16> -> tensor<1x1x10x5xf16>
  %1 = IE.ExtractImagePatches(%0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [1, 5], strides = [1, 1]} : tensor<1x1x10x5xf16> -> tensor<1x5x10x1xf16>
  %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x5x10x1xf16> -> tensor<1x10x1x5xf16>
  %3 = IE.ReduceSum(%2) {axes_value = [3]}: tensor<1x10x1x5xf16> -> tensor<1x10x1xf16>
  return %3 : tensor<1x10x1xf16>

  //CHECK: [[VAL0:%.*]] = IE.ReduceSum(%arg0) {axes_value = [1, 3]} : tensor<1x8x10x5xf16> -> tensor<1x10xf16>
  //CHECK: [[VAL1:%.*]] = IE.Unsqueeze([[VAL0]]) {axes_value = [2]} : tensor<1x10xf16> -> tensor<1x10x1xf16>
  //CHECK: return [[VAL1]] : tensor<1x10x1xf16>
}

// -----

// CHECK-LABEL: @FuseReduceSumExtractImagePatchesTransposeReduceSumToReduceSum
func.func @FuseReduceSumExtractImagePatchesTransposeReduceSumToReduceSum(%arg0: tensor<1x8x10x5xf16>) -> tensor<1x10x1x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x8x10x5xf16> -> tensor<1x1x10x5xf16>
  %1 = IE.ExtractImagePatches(%0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [1, 5], strides = [1, 1]} : tensor<1x1x10x5xf16> -> tensor<1x5x10x1xf16>
  %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x5x10x1xf16> -> tensor<1x10x1x5xf16>
  %3 = IE.ReduceSum(%2) {axes_value = [3], keep_dims} : tensor<1x10x1x5xf16> -> tensor<1x10x1x1xf16>
  return %3 : tensor<1x10x1x1xf16>

  //CHECK: [[VAL0:%.*]] = IE.ReduceSum(%arg0) {axes_value = [1, 3]} : tensor<1x8x10x5xf16> -> tensor<1x10xf16>
  //CHECK: [[VAL1:%.*]] = IE.Unsqueeze([[VAL0]]) {axes_value = [2, 3]} : tensor<1x10xf16> -> tensor<1x10x1x1xf16>
  //CHECK: return [[VAL1]] : tensor<1x10x1x1xf16>
}

// -----
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @ConvertExtractImagePatchesToTranspose
func.func @ConvertExtractImagePatchesToTranspose(%arg0: tensor<1x16x8x12xf16>) -> tensor<1x12x8x1xf16> {
  %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x8x12xf16> -> tensor<1x1x8x12xf16>
  %1 = IE.ExtractImagePatches(%0) {autoPad = #IE.pad_type<VALID>, rates = [1, 1], sizes = [1, 12], strides = [1, 1]} : tensor<1x1x8x12xf16> -> tensor<1x12x8x1xf16>
  return %1 : tensor<1x12x8x1xf16>

  //CHECK: [[VAL0:%.*]] = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x16x8x12xf16> -> tensor<1x1x8x12xf16>
  //CHECK: [[VAL1:%.*]] = IE.Transpose([[VAL0]]) {order_value = #NWHC} : tensor<1x1x8x12xf16> -> tensor<1x12x8x1xf16>
  //CHECK: return [[VAL1]] : tensor<1x12x8x1xf16>
}

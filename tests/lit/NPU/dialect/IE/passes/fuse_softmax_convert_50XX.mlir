//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-softmax-convert --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @FuseSoftmaxConvert
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x548x600xf32>)
func.func @FuseSoftmaxConvert(%arg0: tensor<1x548x600xf32>) -> tensor<1x548x600xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x548x600xf32> -> tensor<1x548x600xf16>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x548x600xf16> -> tensor<1x548x600xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x548x600xf16> -> tensor<1x548x600xf32>

    return %2 : tensor<1x548x600xf32>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x548x600xf32> -> tensor<1x548x600xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONVERT]]) {axisInd = 2 : i64, dstElemType = f32} : tensor<1x548x600xf16> -> tensor<1x548x600xf32>
    // CHECK: return [[SOFTMAX]] : tensor<1x548x600xf32>
}

// -----

// CHECK-LABEL: @DoNotFuseSoftmaxConvertAxisNotInner
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x548x600xf32>)
func.func @DoNotFuseSoftmaxConvertAxisNotInner(%arg0: tensor<1x548x600xf32>) -> tensor<1x548x600xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x548x600xf32> -> tensor<1x548x600xf16>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x548x600xf16> -> tensor<1x548x600xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x548x600xf16> -> tensor<1x548x600xf32>

    return %2 : tensor<1x548x600xf32>

    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x548x600xf32> -> tensor<1x548x600xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONVERT1]]) {axisInd = 1 : i64} : tensor<1x548x600xf16> -> tensor<1x548x600xf16>
    // CHECK: [[CONVERT2:%.+]] = IE.Convert([[SOFTMAX]]) {dstElemType = f32} : tensor<1x548x600xf16> -> tensor<1x548x600xf32>
    // CHECK: return [[CONVERT2]] : tensor<1x548x600xf32>
}

// -----

// CHECK-LABEL: @DoNotFuseSoftMaxConvertMoreUses
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x548x600xf32>)
func.func @DoNotFuseSoftMaxConvertMoreUses(%arg0: tensor<1x548x600xf32>) -> (tensor<1x548x600xf32>, tensor<1x548x600xf16>) {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x548x600xf32> -> tensor<1x548x600xf16>
    %1 = IE.SoftMax(%0) {axisInd = 2 : i64} : tensor<1x548x600xf16> -> tensor<1x548x600xf16>
    %2 = IE.Convert(%1) {dstElemType = f32} : tensor<1x548x600xf16> -> tensor<1x548x600xf32>

    return %2, %1 : tensor<1x548x600xf32>, tensor<1x548x600xf16>

    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x548x600xf32> -> tensor<1x548x600xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONVERT1]]) {axisInd = 2 : i64} : tensor<1x548x600xf16> -> tensor<1x548x600xf16>
    // CHECK: [[CONVERT2:%.+]] = IE.Convert([[SOFTMAX]]) {dstElemType = f32} : tensor<1x548x600xf16> -> tensor<1x548x600xf32>
    // CHECK: return [[CONVERT2]], [[SOFTMAX]] : tensor<1x548x600xf32>, tensor<1x548x600xf16>
}

// -----

// CHECK-LABEL: @FuseSoftMaxShapeCastConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x64x33xf32>)
func.func @FuseSoftMaxShapeCastConvert(%arg0: tensor<1x1x64x33xf32>) -> tensor<1x1x64x33xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x64x33xf32> -> tensor<1x1x64x33xf16>
    %1 = IE.ShapeCast {shape = [1, 8, 8, 33]} inputs(%0 : tensor<1x1x64x33xf16>) -> tensor<1x8x8x33xf16>
    %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x8x33xf16> -> tensor<1x8x8x33xf16>
    %3 = IE.ShapeCast {shape = [1, 1, 64, 33]} inputs(%2 : tensor<1x8x8x33xf16>) -> tensor<1x1x64x33xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x64x33xf16> -> tensor<1x1x64x33xf32>

    return %4 : tensor<1x1x64x33xf32>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x64x33xf32> -> tensor<1x1x64x33xf16>
    // CHECK: [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 8, 8, 33]} inputs([[CONVERT]] : tensor<1x1x64x33xf16>) -> tensor<1x8x8x33xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[SHAPECAST1]]) {axisInd = 3 : i64, dstElemType = f32} : tensor<1x8x8x33xf16> -> tensor<1x8x8x33xf32>
    // CHECK: [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 1, 64, 33]} inputs([[SOFTMAX]] : tensor<1x8x8x33xf32>) -> tensor<1x1x64x33xf32>
    // CHECK: return [[SHAPECAST2:%.+]] : tensor<1x1x64x33xf32>
}

// -----

// CHECK-LABEL: @FuseSoftMaxShapeCastAffineReshapeConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<55x32x64x33xf32>)
func.func @FuseSoftMaxShapeCastAffineReshapeConvert(%arg0: tensor<55x32x64x33xf32>) -> tensor<55x32x64x33xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<55x32x64x33xf32> -> tensor<55x32x64x33xf16>
    %1 = IE.ShapeCast {shape = [1, 1760, 64, 33]} inputs(%0 : tensor<55x32x64x33xf16>) -> tensor<1x1760x64x33xf16>
    %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x1760x64x33xf16> -> tensor<1x1760x64x33xf16>
    %3 = IE.ShapeCast {shape = [55, 32, 64, 33]} inputs(%2 : tensor<1x1760x64x33xf16>) -> tensor<55x32x64x33xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1], [1], [2], [3]], shape_value = [1, 1760, 64, 33]} : tensor<55x32x64x33xf16> -> tensor<1x1760x64x33xf16>
    %5 = IE.Convert(%4) {dstElemType = f32} : tensor<1x1760x64x33xf16> -> tensor<1x1760x64x33xf32>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [0, 1], [2], [3]], shape_value = [55, 32, 64, 33]} : tensor<1x1760x64x33xf32> -> tensor<55x32x64x33xf32>

    return %6 : tensor<55x32x64x33xf32>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<55x32x64x33xf32> -> tensor<55x32x64x33xf16>
    // CHECK: [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 1760, 64, 33]} inputs([[CONVERT]] : tensor<55x32x64x33xf16>) -> tensor<1x1760x64x33xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[SHAPECAST1]]) {axisInd = 3 : i64, dstElemType = f32} : tensor<1x1760x64x33xf16> -> tensor<1x1760x64x33xf32>
    // CHECK: [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [55, 32, 64, 33]} inputs([[SOFTMAX]] : tensor<1x1760x64x33xf32>) -> tensor<55x32x64x33xf32>
    // CHECK: return [[SHAPECAST2:%.+]] : tensor<55x32x64x33xf32>
}

// -----

// CHECK-LABEL: @FuseSoftMaxAffineReshapeConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16800x1x2xf32>)
func.func @FuseSoftMaxAffineReshapeConvert(%arg0: tensor<1x16800x1x2xf32>) -> tensor<1x16800x2xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x16800x1x2xf32> -> tensor<1x16800x1x2xf16>
    %1 = IE.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x16800x1x2xf16> -> tensor<1x16800x1x2xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 16800, 2]} : tensor<1x16800x1x2xf16> -> tensor<1x1x16800x2xf16>
    %3 = IE.Convert(%2) {dstElemType = f32} : tensor<1x1x16800x2xf16> -> tensor<1x1x16800x2xf32>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 16800, 2]} : tensor<1x1x16800x2xf32> -> tensor<1x16800x2xf32>

    return %4 : tensor<1x16800x2xf32>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x16800x1x2xf32> -> tensor<1x16800x1x2xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONVERT]]) {axisInd = 3 : i64, dstElemType = f32} : tensor<1x16800x1x2xf16> -> tensor<1x16800x1x2xf32>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[SOFTMAX]]) {dim_mapping = {{\[}}[0], [1], [1], [2]{{\]}}, shape_value = [1, 16800, 2]} : tensor<1x16800x1x2xf32> -> tensor<1x16800x2xf32>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x16800x2xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @FuseSoftMaxPermuteCastAffineReshapeConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x2x1x1xf32, {order = #NHWC}>)
func.func @FuseSoftMaxPermuteCastAffineReshapeConvert(%arg0: tensor<1x2x1x1xf32, {order = #NHWC}>) -> tensor<1x2xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x1x1xf32, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x2x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    %2 = IE.PermuteCast(%1) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x2x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 2]} : tensor<1x2x1x1xf16> -> tensor<1x1x1x2xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x1x2xf16> -> tensor<1x1x1x2xf32>
    %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 2]} : tensor<1x1x1x2xf32> -> tensor<1x2xf32>

    return %5 : tensor<1x2xf32>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x2x1x1xf32, {order = #NHWC}> -> tensor<1x2x1x1xf16, {order = #NHWC}>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONVERT]]) {axisInd = 1 : i64, dstElemType = f32} : tensor<1x2x1x1xf16, {order = #NHWC}> -> tensor<1x2x1x1xf32, {order = #NHWC}>
    // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[SOFTMAX]]) {dst_order = #NCHW, mem_perm = #NWHC} : tensor<1x2x1x1xf32, {order = #NHWC}> -> tensor<1x2x1x1xf32>
    // CHECK: [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[PERMUTECAST]]) {dim_mapping = {{\[}}[0], [1], [1], [1]{{\]}}, shape_value = [1, 2]} : tensor<1x2x1x1xf32> -> tensor<1x2xf32>
    // CHECK: return [[AFFINERESHAPE1]] : tensor<1x2xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @FuseSoftMaxSlicePermuteCastAffineReshapeConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1008x1x1xf32, {order = #NHWC}>)
func.func @FuseSoftMaxSlicePermuteCastAffineReshapeConvert(%arg0: tensor<1x1008x1x1xf32, {order = #NHWC}>) -> tensor<1x1000xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1008x1x1xf32, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64, padSize = 8 : i64} : tensor<1x1008x1x1xf16, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    %2 = IE.Slice %1 [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf16, {order = #NHWC}> to tensor<1x1000x1x1xf16, {order = #NHWC}>
    %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf16, {order = #NHWC}> -> tensor<1x1000x1x1xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000x1x1xf16> -> tensor<1x1x1x1000xf16>
    %5 = IE.Convert(%4) {dstElemType = f32} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf32>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf32> -> tensor<1x1000xf32>

    return %6 : tensor<1x1000xf32>

    // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1008x1x1xf32, {order = #NHWC}> -> tensor<1x1008x1x1xf16, {order = #NHWC}>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[CONVERT]]) {axisInd = 1 : i64, dstElemType = f32, padSize = 8 : i64} : tensor<1x1008x1x1xf16, {order = #NHWC}> -> tensor<1x1008x1x1xf32, {order = #NHWC}>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 1000, 1, 1] : tensor<1x1008x1x1xf32, {order = #NHWC}> to tensor<1x1000x1x1xf32, {order = #NHWC}>
    // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[SLICE]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1000x1x1xf32, {order = #NHWC}> -> tensor<1x1000x1x1xf32>
    // CHECK: [[AFFINERESHAPE:%.+]] = IE.AffineReshape([[PERMUTECAST]]) {dim_mapping = {{\[}}[0], [1], [1], [1]{{\]}}, shape_value = [1, 1000]} : tensor<1x1000x1x1xf32> -> tensor<1x1000xf32>
    // CHECK: return [[AFFINERESHAPE]] : tensor<1x1000xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @FuseSoftMaxSliceAffineReshapePermuteCastConvert
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x32640x250x4xf16, {order = #NHWC}>)
func.func @FuseSoftMaxSliceAffineReshapePermuteCastConvert(%arg0: tensor<1x32640x250x4xf16, {order = #NHWC}>) -> tensor<1000x1x32632xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64, padSize = 8 : i64} : tensor<1x32640x250x4xf16, {order = #NHWC}> -> tensor<1x32640x250x4xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 32632, 250, 4] : tensor<1x32640x250x4xf16, {order = #NHWC}> to tensor<1x32632x250x4xf16, {order = #NHWC}>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 32632, 1000, 1]} : tensor<1x32632x250x4xf16, {order = #NHWC}> -> tensor<1x32632x1000x1xf16, {order = #NHWC}>
    %3 = IE.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #map} : tensor<1x32632x1000x1xf16, {order = #NHWC}> -> tensor<1000x1x1x32632xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3]], shape_value = [1000, 1, 32632, 1]} : tensor<1000x1x1x32632xf16> -> tensor<1000x1x32632x1xf16>
    %5 = IE.Convert(%4) {dstElemType = f32} : tensor<1000x1x32632x1xf16> -> tensor<1000x1x32632x1xf32>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0], [1], [2], [2]], shape_value = [1000, 1, 32632]} : tensor<1000x1x32632x1xf32> -> tensor<1000x1x32632xf32>

    return %6 : tensor<1000x1x32632xf32>

    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 1 : i64, dstElemType = f32, padSize = 8 : i64} : tensor<1x32640x250x4xf16, {order = #NHWC}> -> tensor<1x32640x250x4xf32, {order = #NHWC}>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[SOFTMAX]] [0, 0, 0, 0] [1, 32632, 250, 4] : tensor<1x32640x250x4xf32, {order = #NHWC}> to tensor<1x32632x250x4xf32, {order = #NHWC}>
    // CHECK: [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[SLICE]]) {dim_mapping = {{\[}}[0], [1], [2], [2, 3]{{\]}}, shape_value = [1, 32632, 1000, 1]} : tensor<1x32632x250x4xf32, {order = #NHWC}> -> tensor<1x32632x1000x1xf32, {order = #NHWC}>
    // CHECK: [[PERMUTECAST:%.+]] = IE.PermuteCast([[AFFINERESHAPE1]]) {dst_order = #NCHW, mem_perm = #map} : tensor<1x32632x1000x1xf32, {order = #NHWC}> -> tensor<1000x1x1x32632xf32>
    // CHECK: [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[PERMUTECAST]]) {dim_mapping = {{\[}}[0], [1], [1], [2]{{\]}}, shape_value = [1000, 1, 32632]} : tensor<1000x1x1x32632xf32> -> tensor<1000x1x32632xf32>
    // CHECK: return [[AFFINERESHAPE2]] : tensor<1000x1x32632xf32>
}

// -----

// CHECK-LABEL: @DoNotFuseSoftMaxConvertMultipleUses
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x64x33xf32>)
func.func @DoNotFuseSoftMaxConvertMultipleUses(%arg0: tensor<1x1x64x33xf32>) -> (tensor<1x1x64x33xf32>, tensor<1x4x8x33xf16>) {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x64x33xf32> -> tensor<1x1x64x33xf16>
    %1 = IE.ShapeCast {shape = [1, 8, 8, 33]} inputs(%0 : tensor<1x1x64x33xf16>) -> tensor<1x8x8x33xf16>
    %2 = IE.SoftMax(%1) {axisInd = 3 : i64} : tensor<1x8x8x33xf16> -> tensor<1x8x8x33xf16>
    %3 = IE.ShapeCast {shape = [1, 1, 64, 33]} inputs(%2 : tensor<1x8x8x33xf16>) -> tensor<1x1x64x33xf16>
    %4 = IE.Convert(%3) {dstElemType = f32} : tensor<1x1x64x33xf16> -> tensor<1x1x64x33xf32>
    %5 = IE.Slice %3 [0, 0, 0, 0] [1, 1, 32, 33] : tensor<1x1x64x33xf16> to tensor<1x1x32x33xf16>
    %6 = IE.ShapeCast {shape = [1, 4, 8, 33]} inputs(%5 : tensor<1x1x32x33xf16>) -> tensor<1x4x8x33xf16>

    return %4, %6 : tensor<1x1x64x33xf32>, tensor<1x4x8x33xf16>

    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x64x33xf32> -> tensor<1x1x64x33xf16>
    // CHECK: [[SHAPECAST1:%.+]] = IE.ShapeCast {shape = [1, 8, 8, 33]} inputs([[CONVERT1]] : tensor<1x1x64x33xf16>) -> tensor<1x8x8x33xf16>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[SHAPECAST1]]) {axisInd = 3 : i64} : tensor<1x8x8x33xf16> -> tensor<1x8x8x33xf16>
    // CHECK: [[SHAPECAST2:%.+]] = IE.ShapeCast {shape = [1, 1, 64, 33]} inputs([[SOFTMAX]] : tensor<1x8x8x33xf16>) -> tensor<1x1x64x33xf16>
    // CHECK: [[CONVERT2:%.+]] = IE.Convert([[SHAPECAST2]]) {dstElemType = f32} : tensor<1x1x64x33xf16> -> tensor<1x1x64x33xf32>
    // CHECK: [[SLICE:%.+]] = IE.Slice [[SHAPECAST2]] [0, 0, 0, 0] [1, 1, 32, 33] : tensor<1x1x64x33xf16> to tensor<1x1x32x33xf16>
    // CHECK: [[SHAPECAST3:%.+]] = IE.ShapeCast {shape = [1, 4, 8, 33]} inputs([[SLICE]] : tensor<1x1x32x33xf16>) -> tensor<1x4x8x33xf16>
    // CHECK: return [[CONVERT2]], [[SHAPECAST3]] : tensor<1x1x64x33xf32>, tensor<1x4x8x33xf16>
}

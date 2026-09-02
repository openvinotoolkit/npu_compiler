//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-auto-padding-odu" --adjust-input-shape --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @ExpandAddToShapeCastAddWithTwoExpands
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x3x32x32xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x3x32x32xf16>
func.func @ExpandAddToShapeCastAddWithTwoExpands(%arg0: tensor<1x3x32x32xf16>, %arg1: tensor<1x3x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %1 = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, input_padding = [0, 13, 0, 0], output_padding = [0, 0, 0, 0]} : tensor<1x16x32x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x3x32x32xf16>
    %3 = IE.Expand(%2) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    return %3 : tensor<1x16x32x32xf16>

    // CHECK-NOT:   IE.Expand
    // CHECK-DAG:   [[CAST1:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT1]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>
    // CHECK-DAG:   [[CAST2:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 12]} inputs([[INPUT2]] : tensor<1x3x32x32xf16>) -> tensor<1x16x16x12xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CAST1]], [[CAST2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x16x12xf16>, tensor<1x16x16x12xf16> -> tensor<1x16x16x12xf16>
    // CHECK:       [[CAST_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 3, 32, 32]} inputs([[ADD]] : tensor<1x16x16x12xf16>) -> tensor<1x3x32x32xf16>
    // CHECK:       [[EXPAND_OUTPUT:%.+]] = IE.Expand([[CAST_OUTPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[EXPAND_OUTPUT]]
}

// -----

// CHECK-LABEL: @AdjustAvgPoolingToShapeCastAvgPooling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1x2048xf16>
func.func @AdjustAvgPoolingToShapeCastAvgPooling(%arg0: tensor<1x1x1x2048xf16>) -> tensor<1x16x1x2048xf16> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 0.10000000149011612 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1], input_padding = [0, 15, 0, 0], output_padding = [0, 0, 0, 0]} : tensor<1x16x1x2048xf16> -> tensor<1x1x1x2048xf16>
    %2 = IE.Expand(%1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    return %2 : tensor<1x16x1x2048xf16>

    // ExpandPoolingRewriter will be used instead of ExpandSingleChannelPoolingRewriter due to benefit level setting
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 16, 16, 8]} inputs([[INPUT]] : tensor<1x1x1x2048xf16>) -> tensor<1x16x16x8xf16>
    // CHECK:   [[POOLING:%.+]] = IE.AvgPool([[SHAPECAST0]])
    // CHECK:   [[SHAPECAST0:%.+]] = IE.ShapeCast {shape = [1, 1, 1, 2048]} inputs([[POOLING]] : tensor<1x16x16x8xf16>) -> tensor<1x1x1x2048xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[SHAPECAST0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 15, 0, 0]} : tensor<1x1x1x2048xf16> -> tensor<1x16x1x2048xf16>
    // CHECK:       return [[EXPAND]] : tensor<1x16x1x2048xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @DoNotPropagateShapeCastAfterPaddedEltwise
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x48x512x32xf16, {order = #NHWC}>
func.func @DoNotPropagateShapeCastAfterPaddedEltwise(%input: tensor<1x48x512x32xf16, {order = #NHWC}>)
            -> tensor<1x16x256x192xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%input, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
        : tensor<1x48x512x32xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}> -> tensor<1x48x512x32xf16, {order = #NHWC}>
    %shapecast = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs(%conv : tensor<1x48x512x32xf16, {order = #NHWC}>) -> tensor<1x16x256x192xf16, {order = #NHWC}>
    %eltwise = IE.Add(%shapecast, %shapecast) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, input_padding = [0, 13, 0, 0], output_padding = [0, 13, 0, 0]}
        : tensor<1x16x256x192xf16, {order = #NHWC}>, tensor<1x16x256x192xf16, {order = #NHWC}> -> tensor<1x16x256x192xf16, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x192xf16, {order = #NHWC}>

    // CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}>
    // CHECK:  [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK:  [[SHAPE_CAST:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 192]} inputs([[CONV]]
    // CHECK:  [[ADD:%.+]] = IE.Add([[SHAPE_CAST]], [[SHAPE_CAST]]
    // CHECK:  return [[ADD]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustBatchedMultiplyShape
// CHECK-SAME:        [[IN1:%[^:]+]]: tensor<8x16x1x1xf16>,
// CHECK-SAME:        [[IN2:%[^:]+]]: tensor<8x16x1x1xf16>
func.func @AdjustBatchedMultiplyShape(%arg0: tensor<8x16x1x1xf16>, %arg1: tensor<8x16x1x1xf16>) -> tensor<8x16x1x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<8x16x1x1xf16>, tensor<8x16x1x1xf16> -> tensor<8x16x1x1xf16>
    return %0 : tensor<8x16x1x1xf16>

    // CHECK:       [[PC1:%.+]] = IE.PermuteCast([[IN1]]) {dst_order = #NHWC, mem_perm = {{.*}}} : tensor<8x16x1x1xf16> -> tensor<1x16x8x1xf16, {order = #NHWC}>
    // CHECK:       [[AR1:%.+]] = IE.AffineReshape([[PC1]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 2, 4]}
    // CHECK-SAME:             tensor<1x16x8x1xf16, {order = #NHWC}> -> tensor<1x16x2x4xf16, {order = #NHWC}>
    // CHECK:       [[PC2:%.+]] = IE.PermuteCast([[IN2]]) {dst_order = #NHWC, mem_perm = {{.*}}} : tensor<8x16x1x1xf16> -> tensor<1x16x8x1xf16, {order = #NHWC}>
    // CHECK:       [[AR2:%.+]] = IE.AffineReshape([[PC2]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 2, 4]}
    // CHECK-SAME:             tensor<1x16x8x1xf16, {order = #NHWC}> -> tensor<1x16x2x4xf16, {order = #NHWC}>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[AR1]], [[AR2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x2x4xf16, {order = #NHWC}>, tensor<1x16x2x4xf16, {order = #NHWC}> -> tensor<1x16x2x4xf16, {order = #NHWC}>
    // CHECK:       [[AR_OUT:%.+]] = IE.AffineReshape([[MUL]])
    // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 16, 8, 1]}
    // CHECK-SAME:             tensor<1x16x2x4xf16, {order = #NHWC}> -> tensor<1x16x8x1xf16, {order = #NHWC}>
    // CHECK:       [[PC_OUT:%.+]] = IE.PermuteCast([[AR_OUT]]) {dst_order = #NCHW, mem_perm = {{.*}}} : tensor<1x16x8x1xf16, {order = #NHWC}> -> tensor<8x16x1x1xf16>
    // CHECK:       return [[PC_OUT]] : tensor<8x16x1x1xf16>
}

// -----

// CHECK-LABEL: @DoNotAdjustBatchedMultiplyShapeWhenBatchNotDivisibleBy4
// CHECK-SAME:        [[IN1:%[^:]+]]: tensor<6x16x1x1xf16>,
// CHECK-SAME:        [[IN2:%[^:]+]]: tensor<6x16x1x1xf16>
func.func @DoNotAdjustBatchedMultiplyShapeWhenBatchNotDivisibleBy4(
        %arg0: tensor<6x16x1x1xf16>, %arg1: tensor<6x16x1x1xf16>) -> tensor<6x16x1x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<6x16x1x1xf16>, tensor<6x16x1x1xf16> -> tensor<6x16x1x1xf16>
    return %0 : tensor<6x16x1x1xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK-NOT:   IE.AffineReshape
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<6x16x1x1xf16>, tensor<6x16x1x1xf16> -> tensor<6x16x1x1xf16>
    // CHECK:       return [[MUL]] : tensor<6x16x1x1xf16>
}

// -----

// CHECK-LABEL: @DoNotAdjustBatchedMultiplyShapeWhenBatchIsOne
// CHECK-SAME:        [[IN1:%[^:]+]]: tensor<1x16x1x1xf16>,
// CHECK-SAME:        [[IN2:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @DoNotAdjustBatchedMultiplyShapeWhenBatchIsOne(
        %arg0: tensor<1x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    return %0 : tensor<1x16x1x1xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK-NOT:   IE.AffineReshape
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       return [[MUL]] : tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @DoNotAdjustBatchedMultiplyShapeWhenSpatialDimsNotOne
// CHECK-SAME:        [[IN1:%[^:]+]]: tensor<4x16x2x1xf16>,
// CHECK-SAME:        [[IN2:%[^:]+]]: tensor<4x16x2x1xf16>
func.func @DoNotAdjustBatchedMultiplyShapeWhenSpatialDimsNotOne(
        %arg0: tensor<4x16x2x1xf16>, %arg1: tensor<4x16x2x1xf16>) -> tensor<4x16x2x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<4x16x2x1xf16>, tensor<4x16x2x1xf16> -> tensor<4x16x2x1xf16>
    return %0 : tensor<4x16x2x1xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK-NOT:   IE.AffineReshape
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<4x16x2x1xf16>, tensor<4x16x2x1xf16> -> tensor<4x16x2x1xf16>
    // CHECK:       return [[MUL]] : tensor<4x16x2x1xf16>
}

// -----

// CHECK-LABEL: @DoNotAdjustBatchedMultiplyShapeWithBroadcast
// CHECK-SAME:        [[IN1:%[^:]+]]: tensor<8x16x1x1xf16>,
// CHECK-SAME:        [[IN2:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @DoNotAdjustBatchedMultiplyShapeWithBroadcast(
        %arg0: tensor<8x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<8x16x1x1xf16> {
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<8x16x1x1xf16>
    return %0 : tensor<8x16x1x1xf16>

    // CHECK-NOT:   IE.PermuteCast
    // CHECK-NOT:   IE.AffineReshape
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[IN1]], [[IN2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<8x16x1x1xf16>
    // CHECK:       return [[MUL]] : tensor<8x16x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW_TO_NHWC_8_16 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

// CHECK-LABEL: @AdjustBatchedMultiplyShapeDerivesConsumerWThroughPermuteCast
// CHECK-SAME:        [[IN1:%[^:]+]]: tensor<8x16x1x1xf16>,
// CHECK-SAME:        [[IN2:%[^:]+]]: tensor<8x16x1x1xf16>
func.func @AdjustBatchedMultiplyShapeDerivesConsumerWThroughPermuteCast(
        %arg0: tensor<8x16x1x1xf16>, %arg1: tensor<8x16x1x1xf16>) -> tensor<1x16x1x8xf16, {order = #NHWC}> {
    %mul = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<8x16x1x1xf16>, tensor<8x16x1x1xf16> -> tensor<8x16x1x1xf16>

    %pc  = IE.PermuteCast(%mul) {dst_order = #NHWC, mem_perm = #NCHW_TO_NHWC_8_16} : tensor<8x16x1x1xf16> -> tensor<1x16x1x8xf16, {order = #NHWC}>
    %pool = IE.MaxPool(%pc) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
                             rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x16x1x8xf16, {order = #NHWC}> -> tensor<1x16x1x8xf16, {order = #NHWC}>
    return %pool : tensor<1x16x1x8xf16, {order = #NHWC}>

    // CHECK:       [[PC1:%.+]] = IE.PermuteCast([[IN1]])
    // CHECK:       [[AR1:%.+]] = IE.AffineReshape([[PC1]])
    // CHECK-SAME:       shape_value = [1, 16, 1, 8]
    // CHECK:       [[PC2:%.+]] = IE.PermuteCast([[IN2]])
    // CHECK:       [[AR2:%.+]] = IE.AffineReshape([[PC2]])
    // CHECK-SAME:       shape_value = [1, 16, 1, 8]
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[AR1]], [[AR2]])
    // CHECK-SAME:       tensor<1x16x1x8xf16, {order = #NHWC}>, tensor<1x16x1x8xf16, {order = #NHWC}> -> tensor<1x16x1x8xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool({{.*}})
    // CHECK-SAME:       tensor<1x16x1x8xf16, {order = #NHWC}> -> tensor<1x16x1x8xf16, {order = #NHWC}>
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x1x8xf16, {order = #NHWC}>
}

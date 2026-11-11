//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-tile-with-eltwise-to-nce-pool %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileMultiplyToMaxPool
// CHECK-SAME:      ([[SCALE_PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileMultiplyToMaxPool(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %scale = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
    } -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %tile = VPU.Tile(%scale) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %mul = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<MULTIPLY>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %mul : tensor<1x16x8x8xf16, {order = #NHWC}>

    // The Tile -> NCE.Eltwise<MULTIPLY> pair is folded into a single NCE.MaxPool that consumes the
    // broadcast operand as a per-channel scale table. On NPU60+, the broadcast AvgPool producer can
    // output f32 directly before NCHW materialization and the per-channel reshape.
    // CHECK:       [[SCALE_TABLE:%.+]] = VPU.NCE.AveragePool([[SCALE_PRODUCER]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK-NOT:   VPU.Convert([[SCALE_TABLE]])
    // CHECK:       [[SCALE_NCHW:%.+]] = VPU.PermuteCast([[SCALE_TABLE]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[SCALE_RESHAPE:%.+]] = VPU.AffineReshape([[SCALE_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK-SAME:      tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
    // CHECK:       [[ZERO_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[SCALE_RESHAPE]], [[ZERO_BIAS]])
    // CHECK-SAME:      -> tensor<1x16x8x8xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileAddToMaxPool
// CHECK-SAME:      ([[BIAS_PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileAddToMaxPool(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %bias = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
    } -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %tile = VPU.Tile(%bias) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %add = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // The Tile -> NCE.Eltwise<ADD> pair is folded into a single NCE.MaxPool that consumes the
    // broadcast operand as a per-channel bias table. On NPU60+, the broadcast AvgPool producer can
    // output f32 directly before NCHW materialization and the per-channel reshape.
    // CHECK:       [[BIAS_TABLE:%.+]] = VPU.NCE.AveragePool([[BIAS_PRODUCER]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK-NOT:   VPU.Convert([[BIAS_TABLE]])
    // CHECK:       [[BIAS_NCHW:%.+]] = VPU.PermuteCast([[BIAS_TABLE]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[BIAS_RESHAPE:%.+]] = VPU.AffineReshape([[BIAS_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK-SAME:      tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
    // CHECK:       [[UNIT_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[UNIT_SCALE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:      -> tensor<1x16x8x8xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileAddArgToMaxPoolViaMaxPool
// CHECK-SAME:      ([[BIAS_PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileAddArgToMaxPoolViaMaxPool(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %add = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [2.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // Runtime f16 bias table input is materialized as f32 through a 1x1 NCE.MaxPool instead
    // of VPU.Convert, matching the faster DPU conversion path used on NPU60+.
    // CHECK:       [[BIAS_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[BIAS_PRODUCER]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK-NOT:   VPU.Convert([[BIAS_PRODUCER]])
    // CHECK:       [[BIAS_NCHW:%.+]] = VPU.PermuteCast([[BIAS_MAXPOOL]])
    // CHECK:       [[BIAS_RESHAPE:%.+]] = VPU.AffineReshape([[BIAS_NCHW]])
    // CHECK:       [[UNIT_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[UNIT_SCALE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELU>
    // CHECK-SAME:      prelu_alpha = [2.000000e-01]
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileMultiplyArgToMaxPoolViaMaxPool
// CHECK-SAME:      ([[SCALE_PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileMultiplyArgToMaxPoolViaMaxPool(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %mul = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<MULTIPLY>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [2.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %mul : tensor<1x16x8x8xf16, {order = #NHWC}>

    // Runtime f16 scale table input is materialized as f32 through a 1x1 NCE.MaxPool instead
    // of VPU.Convert, matching the faster DPU conversion path used on NPU60+.
    // CHECK:       [[SCALE_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SCALE_PRODUCER]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK-NOT:   VPU.Convert([[SCALE_PRODUCER]])
    // CHECK:       [[SCALE_NCHW:%.+]] = VPU.PermuteCast([[SCALE_MAXPOOL]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[SCALE_RESHAPE:%.+]] = VPU.AffineReshape([[SCALE_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK:       [[ZERO_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[SCALE_RESHAPE]], [[ZERO_BIAS]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELU>
    // CHECK-SAME:      prelu_alpha = [2.000000e-01]
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Eltwise type SUBTRACT is not supported by the scale/bias-table rewrite: keep Tile + Eltwise.
// CHECK-LABEL: @NotConvertUnsupportedEltwiseType
// CHECK-SAME:      ([[PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @NotConvertUnsupportedEltwiseType(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %sub = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<SUBTRACT>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %sub : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK:       [[TILE:%.+]] = VPU.Tile([[PRODUCER]]) {repeats_values = [1, 1, 8, 8]}
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[TILE]], [[INPUT]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<SUBTRACT>
    // CHECK:       return [[ELTWISE]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Eltwise with reduce_xy_min has an extra result that is not preserved by the MaxPool rewrite.
// CHECK-LABEL: @NotConvertEltwiseWithReduceXyMin
// CHECK-SAME:      ([[PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @NotConvertEltwiseWithReduceXyMin(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>) {
        %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
        %add, %reduce_min = VPU.NCE.Eltwise(%tile, %arg1) {
                axes_value = [1],
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                resultSegmentSizes = array<i32: 1, 0, 1, 0>
        } : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>
            -> tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>
        return %add, %reduce_min : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x1x8x8xf16, {order = #NHWC}>

        // CHECK:       [[TILE:%.+]] = VPU.Tile([[PRODUCER]]) {repeats_values = [1, 1, 8, 8]}
        // CHECK:       [[ELTWISE:%.+]], [[REDUCE_MIN:%.+]] = VPU.NCE.Eltwise([[TILE]], [[INPUT]])
        // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
        // CHECK:       return [[ELTWISE]], [[REDUCE_MIN]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Tile input is not flattened (H != 1, broadcast over W only): cannot map to a per-channel
// scale/bias table, so keep Tile + Eltwise.
// CHECK-LABEL: @NotConvertNonFlattenedTile
// CHECK-SAME:      ([[PRODUCER:%[^:]+]]: tensor<1x16x2x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x2x8xf16, {order = #NHWC}>)
func.func @NotConvertNonFlattenedTile(%arg0: tensor<1x16x2x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x2x8xf16, {order = #NHWC}>) -> tensor<1x16x2x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 1, 8]} : tensor<1x16x2x1xf16, {order = #NHWC}> -> tensor<1x16x2x8xf16, {order = #NHWC}>
    %add = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x2x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x2x8xf16, {order = #NHWC}>

    // CHECK:       [[TILE:%.+]] = VPU.Tile([[PRODUCER]]) {repeats_values = [1, 1, 1, 8]}
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[TILE]], [[INPUT]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK:       return [[ELTWISE]] : tensor<1x16x2x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Tile result has more than one user: the broadcast copy is still needed, so keep Tile + Eltwise.
// CHECK-LABEL: @NotConvertTileWithMultipleUsers
// CHECK-SAME:      ([[PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @NotConvertTileWithMultipleUsers(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> (tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>) {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %add = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %tile, %add : tensor<1x16x8x8xf16, {order = #NHWC}>, tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK:       [[TILE:%.+]] = VPU.Tile([[PRODUCER]]) {repeats_values = [1, 1, 8, 8]}
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[TILE]], [[INPUT]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK:       return [[TILE]], [[ELTWISE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Existing NCEEltwise weight-table operands already encode scale/bias semantics. The Tile rewrite
// must not drop or try to merge them.
// CHECK-LABEL: @NotConvertEltwiseWithExistingWeightTables
// CHECK-SAME:      ([[PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @NotConvertEltwiseWithExistingWeightTables(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %scale = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    %bias = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    %add = VPU.NCE.Eltwise(%tile, %arg1, %scale, %bias) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK:       [[TILE:%.+]] = VPU.Tile([[PRODUCER]]) {repeats_values = [1, 1, 8, 8]}
    // CHECK:       [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[TILE]], [[INPUT]], [[SCALE]], [[BIAS]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK:       return [[ELTWISE]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Dynamic Tile input shape is not supported by this static scale/bias-table rewrite.
// CHECK-LABEL: @NotConvertDynamicTileInput
// CHECK-SAME:      ([[PRODUCER:%[^:]+]]: tensor<1x?x1x1xf16
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x?x1x1xf16
func.func @NotConvertDynamicTileInput(%arg0: tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 1, 1]} : tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    %mul = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<MULTIPLY>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>
    return %mul : tensor<1x?x1x1xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1, 1]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[TILE:%.+]] = VPU.Tile([[PRODUCER]]) {repeats_values = [1, 1, 1, 1]}
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[TILE]], [[INPUT]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK:       return [[ELTWISE]] : tensor<1x?x1x1xf16
}

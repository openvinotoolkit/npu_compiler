//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-tile-with-eltwise-to-nce-pool %s | FileCheck %s
// REQUIRES: platform-NPU5010

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
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %mul : tensor<1x16x8x8xf16, {order = #NHWC}>

    // The Tile -> NCE.Eltwise<MULTIPLY> pair is folded into a single NCE.MaxPool that consumes a
    // legacy [OC, 1, 1, 4] xsi32 weights table: the broadcast operand's AvgPool producer is reused
    // in place to output FP32 directly, reshaped to [C, 1, 1, 1], and packed together with a zero
    // sparsity pointer and a dummy unit/zero bias by Concat + Core.ReinterpretCast to si32.
    // CHECK:       [[SCALE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[SCALE_PRODUCER]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[SCALE_NCHW:%.+]] = VPU.PermuteCast([[SCALE_AVGPOOL]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[SCALE_RESHAPE:%.+]] = VPU.AffineReshape([[SCALE_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK-SAME:      tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
    // CHECK:       [[ZERO_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[SCALE_RESHAPE]], [[ZERO_BIAS]])
    // CHECK-SAME:      -> tensor<16x1x1x4xf32>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32>
    // The original NCE.Eltwise's mpe_engine is forwarded to the folded MaxPool.
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
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
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK:       [[BIAS_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[BIAS_PRODUCER]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_NCHW:%.+]] = VPU.PermuteCast([[BIAS_AVGPOOL]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[BIAS_RESHAPE:%.+]] = VPU.AffineReshape([[BIAS_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK-SAME:      tensor<1x16x1x1xf32> -> tensor<16x1x1x1xf32>
    // CHECK:       [[UNIT_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[UNIT_SCALE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:      -> tensor<16x1x1x4xf32>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32>
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME:      -> tensor<1x16x8x8xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileNonIdentityAvgPoolToMaxPool
// CHECK-SAME:      ([[SCALE_PRODUCER:%[^:]+]]: tensor<1x16x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileNonIdentityAvgPoolToMaxPool(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %scale = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.111111e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
    } -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %tile = VPU.Tile(%scale) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %mul = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<MULTIPLY>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %mul : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK:       [[SCALE_AVGPOOL:%.+]] = VPU.NCE.AveragePool([[SCALE_PRODUCER]])
    // CHECK-SAME:      kernel_size = [3, 3]
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[SCALE_NCHW:%.+]] = VPU.PermuteCast([[SCALE_AVGPOOL]])
    // CHECK:       [[SCALE_RESHAPE:%.+]] = VPU.AffineReshape([[SCALE_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK:       [[ZERO_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[SCALE_RESHAPE]], [[ZERO_BIAS]])
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
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
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [2.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // Runtime f16 bias operand is materialized as f32 through a 1x1 identity NCE.MaxPool instead
    // of VPU.Convert, matching the faster DPU conversion path also used for the AvgPool-producer case.
    // CHECK:       [[BIAS_CONVERT:%.+]] = VPU.NCE.MaxPool([[BIAS_PRODUCER]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_NCHW:%.+]] = VPU.PermuteCast([[BIAS_CONVERT]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[BIAS_RESHAPE:%.+]] = VPU.AffineReshape([[BIAS_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK:       [[UNIT_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[UNIT_SCALE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:      -> tensor<16x1x1x4xf32>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32>
    // The original NCE.Eltwise LRELU mode/prelu_alpha and mpe_engine are preserved on the folded MaxPool.
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
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
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [2.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %mul : tensor<1x16x8x8xf16, {order = #NHWC}>

    // Runtime f16 scale operand is materialized as f32 through a 1x1 identity NCE.MaxPool instead
    // of VPU.Convert, matching the faster DPU conversion path also used for the AvgPool-producer case.
    // CHECK:       [[SCALE_CONVERT:%.+]] = VPU.NCE.MaxPool([[SCALE_PRODUCER]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[SCALE_NCHW:%.+]] = VPU.PermuteCast([[SCALE_CONVERT]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[SCALE_RESHAPE:%.+]] = VPU.AffineReshape([[SCALE_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK:       [[ZERO_BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[SCALE_RESHAPE]], [[ZERO_BIAS]])
    // CHECK-SAME:      -> tensor<16x1x1x4xf32>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32>
    // The original NCE.Eltwise LRELU mode/prelu_alpha and mpe_engine are preserved on the folded MaxPool.
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELU>
    // CHECK-SAME:      prelu_alpha = [2.000000e-01]
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileAsSecondEltwiseOperandToMaxPool
// CHECK-SAME:      ([[BIAS_PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileAsSecondEltwiseOperandToMaxPool(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %add = VPU.NCE.Eltwise(%arg1, %tile) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        op_type = #VPU.eltwise_type<ADD>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // Tile output is the Eltwise's second operand; the non-Tile operand is still identified
    // correctly as the pool input.
    // CHECK:       [[BIAS_CONVERT:%.+]] = VPU.NCE.MaxPool([[BIAS_PRODUCER]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_NCHW:%.+]] = VPU.PermuteCast([[BIAS_CONVERT]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[BIAS_RESHAPE:%.+]] = VPU.AffineReshape([[BIAS_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK:       [[UNIT_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[UNIT_SCALE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:      -> tensor<16x1x1x4xf32>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32>
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME:      -> tensor<1x16x8x8xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Tile
    // CHECK-NOT:   VPU.NCE.Eltwise
    // CHECK:       return [[MAXPOOL]] : tensor<1x16x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertTileAddToMaxPoolWithPadding
// CHECK-SAME:      ([[BIAS_PRODUCER:%[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:       [[INPUT:%[^:]+]]: tensor<1x16x8x8xf16, {order = #NHWC}>)
func.func @ConvertTileAddToMaxPoolWithPadding(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x16x8x8xf16, {order = #NHWC}>) -> tensor<1x16x8x8xf16, {order = #NHWC}> {
    %tile = VPU.Tile(%arg0) {repeats_values = [1, 1, 8, 8]} : tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %add = VPU.NCE.Eltwise(%tile, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        input_padding = [0, 4, 0, 0],
        output_padding = [0, 4, 0, 0],
        op_type = #VPU.eltwise_type<ADD>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x16x8x8xf16, {order = #NHWC}>
    return %add : tensor<1x16x8x8xf16, {order = #NHWC}>

    // CHECK:       [[BIAS_CONVERT:%.+]] = VPU.NCE.MaxPool([[BIAS_PRODUCER]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      -> tensor<1x16x1x1xf32, {order = #NHWC}>
    // CHECK:       [[BIAS_NCHW:%.+]] = VPU.PermuteCast([[BIAS_CONVERT]])
    // CHECK-SAME:      -> tensor<1x16x1x1xf32>
    // CHECK:       [[BIAS_RESHAPE:%.+]] = VPU.AffineReshape([[BIAS_NCHW]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 1]
    // CHECK:       [[UNIT_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    // CHECK:       [[SPARSITY_PTR:%.+]] = const.Declare tensor<16x1x1x2xf32> = dense<{{.*2.35098856E-38.*}}> : tensor<16x1x1x2xf32>
    // CHECK:       [[WT_CONCAT:%.+]] = VPU.Concat([[SPARSITY_PTR]], [[UNIT_SCALE]], [[BIAS_RESHAPE]])
    // CHECK-SAME:      -> tensor<16x1x1x4xf32>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = Core.ReinterpretCast([[WT_CONCAT]])
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32>
    // The original NCE.Eltwise's input_padding/output_padding/mpe_engine attributes are forwarded
    // unchanged to the folded MaxPool.
    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]], [[WEIGHTS_TABLE]]
    // CHECK-SAME:      input_padding = [0, 4, 0, 0]
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME:      output_padding = [0, 4, 0, 0]
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

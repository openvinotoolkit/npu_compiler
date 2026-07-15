//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp32to128
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClamp32to128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            clamp = {min = 3.200000e+01 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 32 : i64,
    // CHECK-SAME:          clamp_high = 128 : i64,
    // CHECK-SAME:          lrelu_mult = 1638 : i64,
    // CHECK-SAME:          lrelu_shift = 14 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 0.10000000149011612 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FloatClamp0to120
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @FloatClamp0to120(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            clamp = {min = 0.000000e+00 : f64, max = 1.200000e+02 : f64}
        } : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x256x56x56xf16, {order = #NHWC}>
            -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = -2147483648 : i64,
    // CHECK-SAME:          clamp_high = 22400 : i64,
    // CHECK-SAME:          lrelu_mult = 1638 : i64,
    // CHECK-SAME:          lrelu_shift = 14 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 0.10000000149011612 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:255>:f16, 0.5:127>

// CHECK-LABEL: @QuantClampNeg6to6
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClampNeg6to6(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            clamp = {min = -6.000000e+00 : f64, max = 6.000000e+00 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 115 : i64,
    // CHECK-SAME:          clamp_high = 139 : i64,
    // CHECK-SAME:          lrelu_mult = 1638 : i64,
    // CHECK-SAME:          lrelu_shift = 14 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 0.10000000149011612 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @QuantClamp0To128
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @QuantClamp0To128(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                       -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            clamp = {min = 0.000000e+00 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0 : i64,
    // CHECK-SAME:          clamp_high = 128 : i64,
    // CHECK-SAME:          lrelu_mult = 1638 : i64,
    // CHECK-SAME:          lrelu_shift = 14 : i64,
    // CHECK-SAME:          fp_prelu_alpha = 0.10000000149011612 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionFloatClamp0to192
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionFloatClamp0to192(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 1.920000e+02 : f64},
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) rawFilterShape [64, 16, 3, 3] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEInt<
    // CHECK-SAME:            mode = <LRELUX>,
    // CHECK-SAME:            clamp_low = -2147483648 : i64,
    // CHECK-SAME:            clamp_high = 23040 : i64,
    // CHECK-SAME:            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:        >,
    // CHECK-SAME:        strides = [2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> tensor<1x64x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionLReluFloatClamp0to192
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionLReluFloatClamp0to192(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 1.920000e+02 : f64},
        post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) rawFilterShape [64, 16, 3, 3] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEInt<
    // CHECK-SAME:            mode = <LRELUX>,
    // CHECK-SAME:            clamp_low = -2147483648 : i64,
    // CHECK-SAME:            clamp_high = 23040 : i64,
    // CHECK-SAME:            lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64, fp_prelu_alpha = 0.10000000149011612 : f64
    // CHECK-SAME:        >,
    // CHECK-SAME:        strides = [2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> tensor<1x64x64x64xf16, {order = #NHWC}>
}

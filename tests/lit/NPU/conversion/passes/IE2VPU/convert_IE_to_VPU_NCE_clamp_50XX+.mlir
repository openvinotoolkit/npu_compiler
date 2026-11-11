//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Q_32To128_LRelu
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_32To128_LRelu(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
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
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e-01],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @F16_0To120_Tanh
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56xf16, {order = #NHWC}>)
func.func @F16_0To120_Tanh(%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                       %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                       -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.Tanh<>,
            clamp = {min = 0.000000e+00 : f64, max = 1.200000e+02 : f64}
        } : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<1x256x56x56xf16, {order = #NHWC}>
            -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <TANH>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.200000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:255>:f16, 0.5:127>

// CHECK-LABEL: @Q_Neg6To6_Exp
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_Neg6To6_Exp(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            post_op = #IE.Exp<>,
            clamp = {min = -6.000000e+00 : f64, max = 6.000000e+00 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <EXP>,
    // CHECK-SAME:          clamp_low = -1.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.200000e+01 : f64,
    // CHECK-SAME:          prelu_alpha = [2.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 1.270000e+02 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Q_0To128_LRelu
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_0To128_LRelu(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
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
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e-01],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @Q_32To128_NoOp
// CHECK-SAME:  ([[INPUT_0:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
// CHECK-SAME:   [[INPUT_1:%[^:]+]]: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
func.func @Q_32To128_NoOp(%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                        %arg1: tensor<1x256x56x56x!qElemType, {order = #NHWC}>)
                        -> tensor<1x256x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Add(%arg0, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            clamp = {min = 3.200000e+01 : f64, max = 1.280000e+02 : f64}
        } : tensor<1x256x56x56x!qElemType, {order = #NHWC}>, tensor<1x256x56x56x!qElemType, {order = #NHWC}>
            -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.280000e+02 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:  } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To175_Sigmoid
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To175_Sigmoid(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 1.750000e+02 : f64},
        post_op = #IE.Sigmoid<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <SIGMOID>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.750000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_32To192
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_32To192(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 3.200000e+01 : f64, max = 1.920000e+02 : f64},
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = 3.200000e+01 : f64,
    // CHECK-SAME:          clamp_high = 1.920000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To175_Swish
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To175_Swish(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 1.750000e+02 : f64},
        post_op = #IE.Swish<beta=1.5>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <SWISH>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 1.750000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To64_Gelu
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To64_Gelu(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 6.400000e+01 : f64},
        post_op = #IE.Gelu<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <GELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.400000e+01 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionF16_0To64_HSwish
// CHECK-SAME:  ([[INPUT:%[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>)
func.func @ConvolutionF16_0To64_HSwish(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        clamp = {min = 0.000000e+00 : f64, max = 6.400000e+01 : f64},
        post_op = #IE.HSwish<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<
    // CHECK-SAME:          mode = <HSWISH>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.400000e+01 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense<[[SPRLUT_DATA:".+"]]> : tensor<[[SPLRLUT_SIZE:.+]]xui16>
    // CHECK-SAME:      >
    // CHECK-SAME:      strides = [2, 2]
}

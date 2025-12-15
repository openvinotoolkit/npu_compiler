//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu=true enable-is-reduce-supported=true" --mlir-elide-elementsattrs-if-larger 64 --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @ClampLowInF16PReLU
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @ClampLowInF16PReLU(%input: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x3x3xf16, {order = #NHWC}>
    %CONV = IE.Convolution(%input, %WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = -2.312500e+00 : f64>,
        strides = [2, 2]
    } : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x3x3xf16, {order = #NHWC}> -> tensor<1x64x64x64xf16, {order = #NHWC}>

    return %CONV : tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:    [[BIAS:%.+]] = const.Declare tensor<64x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:            clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:            scale = 1.000000e+00 : f64, prelu_alpha = [-2.312500e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:        rawFilterShape = [64, 16, 3, 3], strides = [2, 2]
    // CHECK-SAME:    }
    // CHECK-SAME:    -> tensor<1x64x64x64xf16, {order = #NHWC}>

    // CHECK: return [[CONV]] : tensor<1x64x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 0.078737745098039214>

// CHECK-LABEL: @BiasFuncForI8Weights
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @BiasFuncForI8Weights(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:          0, 0, 1033978177, 1095447755
    // CHECK-SAME:          16, 0, 1033978177, 1095447755
    // CHECK-SAME:          32, 0, 1033978177, 1095447755
    // CHECK-SAME:          48, 0, 1033978177, 1095447755
    // CHECK-SAME:          64, 0, 1033978177, 1095447755
    // CHECK-SAME:          80, 0, 1033978177, 1095447755
    // CHECK-SAME:          96, 0, 1033978177, 1095447755
    // CHECK-SAME:          112, 0, 1033978177, 1095447755
    // CHECK-SAME:          128, 0, 1033978177, 1095447755
    // CHECK-SAME:          144, 0, 1033978177, 1095447755
    // CHECK-SAME:          160, 0, 1033978177, 1095447755
    // CHECK-SAME:          176, 0, 1033978177, 1095447755
    // CHECK-SAME:          192, 0, 1033978177, 1095447755
    // CHECK-SAME:          208, 0, 1033978177, 1095447755
    // CHECK-SAME:          224, 0, 1033978177, 1095447755
    // CHECK-SAME:          240, 0, 1033978177, 1095447755
    // CHECK-SAME:          : tensor<16x1x1x4xsi32>


    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 0.078737745098039214 : f64, prelu_alpha = [1.000000e-01], bias = 12.700389105058367 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @I4WeightsConvToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @I4WeightsConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %weights) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x32x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
    // CHECK-SAME:      tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 16]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 16]>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> =
    // CHECK-SAME{LITERAL}:    dense<[[[[0, 0, 1068193109, 0]]], [[[16, 0, 1068193109, 0]]], [[[32, 0, 1068193109, 0]]], [[[48, 0, 1068193109, 0]]], [[[64, 0, 1068193109, 0]]], [[[80, 0, 1068193109, 0]]], [[[96, 0, 1068193109, 0]]], [[[112, 0, 1068193109, 0]]], [[[128, 0, 1068193109, 0]]], [[[144, 0, 1068193109, 0]]], [[[160, 0, 1068193109, 0]]], [[[176, 0, 1068193109, 0]]], [[[192, 0, 1068193109, 0]]], [[[208, 0, 1068193109, 0]]], [[[224, 0, 1068193109, 0]]], [[[240, 0, 1068193109, 0]]]]> : tensor<16x1x1x4xsi32>
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.3385416666666667 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1], strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>,
// CHECK-SAME:  [[ARG1:%.+]]: tensor<16x16x1x1xf16, {order = #NHWC}>)
func.func @ConvWithStaticScale(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
                               %arg1: tensor<16x16x1x1xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        // Note: 5.24156e-40 is '374050' when bit-cast to int
        static_scale = 5.24156e-40 : f32
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<[
    // CHECK-SAME{LITERAL}: [[[0, 0, 374050, 0]]], [[[32, 0, 374050, 0]]],
    // CHECK-SAME{LITERAL}: [[[448, 0, 374050, 0]]], [[[480, 0, 374050, 0]]]
    // CHECK-SAME:  ]> : tensor<16x1x1x4xsi32>

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[WEIGHTS]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 5.2415569058069783E-40 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:      strides = [1, 1]}

    // CHECK:       return [[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCE(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE4channelsIDUAutopad
module @ConvToNCE4channelsIDUAutopad {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

        %0 = IE.Convolution(%arg0, %weights, %bias) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1],
                post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
                input_padding = [0, 12, 0, 0],
                output_padding = [0, 12, 0, 0]
            } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
                -> tensor<1x16x16x16xf16, {order = #NHWC}>

        return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:      input_padding = [0, 12, 0, 0],
        // CHECK-SAME:      output_padding = [0, 12, 0, 0],
        // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
        // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
        // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
        // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
        // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1]
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCEWithBiasBroadCast
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCEWithBiasBroadCast(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithReluRewriter
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvWithReluRewriter(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.Relu<>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELU>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [-0.000000e+00], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvWithSprLUTTanh
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvWithSprLUTTanh(%input: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %output = IE.Convolution(%input, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.Tanh<>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %output : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:   [[MAP:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[MAP]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <TANH>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 1.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense_resource<__elided__>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, -0.078737745098039214>
// CHECK: !qElemType = !quant.uniform<i8:f16, -0.078737745098039214>

// CHECK-LABEL: @QuantConvWithNegativeScales
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @QuantConvWithNegativeScales(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1x!qElemType, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:          0, 0, -1113505471, -1052035893
    // CHECK-SAME:          16, 0, -1113505471, -1052035893
    // CHECK-SAME:          32, 0, -1113505471, -1052035893
    // CHECK-SAME:          48, 0, -1113505471, -1052035893
    // CHECK-SAME:          64, 0, -1113505471, -1052035893
    // CHECK-SAME:          80, 0, -1113505471, -1052035893
    // CHECK-SAME:          96, 0, -1113505471, -1052035893
    // CHECK-SAME:          112, 0, -1113505471, -1052035893
    // CHECK-SAME:          128, 0, -1113505471, -1052035893
    // CHECK-SAME:          144, 0, -1113505471, -1052035893
    // CHECK-SAME:          160, 0, -1113505471, -1052035893
    // CHECK-SAME:          176, 0, -1113505471, -1052035893
    // CHECK-SAME:          192, 0, -1113505471, -1052035893
    // CHECK-SAME:          208, 0, -1113505471, -1052035893
    // CHECK-SAME:          224, 0, -1113505471, -1052035893
    // CHECK-SAME:          240, 0, -1113505471, -1052035893
    // CHECK-SAME:          : tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = -0.078737745098039214 : f64, prelu_alpha = [1.000000e-01], bias = -12.700389105058367 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>

// CHECK: func.func @LowerMatMulToNCE([[INPUT:%.+]]: tensor<1x128x64x32xf16>)
func.func @LowerMatMulToNCE(%input : tensor<1x128x64x32xf16>) -> tensor<1x128x64x64xf16> {
    %matmul = IE.MatMul(%input, %input) { transpose_b } : tensor<1x128x64x32xf16>, tensor<1x128x64x32xf16> -> tensor<1x128x64x64xf16>
    return %matmul : tensor<1x128x64x64xf16>

    // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:      tensor<1x128x64x32xf16> -> tensor<128x64x32x1x1xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_0]])
    // CHECK-SAME:      tensor<128x64x32x1x1xf16> -> tensor<128x1x32x64x1xf16, {order = #GNHWC}>

    // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME:      tensor<1x128x64x32xf16> -> tensor<128x64x32x1x1xf16>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[AFFINE_RESHAPE_1]])
    // CHECK-SAME:      tensor<128x64x32x1x1xf16> -> tensor<128x64x32x1x1xf16, {order = #GNHWC}>


    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x64x1x1x4xsi32>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) {

    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64
    // CHECK-SAME:      clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:      scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } -> tensor<128x1x64x64x1xf16, {order = #GNHWC}>


    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[MATMUL]])
    // CHECK-SAME:      tensor<128x1x64x64x1xf16, {order = #GNHWC}> -> tensor<128x64x64x1x1xf16>

    // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[MEMPERM]])
    // CHECK-SAME:      tensor<128x64x64x1x1xf16> -> tensor<1x128x64x64xf16>


    // CHECK:       return [[AFFINE_RESHAPE_2]] : tensor<1x128x64x64xf16>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<u8:f16, 0.021319432352103439:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0033220431383918312>
!qElemType2 = !quant.uniform<u8:f16, 0.029685012966978782:128>

// CHECK: func.func @LowerQuantMatMulToNCE()
func.func @LowerQuantMatMulToNCE() -> tensor<1x12x576x32x!quant.uniform<u8:f16, 0.021319432352103439:128>> {
    %INP1 = const.Declare tensor<1x12x576x144x!quant.uniform<u8:f16, 0.0033220431383918312>> = dense<1.0> : tensor<1x12x576x144xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.0033220431383918312>> ]
    %INP2 = const.Declare tensor<1x12x32x144x!quant.uniform<u8:f16, 0.029685012966978782:128>> = dense<1.0> : tensor<1x12x32x144xf16>, [ #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8:f16, 0.029685012966978782:128>> ]
    %MATMUL = IE.MatMul(%INP1, %INP2) {transpose_b} : tensor<1x12x576x144x!quant.uniform<u8:f16, 0.0033220431383918312>>,
     tensor<1x12x32x144x!quant.uniform<u8:f16, 0.029685012966978782:128>> -> tensor<1x12x576x32x!quant.uniform<u8:f16, 0.021319432352103439:128>>
    return %MATMUL : tensor<1x12x576x32x!quant.uniform<u8:f16, 0.021319432352103439:128>>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<12x1x144x576x1x!qElemType1, {order = #GNHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<12x32x144x1x1x!qElemType2, {order = #GNHWC}>


    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<12x32x1x1x4xsi32>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>
    // CHECK-SAME:      clamp_low = -1.280000e+02 : f64
    // CHECK-SAME:      clamp_high = 1.270000e+02 : f64
    // CHECK-SAME:      scale = 0.0046255872112980888 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } -> tensor<12x1x32x576x1x!qElemType, {order = #GNHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
!qElemType = !quant.uniform<f8E4M3FN:f16, 0.0045223289302417213>

// CHECK: func.func @LowerMixedPrecisionMatMulToNCE
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x1x64xf16>)
func.func @LowerMixedPrecisionMatMulToNCE(%arg0: tensor<1x8x1x64xf16>) -> tensor<1x8x1x128xf16> {
    %cst = const.Declare tensor<1x8x128x64x!qElemType> = dense<1.000000e+00> : tensor<1x8x128x64xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %quantize = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType>
    %matmul = IE.MatMul(%quantize, %cst) {transpose_b} : tensor<1x8x1x64x!qElemType>, tensor<1x8x128x64x!qElemType> -> tensor<1x8x1x128xf16>
    return %matmul : tensor<1x8x1x128xf16>

    // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast(
    // CHECK-SAME:     -> tensor<8x1x64x1x1x!qElemType, {order = #GNHWC}>

    // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast(
    // CHECK-SAME:      -> tensor<8x128x64x1x1x!qElemType, {order = #GNHWC}>


    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<8x128x1x1x4xsi32>

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:      scale = 2.0451458953301231E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:       rawFilterShape = [8, 128, 64, 1, 1], strides = [1, 1]}
    // CHECK-SAME:     -> tensor<8x1x128x1x1xf16, {order = #GNHWC}>
}

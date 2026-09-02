//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu=true enable-is-reduce-supported=true" --mlir-elide-elementsattrs-if-larger 64 --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @AvgPoolToNCE(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK:         [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [3, 3]
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:           clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:           clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:           scale = 0.1111111111111111 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK:           return [[OUT]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

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

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) rawFilterShape [64, 16, 3, 3] {
    // CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:            clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:            scale = 1.000000e+00 : f64, prelu_alpha = [-2.312500e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

    // CHECK-SAME:            strides = [2, 2]
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
    // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) rawFilterShape [16, 16, 1, 1]
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.3385416666666667 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:          strides = [1, 1]}
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

    // CHECK:   [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]], [[WEIGHTS]]) rawFilterShape [16, 16, 1, 1]
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 5.2415569058069783E-40 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
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

        // CHECK:       [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) rawFilterShape [16, 16, 1, 1]
        // CHECK-SAME:      input_padding = [0, 12, 0, 0],
        // CHECK-SAME:      output_padding = [0, 12, 0, 0],
        // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
        // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
        // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
        // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
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

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) rawFilterShape [128, 64, 32, 1, 1] {

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

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) rawFilterShape [12, 32, 144, 1, 1] {
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

    // CHECK:       [[MATMUL:%.+]] = VPU.NCE.MatMul([[PERMUTE_CAST_0]], [[PERMUTE_CAST_1]], [[WEIGHTS_TABLE]]) rawFilterShape [8, 128, 64, 1, 1] {
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:  ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:      scale = 2.0451458953301231E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:     -> tensor<8x1x128x1x1xf16, {order = #GNHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            clamp = {min = 0.000000e+00 : f64, max = 6.000000e+00 : f64}
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {kernel_size = [1, 1],
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.000000e+00 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolFP32ToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolFP32ToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf32, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf32, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf32, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {kernel_size = [1, 1],
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4xf32, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AveragePoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @AveragePoolToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x14x14xf16, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [2, 2]
         } : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [2, 2],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:          scale = 2.500000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x14x14xf16, {order = #NHWC}>
}

// -----

// Check that an IE.Convolution whose filter uses !QuantileType.quantile weights
// (produced by a QuantizeCast → AffineReshape → QuantizeCast → AffineReshape →
// QuantizeCast → PermuteCast → Concat chain) is lowered to VPU.NCE.Convolution
// with the correct weight-table constants and NCE weight-tensor layout
// (Reshape → Expand → LayoutCast).

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Quantile quant type: lookup-table index encoding, per-output-channel scales.
// !qElemType0 – the i4 per-channel type before QuantileType cast
// !qElemType1 – the QuantileType.quantile weight type
!qElemType0 = !quant.uniform<i4:f16:0, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>
!qElemType1 = !quant.uniform<!QuantileType.quantile<ui4:si8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16:0, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>

// CHECK-DAG: [[QTYPE0:!.+]] = !quant.uniform<i4:f16:0,
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<!QuantileType.quantile<ui4:si8,

// CHECK: @NCEConvolutionWithQuantileTypeFilter
// CHECK-SAME: ([[ACT:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[WEIGHTS:%.+]]: tensor<16x4xsi4>)
func.func @NCEConvolutionWithQuantileTypeFilter(
        %arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
        %arg1: tensor<16x4xsi4>
) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    // Zero-padding constant for the filter (input-channel dimension: 4 → 16).
    %cst_zero = const.Declare tensor<16x12x1x1x!qElemType1, {order = #NHWC}> =
        dense<0> : tensor<16x12x1x1xui8, {order = #NHWC}>

    // Filter producer chain: si4 weights → QuantileType per-channel quant tensor.
    %0 = IE.QuantizeCast(%arg1) {dstElemType = !quant.uniform<i4:f16, 1.000000e+00>}
        : tensor<16x4xsi4> -> tensor<16x4x!quant.uniform<i4:f16, 1.000000e+00>>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 16, 1, 4]}
        : tensor<16x4x!quant.uniform<i4:f16, 1.000000e+00>>
        -> tensor<1x16x1x4x!quant.uniform<i4:f16, 1.000000e+00>>
    %2 = IE.QuantizeCast(%1) {dstElemType = !quant.uniform<i4:f16:1, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>}
        : tensor<1x16x1x4x!quant.uniform<i4:f16, 1.000000e+00>>
        -> tensor<1x16x1x4x!quant.uniform<i4:f16:1, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16, 4, 1, 1]}
        : tensor<1x16x1x4x!quant.uniform<i4:f16:1, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>>
        -> tensor<16x4x1x1x!qElemType0>
    %4 = IE.QuantizeCast(%3) {dstElemType = !qElemType1}
        : tensor<16x4x1x1x!qElemType0> -> tensor<16x4x1x1x!qElemType1>
    %5 = IE.PermuteCast(%4) {dst_order = #NHWC, mem_perm = #NHWC}
        : tensor<16x4x1x1x!qElemType1> -> tensor<16x4x1x1x!qElemType1, {order = #NHWC}>
    %6 = IE.Concat(%5, %cst_zero) {static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]}
        : tensor<16x4x1x1x!qElemType1, {order = #NHWC}>,
          tensor<16x12x1x1x!qElemType1, {order = #NHWC}>
          -> tensor<16x16x1x1x!qElemType1, {order = #NHWC}>

    %7 = IE.Convolution(%arg0, %6) {
        dilations = [1, 1], input_padding = [0, 12, 0, 0], output_padding = [0, 0, 0, 0],
        pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x1x1xf16, {order = #NHWC}>,
        tensor<16x16x1x1x!qElemType1, {order = #NHWC}>
        -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %7 : tensor<1x16x1x1xf16, {order = #NHWC}>

    // Filter producer chain is preserved unchanged up to and including the Concat.
    // CHECK-DAG:   [[ZERO:%.+]] = const.Declare tensor<16x12x1x1x[[QTYPE1]], {order = #NHWC}> = dense<0>

    // CHECK:       [[Q_CAST0:%.+]] = IE.QuantizeCast([[WEIGHTS]])
    // CHECK:       [[AFF_RESHAPE0:%.+]] = IE.AffineReshape([[Q_CAST0]])
    // CHECK:       [[Q_CAST1:%.+]] = IE.QuantizeCast([[AFF_RESHAPE0]])
    // CHECK:       [[AFF_RESHAPE1:%.+]] = IE.AffineReshape([[Q_CAST1]])
    // CHECK:       [[Q_CAST2:%.+]] = IE.QuantizeCast([[AFF_RESHAPE1]])
    // CHECK:       [[FILTER_NHWC:%.+]] = IE.PermuteCast([[Q_CAST2]])
    // CHECK-SAME:      -> tensor<16x4x1x1x[[QTYPE1]], {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[FILTER_NHWC]], [[ZERO]])
    // CHECK-SAME{{LITERAL}}:      static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]
    // CHECK-SAME:      -> tensor<16x16x1x1x[[QTYPE1]], {order = #NHWC}>

    // The 16x16 filter is reshaped and expanded into NCE weight layout [OC, 1, 1, IC_aligned]
    // then LayoutCast to NHWC before being passed to VPU.NCE.Convolution.
    // CHECK:       [[RESHAPE:%.+]] = VPU.Reshape([[CONCAT]])
    // CHECK-SAME:      shape_value = [16, 1, 1, 16]
    // CHECK-SAME:      -> tensor<16x1x1x16x[[QTYPE1]]>
    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[RESHAPE]])
    // CHECK-SAME:      pads_end = [0, 0, 0, 16]
    // CHECK-SAME:      -> tensor<16x1x1x32x[[QTYPE1]]>
    // CHECK:       [[LAYOUT:%.+]] = VPU.LayoutCast([[EXPAND]])
    // CHECK-SAME:      -> tensor<16x1x1x32x[[QTYPE1]], {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[ACT]], [[LAYOUT]]
    // CHECK-SAME:      input_padding = [0, 12, 0, 0]
    // CHECK-SAME:      weight_zp = 0
    // CHECK-SAME:      output_padding = [0, 0, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[OUT]]
}
